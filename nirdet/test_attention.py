"""
test_attention.py — Unit tests for EdgeAwareAttention (EAA)

Tests required:
  T1  Shape:          img=(1,1,640,640), feat=(1,64,80,80) → output=(1,64,80,80)
  T2  Attention range: all attention values in [0, 1]
  T3  Gradient:       loss=output.sum(); loss.backward() — no None grads,
                      edge_conv weights have non-zero gradients
  T4  Edge sensitivity: synthetic vertical-line image vs zeros →
                        line image must have higher activation at line location

Run with:
    python test_attention.py
or:
    pytest test_attention.py -v
"""

import sys
import torch
import torch.nn.functional as F

# ── import the module under test ──────────────────────────────────────────────
try:
    from attention import EdgeAwareAttention, build_eaa
except ImportError as exc:
    sys.exit(f"[ERROR] Cannot import attention.py: {exc}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running tests on device: {DEVICE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_eaa(**kwargs) -> EdgeAwareAttention:
    """Create an EAA in fully-trainable mode (freeze_epochs=0)."""
    return EdgeAwareAttention(freeze_epochs=0, **kwargs).to(DEVICE)


def pass_fail(name: str, ok: bool) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    return ok


# ---------------------------------------------------------------------------
# Test 1 — Output shape
# ---------------------------------------------------------------------------

def test_shape() -> bool:
    """img=(1,1,640,640), feat=(1,64,80,80) → output=(1,64,80,80)"""
    eaa = make_eaa()
    img  = torch.randn(1, 1, 640, 640, device=DEVICE)
    feat = torch.randn(1, 64, 80, 80,  device=DEVICE)

    with torch.no_grad():
        out = eaa(feat, img)

    expected = (1, 64, 80, 80)
    ok = (tuple(out.shape) == expected)
    if not ok:
        print(f"      Expected shape {expected}, got {tuple(out.shape)}")
    return pass_fail("T1 — Output shape (1,64,80,80)", ok)


# ---------------------------------------------------------------------------
# Test 2 — Attention values in [0, 1]
# ---------------------------------------------------------------------------

def test_attention_range() -> bool:
    """All attention values must be in [0, 1] (sigmoid output)."""
    eaa = make_eaa()
    eaa.eval()

    # Hook to capture the sigmoid output
    captured = {}

    def hook(module, inp, out):
        # This hook fires on the sigmoid — but we patch forward instead
        pass

    # Patch forward to expose internal attn
    original_forward = eaa.forward

    attn_values = []

    def patched_forward(feat, img):
        B, C, Hf, Wf = feat.shape
        _, _, H, W   = img.shape
        img_pad = F.pad(img, (1, 1, 1, 1), mode=eaa.padding_mode)
        e_map   = eaa.edge_conv(img_pad)
        e_mag   = torch.abs(e_map)
        e_ds    = F.adaptive_avg_pool2d(e_mag, (Hf, Wf))
        a_raw   = eaa.proj(e_ds)
        attn    = torch.sigmoid(a_raw)
        attn_values.append(attn.detach().clone())
        if eaa.residual_scale is not None:
            return feat * (1.0 + eaa.residual_scale * attn)
        return feat * attn

    eaa.forward = patched_forward

    img  = torch.randn(2, 1, 320, 320, device=DEVICE)
    feat = torch.randn(2, 32, 40, 40,  device=DEVICE)

    with torch.no_grad():
        _ = eaa(feat, img)

    eaa.forward = original_forward  # restore

    if not attn_values:
        return pass_fail("T2 — Attention range [0,1]", False)

    a = attn_values[0]
    mn, mx = a.min().item(), a.max().item()
    ok = (mn >= 0.0 - 1e-6) and (mx <= 1.0 + 1e-6)
    if not ok:
        print(f"      min={mn:.6f}, max={mx:.6f} — outside [0,1]")
    else:
        print(f"      attention min={mn:.6f}, max={mx:.6f}")
    return pass_fail("T2 — Attention range [0,1]", ok)


# ---------------------------------------------------------------------------
# Test 3 — Gradients flow to edge_conv weights
# ---------------------------------------------------------------------------

def test_gradients() -> bool:
    """
    loss = output.sum(); loss.backward()
      • No parameter has grad == None
      • edge_conv.weight.grad is not all-zero
    """
    eaa = make_eaa()
    eaa.train()

    img  = torch.randn(1, 1, 640, 640, device=DEVICE, requires_grad=False)
    feat = torch.randn(1, 64, 80, 80,  device=DEVICE, requires_grad=False)

    out  = eaa(feat, img)
    loss = out.sum()
    loss.backward()

    # Check no parameter has None grad
    none_grad_params = []
    for name, p in eaa.named_parameters():
        if p.requires_grad and p.grad is None:
            none_grad_params.append(name)

    # Check edge_conv weight has non-zero gradient
    edge_grad = eaa.edge_conv.weight.grad
    has_edge_grad = (edge_grad is not None) and (edge_grad.abs().sum().item() > 0.0)

    ok = (len(none_grad_params) == 0) and has_edge_grad

    if none_grad_params:
        print(f"      Params with None grad: {none_grad_params}")
    if not has_edge_grad:
        print(f"      edge_conv.weight.grad: {edge_grad}")
    else:
        print(f"      edge_conv.weight.grad norm = {edge_grad.norm():.6f}")

    return pass_fail("T3 — Gradients non-None, edge_conv non-zero", ok)


# ---------------------------------------------------------------------------
# Test 4 — Edge sensitivity: vertical line vs zeros
# ---------------------------------------------------------------------------

def test_edge_sensitivity() -> bool:
    """
    A synthetic image with a vertical white line should produce higher
    attention activation at the line location than an all-zeros image.

    Protocol:
      1. Create img_line:  1×1×64×64, all zeros except column 32 = 1.0
      2. Create img_flat:  1×1×64×64, all zeros
      3. Run both through EAA with the SAME feature map (pure multiplicative,
         residual_scale=None) so output directly reflects attn amplitude.
      4. Extract attention at column 32 from both outputs.
      5. Assert mean(out_line[:, :, :, 32]) > mean(out_flat[:, :, :, 32])
    """
    # Use pure multiplicative so output magnitude = feat * attn directly
    eaa = make_eaa(residual_scale=None)
    eaa.eval()

    H, W   = 64, 64
    Hf, Wf = 8, 8   # 1/8 scale → column 32 maps to column 4 in feat space

    # Vertical line at column 32
    img_line = torch.zeros(1, 1, H, W, device=DEVICE)
    img_line[:, :, :, 32] = 1.0   # bright vertical stripe

    img_flat = torch.zeros(1, 1, H, W, device=DEVICE)

    # Shared feature map: all-ones so output = attn (no feat influence)
    feat = torch.ones(1, 16, Hf, Wf, device=DEVICE)

    with torch.no_grad():
        out_line = eaa(feat, img_line)   # (1, 16, 8, 8)
        out_flat = eaa(feat, img_flat)   # (1, 16, 8, 8)

    # Column 32 in img maps to column 4 in feat (32 / (64/8) = 4)
    col_feat = int(32 / (W / Wf))

    mean_line = out_line[:, :, :, col_feat].mean().item()
    mean_flat = out_flat[:, :, :, col_feat].mean().item()

    ok = mean_line > mean_flat
    print(f"      mean at line col ({col_feat}): line={mean_line:.6f}, flat={mean_flat:.6f}")

    return pass_fail("T4 — Edge sensitivity: line > flat at edge location", ok)


# ---------------------------------------------------------------------------
# Test 4b — Multi-scale: all three FPN scales produce correct shapes
# ---------------------------------------------------------------------------

def test_multiscale_shapes() -> bool:
    """Verify P3, P4, P5 scales all work with the same EAA instance."""
    eaa = make_eaa()

    scales = [
        # (C,  H', W',  label)
        (64,   80,  80, "P3"),
        (128,  40,  40, "P4"),
        (256,  20,  20, "P5"),
    ]
    img = torch.randn(2, 1, 640, 640, device=DEVICE)

    all_ok = True
    for C, Hf, Wf, label in scales:
        feat = torch.randn(2, C, Hf, Wf, device=DEVICE)
        with torch.no_grad():
            out = eaa(feat, img)
        expected = (2, C, Hf, Wf)
        ok = (tuple(out.shape) == expected)
        if not ok:
            print(f"      {label}: expected {expected}, got {tuple(out.shape)}")
            all_ok = False
        else:
            print(f"      {label}: shape {tuple(out.shape)} ✓")

    return pass_fail("T4b — Multi-scale shapes P3/P4/P5", all_ok)


# ---------------------------------------------------------------------------
# Test 5 — Freeze / unfreeze epoch logic
# ---------------------------------------------------------------------------

def test_freeze_unfreeze() -> bool:
    """
    With freeze_epochs=3:
      • At epoch 1  → edge_conv.weight.requires_grad == False
      • After epoch 3 step → still frozen
      • After epoch 4 step → unfrozen
    """
    eaa = EdgeAwareAttention(freeze_epochs=3).to(DEVICE)

    # Simulate initial state before any epoch step — weights frozen
    eaa._update_grad_state()  # reflect epoch=0
    # At epoch=0 (< freeze_epochs=3), should be frozen
    frozen_at_0 = not eaa.edge_conv.weight.requires_grad

    # Simulate epoch 3 — still frozen
    eaa._current_epoch = 3
    eaa._update_grad_state()
    still_frozen_at_3 = not eaa.edge_conv.weight.requires_grad

    # Simulate epoch 4 — unfrozen
    eaa._current_epoch = 4
    eaa._update_grad_state()
    unfrozen_at_4 = eaa.edge_conv.weight.requires_grad

    ok = frozen_at_0 and still_frozen_at_3 and unfrozen_at_4
    print(f"      frozen@epoch0={frozen_at_0}, frozen@epoch3={still_frozen_at_3}, "
          f"trainable@epoch4={unfrozen_at_4}")
    return pass_fail("T5 — Freeze/unfreeze epoch logic", ok)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EAA Unit Tests")
    print("=" * 60)

    results = []
    for test_fn in [
        test_shape,
        test_attention_range,
        test_gradients,
        test_edge_sensitivity,
        test_multiscale_shapes,
        test_freeze_unfreeze,
    ]:
        print(f"\n{test_fn.__doc__.strip().splitlines()[0]}")
        try:
            results.append(test_fn())
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total  = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed < total:
        sys.exit(1)
