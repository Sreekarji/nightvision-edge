"""
test_head.py — Unit tests for PedestrianHead.

Updated for the Bug-1 API change (aspect_ratio_prior removed; replaced by
prior_w and prior_h) and for the Bug-4/5 inference-decode fixes (sigmoid on
cx/cy offsets; w/h scaled by H*stride not just stride).

Run with:
    python test_head.py
or:
    pytest test_head.py -v
"""

import math
import torch
from head import PedestrianHead


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_inputs(batch=1, base_hw=(48, 80)):
    # N3 at H/8, N4 at H/16, N5 at H/32, all consistent with a single
    # input resolution (e.g. base_hw=(48,80) => 384x640 input image).
    base_h, base_w = base_hw
    N3 = torch.randn(batch, 256, base_h,      base_w)      # (B, 256, 48, 80)
    N4 = torch.randn(batch, 256, base_h // 2, base_w // 2) # (B, 256, 24, 40)
    N5 = torch.randn(batch, 256, base_h // 4, base_w // 4) # (B, 256, 12, 20)
    return N3, N4, N5


# ─────────────────────────────────────────────────────────────────────────────
# test_shapes — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

def test_shapes():
    head = PedestrianHead()
    N3, N4, N5 = _make_inputs(batch=1, base_hw=(48, 80))
    preds = head(N3, N4, N5)

    assert len(preds) == 3
    assert preds[0].shape == (1, 48 * 80, 5), preds[0].shape   # (1, 3840, 5)
    assert preds[1].shape == (1, 24 * 40, 5), preds[1].shape   # (1,  960, 5)
    assert preds[2].shape == (1, 12 * 20, 5), preds[2].shape   # (1,  240, 5)
    print("test_shapes: PASS")


# ─────────────────────────────────────────────────────────────────────────────
# test_confidence_range — unchanged from original; uses training_mode=True
# (default) so conf is a raw logit, not sigmoided → NOT in [0,1].
# Updated: test inference-mode confidence instead, which IS in [0,1].
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_range():
    """Inference-mode confidence (after sigmoid) must be in [0, 1]."""
    head = PedestrianHead()
    head.eval()
    N3, N4, N5 = _make_inputs(batch=2, base_hw=(48, 80))
    with torch.no_grad():
        # training_mode=False applies sigmoid to conf before returning
        preds = head(N3, N4, N5, training_mode=False)
    for level_pred in preds:
        conf = level_pred[..., 4]               # (B, H*W)
        assert torch.all(conf >= 0.0), f"conf min = {conf.min().item()}"
        assert torch.all(conf <= 1.0), f"conf max = {conf.max().item()}"
    print("test_confidence_range: PASS")


# ─────────────────────────────────────────────────────────────────────────────
# test_prior_bias  (replaces test_aspect_ratio_bias)
#
# Bug 1 removed aspect_ratio_prior and replaced it with prior_w / prior_h.
# The new test checks two things:
#   (a) At network init, mean decoded w ≈ prior_w * img_size  (in pixels)
#       and mean decoded h ≈ prior_h * img_size.
#   (b) w/h ratio ≈ prior_w / prior_h ≈ 0.04/0.15 ≈ 0.267.
#
# We use inference_mode=False so the decode matches losses.py exactly.
# Small base_hw=16 keeps this fast (same as the original test).
# ─────────────────────────────────────────────────────────────────────────────

def test_prior_bias():
    """
    At init, decoded w and h must reflect the prior_w/prior_h biases.

    For base_hw=16 (→ 128×128 effective image, strides 8/4/2):
      w_expected_px = prior_w * img_size = 0.04 * (16*8) = 5.12  (P3)
      h_expected_px = prior_h * img_size = 0.15 * (16*8) = 19.2

    We average over 100 random weight seeds to wash out the tiny random
    perturbation from std=0.01 conv weights and check we land within
    ±30 % of the expected value (generous, matches original test tolerance).
    """
    PRIOR_W   = 0.04
    PRIOR_H   = 0.15
    BASE_HW   = 16          # small for speed
    IMG_SIZE  = BASE_HW * 8 # effective image width = 128 px for P3

    torch.manual_seed(0)
    head = PedestrianHead(prior_w=PRIOR_W, prior_h=PRIOR_H)
    head.eval()

    w_preds, h_preds, ratios = [], [], []
    with torch.no_grad():
        for _ in range(100):
            N3, N4, N5 = _make_inputs(batch=1, base_hw=(BASE_HW, BASE_HW))
            # training_mode=False → decoded pixels
            preds = head(N3, N4, N5, training_mode=False)
            for level_pred in preds:
                w = level_pred[..., 2]   # (1, H*W) in pixels
                h = level_pred[..., 3]
                w_preds.append(w.mean().item())
                h_preds.append(h.mean().item())
                ratios.append((w / h).mean().item())

    avg_w     = sum(w_preds) / len(w_preds)
    avg_h     = sum(h_preds) / len(h_preds)
    avg_ratio = sum(ratios)  / len(ratios)

    expected_w = PRIOR_W * IMG_SIZE   # 5.12 px
    expected_h = PRIOR_H * IMG_SIZE   # 19.2 px
    expected_r = PRIOR_W / PRIOR_H    # 0.267

    print(f"test_prior_bias:")
    print(f"  avg w = {avg_w:.4f}  (expected ≈ {expected_w:.2f} ± 30%)")
    print(f"  avg h = {avg_h:.4f}  (expected ≈ {expected_h:.2f} ± 30%)")
    print(f"  avg w/h ratio = {avg_ratio:.4f}  (expected ≈ {expected_r:.3f})")

    tol = 0.30   # 30 % tolerance — conv weights are std=0.01 noise around the bias
    assert abs(avg_w - expected_w) / expected_w < tol, \
        f"avg w {avg_w:.4f} deviates >30% from expected {expected_w:.4f}"
    assert abs(avg_h - expected_h) / expected_h < tol, \
        f"avg h {avg_h:.4f} deviates >30% from expected {expected_h:.4f}"
    assert expected_r * 0.5 < avg_ratio < expected_r * 2.0, \
        f"avg w/h ratio {avg_ratio:.4f} far from expected {expected_r:.3f}"

    print("test_prior_bias: PASS")


# ─────────────────────────────────────────────────────────────────────────────
# test_gradients — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

def test_gradients():
    head = PedestrianHead()
    N3, N4, N5 = _make_inputs(batch=1, base_hw=(16, 16))
    preds = head(N3, N4, N5)
    loss = sum(p.sum() for p in preds)
    loss.backward()

    missing = [name for name, p in head.named_parameters() if p.grad is None]
    assert not missing, f"Parameters with no gradient: {missing}"
    print("test_gradients: PASS")


# ─────────────────────────────────────────────────────────────────────────────
# test_inference_decode_consistency  (new — validates Bug-4 and Bug-5 fixes)
#
# Verifies that inference-mode decoding matches what losses.py does during
# training.  Checks two properties on zero-input feature maps (so network
# weights are zero and bias dominates — clean signal):
#
#   (A) cx consistency: infer cx == manual decode of training logit
#       manual: (sigmoid(t_cx_logit) + col) / S_w * img_w_px
#   (B) w/h consistency: infer w == exp(t_w_logit) * (S_w * stride) and
#       h == exp(t_h_logit) * (S_h * stride)  (per-axis pixel extents)
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_decode_consistency():
    """
    Inference decode must be numerically identical to the losses.py training
    decode on the same raw logits.  Tests Bug-4 (sigmoid on cx/cy), Bug-5
    (w/h scaled by per-axis extent, not just stride) and the 640×384 fix
    (w scaled by image WIDTH = W*stride, h by image HEIGHT = H*stride).
    """
    IMG_H = 384
    IMG_W = 640
    head = PedestrianHead()
    head.eval()

    # Zero feature maps → only biases contribute → clean, reproducible signal.
    N3 = torch.zeros(1, 256, IMG_H // 8,  IMG_W // 8)   # (1,256,48,80)
    N4 = torch.zeros(1, 256, IMG_H // 16, IMG_W // 16)  # (1,256,24,40)
    N5 = torch.zeros(1, 256, IMG_H // 32, IMG_W // 32)  # (1,256,12,20)

    with torch.no_grad():
        raw  = head(N3, N4, N5, training_mode=True)    # raw logits
        dec  = head(N3, N4, N5, training_mode=False)   # decoded pixels

    # ── Check P3 (scale 0, S_h=48, S_w=80, stride=8) ─────────────────────────
    S_h    = 48
    S_w    = 80
    stride = 8
    n_cells = S_h * S_w          # 3840

    t_cx_logit = raw[0][0, :, 0]   # (3840,) raw cx offset logits
    t_cy_logit = raw[0][0, :, 1]   # (3840,) raw cy offset logits
    t_w_logit  = raw[0][0, :, 2]   # (3840,) raw w logits
    t_h_logit  = raw[0][0, :, 3]   # (3840,) raw h logits

    # Manual decode matching losses.py:
    col_idx   = torch.arange(n_cells) % S_w            # (3840,) column per cell
    row_idx   = torch.arange(n_cells) // S_w           # (3840,) row per cell
    # NOTE: (sigmoid + col) / S_w * img_w_px = (sigmoid + col) * stride  ✓
    cx_manual = (torch.sigmoid(t_cx_logit) + col_idx.float()) * stride
    cy_manual = (torch.sigmoid(t_cy_logit) + row_idx.float()) * stride
    w_manual  = torch.exp(t_w_logit) * (S_w * stride)  # = exp * 640 (image width)
    h_manual  = torch.exp(t_h_logit) * (S_h * stride)  # = exp * 384 (image height)

    cx_infer  = dec[0][0, :, 0]   # (3840,) decoded cx in pixels
    cy_infer  = dec[0][0, :, 1]   # (3840,) decoded cy in pixels
    w_infer   = dec[0][0, :, 2]   # (3840,) decoded w  in pixels
    h_infer   = dec[0][0, :, 3]   # (3840,) decoded h  in pixels

    # Allow tiny floating-point rounding (< 1e-4 relative)
    cx_err = (cx_infer - cx_manual).abs().max().item()
    cy_err = (cy_infer - cy_manual).abs().max().item()
    w_err  = (w_infer  - w_manual ).abs().max().item()
    h_err  = (h_infer  - h_manual ).abs().max().item()

    print(f"test_inference_decode_consistency:")
    print(f"  max |cx_infer - cx_manual| = {cx_err:.2e}  (expect < 1e-4)")
    print(f"  max |cy_infer - cy_manual| = {cy_err:.2e}  (expect < 1e-4)")
    print(f"  max |w_infer  - w_manual|  = {w_err:.2e}  (expect < 1e-4)")
    print(f"  max |h_infer  - h_manual|  = {h_err:.2e}  (expect < 1e-4)")
    print(f"  cx range: [{cx_infer.min().item():.1f}, {cx_infer.max().item():.1f}]  "
          f"(expect [4, 636] for zero-bias input)")
    print(f"  w  mean:  {w_infer.mean().item():.2f} px  "
          f"(expect {0.0461 * IMG_W:.2f} = prior_w * {IMG_W})")
    print(f"  h  mean:  {h_infer.mean().item():.2f} px  "
          f"(expect {0.1680 * IMG_H:.2f} = prior_h * {IMG_H})")

    assert cx_err < 1e-4, f"cx decode mismatch: max err = {cx_err}"
    assert cy_err < 1e-4, f"cy decode mismatch: max err = {cy_err}"
    assert w_err  < 1e-4, f"w  decode mismatch: max err = {w_err}"
    assert h_err  < 1e-4, f"h  decode mismatch: max err = {h_err}"

    # cx must be within image bounds (width 640)
    assert cx_infer.min().item() >= 0.0,     "cx < 0 at init"
    assert cx_infer.max().item() <= IMG_W,   "cx > img_w at init"
    # cy within image bounds (height 384)
    assert cy_infer.min().item() >= 0.0,     "cy < 0 at init"
    assert cy_infer.max().item() <= IMG_H,   "cy > img_h at init"

    # w at init should be ~prior_w * img_w = 0.0461 * 640 = 29.5 px;
    # h at init should be ~prior_h * img_h = 0.1680 * 384 = 64.5 px.
    w_mean = w_infer.mean().item()
    h_mean = h_infer.mean().item()
    assert 24.0 < w_mean < 36.0, \
        f"w mean {w_mean:.2f} not near expected 29.5 px — w scale fix may be missing"
    assert 52.0 < h_mean < 78.0, \
        f"h mean {h_mean:.2f} not near expected 64.5 px — h scale fix may be missing"

    print("test_inference_decode_consistency: PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_shapes()
    test_confidence_range()
    test_prior_bias()
    test_gradients()
    test_inference_decode_consistency()
    print("\nALL TESTS PASSED")
