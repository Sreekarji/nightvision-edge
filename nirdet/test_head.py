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

def _make_inputs(batch=1, base_hw=80):
    # N3 at H/8, N4 at H/16, N5 at H/32, all consistent with a single
    # input resolution (e.g. base_hw=80 => 640x640 input image).
    N3 = torch.randn(batch, 256, base_hw,      base_hw)        # (B, 256, 80, 80)
    N4 = torch.randn(batch, 256, base_hw // 2, base_hw // 2)   # (B, 256, 40, 40)
    N5 = torch.randn(batch, 256, base_hw // 4, base_hw // 4)   # (B, 256, 20, 20)
    return N3, N4, N5


# ─────────────────────────────────────────────────────────────────────────────
# test_shapes — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

def test_shapes():
    head = PedestrianHead()
    N3, N4, N5 = _make_inputs(batch=1, base_hw=80)
    preds = head(N3, N4, N5)

    assert len(preds) == 3
    assert preds[0].shape == (1, 80 * 80, 5), preds[0].shape   # (1, 6400, 5)
    assert preds[1].shape == (1, 40 * 40, 5), preds[1].shape   # (1, 1600, 5)
    assert preds[2].shape == (1, 20 * 20, 5), preds[2].shape   # (1,  400, 5)
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
    N3, N4, N5 = _make_inputs(batch=2, base_hw=80)
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
            N3, N4, N5 = _make_inputs(batch=1, base_hw=BASE_HW)
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
    N3, N4, N5 = _make_inputs(batch=1, base_hw=16)
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
#       manual: (sigmoid(t_cx_logit) + col) / S * img_size
#   (B) w  consistency: infer w  == exp(t_w_logit) * img_size
#       (w is in pixels; img_size = H * stride for the P3 scale)
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_decode_consistency():
    """
    Inference decode must be numerically identical to the losses.py training
    decode on the same raw logits.  Tests Bug-4 (sigmoid on cx/cy) and
    Bug-5 (w/h scaled by H*stride, not just stride).
    """
    IMG_SIZE = 640
    head = PedestrianHead()
    head.eval()

    # Zero feature maps → only biases contribute → clean, reproducible signal.
    N3 = torch.zeros(1, 256, IMG_SIZE // 8,  IMG_SIZE // 8)   # (1,256,80,80)
    N4 = torch.zeros(1, 256, IMG_SIZE // 16, IMG_SIZE // 16)  # (1,256,40,40)
    N5 = torch.zeros(1, 256, IMG_SIZE // 32, IMG_SIZE // 32)  # (1,256,20,20)

    with torch.no_grad():
        raw  = head(N3, N4, N5, training_mode=True)    # raw logits
        dec  = head(N3, N4, N5, training_mode=False)   # decoded pixels

    # ── Check P3 (scale 0, S=80, stride=8) ──────────────────────────────────
    S      = 80
    stride = 8

    t_cx_logit = raw[0][0, :, 0]   # (6400,) raw cx offset logits
    t_w_logit  = raw[0][0, :, 2]   # (6400,) raw w logits

    # Manual decode matching losses.py:
    col_idx   = torch.arange(S * S) % S           # (6400,) column per cell
    cx_manual = (torch.sigmoid(t_cx_logit) + col_idx.float()) * stride
    # NOTE: (sigmoid + col) / S * img_size = (sigmoid + col) * stride  ✓
    w_manual  = torch.exp(t_w_logit) * (S * stride)  # = exp * img_size = exp * 640

    cx_infer  = dec[0][0, :, 0]   # (6400,) decoded cx in pixels
    w_infer   = dec[0][0, :, 2]   # (6400,) decoded w  in pixels

    # Allow tiny floating-point rounding (< 1e-4 relative)
    cx_err = (cx_infer - cx_manual).abs().max().item()
    w_err  = (w_infer  - w_manual ).abs().max().item()

    print(f"test_inference_decode_consistency:")
    print(f"  max |cx_infer - cx_manual| = {cx_err:.2e}  (expect < 1e-4)")
    print(f"  max |w_infer  - w_manual|  = {w_err:.2e}  (expect < 1e-4)")
    print(f"  cx range: [{cx_infer.min().item():.1f}, {cx_infer.max().item():.1f}]  "
          f"(expect [4, 636] for zero-bias input)")
    print(f"  w  mean:  {w_infer.mean().item():.2f} px  "
          f"(expect {0.04 * IMG_SIZE:.2f} = prior_w * {IMG_SIZE})")

    assert cx_err < 1e-4, f"cx decode mismatch: max err = {cx_err}"
    assert w_err  < 1e-4, f"w  decode mismatch: max err = {w_err}"

    # cx must be within image bounds
    assert cx_infer.min().item() >= 0.0,     "cx < 0 at init"
    assert cx_infer.max().item() <= IMG_SIZE, "cx > img_size at init"

    # w at init should be ~prior_w * img_size = 0.04 * 640 = 25.6 px
    w_mean = w_infer.mean().item()
    assert 20.0 < w_mean < 32.0, \
        f"w mean {w_mean:.2f} not near expected 25.6 px — w/h scale fix may be missing"

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
