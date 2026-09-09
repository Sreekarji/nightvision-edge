"""
test_head.py — PedestrianHead forward-pass smoke test.

PedestrianHead uses strides (8, 16, 32) — three levels for 384x640 geometry:
  N3 (stride  8) : (1, 64, 48, 80)  -> (1, 3840, 5)
  N4 (stride 16) : (1, 64, 24, 40)  -> (1,  960, 5)
  N5 (stride 32) : (1, 64, 12, 20)  -> (1,  240, 5)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from head import PedestrianHead


def test_head_shapes():
    head = PedestrianHead(
        in_channels=64,
        feat_channels=64,
        strides=(8, 16, 32),
        num_branch_convs=1,
        use_stem=True,
    )
    head.eval()

    # Three-level feature maps matching 384x640 input
    feats = [
        torch.zeros(1, 64, 48, 80),   # N3: stride 8  -> 48x80 grid
        torch.zeros(1, 64, 24, 40),   # N4: stride 16 -> 24x40 grid
        torch.zeros(1, 64, 12, 20),   # N5: stride 32 -> 12x20 grid
    ]

    with torch.no_grad():
        # Training mode: raw logits
        raw_out = head(feats, training_mode=True)
        # Inference mode: decoded pixel coordinates
        dec_out = head(feats, training_mode=False)

    print("  Training-mode (raw logit) shapes:")
    for i, t in enumerate(raw_out):
        print(f"    Level {i}: {tuple(t.shape)}")

    print("  Inference-mode (decoded) shapes:")
    for i, t in enumerate(dec_out):
        print(f"    Level {i}: {tuple(t.shape)}")

    # Verify shapes
    assert len(raw_out) == 3, f"Expected 3 levels, got {len(raw_out)}"
    assert tuple(raw_out[0].shape) == (1, 3840, 5), \
        f"N3 raw shape mismatch: expected (1, 3840, 5), got {tuple(raw_out[0].shape)}"
    assert tuple(raw_out[1].shape) == (1, 960, 5), \
        f"N4 raw shape mismatch: expected (1, 960, 5), got {tuple(raw_out[1].shape)}"
    assert tuple(raw_out[2].shape) == (1, 240, 5), \
        f"N5 raw shape mismatch: expected (1, 240, 5), got {tuple(raw_out[2].shape)}"

    assert len(dec_out) == 3, f"Expected 3 decoded levels, got {len(dec_out)}"
    assert tuple(dec_out[0].shape) == (1, 3840, 5), \
        f"N3 decoded shape mismatch: {tuple(dec_out[0].shape)}"
    assert tuple(dec_out[1].shape) == (1, 960, 5), \
        f"N4 decoded shape mismatch: {tuple(dec_out[1].shape)}"
    assert tuple(dec_out[2].shape) == (1, 240, 5), \
        f"N5 decoded shape mismatch: {tuple(dec_out[2].shape)}"

    # Sanity-check prior biases
    w_mean = float(dec_out[0][..., 2].mean())
    h_mean = float(dec_out[0][..., 3].mean())
    print(f"  Prior w_mean={w_mean:.1f}px (expect ~{0.0461*640:.1f}px)")
    print(f"  Prior h_mean={h_mean:.1f}px (expect ~{0.1680*384:.1f}px)")
    assert abs(w_mean - 0.0461 * 640) < 5.0, \
        f"Prior w unexpected: {w_mean:.1f} vs {0.0461*640:.1f}"
    assert abs(h_mean - 0.1680 * 384) < 5.0, \
        f"Prior h unexpected: {h_mean:.1f} vs {0.1680*384:.1f}"

    params = sum(p.numel() for p in head.parameters())
    print(f"  Head parameters: {params:,}")

    print("PASS")


if __name__ == "__main__":
    print("=" * 50)
    print("  test_head.py — PedestrianHead forward pass")
    print("=" * 50)
    test_head_shapes()
