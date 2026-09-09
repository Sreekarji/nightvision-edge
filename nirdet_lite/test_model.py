"""
test_model.py — NIRDet-Lite forward-pass smoke test.

NIRDet uses strides (8, 16, 32), giving three output levels for a 384x640 input:
  stride  8 -> grid 48x80 -> 3840 cells -> (1, 3840, 5)
  stride 16 -> grid 24x40 ->  960 cells -> (1,  960, 5)
  stride 32 -> grid 12x20 ->  240 cells -> (1,  240, 5)
"""

import sys
import os

# Make sure imports resolve from this directory
sys.path.insert(0, os.path.dirname(__file__))

import torch
from model import NIRDet

def test_forward():
    m = NIRDet()
    m.eval()

    x = torch.zeros(1, 1, 384, 640)

    with torch.no_grad():
        # training_mode=True -> returns raw logit tensors, one per level
        preds = m(x, training_mode=True)

    print(f"  Number of output levels: {len(preds)}")
    for i, t in enumerate(preds):
        print(f"  Level {i}: shape={tuple(t.shape)}")

    # Three-level model: strides (8, 16, 32)
    assert len(preds) == 3, f"Expected 3 levels, got {len(preds)}"
    assert tuple(preds[0].shape) == (1, 3840, 5), \
        f"Level 0 shape mismatch: expected (1, 3840, 5), got {tuple(preds[0].shape)}"
    assert tuple(preds[1].shape) == (1, 960, 5), \
        f"Level 1 shape mismatch: expected (1, 960, 5), got {tuple(preds[1].shape)}"
    assert tuple(preds[2].shape) == (1, 240, 5), \
        f"Level 2 shape mismatch: expected (1, 240, 5), got {tuple(preds[2].shape)}"

    print("PASS")


if __name__ == "__main__":
    print("=" * 50)
    print("  test_model.py — NIRDet forward pass")
    print("=" * 50)
    test_forward()
