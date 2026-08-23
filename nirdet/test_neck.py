# test_neck.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for LightweightFPN (neck.py).
#
# Run with:
#   pytest test_neck.py -v
# or:
#   python test_neck.py
#
# All three tests must pass before Phase 4 (detection head) begins.
#
# Channel values are read exclusively from the Phase 0 config via get_config().
# No constants are hardcoded here.
# ─────────────────────────────────────────────────────────────────────────────

import pytest
import torch

# ── Single source of truth ────────────────────────────────────────────────────
from config import get_config
from neck import LightweightFPN, _C3, _C4, _C5, _OUT

# Load the live config so tests document which values they are using.
_cfg = get_config()
# backbone_channels: (16, 32, 64, 128, 256) at default config
# _C3  = backbone_channels[2] = 64   (stride-8  backbone output)
# _C4  = backbone_channels[3] = 128  (stride-16 backbone output)
# _C5  = backbone_channels[4] = 256  (stride-32 backbone output)
# _OUT = backbone_channels[-1] = 256 (unified neck output width)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture — one model instance per test module run.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fpn() -> LightweightFPN:
    """Return a LightweightFPN built from config defaults, in eval mode on CPU."""
    model = LightweightFPN()  # uses _C3, _C4, _C5, _OUT derived from config
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Output shape correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shapes(fpn: LightweightFPN) -> None:
    """
    Given config backbone_channels = (16, 32, 64, 128, 256):
        C3  = backbone_channels[2] = 64   → P3 shape (1, 64,  48, 80)
        C4  = backbone_channels[3] = 128  → P4 shape (1, 128, 24, 40)
        C5  = backbone_channels[4] = 256  → P5 shape (1, 256, 12, 20)
        OUT = backbone_channels[-1] = 256

    Spatial sizes match a 640×384 input (384/8=48 rows, 640/8=80 cols, etc.).

    Expect:
        N3 = (1, 256, 48, 80)
        N4 = (1, 256, 24, 40)
        N5 = (1, 256, 12, 20)
    """
    # Build dummy inputs — shapes are driven entirely by _C3/_C4/_C5 from config
    p3 = torch.zeros(1, _C3, 48, 80)   # (1, 64,  48, 80)
    p4 = torch.zeros(1, _C4, 24, 40)   # (1, 128, 24, 40)
    p5 = torch.zeros(1, _C5, 12, 20)   # (1, 256, 12, 20)

    with torch.no_grad():
        n3, n4, n5 = fpn(p3, p4, p5)

    # ── N3 ────────────────────────────────────────────────────────────────────
    assert n3.shape == (1, _OUT, 48, 80), (
        f"N3 shape mismatch: expected (1, {_OUT}, 48, 80), got {tuple(n3.shape)}"
    )

    # ── N4 ────────────────────────────────────────────────────────────────────
    assert n4.shape == (1, _OUT, 24, 40), (
        f"N4 shape mismatch: expected (1, {_OUT}, 24, 40), got {tuple(n4.shape)}"
    )

    # ── N5 ────────────────────────────────────────────────────────────────────
    assert n5.shape == (1, _OUT, 12, 20), (
        f"N5 shape mismatch: expected (1, {_OUT}, 12, 20), got {tuple(n5.shape)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Gradient flow
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flow() -> None:
    """
    Compute loss = N3.sum() + N4.sum() + N5.sum() and call loss.backward().
    Every parameter that was used in the forward pass must have a non-None .grad.

    A None gradient means that parameter was not connected to the loss —
    i.e. the computation graph is broken somewhere in the neck.
    """
    # Use a fresh model in *train* mode so BN statistics are tracked
    # and gradients are enabled.
    model = LightweightFPN()   # defaults from config via _C3, _C4, _C5, _OUT
    model.train()

    # Input shapes from config — no literals
    p3 = torch.randn(1, _C3, 48, 80, requires_grad=True)   # (1, 64,  48, 80)
    p4 = torch.randn(1, _C4, 24, 40, requires_grad=True)   # (1, 128, 24, 40)
    p5 = torch.randn(1, _C5, 12, 20, requires_grad=True)   # (1, 256, 12, 20)

    n3, n4, n5 = model(p3, p4, p5)

    # Scalar loss — backpropagate through all three output maps.
    loss = n3.sum() + n4.sum() + n5.sum()
    loss.backward()

    # Every named parameter must have a gradient tensor (not None).
    null_grads = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]

    assert len(null_grads) == 0, (
        f"Parameters with None gradients (broken graph): {null_grads}"
    )

    # Also check the three input tensors received gradients.
    assert p3.grad is not None, "p3 has no gradient — top-down path may be detached"
    assert p4.grad is not None, "p4 has no gradient — lateral merge may be detached"
    assert p5.grad is not None, "p5 has no gradient — lateral projection may be detached"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Information flow (P5 → N3)
# ─────────────────────────────────────────────────────────────────────────────

def test_information_flow(fpn: LightweightFPN) -> None:
    """
    Set P5 = ones, P3 = P4 = zeros.
    The top-down path must carry P5 signal all the way to N3.
    Therefore N3 must NOT be all-zeros.

    Mechanism:
        L5 = lat5(ones) ≠ 0         (Kaiming init → nonzero weights)
        TD5 = L5                    ≠ 0
        TD5_up = upsample(TD5)      ≠ 0
        L4 = lat4(zeros) = 0
        TD4 = L4 + TD5_up = TD5_up ≠ 0
        TD4_up = upsample(TD4)      ≠ 0
        L3 = lat3(zeros) = 0
        TD3 = L3 + TD4_up = TD4_up ≠ 0
        N3 = out3(TD3)              ≠ 0  (Kaiming init + BN with gamma=1)
    """
    # Input shapes from config — no literals
    p3 = torch.zeros(1, _C3, 48, 80)   # (1, 64,  48, 80) all zeros
    p4 = torch.zeros(1, _C4, 24, 40)   # (1, 128, 24, 40) all zeros
    p5 = torch.ones(1,  _C5, 12, 20)   # (1, 256, 12, 20) all ones — only signal

    with torch.no_grad():
        n3, n4, n5 = fpn(p3, p4, p5)

    # N3 must contain at least some non-zero values.
    n3_abs_sum = n3.abs().sum().item()
    assert n3_abs_sum > 0.0, (
        f"N3 is all-zeros even though P5=ones — top-down path is broken. "
        f"|N3|.sum() = {n3_abs_sum}"
    )

    # N5 must also be non-zero (closest to P5 source, trivially true).
    n5_abs_sum = n5.abs().sum().item()
    assert n5_abs_sum > 0.0, (
        f"N5 is all-zeros even though P5=ones — lat5 or out5 is broken. "
        f"|N5|.sum() = {n5_abs_sum}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (python test_neck.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print(f"Config backbone_channels : {_cfg.model.backbone_channels}")
    print(f"Neck inputs  : C3={_C3}, C4={_C4}, C5={_C5}")
    print(f"Neck output  : OUT={_OUT}")
    print()

    passed = 0
    failed = 0

    _fpn = LightweightFPN().eval()

    tests = [
        ("Test 1 — Output shapes",      lambda: test_output_shapes(_fpn)),
        ("Test 2 — Gradient flow",       test_gradient_flow),
        ("Test 3 — Information flow",    lambda: test_information_flow(_fpn)),
    ]

    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {name}")
            print(f"        {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR {name}")
            print(f"        {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)
