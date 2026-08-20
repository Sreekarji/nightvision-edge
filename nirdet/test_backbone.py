"""
test_backbone.py — NIRDet Phase 1 unit tests
=============================================
Run:  python test_backbone.py

All four tests must PASS before the backbone is integrated into Phase 2.
A failed test prints a descriptive error and exits with code 1.
"""

import sys
import math
import torch
import torch.nn as nn

# ── import the backbone under test ──────────────────────────────────────────
try:
    from backbone import NIRBackbone, NIRStem
except ImportError as e:
    print(f"[FATAL] Could not import backbone.py: {e}")
    sys.exit(1)


# ── helpers ─────────────────────────────────────────────────────────────────

PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
SEP   = "─" * 60


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def assert_eq(label: str, got, expected) -> None:
    ok = (got == expected)
    mark = PASS if ok else FAIL
    print(f"  {mark}  {label}: got {got}, expected {expected}")
    if not ok:
        raise AssertionError(f"{label} mismatch: {got} != {expected}")


def assert_true(label: str, condition: bool, detail: str = "") -> None:
    mark = PASS if condition else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  {mark}  {label}{suffix}")
    if not condition:
        raise AssertionError(f"{label} failed. {detail}")


# ── Test 1: output shapes ────────────────────────────────────────────────────

def test_shapes() -> None:
    section("TEST 1 — Output shapes  (640 × 640 NIR input)")

    model = NIRBackbone(base_ch=32)
    model.eval()

    x = torch.zeros(1, 1, 640, 640)
    print(f"  Input shape:  {tuple(x.shape)}")

    with torch.no_grad():
        P3, P4, P5 = model(x)

    print(f"  P3 shape:     {tuple(P3.shape)}")
    print(f"  P4 shape:     {tuple(P4.shape)}")
    print(f"  P5 shape:     {tuple(P5.shape)}")

    # Batch dimension
    assert_eq("P3 batch",   P3.shape[0], 1)
    assert_eq("P4 batch",   P4.shape[0], 1)
    assert_eq("P5 batch",   P5.shape[0], 1)

    # Channel counts
    C3, C4, C5 = model.out_channels
    assert_eq("P3 channels", P3.shape[1], C3)   # 128
    assert_eq("P4 channels", P4.shape[1], C4)   # 256
    assert_eq("P5 channels", P5.shape[1], C5)   # 256

    # Spatial dimensions — stride 8, 16, 32
    assert_eq("P3 height", P3.shape[2], 640 // 8)   # 80
    assert_eq("P3 width",  P3.shape[3], 640 // 8)   # 80
    assert_eq("P4 height", P4.shape[2], 640 // 16)  # 40
    assert_eq("P4 width",  P4.shape[3], 640 // 16)  # 40
    assert_eq("P5 height", P5.shape[2], 640 // 32)  # 20
    assert_eq("P5 width",  P5.shape[3], 640 // 32)  # 20

    print(f"\n  Backbone out_channels attribute: {model.out_channels}")
    print(f"  {PASS}  All shape assertions passed.")


# ── Test 2: parameter count ──────────────────────────────────────────────────

def test_params() -> None:
    section("TEST 2 — Parameter count  (must be under 2,500,000)")

    model = NIRBackbone(base_ch=32)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Per-module breakdown (useful for debugging budget overruns)
    print("\n  Per-module breakdown:")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"    {name:15s}: {n:>10,} params")

    limit = 2_500_000
    assert_true(
        f"Total params ({total_params:,}) < {limit:,}",
        total_params < limit,
        f"Exceeded by {total_params - limit:,}" if total_params >= limit else "",
    )
    print(f"\n  {PASS}  Parameter budget satisfied.")


# ── Test 3: gradient flow ────────────────────────────────────────────────────

def test_gradients() -> None:
    section("TEST 3 — Gradient flow  (no None gradients after backward)")

    model = NIRBackbone(base_ch=32)
    model.train()

    x     = torch.zeros(1, 1, 640, 640, requires_grad=False)
    P3, P4, P5 = model(x)

    loss = P3.sum() + P4.sum() + P5.sum()
    loss.backward()

    none_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            none_grads.append(name)

    if none_grads:
        print(f"  {FAIL}  Parameters with None gradient:")
        for n in none_grads:
            print(f"    • {n}")
        raise AssertionError(f"{len(none_grads)} parameter(s) received no gradient.")

    # Also verify at least one gradient has non-zero magnitude
    # (catches the degenerate case where all grads are zero tensors, not None)
    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    max_grad = max(grad_norms)
    assert_true(
        "At least one non-zero gradient exists",
        max_grad > 0,
        f"max grad norm = {max_grad:.6f}",
    )

    print(f"\n  Gradient stats:")
    print(f"    # params with grad : {len(grad_norms)}")
    print(f"    max |grad|         : {max_grad:.6e}")
    print(f"    min |grad|         : {min(grad_norms):.6e}")
    print(f"\n  {PASS}  All parameters received valid gradients.")


# ── Test 4: edge initialisation verification ─────────────────────────────────

def test_edge_init() -> None:
    section("TEST 4 — Edge-kernel initialisation")

    stem = NIRStem(out_ch=32, n_edge_init=6)

    # Retrieve first-layer weights: shape (32, 1, 3, 3)
    w = stem.conv.weight.data
    print(f"  Stem weight tensor shape: {tuple(w.shape)}")

    # ── Canonical Sobel Gx (normalised) ─────────────────────────────────
    Gx_raw = torch.tensor([
        [-1.,  0.,  1.],
        [-2.,  0.,  2.],
        [-1.,  0.,  1.],
    ])
    Gx_norm = Gx_raw / Gx_raw.norm()

    # ── Check first 6 kernels resemble Sobel/Laplacian patterns ─────────
    print("\n  First 6 kernel values (should reflect Sobel/Laplacian patterns):")
    print(f"  {'Kernel':>8}  {'Max|w|':>8}  {'Cosine similarity with Gx':>28}  {'Interpretation'}")
    print(f"  {'------':>8}  {'------':>8}  {'-------------------------':>28}  {'-------------'}")

    interpretations = ["Sobel Gx", "Sobel Gy", "Sobel 45°", "Sobel 135°", "Laplacian", "Laplacian-D"]

    for i in range(6):
        k = w[i, 0]                    # (3, 3)
        max_val = k.abs().max().item()
        # Cosine similarity with Gx to give a rough "edge-ness" measure
        cos_sim = torch.nn.functional.cosine_similarity(
            k.flatten().unsqueeze(0),
            Gx_norm.flatten().unsqueeze(0),
        ).item()
        print(f"  {i:>8d}  {max_val:>8.4f}  {cos_sim:>28.4f}  {interpretations[i]}")

    # ── Verification criterion ───────────────────────────────────────────
    # The first 6 kernels should NOT be all-zero (Kaiming can produce small values,
    # but an edge kernel has guaranteed non-zero structure).
    all_nonzero = all(w[i, 0].abs().max().item() > 1e-6 for i in range(6))
    assert_true("First 6 kernels are non-zero", all_nonzero)

    # The first kernel (Sobel Gx) should have a positive centre-column,
    # negative left-column structure: check sign pattern
    k0 = w[0, 0]   # Sobel Gx
    left_col_neg  = (k0[:, 0] < 0).all().item()
    right_col_pos = (k0[:, 2] > 0).all().item()
    centre_zero   = (k0[:, 1].abs() < 1e-6).all().item()

    assert_true("Kernel 0 left column is negative (Sobel Gx pattern)",  left_col_neg)
    assert_true("Kernel 0 right column is positive (Sobel Gx pattern)", right_col_pos)
    assert_true("Kernel 0 centre column is zero (Sobel Gx pattern)",    centre_zero)

    # Kernels 6–31 should be Kaiming-style: diverse, not all matching Gx
    kaiming_kernels = w[6:]                          # (26, 1, 3, 3)
    cos_sims_random = torch.nn.functional.cosine_similarity(
        kaiming_kernels.view(26, -1),
        Gx_norm.flatten().unsqueeze(0).expand(26, -1),
    )
    # If all random kernels had |cos_sim| > 0.9 they'd also be Sobel — they shouldn't
    all_different = (cos_sims_random.abs() < 0.95).all().item()
    assert_true(
        "Kernels 6–31 are diverse (Kaiming, not copies of Gx)",
        all_different,
        f"max |cos_sim| = {cos_sims_random.abs().max().item():.4f}",
    )

    print(f"\n  {PASS}  Edge-kernel initialisation verified.")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  NIRDet backbone.py — Unit Test Suite")
    print("═" * 60)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")

    results = {}

    for name, fn in [
        ("shapes",    test_shapes),
        ("params",    test_params),
        ("gradients", test_gradients),
        ("edge_init", test_edge_init),
    ]:
        try:
            fn()
            results[name] = True
        except AssertionError as e:
            print(f"\n  {FAIL}  {name}: {e}")
            results[name] = False
        except Exception as e:
            print(f"\n  {FAIL}  {name}: unexpected error — {e}")
            results[name] = False

    # ── Summary ─────────────────────────────────────────────────────────
    section("SUMMARY")
    all_passed = True
    for name, ok in results.items():
        mark = PASS if ok else FAIL
        print(f"  {mark}  {name}")
        all_passed = all_passed and ok

    print()
    if all_passed:
        print("  ✅  All tests passed.  backbone.py is ready for Phase 2.")
    else:
        print("  ❌  One or more tests failed.  See output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
