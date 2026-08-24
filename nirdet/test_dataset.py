"""
test_dataset.py — NIRDet Phase 7 unit tests

Run with:
    python test_dataset.py --root C:/projects/nightvision/data/raw/miniNIRPed

All four tests must pass with no errors.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Make the dataset importable when run from the same directory ──────────
sys.path.insert(0, str(Path(__file__).parent))
from dataset import NIRDetDataset, collate_fn, build_train_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def header(title: str) -> None:
    sep = "─" * 60
    print(f"\n{sep}\n  {title}\n{sep}")


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def fail(msg: str) -> None:
    print(f"  ✗  {msg}", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Load test
# ─────────────────────────────────────────────────────────────────────────────

def test_load(root: Path, split: str = "train",
              img_h: int = 384, img_w: int = 640) -> NIRDetDataset:
    """
    Load the entire split without errors.
    Print: total images, total annotations, images with zero annotations.
    """
    header("Test 1 — Load test")

    ds = NIRDetDataset(root, split=split, img_h=img_h, img_w=img_w, augment=False)
    total_images = len(ds)
    total_annotations = 0
    zero_annotation_images = 0
    errors = []

    t0 = time.time()
    for idx in range(total_images):
        try:
            img, labels = ds[idx]
            n = labels.shape[0]
            total_annotations += n
            if n == 0:
                zero_annotation_images += 1
        except Exception as e:
            errors.append((idx, str(e)))

    elapsed = time.time() - t0

    if errors:
        for idx, msg in errors[:5]:
            print(f"    Error at index {idx}: {msg}", file=sys.stderr)
        fail(f"{len(errors)} images failed to load.")

    print(f"  Total images              : {total_images}")
    print(f"  Total annotations         : {total_annotations}")
    print(f"  Images with 0 annotations : {zero_annotation_images}")
    print(f"  Load time                 : {elapsed:.2f}s  "
          f"({elapsed / max(total_images, 1) * 1000:.1f} ms/image)")
    ok("All images loaded without errors.")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Shape test
# ─────────────────────────────────────────────────────────────────────────────

def test_shapes(root: Path, split: str = "train",
                img_h: int = 384, img_w: int = 640) -> None:
    """
    Single item: image shape == (1, H, W); labels shape == (N, 5).
    Batched:     images shape == (B, 1, H, W).
    """
    header("Test 2 — Shape test")

    ds = NIRDetDataset(root, split=split, img_h=img_h, img_w=img_w, augment=False)
    B = min(4, len(ds))

    # --- Single item ---
    img, labels = ds[0]
    assert img.ndim == 3, f"Expected 3-D tensor, got {img.ndim}-D"
    C, H, W = img.shape
    assert C == 1, f"Expected 1 channel, got {C}"
    assert H == img_h and W == img_w, (
        f"Expected ({img_h},{img_w}), got ({H},{W})"
    )
    assert labels.ndim == 2, f"Labels must be 2-D, got {labels.ndim}-D"
    assert labels.shape[1] == 5, (
        f"Labels must have 5 columns, got {labels.shape[1]}"
    )
    ok(f"Single item — image: {tuple(img.shape)}, labels: {tuple(labels.shape)}")

    # --- Batched ---
    loader = DataLoader(ds, batch_size=B, shuffle=False, collate_fn=collate_fn)
    batch_imgs, batch_labels = next(iter(loader))

    assert batch_imgs.shape == (B, 1, img_h, img_w), (
        f"Batched images: expected ({B},1,{img_h},{img_w}), "
        f"got {tuple(batch_imgs.shape)}"
    )
    assert isinstance(batch_labels, list) and len(batch_labels) == B, (
        "batch_labels must be a list of length B"
    )
    for i, lbl in enumerate(batch_labels):
        assert lbl.ndim == 2 and lbl.shape[1] == 5, (
            f"labels[{i}] has unexpected shape {tuple(lbl.shape)}"
        )
    ok(f"Batched  — images: {tuple(batch_imgs.shape)}, "
       f"labels: list[{B}] of (N,5) tensors")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Normalisation test
# ─────────────────────────────────────────────────────────────────────────────

def test_normalisation(root: Path, split: str = "train",
                       img_h: int = 384, img_w: int = 640,
                       n_samples: int = 20) -> None:
    """All pixel values must be in [0, 1]."""
    header("Test 3 — Normalisation test")

    ds = NIRDetDataset(root, split=split, img_h=img_h, img_w=img_w, augment=False)
    n_samples = min(n_samples, len(ds))

    for idx in range(n_samples):
        img, _ = ds[idx]
        lo, hi = img.min().item(), img.max().item()
        if lo < 0.0 or hi > 1.0:
            fail(
                f"Sample {idx}: pixel range [{lo:.4f}, {hi:.4f}] "
                "is outside [0, 1]."
            )

    ok(f"All {n_samples} sampled images have pixel values in [0, 1].")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Augmentation sanity test
# ─────────────────────────────────────────────────────────────────────────────

def test_augmentation(
    root: Path,
    split: str = "train",
    img_h: int = 384,
    img_w: int = 640,
    n_versions: int = 10,
    debug_path: str = "debug_aug.jpg",
) -> None:
    """
    Take one annotated image and produce n_versions augmented copies.
    For every copy, assert all bounding boxes satisfy:
        cx ∈ (0, 1),  cy ∈ (0, 1),  w > 0,  h > 0
    Also save one augmented image with boxes drawn to debug_aug.jpg.
    """
    header("Test 4 — Augmentation sanity test")

    ds = NIRDetDataset(root, split=split, img_h=img_h, img_w=img_w, augment=True)

    # Find the first image that has at least one annotation.
    seed_idx = None
    for i in range(len(ds)):
        _, lbl = NIRDetDataset(root, split=split, img_h=img_h, img_w=img_w,
                               augment=False)[i]
        if lbl.shape[0] > 0:
            seed_idx = i
            break

    if seed_idx is None:
        print("  ⚠  No annotated images found — skipping augmentation test.")
        return

    violations = []

    for v in range(n_versions):
        img_t, labels_t = ds[seed_idx]   # each call samples a fresh random aug

        # img_t: (1, H, W) float in [0,1]
        # labels_t: (N, 5)  columns: class, cx, cy, w, h
        for row_idx, row in enumerate(labels_t):
            cls, cx, cy, w, h = row.tolist()
            bad = []
            if not (0.0 <= cx <= 1.0):
                bad.append(f"cx={cx:.4f}")
            if not (0.0 <= cy <= 1.0):
                bad.append(f"cy={cy:.4f}")
            if w <= 0.0:
                bad.append(f"w={w:.4f}")
            if h <= 0.0:
                bad.append(f"h={h:.4f}")
            if bad:
                violations.append(
                    f"version={v}, box={row_idx}: " + ", ".join(bad)
                )

    if violations:
        for v in violations[:5]:
            print(f"    VIOLATION: {v}", file=sys.stderr)
        fail(f"{len(violations)} bounding-box violations found.")

    ok(f"All {n_versions} augmented versions have valid bounding boxes.")

    # ── Draw one augmented sample and save to disk ─────────────────────────
    img_t, labels_t = ds[seed_idx]
    img_uint8 = (img_t.squeeze(0).numpy() * 255).astype(np.uint8)
    # Convert grayscale → BGR so we can draw coloured boxes
    vis = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    H, W = vis.shape[:2]

    for row in labels_t:
        cls, cx, cy, w, h = row.tolist()
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"cls={int(cls)}",
            (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

    out_path = Path(debug_path).resolve()
    cv2.imwrite(str(out_path), vis)
    ok(f"Debug image saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIRDet dataset unit tests")
    parser.add_argument(
        "--root",
        required=True,
        help="Path to miniNIRPed root (contains images/ and labels/)",
    )
    parser.add_argument("--split", default="train", help="Which split to test")
    parser.add_argument("--img-h", type=int, default=384,
                        help="Resize height (pipeline default 384)")
    parser.add_argument("--img-w", type=int, default=640,
                        help="Resize width (pipeline default 640)")
    parser.add_argument("--debug-path", default="debug_aug.jpg")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"Root path does not exist: {root}")

    print(f"\nNIRDet Dataset Tests")
    print(f"  root   : {root}")
    print(f"  split  : {args.split}")
    print(f"  img    : {args.img_h} x {args.img_w}")

    test_load(root, split=args.split, img_h=args.img_h, img_w=args.img_w)
    test_shapes(root, split=args.split, img_h=args.img_h, img_w=args.img_w)
    test_normalisation(root, split=args.split, img_h=args.img_h, img_w=args.img_w)
    test_augmentation(root, split=args.split, img_h=args.img_h, img_w=args.img_w,
                      debug_path=args.debug_path)

    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
