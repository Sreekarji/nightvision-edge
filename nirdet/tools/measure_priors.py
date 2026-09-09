#!/usr/bin/env python
"""
tools/measure_priors.py — Derive dataset-specific constants for NIRDet.

Run this FIRST on any new dataset and after ANY resolution change.
prior_w and prior_h are normalised — they change with resolution even
for the same physical scene.

Usage:
    python tools/measure_priors.py \
        --root C:/projects/nightvision/data/raw/miniNIRPed \
        --split train --img-h 384 --img-w 640

Output: paste-ready CFG block for train.py + anomaly warnings.

When to re-run:
    - Added custom Pi captures
    - Changed img_h or img_w
    - Switched to a different NIR dataset
    - Any time the size histogram looks different from what you expect
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import NIRDetDataset

STRIDES = (8, 16, 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",     required=True)
    ap.add_argument("--split",    default="train")
    ap.add_argument("--img-h",    type=int, default=384)
    ap.add_argument("--img-w",    type=int, default=640)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    H, W = a.img_h, a.img_w

    if H % 32 or W % 32:
        sys.exit(
            f"ERROR: {W}x{H} not divisible by 32 on both axes. "
            f"Nearest valid: {round(W/32)*32}x{round(H/32)*32}"
        )

    print(f"\nMeasuring priors: {a.root}  split={a.split}  @ {W}x{H}")
    ds = NIRDetDataset(a.root, split=a.split, img_h=H, img_w=W, augment=False)
    print(f"Loaded {len(ds)} images.")

    cx_l, cy_l, w_l, h_l = [], [], [], []
    per_img_n = []
    n_empty = 0
    dup_cols = Counter()

    for i in range(len(ds)):
        _, lbl = ds[i]
        n = lbl.shape[0]
        per_img_n.append(n)
        if n == 0:
            n_empty += 1
            continue
        cx_l += lbl[:, 1].tolist()
        cy_l += lbl[:, 2].tolist()
        w_l  += lbl[:, 3].tolist()
        h_l  += lbl[:, 4].tolist()

        for s, stride in enumerate(STRIDES):
            Sh, Sw = H // stride, W // stride
            seen = set()
            for r in lbl:
                col = min(int(r[1] * Sw), Sw - 1)
                row = min(int(r[2] * Sh), Sh - 1)
                key = row * Sw + col
                if key in seen:
                    dup_cols[s] += 1
                seen.add(key)

    if not w_l:
        sys.exit("ERROR: No annotations found. Check --root and --split.")

    cx = np.array(cx_l); cy = np.array(cy_l)
    w  = np.array(w_l);  h  = np.array(h_l)
    wp = w * W;          hp = h * H
    sz = np.maximum(wp, hp)
    n_labels = len(w)
    n_images = len(ds)

    print("\n" + "=" * 68)
    print(f"  Dataset @ {W}x{H}   split={a.split}")
    print("=" * 68)
    print(f"  Images:          {n_images}")
    print(f"  Annotations:     {n_labels}")
    print(f"  Mean GT/image:   {n_labels/max(n_images,1):.2f}")
    print(f"  Empty images:    {n_empty} ({100*n_empty/max(n_images,1):.1f}%)")

    print("\n  Normalised box statistics (resolution-dependent):")
    for name, v, scale in [("w", w, W), ("h", h, H)]:
        print(f"    {name}: median={np.median(v):.4f}  mean={v.mean():.4f}  "
              f"p5={np.percentile(v,5):.4f}  p95={np.percentile(v,95):.4f}  "
              f"(median={np.median(v)*scale:.1f}px)")

    print(f"\n  Aspect w/h: median={np.median(wp/hp):.3f}")
    print(f"  Size max(w,h) px: p5={np.percentile(sz,5):.0f}  "
          f"median={np.median(sz):.0f}  p95={np.percentile(sz,95):.0f}")

    print("\n  Size histogram — use to set scale_ranges in LabelAssigner:")
    edges = [0, 16, 32, 48, 64, 96, 128, 192, 256, int(1e5)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        n  = int(((sz >= lo) & (sz < hi)).sum())
        bar = "#" * int(50 * n / max(n_labels, 1))
        lv = "P3" if hi <= 64 else ("P4" if hi <= 160 else "P5")
        hs = str(hi) if hi < 1e5 else "inf"
        print(f"    [{lo:>5},{hs:>5}) {lv}  {n:>5}  {bar}")

    med_w  = float(np.median(w))
    med_h  = float(np.median(h))
    mean_gt = n_labels / max(n_images, 1)
    p30 = float(np.percentile(sz, 30))
    p70 = float(np.percentile(sz, 70))
    p90 = float(np.percentile(sz, 90))
    s_p3_hi = max(32.0, round(p30 / 8) * 8)
    s_p4_lo = max(24.0, round(p30 * 0.75 / 8) * 8)
    s_p4_hi = max(s_p4_lo + 32, round(p90 / 8) * 8)
    s_p5_lo = max(s_p4_hi * 0.75, round(p70 / 8) * 8)

    print("\n" + "-" * 68)
    print("  PASTE INTO train.py CFG:")
    print(f'    "prior_w": {med_w:.4f},')
    print(f'    "prior_h": {med_h:.4f},')
    print(f'    # Suggested scale_ranges for LabelAssigner:')
    print(f'    # scale_ranges=((0.0,{s_p3_hi:.0f}),({s_p4_lo:.0f},{s_p4_hi:.0f}),({s_p5_lo:.0f},1e5))')
    print("-" * 68)

    warnings_found = []
    if (sz < 8).mean() > 0.02:
        warnings_found.append(
            f"{100*(sz<8).mean():.1f}% of boxes are <8px — sub-cell at P3 stride-8. "
            f"These cannot be detected. Filter them or increase resolution."
        )
    if n_empty / max(n_images, 1) > 0.35:
        warnings_found.append(
            f"{100*n_empty/n_images:.0f}% of images are empty. "
            f"High empty rate — verify batch-level loss normalisation is used."
        )
    for s, c in sorted(dup_cols.items()):
        if c > int(0.02 * n_labels):
            warnings_found.append(
                f"P{3+s} (stride {STRIDES[s]}): {c} GT pairs share a cell "
                f"(last-write-wins loses supervision). Consider cross3 assignment."
            )
    if (w > 0.9).any() or (h > 0.9).any():
        warnings_found.append("Some boxes span >90% of an axis — check for full-image labels.")

    if warnings_found:
        print("\n  ANOMALIES:")
        for msg in warnings_found:
            print(f"    ! {msg}")
    else:
        print("\n  No anomalies detected.")

    if a.json_out:
        out = {
            "prior_w": med_w, "prior_h": med_h,
            "n_images": n_images, "n_labels": n_labels,
            "mean_gt_per_image": mean_gt,
            "empty_frac": n_empty / max(n_images, 1),
            "size_p5_px": float(np.percentile(sz, 5)),
            "size_median_px": float(np.median(sz)),
            "size_p95_px": float(np.percentile(sz, 95)),
            "resolution": [W, H],
            "suggested_scale_ranges": [
                [0.0, s_p3_hi], [s_p4_lo, s_p4_hi], [s_p5_lo, 1e5]
            ],
        }
        Path(a.json_out).write_text(json.dumps(out, indent=2))
        print(f"\n  JSON saved → {a.json_out}")
    print()


if __name__ == "__main__":
    main()
