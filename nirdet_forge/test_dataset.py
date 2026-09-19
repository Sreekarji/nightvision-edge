"""
test_dataset.py — dataset geometry + augmentation invariants
============================================================
    python test_dataset.py

Runs with NO dataset mounted: T1-T6 and T10 use synthetic arrays and temp
label files; T7-T9 build a throwaway YOLO-layout directory tree.

Follows the check()/PASS/FAIL pattern of test_dataset_profiles.py and
test_decode_contract.py. Exit code 0 on all-pass, 1 on any failure.
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
from typing import List, Tuple

import numpy as np
import torch

from config import get_config
from dataset import (NIRPedDataset, cxcywh_to_xyxy_px, _iou_matrix,
                     boxes_to_canvas, load_flat_field, preprocess_frame,
                     read_yolo_labels)

_failures: list = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _failures.append(msg)


def _synth_ds_dir(n_images: int = 12, h: int = 300, w: int = 400,
                  splits: Tuple[str, ...] = ("train", "val", "test")
                  ) -> str:
    """A minimal YOLO-layout dataset of noise images with one box each."""
    import cv2
    root = tempfile.mkdtemp(prefix="nirdet_test_ds_")
    rng = np.random.default_rng(7)
    for sp in splits:
        os.makedirs(os.path.join(root, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", sp), exist_ok=True)
        for i in range(n_images):
            img = rng.integers(0, 256, (h, w), dtype=np.uint8)
            cv2.imwrite(os.path.join(root, "images", sp, f"{i:03d}.png"), img)
            cx, cy = 0.3 + 0.4 * (i % 5) / 4.0, 0.35 + 0.3 * (i % 3) / 2.0
            with open(os.path.join(root, "labels", sp, f"{i:03d}.txt"),
                      "w", encoding="utf-8") as fh:
                fh.write(f"0 {cx:.4f} {cy:.4f} 0.12 0.30\n")
    return root


# ===========================================================================
# T1  letterbox geometry
# ===========================================================================

def t1_letterbox() -> None:
    print("\nT1  letterbox geometry: 1280x720 -> 512x288")
    img = (np.ones((720, 1280), dtype=np.uint8) * 128)
    ret = preprocess_frame(img, 288, 512)
    check(isinstance(ret, tuple) and len(ret) == 4,
          f"returns a 4-tuple (canvas, scale, pad_x, pad_y), got len={len(ret)}")
    canvas, scale, pad_x, pad_y = ret
    check(canvas.shape == (288, 512), f"canvas.shape {canvas.shape} == (288, 512)")
    check(canvas.dtype == np.float32, f"canvas.dtype {canvas.dtype} == float32")
    check(0.0 <= float(canvas.min()) and float(canvas.max()) <= 1.0,
          f"canvas in [0,1]: min {float(canvas.min()):.4f} "
          f"max {float(canvas.max()):.4f}")
    check(abs(scale - 0.4) < 1e-6, f"scale {scale} == 0.4 (1280*0.4 == 512)")
    check(pad_x == 0 and pad_y == 0,
          f"pads zero ({pad_x}, {pad_y}) — 720*0.4 == 288 exactly")


# ===========================================================================
# T2  boxes_to_canvas round-trip, no padding
# ===========================================================================

def t2_round_trip_no_pad() -> None:
    print("\nT2  boxes_to_canvas identity when scale fits and pads are 0")
    b = np.array([[0.5, 0.5, 0.1, 0.2]], dtype=np.float32)
    out = boxes_to_canvas(b, 720, 1280, 288, 512, 0.4, 0, 0)
    check(out.shape == (1, 4), f"shape {out.shape}")
    check(bool(np.all(np.abs(out - b) < 1e-5)),
          f"unpadded geometry is an identity transform: {out[0]} == {b[0]}")


# ===========================================================================
# T3  boxes_to_canvas with padding
# ===========================================================================

def t3_padded() -> None:
    print("\nT3  boxes_to_canvas shifts by the pad")
    box = np.array([[0.5, 0.1, 0.2, 0.15]], dtype=np.float32)  # near top edge

    # NOTE: letterbox scale is min(out/w, out/h), so 640x480 -> 512x288
    # scales by 0.6 and pads HORIZONTALLY (pad_x=64, pad_y=0) — it does NOT
    # produce the vertical pad the original prompt arithmetic assumed
    # (0.8 would overflow the canvas). Both padding axes are pinned here.

    # horizontal-pad case: 640x480
    canvas, scale, pad_x, pad_y = preprocess_frame(
        np.ones((480, 640), np.uint8), 288, 512, clahe_enabled=False)
    check(scale == 0.6 and pad_x == 64 and pad_y == 0,
          f"640x480: scale {scale} pad_x {pad_x} pad_y {pad_y} (pads sideways)")
    out = boxes_to_canvas(box, 480, 640, 288, 512, scale, pad_x, pad_y)
    check(out.shape[0] == 1 and bool(((out >= 0.0) & (out <= 1.0)).all()),
          f"box stays inside [0,1] after horizontal pad shift: {out[0]}")
    # cy must NOT move: pad_y == 0
    check(abs(out[0, 1] - box[0, 1]) < 1e-6,
          f"cy unchanged {out[0, 1]:.4f} == {box[0, 1]:.4f} when pad_y == 0")

    # vertical-pad case: anamorphic source 800x300 -> 512x288
    canvas, scale, pad_x, pad_y = preprocess_frame(
        np.ones((300, 800), np.uint8), 288, 512, clahe_enabled=False)
    check(pad_y > 0 and pad_x == 0,
          f"800x300: scale {scale} pad_x {pad_x} pad_y {pad_y} (pads vertically)")
    out = boxes_to_canvas(box, 300, 800, 288, 512, scale, pad_x, pad_y)
    pure_scale = (box[0, 1] * 300 * scale) / 288
    shift = (out[0, 1] - pure_scale)
    check(shift > 0, f"top-edge box moved DOWN by the pad: shift {shift:.4f}")
    check(abs(shift - pad_y / 288.0) < 1e-6,
          f"shift {shift:.6f} == pad_y/out_h {pad_y / 288.0:.6f} "
          f"(half the 2*pad_y total padding)")
    check(0.0 <= float(out[0, 1]), f"out cy {float(out[0, 1]):.4f} >= 0")
    check(bool(((out >= 0.0) & (out <= 1.0)).all()),
          f"box fully inside [0,1] after vertical pad shift: {out[0]}")


# ===========================================================================
# T4  preprocess_frame is deterministic
# ===========================================================================

def t4_deterministic() -> None:
    print("\nT4  preprocess_frame pixel-for-pixel reproducible")
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (720, 1280), dtype=np.uint8)
    kw = dict(clahe_enabled=True, clahe_clip=2.0, clahe_grid=8)
    c1 = preprocess_frame(img.copy(), 288, 512, **kw)[0]
    c2 = preprocess_frame(img.copy(), 288, 512, **kw)[0]
    check(np.array_equal(c1, c2),
          "two identical calls -> identical canvases (no stochastic side effects)")


# ===========================================================================
# T5  kwarg contract (F42 can never re-occur silently)
# ===========================================================================

def t5_kwarg_contract() -> None:
    print("\nT5  preprocess_frame kwarg contract")
    img = (np.ones((720, 1280), np.uint8) * 128)
    try:
        preprocess_frame(img, 288, 512, apply_clahe=True)
        check(False, "apply_clahe= must raise TypeError")
    except TypeError as exc:
        check(True, f"apply_clahe=True -> TypeError ({str(exc)[:60]})")
    try:
        c, s, px, py = preprocess_frame(img, 288, 512, clahe_enabled=False)
        check(c.shape == (288, 512), "clahe_enabled=False succeeds")
    except Exception as exc:
        check(False, f"clahe_enabled=False raised {type(exc).__name__}")


# ===========================================================================
# T6  read_yolo_labels rejects degenerate boxes
# ===========================================================================

def t6_label_filter() -> None:
    print("\nT6  read_yolo_labels degeneracy + F22 oversize filter")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "labels.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("0 0.5 0.5 0.05 0.18\n")    # valid
            fh.write("0 0.5 0.5 0.0 0.18\n")     # zero width
            fh.write("0 0.5 0.5 -0.05 0.18\n")   # negative width
            fh.write("0 0.5 0.5 2.0 0.18\n")     # w > 1.5, F22 clamp
        got = read_yolo_labels(p)
        check(got.shape[0] == 1,
              f"only the valid box survives: got {got.shape[0]} of 4")
        if got.shape[0] == 1:
            check(bool(np.allclose(got[0], [0.5, 0.5, 0.05, 0.18], atol=1e-6)),
                  f"survivor is the valid row: {got[0]}")


# ===========================================================================
# T7  affine never emits boxes outside the canvas
# ===========================================================================

def t7_affine_bounds() -> None:
    print("\nT7  _augment affine-only keeps boxes inside (0,1)")
    cfg = get_config()
    a = cfg.aug
    a.affine_p, a.hflip_p = 1.0, 0.0
    a.brightness_contrast_p = a.motion_blur_p = 0.0
    a.downscale_p = a.gauss_noise_p = a.cutout_p = 0.0
    with tempfile.TemporaryDirectory() as d:
        root = _synth_ds_dir(4)
        cfg.data.root = root
        try:
            ds = NIRPedDataset(cfg, split="train", limit=4)
            ds._copy_paste_p = 0.0
            base_img = np.random.default_rng(3).random(
                (ds.img_h, ds.img_w), dtype=np.float32)
            brng = np.random.default_rng(4)
            base_boxes = np.column_stack([
                brng.uniform(0.1, 0.9, 5), brng.uniform(0.1, 0.9, 5),
                brng.uniform(0.04, 0.15, 5), brng.uniform(0.10, 0.35, 5),
            ]).astype(np.float32)

            bad = 0
            total_kept = 0
            for s in range(50):
                rng = random.Random(s)
                npg = np.random.default_rng(s)
                _, boxes = ds._augment(base_img.copy(), base_boxes.copy(),
                                       rng, 0, npg=npg)
                total_kept += boxes.shape[0]
                if boxes.shape[0]:
                    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], \
                        boxes[:, 2], boxes[:, 3]
                    ok = ((cx > 0) & (cx < 1) & (cy > 0) & (cy < 1)
                          & (bw > 0) & (bh > 0))
                    bad += int((~ok).sum())
            check(bad == 0, f"50 affine runs: {bad} out-of-canvas components")
            # guard against a vacuous pass — affine keeps must not be empty
            check(total_kept >= 50,
                  f"boxes actually survived the warp: {total_kept}/250 kept")
        except Exception as exc:
            check(False, f"affine harness raised {type(exc).__name__}: {exc}")


# ===========================================================================
# T8  copy-paste honours avoid=idx and max_iou
# ===========================================================================

def t8_copy_paste() -> None:
    print("\nT8  _copy_paste avoid=idx + copy_paste_max_iou")
    cfg = get_config()
    cfg.data.root = _synth_ds_dir(12)          # >= 10 images, all boxed
    ds = NIRPedDataset(cfg, split="train")
    h, w = ds.img_h, ds.img_w
    max_iou = float(cfg.aug.copy_paste_max_iou)

    rng_img = np.random.default_rng(21)
    base_img = rng_img.random((h, w), dtype=np.float32)
    boxes0 = np.array([[0.50, 0.50, 0.10, 0.30],
                       [0.25, 0.35, 0.08, 0.22]], dtype=np.float32)

    sources: List[int] = []
    orig_load = ds._load_base
    ds._load_base = lambda i, _o=orig_load, _r=sources: (_r.append(i), _o(i))[1]

    worst = 0.0
    n_pasted = 0
    avoid_ok = True
    for s in range(20):
        sources.clear()
        rng = random.Random(1000 + s)
        img, boxes = ds._copy_paste(base_img.copy(), boxes0.copy(), rng, idx=0)
        if any(j == 0 for j in sources):
            avoid_ok = False
            print(f"        run {s}: avoid=0 violated, sources={sources}")
        added = boxes[boxes0.shape[0]:]
        if added.shape[0]:
            a_xy = cxcywh_to_xyxy_px(added, h, w)
            o_xy = cxcywh_to_xyxy_px(boxes0, h, w)
            ious = _iou_matrix(a_xy, o_xy)
            worst = max(worst, float(ious.max()))
            n_pasted += int(added.shape[0])
    ds._load_base = orig_load
    check(avoid_ok, "source index was never the avoid index in 20 runs")
    check(n_pasted > 0, f"copy-paste actually fired: {n_pasted} boxes pasted")
    check(worst <= max_iou + 1e-6,
          f"max pasted-vs-existing IoU {worst:.4f} <= {max_iou} "
          f"(worst over {n_pasted} pastes)")


# ===========================================================================
# T9  seed wiring: same seed+epoch reproducible, epoch term mixes in
# ===========================================================================

def t9_epoch_seed() -> None:
    print("\nT9  __getitem__ seed: reproducible per epoch, varies across epochs")
    cfg = get_config()
    a = cfg.aug
    # force a continuous-valued augmentation so different seeds visibly
    # change pixels, and every other draw is inert
    a.affine_p, a.hflip_p = 1.0, 0.0
    a.brightness_contrast_p = a.motion_blur_p = 0.0
    a.downscale_p = a.gauss_noise_p = a.cutout_p = 0.0
    cfg.data.root = _synth_ds_dir(4)
    ds = NIRPedDataset(cfg, split="train", limit=4)
    ds._copy_paste_p = 0.0

    torch.manual_seed(1234)                    # fixes torch.initial_seed()
    im1 = ds[0][0].numpy().copy()
    im2 = ds[0][0].numpy().copy()              # same seed, _epoch still 0
    check(np.array_equal(im1, im2),
          "_epoch=0 twice -> pixel-identical augmentations (reproducible)")

    ds.set_epoch(1)
    im3 = ds[0][0].numpy()
    check(not np.array_equal(im1, im3),
          "_epoch=1 -> different augmentation (epoch term mixes into the seed)")
    ds.set_epoch(0)
    im4 = ds[0][0].numpy()
    check(np.array_equal(im1, im4),
          "set_epoch(0) again restores the epoch-0 stream (round trip)")


# ===========================================================================
# T10  load_flat_field contract
# ===========================================================================

def t10_flat_field() -> None:
    print("\nT10  load_flat_field contract (F41 can never re-occur silently)")
    check(load_flat_field(None) is None, "None -> None")
    try:
        load_flat_field(get_config())
        check(False, "Config object must not be accepted")
    except TypeError as exc:
        check(True, f"load_flat_field(cfg) -> TypeError ({str(exc)[:56]})")
    # Documented behaviour: a missing path RAISES FileNotFoundError rather
    # than returning None — a silent no-op would turn a typo'd flat-field
    # path into a calibration/deploy drift that nothing downstream detects.
    try:
        r = load_flat_field("/nonexistent/path.npy")
        check(False, f"missing path returned {r} instead of raising")
    except FileNotFoundError as exc:
        check("not found" in str(exc), f"missing path -> clear FileNotFoundError")


# ===========================================================================

def main() -> int:
    print("=" * 70)
    print("  dataset geometry / augmentation tests  (no dataset required)")
    print("=" * 70)
    t1_letterbox()
    t2_round_trip_no_pad()
    t3_padded()
    t4_deterministic()
    t5_kwarg_contract()
    t6_label_filter()
    t7_affine_bounds()
    t8_copy_paste()
    t9_epoch_seed()
    t10_flat_field()
    print("=" * 70)
    if _failures:
        print(f"  {len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  all dataset tests PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
