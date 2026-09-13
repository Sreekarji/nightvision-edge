"""
dataset.py — NIRPed / miniNIRPed loader and the PREPROCESSING CONTRACT
=======================================================================
This file owns the preprocessing chain. Four consumers must apply it
identically, and all four import from here rather than reimplementing:

    dataset.py        training / validation     (this file)
    quantize_qdq.py   INT8 calibration
    evaluate_onnx.py  INT8 evaluation
    live_nirdet.py    Pi 5 deployment

THE CHAIN, IN ORDER
-------------------
    1. read as single-channel uint8          (850 nm reflective NIR)
    2. flat-field correction  (optional)     multiplicative, RAW resolution
    3. CLAHE                  (switch)       tile-local, RAW resolution
    4. letterbox to 512x288                  fixed canvas
    5. /255 -> float32 [0, 1]
    6. augmentation                          TRAINING ONLY, inside the canvas

Steps 2 and 3 run at raw resolution because both are illumination
corrections and both are defined against the sensor's own geometry: the
850 nm beam profile is fixed in sensor coordinates, and CLAHE's tile grid
should partition the real field of view, not a padded canvas. Step 4 runs
BEFORE augmentation so that every geometric augmentation is expressed in
canvas coordinates — the same coordinates the loss, the head grid and the
on-device decoder all use. Augmenting first and letterboxing after would
make the effective augmentation magnitude depend on the source resolution.

At 512x288 with 1280x720 sources the letterbox scale is min(512/1280,
288/720) = 0.4 in BOTH axes, so pad_x = pad_y = 0 and canvas-normalised box
coordinates are numerically equal to the raw YOLO label values. That is why
dataset_profiles.py can derive canvas-space priors without re-deriving the
letterbox per image. The code does not assume it.

clahe_enabled IS A SWITCH, NOT A PROBABILITY
--------------------------------------------
It is applied to train, val, test and deployment alike. It is the reason
EdgeAwareAttention could drop its runtime mean-normalisation (a per-image
Div, which is SW_INT on Neural-ART). Randomising it would put the edge
statistics that calibrate_bias() measured on a different distribution than
the one seen at inference.

LABEL CACHE
-----------
__init__ scans every label file exactly once and caches the per-file box
count in ``self._box_counts``. _paste_pool() reads the cache; __getitem__
never touches a label file it has not already parsed for the sample it is
building. At 261 images the difference is invisible; at 146k NIRPed images
times num_workers it is a multi-second stall on every worker restart.
"""

from __future__ import annotations

import glob
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config, copy_paste_p_for, get_config

# OpenCV inside a DataLoader worker will otherwise spawn its own thread pool
# per worker and oversubscribe the CPU.
cv2.setNumThreads(0)

IMG_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif",
                             ".tiff", ".pgm")

_SPLIT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


# ===========================================================================
# STANDALONE PREPROCESSING — import these, never reimplement them
# ===========================================================================

def load_flat_field(path: Optional[str],
                    dtype: np.dtype = np.float32) -> Optional[np.ndarray]:
    """
    Load a flat-field correction map.

    Accepts .npy (float32, already a gain map) or any image format readable
    by OpenCV (interpreted as a uniform-surface capture and converted to a
    gain map as ``mean(f) / f``).

    Returns a float32 gain array, or None if ``path`` is None.
    """
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"flat-field map not found: {path}")

    if path.lower().endswith(".npy"):
        ff = np.load(path).astype(dtype)
    else:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"could not decode flat-field map {path}")
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        raw = raw.astype(dtype)
        m = float(raw.mean())
        if m <= 0.0:
            raise RuntimeError(f"flat-field map {path} has non-positive mean")
        ff = m / np.clip(raw, 1e-3, None)

    if ff.ndim != 2:
        raise RuntimeError(f"flat-field map must be 2-D, got {ff.shape}")
    # A gain map far from 1.0 usually means a raw capture was passed as .npy.
    g = float(ff.mean())
    if not (0.2 < g < 5.0):
        raise RuntimeError(
            f"flat-field gain map has mean {g:.3f}; expected ~1.0. A gain map "
            f"is mean(uniform_capture)/uniform_capture, not the capture.")
    return np.ascontiguousarray(ff, dtype=dtype)


def apply_flat_field(img: np.ndarray,
                     flat_field: Optional[np.ndarray]) -> np.ndarray:
    """
    Multiplicative flat-field correction on a uint8 or float32 image.

    The 850 nm illuminator beam profile is not uniform: the periphery is
    systematically darker than the centre. CLAHE is tile-local, so it only
    partially compensates for a global gradient. Returns the same dtype it
    was given.
    """
    if flat_field is None:
        return img
    ff = flat_field
    if ff.shape != img.shape[:2]:
        ff = cv2.resize(ff, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    if img.dtype == np.uint8:
        out = img.astype(np.float32) * ff
        return np.clip(out, 0.0, 255.0).astype(np.uint8)
    return (img.astype(np.float32) * ff).astype(img.dtype)


def apply_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8
                ) -> np.ndarray:
    """
    CLAHE on a single-channel uint8 image.

    Deterministic and unconditional: cfg.aug.clahe_enabled is a switch, so
    this runs on train, val, test and on the Pi. A new CLAHE object per call
    keeps this function reentrant across DataLoader workers.
    """
    if img.dtype != np.uint8:
        raise RuntimeError(f"apply_clahe expects uint8, got {img.dtype}")
    if img.ndim != 2:
        raise RuntimeError(f"apply_clahe expects HxW, got {img.shape}")
    g = max(1, int(grid))
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(g, g))
    return c.apply(img)


def letterbox(img: np.ndarray, out_h: int, out_w: int,
              pad_value: int = 0) -> Tuple[np.ndarray, float, int, int]:
    """
    Aspect-preserving resize into a fixed (out_h, out_w) canvas.

    Returns (canvas, scale, pad_x, pad_y). Box mapping is

        x_canvas = x_raw * scale + pad_x
        y_canvas = y_raw * scale + pad_y

    At 512x288 from 1280x720 the scale is 0.4 in both axes and both pads are
    zero, so this is a pure resize — but the pads are returned and used
    anyway so a different source resolution stays correct.
    """
    if img.ndim != 2:
        raise RuntimeError(f"letterbox expects HxW single channel, "
                           f"got {img.shape}")
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise RuntimeError(f"letterbox got degenerate image {img.shape}")

    scale = min(float(out_w) / float(w), float(out_h) / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    pad_x = (out_w - nw) // 2
    pad_y = (out_h - nh) // 2
    canvas = np.full((out_h, out_w), pad_value, dtype=img.dtype)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def preprocess_frame(
    img_u8: np.ndarray,
    out_h: int,
    out_w: int,
    clahe_enabled: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    flat_field: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int, int]:
    """
    THE canonical deployment chain. Returns
        (float32 canvas in [0, 1] shaped (out_h, out_w), scale, pad_x, pad_y)

    quantize_qdq.py, evaluate_onnx.py and live_nirdet.py all call exactly
    this, so INT8 calibration, INT8 evaluation and on-device inference cannot
    drift from training.
    """
    if img_u8.ndim == 3:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    if img_u8.dtype != np.uint8:
        raise RuntimeError(f"preprocess_frame expects uint8, got {img_u8.dtype}")

    img_u8 = apply_flat_field(img_u8, flat_field)
    if clahe_enabled:
        img_u8 = apply_clahe(img_u8, clahe_clip, clahe_grid)

    canvas, scale, pad_x, pad_y = letterbox(img_u8, out_h, out_w, pad_value=0)
    return canvas.astype(np.float32) / 255.0, scale, pad_x, pad_y


# ===========================================================================
# path / label helpers
# ===========================================================================

def resolve_split_dirs(root: str, split: str) -> Tuple[str, str]:
    """
    Find (image_dir, label_dir) for a split under a YOLO-format root.

    Tries the two conventional layouts and every common split alias, then
    fails with the full list of what it tried. A silent fallback to an empty
    directory here surfaces two hours later as "TAL found no positives".
    """
    aliases = _SPLIT_ALIASES.get(split, (split,))
    tried: List[str] = []
    for a in aliases:
        for img_d, lbl_d in ((os.path.join(root, "images", a),
                              os.path.join(root, "labels", a)),
                             (os.path.join(root, a, "images"),
                              os.path.join(root, a, "labels"))):
            tried.append(img_d)
            if os.path.isdir(img_d) and os.path.isdir(lbl_d):
                return img_d, lbl_d
    raise FileNotFoundError(
        f"could not locate split '{split}' under {root!r}.\n"
        f"Expected <root>/images/<split> + <root>/labels/<split>, or "
        f"<root>/<split>/images + <root>/<split>/labels.\nTried:\n  "
        + "\n  ".join(tried))


def list_images(img_dir: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(img_dir, f"*{ext}"))
        files += glob.glob(os.path.join(img_dir, f"*{ext.upper()}"))
    return sorted(set(files))


def label_path_for(img_path: str, label_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(label_dir, stem + ".txt")


def read_yolo_labels(path: str) -> np.ndarray:
    """
    Read a YOLO label file -> (N, 4) float32 of normalised (cx, cy, w, h).

    Single class: the class column is parsed and discarded. Degenerate and
    out-of-range boxes are dropped here rather than poisoning the priors.
    """
    if not os.path.isfile(path):
        return np.zeros((0, 4), dtype=np.float32)
    out: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 5:
                raise RuntimeError(
                    f"malformed label {path}:{ln}: expected "
                    f"'cls cx cy w h', got {line!r}")
            try:
                cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
            except ValueError as exc:
                raise RuntimeError(f"malformed label {path}:{ln}: {exc}") from exc
            if w <= 0.0 or h <= 0.0:
                continue
            if not (-0.5 < cx < 1.5 and -0.5 < cy < 1.5):
                continue
            out.append([cx, cy, w, h])
    if not out:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def boxes_to_canvas(boxes_n: np.ndarray, raw_h: int, raw_w: int,
                    out_h: int, out_w: int, scale: float,
                    pad_x: int, pad_y: int) -> np.ndarray:
    """Raw-normalised (cx,cy,w,h) -> canvas-normalised (cx,cy,w,h)."""
    if boxes_n.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    b = boxes_n.astype(np.float32).copy()
    cx = b[:, 0] * raw_w * scale + pad_x
    cy = b[:, 1] * raw_h * scale + pad_y
    bw = b[:, 2] * raw_w * scale
    bh = b[:, 3] * raw_h * scale
    return np.stack([cx / out_w, cy / out_h, bw / out_w, bh / out_h],
                    axis=1).astype(np.float32)


def _cxcywh_to_xyxy_px(b: np.ndarray, h: int, w: int) -> np.ndarray:
    if b.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, bw, bh = b[:, 0] * w, b[:, 1] * h, b[:, 2] * w, b[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                    axis=1).astype(np.float32)


def _xyxy_px_to_cxcywh(b: np.ndarray, h: int, w: int) -> np.ndarray:
    if b.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx = (b[:, 0] + b[:, 2]) * 0.5 / w
    cy = (b[:, 1] + b[:, 3]) * 0.5 / h
    bw = (b[:, 2] - b[:, 0]) / w
    bh = (b[:, 3] - b[:, 1]) / h
    return np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (M,4) xyxy, b (N,4) xyxy -> (M,N)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return (inter / (aa[:, None] + ab[None, :] - inter + 1e-9)).astype(np.float32)


# ===========================================================================
# Dataset
# ===========================================================================

class NIRPedDataset(Dataset):
    """
    Single-class NIR pedestrian dataset.

    __getitem__ -> (img (1, H, W) float32 in [0,1],
                    boxes (N, 4) float32 canvas-normalised (cx, cy, w, h),
                    meta dict)

    ``meta`` carries img_path, raw_h, raw_w, scale, pad_x, pad_y so evaluate.py
    can map canvas predictions back to source coordinates if it ever needs to.
    Evaluation itself is done in canvas space, which is where both the model
    and the on-device decoder live.
    """

    def __init__(self, cfg: Config, split: str = "train",
                 augment: Optional[bool] = None,
                 limit: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = str(split)
        self.augment = (split == "train") if augment is None else bool(augment)
        self.img_h = int(cfg.data.img_h)
        self.img_w = int(cfg.data.img_w)
        self.aug = cfg.aug

        self.img_dir, self.label_dir = resolve_split_dirs(cfg.data.root, split)
        self.images: List[str] = list_images(self.img_dir)
        if limit is not None:
            self.images = self.images[: int(limit)]
        if not self.images:
            raise RuntimeError(f"no images found in {self.img_dir}")

        # ---- flat-field, loaded once ----
        self.flat_field = load_flat_field(cfg.aug.flat_field_path)

        # ---- single label scan ----
        # _box_counts is the whole reason _paste_pool() is free. Do not add a
        # second read_yolo_labels() call on the __getitem__ hot path.
        self._labels: Dict[str, np.ndarray] = {}
        self._box_counts: Dict[str, int] = {}
        total_boxes = 0
        for p in self.images:
            lb = read_yolo_labels(label_path_for(p, self.label_dir))
            self._labels[p] = lb
            self._box_counts[p] = int(lb.shape[0])
            total_boxes += int(lb.shape[0])
        self.n_boxes = int(total_boxes)

        # ---- copy-paste probability is DERIVED, never hardcoded ----
        # It attacks positive scarcity, so its usefulness scales inversely
        # with how many positives the split already has.
        declared = int(getattr(cfg.data, "n_train_boxes", 0) or 0)
        self.n_train_boxes = declared if declared > 0 else self.n_boxes
        self._copy_paste_p = (
            min(float(cfg.aug.copy_paste_p_max),
                copy_paste_p_for(self.n_train_boxes))
            if self.augment else 0.0)

        self._paste_indices: List[int] = [
            i for i, p in enumerate(self.images) if self._box_counts[p] > 0
        ]

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.images)

    @property
    def copy_paste_p(self) -> float:
        return self._copy_paste_p

    def box_counts(self) -> Dict[str, int]:
        return dict(self._box_counts)

    def _paste_pool(self) -> List[int]:
        """Indices of images known to contain at least one box. Cache only."""
        return self._paste_indices

    # ------------------------------------------------------------------ #
    # base sample: chain steps 1-5, no augmentation
    # ------------------------------------------------------------------ #

    def _load_base(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        path = self.images[idx]
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise RuntimeError(f"could not decode image {path}")
        if raw.ndim != 2:
            raise RuntimeError(f"bad image shape {raw.shape} in {path}")
        raw_h, raw_w = raw.shape[:2]

        canvas, scale, pad_x, pad_y = preprocess_frame(
            raw, self.img_h, self.img_w,
            clahe_enabled=bool(self.aug.clahe_enabled),
            clahe_clip=float(self.aug.clahe_clip),
            clahe_grid=int(self.aug.clahe_grid),
            flat_field=self.flat_field,
        )

        boxes = boxes_to_canvas(self._labels[path], raw_h, raw_w,
                                self.img_h, self.img_w, scale, pad_x, pad_y)
        meta = {"img_path": path, "raw_h": raw_h, "raw_w": raw_w,
                "scale": float(scale), "pad_x": int(pad_x), "pad_y": int(pad_y)}
        return canvas, boxes, meta

    # ------------------------------------------------------------------ #
    # augmentation, all inside the fixed canvas
    # ------------------------------------------------------------------ #

    def _augment(self, img: np.ndarray, boxes: np.ndarray,
                 rng: random.Random, idx: int = -1
                 ) -> Tuple[np.ndarray, np.ndarray]:
        a = self.aug

        if rng.random() < self._copy_paste_p:
            img, boxes = self._copy_paste(img, boxes, rng, idx)

        if rng.random() < a.hflip_p:
            img = np.ascontiguousarray(img[:, ::-1])
            if boxes.shape[0]:
                boxes = boxes.copy()
                boxes[:, 0] = 1.0 - boxes[:, 0]

        if rng.random() < a.affine_p:
            img, boxes = self._affine(img, boxes, rng)

        if rng.random() < a.brightness_contrast_p:
            b = rng.uniform(-a.brightness_limit, a.brightness_limit)
            c = rng.uniform(-a.contrast_limit, a.contrast_limit)
            img = np.clip(img * (1.0 + c) + b, 0.0, 1.0)

        if rng.random() < a.motion_blur_p:
            img = self._motion_blur(img, rng, int(a.motion_blur_limit))

        if rng.random() < a.downscale_p:
            lo, hi = a.downscale_range
            f = rng.uniform(float(lo), float(hi))
            sh = max(8, int(self.img_h * f))
            sw = max(8, int(self.img_w * f))
            small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (self.img_w, self.img_h),
                             interpolation=cv2.INTER_LINEAR)

        if rng.random() < a.gauss_noise_p:
            sigma = rng.uniform(0.01, 0.05)
            img = np.clip(img + np.random.normal(0.0, sigma,
                                                 img.shape).astype(np.float32),
                          0.0, 1.0)

        if rng.random() < a.cutout_p:
            img = self._cutout(img, rng)

        return np.ascontiguousarray(img, dtype=np.float32), boxes

    def _affine(self, img: np.ndarray, boxes: np.ndarray,
                rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
        a = self.aug
        h, w = self.img_h, self.img_w
        cx, cy = w * 0.5, h * 0.5

        ang = rng.uniform(-a.affine_rotate, a.affine_rotate)
        sc = rng.uniform(float(a.affine_scale[0]), float(a.affine_scale[1]))
        tx = rng.uniform(-a.affine_translate, a.affine_translate) * w
        ty = rng.uniform(-a.affine_translate, a.affine_translate) * h
        sh = math.tan(math.radians(rng.uniform(-a.affine_shear, a.affine_shear)))

        R = np.eye(3, dtype=np.float32)
        R[:2] = cv2.getRotationMatrix2D((cx, cy), ang, sc)
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = sh
        S[0, 2] = -sh * cy
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = tx
        T[1, 2] = ty
        M = (T @ R @ S).astype(np.float32)

        out = cv2.warpAffine(img, M[:2], (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        if boxes.shape[0] == 0:
            return out, boxes

        xyxy = _cxcywh_to_xyxy_px(boxes, h, w)
        n = xyxy.shape[0]
        corners = np.stack([
            np.stack([xyxy[:, 0], xyxy[:, 1]], 1),
            np.stack([xyxy[:, 2], xyxy[:, 1]], 1),
            np.stack([xyxy[:, 2], xyxy[:, 3]], 1),
            np.stack([xyxy[:, 0], xyxy[:, 3]], 1),
        ], axis=1).reshape(-1, 2)                      # (4N, 2)
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        warped = (np.concatenate([corners, ones], 1) @ M.T)[:, :2]
        warped = warped.reshape(n, 4, 2)

        x1 = warped[:, :, 0].min(1)
        x2 = warped[:, :, 0].max(1)
        y1 = warped[:, :, 1].min(1)
        y2 = warped[:, :, 1].max(1)

        area_before = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        x1c = np.clip(x1, 0, w)
        x2c = np.clip(x2, 0, w)
        y1c = np.clip(y1, 0, h)
        y2c = np.clip(y2, 0, h)
        bw = x2c - x1c
        bh = y2c - y1c
        area_after = np.clip(bw, 0, None) * np.clip(bh, 0, None)

        keep = (bw > 2.0) & (bh > 2.0) & \
               (area_after > 0.2 * np.maximum(area_before, 1e-6))
        if not bool(keep.any()):
            return out, np.zeros((0, 4), dtype=np.float32)

        kept = np.stack([x1c, y1c, x2c, y2c], 1)[keep]
        return out, _xyxy_px_to_cxcywh(kept, h, w)

    @staticmethod
    def _motion_blur(img: np.ndarray, rng: random.Random,
                     limit: int) -> np.ndarray:
        k = max(3, int(limit))
        if k % 2 == 0:
            k += 1
        kern = np.zeros((k, k), dtype=np.float32)
        if rng.random() < 0.5:
            kern[k // 2, :] = 1.0
        else:
            kern[:, k // 2] = 1.0
        kern /= float(kern.sum())
        return cv2.filter2D(img, -1, kern)

    def _cutout(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        h, w = self.img_h, self.img_w
        fill = float(img.mean())
        for _ in range(rng.randint(1, 3)):
            cw = rng.randint(max(4, w // 32), max(8, w // 8))
            ch = rng.randint(max(4, h // 32), max(8, h // 8))
            x0 = rng.randint(0, max(0, w - cw))
            y0 = rng.randint(0, max(0, h - ch))
            img[y0:y0 + ch, x0:x0 + cw] = fill
        return img

    # ------------------------------------------------------------------ #
    # copy-paste
    # ------------------------------------------------------------------ #

    def _paste_source(self, rng: random.Random, avoid: int
                      ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (canvas, boxes) for an image GUARANTEED to contain >= 1 box.

        The pool comes from the __init__ label cache, so this loop terminates
        on the first iteration unless the chosen index is the sample being
        built. A single-retry self-paste is not sufficient: on a sparse split
        it silently degrades copy-paste into a no-op for exactly the images
        that need it most.
        """
        pool = self._paste_pool()
        if not pool:
            return None
        if len(pool) == 1 and pool[0] == avoid:
            return None
        tries = 0
        while True:
            j = pool[rng.randrange(len(pool))]
            if j != avoid:
                img, boxes, _ = self._load_base(j)
                if boxes.shape[0]:
                    return img, boxes
            tries += 1
            if tries > 4 * len(pool) + 8:
                return None

    def _copy_paste(self, img: np.ndarray, boxes: np.ndarray,
                    rng: random.Random, idx: int = -1
                    ) -> Tuple[np.ndarray, np.ndarray]:
        a = self.aug
        h, w = self.img_h, self.img_w
        cur = _cxcywh_to_xyxy_px(boxes, h, w)
        added: List[np.ndarray] = []

        n_try = rng.randint(1, max(1, int(a.copy_paste_max_objs)))
        for _ in range(n_try):
            # avoid=idx, not -1: pasting an image's own crops back into itself
            # duplicates existing positives instead of adding new ones.
            src = self._paste_source(rng, avoid=idx)
            if src is None:
                break
            s_img, s_boxes = src
            s_xyxy = _cxcywh_to_xyxy_px(s_boxes, h, w)
            bi = rng.randrange(s_xyxy.shape[0])
            sx1, sy1, sx2, sy2 = [int(round(v)) for v in s_xyxy[bi]]
            sx1 = max(0, min(sx1, w - 2))
            sy1 = max(0, min(sy1, h - 2))
            sx2 = max(sx1 + 2, min(sx2, w))
            sy2 = max(sy1 + 2, min(sy2, h))
            patch = s_img[sy1:sy2, sx1:sx2]
            if patch.size == 0:
                continue

            f = rng.uniform(float(a.copy_paste_scale[0]),
                            float(a.copy_paste_scale[1]))
            pw = max(3, int(round(patch.shape[1] * f)))
            ph = max(3, int(round(patch.shape[0] * f)))
            if pw >= w or ph >= h:
                continue
            patch = cv2.resize(patch, (pw, ph),
                               interpolation=cv2.INTER_LINEAR)

            placed = False
            for _ in range(12):
                x0 = rng.randint(0, w - pw)
                y0 = rng.randint(0, h - ph)
                cand = np.array([[x0, y0, x0 + pw, y0 + ph]], dtype=np.float32)
                ref = cur if len(added) == 0 else np.concatenate(
                    [cur, np.stack(added)], 0)
                if ref.shape[0]:
                    if float(_iou_matrix(cand, ref).max()) > float(a.copy_paste_max_iou):
                        continue
                placed = True
                break
            if not placed:
                continue

            fe = max(0, int(a.copy_paste_feather))
            alpha = np.ones((ph, pw), dtype=np.float32)
            if fe > 0:
                k = 2 * fe + 1
                inner = np.zeros_like(alpha)
                iy0, iy1 = min(fe, ph // 2), max(ph - fe, ph // 2 + 1)
                ix0, ix1 = min(fe, pw // 2), max(pw - fe, pw // 2 + 1)
                inner[iy0:iy1, ix0:ix1] = 1.0
                alpha = cv2.GaussianBlur(inner, (k, k), 0)
                alpha = np.clip(alpha, 0.0, 1.0)

            roi = img[y0:y0 + ph, x0:x0 + pw]
            img[y0:y0 + ph, x0:x0 + pw] = alpha * patch + (1.0 - alpha) * roi
            added.append(np.array([x0, y0, x0 + pw, y0 + ph], dtype=np.float32))

        if not added:
            return img, boxes
        all_xyxy = np.concatenate([cur, np.stack(added)], 0) if cur.shape[0] \
            else np.stack(added)
        return img, _xyxy_px_to_cxcywh(all_xyxy, h, w)

    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int):
        img, boxes, meta = self._load_base(idx)

        if self.augment:
            # Per-sample RNG seeded from the worker's torch seed so that
            # augmentation is reproducible under a fixed cfg.train.seed and
            # still differs across workers and epochs.
            seed = int(torch.initial_seed() % (2 ** 31 - 1)) ^ (idx * 2654435761 % (2 ** 31))
            rng = random.Random(seed)
            img, boxes = self._augment(img, boxes, rng, idx)

        if img.shape != (self.img_h, self.img_w):
            raise RuntimeError(
                f"bad image shape {img.shape} in {meta['img_path']}: expected "
                f"({self.img_h}, {self.img_w}) after letterbox")

        if boxes.shape[0]:
            boxes = np.clip(boxes, 0.0, 1.0).astype(np.float32)
            keep = (boxes[:, 2] > 2.0 / self.img_w) & \
                   (boxes[:, 3] > 2.0 / self.img_h)
            boxes = boxes[keep]

        t_img = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float()
        t_box = torch.from_numpy(np.ascontiguousarray(boxes)).float() \
            if boxes.shape[0] else torch.zeros((0, 4), dtype=torch.float32)
        return t_img, t_box, meta


# ===========================================================================
# collate / loaders
# ===========================================================================

def seed_worker(worker_id: int) -> None:
    """Per-worker seeding for numpy and random global generators.

    PyTorch seeds torch per worker automatically, but NOT numpy. Without this,
    every worker draws the same Gaussian-noise sequence, reducing augmentation
    variance and making results non-reproducible under a fixed cfg.train.seed.
    """
    base = torch.initial_seed() % (2 ** 31 - 1)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
    cv2.setNumThreads(0)


def collate_fn(batch):
    """-> (imgs (B,1,H,W), targets list of (N_i,4), metas list of dict)."""
    imgs = torch.stack([b[0] for b in batch], 0)
    targets = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return imgs, targets, metas


def build_dataloader(cfg: Config, split: str, batch_size: Optional[int] = None,
                     shuffle: Optional[bool] = None,
                     augment: Optional[bool] = None,
                     limit: Optional[int] = None,
                     num_workers: Optional[int] = None
                     ) -> Tuple[DataLoader, NIRPedDataset]:
    ds = NIRPedDataset(cfg, split=split, augment=augment, limit=limit)
    bs = int(cfg.train.batch_size if batch_size is None else batch_size)
    sh = (split == "train") if shuffle is None else bool(shuffle)
    nw = int(cfg.data.num_workers if num_workers is None else num_workers)
    g = torch.Generator()
    g.manual_seed(int(cfg.train.seed))
    dl = DataLoader(
        ds, batch_size=bs, shuffle=sh, num_workers=nw,
        pin_memory=bool(cfg.data.pin_memory), collate_fn=collate_fn,
        drop_last=(split == "train" and len(ds) > bs),
        persistent_workers=bool(nw > 0),
        worker_init_fn=seed_worker, generator=g,
    )
    return dl, ds


def build_dataloaders(cfg: Config) -> Dict[str, Tuple[DataLoader, NIRPedDataset]]:
    out: Dict[str, Tuple[DataLoader, NIRPedDataset]] = {}
    for split in ("train", "val", "test"):
        try:
            out[split] = build_dataloader(
                cfg, split,
                batch_size=cfg.train.batch_size if split == "train" else 1,
                shuffle=(split == "train"),
                augment=(split == "train"),
            )
        except FileNotFoundError as exc:
            print(f"[data] split '{split}' unavailable: {exc}")
    return out


if __name__ == "__main__":
    cfg = get_config()
    print(f"root: {cfg.data.root}")
    try:
        dl, ds = build_dataloader(cfg, "train", num_workers=0)
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1)

    print(f"images        : {len(ds)}")
    print(f"boxes         : {ds.n_boxes}")
    print(f"copy_paste_p  : {ds.copy_paste_p:.3f}  (derived from "
          f"{ds.n_train_boxes} boxes)")
    print(f"flat-field    : {cfg.aug.flat_field_path}")
    print(f"clahe_enabled : {cfg.aug.clahe_enabled}")

    imgs, tgts, metas = next(iter(dl))
    print(f"batch imgs    : {tuple(imgs.shape)} "
          f"[{float(imgs.min()):.3f}, {float(imgs.max()):.3f}]")
    print(f"boxes/img     : {[int(t.shape[0]) for t in tgts]}")

    # preprocessing determinism: the deployment chain must be a pure function
    raw = cv2.imread(ds.images[0], cv2.IMREAD_GRAYSCALE)
    a, *_ = preprocess_frame(raw, cfg.data.img_h, cfg.data.img_w,
                             cfg.aug.clahe_enabled, cfg.aug.clahe_clip,
                             cfg.aug.clahe_grid, ds.flat_field)
    b, *_ = preprocess_frame(raw, cfg.data.img_h, cfg.data.img_w,
                             cfg.aug.clahe_enabled, cfg.aug.clahe_clip,
                             cfg.aug.clahe_grid, ds.flat_field)
    print(f"preprocess deterministic: {bool(np.array_equal(a, b))}")
