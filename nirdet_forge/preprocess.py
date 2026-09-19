"""
preprocess.py — THE preprocessing contract, PyTorch-free (F57)

numpy + cv2 ONLY. This module exists so the Pi runtime (live_nirdet.py) can
apply the deployment preprocessing chain without importing torch, which is
not installed on the deployment host. dataset.py re-exports everything here
so existing importers (`from dataset import preprocess_frame, ...`) keep
working.

THE CHAIN, IN ORDER
-------------------
    1. read as single-channel uint8          (850 nm reflective NIR)
    2. flat-field correction  (optional)     multiplicative, RAW resolution
    3. CLAHE                  (switch)       tile-local, RAW resolution
    4. letterbox to 512x288                  fixed canvas
    5. /255 -> float32 [0, 1]

Steps 2 and 3 run at raw resolution because both are illumination corrections
defined against the sensor's own geometry.

clahe_enabled IS A SWITCH, NOT A PROBABILITY. It is applied to train, val,
test and deployment alike, and it is part of the hashed deploy contract.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif",
                             ".tiff", ".pgm")

_SPLIT_ALIASES: Dict = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}

_ROOT_UNSET_MSG = (
    "cfg.data.root is unset. Apply a dataset profile before building a "
    "dataset:\n"
    "  python dataset_profiles.py --root <path> --out datasets/<n>.yaml\n"
    "  then pass --profile datasets/<n>.yaml")


def load_flat_field(path: Optional[str],
                    dtype: np.dtype = np.float32) -> Optional[np.ndarray]:
    """
    Load a flat-field correction map.

    Accepts .npy (float32, already a gain map) or any image format readable
    by OpenCV (interpreted as a uniform-surface capture and converted to a
    gain map as ``mean(f) / f``).

    A non-path argument is a TypeError BY CONTRACT (F95), not by accident of
    os.path.isfile's argument validation.
    """
    if path is None:
        return None
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(
            f"load_flat_field expects a path or None, got "
            f"{type(path).__name__}. Pass cfg.aug.flat_field_path, not cfg.")
    path = os.fspath(path)
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

    A shape mismatch is resized but WARNED ABOUT ONCE (F18): a silent resize
    applies the correction at the wrong scale, every frame, including on the
    Pi at inference time.
    """
    if flat_field is None:
        return img
    ff = flat_field
    if ff.shape != img.shape[:2]:
        if not getattr(apply_flat_field, "_warned", False):
            print(f"[prep] WARNING flat-field map is {ff.shape} but the image "
                  f"is {img.shape[:2]}; resizing the gain map on EVERY frame. "
                  f"The beam profile is fixed in sensor coordinates — capture "
                  f"the map at sensor resolution.")
            apply_flat_field._warned = True
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

    quantize_qdq.py, evaluate_onnx.py, export_ncnn.py and live_nirdet.py all
    call exactly this, so INT8 calibration, INT8 evaluation and on-device
    inference cannot drift from training.
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


def resolve_split_dirs(root: Optional[str], split: str) -> Tuple[str, str]:
    """
    Find (image_dir, label_dir) for a split under a YOLO-format root.

    An unset root raises FileNotFoundError, NOT the TypeError that
    os.path.join(None, ...) used to produce (F15) — every caller in this
    project guards on FileNotFoundError.
    """
    if not root:
        raise FileNotFoundError(_ROOT_UNSET_MSG)
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
