"""
dataset.py — NIRPed-format loader and the PREPROCESSING CONTRACT
==================================================================
The preprocessing chain itself lives in preprocess.py (numpy+cv2 only) and is
re-exported here. Four consumers apply it identically: this file, quantize_qdq,
evaluate_onnx and live_nirdet.

SCALE NOTES (NIRPed val split used as train set, ~10-100x miniNIRPed)
---------------------------------------------------------------------
* The label cache key is (root-relative path, size, mtime_ns) per label file,
  NOT the file CONTENTS. Hashing contents re-read every label byte on every
  DataLoader construction — at 100k labels that is the same I/O the cache was
  supposed to avoid, paid three times per run (train/val/test loaders).
* The cache is written next to the dataset if writable, otherwise into a
  per-user temp directory: a 38 GB dataset is very often on a read-only mount.
* read_yolo_labels' per-file "dropped boxes" print is rate-limited; at 100k
  files an unconditional print is megabytes of log noise that hides real
  warnings.
* The copy-paste patch cache is bounded by COUNT and the paste pool is an
  index list, both independent of dataset size.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config, copy_paste_p_for, get_config
from preprocess import *          # chain lives in preprocess.py
from preprocess import (IMG_EXTS, _ROOT_UNSET_MSG, _SPLIT_ALIASES,
                        apply_clahe, apply_flat_field, letterbox,
                        list_images, load_flat_field, preprocess_frame,
                        resolve_split_dirs)

# cv2.setNumThreads(0) is set in seed_worker — WORKERS ONLY.

_LABEL_WARN_BUDGET = 20
_label_warns_emitted = 0


def label_path_for(img_path: str, label_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(label_dir, stem + ".txt")


def _label_warn(msg: str) -> None:
    """Rate-limited label warning: unbounded prints do not scale to 100k files."""
    global _label_warns_emitted
    if _label_warns_emitted < _LABEL_WARN_BUDGET:
        print(msg)
    elif _label_warns_emitted == _LABEL_WARN_BUDGET:
        print(f"[labels] ... further per-file label warnings suppressed "
              f"(> {_LABEL_WARN_BUDGET}); totals are reported by "
              f"dataset_profiles.py")
    _label_warns_emitted += 1


def read_yolo_labels(path: str) -> np.ndarray:
    """YOLO label file -> (N,4) float32 normalised (cx,cy,w,h). Class dropped."""
    if not os.path.isfile(path):
        return np.zeros((0, 4), dtype=np.float32)
    out: List[List[float]] = []
    n_raw = 0
    n_dropped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 5:
                _label_warn(f"[labels] {os.path.basename(path)}:{ln}: "
                            f"malformed line dropped (expected "
                            f"'cls cx cy w h', got {line!r})")
                n_raw += 1
                n_dropped += 1
                continue
            try:
                cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
            except ValueError as exc:
                _label_warn(f"[labels] {os.path.basename(path)}:{ln}: "
                            f"unparseable field dropped ({exc})")
                n_raw += 1
                n_dropped += 1
                continue
            n_raw += 1
            if not (math.isfinite(cx) and math.isfinite(cy)
                    and math.isfinite(w) and math.isfinite(h)):
                n_dropped += 1
                continue
            if w <= 0.0 or h <= 0.0:
                n_dropped += 1
                continue
            if not (-0.5 < cx < 1.5 and -0.5 < cy < 1.5):
                n_dropped += 1
                continue
            out.append([cx, cy, w, h])
    if not out:
        if n_dropped > 0:
            _label_warn(f"[labels] {os.path.basename(path)}: dropped "
                        f"{n_dropped}/{n_raw} boxes (degenerate/out-of-range)")
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(out, dtype=np.float32)
    keep = (boxes[:, 2] <= 1.5) & (boxes[:, 3] <= 1.5)
    total_dropped = n_dropped + int((~keep).sum())
    if total_dropped > 0:
        _label_warn(f"[labels] {os.path.basename(path)}: dropped "
                    f"{total_dropped}/{n_raw} boxes (degenerate/out-of-range)")
    return boxes[keep]


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


def cxcywh_to_xyxy_px(b: np.ndarray, h: int, w: int) -> np.ndarray:
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
# label cache
# ===========================================================================

def _label_scan_fingerprint(label_files: List[str], root: str = "") -> str:
    """
    STAT-based fingerprint: (root-relative path, size, mtime_ns) per file.

    Content hashing read every label byte on every construction, which is the
    cost the cache exists to remove. A stat() is ~100x cheaper and still
    detects edits, additions, removals and cross-split moves.
    """
    h = hashlib.sha256()
    for p in sorted(label_files):
        rel = (os.path.relpath(p, root).replace("\\", "/")
               if root else os.path.basename(p))
        h.update(rel.encode("utf-8"))
        try:
            st = os.stat(p)
            h.update(f"|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))
        except OSError:
            h.update(b"|missing\n")
    return h.hexdigest()


def _cache_candidates(root: str, split: str) -> List[str]:
    """Dataset-local path first, then a per-user temp path (read-only mounts)."""
    local = os.path.join(root, f".labelcache-{split}.npz")
    digest = hashlib.sha256(os.path.abspath(root).encode("utf-8")).hexdigest()[:12]
    tmp = os.path.join(tempfile.gettempdir(),
                       f"nirdet-labelcache-{digest}-{split}.npz")
    return [local, tmp]


class NIRPedDataset(Dataset):
    """
    __getitem__ -> (img (1,H,W) float32 [0,1],
                    boxes (N,4) canvas-normalised (cx,cy,w,h),
                    meta dict)
    """

    _PATCH_CACHE_MAX = 64

    def __init__(self, cfg: Config, split: str = "train",
                 augment: Optional[bool] = None,
                 limit: Optional[int] = None) -> None:
        super().__init__()
        if not cfg.data.root:
            raise FileNotFoundError(_ROOT_UNSET_MSG)
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

        self.flat_field = load_flat_field(cfg.aug.flat_field_path)

        root = cfg.data.root
        label_files = [label_path_for(p, self.label_dir) for p in self.images]
        cache_key = _label_scan_fingerprint(label_files, root)
        paths = _cache_candidates(root, split)

        labels_from_cache = None
        for cp in paths:
            try:
                cached = np.load(cp, allow_pickle=True)
            except Exception:
                continue
            try:
                if str(cached["key"]) != cache_key:
                    continue
                imgs_c = [str(p) for p in list(cached["images"])]
                if imgs_c != [str(p) for p in self.images]:
                    continue
                labs_c = list(cached["labels"])
                labels_from_cache = {
                    p: np.asarray(l, dtype=np.float32).reshape(-1, 4)
                    for p, l in zip(imgs_c, labs_c)}
                break
            except Exception:
                continue

        self._labels: Dict[str, np.ndarray] = {}
        self._box_counts: Dict[str, int] = {}
        total_boxes = 0
        if labels_from_cache is not None:
            for p in self.images:
                lb = labels_from_cache[p]
                self._labels[p] = lb
                self._box_counts[p] = int(lb.shape[0])
                total_boxes += int(lb.shape[0])
        else:
            n = len(self.images)
            report = max(1, n // 10)
            for i, p in enumerate(self.images, 1):
                lb = read_yolo_labels(label_path_for(p, self.label_dir))
                self._labels[p] = lb
                self._box_counts[p] = int(lb.shape[0])
                total_boxes += int(lb.shape[0])
                if n > 5000 and (i % report == 0 or i == n):
                    print(f"[data] label scan {split}: {i}/{n}", flush=True)
            wrote = False
            if limit is not None:
                # A truncated scan must never overwrite the full-split cache:
                # the key embeds the image list, so a 10-image --overfit-test
                # invalidates the full cache and forces a complete rescan on
                # the next real run.
                wrote = True
                print(f"[data] label cache not written (limit={int(limit)})")
            for cp in (paths if limit is None else []):
                try:
                    tmp = cp + f".tmp-{os.getpid()}"
                    np.savez(tmp, key=np.array(cache_key),
                             images=np.array(self.images, dtype=object),
                             labels=np.array([self._labels[p]
                                              for p in self.images],
                                             dtype=object))
                    os.replace(tmp + ".npz", cp)
                    wrote = True
                    break
                except OSError:
                    continue
            if not wrote:
                print(f"[data] WARNING label cache not writable "
                      f"({paths[0]} / {paths[1]}); continuing uncached")
        self.n_boxes = int(total_boxes)

        declared = int(getattr(cfg.data, "n_train_boxes", 0) or 0)
        self.n_train_boxes = declared if declared > 0 else self.n_boxes
        self._copy_paste_p = (
            min(float(cfg.aug.copy_paste_p_max),
                copy_paste_p_for(self.n_train_boxes))
            if self.augment else 0.0)

        self._paste_indices: List[int] = [
            i for i, p in enumerate(self.images) if self._box_counts[p] > 0]
        self._patch_cache: Dict[int, List[Tuple[np.ndarray, int]]] = {}
        self._epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        """Call at the top of every epoch. Only reaches workers because
        build_dataloader keeps persistent_workers off for augmented splits."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.images)

    @property
    def copy_paste_p(self) -> float:
        return self._copy_paste_p

    def box_counts(self) -> Dict[str, int]:
        return dict(self._box_counts)

    def _paste_pool(self) -> List[int]:
        return self._paste_indices

    # ---------------- base sample: chain steps 1-5 ----------------

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

    # ---------------- augmentation, inside the canvas ----------------

    def _augment(self, img: np.ndarray, boxes: np.ndarray,
                 rng: random.Random, idx: int = -1,
                 npg: Optional[np.random.Generator] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        a = self.aug
        img = img.copy()
        if npg is None:
            npg = np.random.default_rng()

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
            img = np.clip(img + npg.normal(0.0, sigma,
                                           img.shape).astype(np.float32),
                          0.0, 1.0)

        if rng.random() < a.cutout_p:
            img = self._cutout(img, rng, boxes)

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

        xyxy = cxcywh_to_xyxy_px(boxes, h, w)
        n = xyxy.shape[0]
        corners = np.stack([
            np.stack([xyxy[:, 0], xyxy[:, 1]], 1),
            np.stack([xyxy[:, 2], xyxy[:, 1]], 1),
            np.stack([xyxy[:, 2], xyxy[:, 3]], 1),
            np.stack([xyxy[:, 0], xyxy[:, 3]], 1),
        ], axis=1).reshape(-1, 2)
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        warped = (np.concatenate([corners, ones], 1) @ M.T)[:, :2].reshape(n, 4, 2)

        x1 = warped[:, :, 0].min(1)
        x2 = warped[:, :, 0].max(1)
        y1 = warped[:, :, 1].min(1)
        y2 = warped[:, :, 1].max(1)

        area_before = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        x1c, x2c = np.clip(x1, 0, w), np.clip(x2, 0, w)
        y1c, y2c = np.clip(y1, 0, h), np.clip(y2, 0, h)
        bw, bh = x2c - x1c, y2c - y1c
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

    def _cutout(self, img: np.ndarray, rng: random.Random,
                boxes: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = self.img_h, self.img_w
        fill = 0.5
        xy = (cxcywh_to_xyxy_px(boxes, h, w)
              if boxes is not None and boxes.shape[0] else None)
        for _ in range(rng.randint(1, 3)):
            for _attempt in range(8):
                cw = rng.randint(max(4, w // 32), max(8, w // 8))
                ch = rng.randint(max(4, h // 32), max(8, h // 8))
                x0 = rng.randint(0, max(0, w - cw))
                y0 = rng.randint(0, max(0, h - ch))
                if xy is not None:
                    cand = np.array([[x0, y0, x0 + cw, y0 + ch]],
                                    dtype=np.float32)
                    ix1 = np.maximum(cand[0, 0], xy[:, 0])
                    iy1 = np.maximum(cand[0, 1], xy[:, 1])
                    ix2 = np.minimum(cand[0, 2], xy[:, 2])
                    iy2 = np.minimum(cand[0, 3], xy[:, 3])
                    inter = (np.clip(ix2 - ix1, 0, None) *
                             np.clip(iy2 - iy1, 0, None))
                    area = (np.clip(xy[:, 2] - xy[:, 0], 1e-6, None) *
                            np.clip(xy[:, 3] - xy[:, 1], 1e-6, None))
                    # Never occlude more than 60% of any labelled box.
                    if float((inter / area).max()) > 0.60:
                        continue
                img[y0:y0 + ch, x0:x0 + cw] = fill
                break
        return img

    # ---------------- copy-paste (bounded patch cache) ----------------

    def _crops_for(self, j: int) -> List[Tuple[np.ndarray, int]]:
        cached = self._patch_cache.get(j)
        if cached is not None:
            return cached
        img, boxes, _ = self._load_base(j)
        xy = cxcywh_to_xyxy_px(boxes, self.img_h, self.img_w)
        crops: List[Tuple[np.ndarray, int]] = []
        for b in xy.astype(int):
            x1 = max(0, min(int(b[0]), self.img_w - 2))
            y1 = max(0, min(int(b[1]), self.img_h - 2))
            x2 = max(x1 + 2, min(int(b[2]), self.img_w))
            y2 = max(y1 + 2, min(int(b[3]), self.img_h))
            c = img[y1:y2, x1:x2]
            if c.size:
                crops.append((np.ascontiguousarray(c), int((y1 + y2) // 2)))
        if len(self._patch_cache) >= self._PATCH_CACHE_MAX:
            self._patch_cache.pop(next(iter(self._patch_cache)))
        self._patch_cache[j] = crops
        return crops

    def _paste_patch(self, rng: random.Random,
                     avoid: int) -> Optional[Tuple[np.ndarray, int]]:
        pool = self._paste_pool()
        if not pool:
            return None
        if len(pool) == 1 and pool[0] == avoid:
            return None
        # Bounded attempts: 4*len(pool) is O(dataset size) and at 100k images
        # a pathological pool would spin for millions of iterations.
        for _ in range(16):
            j = pool[rng.randrange(len(pool))]
            if j == avoid:
                continue
            crops = self._crops_for(j)
            if crops:
                return crops[rng.randrange(len(crops))]
        return None

    def _copy_paste(self, img: np.ndarray, boxes: np.ndarray,
                    rng: random.Random, idx: int = -1
                    ) -> Tuple[np.ndarray, np.ndarray]:
        a = self.aug
        h, w = self.img_h, self.img_w
        cur = cxcywh_to_xyxy_px(boxes, h, w)
        added: List[np.ndarray] = []

        n_try = rng.randint(1, max(1, int(a.copy_paste_max_objs)))
        for _ in range(n_try):
            got = self._paste_patch(rng, avoid=idx)
            if got is None:
                break
            patch, patch_row = got
            if patch.size == 0:
                continue

            f = rng.uniform(float(a.copy_paste_scale[0]),
                            float(a.copy_paste_scale[1]))
            pw = max(3, int(round(patch.shape[1] * f)))
            ph = max(3, int(round(patch.shape[0] * f)))
            if pw >= w or ph >= h:
                continue
            patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)

            # Ground-plane prior: apparent height ~ row offset from the
            # horizon, so a patch scaled by f belongs at h0 + f*(src_row-h0).
            src_row = patch_row
            h0 = float(self.aug.copy_paste_horizon_row)
            tgt_row = int(np.clip(h0 + float(f) * (src_row - h0),
                                  ph // 2, h - ph // 2))
            band = max(4, int(0.05 * h))
            lo_y = max(0, min(h - ph, tgt_row - ph // 2 - band))
            hi_y = max(lo_y, min(h - ph, tgt_row - ph // 2 + band))

            placed = False
            x0 = y0 = 0
            for _ in range(12):
                x0 = rng.randint(0, w - pw)
                y0 = int(rng.randint(lo_y, hi_y))
                cand = np.array([[x0, y0, x0 + pw, y0 + ph]], dtype=np.float32)
                ref = cur if len(added) == 0 else np.concatenate(
                    [cur, np.stack(added)], 0) if cur.shape[0] \
                    else np.stack(added)
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
                alpha = np.clip(cv2.GaussianBlur(inner, (k, k), 0), 0.0, 1.0)

            roi = img[y0:y0 + ph, x0:x0 + pw]
            img[y0:y0 + ph, x0:x0 + pw] = alpha * patch + (1.0 - alpha) * roi
            added.append(np.array([x0, y0, x0 + pw, y0 + ph], dtype=np.float32))

        if not added:
            return img, boxes
        all_xyxy = np.concatenate([cur, np.stack(added)], 0) if cur.shape[0] \
            else np.stack(added)
        return img, _xyxy_px_to_cxcywh(all_xyxy, h, w)

    # ---------------------------------------------------------------- #

    def __getitem__(self, idx: int):
        img, boxes, meta = self._load_base(idx)

        if self.augment:
            seed = (int(torch.initial_seed() % (2 ** 31 - 1))
                    ^ ((idx * 2654435761) % (2 ** 31))
                    ^ (((int(getattr(self, "_epoch", 0)) + 1) * 40503) % (2 ** 31)))
            rng = random.Random(seed)
            npg = np.random.default_rng(seed ^ 0x9E3779B9)
            img, boxes = self._augment(img, boxes, rng, idx, npg=npg)

        if img.shape != (self.img_h, self.img_w):
            raise RuntimeError(
                f"bad image shape {img.shape} in {meta['img_path']}: expected "
                f"({self.img_h}, {self.img_w}) after letterbox")

        if boxes.shape[0]:
            xy = cxcywh_to_xyxy_px(boxes, self.img_h, self.img_w)
            xy[:, 0::2] = np.clip(xy[:, 0::2], 0.0, float(self.img_w))
            xy[:, 1::2] = np.clip(xy[:, 1::2], 0.0, float(self.img_h))
            boxes = _xyxy_px_to_cxcywh(xy, self.img_h, self.img_w)
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
    base = torch.initial_seed() % (2 ** 31 - 1)
    np.random.seed(base)
    random.seed(base)
    cv2.setNumThreads(0)      # WORKERS ONLY


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    return imgs, [b[1] for b in batch], [b[2] for b in batch]


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
    overfit = limit is not None
    # persistent_workers MUST be False on any split whose Dataset carries
    # per-epoch state: workers hold a pickled copy, so set_epoch() would
    # never reach them.
    persist = bool(nw > 0) and not ds.augment

    g_shuffle = torch.Generator()
    g_shuffle.manual_seed(int(cfg.train.seed))
    kwargs = dict(
        batch_size=bs, shuffle=sh, num_workers=nw,
        pin_memory=bool(cfg.data.pin_memory), collate_fn=collate_fn,
        drop_last=(split == "train" and not overfit and len(ds) > bs),
        persistent_workers=persist,
        worker_init_fn=seed_worker, generator=g_shuffle,
    )
    if nw > 0:
        # Bounded prefetch: the default (2 * num_workers batches) holds
        # 2*nw*bs decoded canvases in RAM, which matters on a 38 GB set.
        kwargs["prefetch_factor"] = 2
    dl = DataLoader(ds, **kwargs)
    dl._nirdet_shuffle_generator = g_shuffle
    return dl, ds





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
    imgs, tgts, metas = next(iter(dl))
    print(f"batch imgs    : {tuple(imgs.shape)} "
          f"[{float(imgs.min()):.3f}, {float(imgs.max()):.3f}]")
    print(f"boxes/img     : {[int(t.shape[0]) for t in tgts]}")
    raw = cv2.imread(ds.images[0], cv2.IMREAD_GRAYSCALE)
    a, *_ = preprocess_frame(raw, cfg.data.img_h, cfg.data.img_w,
                             cfg.aug.clahe_enabled, cfg.aug.clahe_clip,
                             cfg.aug.clahe_grid, ds.flat_field)
    b, *_ = preprocess_frame(raw, cfg.data.img_h, cfg.data.img_w,
                             cfg.aug.clahe_enabled, cfg.aug.clahe_clip,
                             cfg.aug.clahe_grid, ds.flat_field)
    print(f"preprocess deterministic: {bool(np.array_equal(a, b))}")
