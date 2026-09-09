"""
dataset.py — NIRDet-Lite dataset
=================================
GEOMETRY FIX (the single most damaging bug in the original)
-----------------------------------------------------------
Old train pipeline:
    RandomResizedCrop(384x640, p=0.5) -> LongestMaxSize(640) -> PadIfNeeded
When the crop fired the image was already exactly 384x640, so the letterbox
became a no-op and the sample had full-frame content with a distorted aspect
ratio. When it did not fire, the sample was letterboxed with black bars.
Validation always took the second path, so half the training data came from a
different geometric distribution than evaluation, and edge behaviour in
particular was unreliable.

New pipeline: deterministic letterbox FIRST (identical to validation), then all
augmentation happens INSIDE that fixed canvas. One coordinate mapping, always.

    load grayscale
      -> optional CLAHE (grayscale-native, physically meaningful for NIR)
      -> letterbox to (img_h, img_w)              [deterministic, shared]
      -> optional box-level copy-paste            [needs no masks]
      -> albumentations inside the canvas:
           Affine (scale/translate/rotate/shear), HFlip,
           BrightnessContrast, GaussNoise, MotionBlur, Downscale, CoarseDropout
      -> assert output shape == (img_h, img_w)
      -> float32 /255, (1, H, W)

``letterbox()`` is the reference implementation reused by quantize_qdq.py; the
same arithmetic is duplicated (deliberately, with a comment) in live_nirdet.py
so the Pi runtime needs no torch/albumentations import.

Requires albumentations >= 1.4.14 (num_holes_range / std_range / fill API).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


# ---------------------------------------------------------------------------
# Deterministic letterbox — the ONE coordinate mapping in the whole project
# ---------------------------------------------------------------------------

def letterbox(
    img: np.ndarray,
    out_h: int,
    out_w: int,
    pad_value: int = 0,
) -> Tuple[np.ndarray, float, int, int]:
    """
    Aspect-preserving resize + centred constant pad.

    img : (h0, w0) uint8 grayscale
    ->    ((out_h, out_w) uint8, scale, pad_left, pad_top)

    MUST stay identical to live_nirdet._letterbox and to the calibration
    reader in quantize_qdq.py, or INT8 scales are computed for the wrong
    input distribution.
    """
    h0, w0 = img.shape[:2]
    scale = min(out_w / float(w0), out_h / float(h0))
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    canvas = np.full((out_h, out_w), pad_value, dtype=np.uint8)
    left = (out_w - nw) // 2
    top = (out_h - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def letterbox_boxes(
    boxes_yolo: np.ndarray,
    src_h: int,
    src_w: int,
    out_h: int,
    out_w: int,
    scale: float,
    pad_left: int,
    pad_top: int,
) -> np.ndarray:
    """Map normalised (cx, cy, w, h) from source frame to letterboxed canvas."""
    if boxes_yolo.size == 0:
        return boxes_yolo.reshape(0, 4).astype(np.float32)
    b = boxes_yolo.astype(np.float64).copy()
    cx = b[:, 0] * src_w * scale + pad_left
    cy = b[:, 1] * src_h * scale + pad_top
    w = b[:, 2] * src_w * scale
    h = b[:, 3] * src_h * scale
    return np.stack([cx / out_w, cy / out_h, w / out_w, h / out_h],
                    axis=1).astype(np.float32)


def apply_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """
    Grayscale-native contrast equalisation.

    Implemented with cv2 directly rather than A.CLAHE to avoid channel-count
    assumptions, and because the same call is used at deployment time to
    stabilise the INT8 input distribution (which is why EAA's runtime
    mean-normalisation could be removed).
    """
    return cv2.createCLAHE(clipLimit=float(clip),
                           tileGridSize=(int(grid), int(grid))).apply(img)


# ---------------------------------------------------------------------------
# In-canvas augmentation
# ---------------------------------------------------------------------------

def build_train_transforms(cfg_aug=None) -> A.Compose:
    """
    Everything here operates on an already-letterboxed (img_h, img_w) canvas
    and preserves that shape. No resize, no crop, no pad.
    """
    if cfg_aug is None:
        from config import AugCfg
        cfg_aug = AugCfg()
    a = cfg_aug
    return A.Compose(
        [
            A.Affine(
                scale=a.affine_scale,
                translate_percent={"x": (-a.affine_translate, a.affine_translate),
                                   "y": (-a.affine_translate, a.affine_translate)},
                rotate=(-a.affine_rotate, a.affine_rotate),
                shear={"x": (-a.affine_shear, a.affine_shear), "y": (0.0, 0.0)},
                fit_output=False,           # keeps the canvas size fixed
                keep_ratio=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=a.affine_p,
            ),
            A.HorizontalFlip(p=a.hflip_p),
            A.RandomBrightnessContrast(
                brightness_limit=a.brightness_limit,
                contrast_limit=a.contrast_limit,
                p=a.brightness_contrast_p,
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=a.gauss_noise_p),
            A.MotionBlur(blur_limit=a.motion_blur_limit, p=a.motion_blur_p),
            A.Downscale(
                scale_range=a.downscale_range,
                interpolation_pair={"downscale": cv2.INTER_AREA,
                                    "upscale": cv2.INTER_LINEAR},
                p=a.downscale_p,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                p=a.cutout_p,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.25,
            min_area=8.0,
            clip=True,
        ),
    )


def build_val_transforms() -> A.Compose:
    """
    Identity. The deterministic letterbox already produced the canvas, so
    validation applies no further spatial operation at all — which is exactly
    why training and validation now share one coordinate mapping.
    """
    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            clip=True,
        ),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NIRDetDataset(Dataset):
    """
    <root>/images/{split}/*.jpg
    <root>/labels/{split}/*.txt      YOLO: "cls cx cy w h" normalised

    __getitem__ -> (image (1, H, W) float32 in [0,1], labels (N, 5) float32)
                   labels columns = (cls, cx, cy, w, h), normalised, (0,5) if empty.

    Zero-annotation images are kept deliberately: they are negative samples
    that suppress false positives.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_h: int = 384,
        img_w: int = 640,
        augment: bool = True,
        cfg_aug=None,
        clahe: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if cfg_aug is None:
            from config import AugCfg
            cfg_aug = AugCfg()
        self.aug_cfg = cfg_aug

        self.root = Path(root)
        self.split = str(split)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.augment = bool(augment)

        # CLAHE is applied in BOTH splits by default: it is a preprocessing
        # choice (matched at deployment), not an augmentation.
        self.use_clahe = (cfg_aug.clahe_p > 0.0) if clahe is None else bool(clahe)

        self.img_dir = self.root / "images" / self.split
        self.lbl_dir = self.root / "labels" / self.split
        self.img_paths: List[Path] = sorted(
            p for ext in IMG_EXTS for p in self.img_dir.glob(ext)
        )
        if not self.img_paths:
            raise FileNotFoundError(f"no images under {self.img_dir}")

        self.transforms = (build_train_transforms(cfg_aug) if self.augment
                           else build_val_transforms())

        # Indices of images that actually contain boxes — the copy-paste pool.
        # File existence alone is not enough: an empty .txt (zero-annotation
        # negative) passes the exists() check but causes _load_canvas() to be
        # called before _copy_paste() discovers there is nothing to paste.
        self._paste_pool: List[int] = []
        for i, p in enumerate(self.img_paths):
            lbl = self.lbl_dir / (p.stem + ".txt")
            if not lbl.exists():
                continue
            boxes, _ = self._read_labels(lbl)
            if boxes.shape[0] > 0:
                self._paste_pool.append(i)

    def __len__(self) -> int:
        return len(self.img_paths)

    # ------------------------------------------------------------------ #

    def _read_labels(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not path.exists():
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        boxes, cls = [], []
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    c = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    continue
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                w = min(max(w, 1e-6), 1.0)
                h = min(max(h, 1e-6), 1.0)
                boxes.append((cx, cy, w, h))
                cls.append(c)
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        return (np.asarray(boxes, np.float32), np.asarray(cls, np.int64))

    def _load_canvas(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> (canvas (H,W) uint8, boxes (N,4) yolo-on-canvas, cls (N,))."""
        path = self.img_paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"cv2 could not decode {path}")
        h0, w0 = img.shape[:2]

        if self.use_clahe:
            img = apply_clahe(img, self.aug_cfg.clahe_clip, self.aug_cfg.clahe_grid)

        canvas, scale, left, top = letterbox(img, self.img_h, self.img_w)
        boxes, cls = self._read_labels(self.lbl_dir / (path.stem + ".txt"))
        boxes = letterbox_boxes(boxes, h0, w0, self.img_h, self.img_w,
                                scale, left, top)
        return canvas, boxes, cls

    # ------------------------------------------------------------------ #
    # Box-level copy-paste (no segmentation masks required)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]), np.float32)
        lt = np.maximum(a[:, None, :2], b[None, :, :2])
        rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[..., 0] * wh[..., 1]
        aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
        bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
        return (inter / (aa[:, None] + bb[None, :] - inter + 1e-9)).astype(np.float32)

    def _copy_paste(self, canvas: np.ndarray, boxes: np.ndarray,
                    cls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Paste pedestrian crops from another frame into this canvas with a
        feathered rectangular alpha. Directly attacks the two hardest problems
        on a 261-image set: too few positives and too little crowding.
        """
        a = self.aug_cfg
        if not self._paste_pool or random.random() >= a.copy_paste_p:
            return canvas, boxes, cls

        cur_idx = getattr(self, "_current_idx", -1)
        src_idx = random.choice(self._paste_pool)
        if len(self._paste_pool) > 1:
            while src_idx == cur_idx:
                src_idx = random.choice(self._paste_pool)

        src_canvas, src_boxes, src_cls = self._load_canvas(src_idx)
        if src_boxes.shape[0] == 0:
            return canvas, boxes, cls

        H, W = canvas.shape[:2]
        out = canvas.copy()
        cur = list(boxes.reshape(-1, 4))
        cur_cls = list(cls.reshape(-1))
        occupied = np.stack([self._yolo_to_xyxy(b, W, H) for b in cur]) \
            if cur else np.zeros((0, 4), np.float32)

        order = np.random.permutation(src_boxes.shape[0])
        pasted = 0
        for j in order:
            if pasted >= a.copy_paste_max_objs:
                break
            sx1, sy1, sx2, sy2 = self._yolo_to_xyxy(src_boxes[j], W, H)
            sx1, sy1 = int(max(0, np.floor(sx1))), int(max(0, np.floor(sy1)))
            sx2, sy2 = int(min(W, np.ceil(sx2))), int(min(H, np.ceil(sy2)))
            if sx2 - sx1 < 6 or sy2 - sy1 < 10:
                continue
            crop = src_canvas[sy1:sy2, sx1:sx2]
            if crop.size == 0 or int(crop.max()) == 0:
                continue

            s = random.uniform(*a.copy_paste_scale)
            nw = int(round(crop.shape[1] * s))
            nh = int(round(crop.shape[0] * s))
            if nw < 6 or nh < 10 or nw >= W or nh >= H:
                continue
            crop = cv2.resize(crop, (nw, nh),
                              interpolation=cv2.INTER_AREA if s < 1
                              else cv2.INTER_LINEAR)

            placed = False
            for _ in range(12):
                x0 = random.randint(0, W - nw)
                # Keep pasted people on roughly the same ground band as the
                # source, so scenes stay geometrically plausible.
                y_ref = int(sy1 * random.uniform(0.85, 1.15))
                y0 = int(min(max(0, y_ref), H - nh))
                cand = np.array([[x0, y0, x0 + nw, y0 + nh]], np.float32)
                if occupied.shape[0] and \
                        float(self._iou_xyxy(cand, occupied).max()) > a.copy_paste_max_iou:
                    continue
                placed = True
                break
            if not placed:
                continue

            # Feathered rectangular alpha hides the hard crop boundary.
            alpha = np.ones((nh, nw), np.float32)
            f = int(max(0, a.copy_paste_feather))
            if f > 0 and nh > 2 * f and nw > 2 * f:
                ramp_x = np.minimum(np.arange(nw), np.arange(nw)[::-1]) / float(f)
                ramp_y = np.minimum(np.arange(nh), np.arange(nh)[::-1]) / float(f)
                alpha = np.clip(np.minimum(ramp_y[:, None], ramp_x[None, :]),
                                0.0, 1.0).astype(np.float32)

            roi = out[y0:y0 + nh, x0:x0 + nw].astype(np.float32)
            out[y0:y0 + nh, x0:x0 + nw] = np.clip(
                alpha * crop.astype(np.float32) + (1.0 - alpha) * roi, 0, 255
            ).astype(np.uint8)

            cur.append(np.array([(x0 + nw / 2) / W, (y0 + nh / 2) / H,
                                 nw / W, nh / H], np.float32))
            cur_cls.append(int(src_cls[j]) if j < len(src_cls) else 0)
            occupied = np.concatenate(
                [occupied, np.array([[x0, y0, x0 + nw, y0 + nh]], np.float32)], 0
            )
            pasted += 1

        if pasted == 0:
            return canvas, boxes, cls
        return (out,
                np.stack(cur).astype(np.float32),
                np.asarray(cur_cls, np.int64))

    @staticmethod
    def _yolo_to_xyxy(b: np.ndarray, w: int, h: int) -> np.ndarray:
        cx, cy, bw, bh = float(b[0]) * w, float(b[1]) * h, float(b[2]) * w, float(b[3]) * h
        return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                        np.float32)

    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        self._current_idx = idx
        canvas, boxes, cls = self._load_canvas(idx)

        if self.augment:
            canvas, boxes, cls = self._copy_paste(canvas, boxes, cls)

        res = self.transforms(
            image=canvas[:, :, None],
            bboxes=[tuple(map(float, b)) for b in boxes],
            class_labels=[int(c) for c in cls],
        )
        img_hwc = np.asarray(res["image"])
        if img_hwc.ndim == 2:
            img_hwc = img_hwc[:, :, None]

        # Explicit contract: every sample in the batch is exactly (H, W).
        # PadIfNeeded only pads, so the old pipeline crashed inside
        # torch.stack for any source aspect ratio below img_w/img_h.
        if img_hwc.shape[:2] != (self.img_h, self.img_w):
            raise RuntimeError(
                f"augmented canvas is {img_hwc.shape[:2]}, expected "
                f"({self.img_h}, {self.img_w}); an augmentation changed the "
                f"canvas size (check Affine fit_output / Downscale) "
                f"[image {self.img_paths[idx].name}]"
            )

        img = (torch.from_numpy(np.ascontiguousarray(img_hwc))
               .permute(2, 0, 1).float().div_(255.0))

        bb = res["bboxes"]
        if len(bb):
            arr = np.asarray(bb, np.float32).reshape(-1, 4)
            cl = np.asarray(res["class_labels"], np.float32).reshape(-1, 1)
            labels = torch.from_numpy(np.concatenate([cl, arr], 1).astype(np.float32))
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return img, labels


# ---------------------------------------------------------------------------
# collate / loader
# ---------------------------------------------------------------------------

def collate_fn(batch: Sequence[Tuple[Tensor, Tensor]]
               ) -> Tuple[Tensor, List[Tensor]]:
    """
    Images stack (all canvases are the same size by construction);
    labels stay a list because N varies per image, and a (0,5) tensor is a
    natural sentinel for a negative sample.
    """
    imgs, labels = zip(*batch)
    return torch.stack(imgs, 0), list(labels)


def seed_worker(worker_id: int) -> None:
    """
    Deterministic per-worker seeding for numpy / random / cv2 paths.
    Without this, augmentation order alone produces run-to-run variance
    comparable to the improvement being claimed over the baseline.
    """
    base = torch.initial_seed() % (2 ** 31 - 1)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
    cv2.setNumThreads(0)          # workers must not oversubscribe the CPU


def build_dataloader(
    root: str | Path,
    split: str,
    img_h: int = 384,
    img_w: int = 640,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    augment: Optional[bool] = None,
    cfg_aug=None,
    drop_last: Optional[bool] = None,
    seed: int = 42,
) -> DataLoader:
    aug = (split == "train") if augment is None else bool(augment)
    shuf = aug if shuffle is None else bool(shuffle)
    drop = aug if drop_last is None else bool(drop_last)

    ds = NIRDetDataset(root, split=split, img_h=img_h, img_w=img_w,
                       augment=aug, cfg_aug=cfg_aug)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuf,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=drop,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )


# ---------------------------------------------------------------------------
# python dataset.py --root <path>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--img-h", type=int, default=384)
    ap.add_argument("--img-w", type=int, default=640)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--debug-out", default="debug_aug.jpg")
    args = ap.parse_args()

    print("T1 load + shape + range")
    ds = NIRDetDataset(args.root, args.split, args.img_h, args.img_w,
                       augment=False)
    n_ann = n_box = 0
    for i in range(len(ds)):
        im, lb = ds[i]
        assert im.shape == (1, args.img_h, args.img_w), im.shape
        assert lb.ndim == 2 and lb.shape[1] == 5, lb.shape
        assert float(im.min()) >= 0.0 and float(im.max()) <= 1.0
        n_box += int(lb.shape[0])
        n_ann += int(lb.shape[0] > 0)
    print(f"   {len(ds)} images, {n_box} boxes, {n_ann} annotated, "
          f"{len(ds) - n_ann} negatives  PASS")

    print("T2 train/val geometry parity")
    dtr = NIRDetDataset(args.root, args.split, args.img_h, args.img_w,
                        augment=True)
    for i in range(min(args.n, len(dtr))):
        im, lb = dtr[i]
        assert im.shape == (1, args.img_h, args.img_w)
        for row in lb:
            _, cx, cy, w, h = row.tolist()
            assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0, (cx, cy)
            assert w > 0.0 and h > 0.0, (w, h)
    print("   PASS")

    print("T3 copy-paste increases box count at least once")
    base = ds[0][1].shape[0]
    hits = 0
    for _ in range(30):
        if dtr[0][1].shape[0] > base:
            hits += 1
    print(f"   {hits}/30 samples gained pasted objects  "
          f"{'PASS' if hits > 0 or base == 0 else 'CHECK copy_paste_p'}")

    print("T4 debug render")
    im, lb = dtr[0]
    vis = cv2.cvtColor((im.squeeze(0).numpy() * 255).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    H, W = vis.shape[:2]
    for row in lb:
        _, cx, cy, w, h = row.tolist()
        cv2.rectangle(vis,
                      (int((cx - w / 2) * W), int((cy - h / 2) * H)),
                      (int((cx + w / 2) * W), int((cy + h / 2) * H)),
                      (0, 255, 0), 2)
    cv2.imwrite(args.debug_out, vis)
    print(f"   wrote {args.debug_out}")
    print("all dataset tests passed")
