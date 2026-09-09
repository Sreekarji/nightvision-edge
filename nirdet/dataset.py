"""
dataset.py — NIRDet Phase 7
Single-channel NIR pedestrian detection dataset.

Layout expected:
  <root>/images/train/*.jpg   (or val)
  <root>/labels/train/*.txt   (YOLO format: "0 cx cy w h" per line)
  <root>/labels/val/*.txt

Returns per sample:
  image  : torch.float32  (1, H, W)   pixels in [0, 1]
  labels : torch.float32  (N, 5)      columns = (class, cx, cy, w, h)
           shape is (0, 5) for images that have no annotations
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.core.bbox_utils import BboxProcessor


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def build_train_transforms(img_h: int = 384, img_w: int = 640) -> A.Compose:
    """
    Albumentations pipeline for training.

    Chosen over torchvision.transforms.v2 because:
      - A.Compose natively propagates bounding-box coordinates through every
        spatial transform, with built-in clip-and-filter so boxes never leave
        [0,1].  torchvision.v2 can do this too but requires wrapping labels in
        TVTensor objects and is slower on CPU by ~240 % (benchmark: rumn/medium,
        2024; albumentations.ai/docs/benchmarks).
      - albumentations works on NumPy uint8 arrays (the native cv2 output),
        removing a redundant PIL/tensor conversion step in the hot path.

    Valid augmentations for single-channel NIR grayscale
    -------------------------------------------------------
    HorizontalFlip   — pedestrians are laterally symmetric; NIR intensity
                       is unaffected. Standard in all pedestrian detectors.
    RandomBrightnessContrast — NIR cameras are sensitive to illumination
                       variation (IR flood, ambient light, distance).
                       Simulates those changes without altering spatial
                       semantics.  Operates per-pixel, channel-agnostic.
    RandomCrop / LongestMaxSize + PadIfNeeded — spatial crops preserve
                       relative object sizes and NIR texture. Safe.
    GaussNoise       — sensor noise is prominent in NIR; adding it during
                       training improves robustness.
    CoarseDropout    — random-erase variant; occlusion simulation.
                       Bounding boxes are preserved as-is (the erased region
                       is inside the image, not a spatial warp).

    Invalid augmentations for NIR grayscale (excluded)
    -------------------------------------------------------
    HueSaturationValue / RGBShift / ChannelShuffle
                     — NIR images are single channel; hue and saturation do
                       not exist.  Applying these would either error or
                       silently operate on a broadcast 3-channel copy,
                       introducing artefacts meaningless for grayscale.
    VerticalFlip     — Pedestrians in surveillance / nightvision footage are
                       always upright.  Flipping vertically produces a
                       physically impossible orientation that the model will
                       never encounter at inference.  This is confirmed by
                       pedestrian-detection literature (e.g. miniNIRPed
                       imagery is from fixed overhead / angled cameras).
    Rotate (large angle) — Same reasoning as vertical flip; large rotations
                       produce non-real orientations.
    Mosaic           — Discussed separately in Q2; excluded at the Dataset
                       level.  If desired, implement as a collate-level
                       transform so bounding-box bookkeeping stays clean.
    """
    return A.Compose(
        [
            # --- Geometric (bbox-aware) ---
            # FIX: crop/scale happens BEFORE the letterbox (LongestMaxSize +
            # PadIfNeeded), not after. Cropping after padding crops into the
            # black pad border / rescaled content, corrupting box coords.
            A.RandomResizedCrop(
                size=(img_h, img_w),
                scale=(0.8, 1.0),
                ratio=(1.2, 2.2),   # recentred on W/H = 640/384 = 1.667; the square
                                    # default (0.75, 1.33) squashes box w/h toward 1.0
                                    # at a non-square target (measured +9.5% + dropout)
                p=0.5,
            ),
            A.LongestMaxSize(max_size=max(img_h, img_w), p=1.0),
            A.PadIfNeeded(
                min_height=img_h,
                min_width=img_w,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            # --- Photometric (image only, bbox unchanged) ---
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            # FIX: shrink holes 16-48 → 8-16 px and p 0.2 → 0.1. 48 px holes at
            # 384×640 can fully erase a small pedestrian (~30-50 px tall).
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                p=0.1,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",          # (cx, cy, w, h), all normalised [0,1]
            label_fields=["class_labels"],
            min_visibility=0.2,     # drop boxes that become < 20 % visible
            clip=True,              # clamp boxes to [0,1]
        ),
    )


def build_val_transforms(img_h: int = 384, img_w: int = 640) -> A.Compose:
    """Deterministic resize + pad only — no augmentation."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(img_h, img_w), p=1.0),
            A.PadIfNeeded(
                min_height=img_h,
                min_width=img_w,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
        ],
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
    PyTorch Dataset for single-channel NIR pedestrian detection.

    Parameters
    ----------
    root : str | Path
        Root directory that contains ``images/{split}`` and
        ``labels/{split}`` sub-directories.
    split : str
        "train" or "val".
    img_h : int
        Target image height (applied by the augmentation pipeline).
    img_w : int
        Target image width (applied by the augmentation pipeline).
    augment : bool
        If True, use the training augmentation pipeline; otherwise use the
        deterministic validation pipeline.

    __getitem__ returns
    -------------------
    image  : torch.float32 tensor of shape (1, img_h, img_w)
             Values are in [0.0, 1.0].  No ImageNet normalisation is applied
             because ImageNet statistics are computed from RGB imagery.  For
             single-channel NIR, the correct approach is either:
               (a) [0, 1] range normalisation (done here) and let the
                   first conv adapt, or
               (b) compute dataset-specific mean/std and pass them to the
                   trainer's Normalize call.
             We choose (a) so this Dataset stays decoupled from a particular
             per-dataset statistic.

    labels : torch.float32 tensor of shape (N, 5)
             Each row = (class_id, cx, cy, w, h) with coordinates
             normalised to [0, 1].  If the image has no annotations the
             tensor has shape (0, 5).  Keeping zero-annotation images is
             intentional — they act as negative training samples that
             suppress false positives (Ultralytics YOLOv5 docs; TraCon
             paper, 2022).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_h: int = 384,
        img_w: int = 640,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_h = img_h
        self.img_w = img_w

        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split

        # Collect all image paths; sort for reproducibility.
        self.img_paths: List[Path] = sorted(self.img_dir.glob("*.jpg"))
        if not self.img_paths:
            # Also accept .jpeg / .png if present
            self.img_paths = sorted(
                p
                for ext in ("*.jpg", "*.jpeg", "*.png")
                for p in self.img_dir.glob(ext)
            )
        if not self.img_paths:
            raise FileNotFoundError(
                f"No images found under {self.img_dir}. "
                "Expected *.jpg / *.jpeg / *.png files."
            )

        self.transforms = (
            build_train_transforms(img_h, img_w)
            if augment
            else build_val_transforms(img_h, img_w)
        )

    # ------------------------------------------------------------------
    # Required Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of images in this split."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Load, augment, and return one (image, labels) pair.

        Image loading strategy — cv2.imread + IMREAD_GRAYSCALE
        -------------------------------------------------------
        cv2.IMREAD_GRAYSCALE is the fastest single-step loader for JPEG
        files: the JPEG codec's internal DCT path can skip chrominance
        decoding entirely, which is faster than PIL.convert('L') (which
        decodes to RGB first, then downsamples) and avoids the extra
        array copy that torchvision.io's GRAY mode performs when the
        source is a colour JPEG.  Error handling is explicit: cv2.imread
        returns None on failure rather than raising, so we check and
        raise with a useful message.

        The key gotcha with cv2.IMREAD_GRAYSCALE on colour JPEGs:
        cv2 applies the ITU-R BT.601 luminance formula
          Y = 0.299·R + 0.587·G + 0.114·B
        which differs slightly from PIL's formula (BT.601 with rounding
        quirks) and from torchvision's (which uses BT.709 coefficients
        internally for some modes).  For NIR imagery stored as colour
        JPEG (pseudo-colour), the three channels are often near-identical
        or carry a single channel duplicated, so the formula difference
        is negligible in practice.
        """
        img_path = self.img_paths[idx]

        # ---- 1. Load image as uint8 grayscale (H, W) ----
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise RuntimeError(
                f"cv2 could not open image: {img_path}. "
                "Check that the file exists and is a valid JPEG/PNG."
            )

        # Albumentations expects H×W×C (channel-last); add the channel dim.
        img_hwc = img_gray[:, :, np.newaxis]  # shape (H, W, 1)

        # ---- 2. Load YOLO labels ----
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        bboxes, class_labels = self._load_labels(lbl_path)

        # ---- 3. Apply augmentation pipeline ----
        # Albumentations clips boxes to [0,1] and drops boxes with
        # visibility < min_visibility automatically.
        transformed = self.transforms(
            image=img_hwc,
            bboxes=bboxes,
            class_labels=class_labels,
        )

        aug_img_hwc: np.ndarray = transformed["image"]   # uint8 (H, W, 1)
        aug_bboxes: list = transformed["bboxes"]          # list of (cx, cy, w, h)
        aug_classes: list = transformed["class_labels"]   # list of int

        # ---- 4. Convert image to float tensor (1, H, W) in [0, 1] ----
        # Divide by 255 here rather than using ToTensor() so we stay on the
        # numpy→tensor path without an intermediate PIL conversion.
        img_tensor = (
            torch.from_numpy(aug_img_hwc)   # (H, W, 1) uint8
            .permute(2, 0, 1)               # (1, H, W)
            .float()
            .div(255.0)                     # [0, 1]
        )

        # ---- 5. Build labels tensor (N, 5) ----
        if len(aug_bboxes) > 0:
            boxes_arr = np.array(aug_bboxes, dtype=np.float32)    # (N, 4)
            cls_arr = np.array(aug_classes, dtype=np.float32)[:, None]  # (N,1)
            labels = torch.from_numpy(
                np.concatenate([cls_arr, boxes_arr], axis=1)
            )  # (N, 5): [class, cx, cy, w, h]
        else:
            # Zero-annotation image: return empty (0, 5) tensor.
            # The collate function handles this without padding.
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return img_tensor, labels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_labels(
        self, lbl_path: Path
    ) -> Tuple[List[Tuple[float, float, float, float]], List[int]]:
        """
        Parse a YOLO-format label file.

        Returns
        -------
        bboxes       : list of (cx, cy, w, h) tuples, all in [0, 1]
        class_labels : list of int class IDs

        Missing label files (images with no annotations) are handled
        gracefully — they return empty lists, which causes __getitem__
        to return a (0, 5) label tensor.  This is the correct behaviour
        for negative training samples.
        """
        if not lbl_path.exists():
            return [], []

        bboxes = []
        class_labels = []
        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    # Malformed line — skip rather than crash
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                # Sanity-clamp: label files sometimes contain tiny float
                # errors that push values marginally outside [0, 1].
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w  = max(1e-6, min(1.0, w))
                h  = max(1e-6, min(1.0, h))
                bboxes.append((cx, cy, w, h))
                class_labels.append(cls)

        return bboxes, class_labels


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, List[Tensor]]:
    """
    Collate (image, labels) pairs into a batch.

    Why a custom collate_fn?
    ------------------------
    PyTorch's default_collate calls torch.stack on every element of the
    batch.  Because different images contain different numbers of people,
    the label tensors have shape (N_i, 5) with N_i varying per image.
    torch.stack requires equal shapes → RuntimeError.

    Two standard approaches in the literature:
      (A) Ultralytics YOLO — prepend a batch-index column and concatenate
          all labels into one flat tensor of shape (Σ N_i, 6), where
          column 0 is the image index within the batch.  This is efficient
          for the YOLO loss function, which iterates over the flat tensor.
      (B) torchvision / detectron2 — return images as a stacked tensor and
          labels as a Python list of variable-length tensors.  The loss
          function zips over the list.

    We choose approach (B) — list of tensors — because:
      - It is simpler (one line of change at the loss site vs. the index-
        prepend bookkeeping of approach A).
      - It matches what torchvision Faster-RCNN / RetinaNet expect, so it
        is easy to swap the head later.
      - For zero-annotation images, the (0, 5) tensor in the list is a
        natural sentinel that the loss can skip with `if labels.numel() == 0`.

    Parameters
    ----------
    batch : list of (image_tensor, label_tensor) tuples from __getitem__

    Returns
    -------
    images : FloatTensor of shape (B, 1, H, W)
    labels : list of B FloatTensors, each of shape (N_i, 5)
             (class, cx, cy, w, h), N_i may be 0.
    """
    images, labels = zip(*batch)          # two tuples of length B
    images = torch.stack(images, dim=0)   # (B, 1, H, W) — all same size
    return images, list(labels)           # labels stays as a list


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_dataloader(
    root: str | Path,
    split: str,
    img_h: int = 384,
    img_w: int = 640,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Return a DataLoader with the custom collate function."""
    augment = split == "train"
    if shuffle is None:
        shuffle = augment
    ds = NIRDetDataset(root, split=split, img_h=img_h, img_w=img_w, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=augment,   # drop last incomplete batch during training only
    )
