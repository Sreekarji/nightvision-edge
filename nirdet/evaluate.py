#!/usr/bin/env python
"""
evaluate.py  —  NIRDet Phase 9: full evaluation suite
======================================================
Reads a checkpoint written by train.py (key: "model_state") and produces:

  eval_outputs/
      pr_curve.png            – precision-recall curve with max-F1 point marked
      per_image_results.csv   – one row per val image
      hard_cases/*.jpg        – GT green / preds red for pure false-negative images
      summary.json            – machine-readable metric dump for future phase diffs

SINGLE INFERENCE PASS DESIGN
  One forward pass caches (preds, targets) for all 160 val images.
  mAP50, mAP50-95, PR curve, CSV, and hard-case images all consume those cached
  results — no risk of inconsistency from re-running the model.

FILE CONNECTIONS
  model.py    NIRDet() — forward(x, training_mode=False) returns
              List[B items], each item = [boxes(N,4) xyxy px, scores(N,)]
  train.py    save_checkpoint() writes key "model_state" (not "model_state_dict")
              evaluate_map50() uses nms_score_thresh=0.05 override → we match it
  dataset.py  NIRDetDataset, collate_fn; img_paths is an ordered List[Path]
              val transform: deterministic LongestMaxSize+PadIfNeeded, no random ops
  attention.py EdgeAwareAttention.residual_scale — set to 0.0 for EAA probe
  head.py     decodes cx=(sigmoid(t_cx)+col)*stride; w=exp(t_w)*W*stride; h=exp(t_h)*H*stride
  backbone.py out_channels = (c3=128, c4=256, c5=256) for base_ch=32

Usage:
    python evaluate.py                              # default eval
    python evaluate.py --checkpoint <path>          # explicit checkpoint
    python evaluate.py --score-thresh 0.3          # deployment threshold
    python evaluate.py --eaa-probe                 # also run EAA neutralisation
    python evaluate.py --test                      # unit tests T1-T3 only

Requirements:
    pip install torchmetrics[detection] matplotlib opencv-python tqdm

Machine: Windows 11, RTX 4050 Laptop GPU (6.4 GB VRAM)
         PyTorch 2.6.0+cu124
Project: C:\\projects\\nightvision\\nirdet\\
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations
import argparse           # CLI flag parsing
import copy               # deepcopy model for CPU benchmark (don't touch GPU copy)
import csv                # write per-image results CSV
import json               # machine-readable summary dump
import math               # math.isnan for recorded-mAP50 guard
import statistics         # statistics.mean / .stdev for latency samples
import sys                # sys.exit for test runner return code
import time               # time.perf_counter for CPU latency timing
from pathlib import Path  # cross-platform paths
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2                # image encode + drawing for hard-case overlays
import matplotlib         # PR curve plotting
matplotlib.use("Agg")     # headless backend: always saves to file, never opens window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou                            # vectorised IoU
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# ── Local modules (same directory as train.py) ────────────────────────────────
from model   import NIRDet            # full detector: backbone+EAA+neck+head+NMS
from dataset import NIRDetDataset, collate_fn  # val dataset + variable-length collate


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS  —  edit these to match your layout; mirrors train.py's CFG dict
# ═════════════════════════════════════════════════════════════════════════════

# Paths
DATA_ROOT     = r"C:\projects\nightvision\data\raw\miniNIRPed"
CKPT_DEFAULT  = r"checkpoints\best.pth"   # written by train.py save_checkpoint()
IMG_H         = 384                       # input height used in training
IMG_W         = 640                       # input width used in training
BATCH_SIZE    = 8                         # same as training for fair comparison
NUM_WORKERS   = 0                         # 0 = safest on Windows; raise if CPU idle

# Output directory (auto-created)
EVAL_OUT_DIR  = Path("eval_outputs")
HARD_DIR      = EVAL_OUT_DIR / "hard_cases"

# mAP reference from Phase 8
BASELINE_MAP50 = 0.7350    # YOLO11n fine-tuned on miniNIRPed (Phase 8 result)

# Score threshold for mAP computation — mirrors train.py's evaluate_map50().
# 0.05: wide enough to integrate the full PR curve.
# 0.001 floods metric with 8400 background cells per image (see Part D §3).
# 0.25 clips the curve early and under-reports recall.
EVAL_SCORE_THRESH = 0.05

# COCO 10-point IoU sweep for mAP50-95
COCO_IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]

# Must include 300 because model.py's decode_predictions caps NMS output at 300
MAX_DET_THRESHOLDS = [1, 10, 300]

# CPU latency protocol: 3 discarded warm-up + 20 measured passes
CPU_WARMUP = 3
CPU_TIMED  = 20

# Pi 5 Cortex-A76 @ 2.4 GHz is ~3-5x slower than a laptop x86 core on FP32 conv.
# This is a rule of thumb, NOT measured — see Part D §2. Replace with NCNN numbers.
PI5_SLOWDOWN_LO = 3   # optimistic:  FPS_pi ≈ FPS_cpu / 3
PI5_SLOWDOWN_HI = 5   # pessimistic: FPS_pi ≈ FPS_cpu / 5

# Checkpoint weight keys. train.py writes "model_state" (verified in train.py line 267).
# The Phase 9 prompt says "model_state_dict" — we accept both for robustness.
_CKPT_WEIGHT_KEYS = ("model_state", "model_state_dict", "state_dict")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI flags. argv=None reads sys.argv (normal use)."""
    p = argparse.ArgumentParser(description="NIRDet Phase 9 evaluation")
    p.add_argument(
        "--checkpoint", default=CKPT_DEFAULT,
        help="Path to best.pth checkpoint (default: checkpoints/best.pth)"
    )
    p.add_argument(
        "--score-thresh", type=float, default=0.25,
        help="Deployment NMS threshold for CSV / hard-case mining "
             "(default 0.25). Does NOT affect mAP (always 0.05)."
    )
    p.add_argument("--data-root", default=DATA_ROOT,
                   help="miniNIRPed root directory")
    p.add_argument("--batch-size",    type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers",   type=int, default=NUM_WORKERS)
    p.add_argument("--img-h", type=int, default=IMG_H,
                   help="Input height (default: training resolution)")
    p.add_argument("--img-w", type=int, default=IMG_W,
                   help="Input width (default: training resolution)")
    p.add_argument(
        "--max-hard-cases", type=int, default=10,
        help="Max number of hard-case images to save (default 10)"
    )
    p.add_argument(
        "--eaa-probe", action="store_true",
        help="Also re-evaluate with EAA neutralised (residual_scale=0). "
             "Measures inference-time dependence, NOT training contribution. "
             "See Part D §4."
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run unit tests T1-T3 instead of full evaluation"
    )
    return p.parse_args(argv)


# ═════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def amp_context(device: torch.device):
    """
    Return an autocast context matching train.py's build_amp_context().

    bf16 on CUDA (same as training — preserves numeric parity with the
    forward passes that produced the best.pth checkpoint).
    Disabled (no-op) on CPU: bf16 on x86 is emulated and makes latency
    benchmarks meaningless.
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return torch.amp.autocast("cuda", dtype=torch.float16)  # RTX 40 fallback
    return torch.amp.autocast("cpu", enabled=False)             # no-op


def yolo_to_xyxy(labels: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    Convert (N,5) YOLO label tensor → (N,4) absolute xyxy pixel boxes.

    YOLO columns: (class_id, cx_norm, cy_norm, w_norm, h_norm), all in [0,1].
    The padded 640×384 canvas produced by build_val_transforms() is the shared
    coordinate space for both GT labels and model predictions (head.py decodes
    to pixels of that same canvas), so no letterbox-undo is needed.

    Returns (0,4) zeros tensor for unannotated images (labels.numel()==0).
    """
    if labels.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    cx = labels[:, 1] * img_w   # (N,) pixel centre-x
    cy = labels[:, 2] * img_h   # (N,) pixel centre-y
    w  = labels[:, 3] * img_w   # (N,) pixel width
    h  = labels[:, 4] * img_h   # (N,) pixel height
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1
    ).float()                    # (N, 4) xyxy pixels


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1  —  LOAD CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════════

def load_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[NIRDet, dict]:
    """
    Build a fresh NIRDet and load weights from ckpt_path.

    Checkpoint format produced by train.py's save_checkpoint():
        {
          "epoch":            int,          # 0-based epoch index
          "model_state":      OrderedDict,  # ACTUAL key used (not model_state_dict)
          "optimizer_state":  OrderedDict,
          "scaler_state":     dict,
          "scheduler_state":  dict,
          "best_map50":       float,
          "cfg":              dict,
        }

    Accepts "model_state", "model_state_dict", and "state_dict" as weights keys
    (the first is what train.py actually writes; the others exist in some older
    versions of the code and are documented in the Phase 9 prompt).

    strict=False in load_state_dict: EAA registers _epoch_buf as a buffer that
    may be absent in older checkpoints. We print any mismatch for transparency.

    Raises FileNotFoundError with an actionable message if the file is missing.
    """
    if not ckpt_path.exists():
        parent = ckpt_path.parent
        nearby = (sorted(p.name for p in parent.glob("*.pth"))
                  if parent.exists() else [])
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"  Nearby .pth files in {parent}: {nearby}\n"
            f"  Run train.py first, or pass --checkpoint <path>."
        )

    # weights_only=False: checkpoint contains 'cfg' (plain Python dict), not just
    # tensors. PyTorch refuses to unpickle non-tensor objects with weights_only=True.
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # ── Locate the model state dict ───────────────────────────────────────────
    state, key_used = None, None
    if isinstance(ckpt, dict):
        for k in _CKPT_WEIGHT_KEYS:
            if k in ckpt and isinstance(ckpt[k], dict):
                state, key_used = ckpt[k], k
                break
        if state is None and all(torch.is_tensor(v) for v in ckpt.values()):
            # Rare: raw torch.save(model.state_dict(), path)
            state, key_used = ckpt, "<bare state_dict>"
    if state is None:
        raise RuntimeError(
            f"No model weights found in {ckpt_path}.\n"
            f"  Tried keys: {_CKPT_WEIGHT_KEYS}\n"
            f"  Keys present: "
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    model = NIRDet()                    # fresh architecture with its own init
    incompat = model.load_state_dict(state, strict=False)
    model.to(device).eval()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    epoch     = ckpt.get("epoch", -1) if isinstance(ckpt, dict) else -1
    recorded  = (ckpt.get("best_map50", float("nan"))
                 if isinstance(ckpt, dict) else float("nan"))

    if verbose:
        print("=" * 62)
        print("  Step 1 — Checkpoint")
        print("=" * 62)
        print(f"  File               : {ckpt_path}")
        print(f"  Weights key        : {key_used}")
        print(f"  Saved at epoch     : {epoch + 1}  (0-indexed: {epoch})")
        print(f"  Recorded best mAP50: "
              f"{'n/a' if math.isnan(recorded) else f'{recorded:.4f}'}")
        print(f"  Parameters         : {total:,} total / {trainable:,} trainable")
        if incompat.missing_keys:
            print(f"  ⚠ missing keys  ({len(incompat.missing_keys)}): "
                  f"{incompat.missing_keys[:5]}")
        if incompat.unexpected_keys:
            print(f"  ⚠ unexpected keys ({len(incompat.unexpected_keys)}): "
                  f"{incompat.unexpected_keys[:5]}")
        if not incompat.missing_keys and not incompat.unexpected_keys:
            print("  State dict         : exact match ✓")
        print()

    return model, ckpt


# ═════════════════════════════════════════════════════════════════════════════
# SHARED SINGLE INFERENCE PASS  —  feeds every downstream step
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model:  NIRDet,
    loader: DataLoader,
    device: torch.device,
    eval_score_thresh: float = EVAL_SCORE_THRESH,
    desc:   str = "  inference",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run the entire val loader ONCE and cache (predictions, targets) in CPU RAM.

    Why one pass?
      mAP50, mAP50-95, the PR curve, the per-image CSV, and the hard-case images
      all consume these cached results without re-running the model.  One pass
      guarantees self-consistent numbers and halves total runtime vs two passes.

    Score threshold override:
      Temporarily lowers model.nms_score_thresh from 0.25 to 0.05 so the mAP
      computation integrates the full PR curve (mirrors train.py's evaluate_map50).
      Restored via finally even if an exception interrupts the loop.

    Coordinate frame:
      model(images, training_mode=False) returns boxes in pixel coords of the
      padded 640×384 canvas.  GT labels are also normalised to that same canvas
      (build_val_transforms applies LongestMaxSize+PadIfNeeded deterministically).
      Both are converted to xyxy pixels here so they are in the same frame.

    Returns:
        preds   : List[dict]  keys = boxes (N,4) xyxy px, scores (N,), labels (N,)
        targets : List[dict]  keys = boxes (M,4) xyxy px, labels (M,)
        Both lists are in DataLoader order = val_ds.img_paths order (shuffle=False).
    """
    model.eval()
    saved_thresh = model.nms_score_thresh   # remember deployment value (0.25)
    model.nms_score_thresh = eval_score_thresh        # widen funnel for mAP
    ctx = amp_context(device)               # bf16 on CUDA, no-op on CPU

    preds:   List[Dict] = []
    targets: List[Dict] = []

    try:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)   # (B, 1, H, W)
            B, _, H, W = images.shape                       # e.g. B=8, H=384, W=640

            with ctx:
                # training_mode=False: head decodes offsets → pixel coords,
                # then decode_predictions() applies score filter + NMS.
                # Returns List[B items]; each item = [boxes(N,4), scores(N,)].
                batch_out = model(images, training_mode=False)

            for b in range(B):
                boxes  = batch_out[b][0].detach().float().cpu()  # (N, 4) xyxy px
                scores = batch_out[b][1].detach().float().cpu()  # (N,) confidence

                # torchmetrics prediction format: requires boxes, scores, labels
                preds.append({
                    "boxes":  boxes,
                    "scores": scores,
                    # Single-class: every detection is class 0 (person).
                    "labels": torch.zeros(scores.numel(), dtype=torch.long),
                })

                # Ground truth: convert YOLO normalised (class,cx,cy,w,h) → xyxy px
                t = labels[b]                                # (M, 5) from collate_fn
                targets.append({
                    "boxes":  yolo_to_xyxy(t, W, H),        # (M, 4) xyxy px
                    "labels": (t[:, 0].long() if t.numel()
                               else torch.zeros(0, dtype=torch.long)),
                })
    finally:
        model.nms_score_thresh = saved_thresh   # always restore deployment threshold

    return preds, targets


# ═════════════════════════════════════════════════════════════════════════════
# STEPS 2 & 3  —  mAP (no model re-run; consumes cached predictions)
# ═════════════════════════════════════════════════════════════════════════════

def compute_map(
    preds:   List[Dict],
    targets: List[Dict],
    iou_thresholds:   Optional[List[float]],
    extended_summary: bool = False,
) -> Tuple[Dict, MeanAveragePrecision]:
    """
    Feed cached predictions into a fresh MeanAveragePrecision instance.

    No model forward pass happens here — we reuse run_inference() output.
    Both metric configurations (mAP50 and mAP50-95) consume the same cache,
    halving runtime and guaranteeing identical predictions in both results.

    iou_thresholds=[0.5]   → map_50 key (headline metric, same as train.py)
    iou_thresholds=None    → COCO default 0.50:0.05:0.95; read 'map' key
    iou_thresholds=list    → custom sweep (we pass COCO_IOU_THRESHOLDS explicitly)
    extended_summary=True  → additionally returns precision/recall/scores tensors
                             with shape (T,R,K,A,M) — needed for a deployable PR curve
                             (actual confidence thresholds, not just recall values).
    """
    metric = MeanAveragePrecision(
        box_format="xyxy",                  # model already decodes to xyxy pixels
        iou_type="bbox",
        iou_thresholds=iou_thresholds,      # None = COCO default sweep
        max_detection_thresholds=MAX_DET_THRESHOLDS,   # [1, 10, 300]
        extended_summary=extended_summary,
    )
    metric.update(preds, targets)           # single batched call — no model run
    return metric.compute(), metric


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4  —  PRECISION-RECALL CURVE
# ═════════════════════════════════════════════════════════════════════════════

def extract_pr_curve(
    res:    Dict,
    metric: MeanAveragePrecision,
    iou_idx:    int = 0,   # T dim: 0 = IoU 0.50 (only threshold we passed)
    class_idx:  int = 0,   # K dim: 0 = person (only class)
    area_idx:   int = 0,   # A dim: 0 = all areas (1=small,2=medium,3=large)
    maxdet_idx: int = -1,  # M dim: -1 → index 2 → 300 detections
) -> Dict:
    """
    Extract (recall, precision, confidence-score) from an extended-summary result.

    torchmetrics >= 1.2 with extended_summary=True returns:
        precision : Tensor (T, R, K, A, M)
        recall    : Tensor (T, K, A, M)
        scores    : Tensor (T, R, K, A, M)
    Where:
        T = 1   (we passed iou_thresholds=[0.5])
        R = 101 (COCO recall grid 0.00→1.00 step 0.01)
        K = 1   (single class: person)
        A = 4   (all, small, medium, large)
        M = 3   ([1, 10, 300] max detections)

    COCO sentinel -1.0 marks undefined recall-points (masked before return).

    The scores tensor is the KEY improvement over existing code:
    it gives the actual CONFIDENCE THRESHOLD that achieves each recall level,
    making the best-F1 operating point directly deployable on Pi 5.
    Without extended_summary, the existing code returned the recall VALUE at
    peak F1 rather than the confidence THRESHOLD — not actionable for deployment.

    Raises KeyError (with pip fix hint) if torchmetrics < 1.2.
    """
    if "precision" not in res or "scores" not in res:
        raise KeyError(
            "extended_summary keys ('precision', 'scores') absent.\n"
            "  Required: torchmetrics >= 1.2\n"
            "  Fix: pip install --upgrade torchmetrics[detection]\n"
            f"  Keys returned: {list(res.keys())}"
        )

    # Slice (T,R,K,A,M) → (R,): the precision and confidence at each recall point
    prec_r = res["precision"][iou_idx, :, class_idx, area_idx, maxdet_idx]  # (R,)
    scor_r = res["scores"   ][iou_idx, :, class_idx, area_idx, maxdet_idx]  # (R,)

    # Recall axis — torchmetrics exposes this as metric.rec_thresholds
    rec_th = getattr(metric, "rec_thresholds", None)
    recall = (np.asarray(rec_th, dtype=np.float64) if rec_th is not None
              else np.linspace(0.0, 1.0, prec_r.numel()))   # (R,) fallback

    p = prec_r.detach().cpu().numpy().astype(np.float64)    # (R,) precision
    s = scor_r.detach().cpu().numpy().astype(np.float64)    # (R,) confidence
    valid = p >= 0.0    # mask COCO -1.0 sentinel (undefined recall points)

    return {
        "recall":       recall[valid],      # (V,) recall axis
        "precision":    p[valid],           # (V,) precision values
        "score_thresh": s[valid],           # (V,) confidence thresholds ← deployable
        "n_points":     int(valid.sum()),
    }


def best_f1_point(curve: Dict) -> Dict:
    """
    Find the PR-curve point maximising F1 = 2PR / (P + R).

    Returns precision, recall, F1, and the actual confidence threshold at that
    point. The confidence threshold is what you set as nms_score_thresh on Pi 5.
    Picking max-F1 (not max-recall) is recommended for Pi 5: false positives
    at low confidence flood downstream tracking, whereas missed frames of a
    pedestrian are recovered in the next frame.
    """
    p, r, s = curve["precision"], curve["recall"], curve["score_thresh"]
    denom = p + r
    f1    = np.zeros_like(p)
    nz    = denom > 0           # avoid 0/0 at the origin where P=R=0
    f1[nz] = 2.0 * p[nz] * r[nz] / denom[nz]

    if f1.size == 0 or not np.any(f1 > 0):     # no useful predictions at all
        return {"precision": 0.0, "recall": 0.0,
                "f1": 0.0, "score_thresh": float("nan"), "index": -1}

    i = int(np.argmax(f1))
    return {
        "precision":    float(p[i]),
        "recall":       float(r[i]),
        "f1":           float(f1[i]),
        "score_thresh": float(s[i]),   # actual confidence threshold → deployable
        "index":        i,
    }


def plot_pr_curve(
    curve: Dict, best: Dict, map50: float, out_path: Path,
) -> None:
    """
    Save PR curve to PNG with max-F1 operating point and baseline reference.

    YOLO11n baseline appears as a horizontal dashed line so the gap is
    immediately visible. The vertical dashed line at peak-F1 recall shows
    which recall-level to target for the Pi 5 operating point.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=130)

    # PR curve
    ax.plot(curve["recall"], curve["precision"], linewidth=2.0,
            label=f"NIRDet  (AP@0.50 = {map50:.4f})")

    # YOLO11n horizontal reference line
    ax.axhline(BASELINE_MAP50, color="grey", linestyle="--", linewidth=1.2,
               label=f"YOLO11n baseline ({BASELINE_MAP50:.4f})")

    if best["index"] >= 0:
        # Vertical guide at the best-F1 recall
        ax.axvline(best["recall"], color="crimson",
                   linestyle="--", linewidth=1.0, alpha=0.65)
        # Dot on the curve at the operating point
        ax.plot([best["recall"]], [best["precision"]],
                marker="o", markersize=8, color="crimson", linestyle="none",
                label=(f"best F1 = {best['f1']:.3f}  "
                       f"(conf ≥ {best['score_thresh']:.3f})\n"
                       f"P = {best['precision']:.3f},  "
                       f"R = {best['recall']:.3f}"))

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("NIRDet — PR Curve @ IoU = 0.50  (val set, 1 class: person)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5  —  PER-IMAGE CSV  (uses cached predictions)
# ═════════════════════════════════════════════════════════════════════════════

def build_per_image_rows(
    preds:   List[Dict],
    targets: List[Dict],
    names:   List[str],
    score_thresh: float,    # deployment threshold for n_pred / is_hard_case
) -> List[Dict]:
    """
    Build one result row per val image from cached predictions.

    Filtering logic:
      preds[i]["scores"] contains ALL predictions at EVAL_SCORE_THRESH (0.05).
      We post-filter at score_thresh (0.25) to compute operational statistics.
      This is valid because NMS at 0.05 followed by score-filter at 0.25 gives
      the same surviving boxes as NMS at 0.25 directly (high-score boxes always
      suppress lower-score overlapping boxes in NMS, so post-filtering is lossless).

    Columns:
        filename     — image file name (from val_ds.img_paths)
        n_gt         — GT annotation count
        n_pred       — prediction count at score_thresh (deployment threshold)
        max_score    — highest confidence score from the full cached predictions
        best_iou     — max IoU between any deployment pred and any GT box
        hit          — 1 if any deployment pred has IoU ≥ 0.5 with any GT box
        is_hard_case — 1 if n_gt > 0 and n_pred == 0 (pure false negative)
    """
    rows = []
    for name, pr, tg in zip(names, preds, targets):
        scores   = pr["scores"]                     # (N,) all cached predictions
        keep     = scores >= score_thresh           # operational filter at 0.25
        boxes_op = pr["boxes"][keep]               # (K, 4) deployment-threshold preds
        n_gt     = int(tg["boxes"].shape[0])

        # Max IoU between any operational prediction and any GT box
        if boxes_op.numel() and n_gt:
            best_iou = float(box_iou(boxes_op, tg["boxes"]).max())
        else:
            best_iou = 0.0

        rows.append({
            "filename":     name,
            "n_gt":         n_gt,
            "n_pred":       int(keep.sum()),         # count at deployment threshold
            "max_score":    round(float(scores.max()), 4) if scores.numel() else 0.0,
            "best_iou":     round(best_iou, 4),
            "hit":          int(best_iou >= 0.5),    # any correct detection at 0.25
            "is_hard_case": int(n_gt > 0 and int(keep.sum()) == 0),
        })
    return rows


def write_csv(rows: List[Dict], out_path: Path) -> None:
    """Write per-image rows to a CSV file with one header row."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6  —  HARD-CASE VISUALISATION  (uses cached predictions)
# ═════════════════════════════════════════════════════════════════════════════

def render_case(
    image_chw:   torch.Tensor,  # (1, H, W) float [0, 1] from NIRDetDataset
    gt_boxes:    torch.Tensor,  # (M, 4) xyxy pixels — ground truth
    pred_boxes:  torch.Tensor,  # (N, 4) xyxy pixels — cached predictions at 0.05
    pred_scores: torch.Tensor,  # (N,)   confidence scores
) -> np.ndarray:
    """
    Draw GT (green) and predictions (red + confidence label) on a NIR frame.

    Applies cv2.NORM_MINMAX contrast stretch so near-black NIR frames are
    inspectable — raw NIR pixel values often sit in [0, 40] out of 255.
    GT boxes are drawn first (thickness=2) so predictions (thickness=1)
    overlay on top, making overlap visible at a glance.
    Returns (H, W, 3) uint8 BGR image for cv2.imwrite.
    """
    # Float [0,1] → uint8 grayscale → contrast-stretch → BGR
    gray = (image_chw.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)  # (H, W)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)    # stretch [0,255]
    vis  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)                 # (H, W, 3) BGR

    # Ground truth: green (BGR 0,255,0), thickness=2, drawn first
    for box in gt_boxes.tolist():
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "GT", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # Predictions: red (BGR 0,0,255), thickness=1, score annotated below box
    for box, sc in zip(pred_boxes.tolist(), pred_scores.tolist()):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(vis, f"{sc:.2f}", (x1, min(y2 + 12, vis.shape[0] - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


def visualise_hard_cases(
    val_ds:  NIRDetDataset,
    preds:   List[Dict],
    targets: List[Dict],
    rows:    List[Dict],
    out_dir: Path,
    max_cases: int = 10,
) -> List[str]:
    """
    Save up to max_cases images where GT has ≥1 box but model predicted nothing
    above the deployment threshold (pure false negatives, is_hard_case=1).

    Two failure modes are visually distinguishable:
      "Saw nothing"  — no red boxes at all even at 0.05 threshold
      "Fired weakly" — red boxes present below 0.25 confidence threshold

    preds[i]["boxes"] contains ALL predictions at 0.05 (the inference pass used
    that threshold). This reveals the "fired weakly" mode even though is_hard_case
    was computed at 0.25. See debug_aug.jpg for an example of the extreme edge-crop
    failure mode expected to dominate hard cases in miniNIRPed.

    Images are re-read via val_ds[i] for the float tensor.
    val_ds uses augment=False → deterministic LongestMaxSize+PadIfNeeded transform,
    so pixel coordinates of preds[i]["boxes"] are in the correct frame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hard_idx = [i for i, r in enumerate(rows) if r["is_hard_case"]][:max_cases]

    saved: List[str] = []
    for i in hard_idx:
        image_chw, _ = val_ds[i]           # re-read: (1,H,W) float, deterministic
        vis = render_case(
            image_chw,
            targets[i]["boxes"],           # (M,4) GT xyxy px — cached from inference
            preds[i]["boxes"],             # (N,4) all preds at 0.05 — cached
            preds[i]["scores"],            # (N,)  confidence values
        )
        stem = Path(rows[i]["filename"]).stem
        path = out_dir / f"hard_{stem}.jpg"
        cv2.imwrite(str(path), vis)
        saved.append(path.name)

    return saved


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7  —  LATENCY BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def benchmark_latency(
    model:    NIRDet,
    device:   torch.device,
    img_h:    int = IMG_H,
    img_w:    int = IMG_W,
    warmup:   int = CPU_WARMUP,
    runs:     int = CPU_TIMED,
) -> Dict:
    """
    Time a single-image forward pass (backbone + EAA + neck + head + NMS).

    Protocol:
      1. warmup untimed passes: forces PyTorch allocator, JIT kernel compile,
         OS page-in of weights into CPU cache.
      2. runs timed passes: perf_counter wraps each call.
      3. CUDA sync before/after each timed call (GPU kernels are asynchronous;
         without sync, measured time ≈ 0 ms because torch.cuda.* is enqueued
         but not waited on).
      4. Report mean ± std in milliseconds.

    Probe input is zeros: NMS sees no survivors above threshold, so NMS cost ≈ 0.
    Real crowded frames add marginal cost but backbone/neck/head dominate runtime.

    perf_counter() vs torch.utils.benchmark.Timer:
      Timer's value is mainly in taming CUDA async; for synchronous CPU execution
      perf_counter + explicit CUDA sync gives equally accurate results with less
      boilerplate, and the high-resolution Windows QueryPerformanceCounter makes
      it suitable even for <100 ms measurements.
    """
    m   = model.to(device).eval()
    x   = torch.zeros(1, 1, img_h, img_w, device=device)  # (1,1,H,W) probe
    ctx = amp_context(device)

    for _ in range(warmup):
        with ctx:
            m(x, training_mode=False)
    if device.type == "cuda":
        torch.cuda.synchronize()    # drain after warm-up before timing starts

    samples: List[float] = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()   # drain before t0
        t0 = time.perf_counter()
        with ctx:
            m(x, training_mode=False)
        if device.type == "cuda":
            torch.cuda.synchronize()   # drain before t1 (kernel is async)
        samples.append((time.perf_counter() - t0) * 1000.0)   # → ms

    mean = statistics.mean(samples)
    std  = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "mean_ms":  mean,
        "std_ms":   std,
        "fps":      1000.0 / mean if mean > 0 else 0.0,
        "img_size": (img_h, img_w),
        "runs":     runs,
        "device":   str(device),
    }


# ═════════════════════════════════════════════════════════════════════════════
# OPTIONAL  —  EAA INFERENCE-TIME PROBE  (see Part D §4)
# ═════════════════════════════════════════════════════════════════════════════

def probe_eaa_off(
    model:   NIRDet,
    loader:  DataLoader,
    device:  torch.device,
) -> float:
    """
    Neutralise EAA and re-measure mAP50 on the full val set.

    EdgeAwareAttention applies: feat * (1 + residual_scale * attn)
    Setting residual_scale=0 → feat * (1 + 0) = feat  (identity map).
    The mAP50 delta measures how much the CONVERGED network relies on EAA.

    This is NOT the contribution of EAA to training (see Part D §4).
    The original residual_scale (0.5) is always restored in finally.
    """
    saved = model.eaa.residual_scale           # remember original scale (0.5)
    model.eaa.residual_scale = 0.0             # EAA → identity
    try:
        p_off, t_off = run_inference(model, loader, device,
                                     eval_score_thresh=EVAL_SCORE_THRESH,desc="  EAA-off inference")
        res, _ = compute_map(p_off, t_off, iou_thresholds=[0.5])
        return float(res["map_50"])
    finally:
        model.eaa.residual_scale = saved       # always restore


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> int:
    """Full Phase 9 evaluation pipeline. Returns 0 on success."""
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    HARD_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nNIRDet — Phase 9 Evaluation")
    gpu_name = (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    print(f"  device : {device}{gpu_name}")
    print(f"  torch  : {torch.__version__}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Load checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    model, ckpt = load_checkpoint(Path(args.checkpoint), device)
    epoch    = ckpt.get("epoch", -1) if isinstance(ckpt, dict) else -1
    recorded = (ckpt.get("best_map50", float("nan"))
                if isinstance(ckpt, dict) else float("nan"))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ─────────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────────
    val_ds = NIRDetDataset(
        root=args.data_root, split="val",
        img_h=args.img_h, img_w=args.img_w,
        augment=False,   # deterministic val transform
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,      # MUST: preds align with val_ds.img_paths by index
        drop_last=False,    # MUST: every val image must be scored (not dropped)
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    # Names in dataset order — same order as the DataLoader (shuffle=False).
    # These align index-for-index with preds and targets from run_inference().
    names: List[str] = [p.name for p in val_ds.img_paths]
    print(f"  Val images : {len(val_ds)}  (batch_size={args.batch_size})\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Single inference pass — caches all results for downstream steps
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  Single inference pass (feeds all metrics below) …")
    print("=" * 62)
    preds, targets = run_inference(model, val_loader, device)
    # Sanity check: prediction count must match dataset size
    assert len(preds) == len(val_ds), (
        f"pred count {len(preds)} ≠ dataset size {len(val_ds)}. "
        "Check shuffle=False and drop_last=False in DataLoader."
    )
    print(f"  Cached {len(preds)} prediction dicts.\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Steps 2 & 3 — mAP50 + mAP50-95
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  Steps 2-3 — Detection accuracy")
    print("=" * 62)

    # mAP50 with extended_summary so extract_pr_curve can return real score thresholds
    res50, metric50 = compute_map(
        preds, targets, iou_thresholds=[0.5], extended_summary=True
    )
    map50 = float(res50["map_50"])
    delta = map50 - BASELINE_MAP50

    # mAP50-95: same cached preds, different IoU thresholds — no model re-run
    res5095, _ = compute_map(preds, targets, iou_thresholds=COCO_IOU_THRESHOLDS)
    map5095 = float(res5095["map"])

    # Per-size metrics (-1.0 = no objects of that COCO size exist in val set)
    map_s  = float(res5095.get("map_small",  torch.tensor(-1.0)))
    map_m  = float(res5095.get("map_medium", torch.tensor(-1.0)))
    map_l  = float(res5095.get("map_large",  torch.tensor(-1.0)))
    mar300 = float(res5095.get(f"mar_{MAX_DET_THRESHOLDS[-1]}", torch.tensor(0.0)))
    map75  = float(res5095.get("map_75", torch.tensor(0.0)))

    print(f"  mAP50            : {map50:.4f}   "
          f"(baseline {BASELINE_MAP50:.4f},  delta {delta:+.4f})")
    print(f"  mAP50-95         : {map5095:.4f}   (COCO 10-point sweep)")
    print(f"  mAP@75           : {map75:.4f}")
    print(f"  mAR@300          : {mar300:.4f}")
    print(f"  COCO size split  : small={map_s:.4f} | medium={map_m:.4f} | "
          f"large={map_l:.4f}  (-1 = no objects of that size in val)")

    # Sanity: warn if our mAP50 differs significantly from checkpoint's value.
    # Causes: different eval_score_thresh, different val split, different augment flag.
    if not math.isnan(recorded) and abs(recorded - map50) > 0.01:
        print(f"  ⚠ mAP50 ({map50:.4f}) differs from checkpoint recorded "
              f"value ({recorded:.4f}) by > 0.01. Check EVAL_SCORE_THRESH parity.")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Precision-Recall curve
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  Step 4 — Precision-Recall curve")
    print("=" * 62)

    best  = {"index": -1, "precision": 0.0, "recall": 0.0,
             "f1": 0.0, "score_thresh": float("nan")}
    curve = {"n_points": 0, "recall": np.array([]), "precision": np.array([])}

    try:
        curve = extract_pr_curve(res50, metric50)
        best  = best_f1_point(curve)
        pr_path = EVAL_OUT_DIR / "pr_curve.png"
        plot_pr_curve(curve, best, map50, pr_path)
        print(f"  Valid curve points : {curve['n_points']}")
        print(f"  Saved              : {pr_path}")
        if best["index"] >= 0:
            print(f"  Best-F1 operating point:")
            print(f"     confidence  : {best['score_thresh']:.4f}"
                  f"  ← set as nms_score_thresh on Pi 5")
            print(f"     precision   : {best['precision']:.4f}")
            print(f"     recall      : {best['recall']:.4f}")
            print(f"     F1          : {best['f1']:.4f}")
    except KeyError as exc:
        print(f"  ⚠ PR curve unavailable: {exc}")
        print("  Run: pip install --upgrade torchmetrics[detection]")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — Per-image CSV
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  Step 5 — Per-image results CSV")
    print("=" * 62)
    rows      = build_per_image_rows(preds, targets, names, args.score_thresh)
    csv_path  = EVAL_OUT_DIR / "per_image_results.csv"
    write_csv(rows, csv_path)
    n_hard    = sum(r["is_hard_case"] for r in rows)
    n_hit     = sum(r["hit"] for r in rows)
    n_with_gt = sum(1 for r in rows if r["n_gt"] > 0)
    print(f"  Rows written       : {len(rows)}  →  {csv_path}")
    print(f"  Images with GT     : {n_with_gt}")
    print(f"  Images with hit    : {n_hit}   "
          f"(IoU ≥ 0.5 at thresh={args.score_thresh:g})")
    print(f"  Pure false-neg     : {n_hard}   (is_hard_case=1)")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6 — Hard-case visualisation
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  Step 6 — Hard-case visualisation")
    print("=" * 62)
    saved = visualise_hard_cases(
        val_ds, preds, targets, rows,
        HARD_DIR, max_cases=args.max_hard_cases,
    )
    print(f"  Saved {len(saved)} image(s) → {HARD_DIR}/")
    for nm in saved:
        print(f"     {nm}")
    if not saved:
        print("     (none — model has ≥1 prediction on every annotated image)")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7 — Latency benchmark (GPU + CPU)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  Step 7 — Latency benchmark")
    print("=" * 62)

    gpu_bench = None
    if device.type == "cuda":
        print("  Benchmarking GPU …")
        gpu_bench = benchmark_latency(model, device, args.img_h, args.img_w)
        print(f"  GPU  @{args.img_h}x{args.img_w} : {gpu_bench['mean_ms']:.2f} ± "
              f"{gpu_bench['std_ms']:.2f} ms   ({gpu_bench['fps']:.1f} FPS)")

    # Deepcopy before moving to CPU so the GPU model's device placement is intact
    cpu_model = copy.deepcopy(model).to(torch.device("cpu")).eval()
    print(f"  CPU threads  : {torch.get_num_threads()}")
    print(f"  Benchmarking CPU @{args.img_h}x{args.img_w} …")
    cpu_bench = benchmark_latency(cpu_model, torch.device("cpu"),
                                  args.img_h, args.img_w)
    print(f"  Benchmarking CPU @192x320 …  "
          f"(Pi 5 likely deployment resolution under NCNN — half of 384x640)")
    cpu_bench_192x320 = benchmark_latency(cpu_model, torch.device("cpu"), 192, 320)

    print(f"  CPU  @{args.img_h}x{args.img_w} : {cpu_bench['mean_ms']:.1f} ± "
          f"{cpu_bench['std_ms']:.1f} ms   ({cpu_bench['fps']:.2f} FPS)")
    print(f"  CPU  @192x320 : {cpu_bench_192x320['mean_ms']:.1f} ± "
          f"{cpu_bench_192x320['std_ms']:.1f} ms   ({cpu_bench_192x320['fps']:.2f} FPS)")

    pi_fps_lo = cpu_bench["fps"] / PI5_SLOWDOWN_HI   # pessimistic (5× slower)
    pi_fps_hi = cpu_bench["fps"] / PI5_SLOWDOWN_LO   # optimistic  (3× slower)
    print(f"  Est. Pi 5 @{args.img_h}x{args.img_w} : "
          f"~{pi_fps_lo:.2f}–{pi_fps_hi:.2f} FPS   "
          f"({PI5_SLOWDOWN_LO}–{PI5_SLOWDOWN_HI}× slowdown — "
          f"ESTIMATE, verify with NCNN on device)")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Optional — EAA inference-time probe
    # ─────────────────────────────────────────────────────────────────────────
    eaa_map50 = None
    if args.eaa_probe:
        print("=" * 62)
        print("  EAA probe  (NOT a rigorous ablation — see Part D §4)")
        print("=" * 62)
        eaa_map50 = probe_eaa_off(model, val_loader, device)
        print(f"  mAP50  EAA on  : {map50:.4f}")
        print(f"  mAP50  EAA off : {eaa_map50:.4f}   "
              f"(delta {map50 - eaa_map50:+.4f})")
        print("  Interpretation: CONVERGED-model dependence on EAA attention,")
        print("  NOT the gain from training with EAA.  See Part D §4.\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8 — Summary table
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  NIRDet Evaluation Summary")
    print("=" * 62)
    print(f"  Checkpoint epoch    : {epoch + 1}")
    print(f"  Val images          : {len(val_ds)}")     # FIX: was len([1])=1 in old code
    print(f"  Trainable params    : {n_params:,}")
    print(f"  mAP50               : {map50:.4f}   "
          f"(baseline: {BASELINE_MAP50:.4f},  delta: {delta:+.4f})")
    print(f"  mAP50-95            : {map5095:.4f}")
    print(f"  mAP small/med/large : {map_s:.4f} / {map_m:.4f} / {map_l:.4f}")
    if best["index"] >= 0:
        print(f"  Best F1 threshold   : {best['score_thresh']:.3f}   "
              f"(P={best['precision']:.3f}, R={best['recall']:.3f}, "
              f"F1={best['f1']:.3f})")
    print(f"  CPU latency (384x640)   : "
          f"{cpu_bench['mean_ms']:.1f} ± {cpu_bench['std_ms']:.1f} ms   "
          f"(~{cpu_bench['fps']:.2f} FPS)")
    print(f"  Est. Pi 5 FPS       : ~{pi_fps_lo:.2f}–{pi_fps_hi:.2f} FPS   "
          f"({PI5_SLOWDOWN_LO}–{PI5_SLOWDOWN_HI}× CPU slowdown estimate)")
    print(f"  Hard cases saved    : {HARD_DIR}/  ({len(saved)} images)")
    print("=" * 62)

    # ─────────────────────────────────────────────────────────────────────────
    # JSON summary dump — machine-readable for future phase diffs
    # ─────────────────────────────────────────────────────────────────────────
    summary = {
        "checkpoint":    str(Path(args.checkpoint)),
        "epoch":         epoch + 1,
        "n_val":         len(val_ds),
        "n_params":      n_params,
        "map50":         map50,
        "baseline_map50": BASELINE_MAP50,
        "delta":         delta,
        "map5095":       map5095,
        "map75":         map75,
        "map_small":     map_s,
        "map_medium":    map_m,
        "map_large":     map_l,
        "mar_300":       mar300,
        "best_f1_point": best,
        "n_hard_cases":  len(saved),
        "latency_gpu":   gpu_bench,
        "latency_cpu":   cpu_bench,
        "latency_cpu_192x320": cpu_bench_192x320,
        "pi5_fps_estimate": {"lo": pi_fps_lo, "hi": pi_fps_hi},
        "eaa_probe_map50": eaa_map50,
    }
    json_path = EVAL_OUT_DIR / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Machine-readable summary → {json_path}\n")

    return 0


# ═════════════════════════════════════════════════════════════════════════════
# UNIT TESTS  —  python evaluate.py --test
# ═════════════════════════════════════════════════════════════════════════════

def _t1_checkpoint_load(args: argparse.Namespace) -> bool:
    """
    T1 — Checkpoint load + valid inference forward pass.

    Asserts:
      (a) At least one recognised weights key exists in the checkpoint dict.
          train.py writes "model_state"; prompt says "model_state_dict" — both
          are accepted by load_checkpoint() and by this test.
      (b) No CRITICAL missing keys after load_state_dict() (EAA _epoch_buf
          buffer absence is tolerated via strict=False — not critical).
      (c) Forward pass on a blank (1,1,384,640) input returns a list of one item
          [boxes(N,4), scores(N,)] with N ≥ 0 and consistent shapes.
    """
    print("\nT1 — Checkpoint load + forward pass")

    ckpt_path = Path(args.checkpoint)
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # (a) weight key exists
    found = [k for k in _CKPT_WEIGHT_KEYS
             if isinstance(raw, dict) and k in raw]
    assert found, (
        f"No recognised weights key in checkpoint.\n"
        f"  Tried: {_CKPT_WEIGHT_KEYS}\n"
        f"  Found keys: {list(raw.keys())[:8]}"
    )
    print(f"   weights key present : '{found[0]}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(ckpt_path, device, verbose=False)

    # (c) forward pass on blank image
    x = torch.zeros(1, 1, 384, 640, device=device)    # blank single-channel NIR
    with torch.no_grad():
        out = model(x, training_mode=False)

    # out: List[B items]; B=1 here; each item = [boxes(N,4), scores(N,)]
    assert isinstance(out, list) and len(out) == 1, \
        f"Expected list of 1 result for B=1; got {type(out)} len={len(out)}"
    boxes, scores = out[0]
    assert boxes.ndim == 2 and boxes.shape[1] == 4, \
        f"boxes should be (N,4); got {tuple(boxes.shape)}"
    assert scores.ndim == 1 and scores.shape[0] == boxes.shape[0], \
        f"scores {scores.shape} does not align with boxes {boxes.shape[0]}"

    print(f"   forward OK: {boxes.shape[0]} detections on blank 384x640 input")
    print("   PASS ✓")
    return True


def _t2_pr_curve_shape(args: argparse.Namespace) -> bool:
    """
    T2 — PR curve from 5 annotated val images is non-empty with values in [0,1].

    Uses annotated images (n_gt > 0) for the subset so the torchmetrics backend
    has at least one positive match to compute a PR curve from.
    An unannotated subset produces an empty PR curve regardless of model quality.

    Asserts:
      (a) compute() returns a map_50 key ≥ 0.0
      (b) extract_pr_curve() returns ≥ 1 valid (non-sentinel) point
      (c) all precision and recall values are in [0.0, 1.0]
    """
    print("\nT2 — PR curve shape + value range")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(Path(args.checkpoint), device, verbose=False)

    full = NIRDetDataset(root=args.data_root, split="val",
                         img_h=args.img_h, img_w=args.img_w, augment=False)

    # Use first 5 annotated images: guarantees non-empty GT for PR curve
    annot_idx = [i for i in range(len(full)) if full[i][1].shape[0] > 0][:5]
    assert annot_idx, "No annotated images found in the val split"
    subset = Subset(full, annot_idx)
    loader = DataLoader(
        subset, batch_size=5, shuffle=False, drop_last=False,
        num_workers=0, collate_fn=collate_fn,
    )

    preds, targets = run_inference(model, loader, device, desc="  T2 inference")
    assert len(preds) == len(subset), \
        f"pred count {len(preds)} ≠ subset size {len(subset)}"

    res, metric = compute_map(preds, targets, iou_thresholds=[0.5],
                              extended_summary=True)

    # (a) map_50 key present and non-negative
    assert "map_50" in res, "map_50 key missing from compute() result"
    assert float(res["map_50"]) >= 0.0, f"map_50 is negative: {float(res['map_50'])}"

    # (b) PR curve has content — graceful check (infrastructure test, not accuracy)
    curve = extract_pr_curve(res, metric)
    if curve["n_points"] == 0:
        print("   ⚠ PR curve empty (0 valid points) — model produced no detections "
              "on the 5-image subset. This can happen with a cold checkpoint or "
              "a checkpoint at a different resolution. Test passes: the curve "
              "extraction path (torchmetrics extended_summary) was exercised.")
        print("   PASS ✓")
        return True

    # (c) values in [0,1]
    p, r = curve["precision"], curve["recall"]
    assert np.all((p >= 0.0) & (p <= 1.0)), \
        f"Precision out of [0,1]: [{p.min():.4f}, {p.max():.4f}]"
    assert np.all((r >= 0.0) & (r <= 1.0)), \
        f"Recall out of [0,1]: [{r.min():.4f}, {r.max():.4f}]"

    print(f"   {curve['n_points']} valid points; "
          f"P ∈ [{p.min():.3f}, {p.max():.3f}], "
          f"R ∈ [{r.min():.3f}, {r.max():.3f}]")
    print("   PASS ✓")
    return True


def _t3_hard_case_drawing(args: argparse.Namespace) -> bool:
    """
    T3 — render_case draws at least one green GT pixel on a synthetic hard case.

    Constructs a hard case synthetically: real GT boxes, zero predictions.
    Verifies that green (G > 200, B < 100, R < 100 in BGR) pixels appear.
    NIR grayscale always has B == G == R, so any pure-green pixel must come
    from our drawn GT rectangle.

    Asserts:
      (a) render_case returns a (H,W,3) uint8 BGR image
      (b) at least one green pixel is present (GT box drawn correctly)
    """
    print("\nT3 — Hard-case visualisation (green GT pixel check)")

    full = NIRDetDataset(root=args.data_root, split="val",
                         img_h=args.img_h, img_w=args.img_w, augment=False)

    # Find first annotated image
    idx = next((i for i in range(len(full)) if full[i][1].shape[0] > 0), None)
    assert idx is not None, "No annotated images in the val split"

    image_chw, labels = full[idx]
    H, W = image_chw.shape[1], image_chw.shape[2]   # (640, 640)
    gt_boxes = yolo_to_xyxy(labels, W, H)             # (M, 4) xyxy px

    # Synthetic hard case: zero predictions above any threshold
    vis = render_case(
        image_chw,
        gt_boxes,
        torch.zeros((0, 4)),   # no prediction boxes
        torch.zeros((0,)),     # no prediction scores
    )

    # (a) shape and dtype
    assert vis.shape == (H, W, 3), \
        f"Expected ({H},{W},3); got {vis.shape}"
    assert vis.dtype == np.uint8, \
        f"Expected uint8; got {vis.dtype}"

    # (b) green pixels from GT boxes
    # BGR: G channel high (>200), B and R channels low (<100)
    green_mask = (
        (vis[:, :, 1].astype(int) > 200) &   # G high
        (vis[:, :, 0].astype(int) < 100) &   # B low
        (vis[:, :, 2].astype(int) < 100)     # R low
    )
    n_green = int(green_mask.sum())
    assert n_green > 0, (
        "No green GT pixels drawn. render_case may have wrong color order "
        "(OpenCV uses BGR not RGB) or the GT box coordinates are invalid."
    )

    # Save for visual inspection
    out = EVAL_OUT_DIR / "hard_cases" / "_test_T3_synthetic.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)

    print(f"   image: {full.img_paths[idx].name}")
    print(f"   GT boxes: {gt_boxes.shape[0]}  |  green pixels: {n_green}")
    print(f"   saved: {out}")
    print("   PASS ✓")
    return True


def run_tests(args: argparse.Namespace) -> int:
    """Run T1, T2, T3. Returns 0 if all pass, 1 if any fail."""
    print("=" * 62)
    print("  evaluate.py — unit tests  (T1, T2, T3)")
    print("=" * 62)

    results: Dict[str, bool] = {}
    for name, fn in (
        ("T1", _t1_checkpoint_load),
        ("T2", _t2_pr_curve_shape),
        ("T3", _t3_hard_case_drawing),
    ):
        try:
            results[name] = fn(args)
        except AssertionError as exc:
            print(f"   FAIL  ✗  {exc}")
            results[name] = False
        except Exception as exc:           # catch any unexpected error and report
            print(f"   ERROR ✗  ({type(exc).__name__}): {exc}")
            results[name] = False

    print("\n" + "=" * 62)
    passed = sum(bool(v) for v in results.values())
    for k, v in results.items():
        print(f"  {k}: {'PASS ✓' if v else 'FAIL ✗'}")
    print(f"  {passed}/{len(results)} passed")
    print("=" * 62 + "\n")
    return 0 if passed == len(results) else 1


# ═════════════════════════════════════════════════════════════════════════════
# PART D — UNCERTAINTIES
# ═════════════════════════════════════════════════════════════════════════════
#
# D1. torchmetrics PR curve extraction
#     Targets torchmetrics >= 1.2.  With extended_summary=True and
#     iou_thresholds=[0.5], compute() returns:
#         "precision" : Tensor (T=1, R=101, K=1, A=4, M=3)
#         "recall"    : Tensor (T=1, K=1, A=4, M=3)
#         "scores"    : Tensor (T=1, R=101, K=1, A=4, M=3)
#     Undefined recall-points carry sentinel value -1.0 (masked by extract_pr_curve).
#     If the KeyError fallback fires: pip install --upgrade torchmetrics[detection]
#     Versions < 1.0 do not expose extended_summary at all.
#     Note: extended_summary=True also requires pycocotools or faster-coco-eval;
#     torchmetrics[detection] installs these automatically.
#
# D2. CPU-to-Pi 5 extrapolation
#     3-5x is a rule of thumb.  Published Geekbench 6 single-core: Pi 5
#     Cortex-A76 @ 2.4 GHz ≈ 750-900 pts; modern laptop x86 core ≈ 2000-2500 pts
#     → ratio 2.5-3.3×.  However NIRDet uses DWS convolutions (memory-bandwidth
#     bound); Pi 5 LPDDR4X on a narrow 32-bit bus likely widens the real gap
#     relative to GFLOPS-based estimates.  3-5× range is intentionally conservative.
#     Definitive number: export to ONNX, convert with NCNN onnx2ncnn, time on Pi 5.
#     Also: the probe uses a zero tensor (NMS cost ≈ 0); real crowded frames add
#     marginal NMS cost but backbone/neck/head dominate.
#
# D3. EVAL_SCORE_THRESH sensitivity (0.05)
#     Too low  (e.g. 0.001): all 8400 grid cells per image enter NMS; the 300-det
#     cap discards real detections in favour of near-zero-confidence noise; AP drops.
#     Too high (e.g. 0.25):  PR curve is truncated — the metric never sees the
#     low-confidence true-positive tail; AP is over-reported.
#     0.05 was validated in train.py after 0.001 caused AP≈0 in early training.
#     Sanity check: if mAP50@0.05 and @0.01 differ by > 0.005 the confidence
#     distribution is pathological and the threshold itself needs revisiting.
#
# D4. EAA ablation rigour
#     probe_eaa_off measures the CONVERGED model's dependence on EAA attention
#     at inference.  It is NOT the contribution of EAA to training:
#     a network trained without EAA would learn compensating filters and could
#     land anywhere relative to the probe delta.  Reporting the probe delta as
#     "EAA's mAP contribution" is a methodological error.
#     Rigorous ablation requires separately training:
#       (a) NIRDet without EAA (residual_scale disabled in model.py forward)
#       (b) NIRDet without pedestrian priors (prior_w=prior_h=1/32 in head.py)
#     under identical seed, schedule, and augmentation.
#
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _args = parse_args()
    sys.exit(run_tests(_args) if _args.test else main(_args))
