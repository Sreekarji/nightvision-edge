"""
train.py — NIRDet Training Loop
================================
Requirements:
    pip install tqdm torchmetrics[detection]

Usage:
    # Fresh training
    python train.py

    # Resume from checkpoint
    python train.py --resume checkpoints/last.pth

    # Smoke test (5 epochs, 10 images)
    python train.py --smoke-test

    # Overfit test (50 epochs, 10 images)
    python train.py --overfit-test

All paths assume execution from C:\\projects\\nightvision\\nirdet\\
"""

import os
import sys
import time
import math
import argparse
import logging
from pathlib import Path
from copy import deepcopy

import csv
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ── Local imports (same directory) ──────────────────────────────────────────
from model import NIRDet
from losses import NIRDetLoss
from dataset import NIRDetDataset, collate_fn

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nirdet")

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — edit here, not scattered through the code
# ════════════════════════════════════════════════════════════════════════════

CFG = {
    # ── Paths ────────────────────────────────────────────────────────────────
    "data_root":     r"C:\projects\nightvision\data\raw\miniNIRPed",
    "checkpoint_dir": r"C:\projects\nightvision\nirdet\checkpoints",

    # ── Model ────────────────────────────────────────────────────────────────
    "num_classes": 1,           # single class: person

    # ── Training schedule ────────────────────────────────────────────────────
    "epochs":        100,       # Keep at 100 — enough to prove the architecture on miniNIRPed without
                                # overfitting. Real production training on larger datasets will use more.
    "warmup_steps":  300,       # Step-based warmup: dataset-size invariant.
                                # At 261 images batch=8: 300 steps ≈ 9 epochs of warmup.
                                # At 10,000 images batch=8: still 300 steps ≈ 0.2 epochs — auto-scales.
    "lr_peak":       1e-3,      # peak LR after warmup (cosine decays to lr_min)
    "lr_start":      1e-5,      # LR at step 0 (start of warmup)
    "lr_min":        1e-6,      # final LR at end of cosine decay
    "weight_decay":  5e-4,
    "momentum":      0.937,     # SGD momentum (ignored if using Adam)

    # ── Optimiser ────────────────────────────────────────────────────────────
    # "sgd" or "adamw" — Adam is friendlier for small datasets from scratch
    "optimizer":     "adamw",

    # ── Data loader ──────────────────────────────────────────────────────────
    "batch_size":    8,         # tune down to 4 if OOM
    "num_workers":   4,         # Windows: set to 0 if DataLoader hangs
    "img_h":         384,       # target input height (640×384 geometry)
    "img_w":         640,       # target input width

    # ── Mixed precision ──────────────────────────────────────────────────────
    # "fp16", "bf16", or "fp32"
    # RTX 4050 supports bf16 — prefer it (fewer NaNs, no GradScaler needed)
    "amp_mode":      "bf16",

    # ── Gradient clipping ────────────────────────────────────────────────────
    "grad_clip_norm": 10.0,     # loosen to 35 if frequently clipping;
                                # tighten to 5 if NaN appears on epoch 1

    # ── Validation / checkpointing ───────────────────────────────────────────
    "val_interval":  5,         # Evaluate every 5 epochs for epochs 1-75.
                                # Dense eval (every epoch) for final 25 epochs — catches the exact peak.
    "save_last":     True,      # always save last.pth after each epoch

    # ── Early stopping (mAP50-based, in eval intervals) ──────────────────────
    "es_enabled":   False,      # Disabled: at 261 images val mAP standard error is ~±0.02.
                                # Early stopping on noisy val signal killed the previous run at epoch 80
                                # before the cosine schedule finished. Let it run all 100 epochs and
                                # pick best.pth. Re-enable with True when dataset > 2000 images.

    # ── Baseline to beat ─────────────────────────────────────────────────────
    "baseline_map50": 0.735,    # YOLO11n fine-tuned reference

    # ── Deployment ───────────────────────────────────────────────────────────
    "deploy_score_thresh": 0.301,  # F1-optimal threshold measured from evaluate.py PR curve.
                                   # Re-derive after Group I training (QFL shifts score distribution).

    # ── EMA (exponential moving average of weights) ──────────────────────────
    "use_ema":      True,
    "ema_decay":    0.995,
    "ema_tau_steps": 160,    # ≈ 5 epochs at 32 steps/epoch on miniNIRPed
}

# ════════════════════════════════════════════════════════════════════════════
# LEARNING-RATE SCHEDULE
# ════════════════════════════════════════════════════════════════════════════

def get_lr(step: int, steps_per_epoch: int, cfg: dict) -> float:
    """
    Step-based LR schedule — dataset-size invariant.

    Warmup is in STEPS not epochs:
      300 steps at 261 images (batch=8) = ~9 epochs
      300 steps at 10,000 images (batch=8) = ~0.2 epochs
    Both give the same number of gradient steps of warmup.

    Phases:
      [0, warmup_steps):           linear ramp lr_start → lr_peak
      [warmup_steps, total_steps]: cosine decay lr_peak → lr_min
    """
    total_steps = cfg["epochs"] * steps_per_epoch
    warm        = cfg["warmup_steps"]
    peak        = cfg["lr_peak"]
    start       = cfg["lr_start"]
    min_lr      = cfg["lr_min"]

    if step < warm:
        frac = step / max(warm - 1, 1)
        return start + frac * (peak - start)
    else:
        progress = (step - warm) / max(total_steps - warm, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine * (peak - min_lr)


def set_lr(optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def build_param_groups(model: nn.Module, weight_decay: float) -> list:
    """Split model parameters into decay (2D+ conv/linear weights) and
    no-decay (1D BN/GN scale+shift + all biases) groups.

    Standard practice from YOLOv5, YOLOX, FCOS: normalisation layer
    scale/shift params and all biases are exempt from weight decay,
    which prevents the effective gain of BN/GN from shrinking over
    training — especially important on small datasets (261 images).
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Decay constant is scaled to THIS project's run:
      Total steps at 100 epochs, 261 images, batch=8: ~3,200 steps.
      YOLOv5 d=0.9999 → tau=10,000 steps > entire run → EMA never converges.
      Correct: d=0.995 → tau=200 steps ≈ 6 epochs → EMA converges in first
      quarter of training and tracks the model usefully throughout.

    Formula: d_eff(step) = d * (1 - exp(-step / tau))
    Ramps from 0 at step 0 (EMA = live weights) to d at large steps.
    Prevents EMA from being dominated by bad early weights.

    BN buffers (running_mean/var) are COPIED not averaged — they are
    already running statistics and averaging them biases variance down.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.995,
        tau_steps: int = 160,    # ≈ 5 epochs at 32 steps/epoch
    ):
        self.ema     = deepcopy(model).eval()
        self.decay   = decay
        self.tau     = tau_steps
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d   = self.decay * (1.0 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        named_params = dict(self.ema.named_parameters())
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point and k in named_params:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return {"ema": self.ema.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd["ema"])
        self.updates = sd.get("updates", 0)


# ════════════════════════════════════════════════════════════════════════════
# MIXED-PRECISION HELPERS
# ════════════════════════════════════════════════════════════════════════════

def build_amp_context(amp_mode: str):
    """
    Returns (autocast_context, scaler).
    GradScaler is only needed for fp16 (not bf16 or fp32).
    """
    if amp_mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            log.warning("GPU does not support bf16 — falling back to fp16.")
            amp_mode = "fp16"
        else:
            ctx    = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            scaler = torch.amp.GradScaler("cuda", enabled=False)  # no-op
            return ctx, scaler

    if amp_mode == "fp16":
        ctx    = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=2**14)
        return ctx, scaler

    # fp32 — no AMP
    ctx    = torch.amp.autocast(device_type="cuda", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return ctx, scaler

# ════════════════════════════════════════════════════════════════════════════
# VAL mAP50 EVALUATION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_map50(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp_ctx,
    iou_threshold: float = 0.5,
    eval_score_thresh: float = 0.05,   # was 0.001 — too low, floods metric with 8400 background cells  # FIX: near-zero threshold so mAP integrates the full PR curve (0.25 clips it to 0 early in training)
) -> float:
    """
    Runs inference on the validation set and returns mAP@IoU=0.50.

    IMPORTANT: This function assumes model.forward() returns a list of
    per-image prediction dicts (boxes, scores, labels) in evaluation mode.
    Adjust if your NIRDet inference interface differs — see Part D.
    """
    model.eval()
    # FIX: temporarily lower the NMS score threshold for evaluation only;
    # deployment keeps model.nms_score_thresh = 0.25. Restored in finally.
    saved_thresh = model.nms_score_thresh
    model.nms_score_thresh = eval_score_thresh
    metric = MeanAveragePrecision(
    iou_thresholds=[iou_threshold],
    box_format="xyxy",
    max_detection_thresholds=[1, 10, 300],
)

    try:
        for images, targets in tqdm(val_loader, desc="  val mAP50", leave=False):
            # collate_fn already stacks images to (B,1,H,W)
            images = images.to(device)
            img_h, img_w = images.shape[2], images.shape[3]

            with amp_ctx:
                # training=False returns list of B pairs: [boxes(N,4), scores(N,)]
                batch_results = model(images, training_mode=False)

            # Build torchmetrics-compatible preds and targets
            preds_cpu = []
            for result in batch_results:
                boxes, scores = result[0], result[1]
                preds_cpu.append({
                    "boxes":  boxes.cpu().float(),
                    "scores": scores.cpu().float(),
                    "labels": torch.zeros(scores.shape[0], dtype=torch.long),
                })

            # targets: list of (N_i,5) tensors with cols [cls, cx, cy, w, h] norm
            targets_cpu = []
            for t in targets:
                if t.numel() == 0:
                    targets_cpu.append({
                        "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                        "labels": torch.zeros(0, dtype=torch.long),
                    })
                else:
                    cx = t[:, 1] * img_w
                    cy = t[:, 2] * img_h
                    w  = t[:, 3] * img_w
                    h  = t[:, 4] * img_h
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    boxes = torch.stack([x1, y1, x2, y2], dim=1)
                    targets_cpu.append({
                        "boxes":  boxes.cpu().float(),
                        "labels": t[:, 0].long().cpu(),
                    })

            metric.update(preds_cpu, targets_cpu)

        result = metric.compute()
        map50  = result["map_50"].item()
        metric.reset()
        return map50
    finally:
        # Always restore training mode even if an exception occurs during eval.
        # Without this, a mid-eval exception leaves the model in .eval() for the
        # next training epoch — BN running statistics and dropout are wrong.
        model.nms_score_thresh = saved_thresh   # FIX: restore deployment threshold
        model.train()

# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer,
    scaler,
    scheduler_state: dict,
    best_map50: float,
    cfg: dict,
    ema=None,
) -> None:
    ckpt = {
        "epoch":          epoch,
        "model_state":    (ema.ema.state_dict() if ema is not None else model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":   scaler.state_dict(),
        "scheduler_state": scheduler_state,
        "best_map50":     best_map50,
        "cfg":            cfg,
        "ema_state":      (ema.state_dict() if ema is not None else None),
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, scaler, ema=None):
    """Load checkpoint and return (start_epoch, best_map50, scheduler_state)."""
    log.info(f"Resuming from {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    if ema is not None and ckpt.get("ema_state"):
        ema.load_state_dict(ckpt["ema_state"])
    return (
        ckpt["epoch"] + 1,
        ckpt.get("best_map50", 0.0),
        ckpt.get("scheduler_state", {}),
    )

# ════════════════════════════════════════════════════════════════════════════
# ONE TRAINING EPOCH
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:      nn.Module,
    criterion:  nn.Module,
    loader:     DataLoader,
    optimizer,
    scaler,
    amp_ctx,
    device:     torch.device,
    epoch:      int,
    cfg:        dict,
    global_step: int,
    steps_per_epoch: int,
    ema=None,
) -> tuple[dict, int]:
    """
    Runs one full epoch of training.
    Returns (mean_losses_dict, updated_global_step).
    """
    model.train()
    total_loss = cls_loss = reg_loss = conf_loss = 0.0
    n_batches  = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d}", leave=True, dynamic_ncols=True)

    for batch_idx, (images, targets) in enumerate(pbar):
        # ── Step-wise LR schedule (dataset-size invariant) ───────────────────
        lr = get_lr(global_step, steps_per_epoch, cfg)
        set_lr(optimizer, lr)
        global_step += 1
        # ── Move to device ───────────────────────────────────────────────────
        # collate_fn already stacks images to (B,1,H,W) — just move to device
        images = images.to(device, non_blocking=True)

        # targets: list of B tensors, each (N_i, 5) with cols [cls, cx, cy, w, h]
        # NIRDetLoss expects gt_batch[b] = list of (4,) tensors [cx, cy, w, h]
        gt_batch = []
        for t in targets:
            t = t.to(device)
            if t.numel() == 0:
                gt_batch.append([])
            else:
                # Drop class column (col 0), keep cx,cy,w,h (cols 1-4)
                # Split into list of (4,) tensors, one per GT box
                gt_batch.append([t[i, 1:] for i in range(t.shape[0])])

        optimizer.zero_grad(set_to_none=True)

        # ── Forward pass under autocast ──────────────────────────────────────
        with amp_ctx:
            # training=True returns raw logits: [(B,3840,5),(B,960,5),(B,240,5)]
            outputs = model(images, training_mode=True)

        # ── Loss computation OUTSIDE autocast (fp32 for numerical safety) ────
        outputs_fp32 = [o.float() for o in outputs]
        losses = criterion(outputs_fp32, gt_batch)

        # NIRDetLoss must return a dict: {total, cls, reg, conf}
        # If it returns a single tensor, wrap it:
        if isinstance(losses, torch.Tensor):
            losses = {"total": losses, "cls": losses, "reg": losses, "conf": losses}

        loss = losses["total"]

        # ── Guard against NaN before backprop ────────────────────────────────
        if not torch.isfinite(loss):
            log.warning(
                f"  [epoch {epoch+1} batch {batch_idx}] "
                f"Non-finite loss={loss.item():.4f} — skipping batch."
            )
            continue

        # ── Backward + gradient clip + optimizer step ────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # unscale before clip so threshold is in true units

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=cfg["grad_clip_norm"]
        )

        scaler.step(optimizer)
        scaler.update()

        # ── EMA update after the optimizer step ──────────────────────────────
        if ema is not None:
            ema.update(model)

        # ── Accumulate losses for logging ────────────────────────────────────
        total_loss += losses["total"].item()
        cls_loss   += losses.get("cls",  losses["total"]).item()
        reg_loss   += losses.get("reg",  losses["total"]).item()
        conf_loss  += losses.get("conf", losses["total"]).item()

        pbar.set_postfix(
            loss=f"{losses['total'].item():.4f}",
            cls=f"{losses.get('cls', losses['total']).item():.4f}",
            reg=f"{losses.get('reg', losses['total']).item():.4f}",
            conf=f"{losses.get('conf', losses['total']).item():.4f}",
            gnorm=f"{grad_norm:.2f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return {
        "total": total_loss / n_batches,
        "cls":   cls_loss   / n_batches,
        "reg":   reg_loss   / n_batches,
        "conf":  conf_loss  / n_batches,
    }, global_step

# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    # ── Deterministic seeding ────────────────────────────────────────────────
    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    cfg = deepcopy(CFG)

    # ── Override config for smoke / overfit tests ─────────────────────────────
    if args.smoke_test:
        cfg["epochs"]       = 5
        cfg["val_interval"] = 5   # eval once at the end
        cfg["es_enabled"]   = False  # early stopping off for smoke test
        log.info("=== SMOKE TEST MODE: 5 epochs, 10 images ===")

    if args.overfit_test:
      cfg["epochs"]        = 300
      cfg["warmup_steps"]  = 0
      cfg["lr_peak"]       = 1e-3
      cfg["lr_start"]      = 1e-3
      cfg["lr_min"]        = 1e-3    # flat LR — cosine decay masks real plateaus
      cfg["batch_size"]    = 2       # 5 steps/epoch × 300 = 1500 gradient steps
      cfg["val_interval"]  = 50
      cfg["es_enabled"]    = False
      log.info("=== OVERFIT TEST MODE: 300 epochs, batch=2, flat LR ===")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    log.info("Loading datasets …")
    if args.smoke_test or args.overfit_test:
        train_ds = NIRDetDataset(
            root=cfg["data_root"], split="train",
            img_h=cfg["img_h"], img_w=cfg["img_w"],
            augment=False   # FIX: disable augmentation for smoke/overfit tests
        )
    else:
        train_ds = NIRDetDataset(
            root=cfg["data_root"], split="train",
            img_h=cfg["img_h"], img_w=cfg["img_w"]
        )
    val_ds = NIRDetDataset(
        root=cfg["data_root"], split="val",
        img_h=cfg["img_h"], img_w=cfg["img_w"],
        augment=False   # FIX: dataset default is augment=True; validation must be deterministic
    )

    if args.smoke_test or args.overfit_test:
        # Use exactly 10 training images; val uses the SAME 10 for overfit check
        indices = list(range(min(10, len(train_ds))))
        train_ds = Subset(train_ds, indices)
        # FIX: overfit/smoke must evaluate on the SAME images it trained on.
        # Previously val_ds was Subset(val split, ...) — a DIFFERENT set of images,
        # so the "overfit test" was actually measuring generalization from 10
        # images (mAP≈0 by definition) instead of verifying the model can fit.
        val_ds   = train_ds
        cfg["batch_size"] = len(train_ds)   # FIX: batch_size = 10, one batch covers all images (no drop_last)
        log.info(f"Subset: {len(train_ds)} train images, {len(val_ds)} val images (same images — overfit check).")
    else:
        log.info(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=(not (args.smoke_test or args.overfit_test)),   # FIX: drop_last only for regular training, not smoke/overfit
        persistent_workers=(cfg["num_workers"] > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg["num_workers"] > 0),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("Building NIRDet …")
    model = NIRDet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # ── EMA shadow model ─────────────────────────────────────────────────────
    # Must be created BEFORE load_checkpoint() so a resumed run restores the
    # EMA state along with the live weights.
    ema = ModelEMA(model, decay=cfg["ema_decay"], tau_steps=cfg["ema_tau_steps"]) \
          if cfg.get("use_ema", True) else None

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Grid sizes derive from (img_h, img_w) ÷ strides (8,16,32):
    #   e.g. 384×640 → ((48,80), (24,40), (12,20)).
    # Each entry is (S_h, S_w) — losses.py must receive the rectangular tuple
    # form, otherwise assign/decode use square math on a non-square grid.
    grid_sizes = tuple(
        (cfg["img_h"] // s, cfg["img_w"] // s) for s in (8, 16, 32)
    )
    criterion = NIRDetLoss(
        grid_sizes=grid_sizes,
        lambda_cls=1.0,
        lambda_reg=2.0,
        quality_target=True,
        qfl_ramp_frac=0.20,       # ramp over first 20 epochs of 100
        total_epochs=cfg["epochs"],
        img_h=cfg["img_h"],
        img_w=cfg["img_w"],
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # FIX: split params into decay/no-decay groups (biases + 1D BN/GN params
    # exempt from weight decay).  Both optimizers consume the same groups.
    param_groups = build_param_groups(model, cfg["weight_decay"])
    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg["lr_start"],         # will be overwritten each epoch
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups,
            lr=cfg["lr_start"],
            momentum=cfg["momentum"],
            nesterov=True,
        )

    # ── Mixed precision ───────────────────────────────────────────────────────
    amp_ctx, scaler = build_amp_context(cfg["amp_mode"])
    log.info(f"AMP mode: {cfg['amp_mode']}")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch  = 0
    best_map50   = 0.0
    es_counter   = 0            # consecutive evals with no improvement

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch, best_map50, _ = load_checkpoint(
                resume_path, model, optimizer, scaler, ema
            )
            log.info(f"Resumed from epoch {start_epoch}, best mAP50 = {best_map50:.4f}")
        else:
            log.warning(f"Checkpoint {args.resume} not found — training from scratch.")

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"  Starting training: epochs {start_epoch+1} → {cfg['epochs']}")
    log.info(f"  Baseline mAP50 to beat: {cfg['baseline_map50']}")
    log.info("=" * 60)

    epoch_times = []

    # ── Global step counter for step-wise LR schedule ─────────────────────────
    global_step = 0
    steps_per_epoch = max(1, len(train_loader))

    for epoch in range(start_epoch, cfg["epochs"]):

        # ── Step EAA freeze schedule ──────────────────────────────────────────
        model.step_eaa_epoch()

        # ── QFL ramp schedule (epoch → soft IoU confidence targets) ──────────
        criterion.set_epoch(epoch)

        # ── Train one epoch (LR updated per-step inside) ──────────────────────
        t0     = time.perf_counter()
        losses, global_step = train_one_epoch(
            model, criterion, train_loader,
            optimizer, scaler, amp_ctx, device, epoch, cfg,
            global_step, steps_per_epoch,
            ema,
        )
        t1     = time.perf_counter()
        elapsed = t1 - t0
        epoch_times.append(elapsed)

        # ── LR at the last training step of this epoch (for logging) ──────────
        lr = get_lr(max(global_step - 1, 0), steps_per_epoch, cfg)

        # ── Print per-epoch stats ─────────────────────────────────────────────
        log.info(
            f"Epoch {epoch+1:03d}/{cfg['epochs']} | "
            f"lr={lr:.2e} | "
            f"total={losses['total']:.4f} | "
            f"cls={losses['cls']:.4f} | "
            f"reg={losses['reg']:.4f} | "
            f"conf={losses['conf']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        # ── After epoch 1: print ETA ──────────────────────────────────────────
        if epoch == start_epoch:
            remaining = cfg["epochs"] - start_epoch - 1
            eta_mins  = elapsed * remaining / 60
            log.info(
                f"  Estimated time per epoch: {elapsed:.1f}s | "
                f"  ETA for full run: {eta_mins:.0f} min "
                f"({eta_mins/60:.1f} hr)"
            )

        # ── Save last checkpoint ──────────────────────────────────────────────
        if cfg["save_last"]:
            save_checkpoint(
                ckpt_dir / "last.pth",
                epoch, model, optimizer, scaler,
                {},     # scheduler_state (we compute lr on the fly)
                best_map50, cfg,
                ema,
            )

        # ── Validation mAP50 ─────────────────────────────────────────────────
        # Evaluate every val_interval epochs for epochs 1-75; every epoch for
        # the final 25 epochs (dense eval catches the exact cosine-decay peak).
        is_last_epoch = (epoch + 1 == cfg["epochs"])
        is_late = (epoch + 1) > int(0.75 * cfg["epochs"])  # final 25 epochs
        interval = 1 if is_late else cfg["val_interval"]
        if ((epoch + 1) % interval == 0) or is_last_epoch:

            log.info(f"  [Eval] Running mAP50 on {len(val_ds)} val images …")
            eval_model = ema.ema if ema is not None else model
            map50 = evaluate_map50(eval_model, val_loader, device, amp_ctx)
            log.info(
                f"  [Eval] mAP50 = {map50:.4f} "
                f"(best = {best_map50:.4f}, baseline = {cfg['baseline_map50']:.4f})"
            )

            # ── Per-epoch CSV history (append one row after every eval) ──────
            hist_path = ckpt_dir / "history.csv"
            write_header = not hist_path.exists()
            with open(hist_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "lr", "total", "cls", "reg", "conf", "map50", "seconds"])
                writer.writerow([
                    epoch + 1,
                    f"{lr:.6e}",
                    f"{losses['total']:.6f}",
                    f"{losses['cls']:.6f}",
                    f"{losses['reg']:.6f}",
                    f"{losses['conf']:.6f}",
                    f"{map50:.6f}",
                    f"{elapsed:.2f}",
                ])

            # ── Best checkpoint ───────────────────────────────────────────────
            if map50 > best_map50 + 0.005:
                best_map50 = map50
                es_counter = 0
                save_checkpoint(
                    ckpt_dir / "best.pth",
                    epoch, model, optimizer, scaler,
                    {}, best_map50, cfg,
                    ema,
                )
                beat_flag = "  ✓ BEATS BASELINE" if map50 >= cfg["baseline_map50"] else ""
                log.info(
                    f"  [Checkpoint] New best mAP50 = {best_map50:.4f} saved.{beat_flag}"
                )

            # ── Early stopping ────────────────────────────────────────────────
            # es_enabled=False: runs all 100 epochs, relies on best.pth.
            if cfg.get("es_enabled", False):
                if map50 > best_map50 + 0.005:
                    es_counter = 0
                else:
                    es_counter += 1
                    log.info(f"  [EarlyStopping] No improvement ({es_counter} eval(s)).")
                if es_counter >= cfg.get("es_patience", 8):
                    log.info(f"  [EarlyStopping] Stopping at epoch {epoch+1}.")
                    break

    # ── End of training ───────────────────────────────────────────────────────
    avg_epoch_time = sum(epoch_times) / max(len(epoch_times), 1)
    log.info("=" * 60)
    log.info(f"Training complete.")
    log.info(f"  Best mAP50:         {best_map50:.4f}")
    log.info(f"  Baseline mAP50:     {cfg['baseline_map50']:.4f}")
    log.info(f"  Delta:              {best_map50 - cfg['baseline_map50']:+.4f}")
    log.info(f"  Avg epoch time:     {avg_epoch_time:.1f}s")
    log.info(f"  Best checkpoint:    {ckpt_dir / 'best.pth'}")
    log.info("=" * 60)

    # ── Auto-evaluate on the best checkpoint just saved ──────────────────────
    # Normal completion only: reached after the training loop finishes (or
    # early-stops cleanly).  Skipped for smoke/overfit modes, which train on
    # a 10-image subset — running the full val suite on that is meaningless.
    # An interrupted run never reaches this point.
    if not (args.smoke_test or args.overfit_test):
        best_ckpt_path = ckpt_dir / "best.pth"
        if best_ckpt_path.exists():
            log.info(f"Auto-evaluating with evaluate.py on {best_ckpt_path} …")
            import subprocess, sys
            subprocess.run([
                sys.executable, "evaluate.py",
                "--checkpoint", str(best_ckpt_path),
                "--data-root",  cfg["data_root"],
            ], check=True)
        else:
            log.warning(
                f"No best checkpoint saved (mAP50 never improved) — "
                f"skipping auto-evaluation."
            )


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NIRDet training loop")
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (e.g. checkpoints/last.pth)"
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="5 epochs on 10 images: verify loss decreases, no NaN, no OOM"
    )
    p.add_argument(
        "--overfit-test", action="store_true",
        help="50 epochs on 10 images: loss should reach near-zero"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
