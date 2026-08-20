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
    "epochs":        100,       # total training epochs
    "warmup_epochs": 5,         # linear LR warmup from lr_start → lr_peak
    "lr_peak":       1e-3,      # peak LR after warmup (cosine decays to lr_min)
    "lr_start":      1e-5,      # LR at epoch 0 (start of warmup)
    "lr_min":        1e-6,      # final LR at end of cosine decay
    "weight_decay":  5e-4,
    "momentum":      0.937,     # SGD momentum (ignored if using Adam)

    # ── Optimiser ────────────────────────────────────────────────────────────
    # "sgd" or "adamw" — Adam is friendlier for small datasets from scratch
    "optimizer":     "adamw",

    # ── Data loader ──────────────────────────────────────────────────────────
    "batch_size":    8,         # tune down to 4 if OOM
    "num_workers":   4,         # Windows: set to 0 if DataLoader hangs
    "img_size":      640,

    # ── Mixed precision ──────────────────────────────────────────────────────
    # "fp16", "bf16", or "fp32"
    # RTX 4050 supports bf16 — prefer it (fewer NaNs, no GradScaler needed)
    "amp_mode":      "bf16",

    # ── Gradient clipping ────────────────────────────────────────────────────
    "grad_clip_norm": 10.0,     # loosen to 35 if frequently clipping;
                                # tighten to 5 if NaN appears on epoch 1

    # ── Validation / checkpointing ───────────────────────────────────────────
    "val_interval":  5,         # run mAP50 eval every N epochs
    "save_last":     True,      # always save last.pth after each epoch

    # ── Early stopping (mAP50-based, in eval intervals) ──────────────────────
    "es_patience":   4,         # stop after 4 consecutive evals (= 20 epochs)
                                # with no mAP50 improvement
    "es_min_delta":  1e-4,      # improvement threshold (absolute mAP50 units)

    # ── Baseline to beat ─────────────────────────────────────────────────────
    "baseline_map50": 0.735,    # YOLO11n fine-tuned reference
}

# ════════════════════════════════════════════════════════════════════════════
# LEARNING-RATE SCHEDULE
# ════════════════════════════════════════════════════════════════════════════

def get_lr(epoch: int, cfg: dict) -> float:
    """
    Returns the LR multiplier (× lr_peak) for a given epoch.

    Epochs [0, warmup_epochs):   linear ramp  lr_start → lr_peak
    Epochs [warmup_epochs, end]: cosine decay lr_peak  → lr_min
    """
    warmup = cfg["warmup_epochs"]
    total  = cfg["epochs"]
    peak   = cfg["lr_peak"]
    start  = cfg["lr_start"]
    min_lr = cfg["lr_min"]

    if warmup > 0 and epoch < warmup:
        # Linear warmup: fraction of the way from start to peak
        frac = epoch / max(warmup - 1, 1)
        return start + frac * (peak - start)
    else:
        # Cosine annealing from peak to min_lr
        progress = (epoch - warmup) / max(total - warmup - 1, 1)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + cosine * (peak - min_lr)


def set_lr(optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr

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
) -> None:
    ckpt = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":   scaler.state_dict(),
        "scheduler_state": scheduler_state,
        "best_map50":     best_map50,
        "cfg":            cfg,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, scaler):
    """Load checkpoint and return (start_epoch, best_map50, scheduler_state)."""
    log.info(f"Resuming from {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
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
) -> dict:
    """
    Runs one full epoch of training.
    Returns dict of mean losses: {total, cls, reg, conf}.
    """
    model.train()
    total_loss = cls_loss = reg_loss = conf_loss = 0.0
    n_batches  = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d}", leave=True, dynamic_ncols=True)

    for batch_idx, (images, targets) in enumerate(pbar):
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
            # training=True returns raw logits: [(B,6400,5),(B,1600,5),(B,400,5)]
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
    }

# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    cfg = deepcopy(CFG)

    # ── Override config for smoke / overfit tests ─────────────────────────────
    if args.smoke_test:
        cfg["epochs"]       = 5
        cfg["val_interval"] = 5   # eval once at the end
        cfg["es_patience"]  = 999  # disable early stopping
        log.info("=== SMOKE TEST MODE: 5 epochs, 10 images ===")

    if args.overfit_test:
      cfg["epochs"]        = 300
      cfg["warmup_epochs"] = 0
      cfg["lr_peak"]       = 1e-3
      cfg["lr_start"]      = 1e-3
      cfg["lr_min"]        = 1e-3    # flat LR — cosine decay masks real plateaus
      cfg["batch_size"]    = 2       # 5 steps/epoch × 300 = 1500 gradient steps
      cfg["val_interval"]  = 50
      cfg["es_patience"]   = 999
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
            img_size=cfg["img_size"], augment=False   # FIX: disable augmentation for smoke/overfit tests
        )
    else:
        train_ds = NIRDetDataset(
            root=cfg["data_root"], split="train",
            img_size=cfg["img_size"]
        )
    val_ds = NIRDetDataset(
        root=cfg["data_root"], split="val",
        img_size=cfg["img_size"], augment=False   # FIX: dataset default is augment=True; validation must be deterministic
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

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = NIRDetLoss(lambda_cls=1.0, lambda_reg=2.0).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr_start"],         # will be overwritten each epoch
            weight_decay=cfg["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["lr_start"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
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
                resume_path, model, optimizer, scaler
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

    for epoch in range(start_epoch, cfg["epochs"]):

        # ── Set LR for this epoch ─────────────────────────────────────────────
        lr = get_lr(epoch, cfg)
        set_lr(optimizer, lr)

        # ── Step EAA freeze schedule ──────────────────────────────────────────
        model.step_eaa_epoch()

        # ── Train one epoch ───────────────────────────────────────────────────
        t0     = time.perf_counter()
        losses = train_one_epoch(
            model, criterion, train_loader,
            optimizer, scaler, amp_ctx, device, epoch, cfg,
        )
        t1     = time.perf_counter()
        elapsed = t1 - t0
        epoch_times.append(elapsed)

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
            )

        # ── Validation mAP50 ─────────────────────────────────────────────────
        # Evaluate on: every val_interval epochs, AND on the final epoch
        is_last_epoch = (epoch + 1 == cfg["epochs"])
        if ((epoch + 1) % cfg["val_interval"] == 0) or is_last_epoch:

            log.info(f"  [Eval] Running mAP50 on {len(val_ds)} val images …")
            map50 = evaluate_map50(model, val_loader, device, amp_ctx)
            log.info(
                f"  [Eval] mAP50 = {map50:.4f} "
                f"(best = {best_map50:.4f}, baseline = {cfg['baseline_map50']:.4f})"
            )

            # ── Best checkpoint ───────────────────────────────────────────────
            # FIX: don't start the early-stop counter until mAP50 has cleared a
            # warmup floor — before that, "no improvement over 0.0" is meaningless
            # and would kill the run before confidence logits have risen.
            ES_WARMUP_FLOOR = 1e-3
            if map50 > best_map50 + cfg["es_min_delta"]:
                best_map50 = map50
                es_counter = 0
                save_checkpoint(
                    ckpt_dir / "best.pth",
                    epoch, model, optimizer, scaler,
                    {}, best_map50, cfg,
                )
                beat_flag = "  ✓ BEATS BASELINE" if map50 >= cfg["baseline_map50"] else ""
                log.info(
                    f"  [Checkpoint] New best mAP50 = {best_map50:.4f} saved.{beat_flag}"
                )
            elif best_map50 > ES_WARMUP_FLOOR:
                es_counter += 1   # FIX: only count non-improvement once mAP50 has left ~0
                log.info(
                    f"  [EarlyStopping] No improvement "
                    f"({es_counter}/{cfg['es_patience']} patience evals used)."
                )
            else:
                log.info("  [EarlyStopping] mAP50 still ~0 — patience counter not started yet.")

            # ── Early stopping check ──────────────────────────────────────────
            if best_map50 > ES_WARMUP_FLOOR and es_counter >= cfg["es_patience"]:
                log.info(
                    f"  [EarlyStopping] Patience exhausted after "
                    f"{es_counter * cfg['val_interval']} epochs without "
                    f"mAP50 improvement. Stopping."
                )
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
