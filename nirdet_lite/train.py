#!/usr/bin/env python
"""
train.py — NIRDet-Lite training
================================
Fixes relative to the original
------------------------------
1. RESUME NO LONGER DESTROYS THE LIVE WEIGHTS. The old checkpoint wrote
   ``ema.ema.state_dict()`` under the key "model_state", so resuming loaded
   smoothed EMA weights into the live model and the true optimisation
   trajectory was lost forever. Live and EMA weights are now stored under
   separate keys ("model_state" / "ema_state") plus a "deploy_state" copy of
   whichever set should be used for inference.

2. EAA freeze state is restored on resume (attention._load_from_state_dict
   plus an explicit set_eaa_epoch after loading).

3. The early-stopping counter can actually reset: the improvement flag is
   computed BEFORE best_map50 is updated. Previously the ES test compared
   against an already-updated best and could never see an improvement.

4. The 0.005 best-checkpoint deadband is removed. On 160 val images the noise
   is larger than the deadband, so it only ever hid real improvements.

5. Single source of truth: every hyperparameter comes from config.py. There is
   no module-level CFG dict competing with the dataclasses.

6. Channel-summed ImageNet stem transfer (train.imagenet_stem_init): the RGB
   weights of a pretrained stem are summed across the channel axis to produce
   1-channel NIR filters. This is transfer learning with no new data, which
   matters a great deal at 261 images.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config, get_config, validate_config
from dataset import NIRDetDataset, collate_fn, seed_worker
from losses import NIRDetLoss
from model import build_nirdet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("nirdet")

CKPT_LIVE_KEY = "model_state"
CKPT_EMA_KEY = "ema_state"
CKPT_DEPLOY_KEY = "deploy_state"


# ===========================================================================
# reproducibility
# ===========================================================================

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ===========================================================================
# LR schedule (step-based: dataset-size invariant)
# ===========================================================================

def get_lr(step: int, steps_per_epoch: int, cfg: Config) -> float:
    t = cfg.train
    total = max(1, t.epochs * steps_per_epoch)
    warm = max(0, t.warmup_steps)
    if step < warm:
        frac = step / max(warm - 1, 1)
        return t.lr_start + frac * (t.lr_peak - t.lr_start)
    prog = (step - warm) / max(total - warm, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return t.lr_min + cos * (t.lr_peak - t.lr_min)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)


# Parameters that are learned offsets rather than weights. Decay on these
# pulls them back toward their uninformative initialisation. Matched by
# substring because reg_level_scale is 2-D and invisible to the ndim <= 1 test.
_NO_DECAY_SUBSTRINGS = ("reg_level_scale",)


def build_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    """1-D tensors (BN scale/shift), biases, and learned offsets are decay-exempt."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        exempt = (p.ndim <= 1
                  or name.endswith(".bias")
                  or any(s in name for s in _NO_DECAY_SUBSTRINGS))
        (no_decay if exempt else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "lr_scale": 1.0},
        {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
    ]


# ===========================================================================
# EMA
# ===========================================================================

class ModelEMA:
    """
    d_eff(step) = decay * (1 - exp(-step / tau))

    tau is scaled to THIS run (~32 steps/epoch): decay 0.995 with tau 160
    converges in about 5 epochs. YOLOv5's 0.9999 implies tau ~10k steps, which
    exceeds the entire run here and would never converge.

    BatchNorm buffers are COPIED, not averaged: they are already running
    statistics and averaging them biases the variance downwards.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995,
                 tau_steps: int = 160) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = float(decay)
        self.tau = max(1, int(tau_steps))
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        named = dict(self.ema.named_parameters())
        for k, v in self.ema.state_dict().items():
            if k not in msd:
                continue
            if v.dtype.is_floating_point and k in named:
                v.mul_(d).add_(msd[k].detach().to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> dict:
        return {"weights": self.ema.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd: dict) -> None:
        self.ema.load_state_dict(sd["weights"])
        self.updates = int(sd.get("updates", 0))


# ===========================================================================
# channel-summed ImageNet stem transfer
# ===========================================================================

@torch.no_grad()
def imagenet_stem_init(model: nn.Module, arch: str = "mobilenet_v3_small",
                       verbose: bool = True) -> bool:
    """
    Transfer low-level ImageNet filters into the single-channel NIR stem.

        w_nir[c, 0] = sum_k w_rgb[c, k]

    Summation (not averaging) preserves the response magnitude for a grayscale
    input replicated across RGB, which is the standard trick.

    Only the stem is transferred: the deeper NIRDet stages have no shape
    correspondence with any torchvision backbone, so they remain scratch-
    initialised. Edge-seeded filters [0, n_edge_init) are never overwritten.
    """
    try:
        import torchvision.models as tvm
    except Exception as exc:                     # pragma: no cover
        log.warning(f"torchvision unavailable, skipping stem transfer: {exc}")
        return False

    builders = {
        "mobilenet_v3_small": (tvm.mobilenet_v3_small, "MobileNet_V3_Small_Weights"),
        "mobilenet_v2": (tvm.mobilenet_v2, "MobileNet_V2_Weights"),
        "shufflenet_v2_x0_5": (tvm.shufflenet_v2_x0_5, "ShuffleNet_V2_X0_5_Weights"),
    }
    if arch not in builders:
        log.warning(f"unknown imagenet_arch '{arch}', skipping stem transfer")
        return False

    fn, wenum = builders[arch]
    try:
        weights = getattr(tvm, wenum).DEFAULT
        src = fn(weights=weights)
    except Exception as exc:
        log.warning(f"could not fetch {arch} ImageNet weights "
                    f"({type(exc).__name__}: {exc}); training stem from scratch")
        return False

    src_w = None
    for m in src.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            src_w = m.weight.detach().clone()
            break
    if src_w is None:
        log.warning(f"no 3-channel conv found in {arch}")
        return False

    stem = model.backbone.stem
    dst = stem.conv.weight
    kh, kw = dst.shape[-2:]
    if src_w.shape[-2:] != (kh, kw):
        src_w = torch.nn.functional.interpolate(
            src_w, size=(kh, kw), mode="bilinear", align_corners=False)

    summed = src_w.sum(dim=1, keepdim=True)          # (C_src, 1, kh, kw)
    start, end = stem.imagenet_slots
    n = min(end - start, summed.shape[0])
    if n <= 0:
        log.warning("no free stem filter slots for transfer")
        return False

    # Match the scratch filters' scale so BatchNorm starts in a sane regime.
    tgt_std = float(dst[start:start + n].std().clamp(min=1e-8))
    src_std = float(summed[:n].std().clamp(min=1e-8))
    dst[start:start + n] = summed[:n] * (tgt_std / src_std)

    if verbose:
        log.info(f"ImageNet stem transfer: {arch} -> stem filters "
                 f"[{start}:{start + n}) (channel-summed, rescaled). "
                 f"Filters [0:{start}) keep their Sobel/Laplacian seeds; "
                 f"deeper stages remain scratch-initialised.")
    return True


# ===========================================================================
# AMP
# ===========================================================================

def build_amp_context(amp_mode: str, device: torch.device):
    if device.type != "cuda":
        return torch.amp.autocast("cpu", enabled=False), \
            torch.amp.GradScaler("cpu", enabled=False)
    if amp_mode == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16), \
                torch.amp.GradScaler("cuda", enabled=False)
        log.warning("bf16 unsupported on this GPU, using fp16")
        amp_mode = "fp16"
    if amp_mode == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16), \
            torch.amp.GradScaler("cuda", enabled=True, init_scale=2 ** 14)
    return torch.amp.autocast("cuda", enabled=False), \
        torch.amp.GradScaler("cuda", enabled=False)


# ===========================================================================
# checkpoints
# ===========================================================================

def save_checkpoint(path: Path, epoch: int, global_step: int,
                    model: nn.Module, optimizer, scaler,
                    best_map50: float, cfg: Config,
                    ema: Optional[ModelEMA], es_counter: int) -> None:
    """
    Live weights and EMA weights are stored under DISTINCT keys.
    ``deploy_state`` is the copy evaluate.py / export_onnx.py should load.
    """
    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        CKPT_LIVE_KEY: model.state_dict(),
        CKPT_EMA_KEY: (ema.state_dict() if ema is not None else None),
        CKPT_DEPLOY_KEY: (ema.ema.state_dict() if ema is not None
                          else model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_map50": float(best_map50),
        "es_counter": int(es_counter),
        "eaa_epoch": int(getattr(model, "eaa").current_epoch),
        "cfg": cfg.to_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, scaler,
                    ema: Optional[ModelEMA]) -> Tuple[int, int, float, int]:
    log.info(f"resuming from {path}")
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    if CKPT_LIVE_KEY not in ckpt:
        raise RuntimeError(f"{path} has no '{CKPT_LIVE_KEY}' entry; this is a "
                           f"pre-fix checkpoint whose live weights were "
                           f"overwritten by EMA and cannot be recovered")

    incompat = model.load_state_dict(ckpt[CKPT_LIVE_KEY], strict=False)
    if incompat.missing_keys:
        log.warning(f"missing keys: {incompat.missing_keys[:6]}")
    if incompat.unexpected_keys:
        log.warning(f"unexpected keys: {incompat.unexpected_keys[:6]}")

    if ema is not None and ckpt.get(CKPT_EMA_KEY):
        ema.load_state_dict(ckpt[CKPT_EMA_KEY])
    elif ema is not None:
        log.warning("checkpoint has no EMA state; reseeding EMA from live weights")
        ema.ema.load_state_dict(model.state_dict())

    if ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])

    # Explicit belt-and-braces: attention._load_from_state_dict already
    # restored this from the _epoch_buf buffer.
    eaa_epoch = int(ckpt.get("eaa_epoch", ckpt.get("epoch", 0) + 1))
    model.set_eaa_epoch(eaa_epoch)
    log.info(f"EAA epoch restored to {eaa_epoch} "
             f"(edges frozen: {model.eaa.edges_frozen})")

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    return (start_epoch,
            int(ckpt.get("global_step", 0)),
            float(ckpt.get("best_map50", 0.0)),
            int(ckpt.get("es_counter", 0)))


# ===========================================================================
# validation
# ===========================================================================

@torch.no_grad()
def evaluate_map50(model: nn.Module, loader: DataLoader,
                   device: torch.device, amp_ctx, cfg: Config) -> float:
    """
    mAP@0.50 at cfg.eval.eval_score_thresh. The threshold is passed as an
    argument to decode_predictions, so no module state is mutated and there is
    nothing to restore in a finally block.
    """
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    was_training = model.training
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  iou_thresholds=[0.5],
                                  max_detection_thresholds=[1, 10, 300])
    try:
        for images, targets in tqdm(loader, desc="  val mAP50", leave=False):
            images = images.to(device, non_blocking=True)
            h, w = images.shape[-2], images.shape[-1]
            with amp_ctx:
                raw = model.head(model.forward_features(images),
                                 training_mode=False)
            results = model.decode_predictions(
                raw, (h, w), score_thresh=cfg.eval.eval_score_thresh)

            preds, tgts = [], []
            for b, (boxes, scores) in enumerate(results):
                preds.append({
                    "boxes": boxes.detach().float().cpu(),
                    "scores": scores.detach().float().cpu(),
                    "labels": torch.zeros(scores.numel(), dtype=torch.long),
                })
                t = targets[b]
                if t.numel() == 0:
                    tgts.append({"boxes": torch.zeros((0, 4)),
                                 "labels": torch.zeros(0, dtype=torch.long)})
                else:
                    cx, cy = t[:, 1] * w, t[:, 2] * h
                    bw, bh = t[:, 3] * w, t[:, 4] * h
                    tgts.append({
                        "boxes": torch.stack([cx - bw / 2, cy - bh / 2,
                                              cx + bw / 2, cy + bh / 2], 1).float(),
                        "labels": t[:, 0].long(),
                    })
            metric.update(preds, tgts)
        return float(metric.compute()["map_50"])
    finally:
        if was_training:
            model.train()


# ===========================================================================
# one epoch
# ===========================================================================

def train_one_epoch(model, criterion, loader, optimizer, scaler, amp_ctx,
                    device, epoch, cfg: Config, global_step: int,
                    steps_per_epoch: int, ema) -> Tuple[Dict[str, float], int]:
    model.train()
    acc = {"total": 0.0, "cls": 0.0, "reg": 0.0, "iou": 0.0, "n_pos": 0.0}
    n = 0
    skipped = 0

    pbar = tqdm(loader, desc=f"epoch {epoch + 1:03d}", leave=True,
                dynamic_ncols=True)
    for images, targets in pbar:
        set_lr(optimizer, get_lr(global_step, steps_per_epoch, cfg))
        global_step += 1

        images = images.to(device, non_blocking=True)
        gt_batch = [t.to(device, non_blocking=True) for t in targets]

        model.zero_grad(set_to_none=True)
        with amp_ctx:
            outputs = model(images, training_mode=True)

        # Loss in fp32 outside autocast: CIoU and QFL have poor bf16 dynamics.
        losses = criterion([o.float() for o in outputs], gt_batch)
        loss = losses["total"]

        if not torch.isfinite(loss):
            skipped += 1
            log.warning(f"  epoch {epoch + 1}: non-finite loss, batch skipped "
                        f"({skipped} so far)")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.train.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        n += 1
        for k in acc:
            acc[k] += float(losses[k])
        pbar.set_postfix(
            loss=f"{float(losses['total']):.4f}",
            cls=f"{float(losses['cls']):.4f}",
            reg=f"{float(losses['reg']):.4f}",
            iou=f"{float(losses['iou']):.3f}",
            pos=f"{int(losses['n_pos'])}",
            gn=f"{float(gnorm):.2f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    d = max(n, 1)
    return {k: v / d for k, v in acc.items()}, global_step


# ===========================================================================
# main
# ===========================================================================

def main(args: argparse.Namespace) -> int:
    cfg = get_config()

    # ---------------- dataset profile (Upgrade H) ----------------
    # MUST run before validate_config() and before any dataset access:
    # verify_fresh() recomputes the sha256 fingerprint over the label files
    # and hard-errors if they changed since the profile was generated, and
    # apply() overrides ONLY cfg.data.root / n_train / n_val and
    # cfg.model.prior_w / prior_h. This makes a stale prior impossible to
    # silently train against: switch datasets without regenerating the
    # profile and training refuses to start.
    if args.profile:
        from dataset_profiles import DatasetProfile
        profile = DatasetProfile.load(args.profile)
        profile.verify_fresh()   # hard error if labels changed since profile was generated
        profile.apply(cfg)
        if args.data_root:
            # A --data-root pointing away from the profiled dataset would
            # re-introduce the exact silent-stale-prior failure this upgrade
            # exists to prevent: fingerprint verified on A, training on B.
            import os
            a = os.path.normcase(os.fspath(Path(args.data_root).resolve()))
            b = os.path.normcase(os.fspath(Path(cfg.data.root).resolve()))
            if a != b:
                raise RuntimeError(
                    f"--data-root '{args.data_root}' points at a different "
                    f"dataset than the profile '{args.profile}' "
                    f"(root '{cfg.data.root}'). Generate a profile for the "
                    f"new dataset: python dataset_profiles.py --root "
                    f"{args.data_root} --out datasets/<name>.yaml")

    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.data_root:
        cfg.data.root = args.data_root
    if args.img_h:
        cfg.data.img_h = args.img_h
    if args.img_w:
        cfg.data.img_w = args.img_w
    if args.no_imagenet:
        cfg.train.imagenet_stem_init = False

    # Priors are canvas-space: refuse a canvas the profile was not derived
    # at (after --img-h/--img-w overrides have been applied).
    if args.profile:
        profile.check_canvas(cfg)

    if args.smoke_test:
        cfg.train.epochs = 5
        cfg.train.val_interval = 5
        cfg.train.es_enabled = False
        log.info("=== SMOKE TEST: 5 epochs, 10 images, no augmentation ===")
    if args.overfit_test:
        cfg.train.epochs = 300
        cfg.train.warmup_steps = 0
        cfg.train.lr_peak = cfg.train.lr_start = cfg.train.lr_min = 1e-3
        cfg.train.val_interval = 25
        cfg.train.es_enabled = False
        cfg.train.imagenet_stem_init = False
        log.info("=== OVERFIT TEST: 300 epochs, flat LR, 10 images ===")

    validate_config(cfg, verbose=True)
    set_seed(cfg.train.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")
    if device.type == "cuda":
        log.info(f"gpu: {torch.cuda.get_device_name(0)} "
                 f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    tiny = args.smoke_test or args.overfit_test
    train_ds = NIRDetDataset(cfg.data.root, "train", cfg.data.img_h,
                             cfg.data.img_w, augment=not tiny, cfg_aug=cfg.aug)
    val_ds = NIRDetDataset(cfg.data.root, "val", cfg.data.img_h,
                           cfg.data.img_w, augment=False, cfg_aug=cfg.aug)
    if tiny:
        idx = list(range(min(10, len(train_ds))))
        train_ds = Subset(train_ds, idx)
        val_ds = train_ds                      # overfit must fit, not generalise
        cfg.train.batch_size = min(cfg.train.batch_size, len(train_ds))
        log.info(f"subset: {len(train_ds)} train == {len(val_ds)} val")
    else:
        log.info(f"train {len(train_ds)} | val {len(val_ds)}")
        # Cross-check the loaders against the profile counts: the label
        # fingerprint covers label files only, so images added or removed
        # after profiling would otherwise slip through. n_train == 0 means
        # no --profile was passed.
        if args.profile:
            if len(train_ds) != cfg.data.n_train or len(val_ds) != cfg.data.n_val:
                raise RuntimeError(
                    f"profile counts do not match the built loaders: "
                    f"profile n_train={cfg.data.n_train} n_val={cfg.data.n_val} "
                    f"vs loaders train={len(train_ds)} val={len(val_ds)}. "
                    f"The profile counts label files while the loaders count "
                    f"image files, so either images were added/removed after "
                    f"profiling or the dataset has images without label "
                    f"files. Regenerate the profile "
                    f"(python dataset_profiles.py --root "
                    f"{cfg.data.root} --out {args.profile})")

    g = torch.Generator()
    g.manual_seed(cfg.train.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"), drop_last=not tiny,
        persistent_workers=(cfg.data.num_workers > 0),
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"), drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    # ---------------- model ----------------
    model = build_nirdet(cfg).to(device)
    if cfg.train.imagenet_stem_init and not args.resume:
        imagenet_stem_init(model, cfg.train.imagenet_arch)
    log.info(f"\n{model!r}")
    log.info(f"total params: {sum(p.numel() for p in model.parameters()):,} "
             f"(trainable now: "
             f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
             f"; EAA edge kernels are frozen for the first "
             f"{cfg.model.eaa_freeze_epochs} epochs)")

    criterion = NIRDetLoss(
        img_h=cfg.data.img_h, img_w=cfg.data.img_w, strides=cfg.model.strides,
        lambda_cls=cfg.loss.lambda_cls, lambda_reg=cfg.loss.lambda_reg,
        qfl_beta=cfg.loss.qfl_beta, qfl_alpha=cfg.loss.qfl_alpha,
        tal_topk=cfg.loss.tal_topk, tal_alpha=cfg.loss.tal_alpha,
        tal_beta=cfg.loss.tal_beta, ramp_frac=cfg.loss.ramp_frac,
        total_epochs=cfg.train.epochs,
    ).to(device)
    log.info(f"loss grids {criterion.grid_sizes} "
             f"(derived from {cfg.data.img_h}x{cfg.data.img_w} / {cfg.model.strides})")

    groups = build_param_groups(model, cfg.train.weight_decay)
    if cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=cfg.train.lr_start,
                                      betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.SGD(groups, lr=cfg.train.lr_start,
                                    momentum=cfg.train.momentum, nesterov=True)

    amp_ctx, scaler = build_amp_context(cfg.train.amp_mode, device)
    log.info(f"amp: {cfg.train.amp_mode}")

    ema = (ModelEMA(model, cfg.train.ema_decay, cfg.train.ema_tau_steps)
           if cfg.train.use_ema else None)

    start_epoch, global_step, best_map50, es_counter = 0, 0, 0.0, 0
    if args.resume:
        p = Path(args.resume)
        if p.exists():
            start_epoch, global_step, best_map50, es_counter = \
                load_checkpoint(p, model, optimizer, scaler, ema)
            log.info(f"resumed at epoch {start_epoch}, best mAP50 {best_map50:.4f}")
        else:
            log.warning(f"{p} not found; starting from scratch")

    steps_per_epoch = max(1, len(train_loader))

    # Rescale warmup to approximately 3 epochs regardless of dataset size.
    # 300 steps is ~9 epochs at 261 images / batch 8 but under 1 epoch at
    # 10k images. Floor at 100 so tiny datasets still get a real warmup.
    # Dataset size is only known after the loaders are built, so this is the
    # "before the training loop" point. Diagnostic runs keep their hand-set
    # warmup: smoke stays at the config default, overfit keeps 0.
    if not tiny:
        cfg.train.warmup_steps = max(100, min(1500, steps_per_epoch * 3))
    log.info(f"warmup {cfg.train.warmup_steps} steps "
             f"({cfg.train.warmup_steps / steps_per_epoch:.1f} epochs "
             f"at {steps_per_epoch} steps/epoch)")

    hist_path = ckpt_dir / "history.csv"
    epoch_times: List[float] = []

    log.info("=" * 62)
    log.info(f"training epochs {start_epoch + 1} -> {cfg.train.epochs} | "
             f"baseline mAP50 {cfg.eval.baseline_map50:.4f}")
    log.info("=" * 62)

    for epoch in range(start_epoch, cfg.train.epochs):
        model.set_eaa_epoch(epoch)          # idempotent, resume-safe
        criterion.set_epoch(epoch)

        t0 = time.perf_counter()
        losses, global_step = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, amp_ctx,
            device, epoch, cfg, global_step, steps_per_epoch, ema)
        dt = time.perf_counter() - t0
        epoch_times.append(dt)

        lr_now = get_lr(max(global_step - 1, 0), steps_per_epoch, cfg)
        log.info(f"epoch {epoch + 1:03d}/{cfg.train.epochs} | lr {lr_now:.2e} | "
                 f"total {losses['total']:.4f} | cls {losses['cls']:.4f} | "
                 f"reg {losses['reg']:.4f} | iou {losses['iou']:.3f} | "
                 f"pos {losses['n_pos']:.1f} | frozen_eaa "
                 f"{model.eaa.edges_frozen} | {dt:.1f}s")
        if epoch == start_epoch:
            rem = (cfg.train.epochs - start_epoch - 1) * dt / 60.0
            log.info(f"  eta {rem:.0f} min ({rem / 60:.1f} h)")

        save_checkpoint(ckpt_dir / "last.pth", epoch, global_step, model,
                        optimizer, scaler, best_map50, cfg, ema, es_counter)

        is_last = (epoch + 1 == cfg.train.epochs)
        if ((epoch + 1) % max(1, cfg.train.val_interval) == 0) or is_last:
            eval_model = ema.ema if ema is not None else model
            eval_model.to(device)
            map50 = evaluate_map50(eval_model, val_loader, device, amp_ctx, cfg)

            # improvement flag computed BEFORE best is updated (ES fix)
            improved = map50 > best_map50
            log.info(f"  [eval] mAP50 {map50:.4f} | best {best_map50:.4f} | "
                     f"baseline {cfg.eval.baseline_map50:.4f} | "
                     f"{'IMPROVED' if improved else 'no change'}")

            new = not hist_path.exists()
            with open(hist_path, "a", newline="") as f:
                wr = csv.writer(f)
                if new:
                    wr.writerow(["epoch", "lr", "total", "cls", "reg", "iou",
                                 "n_pos", "map50", "best_map50", "seconds"])
                wr.writerow([epoch + 1, f"{lr_now:.6e}",
                             f"{losses['total']:.6f}", f"{losses['cls']:.6f}",
                             f"{losses['reg']:.6f}", f"{losses['iou']:.6f}",
                             f"{losses['n_pos']:.2f}", f"{map50:.6f}",
                             f"{max(best_map50, map50):.6f}", f"{dt:.2f}"])

            if improved:                      # no deadband
                best_map50 = map50
                es_counter = 0
                save_checkpoint(ckpt_dir / "best.pth", epoch, global_step,
                                model, optimizer, scaler, best_map50, cfg,
                                ema, es_counter)
                flag = " (beats baseline)" if map50 >= cfg.eval.baseline_map50 else ""
                log.info(f"  [ckpt] new best {best_map50:.4f} saved{flag}")
            else:
                es_counter += 1
                if cfg.train.es_enabled:
                    log.info(f"  [es] no improvement for {es_counter} eval(s)")

            if cfg.train.es_enabled and es_counter >= cfg.train.es_patience:
                log.info(f"  [es] stopping at epoch {epoch + 1}")
                break

    avg = sum(epoch_times) / max(len(epoch_times), 1)
    log.info("=" * 62)
    log.info("training complete")
    log.info(f"  best mAP50    : {best_map50:.4f}")
    log.info(f"  baseline      : {cfg.eval.baseline_map50:.4f}")
    log.info(f"  delta         : {best_map50 - cfg.eval.baseline_map50:+.4f}")
    log.info(f"  avg epoch     : {avg:.1f}s")
    log.info(f"  best ckpt     : {ckpt_dir / 'best.pth'}")
    log.info(f"  history       : {hist_path}")
    log.info("  NOTE: a single val mAP50 on ~160 images has a standard error "
             "of roughly +/-0.02-0.04. Run evaluate.py for bootstrap CIs "
             "before reporting any delta against the baseline.")
    log.info("=" * 62)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("NIRDet-Lite training")
    p.add_argument("--resume", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--img-h", type=int, default=None)
    p.add_argument("--img-w", type=int, default=None)
    p.add_argument("--no-imagenet", action="store_true",
                   help="disable channel-summed ImageNet stem transfer")
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic algorithms (slower, exactly reproducible)")
    p.add_argument("--profile", default=None,
                   help="dataset profile YAML (datasets/*.yaml). Verifies the "
                        "label fingerprint is still fresh, then applies "
                        "root/n_train/n_val/prior_w/prior_h to the config")
    p.add_argument("--smoke-test", action="store_true",
                   help="5 epochs on 10 images: verify no NaN, no OOM")
    p.add_argument("--overfit-test", action="store_true",
                   help="300 epochs, flat LR, 10 images: loss must approach 0")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
