"""
train.py — NIRDet-Lite training loop
=====================================
    python train.py --profile datasets/miniNIRPed_261.yaml
    python train.py --overfit-test
    python train.py --p5-ablate --profile datasets/miniNIRPed_261.yaml
    python train.py --resume checkpoints/last.pth

ORDER OF OPERATIONS AT STARTUP (all of it matters)
--------------------------------------------------
  1. --p5-ablate rewrites cfg.model.strides BEFORE the model is built, so the
     head, the loss geometry and the feature slicing all agree.
  2. --profile applies the dataset profile BEFORE the model is built, because
     prior_w/prior_h feed head.size_pred.bias and that init is what lets TAL
     find positives at epoch 0.
  3. validate_config(cfg).
  4. Build the model. ImageNet stem transfer, if enabled, writes ONLY slots
     [n_edge_init, out_ch) so the Sobel/Laplacian seeds survive.
  5. Pull the FIRST REAL BATCH and call model.calibrate_eaa(images). This
     happens before any optimiser step and before the EAA freeze expires. The
     measured bias is written back into cfg.model.eaa_proj_bias so it lands in
     every checkpoint.
  6. Only then does epoch 0 begin.

Step 5 cannot be reordered. The EAA projection bias is measured from the
pooled edge statistics of the real preprocessing chain (flat-field + CLAHE +
letterbox), so it must be measured on real data; and it must be measured
before the edge kernels start moving, or the statistic describes a filter bank
that no longer exists.

LAYER-WISE LR SCALING
---------------------
STResNet/STYOLO report this single change taking STYOLO-Nano from 21.32 to
26.25 mAP on MS-COCO with no architectural change. backbone 0.2x (its
representations are pretrained or edge-seeded and should be preserved), neck
0.8x, head 1.0x (randomly initialised, must adapt fastest). EAA rides with the
backbone: its kernels are Sobel-seeded, which is the same argument.

Weight decay is applied only to parameters with ndim > 1. That covers conv and
linear weights and excludes every bias and every BatchNorm scale in one test.
reg_level_scale is gone from head.py, so it is not in any no-decay list.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (CKPT_DEPLOY_KEY, CKPT_EMA_KEY, CKPT_LIVE_KEY, Config,
                    ema_tau_for, get_config, validate_config)
from dataset import build_dataloader
from losses import ColdStartError, NIRDetLoss
from model import NIRDet, build_nirdet

# ---- checkpoint keys -------------------------------------------------------
# CKPT_LIVE_KEY   : the raw training weights. Resume continues from these.
# CKPT_EMA_KEY    : the EMA shadow, needed to resume the average itself.
# CKPT_DEPLOY_KEY : the weights to deploy and to report numbers from. This is
#                   the EMA shadow when EMA is enabled and the live weights
#                   otherwise, so evaluate.py / export_onnx.py never have to
#                   know which mode the run used.
# (All three are defined once in config.py — the single source of truth.)


# ===========================================================================
# determinism
# ===========================================================================

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed python / numpy / torch / CUDA.

    NOTE on reproducibility: cudnn.benchmark=True (the default here for
    throughput) selects convolution algorithms by TIMING, which is
    nondeterministic run-to-run even under a fixed seed. Fully reproducible
    runs need --deterministic, which disables benchmark and switches torch
    to deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


# ===========================================================================
# parameter groups
# ===========================================================================

def _section_of(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("eaa."):
        return "eaa"
    if name.startswith("neck."):
        return "neck"
    if name.startswith("head."):
        return "head"
    return "head"        # anything new defaults to the fastest group


def build_param_groups(model: nn.Module, cfg: Config) -> List[dict]:
    """
    Eight groups: {backbone, eaa, neck, head} x {decay, no_decay}.

    Each group's ``lr`` already carries the layer-wise scale, and the LR
    schedule multiplies it — so the ratio between sections is preserved
    through warmup and cosine decay instead of collapsing at the peak.
    """
    scale = {
        "backbone": float(cfg.train.lr_scale_backbone),
        # EAA is part of the edge-seeded front end. Its kernels are frozen for
        # freeze_epochs anyway; the scale governs what happens after.
        "eaa": float(cfg.train.lr_scale_backbone),
        "neck": float(cfg.train.lr_scale_neck),
        "head": float(cfg.train.lr_scale_head),
    }
    buckets: Dict[Tuple[str, bool], List[nn.Parameter]] = {
        (s, d): [] for s in scale for d in (True, False)
    }

    for name, p in model.named_parameters():
        # EAA edge kernels during the freeze window are registered here too:
        # they are re-enabled by eaa._update_grad_state(), and because the
        # optimiser groups are built once, they must sit in their bucket
        # regardless of the current requires_grad flag.
        sec = _section_of(name)
        if "eaa.proj.weight" in name:
            decay = False   # override: proj.weight carries the calibrated gain k;
                            # weight decay would pull k back toward 1/N and re-inert
                            # the EAA gate (see attention.py calibrate_bias docstring)
        else:
            decay = p.ndim > 1        # conv/linear weights only
        buckets[(sec, decay)].append(p)

    groups: List[dict] = []
    for (sec, decay), params in buckets.items():
        if not params:
            continue
        groups.append({
            "params": params,
            "lr": float(cfg.train.lr_peak) * scale[sec],
            "base_lr": float(cfg.train.lr_peak) * scale[sec],
            "weight_decay": float(cfg.train.weight_decay) if decay else 0.0,
            "name": f"{sec}_{'decay' if decay else 'nodecay'}",
            "lr_scale": scale[sec],
        })
    groups.sort(key=lambda g: g["name"])
    return groups


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    groups = build_param_groups(model, cfg)
    kind = str(cfg.train.optimizer).lower()
    if kind == "adamw":
        opt = torch.optim.AdamW(groups, lr=float(cfg.train.lr_peak),
                                betas=(0.9, 0.999), eps=1e-8)
    elif kind == "sgd":
        opt = torch.optim.SGD(groups, lr=float(cfg.train.lr_peak),
                              momentum=float(cfg.train.momentum),
                              nesterov=True)
    else:
        raise ValueError(f"unknown optimizer '{cfg.train.optimizer}'")
    print("[optim] layer-wise LR groups:")
    for g in opt.param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<18} lr {g['lr']:.3e} "
              f"(x{g['lr_scale']:.2f})  wd {g['weight_decay']:.1e}  "
              f"params {n:,}")
    return opt


# ===========================================================================
# LR schedule
# ===========================================================================

class LRSchedule:
    """
    Linear warmup from lr_start to the group's base LR, then cosine to lr_min.

    ``flat`` (used by --overfit-test) holds every group at its base LR: an
    overfit test is asking "can this architecture fit 10 images", and a warmup
    plus a cosine decay makes a negative answer ambiguous.
    """

    def __init__(self, cfg: Config, steps_per_epoch: int,
                 flat: bool = False) -> None:
        self.total = max(1, int(cfg.train.epochs) * max(1, steps_per_epoch))
        self.lr_start = float(cfg.train.lr_start)
        self.lr_peak = float(cfg.train.lr_peak)
        self.lr_min = float(cfg.train.lr_min)
        self.flat = bool(flat)

        # F14: warmup is DERIVED from warmup_epochs_frac — a fraction of the
        # total run, so it is dataset-size-invariant. Cap at 10% of the run so
        # a small dataset does not spend a disproportionate time below peak
        # LR, and a large one still gets a real warmup.
        derived = round(float(cfg.train.warmup_epochs_frac) *
                        float(cfg.train.epochs) * max(1, steps_per_epoch))
        self.warmup = 0 if flat else min(
            derived,
            int(float(cfg.train.epochs) * 0.1 * max(1, steps_per_epoch)),
        )
        self.warmup = max(0, self.warmup)

    def factor(self, step: int) -> float:
        """Multiplier applied to each group's base_lr."""
        if self.flat:
            return 1.0
        if self.warmup > 0 and step < self.warmup:
            t = step / float(self.warmup)
            lr = self.lr_start + t * (self.lr_peak - self.lr_start)
            return lr / self.lr_peak
        denom = max(1, self.total - self.warmup)
        t = min(1.0, max(0, step - self.warmup) / denom)
        lr = self.lr_min + 0.5 * (self.lr_peak - self.lr_min) * \
            (1.0 + math.cos(math.pi * t))
        return lr / self.lr_peak

    def apply(self, opt: torch.optim.Optimizer, step: int) -> float:
        f = self.factor(step)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        return float(opt.param_groups[0]["lr"])


# ===========================================================================
# EMA
# ===========================================================================

class ModelEMA:
    """
    Exponential moving average with a warmup ramp on the decay.

        d_eff(step) = decay * (1 - exp(-step / tau))

    tau comes from ema_tau_for(steps_per_epoch), NOT from a constant. A
    hardcoded tau is a hidden dataset dependence: 160 steps is about five
    epochs at 261 images / batch 8 and under one epoch on full NIRPed.
    """

    def __init__(self, model: nn.Module, decay: float, tau_steps: int) -> None:
        self.decay = float(decay)
        self.tau = max(1, int(tau_steps))
        self.step = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    def effective_decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.step / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = self.effective_decay()
        msd = model.state_dict()
        for k, v in self.shadow.items():
            src = msd[k]
            if not torch.is_floating_point(src):
                v.copy_(src.float())          # counters/buffers: track exactly
            else:
                v.mul_(d).add_(src.detach().float(), alpha=1.0 - d)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "tau": self.tau, "step": self.step,
                "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd.get("decay", self.decay))
        self.tau = int(sd.get("tau", self.tau))
        self.step = int(sd.get("step", 0))
        shadow = sd.get("shadow", {})
        for k in self.shadow:
            if k in shadow:
                self.shadow[k].copy_(shadow[k].float())

    def deploy_state_dict(self, ref: nn.Module) -> Dict[str, torch.Tensor]:
        """Shadow cast back to the reference module's dtypes."""
        ref_sd = ref.state_dict()
        return {k: self.shadow[k].to(ref_sd[k].dtype).clone()
                for k in ref_sd}


# ===========================================================================
# ImageNet stem transfer
# ===========================================================================

def imagenet_stem_init(model: NIRDet, cfg: Config) -> bool:
    """
    Channel-sum the MobileNetV3-small first conv into the FREE stem slots.

    NIRStem seeds filters [0, n_edge_init) with unit-L2 Sobel / Laplacian
    kernels, which is the correct inductive bias for an edge-rich,
    texture-poor 850 nm reflective modality. Those slots are NOT touched.
    Only [n_edge_init, out_ch) is overwritten, which is what
    stem.imagenet_slots reports.

    The pretrained kernel is summed over its RGB axis (the standard
    single-channel transfer) and then rescaled to the std of the slots it
    replaces, so the transferred filters enter at the same magnitude as the
    Kaiming init they displace and do not dominate the stem's BatchNorm.
    """
    if not cfg.train.imagenet_stem_init:
        return False
    try:
        import torchvision.models as tvm
    except ImportError:
        print("[stem] torchvision unavailable; skipping ImageNet transfer")
        return False

    arch = str(cfg.train.imagenet_arch)
    try:
        fn = getattr(tvm, arch)
        try:
            net = fn(weights="DEFAULT")
        except TypeError:
            net = fn(pretrained=True)
    except Exception as exc:
        print(f"[stem] could not load {arch} weights ({exc}); "
              f"skipping ImageNet transfer")
        return False

    src = None
    for m in net.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            src = m.weight.detach().float()
            break
    if src is None:
        print(f"[stem] no 3-channel first conv in {arch}; skipping")
        return False

    stem = model.backbone.stem
    lo, hi = stem.imagenet_slots
    n_free = hi - lo
    if n_free <= 0:
        print("[stem] no free slots; skipping ImageNet transfer")
        return False

    summed = src.sum(dim=1, keepdim=True)              # (M, 1, kh, kw)
    if summed.shape[-2:] != stem.conv.weight.shape[-2:]:
        summed = torch.nn.functional.interpolate(
            summed, size=stem.conv.weight.shape[-2:], mode="bilinear",
            align_corners=False)
    take = min(n_free, summed.shape[0])

    with torch.no_grad():
        target = stem.conv.weight[lo:lo + take]
        ref_std = float(target.std()) if take > 1 else float(target.abs().mean())
        src_std = float(summed[:take].std()) + 1e-12
        stem.conv.weight[lo:lo + take] = \
            summed[:take] * (ref_std / src_std)

    print(f"[stem] ImageNet transfer from {arch}: {take} filter(s) into slots "
          f"[{lo}, {lo + take}); Sobel/Laplacian slots [0, {lo}) preserved")
    return True


# ===========================================================================
# checkpoint
# ===========================================================================

def save_checkpoint(path: str, model: NIRDet, ema: Optional[ModelEMA],
                    opt: torch.optim.Optimizer, cfg: Config, epoch: int,
                    global_step: int, best_map50: float,
                    history: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    live = model.state_dict()
    deploy = ema.deploy_state_dict(model) if ema is not None else \
        {k: v.detach().clone() for k, v in live.items()}
    ckpt = {
        CKPT_LIVE_KEY: live,
        CKPT_EMA_KEY: ema.state_dict() if ema is not None else None,
        CKPT_DEPLOY_KEY: deploy,
        "optimizer": opt.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_map50": float(best_map50),
        "history": history,
        "cfg": cfg.to_dict(),
        # evaluate.py, export_onnx.py and test_decode_contract.py all assert
        # this against the live config. A geometry change cannot silently
        # invalidate a trained checkpoint.
        "deploy_contract_hash": cfg.deploy_contract()["hash"],
        "deploy_contract": cfg.deploy_contract(),
        # F30: dump the full config as JSON beside every checkpoint so a
        # future reader can reconstruct the exact training conditions.
        "cfg_json": json.dumps(cfg.to_dict(), indent=2),
        "eaa_proj_bias": cfg.model.eaa_proj_bias,
        "strides": list(cfg.model.strides),
    }
    # Atomic: torch.save straight onto the target path means a crash or
    # SIGINT during serialisation destroys the only resumable checkpoint.
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def save_history(ckpt_dir: str, history: list) -> None:
    """Atomic history.json write — a crash, ColdStartError return or Ctrl-C
    mid-run must not leave evaluate.py without its per-level n_pos record."""
    path = os.path.join(ckpt_dir, "history.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    os.replace(tmp, path)


# ===========================================================================
# one epoch
# ===========================================================================

# ===========================================================================
# live-output helpers  (#25, #82, session header/footer)
# ===========================================================================

_BAR_W = 20


def _sci(x: float) -> str:
    """LR in scientific notation, 2 sig figs: 0.0012 -> '1.2e-3'."""
    s = f"{x:.1e}"
    m, e = s.split("e")
    return f"{m}e{int(e)}"


def _dur(seconds: float) -> str:
    """Compact duration: 272 -> '4m32s', 9660 -> '2h41m', 8 -> '8s'."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _bar(frac: float, w: int = _BAR_W) -> str:
    frac = max(0.0, min(1.0, frac))
    fill = int(round(frac * w))
    return "▓" * fill + "░" * (w - fill)


class _Tee:
    """Mirror everything written to stdout into a log file."""

    def __init__(self, log_path: str) -> None:
        self._term = sys.stdout
        self._log = open(log_path, "a", buffering=1, encoding="utf-8")

    def write(self, data: str) -> None:
        self._term.write(data)
        # don't write bare \r frames to the log — convert to newlines
        self._log.write(data.replace("\r", "\n") if data.endswith("\r") else data)

    def flush(self) -> None:
        self._term.flush()
        self._log.flush()

    def close(self) -> None:
        try:
            self._log.close()
        except Exception:
            pass


def write_curves(history: list, ckpt_dir: str, best_epoch: Optional[int]) -> None:
    """Write a 2-panel loss+mAP50 PNG after every epoch. (#82)

    Guarded by ImportError so matplotlib absence never crashes training.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history:
        return
    ep = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    for k, lbl in (("total", "total"), ("cls", "cls"),
                   ("reg", "reg"), ("iou", "iou")):
        vals = [h.get(k, float("nan")) for h in history]
        ax0.plot(ep, vals, label=lbl)
    ax0.set_ylabel("train loss")
    ax0.legend(loc="upper right")
    ax0.grid(alpha=0.3)
    map_vals = [h.get("val_map50", float("nan")) for h in history]
    ax1.plot(ep, map_vals, color="tab:green", label="val mAP50")
    ax1.set_ylabel("val mAP50")
    ax1.set_xlabel("epoch")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    if best_epoch is not None:
        for ax in (ax0, ax1):
            ax.axvline(best_epoch, ls="--", color="crimson", alpha=0.7,
                       label=f"best e{best_epoch}")
    fig.tight_layout()
    out = os.path.join(ckpt_dir, "curves.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def print_session_header(run_id: str, device: str, epochs: int,
                         canvas_h: int, canvas_w: int, n_cells: int,
                         torch_ver: str, contract_hash: str) -> None:
    rows = [
        f"NIRDet-Lite  |  run: {run_id}  |  {device}",
        f"epochs: {epochs}  |  canvas: {canvas_h}x{canvas_w}"
        f"  |  cells: {n_cells}",
        f"torch: {torch_ver}  |  contract: {contract_hash[:8]}",
    ]
    w = max(len(r) for r in rows) + 2
    print("╔" + "═" * (w + 1) + "╗")
    for r in rows:
        print(f"║ {r.ljust(w)}║")
    print("╚" + "═" * (w + 1) + "╝")


def print_session_footer(best_map50: float, best_epoch: int,
                         best_path: str) -> None:
    line = "═" * 60
    print(line)
    print(f"DONE  best mAP50 {best_map50:.4f} @ epoch {best_epoch}")
    print(f"to evaluate:  python evaluate.py --checkpoint {best_path}")
    print(line)



def train_one_epoch(model: NIRDet, loader: DataLoader,
                    criterion: NIRDetLoss, opt: torch.optim.Optimizer,
                    sched: LRSchedule, cfg: Config, device: torch.device,
                    epoch: int, global_step: int,
                    ema: Optional[ModelEMA],
                    amp_dtype: Optional[torch.dtype],
                    scaler: Optional[torch.amp.GradScaler],
                    log_every: int = 20) -> Tuple[Dict[str, float], int]:
    model.train()
    n_levels = len(cfg.model.strides)
    acc = {k: 0.0 for k in ("total", "cls", "reg", "iou", "n_pos")}
    for lvl in range(n_levels):
        acc[f"n_pos_l{lvl}"] = 0.0
    n_batches = 0
    n_nonfinite = 0
    t0 = time.time()
    lr_now = 0.0   # first real value assigned by sched.apply() on first batch

    for bi, (imgs, targets, _meta) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in targets]

        lr_now = sched.apply(opt, global_step)

        opt.zero_grad(set_to_none=True)
        use_amp = amp_dtype is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            preds = model(imgs, training_mode=True)
        # The loss runs OUTSIDE autocast: losses.py already upcasts the packed
        # tensor to float32, and the CIoU/alignment arithmetic has enough
        # dynamic range in it that bf16 there buys nothing.
        out = criterion(preds, targets)
        loss = out["total"]

        # A single non-finite loss propagated through backward() poisons
        # every weight; the run continues producing NaN forever.
        if not torch.isfinite(loss):
            n_nonfinite += 1
            print(f"  e{epoch:03d} b{bi:04d}: non-finite loss, batch skipped "
                  f"({n_nonfinite} so far this epoch)")
            opt.zero_grad(set_to_none=True)
            continue

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(cfg.train.grad_clip_norm))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(cfg.train.grad_clip_norm))
            opt.step()

        if ema is not None:
            ema.update(model)

        global_step += 1
        n_batches += 1
        for k in acc:
            if k in out:
                acc[k] += float(out[k].detach())

        if log_every and (bi % log_every == 0 or bi == len(loader) - 1):
            done = bi + 1
            total = len(loader)
            elapsed = time.time() - t0
            rate = elapsed / done
            eta = _dur(rate * (total - done))
            print(f"\r[{epoch + 1:03d}/{cfg.train.epochs:03d}]"
                  f" {_bar(done / total)}"
                  f" {done:>{len(str(total))}d}/{total}"
                  f"  loss {float(out['total']):.4f}"
                  f"  cls {float(out['cls']):.4f}"
                  f"  reg {float(out['reg']):.4f}"
                  f"  iou {float(out['iou']):.4f}"
                  f"  n_pos {float(out['n_pos']):.0f}"
                  f"  lr {_sci(lr_now)}"
                  f"  ETA {eta}",
                  end="", flush=True)

    print()   # close the \r progress line

    if n_batches == 0:
        if n_nonfinite == 0:
            raise RuntimeError("train loader produced zero batches")
        raise RuntimeError(
            f"no usable batches: all {n_nonfinite} batches were non-finite")
    # F66: per-epoch norm-floor abort. The counter is incremented in
    # NIRDetLoss.forward; consume and reset it here, after the batch loop.
    if hasattr(criterion, "_norm_floor_count"):
        nf = criterion._norm_floor_count
        criterion._norm_floor_count = 0
        if nf > int(cfg.loss.norm_floor_abort_thresh):
            raise RuntimeError(
                f"epoch {epoch}: norm was floored (sum < 1) on {nf} of "
                f"{n_batches} batches (>{cfg.loss.norm_floor_abort_thresh}). "
                f"The model is not assigning any positives. Check the dataset "
                f"profile and that head.size_pred.bias was initialised correctly.")
        if nf > 0:
            print(f"  [loss] norm floored on {nf}/{n_batches} batches this epoch")
    stats = {k: v / n_batches for k, v in acc.items()}
    stats["lr"] = lr_now
    stats["secs"] = time.time() - t0
    stats["n_nonfinite"] = n_nonfinite
    return stats, global_step


# ===========================================================================
# main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train NIRDet-Lite.")
    ap.add_argument("--profile", default=None,
                    help="dataset profile YAML (dataset_profiles.py). Fills "
                         "n_train/n_val/n_test/n_train_boxes, prior_w/prior_h "
                         "and deploy_score_thresh, and verifies the canvas "
                         "fingerprint.")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overfit-test", action="store_true",
                    help="10 training images, flat LR, eval every epoch, "
                         "max 100 epochs. An architecture smoke test.")
    ap.add_argument("--p5-ablate", action="store_true",
                    help="strides=(8,16): drop the stride-32 level. The "
                         "one-line P5 utilisation experiment.")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--allow-no-profile", action="store_true",
                    help="override the fail-safe that refuses >5-epoch runs "
                         "without a dataset profile")
    ap.add_argument("--deterministic", action="store_true",
                    help="cudnn.benchmark off + deterministic algorithms. "
                         "Without this, benchmark selects conv algorithms "
                         "by timing and runs are not bit-reproducible even "
                         "under a fixed seed.")
    ap.add_argument("--log-every", type=int, default=20)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = get_config()

    # ---- 1. dataset profile FIRST. The profile's canvas fingerprint covers
    # ITS OWN stride list, so check_canvas must see the unablated strides.
    # prior_w/prior_h are canvas-space box medians and are stride-independent,
    # so applying the profile before the ablation is sound — after is not
    # (F01: --p5-ablate used to die with CanvasMismatchError on every
    # profile, because test_dataset_profiles.t1 deliberately pins that a
    # dropped stride changes the fingerprint).
    profile = None
    profile_path = args.profile
    if profile_path:
        from dataset_profiles import DatasetProfile
        profile = DatasetProfile.load(profile_path)
        profile.apply(cfg)          # raises CanvasMismatchError on mismatch
    else:
        print("\n" + "!" * 74)
        print("  NO --profile GIVEN")
        print("!" * 74)
        print(f"  prior_w = {cfg.model.prior_w}   prior_h = {cfg.model.prior_h}")
        print("  These are CONFIG FALLBACKS. A prior from a different dataset")
        print("  produces a plausible loss curve with a useless regression branch.")
        print("  Generate one:")
        print("    python dataset_profiles.py --root <path> --out datasets/<n>.yaml")
        print("!" * 74 + "\n")
        if int(cfg.train.epochs) > 5 and not args.allow_no_profile \
                and not args.overfit_test:
            raise SystemExit(
                f"refusing to start a {cfg.train.epochs}-epoch run without a "
                f"dataset profile. Pass --profile <yaml>, or "
                f"--allow-no-profile to override.")

    # ---- 2. strides: still before the model, the loss geometry and the
    # deploy-contract hash are built from them.
    if args.p5_ablate:
        cfg.model.strides = (8, 16)
        print("[ablate] strides -> (8, 16); the stride-32 level is removed "
              "from the head and the loss geometry. Compare mAP50 and the "
              "n_pos_l2 history of the baseline run.")

    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.lr is not None:
        cfg.train.lr_peak = float(args.lr)
    if args.workers is not None:
        cfg.data.num_workers = int(args.workers)
    if args.no_ema:
        cfg.train.use_ema = False
    if args.overfit_test:
        # Set BEFORE the no-profile epoch guard so --overfit-test never
        # triggers it. An overfit smoke test does not need a dataset profile.
        cfg.train.epochs = 100
        cfg.train.val_interval = 1
        cfg.train.es_enabled = False

    # ---- 3. validate ----
    validate_config(cfg)

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(cfg.train.seed), deterministic=bool(args.deterministic))
    print(f"[env] device {device}  torch {torch.__version__}"
          + ("  [deterministic]" if args.deterministic else ""))

    # ---- data ----
    train_loader, train_ds = build_dataloader(
        cfg, "train", augment=not args.overfit_test,
        limit=10 if args.overfit_test else None,
        num_workers=0 if args.overfit_test else None)
    print(f"[data] train {len(train_ds)} images / {train_ds.n_boxes} boxes  "
          f"copy_paste_p {train_ds.copy_paste_p:.3f}")
    if profile is not None and not args.overfit_test and \
            len(train_ds) != int(cfg.data.n_train):
        raise RuntimeError(
            f"profile n_train={cfg.data.n_train} but the train loader built "
            f"{len(train_ds)} images. Images were added or removed after "
            f"profiling. Regenerate the profile:\n"
            f"  python dataset_profiles.py --root {cfg.data.root!r} "
            f"--img-h {cfg.data.img_h} --img-w {cfg.data.img_w} "
            f"--out <profile_path>")

    val_loader = None
    if not args.no_eval:
        try:
            val_loader, val_ds = build_dataloader(
                cfg, cfg.eval.select_split, batch_size=1, shuffle=False,
                augment=False, num_workers=0)
            print(f"[data] {cfg.eval.select_split} {len(val_ds)} images "
                  f"(checkpoint selection split)")
        except FileNotFoundError as exc:
            print(f"[data] no selection split: {exc}")

    steps_per_epoch = max(1, len(train_loader))

    # ---- model ----
    model = build_nirdet(cfg).to(device)
    imagenet_stem_init(model, cfg)
    pb = model.param_breakdown()
    print(f"[model] params {pb['total']:,} "
          f"(backbone {pb['backbone']:,} eaa {pb['eaa']:,} "
          f"neck {pb['neck']:,} head {pb['head']:,})")
    if model.count_grouped_convs() != 0:
        raise RuntimeError("grouped/depthwise convolution present; "
                           "backbone.py forbids them")

    # ---- 5. EAA CALIBRATION on a dedicated unaugmented loader ----
    # Before epoch 0, before any optimiser step, before the edge kernels
    # unfreeze. The bias depends on the sensor, the 850 nm beam profile, the
    # flat-field map and the CLAHE settings, so it can only be measured here.
    # A dedicated loader, NOT iter(train_loader): (a) the training loader is
    # augmented, so the measured edge statistics would include cutout, blur
    # and copy-paste noise the Pi never sees, and (b) pulling a batch from a
    # persistent-worker loader spawns the full worker pool and abandons it —
    # a documented source of hangs on Windows.
    # F07: measure cfg.train.eaa_calib_frames (>= 32) frames, not one batch.
    # shuffle=True because a fixed-mount NIR capture stores temporally
    # consecutive frames: the first N sorted files are one scene under one
    # illuminator geometry — the worst estimator for a statistic that is
    # frozen into proj.bias AND the INT8 activation ranges.
    calib_target = max(1, int(getattr(cfg.train, "eaa_calib_frames", 32)))
    calib_bs = max(1, min(8, int(cfg.train.batch_size)))
    calib_loader, _calib_ds = build_dataloader(
        cfg, "train", batch_size=calib_bs, shuffle=True,
        augment=False, num_workers=0,
        limit=10 if args.overfit_test else None)
    _calib_chunks: List[torch.Tensor] = []
    _calib_have = 0
    for _c_imgs, _c_tgt, _c_meta in calib_loader:
        _calib_chunks.append(_c_imgs)
        _calib_have += int(_c_imgs.shape[0])
        if _calib_have >= calib_target:
            break
    if not _calib_chunks:
        raise RuntimeError("EAA calibration loader produced zero batches")
    calib_imgs = torch.cat(_calib_chunks, 0)[:calib_target].to(device)
    del _calib_chunks
    print(f"[eaa] calibrating on {int(calib_imgs.shape[0])} unaugmented "
          f"frame(s) (cfg.train.eaa_calib_frames={calib_target})")
    model.eval()
    # Skip calibration on --resume: the checkpoint's state_dict (loaded below)
    # already carries calibrated proj.weight / proj.bias, and running
    # calibrate_eaa here mutates both before they are overwritten by
    # load_state_dict — wasted work and a misleading log line. The checkpoint
    # also carries eaa_proj_bias in its cfg dict, so cfg stays consistent.
    if not args.resume:
        bias = model.calibrate_eaa(calib_imgs, force=False, verbose=True)
        if bias is not None:
            cfg.model.eaa_proj_bias = float(bias)
            print(f"[eaa] cfg.model.eaa_proj_bias <- {cfg.model.eaa_proj_bias:+.5f} "
                  f"(goes into every checkpoint)")
        elif cfg.model.eaa_proj_bias is None:
            cfg.model.eaa_proj_bias = float(model.eaa.proj_bias_value)
        if not model.eaa.is_calibrated:
            raise RuntimeError(
                "EAA calibration did not run. Without it the spatial gate is a "
                "near-uniform rescale that the following BatchNorm absorbs "
                "entirely, and the whole module contributes nothing.")
    else:
        print("[eaa] --resume: skipping calibration; checkpoint carries "
              "calibrated proj weights (eaa_proj_bias will be read from ck below)")
    del calib_imgs, calib_loader, _calib_ds
    model.train()

    # ---- loss / optim ----
    criterion = NIRDetLoss(
        img_h=cfg.data.img_h, img_w=cfg.data.img_w,
        strides=cfg.model.strides,
        lambda_cls=cfg.loss.lambda_cls, lambda_reg=cfg.loss.lambda_reg,
        qfl_beta=cfg.loss.qfl_beta, qfl_alpha=cfg.loss.qfl_alpha,
        tal_topk=cfg.loss.tal_topk, tal_alpha=cfg.loss.tal_alpha,
        tal_beta=cfg.loss.tal_beta, ramp_frac=cfg.loss.ramp_frac,
        tal_min_pos_target=float(getattr(cfg.loss, "tal_min_pos_target", 0.10)),
        total_epochs=cfg.train.epochs, assert_cold_start=True,
    ).to(device)

    opt = build_optimizer(model, cfg)
    sched = LRSchedule(cfg, steps_per_epoch, flat=bool(args.overfit_test))
    print(f"[sched] steps/epoch {steps_per_epoch}  total "
          f"{sched.total}  warmup {sched.warmup}"
          f"{'  (FLAT, overfit test)' if sched.flat else ''}")
    print(f"[sched] warmup rescaled from cfg target "
          f"warmup_epochs_frac={cfg.train.warmup_epochs_frac:.3f} "
          f"-> {sched.warmup} steps = min(derived, 10% of run).")

    ema = None
    if cfg.train.use_ema:
        tau = ema_tau_for(steps_per_epoch)
        ema = ModelEMA(model, float(cfg.train.ema_decay), tau)
        print(f"[ema] decay {cfg.train.ema_decay} tau {tau} steps "
              f"(= ema_tau_for({steps_per_epoch}), ~5 epochs)")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": None}[str(cfg.train.amp_mode)]
    scaler = torch.amp.GradScaler(
        device.type, enabled=(amp_dtype is torch.float16 and device.type == "cuda"))
    if amp_dtype is torch.bfloat16:
        print("[amp] bf16. head._grid() pins cell indices to float32, and at "
              "512 canvas width the stride-8 grid is 64 columns, well inside "
              "the bf16 exact-integer limit of 256.")

    # ---- resume ----
    start_epoch = 0
    global_step = 0
    best_map50 = -1.0
    history: List[dict] = []
    if args.resume:
        # F30: prefer weights_only=True; fall back loudly for older
        # checkpoints that embed non-tensor objects (history list, cfg dict).
        try:
            ck = torch.load(args.resume, map_location=device, weights_only=True)
        except Exception:
            print(f"[resume] WARNING: weights_only=True failed for {args.resume}; "
                  f"falling back to weights_only=False. This checkpoint may contain "
                  f"arbitrary Python objects. Verify it came from this codebase.")
            ck = torch.load(args.resume, map_location=device, weights_only=False)
        got = str(ck.get("deploy_contract_hash", ""))
        want = cfg.deploy_contract()["hash"]
        if got and got != want:
            raise RuntimeError(
                f"checkpoint deploy_contract_hash {got} != current {want}: "
                f"the checkpoint was trained with different geometry "
                f"(canvas, strides or decode constants). Resuming would "
                f"mis-decode every box.")
        model.load_state_dict(ck[CKPT_LIVE_KEY])
        if ema is not None and ck.get(CKPT_EMA_KEY):
            ema.load_state_dict(ck[CKPT_EMA_KEY])
        elif ema is not None:
            ema.shadow = {k: v.detach().clone().float()
                          for k, v in model.state_dict().items()}
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        global_step = int(ck.get("global_step", 0))
        best_map50 = float(ck.get("best_map50", -1.0))
        history = list(ck.get("history", []))
        if ck.get("eaa_proj_bias") is not None:
            cfg.model.eaa_proj_bias = float(ck["eaa_proj_bias"])
        # attention._load_from_state_dict restored _current_epoch from
        # _epoch_buf, so the Sobel kernels are not re-frozen for another
        # freeze_epochs. Reassert it anyway: cheap and explicit.
        model.set_eaa_epoch(start_epoch)
        print(f"[resume] {args.resume} -> epoch {start_epoch} "
              f"best_map50 {best_map50:.4f} "
              f"eaa frozen={model.eaa.edges_frozen}")

    import datetime as _dt
    _run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(cfg.train.checkpoint_dir, f"run_{_run_id}")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pth")
    best_path = os.path.join(ckpt_dir, "best.pth")

    # #25: Tee stdout to ckpt_dir/train.log
    _tee = _Tee(os.path.join(ckpt_dir, "train.log"))
    sys.stdout = _tee
    print(_dt.datetime.now().isoformat(timespec="seconds")
          + "  cmd: " + " ".join(sys.argv))
    print(f"torch {torch.__version__}  "
          f"cuda {torch.cuda.is_available()}  device {device}")

    val_every = 1 if args.overfit_test else max(1, int(cfg.train.val_interval))
    epochs_no_improve = 0
    _best_epoch = 0

    _contract = cfg.deploy_contract()
    print_session_header(
        run_id=_run_id,
        device=str(device),
        epochs=int(cfg.train.epochs),
        canvas_h=int(cfg.data.img_h),
        canvas_w=int(cfg.data.img_w),
        n_cells=getattr(cfg, "num_cells", sum(
            (cfg.data.img_h // s) * (cfg.data.img_w // s)
            for s in cfg.model.strides)),
        torch_ver=torch.__version__,
        contract_hash=_contract["hash"],
    )

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        criterion.set_epoch(epoch)
        # Per-epoch augmentation reseed (F17): with persistent workers the
        # worker torch seed is fixed for the whole run, so without this the
        # epoch term in the per-sample seed never advances and every epoch
        # replays identical augmentation draws.
        train_ds.set_epoch(epoch)
        # Re-seed the shuffle generator so batch order varies per epoch
        # independently of the per-sample augmentation seed (F17).
        train_loader._nirdet_shuffle_generator.manual_seed(
            int(cfg.train.seed) * 10_000 + epoch)
        try:
            stats, global_step = train_one_epoch(
                model, train_loader, criterion, opt, sched, cfg, device,
                epoch, global_step, ema, amp_dtype, scaler,
                log_every=int(args.log_every))
        except ColdStartError as exc:
            print("\n" + "!" * 70)
            print(exc)
            print("!" * 70)
            if isinstance(sys.stdout, _Tee):
                sys.stdout = sys.stdout._term
                sys.stdout.flush()
            return 2

        n_levels = len(cfg.model.strides)
        per_lvl = "  ".join(f"n_pos_l{l} {stats.get(f'n_pos_l{l}', 0.0):.1f}"
                            for l in range(n_levels))
        print(f"[epoch {epoch:03d}] loss {stats['total']:.4f} "
              f"cls {stats['cls']:.4f} reg {stats['reg']:.4f} "
              f"iou {stats['iou']:.3f} n_pos {stats['n_pos']:.1f}  "
              f"{per_lvl}  lr {stats['lr']:.2e}  {stats['secs']:.1f}s  "
              f"eaa_frozen={model.eaa.edges_frozen}")

        rec = {"epoch": epoch, **{k: float(v) for k, v in stats.items()}}

        # F54: log per-level size bias so a drifting or dead level is
        # immediately visible.
        for lvl, s in enumerate(cfg.model.strides):
            b = model.head.size_pred[lvl].bias.detach().cpu()
            print(f"  size bias L{lvl} (stride {s}): "
                  f"w exp {float(b[0].exp()):.4f}  h exp {float(b[1].exp()):.4f}  "
                  f"(init {cfg.model.prior_w:.4f}/{cfg.model.prior_h:.4f})")

        # ---- validation on the SELECTION split ----
        if val_loader is not None and ((epoch + 1) % val_every == 0 or
                                       epoch == cfg.train.epochs - 1):
            from evaluate import evaluate_split     # lazy: breaks the cycle
            restore = None
            if ema is not None:
                restore = copy.deepcopy(model.state_dict())
                model.load_state_dict(ema.deploy_state_dict(model))
            # F28: an exception during validation must not leave the live
            # model holding EMA weights for the rest of training.
            _t_val_start = time.time()
            try:
                metrics, _cache = evaluate_split(
                    model, val_loader, cfg, device=device,
                    score_thresh=cfg.eval.eval_score_thresh, detailed=False)
            finally:
                if restore is not None:
                    model.load_state_dict(restore)
            _val_time = time.time() - _t_val_start

            map50 = float(metrics.get("map_50", 0.0))
            rec["val_map50"] = map50

            # Improvement is decided BEFORE best_map50 moves.
            improved = map50 > max(best_map50, 0.0)
            delta = map50 - (best_map50 if best_map50 > -1.0 else map50)
            best_mark = "  ★ NEW BEST" if improved else ""
            print(f"[val e{epoch + 1:03d}/{cfg.train.epochs:03d}]"
                  f"  mAP50 {map50:.4f}"
                  f"  (best {max(best_map50, 0.0):.4f} @ e{_best_epoch:03d})"
                  f"  Δ {delta:+.4f}"
                  f"  time {_val_time:.1f}s"
                  f"{best_mark}")

            if improved:
                best_map50 = map50
                _best_epoch = epoch + 1
                epochs_no_improve = 0
                save_checkpoint(best_path, model, ema, opt, cfg, epoch,
                                global_step, best_map50, history + [rec])
                print(f"[ckpt] ★ saved best  e{_best_epoch:03d}"
                      f"  mAP50 {best_map50:.4f}  →  {best_path}")
            else:
                epochs_no_improve += 1

            if cfg.train.es_enabled and \
                    epochs_no_improve >= int(cfg.train.es_patience):
                history.append(rec)
                save_checkpoint(last_path, model, ema, opt, cfg, epoch,
                                global_step, best_map50, history)
                save_history(ckpt_dir, history)
                print(f"[early-stop] no improvement for "
                      f"{epochs_no_improve} evaluations")
                break

        # ---- epoch summary line (#79/#81) ----
        _epoch_secs = stats.get("secs", 0.0)
        _remaining = int(cfg.train.epochs) - (epoch + 1)
        _eta_run = _dur(_epoch_secs * _remaining) if _remaining > 0 else "—"
        print(f"════ e{epoch + 1:03d}/{cfg.train.epochs:03d}"
              f"  train_loss {stats.get('total', 0.0):.4f}"
              f"  val_mAP50 {rec.get('val_map50', float('nan')):.4f}"
              f"  best {max(best_map50, 0.0):.4f}"
              f"  lr {_sci(stats.get('lr', 0.0))}"
              f"  epoch_time {_dur(_epoch_secs)}"
              f"  ETA {_eta_run}"
              f"  patience {epochs_no_improve}/{cfg.train.es_patience}")

        history.append(rec)
        save_checkpoint(last_path, model, ema, opt, cfg, epoch, global_step,
                        best_map50, history)
        # Per-epoch, atomic: a crash / ColdStartError / Ctrl-C must not leave
        # evaluate.py without its per-level n_pos record.
        save_history(ckpt_dir, history)
        write_curves(history, ckpt_dir, _best_epoch if _best_epoch else None)

        # End of epoch: advance the EAA freeze schedule.
        model.step_eaa_epoch()

    print_session_footer(max(best_map50, 0.0), _best_epoch, best_path)
    if isinstance(sys.stdout, _Tee):
        sys.stdout = sys.stdout._term
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
