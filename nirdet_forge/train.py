"""
train.py — NIRDet-Forge training loop
=====================================
    python train.py --profile datasets/<name>.yaml
    python train.py --overfit-test
    python train.py --p5-ablate --profile datasets/<name>.yaml
    python train.py --resume checkpoints/run_*/last.pth

STARTUP ORDER (all of it matters)
  1. --profile applies the dataset profile (canvas fingerprint check sees the
     UNABLATED strides).
  2. --p5-ablate rewrites cfg.model.strides before the model/loss/contract.
  3. validate_config.
  4. Build the model; ImageNet stem transfer writes only the free stem slots.
  5. Measure the EAA projection bias on >= cfg.train.eaa_calib_frames
     UNAUGMENTED frames, before any optimiser step and before the freeze ends.
  6. Epoch 0.

SCALE: the norm-floor abort is a FRACTION of an epoch's batches, the EMA tau
is derived from steps_per_epoch, and validation is time-boxed by
val_interval — nothing here assumes a 32-batch epoch.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
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


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        # cudnn.benchmark picks conv algorithms by TIMING, so runs are not
        # bit-reproducible even under a fixed seed. --deterministic disables it.
        torch.backends.cudnn.benchmark = True


def _section_of(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("eaa."):
        return "eaa"
    if name.startswith("neck."):
        return "neck"
    return "head"


def build_param_groups(model: nn.Module, cfg: Config) -> List[dict]:
    """{backbone, eaa, neck, head} x {decay, no_decay}, layer-wise LR baked in."""
    scale = {
        "backbone": float(cfg.train.lr_scale_backbone),
        "eaa": float(cfg.train.lr_scale_backbone),
        "neck": float(cfg.train.lr_scale_neck),
        "head": float(cfg.train.lr_scale_head),
    }
    buckets: Dict[Tuple[str, bool], List[nn.Parameter]] = {
        (s, d): [] for s in scale for d in (True, False)}

    for name, p in model.named_parameters():
        sec = _section_of(name)
        if "eaa.proj.weight" in name:
            # LOAD-BEARING: proj.weight carries the calibrated gain k; decay
            # would pull it back toward 1/N and re-inert the EAA gate.
            decay = False
        else:
            decay = p.ndim > 1
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
                              momentum=float(cfg.train.momentum), nesterov=True)
    else:
        raise ValueError(f"unknown optimizer '{cfg.train.optimizer}'")
    print("[optim] layer-wise LR groups:")
    for g in opt.param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<18} lr {g['lr']:.3e} (x{g['lr_scale']:.2f})  "
              f"wd {g['weight_decay']:.1e}  params {n:,}")
    return opt


class LRSchedule:
    """Linear warmup to each group's base LR, then cosine to lr_min."""

    def __init__(self, cfg: Config, steps_per_epoch: int,
                 flat: bool = False) -> None:
        self.total = max(1, int(cfg.train.epochs) * max(1, steps_per_epoch))
        self.lr_start = float(cfg.train.lr_start)
        self.lr_peak = float(cfg.train.lr_peak)
        self.lr_min = float(cfg.train.lr_min)
        self.flat = bool(flat)
        derived = round(float(cfg.train.warmup_epochs_frac) *
                        float(cfg.train.epochs) * max(1, steps_per_epoch))
        cap = int(float(cfg.train.epochs) * 0.1 * max(1, steps_per_epoch))
        self.warmup = 0 if flat else max(0, min(int(derived), cap))

    def factor(self, step: int) -> float:
        if self.flat:
            return 1.0
        if self.warmup > 0 and step < self.warmup:
            t = step / float(self.warmup)
            return (self.lr_start + t * (self.lr_peak - self.lr_start)) / self.lr_peak
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


class ModelEMA:
    """EMA with a warmup ramp: d_eff(step) = decay * (1 - exp(-step/tau))."""

    def __init__(self, model: nn.Module, decay: float, tau_steps: int) -> None:
        self.decay = float(decay)
        self.tau = max(1, int(tau_steps))
        self.step = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}
        # Cached float/int partitions — built once on first update() call (P25).
        self._float_keys: Optional[List[str]] = None
        self._int_keys: Optional[List[str]] = None

    def effective_decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.step / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = self.effective_decay()
        msd = model.state_dict()
        if self._float_keys is None:
            # Cache the int/float partition: stable for the model's lifetime.
            self._float_keys = [k for k in self.shadow
                                if torch.is_floating_point(msd[k])]
            self._int_keys = [k for k in self.shadow
                              if k not in set(self._float_keys)]
        for k in self._int_keys:
            self.shadow[k].copy_(msd[k].float())
        dst = [self.shadow[k] for k in self._float_keys]
        src = [msd[k].detach().float() for k in self._float_keys]
        torch._foreach_mul_(dst, d)
        torch._foreach_add_(dst, src, alpha=1.0 - d)

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
        ref_sd = ref.state_dict()
        return {k: self.shadow[k].to(ref_sd[k].dtype).clone() for k in ref_sd}


def imagenet_stem_init(model: NIRDet, cfg: Config) -> bool:
    """Channel-sum a pretrained first conv into the FREE stem slots only."""
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
        print(f"[stem] could not load {arch} weights ({exc}); skipping")
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

    summed = src.sum(dim=1, keepdim=True)
    if summed.shape[-2:] != stem.conv.weight.shape[-2:]:
        summed = torch.nn.functional.interpolate(
            summed, size=stem.conv.weight.shape[-2:], mode="bilinear",
            align_corners=False)
    take = min(n_free, summed.shape[0])
    with torch.no_grad():
        target = stem.conv.weight[lo:lo + take]
        ref_std = float(target.std()) if take > 1 else float(target.abs().mean())
        src_std = float(summed[:take].std()) + 1e-12
        stem.conv.weight[lo:lo + take] = summed[:take] * (ref_std / src_std)
    print(f"[stem] ImageNet transfer from {arch}: {take} filter(s) into slots "
          f"[{lo}, {lo + take}); edge slots [0, {lo}) preserved")
    return True


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
        "deploy_contract_hash": cfg.deploy_contract()["hash"],
        "deploy_contract": cfg.deploy_contract(),
        "cfg_json": json.dumps(cfg.to_dict(), indent=2),
        "eaa_proj_bias": cfg.model.eaa_proj_bias,
        "strides": list(cfg.model.strides),
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def save_history(ckpt_dir: str, history: list) -> None:
    path = os.path.join(ckpt_dir, "history.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    os.replace(tmp, path)


# ---- live output helpers ---------------------------------------------------

_BAR_W = 20


def _sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return f"{m}e{int(e)}"


def _dur(seconds: float) -> str:
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
    def __init__(self, log_path: str) -> None:
        self._term = sys.stdout
        self._log = open(log_path, "a", buffering=1, encoding="utf-8")

    def write(self, data: str) -> None:
        self._term.write(data)
        self._log.write(data.replace("\r", "\n") if data.endswith("\r") else data)

    def flush(self) -> None:
        self._term.flush()
        self._log.flush()

    def close(self) -> None:
        try:
            self._log.close()
        except Exception:
            pass


def write_curves(history: list, ckpt_dir: str,
                 best_epoch: Optional[int]) -> None:
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
    for k in ("total", "cls", "reg", "iou"):
        ax0.plot(ep, [h.get(k, float("nan")) for h in history], label=k)
    ax0.set_ylabel("train loss")
    ax0.legend(loc="upper right")
    ax0.grid(alpha=0.3)
    ax1.plot(ep, [h.get("val_map50", float("nan")) for h in history],
             color="tab:green", label="val mAP50")
    ax1.set_ylabel("val mAP50")
    ax1.set_xlabel("epoch")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    if best_epoch is not None:
        for ax in (ax0, ax1):
            ax.axvline(best_epoch, ls="--", color="crimson", alpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, "curves.png"), dpi=110)
    plt.close(fig)


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
    lr_now = 0.0
    total_batches = len(loader)

    for bi, (imgs, targets, _meta) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in targets]
        lr_now = sched.apply(opt, global_step)

        opt.zero_grad(set_to_none=True)
        use_amp = amp_dtype is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            preds = model(imgs, training_mode=True)
        # Loss OUTSIDE autocast: CIoU / alignment arithmetic stays fp32.
        out = criterion(preds, targets)
        loss = out["total"]

        if not torch.isfinite(loss):
            n_nonfinite += 1
            print(f"\n  e{epoch:03d} b{bi:04d}: non-finite loss, batch skipped "
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
        # Single batched read: one .item() per key per batch, reused for
        # both accumulation and the progress print (P24).
        vals = {k: float(out[k].detach()) for k in acc if k in out}
        for k, v in vals.items():
            acc[k] += v

        if log_every and (bi % log_every == 0 or bi == total_batches - 1):
            done = bi + 1
            elapsed = time.time() - t0
            eta = _dur((elapsed / done) * (total_batches - done))
            print(f"\r[{epoch + 1:03d}/{cfg.train.epochs:03d}]"
                  f" {_bar(done / max(1, total_batches))}"
                  f" {done:>{len(str(total_batches))}d}/{total_batches}"
                  f"  loss {vals['total']:.4f}"
                  f"  cls {vals['cls']:.4f}"
                  f"  reg {vals['reg']:.4f}"
                  f"  iou {vals['iou']:.4f}"
                  f"  n_pos {vals['n_pos']:.0f}"
                  f"  lr {_sci(lr_now)}  ETA {eta}", end="", flush=True)

    print()

    if n_batches == 0:
        if n_nonfinite == 0:
            raise RuntimeError("train loader produced zero batches")
        raise RuntimeError(
            f"no usable batches: all {n_nonfinite} batches were non-finite")

    # Norm-floor abort as a FRACTION of this epoch's batches (dataset-size
    # invariant), with a minimum sample so a smoke run cannot trip it.
    _nfa = getattr(criterion, "_norm_floor_acc", None)
    nf = int(_nfa.item()) if _nfa is not None else 0
    if _nfa is not None:
        _nfa.zero_()
    frac = nf / float(n_batches)
    if n_batches >= int(cfg.loss.norm_floor_min_batches) and \
            frac > float(cfg.loss.norm_floor_abort_frac):
        raise RuntimeError(
            f"epoch {epoch}: norm was floored (target sum < 1) on {nf} of "
            f"{n_batches} batches ({frac * 100:.1f}% > "
            f"{cfg.loss.norm_floor_abort_frac * 100:.1f}%). The model is "
            f"assigning almost no positives. Check the dataset profile "
            f"(prior_w/prior_h) and head.size_pred.bias.")
    if nf > 0:
        print(f"  [loss] norm floored on {nf}/{n_batches} batches "
              f"({frac * 100:.1f}%)")

    stats = {k: v / n_batches for k, v in acc.items()}
    stats["lr"] = lr_now
    stats["secs"] = time.time() - t0
    stats["n_nonfinite"] = n_nonfinite
    stats["norm_floored_frac"] = frac
    return stats, global_step


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train NIRDet-Forge.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prior-w", type=float, default=None,
                    help="canvas-space median box width; smoke tests only")
    ap.add_argument("--prior-h", type=float, default=None,
                    help="canvas-space median box height; smoke tests only")
    ap.add_argument("--overfit-test", action="store_true",
                    help="10 images, flat LR, eval every epoch: a smoke test")
    ap.add_argument("--p5-ablate", action="store_true",
                    help="strides=(8,16): drop the stride-32 level")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--allow-no-profile", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = get_config()

    # 1. profile FIRST: its canvas fingerprint covers its own stride list.
    profile = None
    if args.profile:
        from dataset_profiles import DatasetProfile
        profile = DatasetProfile.load(args.profile)
        profile.apply(cfg)
    else:
        print("\n" + "!" * 74)
        print("  NO --profile GIVEN")
        print("  prior_w/prior_h must come from a dataset measurement:")
        print("    python dataset_profiles.py --root <path> --out datasets/<n>.yaml")
        print("!" * 74 + "\n")

    if args.prior_w is not None or args.prior_h is not None:
        if args.prior_w is None or args.prior_h is None:
            raise SystemExit("--prior-w and --prior-h must be given together")
        cfg.model.prior_w = float(args.prior_w)
        cfg.model.prior_h = float(args.prior_h)
        print(f"[priors] OVERRIDE from CLI: w {cfg.model.prior_w:.6f} "
              f"h {cfg.model.prior_h:.6f} — these are NOT a measurement of "
              f"this dataset unless you measured them yourself.")

    # 2. strides: before the model, the loss geometry and the contract hash.
    if args.p5_ablate:
        cfg.model.strides = (8, 16)
        print("[ablate] strides -> (8, 16); the stride-32 level is removed "
              "from the head and the loss geometry.")

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
        cfg.train.epochs = min(int(cfg.train.epochs), 100)
        cfg.train.val_interval = 1
        cfg.train.es_enabled = False
        if cfg.model.prior_w is None:
            cfg.model.prior_w, cfg.model.prior_h = 0.05, 0.15
            print("[overfit] SYNTHETIC priors 0.05/0.15 (architecture smoke "
                  "test only; never deploy a model trained this way)")

    if not args.profile and int(cfg.train.epochs) > 5 \
            and not args.allow_no_profile and not args.overfit_test:
        raise SystemExit(
            f"refusing to start a {cfg.train.epochs}-epoch run without a "
            f"dataset profile. Pass --profile <yaml>, or --allow-no-profile "
            f"together with --prior-w/--prior-h.")

    # 3. validate
    validate_config(cfg)
    # The config comment promises abspath resolution — make it true. Without
    # this, a run launched from a different CWD scatters checkpoints and eval
    # outputs into separate directory trees.
    cfg.train.checkpoint_dir = os.path.abspath(cfg.train.checkpoint_dir)

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
            f"{len(train_ds)} images. Regenerate the profile:\n"
            f"  python dataset_profiles.py --root {cfg.data.root!r} "
            f"--img-h {cfg.data.img_h} --img-w {cfg.data.img_w} --out <path>")

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
    print(f"[model] params {pb['total']:,} (backbone {pb['backbone']:,} "
          f"eaa {pb['eaa']:,} neck {pb['neck']:,} head {pb['head']:,})")
    if model.count_grouped_convs() != 0:
        raise RuntimeError("grouped/depthwise convolution present; "
                           "backbone.py forbids them")

    # ---- 5. EAA calibration on unaugmented, SHUFFLED frames ----
    if not args.resume:
        calib_target = max(1, int(getattr(cfg.train, "eaa_calib_frames", 32)))
        calib_bs = max(1, min(8, int(cfg.train.batch_size)))
        calib_loader, _cds = build_dataloader(
            cfg, "train", batch_size=calib_bs, shuffle=True, augment=False,
            num_workers=0, limit=10 if args.overfit_test else None)
        chunks: List[torch.Tensor] = []
        have = 0
        for c_imgs, _c_t, _c_m in calib_loader:
            chunks.append(c_imgs)
            have += int(c_imgs.shape[0])
            if have >= calib_target:
                break
        if not chunks:
            raise RuntimeError("EAA calibration loader produced zero batches")
        calib_imgs = torch.cat(chunks, 0)[:calib_target].to(device)
        del chunks
        print(f"[eaa] calibrating on {int(calib_imgs.shape[0])} unaugmented "
              f"frame(s) (cfg.train.eaa_calib_frames={calib_target})")
        model.eval()
        bias = model.calibrate_eaa(calib_imgs, force=False, verbose=True)
        if bias is not None:
            cfg.model.eaa_proj_bias = float(bias)
            print(f"[eaa] cfg.model.eaa_proj_bias <- "
                  f"{cfg.model.eaa_proj_bias:+.5f} (stored in every checkpoint)")
        elif cfg.model.eaa_proj_bias is None:
            cfg.model.eaa_proj_bias = float(model.eaa.proj_bias_value)
        if not model.eaa.is_calibrated:
            raise RuntimeError(
                "EAA calibration did not run. Without it the spatial gate is "
                "a near-uniform rescale the following BatchNorm absorbs.")
        # Save 4 probe frames for per-10-epoch gate-span monitoring (R13).
        imgs_probe = calib_imgs[:4].clone()
        del calib_imgs, calib_loader, _cds
        model.train()
    else:
        imgs_probe = None   # no calibration data on --resume; monitoring skipped
        print("[eaa] --resume: skipping calibration; the checkpoint carries "
              "calibrated proj weights and eaa_proj_bias")

    # ---- loss / optim ----
    criterion = NIRDetLoss(
        img_h=cfg.data.img_h, img_w=cfg.data.img_w, strides=cfg.model.strides,
        lambda_cls=cfg.loss.lambda_cls, lambda_reg=cfg.loss.lambda_reg,
        qfl_beta=cfg.loss.qfl_beta, qfl_alpha=cfg.loss.qfl_alpha,
        tal_topk=cfg.loss.tal_topk, tal_alpha=cfg.loss.tal_alpha,
        tal_beta=cfg.loss.tal_beta, ramp_frac=cfg.loss.ramp_frac,
        tal_min_pos_target=float(getattr(cfg.loss, "tal_min_pos_target", 0.10)),
        total_epochs=cfg.train.epochs, assert_cold_start=True,
    ).to(device)

    opt = build_optimizer(model, cfg)
    sched = LRSchedule(cfg, steps_per_epoch, flat=bool(args.overfit_test))
    print(f"[sched] steps/epoch {steps_per_epoch}  total {sched.total}  "
          f"warmup {sched.warmup}"
          f"{'  (FLAT, overfit test)' if sched.flat else ''}")

    ema = None
    if cfg.train.use_ema:
        tau = ema_tau_for(steps_per_epoch)
        ema = ModelEMA(model, float(cfg.train.ema_decay), tau)
        print(f"[ema] decay {cfg.train.ema_decay} tau {tau} steps "
              f"(= ema_tau_for({steps_per_epoch}), ~5 epochs)")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": None}[str(cfg.train.amp_mode)]
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(amp_dtype is torch.float16 and device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_map50 = -1.0
    history: List[dict] = []
    if args.resume:
        try:
            ck = torch.load(args.resume, map_location=device, weights_only=True)
        except Exception:
            print(f"[resume] WARNING weights_only=True failed for "
                  f"{args.resume}; falling back to weights_only=False. Verify "
                  f"this checkpoint came from this codebase.")
            ck = torch.load(args.resume, map_location=device,
                            weights_only=False)
        got = str(ck.get("deploy_contract_hash", ""))
        want = cfg.deploy_contract()["hash"]
        if got and got != want:
            raise RuntimeError(
                f"checkpoint deploy_contract_hash {got} != current {want}: "
                f"different geometry or preprocessing. Resuming would "
                f"mis-decode every box.")
        if ck.get("eaa_proj_bias") is not None:
            cfg.model.eaa_proj_bias = float(ck["eaa_proj_bias"])
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
        model.set_eaa_epoch(start_epoch)
        # The loss's cold-start guard is about epoch 0 only; a resumed run has
        # already passed it.
        criterion._cold_start_checked = True
        print(f"[resume] {args.resume} -> epoch {start_epoch} "
              f"best_map50 {best_map50:.4f} eaa frozen="
              f"{model.eaa.edges_frozen}")

    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(cfg.train.checkpoint_dir, f"run_{run_id}")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pth")
    best_path = os.path.join(ckpt_dir, "best.pth")

    tee = _Tee(os.path.join(ckpt_dir, "train.log"))
    sys.stdout = tee
    rc = 0
    try:
        print(_dt.datetime.now().isoformat(timespec="seconds")
              + "  cmd: " + " ".join(sys.argv))
        contract = cfg.deploy_contract()
        _w = 50
        _h = contract['hash'][:8]
        print(f"╔{'═' * _w}╗")
        print(f"║ NIRDet-Forge  |  run: {run_id}  |  {str(device):<6}{'':>{_w - 34 - len(str(device))}}║")
        print(f"║ epochs: {cfg.train.epochs:<4}  |  canvas: {cfg.data.img_h}x{cfg.data.img_w}  |  cells: {cfg.num_cells:<4}  ║")
        print(f"║ torch: {torch.__version__:<12}  |  contract: {_h}{'':>{_w - 36}}║")
        print(f"╚{'═' * _w}╝")

        val_every = 1 if args.overfit_test else max(1, int(cfg.train.val_interval))
        epochs_no_improve = 0
        best_epoch = 0

        for epoch in range(start_epoch, int(cfg.train.epochs)):
            criterion.set_epoch(epoch)
            train_ds.set_epoch(epoch)
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
                save_history(ckpt_dir, history)
                rc = 2
                break

            per_lvl = "  ".join(
                f"n_pos_l{l} {stats.get(f'n_pos_l{l}', 0.0):.1f}"
                for l in range(len(cfg.model.strides)))
            print(f"[epoch {epoch:03d}] loss {stats['total']:.4f} "
                  f"cls {stats['cls']:.4f} reg {stats['reg']:.4f} "
                  f"iou {stats['iou']:.3f} n_pos {stats['n_pos']:.1f}  "
                  f"{per_lvl}  lr {stats['lr']:.2e}  {stats['secs']:.1f}s  "
                  f"eaa_frozen={model.eaa.edges_frozen}")

            rec = {"epoch": epoch, **{k: float(v) for k, v in stats.items()}}

            for lvl, s in enumerate(cfg.model.strides):
                b = model.head.size_pred[lvl].bias.detach().cpu()
                print(f"  size bias L{lvl} (stride {s}): "
                      f"w exp {float(b[0].exp()):.4f}  "
                      f"h exp {float(b[1].exp()):.4f}  "
                      f"(init {cfg.model.prior_w:.4f}/{cfg.model.prior_h:.4f})")

            if imgs_probe is not None and model.eaa.is_calibrated and \
                    (epoch % 10 == 0 or epoch == int(cfg.train.epochs) - 1):
                with torch.no_grad():
                    _e = model.eaa.compute_edge_magnitude(imgs_probe)
                    _g = torch.sigmoid(model.eaa.proj(_e))
                    _span = float(_g.max() - _g.min())
                print(f"  eaa gate span {_span:.4f} "
                      f"(calibrated {float(model.eaa._gate_span):.4f})"
                      + ("  <-- COLLAPSED, edge_conv may have drifted "
                         "since calibration" if _span < 0.10 else ""))
            if val_loader is not None and ((epoch + 1) % val_every == 0 or
                                           epoch == cfg.train.epochs - 1):
                from evaluate import evaluate_split   # lazy: breaks the cycle
                restore = None
                if ema is not None:
                    restore = copy.deepcopy(model.state_dict())
                    model.load_state_dict(ema.deploy_state_dict(model))
                t_val = time.time()
                try:
                    metrics, _cache = evaluate_split(
                        model, val_loader, cfg, device=device,
                        score_thresh=cfg.eval.eval_score_thresh, detailed=False)
                finally:
                    if restore is not None:
                        model.load_state_dict(restore)
                val_time = time.time() - t_val

                map50 = float(metrics.get("map_50", float("nan")))
                rec["val_map50"] = map50
                # NaN (a GT-less split) must never count as an improvement.
                improved = bool(np.isfinite(map50)) and map50 > max(best_map50, 0.0)
                delta = (map50 - best_map50) if best_map50 > -1.0 else 0.0
                print(f"[val e{epoch + 1:03d}/{cfg.train.epochs:03d}]  "
                      f"mAP50 {map50:.4f}  (best {max(best_map50, 0.0):.4f} "
                      f"@ e{best_epoch:03d})  Δ {delta:+.4f}  "
                      f"time {val_time:.1f}s"
                      f"{'  ★ NEW BEST' if improved else ''}")

                if improved:
                    best_map50 = map50
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    save_checkpoint(best_path, model, ema, opt, cfg, epoch,
                                    global_step, best_map50, history + [rec])
                    print(f"[ckpt] ★ saved best  e{best_epoch:03d}  mAP50 "
                          f"{best_map50:.4f}  →  {best_path}")
                else:
                    epochs_no_improve += 1

                if cfg.train.es_enabled and \
                        epochs_no_improve >= int(cfg.train.es_patience):
                    history.append(rec)
                    save_checkpoint(last_path, model, ema, opt, cfg, epoch,
                                    global_step, best_map50, history)
                    save_history(ckpt_dir, history)
                    write_curves(history, ckpt_dir,
                                 (best_epoch - 1) if best_epoch else None)
                    print(f"[early-stop] no improvement for "
                          f"{epochs_no_improve} evaluations")
                    model.step_eaa_epoch()
                    break

            remaining = int(cfg.train.epochs) - (epoch + 1)
            eta_run = _dur(stats.get("secs", 0.0) * remaining) if remaining else "-"
            print(f"════ e{epoch + 1:03d}/{cfg.train.epochs:03d}  "
                  f"train_loss {stats.get('total', 0.0):.4f}  "
                  f"val_mAP50 {rec.get('val_map50', float('nan')):.4f}  "
                  f"best {max(best_map50, 0.0):.4f}  "
                  f"lr {_sci(stats.get('lr', 0.0))}  "
                  f"epoch_time {_dur(stats.get('secs', 0.0))}  ETA {eta_run}  "
                  f"patience {epochs_no_improve}/{cfg.train.es_patience}")

            history.append(rec)
            save_checkpoint(last_path, model, ema, opt, cfg, epoch, global_step,
                            best_map50, history)
            save_history(ckpt_dir, history)
            write_curves(history, ckpt_dir,
                         (best_epoch - 1) if best_epoch else None)
            model.step_eaa_epoch()

        if rc == 0:
            print("=" * 60)
            print(f"DONE  best mAP50 {max(best_map50, 0.0):.4f} @ epoch "
                  f"{best_epoch}")
            print(f"to evaluate:  python evaluate.py --checkpoint {best_path}")
            print("=" * 60)
    except KeyboardInterrupt:
        print("\n[main] interrupted; last.pth and history.json are on disk")
        rc = 130
    finally:
        # ALWAYS restore stdout and close the log, on every exit path.
        sys.stdout = tee._term
        tee.close()
        sys.stdout.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
