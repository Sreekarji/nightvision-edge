#!/usr/bin/env python
"""
evaluate.py — NIRDet-Lite evaluation
=====================================
Fixes relative to the original
------------------------------
1. mAP50-95 comes straight from torchmetrics' ``map`` field. The old code
   hand-rolled it by averaging ``precision[t, :, 0, 0, -1]`` with a
   ``>= 0`` mask; COCO uses 0 (not -1) for unreachable recall points, so
   masking them out inflated the number. The premise that ``map`` averages
   across area buckets was also wrong: ``map`` is the IoU-averaged AP for
   area=all.

2. Bootstrap confidence intervals on every headline metric. With 160 val
   images the standard error on mAP50 is ~0.02-0.04, so a bare point estimate
   cannot support a "beats the 0.735 baseline" claim. Images are resampled
   with replacement and the metric recomputed from the cached predictions.

3. The dishonest latency benchmark is gone. Timing fp32 eager PyTorch on a
   zero tensor and dividing by a hard-coded 3-5x cannot predict INT8 NCNN on
   a Cortex-A76. What remains is an explicit placeholder that prints the
   commands which produce a real measurement.

4. ``deploy_state`` is preferred when loading a checkpoint, so evaluation uses
   the EMA weights that training intended for deployment.

5. Thresholds are passed as arguments; no module state is mutated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm

from config import get_config
from dataset import NIRDetDataset, collate_fn
from model import build_nirdet

COCO_IOUS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
MAXDETS = [1, 10, 300]
_WEIGHT_KEYS = ("deploy_state", "model_state", "model_state_dict", "state_dict")


# ===========================================================================
# checkpoint
# ===========================================================================

def load_checkpoint(path: Path, device: torch.device, cfg,
                    prefer: str = "deploy", verbose: bool = True):
    if not path.exists():
        near = sorted(p.name for p in path.parent.glob("*.pth")) \
            if path.parent.exists() else []
        raise FileNotFoundError(f"no checkpoint at {path}. nearby: {near}")

    # weights_only=False is required because checkpoints store cfg.to_dict()
    # as a plain Python dict via pickle. If you switch to saving config as a
    # sidecar JSON, change this to weights_only=True.
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    order = list(_WEIGHT_KEYS)
    if prefer == "live":
        order = ["model_state"] + [k for k in order if k != "model_state"]

    state, used = None, None
    if isinstance(ckpt, dict):
        for k in order:
            if isinstance(ckpt.get(k), dict) and ckpt[k]:
                v = ckpt[k]
                state = v.get("weights", v) if "weights" in v else v
                used = k
                break
        if state is None and all(torch.is_tensor(v) for v in ckpt.values()):
            state, used = ckpt, "<bare state_dict>"
    if state is None:
        raise RuntimeError(f"no weights in {path}; keys="
                           f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    model = build_nirdet(cfg)
    inc = model.load_state_dict(state, strict=False)
    model.to(device).eval()

    if verbose:
        print("=" * 62)
        print("  checkpoint")
        print("=" * 62)
        print(f"  file        : {path}")
        print(f"  weights key : {used}"
              f"{'  (EMA / deployment copy)' if used == 'deploy_state' else ''}")
        print(f"  epoch       : {int(ckpt.get('epoch', -1)) + 1}")
        rec = ckpt.get("best_map50", float('nan'))
        print(f"  recorded    : "
              f"{'n/a' if rec != rec else f'{float(rec):.4f}'}")
        print(f"  params      : {sum(p.numel() for p in model.parameters()):,}")
        if inc.missing_keys:
            print(f"  ! missing   : {inc.missing_keys[:5]}")
        if inc.unexpected_keys:
            print(f"  ! unexpected: {inc.unexpected_keys[:5]}")
        if not inc.missing_keys and not inc.unexpected_keys:
            print("  state dict  : exact match")
        print()
    return model, ckpt


def amp_context(device: torch.device):
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.amp.autocast("cpu", enabled=False)


def yolo_to_xyxy(labels: torch.Tensor, w: int, h: int) -> torch.Tensor:
    if labels.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    cx, cy = labels[:, 1] * w, labels[:, 2] * h
    bw, bh = labels[:, 3] * w, labels[:, 4] * h
    return torch.stack([cx - bw / 2, cy - bh / 2,
                        cx + bw / 2, cy + bh / 2], 1).float()


# ===========================================================================
# single cached inference pass
# ===========================================================================

@torch.no_grad()
def run_inference(model, loader, device, score_thresh: float,
                 desc: str = "  inference") -> Tuple[List[Dict], List[Dict]]:
    """One pass; every metric below consumes this cache, so all numbers are
    guaranteed to describe the same predictions."""
    model.eval()
    ctx = amp_context(device)
    preds: List[Dict] = []
    tgts: List[Dict] = []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        h, w = images.shape[-2], images.shape[-1]
        with ctx:
            raw = model.head(model.forward_features(images), training_mode=False)
        results = model.decode_predictions(raw, (h, w),
                                           score_thresh=score_thresh)
        for b, (boxes, scores) in enumerate(results):
            preds.append({
                "boxes": boxes.detach().float().cpu(),
                "scores": scores.detach().float().cpu(),
                "labels": torch.zeros(scores.numel(), dtype=torch.long),
            })
            t = labels[b]
            tgts.append({
                "boxes": yolo_to_xyxy(t, w, h),
                "labels": (t[:, 0].long() if t.numel()
                           else torch.zeros(0, dtype=torch.long)),
            })
    return preds, tgts


def compute_map(preds: List[Dict], tgts: List[Dict],
                iou_thresholds: Optional[List[float]],
                extended: bool = False):
    m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                             iou_thresholds=iou_thresholds,
                             max_detection_thresholds=MAXDETS,
                             extended_summary=extended)
    m.update(preds, tgts)
    return m.compute(), m


def _f(res: Dict, key: str, default: float = float("nan")) -> float:
    v = res.get(key, None)
    if v is None:
        return default
    return float(v)


# ===========================================================================
# bootstrap confidence intervals
# ===========================================================================

def bootstrap_ci(preds: List[Dict], tgts: List[Dict], n_boot: int,
                 seed: int, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """
    Resample IMAGES with replacement and recompute the metrics from the cached
    predictions. Image-level resampling is the right unit here: annotations
    within one frame are correlated, so resampling boxes would understate the
    interval.

    Returns {metric: {"mean", "std", "lo", "hi", "median"}}.
    """
    if n_boot <= 0 or not preds:
        return {}
    rng = np.random.default_rng(seed)
    n = len(preds)
    acc: Dict[str, List[float]] = {"map50": [], "map5095": [], "map75": [],
                                   "mar300": []}

    for _ in tqdm(range(n_boot), desc="  bootstrap", leave=False):
        idx = rng.integers(0, n, size=n)
        bp = [preds[i] for i in idx]
        bt = [tgts[i] for i in idx]
        if sum(int(t["boxes"].shape[0]) for t in bt) == 0:
            continue                      # degenerate resample: no GT at all
        r, _ = compute_map(bp, bt, iou_thresholds=COCO_IOUS)
        acc["map50"].append(_f(r, "map_50", 0.0))
        acc["map5095"].append(_f(r, "map", 0.0))
        acc["map75"].append(_f(r, "map_75", 0.0))
        acc["mar300"].append(_f(r, f"mar_{MAXDETS[-1]}", 0.0))

    out: Dict[str, Dict[str, float]] = {}
    for k, v in acc.items():
        if not v:
            continue
        a = np.asarray(v, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        out[k] = {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)),
            "lo": float(np.percentile(a, 100.0 * alpha / 2.0)),
            "hi": float(np.percentile(a, 100.0 * (1.0 - alpha / 2.0))),
            "n": int(a.size),
        }
    return out


def prob_beats(preds: List[Dict], tgts: List[Dict], baseline: float,
               n_boot: int, seed: int) -> float:
    """Fraction of bootstrap replicates whose mAP50 exceeds the baseline.
    This is the honest form of the 'do we beat YOLO11n' claim."""
    if n_boot <= 0:
        return float("nan")
    rng = np.random.default_rng(seed + 7)
    n = len(preds)
    wins = total = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bt = [tgts[i] for i in idx]
        if sum(int(t["boxes"].shape[0]) for t in bt) == 0:
            continue
        r, _ = compute_map([preds[i] for i in idx], bt, iou_thresholds=[0.5])
        wins += int(_f(r, "map_50", 0.0) > baseline)
        total += 1
    return wins / max(total, 1)


# ===========================================================================
# PR curve
# ===========================================================================

def extract_pr_curve(res: Dict, metric: MeanAveragePrecision,
                     iou_idx: int = 0, cls_idx: int = 0,
                     area_idx: int = 0, md_idx: int = -1) -> Dict:
    """
    extended_summary=True gives precision/scores as (T, R, K, A, M).
    The ``scores`` tensor is what makes the operating point deployable: it is
    the actual confidence threshold achieving each recall level, not the recall
    value itself.
    """
    if "precision" not in res or "scores" not in res:
        raise KeyError("extended_summary keys absent; "
                       "pip install --upgrade torchmetrics[detection]")
    p = res["precision"][iou_idx, :, cls_idx, area_idx, md_idx]
    s = res["scores"][iou_idx, :, cls_idx, area_idx, md_idx]
    rt = getattr(metric, "rec_thresholds", None)
    recall = (np.asarray(rt, dtype=np.float64) if rt is not None
              else np.linspace(0.0, 1.0, p.numel()))
    pv = p.detach().cpu().numpy().astype(np.float64)
    sv = s.detach().cpu().numpy().astype(np.float64)
    ok = pv >= 0.0
    return {"recall": recall[ok], "precision": pv[ok],
            "score_thresh": sv[ok], "n_points": int(ok.sum())}


def best_f1_point(curve: Dict) -> Dict:
    p, r, s = curve["precision"], curve["recall"], curve["score_thresh"]
    den = p + r
    f1 = np.zeros_like(p)
    nz = den > 0
    f1[nz] = 2.0 * p[nz] * r[nz] / den[nz]
    if f1.size == 0 or not np.any(f1 > 0):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "score_thresh": float("nan"), "index": -1}
    i = int(np.argmax(f1))
    return {"precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f1[i]), "score_thresh": float(s[i]), "index": i}


def plot_pr(curve: Dict, best: Dict, map50: float, ci: Dict,
            baseline: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=130)
    lab = f"NIRDet-Lite  AP@0.50 = {map50:.4f}"
    if "map50" in ci:
        lab += f"  [{ci['map50']['lo']:.3f}, {ci['map50']['hi']:.3f}]"
    ax.plot(curve["recall"], curve["precision"], lw=2.0, label=lab)
    ax.axhline(baseline, color="grey", ls="--", lw=1.2,
               label=f"YOLO11n baseline ({baseline:.3f})")
    if "map50" in ci:
        ax.axhspan(ci["map50"]["lo"], ci["map50"]["hi"], color="tab:blue",
                   alpha=0.08, label="95% CI on AP@0.50 (image bootstrap)")
    if best["index"] >= 0:
        ax.axvline(best["recall"], color="crimson", ls="--", lw=1.0, alpha=0.65)
        ax.plot([best["recall"]], [best["precision"]], "o", ms=8,
                color="crimson", ls="none",
                label=(f"best F1 = {best['f1']:.3f} "
                       f"(conf >= {best['score_thresh']:.3f})\n"
                       f"P = {best['precision']:.3f}, R = {best['recall']:.3f}"))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("NIRDet-Lite — PR curve @ IoU 0.50 (val, class: person)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ===========================================================================
# per-image CSV + visualisation
# ===========================================================================

def per_image_rows(preds, tgts, names, thresh: float) -> List[Dict]:
    """
    n_pred / hit / is_hard_case are evaluated at the DEPLOYMENT threshold by
    post-filtering the cached predictions. That is lossless: NMS at 0.05
    followed by a score filter at 0.30 keeps exactly the boxes that NMS at
    0.30 would have kept, because a higher-scoring box always suppresses a
    lower-scoring overlap.
    """
    rows = []
    for name, pr, tg in zip(names, preds, tgts):
        sc = pr["scores"]
        keep = sc >= thresh
        bx = pr["boxes"][keep]
        n_gt = int(tg["boxes"].shape[0])
        best = float(box_iou(bx, tg["boxes"]).max()) if (bx.numel() and n_gt) else 0.0
        rows.append({
            "filename": name,
            "n_gt": n_gt,
            "n_pred": int(keep.sum()),
            "max_score": round(float(sc.max()), 4) if sc.numel() else 0.0,
            "best_iou": round(best, 4),
            "hit": int(best >= 0.5),
            "is_hard_case": int(n_gt > 0 and int(keep.sum()) == 0),
        })
    return rows


def write_csv(rows: List[Dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def render(img_chw, gt, pb, ps) -> np.ndarray:
    """Contrast-stretched so near-black NIR frames stay inspectable."""
    g = (img_chw.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for b in gt.tolist():
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "GT", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    for b, s in zip(pb.tolist(), ps.tolist()):
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(vis, f"{s:.2f}", (x1, min(y2 + 12, vis.shape[0] - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


def save_hard_cases(ds, preds, tgts, rows, out_dir: Path,
                    max_cases: int) -> List[str]:
    """Pure false negatives at the deployment threshold. The drawn predictions
    are the 0.05-threshold cache, so 'saw nothing' and 'fired weakly' are
    visually distinguishable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i in [j for j, r in enumerate(rows) if r["is_hard_case"]][:max_cases]:
        img, _ = ds[i]
        p = out_dir / f"hard_{Path(rows[i]['filename']).stem}.jpg"
        cv2.imwrite(str(p), render(img, tgts[i]["boxes"],
                                   preds[i]["boxes"], preds[i]["scores"]))
        saved.append(p.name)
    return saved


def save_all_predictions(ds, preds, tgts, names, out_dir: Path,
                         thresh: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(ds)):
        img, _ = ds[i]
        keep = preds[i]["scores"] >= thresh
        cv2.imwrite(str(out_dir / names[i]),
                    render(img, tgts[i]["boxes"],
                           preds[i]["boxes"][keep], preds[i]["scores"][keep]))
    return len(ds)


# ===========================================================================
# latency: honest placeholder
# ===========================================================================

def latency_placeholder(cfg, img_h: int, img_w: int) -> Dict:
    """
    Deliberately NOT a measurement.

    The removed benchmark timed fp32 eager PyTorch on a zero-filled tensor
    (so NMS cost was ~0) and divided by a hard-coded 3-5x to "predict" INT8
    NCNN on a Cortex-A76. Those two things differ by an order of magnitude in
    opposite directions, so the output was noise dressed as data.
    """
    txt = [
        "Latency is NOT measured here. Deployment numbers must come from the",
        "target device, in the target runtime, on real frames.",
        "",
        "Raspberry Pi 5 (NCNN INT8):",
        "  python export_onnx.py --checkpoint checkpoints/best.pth",
        f"  python -m onnxsim {cfg.export.onnx_path} {cfg.export.onnx_sim_path} \\",
        f"      --overwrite-input-shape img:1,1,{img_h},{img_w}",
        f"  onnx2ncnn {cfg.export.onnx_sim_path} nirdet.param nirdet.bin",
        "  ncnnoptimize nirdet.param nirdet.bin nirdet-opt.param nirdet-opt.bin 0",
        "  ncnn2table nirdet-opt.param nirdet-opt.bin calib.txt nirdet.table \\",
        f"      mean=0,0,0 norm=0.00392156862745,0,0 shape=[{img_w},{img_h},1] \\",
        "      pixel=GRAY thread=4 method=kl",
        "  ncnn2int8 nirdet-opt.param nirdet-opt.bin \\",
        "      nirdet-int8.param nirdet-int8.bin nirdet.table",
        "  benchncnn 20 3 0 -1 0      # then: python live_nirdet.py --bench",
        "  # verify first: vcgencmd get_throttled must read 0x0",
        "",
        "STM32N6570-DK (Neural-ART):",
        "  python quantize_qdq.py --onnx <sim.onnx> --data-root <root>",
        "  stedgeai analyze --model nirdet-int8-qdq.onnx --target stm32n6 \\",
        "      --st-neural-art default@user_neuralart.json --input-data-type uint8",
        "  # the analyze report is the go/no-go gate: require ~zero SW_FLOAT",
        "  # layers, internal RAM < ~1.5 MB, weights < ~3 MB",
    ]
    return {"measured": False, "reason": "cross-runtime extrapolation is not a "
                                         "measurement", "instructions": txt}


# ===========================================================================
# main
# ===========================================================================

def main(args: argparse.Namespace) -> int:
    cfg = get_config()

    # ---------------- dataset profile (Upgrade H) ----------------
    # Same contract as train.py: verify the sha256 fingerprint over the
    # label files before touching the dataset, then apply ONLY
    # root/n_train/n_val/prior_w/prior_h. The measured deploy threshold
    # read below was fingerprint-verified too.
    profile = None
    if args.profile:
        from dataset_profiles import DatasetProfile
        profile = DatasetProfile.load(args.profile)
        profile.verify_fresh()   # hard error if labels changed since profile was generated
        profile.apply(cfg)
        if args.data_root:
            # Same guard as train.py: the fingerprint was verified against
            # the profiled root; evaluating a different dataset against
            # this profile's priors/threshold would be silently wrong.
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

    # Deployment threshold precedence: explicit --score-thresh >
    # profile deploy_score_thresh (measured on THIS dataset by a previous
    # full training + evaluation cycle) > cfg.eval.deploy_score_thresh.
    if args.score_thresh is None:
        measured = (profile.data.get("deploy_score_thresh")
                    if profile is not None else None)
        args.score_thresh = (float(measured) if measured is not None
                             else cfg.eval.deploy_score_thresh)

    if args.data_root:
        cfg.data.root = args.data_root
    if args.img_h:
        cfg.data.img_h = args.img_h
    if args.img_w:
        cfg.data.img_w = args.img_w
    # Priors are canvas-space: refuse a canvas the profile was not derived
    # at (after --img-h/--img-w overrides have been applied).
    if profile is not None:
        profile.check_canvas(cfg)
    if args.bootstrap is not None:
        cfg.eval.bootstrap_n = args.bootstrap

    out_dir = Path(args.out_dir or cfg.eval.out_dir)
    hard_dir = out_dir / "hard_cases"
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nNIRDet-Lite evaluation")
    print(f"  device : {device}"
          f"{f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''}")
    print(f"  torch  : {torch.__version__}\n")

    model, ckpt = load_checkpoint(Path(args.checkpoint), device, cfg,
                                  prefer=args.weights)
    epoch = int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else -1
    recorded = float(ckpt.get("best_map50", float("nan"))) \
        if isinstance(ckpt, dict) else float("nan")
    n_params = sum(p.numel() for p in model.parameters())

    ds = NIRDetDataset(cfg.data.root, "val", cfg.data.img_h, cfg.data.img_w,
                       augment=False, cfg_aug=cfg.aug)
    if profile is not None and len(ds) != cfg.data.n_val:
        raise RuntimeError(
            f"profile n_val={cfg.data.n_val} does not match the val loader "
            f"({len(ds)} images): the profile counts label files while the "
            f"loader counts image files, so either images were added/removed "
            f"after profiling or val has images without label files. "
            f"Regenerate the profile (python dataset_profiles.py --root "
            f"{cfg.data.root} --out {args.profile})")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=args.num_workers,
                        collate_fn=collate_fn,
                        pin_memory=(device.type == "cuda"))
    names = [p.name for p in ds.img_paths]
    print(f"  val images : {len(ds)} (batch {args.batch_size})\n")

    print("=" * 62)
    print("  single inference pass (feeds every metric below)")
    print("=" * 62)
    preds, tgts = run_inference(model, loader, device,
                                cfg.eval.eval_score_thresh)
    assert len(preds) == len(ds), f"{len(preds)} preds vs {len(ds)} images"
    n_gt_total = sum(int(t["boxes"].shape[0]) for t in tgts)
    print(f"  cached {len(preds)} predictions | {n_gt_total} GT boxes\n")

    # ---------------- accuracy ----------------
    print("=" * 62)
    print("  detection accuracy")
    print("=" * 62)
    res50, m50 = compute_map(preds, tgts, [0.5], extended=True)
    map50 = _f(res50, "map_50", 0.0)

    res, _ = compute_map(preds, tgts, COCO_IOUS)
    map5095 = _f(res, "map", 0.0)              # torchmetrics directly
    map75 = _f(res, "map_75", 0.0)
    mar300 = _f(res, f"mar_{MAXDETS[-1]}", 0.0)
    map_s = _f(res, "map_small", -1.0)
    map_m = _f(res, "map_medium", -1.0)
    map_l = _f(res, "map_large", -1.0)
    delta = map50 - cfg.eval.baseline_map50

    print(f"  mAP50       : {map50:.4f}   "
          f"(baseline {cfg.eval.baseline_map50:.4f}, delta {delta:+.4f})")
    print(f"  mAP50-95    : {map5095:.4f}   (torchmetrics 'map', area=all)")
    print(f"  mAP75       : {map75:.4f}")
    print(f"  mAR@300     : {mar300:.4f}")
    print(f"  size split  : s={map_s:.4f} m={map_m:.4f} l={map_l:.4f}   "
          f"(-1 = no objects of that COCO size)")
    if recorded == recorded and abs(recorded - map50) > 0.01:
        print(f"  ! mAP50 differs from the recorded {recorded:.4f} by >0.01; "
              f"check eval_score_thresh and the weights key")
    print()

    # ---------------- bootstrap CIs ----------------
    ci: Dict = {}
    p_beat = float("nan")
    if cfg.eval.bootstrap_n > 0:
        print("=" * 62)
        print(f"  bootstrap confidence intervals "
              f"({cfg.eval.bootstrap_n} image resamples)")
        print("=" * 62)
        ci = bootstrap_ci(preds, tgts, cfg.eval.bootstrap_n,
                          cfg.eval.bootstrap_seed)
        for k, label in (("map50", "mAP50"), ("map5095", "mAP50-95"),
                         ("map75", "mAP75"), ("mar300", "mAR@300")):
            if k in ci:
                c = ci[k]
                print(f"  {label:<9}: {c['mean']:.4f} +/- {c['std']:.4f}   "
                      f"95% CI [{c['lo']:.4f}, {c['hi']:.4f}]")
        p_beat = prob_beats(preds, tgts, cfg.eval.baseline_map50,
                            cfg.eval.bootstrap_n, cfg.eval.bootstrap_seed)
        print(f"  P(mAP50 > baseline {cfg.eval.baseline_map50:.3f}) = "
              f"{p_beat:.3f}")
        if "map50" in ci and ci["map50"]["lo"] <= cfg.eval.baseline_map50:
            print("  ! the 95% CI includes the baseline: this run does NOT "
                  "establish an improvement. Report the interval, not the "
                  "point estimate.")
        print(f"  NOTE: best.pth was selected on these same {len(ds)} images "
              f"across many evaluations, so this mAP is an optimistically "
              f"biased estimate. For a publishable number use k-fold CV or a "
              f"held-out test split never used for checkpoint selection.")
        print()

    # ---------------- PR curve ----------------
    print("=" * 62)
    print("  precision-recall curve")
    print("=" * 62)
    # Snapshot the threshold source BEFORE record_deploy_threshold can write
    # it to profile.data: a first-ever evaluation must report "config default",
    # not "profile".
    _thresh_source = (
        "profile"
        if (profile is not None and
            profile.data.get("deploy_score_thresh") is not None)
        else "config default"
    )
    best = {"index": -1, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "score_thresh": float("nan")}
    curve = {"n_points": 0, "recall": np.array([]), "precision": np.array([]),
             "score_thresh": np.array([])}
    try:
        curve = extract_pr_curve(res50, m50)
        best = best_f1_point(curve)
        pr_path = out_dir / "pr_curve.png"
        plot_pr(curve, best, map50, ci, cfg.eval.baseline_map50, pr_path)
        print(f"  valid points : {curve['n_points']}")
        print(f"  saved        : {pr_path}")
        if best["index"] >= 0:
            dst_note = (f"written back to {args.profile}"
                        if profile is not None
                        else "set cfg.eval.deploy_score_thresh / SCORE_TH")
            print(f"  best-F1 operating point")
            print(f"     confidence : {best['score_thresh']:.4f}   "
                  f"<- {dst_note}")
            print(f"     precision  : {best['precision']:.4f}")
            print(f"     recall     : {best['recall']:.4f}")
            print(f"     F1         : {best['f1']:.4f}")
            if profile is not None:
                # deploy_score_thresh is deliberately NOT derived — it needs
                # a training run to know. Persist the measured best-F1
                # operating point into the profile so the next export uses
                # it. record_deploy_threshold() re-verifies the label
                # fingerprint before writing, so a threshold measured
                # against changed labels is never preserved.
                try:
                    profile.record_deploy_threshold(best["score_thresh"],
                                                    path=args.profile)
                except Exception as exc:
                    print(f"  ! deploy_score_thresh NOT written back: {exc}")
    except KeyError as exc:
        print(f"  ! PR curve unavailable: {exc}")
    print()

    # ---------------- per-image CSV ----------------
    print("=" * 62)
    print("  per-image results")
    print("=" * 62)
    rows = per_image_rows(preds, tgts, names, args.score_thresh)
    csv_path = out_dir / "per_image_results.csv"
    write_csv(rows, csv_path)
    n_hard = sum(r["is_hard_case"] for r in rows)
    n_hit = sum(r["hit"] for r in rows)
    n_with_gt = sum(1 for r in rows if r["n_gt"] > 0)
    print(f"  rows        : {len(rows)} -> {csv_path}")
    print(f"  with GT     : {n_with_gt}")
    print(f"  with a hit  : {n_hit} (IoU >= 0.5 at conf {args.score_thresh:g})")
    print(f"  pure FN     : {n_hard}\n")

    print("=" * 62)
    print("  hard cases")
    print("=" * 62)
    saved = save_hard_cases(ds, preds, tgts, rows, hard_dir,
                            cfg.eval.max_hard_cases)
    print(f"  saved {len(saved)} -> {hard_dir}/")
    for s in saved:
        print(f"     {s}")
    if not saved:
        print("     (none: at least one detection on every annotated image)")
    print()

    if not args.no_render_all:
        print("=" * 62)
        print("  all predictions")
        print("=" * 62)
        n = save_all_predictions(ds, preds, tgts, names, pred_dir,
                                 args.score_thresh)
        print(f"  saved {n} -> {pred_dir}/\n")

    # ---------------- latency ----------------
    print("=" * 62)
    print("  latency")
    print("=" * 62)
    lat = latency_placeholder(cfg, cfg.data.img_h, cfg.data.img_w)
    for line in lat["instructions"]:
        print(f"  {line}")
    print()

    # ---------------- summary ----------------
    print("=" * 62)
    print("  summary")
    print("=" * 62)
    print(f"  checkpoint epoch : {epoch}")
    print(f"  val images       : {len(ds)}")
    print(f"  params           : {n_params:,}")
    if "map50" in ci:
        c = ci["map50"]
        print(f"  mAP50            : {map50:.4f}  "
              f"[{c['lo']:.4f}, {c['hi']:.4f}]  delta {delta:+.4f}")
    else:
        print(f"  mAP50            : {map50:.4f}  delta {delta:+.4f}")
    print(f"  mAP50-95         : {map5095:.4f}")
    if best["index"] >= 0:
        print(f"  deploy threshold : {best['score_thresh']:.3f}  "
              f"(P {best['precision']:.3f} / R {best['recall']:.3f} / "
              f"F1 {best['f1']:.3f})")
    print(f"  hard cases       : {hard_dir}/ ({len(saved)})")
    print("=" * 62)

    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "profile": (str(args.profile) if profile is not None else None),
        "deploy_score_thresh_source": _thresh_source,
        "weights_key_preference": args.weights,
        "epoch": epoch,
        "n_val": len(ds),
        "n_gt_boxes": n_gt_total,
        "n_params": n_params,
        "input_size": [cfg.data.img_h, cfg.data.img_w],
        "strides": list(cfg.model.strides),
        "eval_score_thresh": cfg.eval.eval_score_thresh,
        "deploy_score_thresh": args.score_thresh,
        "map50": map50,
        "map5095": map5095,
        "map75": map75,
        "mar_300": mar300,
        "map_small": map_s,
        "map_medium": map_m,
        "map_large": map_l,
        "baseline_map50": cfg.eval.baseline_map50,
        "delta_vs_baseline": delta,
        "bootstrap": ci,
        "prob_beats_baseline": p_beat,
        "best_f1_point": best,
        "n_hard_cases": len(saved),
        "n_pure_false_negative_images": n_hard,
        "latency": lat,
    }
    jp = out_dir / "summary.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  machine-readable summary -> {jp}\n")
    return 0


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    p = argparse.ArgumentParser("NIRDet-Lite evaluation")
    p.add_argument("--checkpoint", default=str(Path(cfg.train.checkpoint_dir)
                                               / "best.pth"))
    p.add_argument("--weights", choices=("deploy", "live"), default="deploy",
                   help="'deploy' uses the EMA copy (default), 'live' the raw "
                        "training weights")
    p.add_argument("--data-root", default=None)
    p.add_argument("--img-h", type=int, default=None)
    p.add_argument("--img-w", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--score-thresh", type=float, default=None,
                   help="deployment threshold for the CSV / renders "
                        "(mAP always uses eval_score_thresh). Default: "
                        "profile deploy_score_thresh if --profile, else "
                        "cfg.eval.deploy_score_thresh")
    p.add_argument("--profile", default=None,
                   help="dataset profile YAML (datasets/*.yaml). Verifies the "
                        "label fingerprint, applies root/priors, and writes "
                        "the measured best-F1 threshold back to the profile "
                        "as deploy_score_thresh")
    p.add_argument("--bootstrap", type=int, default=None,
                   help="number of image resamples (0 disables CIs)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-render-all", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
