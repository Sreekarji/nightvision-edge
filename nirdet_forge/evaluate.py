"""
evaluate.py — fp32 evaluation and deploy-threshold measurement
===============================================================
    python evaluate.py --checkpoint checkpoints/best.pth \
                       --profile datasets/miniNIRPed_261.yaml

SELECTION SPLIT != REPORT SPLIT
-------------------------------
best.pth is chosen by maximising mAP50 on cfg.eval.select_split ('val') across
roughly one evaluation per epoch. Reporting the headline number on that same
split is optimistically biased by construction — it is a maximum over ~100
noisy estimates of the same quantity. The headline therefore comes from
cfg.eval.report_split ('test'), which the selection procedure never saw.
validate_config() errors if the two are equal, and this script runs report-
split inference only after the checkpoint has already been chosen.

evaluate_split() is shared with train.py, so the val-selection number and the
test-report number are produced by identical code.

WHAT deploy_score_thresh IS
---------------------------
The single scalar the Pi and the STM32 need and cannot derive for themselves.
It is measured here as the threshold maximising F1 at IoU 0.5 on the report
split, swept 0.05..0.95 in 0.01 steps, and written back into the profile YAML.
live_nirdet.py --profile reads it from there. No deployment file contains a
threshold literal.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("evaluate.py needs PyYAML: pip install pyyaml") from exc

from config import Config, get_config, validate_config
from dataset import build_dataloader
from model import NIRDet, build_nirdet
from train import CKPT_DEPLOY_KEY

try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "evaluate.py needs torchmetrics: pip install torchmetrics pycocotools"
    ) from exc


# ===========================================================================
# metric helpers
# ===========================================================================

def _map50_metric() -> MeanAveragePrecision:
    """
    Headline / bootstrap metric: IoU 0.5 only, single class.

    Single class means class_metrics=False loses nothing, and restricting to
    one IoU threshold makes the 200 bootstrap recomputations affordable.
    """
    return MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5],
                               class_metrics=False)


def _detail_metric() -> MeanAveragePrecision:
    """
    Full COCO sweep for the secondary numbers: mAP75 and AP small/medium/large
    do not exist at a single IoU threshold, and mAR@300 needs 300 in the
    max-detection list.
    """
    return MeanAveragePrecision(iou_type="bbox", class_metrics=False,
                                max_detection_thresholds=[1, 10, 300])


def _iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return (inter / (aa[:, None] + ab[None, :] - inter + 1e-9)).astype(np.float32)


def _match_greedy(pred_xyxy: np.ndarray, scores: np.ndarray,
                  gt_xyxy: np.ndarray, iou_thr: float = 0.5
                  ) -> Tuple[int, int, int, np.ndarray]:
    """
    Score-ordered greedy matching. -> (tp, fp, fn, matched_iou per TP).
    """
    order = np.argsort(-scores)
    pred_xyxy = pred_xyxy[order]
    iou = _iou_np(pred_xyxy, gt_xyxy)
    used = np.zeros(gt_xyxy.shape[0], dtype=bool)
    tp = 0
    ious: List[float] = []
    for i in range(pred_xyxy.shape[0]):
        if gt_xyxy.shape[0] == 0:
            break
        row = iou[i].copy()
        row[used] = -1.0
        j = int(np.argmax(row)) if row.size else -1
        if j >= 0 and row[j] >= iou_thr:
            used[j] = True
            tp += 1
            ious.append(float(row[j]))
    fp = int(pred_xyxy.shape[0] - tp)
    fn = int(gt_xyxy.shape[0] - tp)
    return tp, fp, fn, np.asarray(ious, dtype=np.float32)


def _ap50_single(pred_xyxy: np.ndarray, scores: np.ndarray,
                 gt_xyxy: np.ndarray) -> float:
    """
    Per-image AP at IoU 0.5, for hard-case ranking only.

    A local implementation rather than 165 torchmetrics instantiations; it is
    used to ORDER images by difficulty, never to report a number.
    """
    n_gt = int(gt_xyxy.shape[0])
    if n_gt == 0:
        return 1.0 if pred_xyxy.shape[0] == 0 else 0.0
    if pred_xyxy.shape[0] == 0:
        return 0.0
    order = np.argsort(-scores)
    p = pred_xyxy[order]
    iou = _iou_np(p, gt_xyxy)
    used = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(p.shape[0], dtype=np.float32)
    fp = np.zeros(p.shape[0], dtype=np.float32)
    for i in range(p.shape[0]):
        row = iou[i].copy()
        row[used] = -1.0
        j = int(np.argmax(row))
        if row[j] >= 0.5:
            used[j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    rec = ctp / float(n_gt)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # 101-point interpolated AP, COCO style.
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        m = prec[rec >= t]
        ap += float(m.max()) if m.size else 0.0
    return ap / 101.0


# ===========================================================================
# shared evaluation
# ===========================================================================

@torch.no_grad()
def evaluate_split(model: NIRDet, loader: DataLoader, cfg: Config,
                   device: Optional[torch.device] = None,
                   score_thresh: Optional[float] = None,
                   detailed: bool = True) -> Tuple[Dict[str, float], dict]:
    """
    Run inference over a loader and compute detection metrics in CANVAS space.

    Used by train.py for selection-split mAP50 and by this script for the
    report split, so the two numbers are never produced by different code.

    ``score_thresh`` defaults to cfg.eval.eval_score_thresh (0.05), which
    integrates essentially the whole PR curve — a deployment threshold here
    would truncate recall and understate mAP.

    Returns (metrics, cache). ``cache`` holds the per-image prediction and GT
    arrays so the bootstrap and the F1 sweep do not re-run the network.
    """
    device = device or next(model.parameters()).device
    st = float(cfg.eval.eval_score_thresh if score_thresh is None
               else score_thresh)
    was_training = model.training
    model.eval()

    m50 = _map50_metric()
    mdet = _detail_metric() if detailed else None

    preds_cache: List[dict] = []
    gts_cache: List[dict] = []
    metas: List[dict] = []

    h, w = int(cfg.data.img_h), int(cfg.data.img_w)

    for imgs, targets, meta in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = model.forward_features(imgs)
        packed = model.head(feats, training_mode=False)
        dets = model.decode_predictions(
            packed, input_size=(h, w), score_thresh=st,
            iou_thresh=cfg.model.nms_iou_thresh, max_det=cfg.model.max_det)

        for i, (boxes, scores) in enumerate(dets):
            pb = boxes.detach().float().cpu()
            ps = scores.detach().float().cpu()
            gt = targets[i].detach().float().cpu()
            if gt.shape[0]:
                cx, cy, bw, bh = gt[:, 0] * w, gt[:, 1] * h, gt[:, 2] * w, gt[:, 3] * h
                gb = torch.stack([cx - bw / 2, cy - bh / 2,
                                  cx + bw / 2, cy + bh / 2], dim=-1)
            else:
                gb = torch.zeros((0, 4))

            p = {"boxes": pb, "scores": ps,
                 "labels": torch.zeros(pb.shape[0], dtype=torch.long)}
            g = {"boxes": gb,
                 "labels": torch.zeros(gb.shape[0], dtype=torch.long)}
            m50.update([p], [g])
            if mdet is not None:
                mdet.update([p], [g])
            preds_cache.append(p)
            gts_cache.append(g)
            metas.append(meta[i] if i < len(meta) else {})

    out: Dict[str, float] = {}
    r50 = m50.compute()
    out["map_50"] = float(r50["map_50"]) if "map_50" in r50 else float(r50["map"])
    if mdet is not None:
        rd = mdet.compute()
        for k in ("map", "map_50", "map_75", "map_small", "map_medium",
                  "map_large", "mar_1", "mar_10", "mar_100"):
            if k in rd:
                out[f"detail_{k}"] = float(rd[k])
        # torchmetrics names the largest max-detection recall mar_100 even when
        # the threshold list ends at 300; with max_detection_thresholds=
        # [1, 10, 300] the third entry IS @300.
        if "mar_100" in rd:
            out["mar_300"] = float(rd["mar_100"])
        out["map_75"] = out.get("detail_map_75", float("nan"))
        out["map_small"] = out.get("detail_map_small", float("nan"))
        out["map_medium"] = out.get("detail_map_medium", float("nan"))
        out["map_large"] = out.get("detail_map_large", float("nan"))

    out["n_images"] = float(len(preds_cache))
    out["n_gt"] = float(sum(int(g["boxes"].shape[0]) for g in gts_cache))
    out["n_pred"] = float(sum(int(p["boxes"].shape[0]) for p in preds_cache))

    if was_training:
        model.train()
    return out, {"preds": preds_cache, "gts": gts_cache, "metas": metas}


# ===========================================================================
# bootstrap CI
# ===========================================================================

def bootstrap_map50(cache: dict, n: int, seed: int = 1234
                    ) -> Tuple[float, float, float]:
    """
    Percentile bootstrap over IMAGES (the sampling unit), not over boxes.

    Resamples image indices with replacement and recomputes mAP50 from the
    cached tensors, so the network runs once regardless of n.
    -> (lo 2.5%, median, hi 97.5%)
    """
    preds, gts = cache["preds"], cache["gts"]
    k = len(preds)
    if k == 0 or n <= 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(int(n)):
        idx = rng.integers(0, k, size=k)
        m = _map50_metric()
        m.update([preds[i] for i in idx], [gts[i] for i in idx])
        r = m.compute()
        vals.append(float(r["map_50"]) if "map_50" in r else float(r["map"]))
    a = np.asarray(vals, dtype=np.float64)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 50.0)),
            float(np.percentile(a, 97.5)))


# ===========================================================================
# threshold sweep
# ===========================================================================

def sweep_deploy_threshold(cache: dict, lo: float = 0.05, hi: float = 0.95,
                           step: float = 0.01, iou_thr: float = 0.5
                           ) -> Tuple[float, dict, List[dict]]:
    """
    Find the score threshold maximising F1 at IoU 0.5 over the whole split.

    This is the number the Pi and the STM32 consume. It is measured on the
    report split, because the deployed system faces held-out data and the
    threshold should be tuned against that, not against the split used to pick
    the checkpoint.
    """
    preds, gts = cache["preds"], cache["gts"]
    pre = [(p["boxes"].numpy(), p["scores"].numpy(), g["boxes"].numpy())
           for p, g in zip(preds, gts)]

    curve: List[dict] = []
    best = {"thresh": float(lo), "f1": -1.0, "precision": 0.0, "recall": 0.0}
    t = float(lo)
    while t <= hi + 1e-9:
        TP = FP = FN = 0
        for pb, ps, gb in pre:
            keep = ps >= t
            tp, fp, fn, _ = _match_greedy(pb[keep], ps[keep], gb, iou_thr)
            TP += tp
            FP += fp
            FN += fn
        prec = TP / max(TP + FP, 1)
        rec = TP / max(TP + FN, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        row = {"thresh": round(t, 4), "precision": prec, "recall": rec,
               "f1": f1, "tp": TP, "fp": FP, "fn": FN}
        curve.append(row)
        if f1 > best["f1"]:
            best = dict(row)
        t += float(step)
    return float(best["thresh"]), best, curve


# ===========================================================================
# hard cases
# ===========================================================================

def save_hard_cases(cache: dict, cfg: Config, score_thresh: float,
                    max_cases: int) -> List[dict]:
    """
    Save the lowest-AP50 images with predicted (green) and GT (blue) boxes.

    Ranked by the local per-image AP50 rather than by loss, because the point
    is to look at what the detector gets wrong, not at what the assigner found
    difficult to assign.
    """
    import cv2
    from dataset import preprocess_frame, load_flat_field

    out_dir = os.path.join(cfg.eval.out_dir, "hard_cases")
    os.makedirs(out_dir, exist_ok=True)
    ff = load_flat_field(cfg.aug.flat_field_path)

    scored: List[Tuple[float, int]] = []
    for i, (p, g) in enumerate(zip(cache["preds"], cache["gts"])):
        s = p["scores"].numpy()
        keep = s >= score_thresh
        ap = _ap50_single(p["boxes"].numpy()[keep], s[keep],
                          g["boxes"].numpy())
        scored.append((ap, i))
    scored.sort(key=lambda x: x[0])

    written: List[dict] = []
    for ap, i in scored[: int(max_cases)]:
        meta = cache["metas"][i] if i < len(cache["metas"]) else {}
        path = meta.get("img_path")
        if not path or not os.path.isfile(path):
            continue
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        canvas, *_ = preprocess_frame(
            raw, cfg.data.img_h, cfg.data.img_w,
            clahe_enabled=cfg.aug.clahe_enabled,
            clahe_clip=cfg.aug.clahe_clip, clahe_grid=cfg.aug.clahe_grid,
            flat_field=ff)
        vis = cv2.cvtColor((canvas * 255.0).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)

        for b in cache["gts"][i]["boxes"].numpy().astype(int):
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (255, 128, 0), 1)
        pb = cache["preds"][i]["boxes"].numpy()
        ps = cache["preds"][i]["scores"].numpy()
        for b, s in zip(pb[ps >= score_thresh].astype(int),
                        ps[ps >= score_thresh]):
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 220, 0), 1)
            cv2.putText(vis, f"{s:.2f}", (b[0], max(10, b[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1)
        cv2.putText(vis, f"AP50={ap:.3f}  {os.path.basename(path)}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        name = f"{ap:.3f}_{os.path.splitext(os.path.basename(path))[0]}.png"
        cv2.imwrite(os.path.join(out_dir, name), vis)
        written.append({"image": os.path.basename(path), "ap50": float(ap)})
    return written


# ===========================================================================
# checkpoint
# ===========================================================================

def load_checkpoint_into(model: NIRDet, path: str, cfg: Config,
                         device: torch.device) -> dict:
    """
    Load CKPT_DEPLOY_KEY (the EMA weights) and verify the decode contract.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ck = torch.load(path, map_location=device, weights_only=False)

    got = str(ck.get("deploy_contract_hash", ""))
    want = cfg.deploy_contract()["hash"]
    if got != want:
        raise RuntimeError(
            f"DEPLOY CONTRACT MISMATCH\n"
            f"  checkpoint : {got or '<absent>'}\n"
            f"  config     : {want}\n"
            f"The checkpoint was trained with a different geometry. The "
            f"contract covers num_classes, canvas h/w, strides, "
            f"DECODE_OFFSET_SCALE/BIAS, REG_LOG_CLAMP_MIN/MAX and the "
            f"blobs-per-level layout. Evaluating across a mismatch decodes "
            f"every box with the wrong grid or the wrong exp range, so the "
            f"mAP it reports is meaningless.\n"
            f"Either evaluate with the config that trained it (checkpoint["
            f"'cfg'] holds a full dump) or retrain at the current geometry.")

    # Alias list in preference order. The deploy key was renamed
    # deploy_state -> deploy_state_dict; without aliases every checkpoint
    # written before the rename fails with a bare KeyError.
    from train import CKPT_LIVE_KEY
    sd, which = None, None
    for key in (CKPT_DEPLOY_KEY, "deploy_state", CKPT_LIVE_KEY,
                "model_state", "model_state_dict", "state_dict"):
        v = ck.get(key)
        if isinstance(v, dict) and v:
            sd = v.get("weights", v) if "weights" in v else v
            which = key
            break
    if sd is None:
        raise RuntimeError(
            f"no weights found in {path}; keys = {sorted(ck)}")
    if which != CKPT_DEPLOY_KEY:
        print(f"[ckpt] no '{CKPT_DEPLOY_KEY}'; falling back to '{which}'")
    model.load_state_dict(sd)
    print(f"[ckpt] {path}: loaded '{which}' from epoch "
          f"{ck.get('epoch', '?')}, best {cfg.eval.select_split} mAP50 "
          f"{float(ck.get('best_map50', float('nan'))):.4f}")
    if ck.get("eaa_proj_bias") is not None:
        cfg.model.eaa_proj_bias = float(ck["eaa_proj_bias"])
        print(f"[ckpt] eaa_proj_bias {cfg.model.eaa_proj_bias:+.5f}")
    return ck


# ===========================================================================
# main
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate NIRDet-Lite (fp32) and measure "
                    "deploy_score_thresh.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--profile", default=None,
                    help="dataset profile YAML; deploy_score_thresh is "
                         "written back into it")
    ap.add_argument("--split", default=None,
                    help="override cfg.eval.report_split")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--no-hard-cases", action="store_true")
    args = ap.parse_args()

    cfg = get_config()
    profile = None
    if args.profile:
        from dataset_profiles import DatasetProfile
        profile = DatasetProfile.load(args.profile)
        profile.apply(cfg)
    validate_config(cfg, verbose=False)

    split = args.split or cfg.eval.report_split
    if split == cfg.eval.select_split and not args.split:
        raise RuntimeError("report split equals selection split")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_nirdet(cfg).to(device)

    ckpt_path = args.checkpoint or os.path.join(cfg.train.checkpoint_dir,
                                                "best.pth")
    load_checkpoint_into(model, ckpt_path, cfg, device)
    model.eval()

    loader, ds = build_dataloader(cfg, split, batch_size=1, shuffle=False,
                                  augment=False, num_workers=0)
    print(f"[eval] report split '{split}': {len(ds)} images, "
          f"{ds.n_boxes} boxes")
    print(f"[eval] selection split was '{cfg.eval.select_split}'; the "
          f"headline number below is from '{split}', which the checkpoint "
          f"selection never saw.")

    metrics, cache = evaluate_split(model, loader, cfg, device=device,
                                    score_thresh=cfg.eval.eval_score_thresh,
                                    detailed=True)

    ci = (float("nan"),) * 3
    if not args.no_bootstrap and int(cfg.eval.bootstrap_n) > 0:
        print(f"[eval] bootstrap {cfg.eval.bootstrap_n} resamples over "
              f"{len(cache['preds'])} images...")
        ci = bootstrap_map50(cache, int(cfg.eval.bootstrap_n),
                             int(cfg.eval.bootstrap_seed))

    thr, best, curve = sweep_deploy_threshold(cache)

    hard: List[dict] = []
    if not args.no_hard_cases:
        hard = save_hard_cases(cache, cfg, thr, int(cfg.eval.max_hard_cases))

    # ---- per-level positives, from the training history ----
    per_level: Dict[str, float] = {}
    hist_path = os.path.join(cfg.train.checkpoint_dir, "history.json")
    if os.path.isfile(hist_path):
        import json
        with open(hist_path, "r", encoding="utf-8") as fh:
            hist = json.load(fh)
        if hist:
            last = hist[-1]
            for k, v in last.items():
                if k.startswith("n_pos"):
                    per_level[k] = float(v)

    # ---- report ----
    print("=" * 70)
    print(f"  NIRDet-Lite fp32 evaluation — split '{split}'")
    print("=" * 70)
    print(f"  mAP50            : {metrics['map_50']:.4f}")
    if np.isfinite(ci[0]):
        print(f"  95% CI           : [{ci[0]:.4f}, {ci[2]:.4f}]  "
              f"(median {ci[1]:.4f}, {cfg.eval.bootstrap_n} resamples)")
    print(f"  mAP50-95         : {metrics.get('detail_map', float('nan')):.4f}")
    print(f"  mAP75            : {metrics.get('map_75', float('nan')):.4f}")
    print(f"  mAR@300          : {metrics.get('mar_300', float('nan')):.4f}")
    print(f"  AP small         : {metrics.get('map_small', float('nan')):.4f}")
    print(f"  AP medium        : {metrics.get('map_medium', float('nan')):.4f}")
    print(f"  AP large         : {metrics.get('map_large', float('nan')):.4f}")
    print(f"  baseline mAP50   : {cfg.eval.baseline_map50:.4f} "
          f"(YOLO11n fine-tune reference)")
    print(f"  delta vs base    : {metrics['map_50'] - cfg.eval.baseline_map50:+.4f}")
    if per_level:
        print("  n_pos (last training epoch):")
        for k in sorted(per_level):
            print(f"    {k:<10} {per_level[k]:.1f}")
        zero_lvls = [k for k in per_level
                     if k.startswith("n_pos_l") and per_level[k] < 1.0]
        if zero_lvls:
            print(f"    ! {', '.join(zero_lvls)} near zero: that level is "
                  f"doing nothing. Confirm with train.py --p5-ablate.")
    print(f"  deploy_score_thresh : {thr:.3f}  "
          f"(F1 {best['f1']:.4f}, P {best['precision']:.4f}, "
          f"R {best['recall']:.4f}, TP {best['tp']} FP {best['fp']} "
          f"FN {best['fn']})")
    print("=" * 70)

    # ---- write back ----
    if profile is not None and args.profile:
        profile.update_deploy_thresh(args.profile, thr)
        print(f"[profile] deploy_score_thresh {thr:.3f} written to "
              f"{args.profile}; live_nirdet.py --profile will read it")
    else:
        print("[profile] no --profile given, so deploy_score_thresh was not "
              "persisted. live_nirdet.py requires either --profile or an "
              "explicit --score-thresh.")

    os.makedirs(cfg.eval.out_dir, exist_ok=True)
    summary = {
        "split": split,
        "checkpoint": os.path.abspath(ckpt_path),
        "deploy_contract_hash": cfg.deploy_contract()["hash"],
        "canvas": f"{cfg.data.img_h}x{cfg.data.img_w}",
        "strides": list(cfg.model.strides),
        "n_images": int(metrics["n_images"]),
        "n_gt": int(metrics["n_gt"]),
        "n_pred": int(metrics["n_pred"]),
        "map50": float(metrics["map_50"]),
        "map50_ci_lo": None if not np.isfinite(ci[0]) else float(ci[0]),
        "map50_ci_hi": None if not np.isfinite(ci[2]) else float(ci[2]),
        "map50_95": float(metrics.get("detail_map", float("nan"))),
        "map75": float(metrics.get("map_75", float("nan"))),
        "mar300": float(metrics.get("mar_300", float("nan"))),
        "ap_small": float(metrics.get("map_small", float("nan"))),
        "ap_medium": float(metrics.get("map_medium", float("nan"))),
        "ap_large": float(metrics.get("map_large", float("nan"))),
        "baseline_map50": float(cfg.eval.baseline_map50),
        "deploy_score_thresh": float(thr),
        "best_f1": float(best["f1"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "n_pos_history": per_level,
        "hard_cases": hard,
        "f1_curve": curve,
    }
    out_yaml = os.path.join(cfg.eval.out_dir, "eval_summary.yaml")
    with open(out_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False, default_flow_style=False)
    print(f"[out] {out_yaml}")
    if hard:
        print(f"[out] {len(hard)} hard cases in "
              f"{os.path.join(cfg.eval.out_dir, 'hard_cases')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
