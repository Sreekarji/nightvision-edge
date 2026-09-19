"""
evaluate.py — fp32 evaluation and deploy-threshold measurement
===============================================================
    python evaluate.py --checkpoint checkpoints/best.pth \
                       --profile datasets/<name>.yaml

SELECTION SPLIT != REPORT SPLIT. best.pth is chosen by maximising mAP50 on
cfg.eval.select_split across ~one evaluation per epoch, so reporting the
headline number on that split is optimistically biased by construction. The
headline comes from cfg.eval.report_split.

deploy_score_thresh is measured here as the threshold maximising F1 at IoU 0.5
on the report split and written back into the profile YAML. No deployment file
contains a threshold literal.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("evaluate.py needs PyYAML: pip install pyyaml") from exc

from config import (CKPT_DEPLOY_KEY, CKPT_LIVE_KEY, Config, get_config,
                    validate_config)
from dataset import build_dataloader
from model import NIRDet, build_nirdet

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
    """Headline / bootstrap metric: IoU 0.5 only, single class."""
    return MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5],
                                class_metrics=False)


def _detail_metric() -> MeanAveragePrecision:
    """Full COCO sweep for mAP75 and AP small/medium/large."""
    return MeanAveragePrecision(iou_type="bbox", class_metrics=False,
                                max_detection_thresholds=[1, 10, 300])


def map50_value(res: dict, report: bool = True) -> float:
    """
    Extract mAP50 defensively (F30), reporting which key supplied it (F35).

    Some torchmetrics versions populate map_50 only for the default
    10-threshold sweep and otherwise set it to -1.0. A silent -1 would be
    reported as a negative mAP and would select the WORST checkpoint.
    """
    if "map_50" in res and float(res["map_50"]) >= 0:
        v = float(res["map_50"])
        _map_key_used = "map_50"
    else:
        v = float(res.get("map", -1.0))
        _map_key_used = "map (fallback)"
    if v < 0.0:
        raise RuntimeError(
            f"torchmetrics returned no usable mAP50 (map_50 and map are both "
            f"negative): {dict(res)}. Check the torchmetrics version against "
            f"the single-IoU-threshold configuration in _map50_metric().")
    if report:
        print(f"[eval] mAP key used: {_map_key_used} = {v:.4f}")
    return v


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
    """Score-ordered greedy matching -> (tp, fp, fn, matched IoU per TP)."""
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
    return tp, int(pred_xyxy.shape[0] - tp), int(gt_xyxy.shape[0] - tp), \
        np.asarray(ious, dtype=np.float32)


def _ap50_single(pred_xyxy: np.ndarray, scores: np.ndarray,
                 gt_xyxy: np.ndarray) -> float:
    """Per-image AP at IoU 0.5, for hard-case RANKING only."""
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
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / float(n_gt)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
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
                   detailed: bool = True,
                   return_cache: bool = False) -> Tuple[Dict[str, float], dict]:
    """
    Run inference over a loader and compute detection metrics in CANVAS space.

    Used by train.py for selection-split mAP50 and by this script for the
    report split, so the two numbers come from identical code.

    F29: ``return_cache=False`` (the default, used for in-training calls)
    skips building the per-image (boxes, scores, gt) cache — a ~3x peak-memory
    reduction on large splits. The detailed path (report split, called with
    detailed=True) always builds it.
    """
    device = device or next(model.parameters()).device
    st = float(cfg.eval.eval_score_thresh if score_thresh is None
               else score_thresh)
    was_training = model.training
    # F28: an exception during evaluation must not leave the model stuck in
    # eval() mode for the rest of training.
    model.eval()
    try:
        m50 = _map50_metric()
        mdet = _detail_metric() if detailed else None

        build_cache = detailed or return_cache
        preds_cache: List[dict] = []
        gts_cache: List[dict] = []
        metas: List[dict] = []
        h, w = int(cfg.data.img_h), int(cfg.data.img_w)
        n_images = 0
        n_gt_total = 0
        n_pred_total = 0

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
                    cx, cy = gt[:, 0] * w, gt[:, 1] * h
                    bw, bh = gt[:, 2] * w, gt[:, 3] * h
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
                n_images += 1
                n_gt_total += int(gb.shape[0])
                n_pred_total += int(pb.shape[0])
                if build_cache:
                    preds_cache.append(p)
                    gts_cache.append(g)
                    metas.append(meta[i] if i < len(meta) else {})

        if n_gt_total == 0:
            print(f"[eval] WARNING: split has 0 GT boxes — "
                  f"mAP is undefined, returning nan.")
            out_zero: Dict[str, float] = {"map_50": float("nan"),
                                          "n_images": float(n_images),
                                          "n_gt": 0.0,
                                          "n_pred": float(n_pred_total)}
            return out_zero, {"preds": preds_cache, "gts": gts_cache,
                               "metas": metas}
        out: Dict[str, float] = {"map_50": map50_value(m50.compute())}
        if mdet is not None:
            rd = mdet.compute()
            for k in ("map", "map_50", "map_75", "map_small", "map_medium",
                      "map_large", "mar_1", "mar_10", "mar_100"):
                if k in rd:
                    out[f"detail_{k}"] = float(rd[k])
            # torchmetrics names the largest max-detection recall mar_100 even
            # when the threshold list ends at 300; the third entry IS @300.
            if "mar_100" in rd:
                out["mar_300"] = float(rd["mar_100"])
            for k in ("map_75", "map_small", "map_medium", "map_large"):
                out[k] = out.get(f"detail_{k}", float("nan"))

        out["n_images"] = float(n_images)
        out["n_gt"] = float(n_gt_total)
        out["n_pred"] = float(n_pred_total)
    finally:
        if was_training:
            model.train()
    return out, {"preds": preds_cache, "gts": gts_cache, "metas": metas}


# ===========================================================================
# bootstrap CI
# ===========================================================================

def bootstrap_map50(cache: dict, n: int, seed: int = 1234
                    ) -> Tuple[float, float, float]:
    """Percentile bootstrap over IMAGES (the sampling unit), not over boxes."""
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
        vals.append(map50_value(m.compute(), report=False))
    a = np.asarray(vals, dtype=np.float64)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 50.0)),
            float(np.percentile(a, 97.5)))


# ===========================================================================
# threshold sweep
# ===========================================================================

def sweep_deploy_threshold(cache: dict, lo: float = 0.05, hi: float = 0.95,
                           step: float = 0.01, iou_thr: float = 0.5
                           ) -> Tuple[float, dict, List[dict]]:
    """Find the score threshold maximising F1 at IoU 0.5 over the split."""
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
    Save the hardest images with predicted (green) and GT (blue) boxes.

    Ranked by per-image AP50 ascending, then by MOST predictions first (the
    second sort key is -n_pred sorted ascending) (F32/F17), so among the
    AP-0.0 images the "hallucinated everything" failures sort ahead of the
    "missed everything" ones instead of collapsing to the same rank. Both
    failure modes appear; only their order within an AP tier is set here.
    """
    import cv2
    from dataset import load_flat_field, preprocess_frame

    out_dir = os.path.join(cfg.eval.out_dir, "hard_cases")
    os.makedirs(out_dir, exist_ok=True)
    ff = load_flat_field(cfg.aug.flat_field_path)

    scored: List[Tuple[float, int, int]] = []
    for i, (p, g) in enumerate(zip(cache["preds"], cache["gts"])):
        s = p["scores"].numpy()
        keep = s >= score_thresh
        ap = _ap50_single(p["boxes"].numpy()[keep], s[keep], g["boxes"].numpy())
        scored.append((ap, -int(keep.sum()), i))
    scored.sort(key=lambda x: (x[0], x[1]))

    written: List[dict] = []
    for idx, (ap, neg_npred, i) in enumerate(scored[: int(max_cases)]):
        meta = cache["metas"][i] if i < len(cache["metas"]) else {}
        path = meta.get("img_path")
        if not path or not os.path.isfile(path):
            continue
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        canvas, *_ = preprocess_frame(
            raw, cfg.data.img_h, cfg.data.img_w,
            clahe_enabled=cfg.aug.clahe_enabled, clahe_clip=cfg.aug.clahe_clip,
            clahe_grid=cfg.aug.clahe_grid, flat_field=ff)
        vis = cv2.cvtColor((canvas * 255.0).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)

        # Explicit int() casts: some OpenCV 4.5-4.7 builds reject numpy.int64
        # in point tuples with "Can't parse 'pt1'" (F31).
        for b in cache["gts"][i]["boxes"].numpy().astype(int).tolist():
            cv2.rectangle(vis, (int(b[0]), int(b[1])),
                          (int(b[2]), int(b[3])), (255, 128, 0), 1)
        pb = cache["preds"][i]["boxes"].numpy()
        ps = cache["preds"][i]["scores"].numpy()
        for b, s in zip(pb[ps >= score_thresh].astype(int).tolist(),
                        ps[ps >= score_thresh]):
            cv2.rectangle(vis, (int(b[0]), int(b[1])),
                          (int(b[2]), int(b[3])), (0, 220, 0), 1)
            cv2.putText(vis, f"{float(s):.2f}", (int(b[0]), max(10, int(b[1]) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1)
        cv2.putText(vis, f"AP50={ap:.3f} npred={-neg_npred} "
                         f"{os.path.basename(path)}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        stem = os.path.splitext(os.path.basename(path))[0]
        # Include a counter to prevent collisions when two images share a stem
        # name (e.g. different subdirectories both contain 0001.png) (F33).
        name = f"{idx:05d}_{ap:.3f}_{stem}.png"
        cv2.imwrite(os.path.join(out_dir, name), vis)
        written.append({"image": os.path.basename(path), "ap50": float(ap),
                        "n_pred": int(-neg_npred)})
    return written


# ===========================================================================
# checkpoint
# ===========================================================================

def load_checkpoint_into(model: NIRDet, path: str, cfg: Config,
                         device: torch.device) -> dict:
    """Load CKPT_DEPLOY_KEY (the EMA weights) and verify the decode contract."""
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
            f"The checkpoint was trained with different geometry or "
            f"preprocessing. The contract covers num_classes, canvas h/w, "
            f"strides, the EAA edge stride/pool factor, "
            f"DECODE_OFFSET_SCALE/BIAS, REG_LOG_CLAMP_MIN/MAX, MIN_BOX_PX, "
            f"the blob layout AND the CLAHE / flat-field parameters. "
            f"Evaluating across a mismatch decodes every box with the wrong "
            f"grid, the wrong exp range or the wrong input distribution.\n"
            f"Either evaluate with the config that trained it "
            f"(checkpoint['cfg'] holds a full dump) or retrain.")

    sd, which = None, None
    for key in (CKPT_DEPLOY_KEY, "deploy_state", CKPT_LIVE_KEY,
                "model_state", "model_state_dict", "state_dict"):
        v = ck.get(key)
        if isinstance(v, dict) and v:
            sd = v.get("weights", v) if "weights" in v else v
            which = key
            break
    if sd is None:
        raise RuntimeError(f"no weights found in {path}; keys = {sorted(ck)}")
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
    ap.add_argument("--split", default=None, help="override report_split")
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
    if args.split and split == cfg.eval.select_split:
        print(f"[eval] WARNING: --split '{split}' equals select_split "
              f"(cfg.eval.select_split='{cfg.eval.select_split}'); "
              f"deploy-threshold P/R/F1 are IN-SAMPLE and will be optimistic.")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_nirdet(cfg).to(device)

    ckpt_path = args.checkpoint or os.path.join(cfg.train.checkpoint_dir,
                                                "best.pth")
    load_checkpoint_into(model, ckpt_path, cfg, device)
    model.eval()

    loader, ds = build_dataloader(cfg, split, batch_size=1, shuffle=False,
                                  augment=False, num_workers=0)
    print(f"[eval] report split '{split}': {len(ds)} images, {ds.n_boxes} boxes")
    print(f"[eval] selection split was '{cfg.eval.select_split}'; the "
          f"headline number below is from '{split}'.")

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
        # Rendering must never destroy an evaluation after the expensive part.
        try:
            hard = save_hard_cases(cache, cfg, thr, int(cfg.eval.max_hard_cases))
        except Exception as exc:
            print(f"[eval] hard-case rendering failed ({type(exc).__name__}: "
                  f"{exc}); the metrics below are unaffected")

    per_level: Dict[str, float] = {}
    hist_path = os.path.join(cfg.train.checkpoint_dir, "history.json")
    if os.path.isfile(hist_path):
        import json
        with open(hist_path, "r", encoding="utf-8") as fh:
            hist = json.load(fh)
        if hist:
            for k, v in hist[-1].items():
                if k.startswith("n_pos"):
                    per_level[k] = float(v)

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
    if cfg.eval.baseline_map50 is not None:
        src = cfg.eval.baseline_source or \
            f"reference from profile '{cfg.data.profile_name or '?'}'"
        print(f"  baseline mAP50   : {cfg.eval.baseline_map50:.4f}  ({src})")
        print(f"  delta vs base    : "
              f"{metrics['map_50'] - cfg.eval.baseline_map50:+.4f}")
    if per_level:
        # F32/fix-95: assert every expected key is present so a key-name
        # divergence raises clearly instead of silently printing 0.
        missing_keys = [f"n_pos_l{i}" for i in range(len(cfg.model.strides))
                        if f"n_pos_l{i}" not in per_level]
        assert not missing_keys, \
            f"per_level history is missing keys: {missing_keys}. " \
            f"The checkpoint was saved with a different number of strides " \
            f"or a different key naming scheme."
        cells = "  ".join(
            f"l{i} {int(per_level[f'n_pos_l{i}']):>5d}"
            for i in range(len(cfg.model.strides)))
        print(f"[eval] n_pos per level:  {cells}")
        print("  n_pos (last training epoch):")
        for k in sorted(per_level):
            print(f"    {k:<14} {per_level[k]:.1f}")
        zero_lvls = [k for k in per_level
                     if k.startswith("n_pos_l") and per_level[k] < 1.0]
        if zero_lvls:
            print(f"    ! {', '.join(zero_lvls)} near zero: that level is "
                  f"doing nothing. Confirm with train.py --p5-ablate.")
    print(f"  deploy_score_thresh : {thr:.3f}  (F1 {best['f1']:.4f}, "
          f"P {best['precision']:.4f}, R {best['recall']:.4f}, "
          f"TP {best['tp']} FP {best['fp']} FN {best['fn']})")
    print("=" * 70)

    if profile is not None and args.profile:
        profile.update_deploy_thresh(args.profile, thr)
        print(f"[profile] deploy_score_thresh {thr:.3f} written to "
              f"{args.profile}; live_nirdet.py --profile will read it")
    else:
        print("[profile] no --profile given, so deploy_score_thresh was not "
              "persisted. live_nirdet.py requires --profile or "
              "--score-thresh.")

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
        "baseline_map50": (float(cfg.eval.baseline_map50)
                           if cfg.eval.baseline_map50 is not None else None),
        "baseline_source": cfg.eval.baseline_source or None,
        "deploy_score_thresh": float(thr),
        "best_f1": float(best["f1"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "n_pos_history": per_level,
        "hard_cases": hard,
        "f1_curve": curve,
    }
    out_yaml = os.path.join(cfg.eval.out_dir, "eval_summary.yaml")
    # F34: atomic write — a crash mid-dump must not leave a truncated summary.
    tmp = out_yaml + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False, default_flow_style=False)
    os.replace(tmp, out_yaml)
    print(f"[out] {out_yaml}")
    if hard:
        print(f"[out] {len(hard)} hard cases in "
              f"{os.path.join(cfg.eval.out_dir, 'hard_cases')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
