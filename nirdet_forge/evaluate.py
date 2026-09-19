"""
evaluate.py — fp32 evaluation and deploy-threshold measurement
===============================================================
    python evaluate.py --checkpoint checkpoints/best.pth \
                       --profile datasets/<name>.yaml

SELECTION SPLIT != REPORT SPLIT. deploy_score_thresh is measured here (best F1
at IoU 0.5 on the report split) and written back into the profile YAML.

SCALE: the bootstrap cost is O(bootstrap_n * n_images). cfg.eval.
bootstrap_max_updates caps the product, so a 20 000-image split reduces the
resample count instead of running for hours. The threshold sweep is vectorised
per image rather than re-matching every image at every one of 91 thresholds.
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


def _map50_metric() -> MeanAveragePrecision:
    return MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5],
                                class_metrics=False)


def _detail_metric() -> MeanAveragePrecision:
    return MeanAveragePrecision(iou_type="bbox", class_metrics=False,
                                max_detection_thresholds=[1, 10, 300])


def map50_value(res: dict, report: bool = True) -> float:
    """Extract mAP50 defensively; a silent -1 would select the WORST model."""
    if "map_50" in res and float(res["map_50"]) >= 0:
        v = float(res["map_50"])
        key = "map_50"
    else:
        v = float(res.get("map", -1.0))
        key = "map (fallback)"
    if v < 0.0:
        raise RuntimeError(
            f"torchmetrics returned no usable mAP50: {dict(res)}. Check the "
            f"torchmetrics version against _map50_metric().")
    if report:
        print(f"[eval] mAP key used: {key} = {v:.4f}")
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





def _match_scores(pred_xyxy: np.ndarray, scores: np.ndarray,
                  gt_xyxy: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """
    Per-image greedy matching ONCE, returning the score of each TP.

    The sweep then only needs to count how many TP scores and how many total
    predictions exceed each threshold — O(images + thresholds) instead of
    O(images * thresholds) greedy matchings.

    Greedy matching is threshold-monotone here: raising the threshold only
    removes the lowest-scoring predictions, and those are matched last.
    """
    if pred_xyxy.shape[0] == 0 or gt_xyxy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    order = np.argsort(-scores)
    p, s = pred_xyxy[order], scores[order]
    iou = _iou_np(p, gt_xyxy)
    used = np.zeros(gt_xyxy.shape[0], dtype=bool)
    tp_scores: List[float] = []
    for i in range(p.shape[0]):
        row = iou[i].copy()
        row[used] = -1.0
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            used[j] = True
            tp_scores.append(float(s[i]))
    return np.asarray(tp_scores, dtype=np.float32)


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


@torch.no_grad()
def evaluate_split(model: NIRDet, loader: DataLoader, cfg: Config,
                   device: Optional[torch.device] = None,
                   score_thresh: Optional[float] = None,
                   detailed: bool = True,
                   return_cache: bool = False) -> Tuple[Dict[str, float], dict]:
    """Inference over a loader + detection metrics in CANVAS space."""
    device = device or next(model.parameters()).device
    st = float(cfg.eval.eval_score_thresh if score_thresh is None
               else score_thresh)
    was_training = model.training
    model.eval()
    try:
        m50 = _map50_metric()
        mdet = _detail_metric() if detailed else None

        build_cache = detailed or return_cache
        preds_cache: List[dict] = []
        gts_cache: List[dict] = []
        metas: List[dict] = []
        h, w = int(cfg.data.img_h), int(cfg.data.img_w)
        n_images = n_gt_total = n_pred_total = 0

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
            print("[eval] WARNING: split has 0 GT boxes — mAP undefined (nan)")
            return ({"map_50": float("nan"), "n_images": float(n_images),
                     "n_gt": 0.0, "n_pred": float(n_pred_total)},
                    {"preds": preds_cache, "gts": gts_cache, "metas": metas})

        out: Dict[str, float] = {"map_50": map50_value(m50.compute())}
        if mdet is not None:
            rd = mdet.compute()
            for k in ("map", "map_50", "map_75", "map_small", "map_medium",
                      "map_large", "mar_1", "mar_10", "mar_100"):
                if k in rd:
                    out[f"detail_{k}"] = float(rd[k])
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


def bootstrap_map50(cache: dict, n: int, seed: int = 1234,
                    max_updates: int = 200_000
                    ) -> Tuple[float, float, float, int]:
    """
    Percentile bootstrap over IMAGES. Returns (lo, median, hi, n_used).

    ``max_updates`` caps resamples * images so the CI stays affordable on a
    large report split instead of quietly costing hours.
    """
    preds, gts = cache["preds"], cache["gts"]
    k = len(preds)
    if k == 0 or n <= 0:
        return (float("nan"),) * 3 + (0,)
    n_used = int(n)
    if max_updates > 0:
        n_used = max(1, min(n_used, int(max_updates // max(1, k))))
    if n_used < n:
        print(f"[eval] bootstrap reduced {n} -> {n_used} resamples to stay "
              f"inside bootstrap_max_updates={max_updates} on {k} images")
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_used):
        idx = rng.integers(0, k, size=k)
        m = _map50_metric()
        m.update([preds[i] for i in idx], [gts[i] for i in idx])
        vals.append(map50_value(m.compute(), report=False))
    a = np.asarray(vals, dtype=np.float64)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 50.0)),
            float(np.percentile(a, 97.5)), n_used)


def sweep_deploy_threshold(cache: dict, lo: float = 0.05, hi: float = 0.95,
                           step: float = 0.01, iou_thr: float = 0.5
                           ) -> Tuple[float, dict, List[dict]]:
    """Threshold maximising F1 at IoU 0.5 — one greedy match per image."""
    preds, gts = cache["preds"], cache["gts"]
    thresholds = np.arange(float(lo), float(hi) + 1e-9, float(step),
                           dtype=np.float64)
    tp = np.zeros(thresholds.shape[0], dtype=np.int64)
    npred = np.zeros(thresholds.shape[0], dtype=np.int64)
    n_gt = 0
    for p, g in zip(preds, gts):
        pb = p["boxes"].numpy()
        ps = p["scores"].numpy()
        gb = g["boxes"].numpy()
        n_gt += int(gb.shape[0])
        if ps.shape[0]:
            npred += np.searchsorted(np.sort(ps), thresholds,
                                     side="left").astype(np.int64) * -1 \
                     + ps.shape[0]
        tps = _match_scores(pb, ps, gb, iou_thr)
        if tps.shape[0]:
            tp += np.searchsorted(np.sort(tps), thresholds,
                                  side="left").astype(np.int64) * -1 \
                  + tps.shape[0]

    curve: List[dict] = []
    best = {"thresh": float(lo), "f1": -1.0, "precision": 0.0, "recall": 0.0,
            "tp": 0, "fp": 0, "fn": n_gt}
    for i, t in enumerate(thresholds):
        TP = int(tp[i])
        FP = int(npred[i] - tp[i])
        FN = int(n_gt - TP)
        prec = TP / max(TP + FP, 1)
        rec = TP / max(TP + FN, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        row = {"thresh": round(float(t), 4), "precision": prec, "recall": rec,
               "f1": f1, "tp": TP, "fp": FP, "fn": FN}
        curve.append(row)
        if f1 > best["f1"]:
            best = dict(row)
    return float(best["thresh"]), best, curve


def save_hard_cases(cache: dict, cfg: Config, score_thresh: float,
                    max_cases: int) -> List[dict]:
    """Hardest images (AP50 ascending, then most predictions first)."""
    import cv2
    # preprocess.py, not dataset.py: rendering needs no torch.
    from preprocess import load_flat_field, preprocess_frame

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
        cv2.imwrite(os.path.join(out_dir, f"{idx:05d}_{ap:.3f}_{stem}.png"), vis)
        written.append({"image": os.path.basename(path), "ap50": float(ap),
                        "n_pred": int(-neg_npred)})
    return written


def load_checkpoint_into(model: NIRDet, path: str, cfg: Config,
                         device: torch.device) -> dict:
    """Load CKPT_DEPLOY_KEY (EMA weights) and verify the decode contract."""
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
            f"preprocessing (canvas, strides, EAA edge stride/pool factor, "
            f"decode constants, blob layout, CLAHE / flat-field). Evaluate "
            f"with the config that trained it (checkpoint['cfg']) or retrain.")

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
    # eaa_proj_bias must be adopted BEFORE load_state_dict so cfg and weights
    # describe the same calibration.
    if ck.get("eaa_proj_bias") is not None:
        cfg.model.eaa_proj_bias = float(ck["eaa_proj_bias"])
        print(f"[ckpt] eaa_proj_bias {cfg.model.eaa_proj_bias:+.5f}")
    model.load_state_dict(sd)
    print(f"[ckpt] {path}: loaded '{which}' from epoch "
          f"{ck.get('epoch', '?')}, best {cfg.eval.select_split} mAP50 "
          f"{float(ck.get('best_map50', float('nan'))):.4f}")
    return ck


def _latest_best(ckpt_dir: str) -> str:
    """train.py writes <ckpt_dir>/run_<timestamp>/best.pth; run_ids sort
    chronologically because they are %Y%m%d_%H%M%S."""
    import glob
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "run_*", "best.pth")))
    flat = os.path.join(ckpt_dir, "best.pth")
    if os.path.isfile(flat):
        cands.append(flat)
    if not cands:
        raise FileNotFoundError(
            f"no checkpoint under {ckpt_dir!r}. train.py writes "
            f"{ckpt_dir}/run_<timestamp>/best.pth. Pass --checkpoint "
            f"explicitly, or check that a run completed a validation epoch.")
    return cands[-1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate NIRDet-Forge (fp32) and measure deploy_score_thresh.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--profile", default=None)
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
        print(f"[eval] WARNING: --split '{split}' equals select_split; "
              f"deploy-threshold P/R/F1 are IN-SAMPLE and optimistic.")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_nirdet(cfg).to(device)

    ckpt_path = args.checkpoint or _latest_best(cfg.train.checkpoint_dir)
    load_checkpoint_into(model, ckpt_path, cfg, device)
    model.eval()

    loader, ds = build_dataloader(cfg, split, batch_size=1, shuffle=False,
                                  augment=False, num_workers=0)
    print(f"[eval] report split '{split}': {len(ds)} images, {ds.n_boxes} boxes")

    metrics, cache = evaluate_split(model, loader, cfg, device=device,
                                    score_thresh=cfg.eval.eval_score_thresh,
                                    detailed=True)

    ci = (float("nan"),) * 3
    n_boot = 0
    if not args.no_bootstrap and int(cfg.eval.bootstrap_n) > 0:
        lo, med, hi, n_boot = bootstrap_map50(
            cache, int(cfg.eval.bootstrap_n), int(cfg.eval.bootstrap_seed),
            max_updates=int(cfg.eval.bootstrap_max_updates))
        ci = (lo, med, hi)

    thr, best, curve = sweep_deploy_threshold(cache)

    hard: List[dict] = []
    if not args.no_hard_cases:
        try:
            hard = save_hard_cases(cache, cfg, thr, int(cfg.eval.max_hard_cases))
        except Exception as exc:
            print(f"[eval] hard-case rendering failed ({type(exc).__name__}: "
                  f"{exc}); the metrics below are unaffected")

    per_level: Dict[str, float] = {}
    hist_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                             "history.json")
    if os.path.isfile(hist_path):
        import json
        with open(hist_path, "r", encoding="utf-8") as fh:
            hist = json.load(fh)
        if hist:
            for k, v in hist[-1].items():
                if k.startswith("n_pos_l"):
                    per_level[k] = float(v)

    print("=" * 70)
    print(f"  NIRDet-Forge fp32 evaluation — split '{split}'")
    print("=" * 70)
    print(f"  mAP50            : {metrics['map_50']:.4f}")
    if np.isfinite(ci[0]):
        print(f"  95% CI           : [{ci[0]:.4f}, {ci[2]:.4f}]  "
              f"(median {ci[1]:.4f}, {n_boot} resamples)")
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
        # Report only the levels the history actually carries: a checkpoint
        # from a --p5-ablate run legitimately has two.
        lvls = sorted(int(k.rsplit("l", 1)[1]) for k in per_level)
        print("[eval] n_pos per level:  " + "  ".join(
            f"l{i} {int(per_level[f'n_pos_l{i}']):>5d}" for i in lvls))
        if len(lvls) != len(cfg.model.strides):
            print(f"  ! history has {len(lvls)} levels but the config has "
                  f"{len(cfg.model.strides)} strides — different run config")
        zero = [f"n_pos_l{i}" for i in lvls if per_level[f"n_pos_l{i}"] < 1.0]
        if zero:
            print(f"    ! {', '.join(zero)} near zero: that level is doing "
                  f"nothing. Confirm with train.py --p5-ablate.")
    print(f"  deploy_score_thresh : {thr:.3f}  (F1 {best['f1']:.4f}, "
          f"P {best['precision']:.4f}, R {best['recall']:.4f}, "
          f"TP {best['tp']} FP {best['fp']} FN {best['fn']})")
    print("=" * 70)

    if profile is not None and args.profile:
        profile.update_deploy_thresh(args.profile, thr)
        print(f"[profile] deploy_score_thresh {thr:.3f} written to {args.profile}")
    else:
        print("[profile] no --profile given, so deploy_score_thresh was not "
              "persisted. live_nirdet.py requires --profile or --score-thresh.")

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
        "bootstrap_resamples": int(n_boot),
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
