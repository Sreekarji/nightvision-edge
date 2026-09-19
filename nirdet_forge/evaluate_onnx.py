"""
evaluate_onnx.py — the INT8 accuracy gate
=========================================
    python evaluate_onnx.py --onnx nirdet-sim.onnx --profile datasets/<n>.yaml
    python evaluate_onnx.py --int8 nirdet-int8-qdq.onnx \
                            --fp32-ref nirdet-sim.onnx --profile datasets/<n>.yaml

The ONLY place the exported graph — fp32 or INT8 QDQ — is scored against
ground truth. The Torch-side number proves the trained weights; it cannot
prove that the export, the simplifier and the QDQ quantiser preserved them.

cfg.eval.int8_max_map50_drop gates the drop from THIS model's fp32 ONNX
(--fp32-ref), not from cfg.eval.baseline_map50, which is a different model
entirely and is printed for orientation only.

No duplication: the decode arithmetic and the MEMBERSHIP contract come from
config (np_sigmoid, clamp_and_filter), preprocessing from
dataset.preprocess_frame, the input name from export_onnx.graph_input_name
(which needs only `onnx`, so a plain fp32 evaluation no longer requires the
quantisation toolchain), and the metric from evaluate._map50_metric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import (DECODE_OFFSET_BIAS, DECODE_OFFSET_SCALE, Config,
                    REG_LOG_CLAMP_MAX, REG_LOG_CLAMP_MIN, clamp_and_filter,
                    get_config, np_sigmoid, validate_config)
from dataset import (boxes_to_canvas, cxcywh_to_xyxy_px, list_images,
                     load_flat_field, preprocess_frame, read_yolo_labels,
                     resolve_split_dirs)
from evaluate import _map50_metric, map50_value
from export_onnx import graph_input_name

try:
    import cv2
except ImportError as exc:                                    # pragma: no cover
    raise SystemExit("evaluate_onnx.py needs opencv: "
                     "pip install opencv-python") from exc

try:
    import onnxruntime as ort
except ImportError as exc:                                    # pragma: no cover
    raise SystemExit("evaluate_onnx.py needs onnxruntime: "
                     "pip install onnxruntime") from exc


# ===========================================================================
# decode
# ===========================================================================

def decode_onnx_outputs(outs: Dict[str, np.ndarray], cfg: Config,
                        score_thresh: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    ONNX output blobs -> (boxes (N,4) xyxy px, scores (N,)), pre-NMS.

    Arithmetic identical to losses.AnchorGeometry.decode,
    head.PedestrianHead.forward, live_nirdet.decode_level and nirdet_pp.c.
    MEMBERSHIP identical too: clamp to the canvas and THEN filter, via the
    shared config.clamp_and_filter, so this decoder cannot drift from the
    runtime's (this function used to apply no clamp at all).

    The threshold is applied to the RAW LOGIT (sigmoid is monotonic).
    """
    st = min(max(float(score_thresh), 1e-6), 1.0 - 1e-6)
    logit_th = float(np.log(st / (1.0 - st)))
    img_h = int(cfg.data.img_h)
    img_w = int(cfg.data.img_w)

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    for s in cfg.model.strides:
        s = int(s)
        for blob in (f"cls{s}", f"off{s}", f"size{s}"):
            if blob not in outs:
                raise KeyError(
                    f"decode_onnx_outputs: missing output blob {blob!r}; "
                    f"present keys {sorted(outs)}. The ONNX must expose the "
                    f"three-per-level names from head.output_names().")
        cls_blob = np.asarray(outs[f"cls{s}"], dtype=np.float32)[0, 0]
        off_blob = np.asarray(outs[f"off{s}"], dtype=np.float32)[0]
        size_blob = np.asarray(outs[f"size{s}"], dtype=np.float32)[0]

        m = cls_blob >= logit_th
        if not bool(m.any()):
            continue
        rows, cols = np.nonzero(m)
        fr = rows.astype(np.float32)
        fc = cols.astype(np.float32)

        conf = np_sigmoid(cls_blob[rows, cols])
        cx = (DECODE_OFFSET_SCALE * np_sigmoid(off_blob[0][rows, cols])
              - DECODE_OFFSET_BIAS + fc) * float(s)
        cy = (DECODE_OFFSET_SCALE * np_sigmoid(off_blob[1][rows, cols])
              - DECODE_OFFSET_BIAS + fr) * float(s)
        bw = np.exp(np.clip(size_blob[0][rows, cols], REG_LOG_CLAMP_MIN,
                            REG_LOG_CLAMP_MAX)) * float(img_w)
        bh = np.exp(np.clip(size_blob[1][rows, cols], REG_LOG_CLAMP_MIN,
                            REG_LOG_CLAMP_MAX)) * float(img_h)

        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                         axis=1).astype(np.float32)
        boxes, conf = clamp_and_filter(boxes, conf.astype(np.float32),
                                       img_h, img_w)
        if boxes.shape[0] == 0:
            continue
        all_boxes.append(boxes)
        all_scores.append(conf)

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    return (np.concatenate(all_boxes, axis=0).astype(np.float32),
            np.concatenate(all_scores, axis=0).astype(np.float32))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float,
         max_det: int) -> np.ndarray:
    """Greedy NMS indices, score-descending. torchvision when available."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    # Pre-NMS cap: match the C heap capacity (Fix 4). The returned indices
    # must stay in the CALLER's boxes/scores domain, so any cap applied to
    # the sliced arrays is remapped through top_idx before returning.
    from config import PRE_NMS_TOPK
    top_idx = None
    if PRE_NMS_TOPK > 0 and scores.shape[0] > PRE_NMS_TOPK:
        top_idx = np.argsort(-scores)[:PRE_NMS_TOPK]
        boxes = boxes[top_idx]
        scores = scores[top_idx]
    try:
        from torchvision.ops import nms as tv_nms
        # tv_nms already returns indices in decreasing score order; the old
        # re-sort was redundant and indexed a torch tensor with a numpy array.
        keep = tv_nms(torch.from_numpy(boxes), torch.from_numpy(scores),
                      float(iou_thresh))
        keep = keep.numpy().astype(np.int64)[:max_det]
        return keep if top_idx is None else top_idx[keep]
    except ImportError:
        pass
    # F37: NumPy fallback — fast greedy NMS matching live_nirdet.greedy_nms
    # (the old O(n^2)-per-survivor mask loop is gone).
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thresh]
    keep = np.asarray(keep, dtype=np.int64)
    return keep if top_idx is None else top_idx[keep]


# ===========================================================================
# session + evaluation
# ===========================================================================

def _open_session(onnx_path: str) -> "ort.InferenceSession":
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    try:
        return ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider",
                                  "CPUExecutionProvider"])
    except Exception:
        return ort.InferenceSession(onnx_path,
                                    providers=["CPUExecutionProvider"])


def _verify_sidecar(onnx_path: str, cfg: Config,
                    allow_missing: bool = False) -> None:
    """
    Compare the graph's .contract.json against the live config (F39).

    Without this the gate happily scores a graph exported at a different
    canvas or with different CLAHE settings, reporting a low mAP that looks
    like a quantisation problem.

    F38: a MISSING sidecar is fatal by default. Every artefact this project
    emits carries one; absence means the graph was not produced by
    export_onnx.py/quantize_qdq.py and its geometry and preprocessing are
    unknown.
    """
    side = os.path.splitext(onnx_path)[0] + ".contract.json"
    if not os.path.isfile(side):
        msg = (
            f"no contract sidecar beside {onnx_path}. Every artefact this "
            f"project emits carries one; absence means the graph was not "
            f"produced by export_onnx.py/quantize_qdq.py and its geometry "
            f"and preprocessing are unknown — any mAP measured here is "
            f"meaningless.")
        if not allow_missing:
            raise SystemExit(msg +
                             " Pass --allow-missing-contract to override.")
        print(f"[eval-onnx] WARNING {msg}")
        return
    with open(side, encoding="utf-8") as fh:
        got = str(json.load(fh).get("hash", ""))
    want = cfg.deploy_contract()["hash"]
    if got != want:
        raise SystemExit(
            f"contract mismatch: {side} carries {got}, current config is "
            f"{want}. The graph was exported at a different geometry or "
            f"preprocessing, so any mAP measured against it is meaningless.")
    print(f"[eval-onnx] contract {got} verified against the live config")


def evaluate_onnx(onnx_path: str, cfg: Config, split: str = "test",
                  score_thresh: Optional[float] = None,
                  nms_iou: Optional[float] = None,
                  allow_missing_contract: bool = False) -> dict:
    """Score one ONNX graph over ``split`` in CANVAS space."""
    if not cfg.data.root or not os.path.isdir(str(cfg.data.root)):
        raise SystemExit(
            f"evaluate_onnx: cfg.data.root is unset or missing "
            f"({cfg.data.root!r}). Pass --profile <yaml> like train.py does.")
    st = float(cfg.eval.eval_score_thresh if score_thresh is None
               else score_thresh)
    nms_iou = float(cfg.model.nms_iou_thresh if nms_iou is None else nms_iou)

    _verify_sidecar(onnx_path, cfg, allow_missing=allow_missing_contract)

    img_dir, lbl_dir = resolve_split_dirs(str(cfg.data.root), split)
    paths = sorted(list_images(img_dir))
    if not paths:
        raise SystemExit(f"evaluate_onnx: split '{split}' at {img_dir} is empty")

    sess = _open_session(onnx_path)
    in_name = graph_input_name(onnx_path)
    out_names = [o.name for o in sess.get_outputs()]
    flat_field = load_flat_field(cfg.aug.flat_field_path)
    h, w = int(cfg.data.img_h), int(cfg.data.img_w)
    max_det = int(cfg.model.max_det)

    m50 = _map50_metric()
    n_gt = n_pred = 0
    n_ok = 0                                   # F40: successfully processed

    for k, p in enumerate(paths, 1):
        raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"  [eval-onnx] unreadable image, skipped: {p}")
            continue
        n_ok += 1
        canvas, scale, pad_x, pad_y = preprocess_frame(
            raw, h, w, clahe_enabled=cfg.aug.clahe_enabled,
            clahe_clip=cfg.aug.clahe_clip, clahe_grid=cfg.aug.clahe_grid,
            flat_field=flat_field)
        t = canvas[np.newaxis, np.newaxis, :, :].astype(np.float32)
        outs = dict(zip(out_names, sess.run(out_names, {in_name: t})))
        boxes, scores = decode_onnx_outputs(outs, cfg, st)
        sel = _nms(boxes, scores, nms_iou, max_det)
        boxes, scores = boxes[sel], scores[sel]

        lbl = os.path.join(lbl_dir,
                           os.path.splitext(os.path.basename(p))[0] + ".txt")
        gt_c = boxes_to_canvas(read_yolo_labels(lbl), raw.shape[0],
                               raw.shape[1], h, w, scale, pad_x, pad_y)
        gb = cxcywh_to_xyxy_px(gt_c, h, w)

        pb_t = torch.from_numpy(np.ascontiguousarray(boxes))
        gb_t = torch.from_numpy(np.ascontiguousarray(gb))
        m50.update(
            [{"boxes": pb_t, "scores": torch.from_numpy(scores),
              "labels": torch.zeros(pb_t.shape[0], dtype=torch.long)}],
            [{"boxes": gb_t,
              "labels": torch.zeros(gb_t.shape[0], dtype=torch.long)}])
        n_gt += int(gb_t.shape[0])
        n_pred += int(pb_t.shape[0])
        if k % 50 == 0 or k == len(paths):
            print(f"  [eval-onnx] {k}/{len(paths)} images", flush=True)

    return {
        "map50": map50_value(m50.compute()),
        "n_images": n_ok, "split": split,      # F40: read successes, not list size
        "n_gt": n_gt, "n_pred": n_pred,
        "score_thresh": st, "nms_iou": nms_iou, "onnx_path": onnx_path,
    }


# ===========================================================================
# paired fp32+int8 evaluation (F39)
# ===========================================================================

def evaluate_onnx_pair(int8_path: str, fp32_path: str, cfg: Config,
                       split: str = "test",
                       score_thresh: Optional[float] = None,
                       nms_iou: Optional[float] = None,
                       allow_missing_contract: bool = False
                       ) -> Tuple[dict, dict]:
    """Run fp32 and int8 over the same preprocessed images in one pass.

    The two serial evaluate_onnx() calls preprocessed EVERY image twice; this
    shares one preprocessing pass per image between both sessions, so the pair
    is guaranteed to see identical inputs and the wall-clock halves.
    """
    if not cfg.data.root or not os.path.isdir(str(cfg.data.root)):
        raise SystemExit(
            f"evaluate_onnx: cfg.data.root is unset or missing "
            f"({cfg.data.root!r}). Pass --profile <yaml> like train.py does.")
    st = float(cfg.eval.eval_score_thresh if score_thresh is None
               else score_thresh)
    nms_iou = float(cfg.model.nms_iou_thresh if nms_iou is None else nms_iou)

    _verify_sidecar(int8_path, cfg, allow_missing=allow_missing_contract)
    _verify_sidecar(fp32_path, cfg, allow_missing=allow_missing_contract)

    img_dir, lbl_dir = resolve_split_dirs(str(cfg.data.root), split)
    paths = sorted(list_images(img_dir))
    if not paths:
        raise SystemExit(f"evaluate_onnx: split '{split}' at {img_dir} is empty")

    sess_int8 = _open_session(int8_path)
    sess_fp32 = _open_session(fp32_path)
    in_name_int8 = graph_input_name(int8_path)
    in_name_fp32 = graph_input_name(fp32_path)
    outs_int8 = [o.name for o in sess_int8.get_outputs()]
    outs_fp32 = [o.name for o in sess_fp32.get_outputs()]

    met_int8 = _map50_metric()
    met_fp32 = _map50_metric()
    flat_field = load_flat_field(cfg.aug.flat_field_path)
    h, w = int(cfg.data.img_h), int(cfg.data.img_w)
    max_det = int(cfg.model.max_det)

    n_ok = 0
    gt8 = pred8 = gt32 = pred32 = 0
    for k, p in enumerate(paths, 1):
        raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"  [eval-onnx] unreadable image, skipped: {p}")
            continue
        canvas, scale, pad_x, pad_y = preprocess_frame(
            raw, h, w, clahe_enabled=cfg.aug.clahe_enabled,
            clahe_clip=cfg.aug.clahe_clip, clahe_grid=cfg.aug.clahe_grid,
            flat_field=flat_field)
        inp = canvas[np.newaxis, np.newaxis].astype(np.float32)

        lbl = os.path.join(lbl_dir,
                           os.path.splitext(os.path.basename(p))[0] + ".txt")
        gt_c = boxes_to_canvas(read_yolo_labels(lbl), raw.shape[0],
                               raw.shape[1], h, w, scale, pad_x, pad_y)
        gb = cxcywh_to_xyxy_px(gt_c, h, w)
        g_t = {"boxes": torch.from_numpy(np.ascontiguousarray(gb)),
               "labels": torch.zeros(gb.shape[0], dtype=torch.long)}

        for sess, out_names, in_name, met in (
                (sess_int8, outs_int8, in_name_int8, met_int8),
                (sess_fp32, outs_fp32, in_name_fp32, met_fp32)):
            outs = dict(zip(out_names, sess.run(out_names, {in_name: inp})))
            boxes, scores = decode_onnx_outputs(outs, cfg, st)
            sel = _nms(boxes, scores, nms_iou, max_det)
            boxes, scores = boxes[sel], scores[sel]
            pb_t = torch.from_numpy(np.ascontiguousarray(boxes))
            met.update(
                [{"boxes": pb_t, "scores": torch.from_numpy(scores),
                  "labels": torch.zeros(pb_t.shape[0], dtype=torch.long)}],
                [g_t])
            if met is met_int8:
                gt8 += int(g_t["boxes"].shape[0])
                pred8 += int(pb_t.shape[0])
            else:
                gt32 += int(g_t["boxes"].shape[0])
                pred32 += int(pb_t.shape[0])
        n_ok += 1
        if k % 50 == 0 or k == len(paths):
            print(f"  [eval-onnx] {k}/{len(paths)} images (paired)", flush=True)

    return (
        {"map50": map50_value(met_int8.compute(), report=True), "n_images": n_ok,
         "split": split, "n_gt": gt8, "n_pred": pred8,
         "score_thresh": st, "nms_iou": nms_iou, "onnx_path": int8_path},
        {"map50": map50_value(met_fp32.compute(), report=False), "n_images": n_ok,
         "split": split, "n_gt": gt32, "n_pred": pred32,
         "score_thresh": st, "nms_iou": nms_iou, "onnx_path": fp32_path},
    )


# ===========================================================================
# CLI
# ===========================================================================

def _boxed(title: str, lines: List[str]) -> None:
    width = max([len(title)] + [len(l) for l in lines]) + 4
    print("+" + "-" * width + "+")
    print(f"|  {title.ljust(width - 4)}  |")
    print("|" + "-" * width + "|")
    for l in lines:
        print(f"|  {l.ljust(width - 4)}  |")
    print("+" + "-" * width + "+")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ONNX fp32 / INT8 accuracy gate for NIRDet-Lite")
    ap.add_argument("--onnx", default=None,
                    help="ONNX graph to evaluate (fp32 or INT8 QDQ)")
    ap.add_argument("--int8", default=None, metavar="PATH",
                    help="alias for --onnx that additionally enforces the "
                         "mAP50-drop gate against --fp32-ref")
    ap.add_argument("--fp32-ref", default=None, metavar="PATH",
                    help="fp32 ONNX the INT8 gate measures the drop against")
    ap.add_argument("--profile", default=None,
                    help="dataset profile YAML, applied before any I/O")
    ap.add_argument("--split", default=None, help="default: report_split")
    ap.add_argument("--score-thresh", type=float, default=None,
                    help="default: cfg.eval.eval_score_thresh (full PR curve)")
    # The old --iou-thresh conflated the NMS IoU with the TP-matching IoU and
    # defaulted to 0.5, while the Torch evaluator uses
    # cfg.model.nms_iou_thresh (0.45) — so the two fp32 numbers were not
    # comparable (F37).
    ap.add_argument("--nms-iou", type=float, default=None,
                    help="NMS IoU; default cfg.model.nms_iou_thresh")
    ap.add_argument("--allow-missing-contract", action="store_true",
                    help="override the F38 fatal error when a graph has no "
                         ".contract.json sidecar (its geometry is then "
                         "UNVERIFIED and any mAP is advisory only)")
    args = ap.parse_args()

    path = args.int8 or args.onnx
    if not path:
        ap.error("give --onnx <path> (or --int8 <path>)")
    is_int8 = args.int8 is not None
    if is_int8 and not args.fp32_ref:
        ap.error(
            "--int8 requires --fp32-ref <path>: the gate measures the drop "
            "from THIS model's own fp32 ONNX, not from "
            "cfg.eval.baseline_map50.")

    cfg = get_config()
    if args.profile:
        from dataset_profiles import DatasetProfile
        DatasetProfile.load(args.profile).apply(cfg)
    else:
        print("\n" + "!" * 74)
        print("  NO --profile GIVEN")
        print("  cfg.data.root is unset — evaluation requires a dataset root.")
        print("  Pass --profile <yaml> (same one used for training).")
        print("!" * 74 + "\n")
        raise SystemExit(
            "--profile is required for evaluate_onnx.py: it supplies "
            "cfg.data.root (where to find the images), prior_w/prior_h "
            "(needed by validate_config), and the score threshold "
            "(deploy_score_thresh written by evaluate.py). "
            "Run: python evaluate_onnx.py --onnx ... --profile datasets/<n>.yaml")
    validate_config(cfg)

    split = args.split or cfg.eval.report_split
    nms_iou = (float(args.nms_iou) if args.nms_iou is not None
               else float(cfg.model.nms_iou_thresh))
    print(f"[eval-onnx] graph   : {path}"
          f"{'  [INT8 gated]' if is_int8 else ''}")
    _st_resolved = float(
        args.score_thresh if args.score_thresh is not None
        else cfg.eval.eval_score_thresh)
    _st_resolved = min(max(_st_resolved, 1e-6), 1.0 - 1e-6)
    print(f"[eval-onnx] split   : {split}  score_thresh "
          f"{_st_resolved:.4f}  nms_iou {nms_iou}"
          f"  (TP matching is fixed at IoU 0.5: mAP50)")

    # F39: when both graphs are given, score them in ONE paired pass over
    # shared preprocessing instead of two serial evaluate_onnx() calls.
    if is_int8:
        res, ref = evaluate_onnx_pair(
            path, args.fp32_ref, cfg, split=split,
            score_thresh=args.score_thresh, nms_iou=nms_iou,
            allow_missing_contract=args.allow_missing_contract)
    else:
        res = evaluate_onnx(path, cfg, split=split,
                            score_thresh=args.score_thresh, nms_iou=nms_iou,
                            allow_missing_contract=args.allow_missing_contract)
        ref = None
    print(f"[eval-onnx] mAP50   : {res['map50']:.4f}   "
          f"({res['n_images']} images, {res['n_gt']} GT, {res['n_pred']} preds)")
    if cfg.eval.baseline_map50 is not None:
        src = cfg.eval.baseline_source or "dataset reference"
        print(f"[eval-onnx] note    : vs cfg.eval.baseline_map50 = "
              f"{cfg.eval.baseline_map50:.4f} ({src}, NOT this model's fp32 "
              f"score) delta {res['map50'] - cfg.eval.baseline_map50:+.4f}")

    if not is_int8:
        return 0
    gate = float(cfg.eval.int8_max_map50_drop)
    drop = ref["map50"] - res["map50"]
    print(f"[gate] fp32 ONNX  : {ref['map50']:.4f}  ({args.fp32_ref})")
    print(f"[gate] int8 ONNX  : {res['map50']:.4f}")
    print(f"[gate] drop       : {drop:+.4f}   allowed <= {gate:.4f}")
    if drop > gate:
        _boxed("INT8 mAP50 DROP VIOLATES THE GATE", [
            f"fp32 {ref['map50']:.4f} -> int8 {res['map50']:.4f}  "
            f"drop {drop:.4f} > {gate:.4f}",
            "",
            "Remediation, in order:",
            " 1. re-quantise with --exclude-first-conv (quantize_qdq.py):",
            "    the stem + eaa.edge_conv sit on the raw sensor distribution",
            "    and blow up per-tensor scales when included.",
            " 2. switch calib method: --method percentile",
            " 3. raise --calib-images so each tensor sees its real range.",
            "",
            f"graph: {path}   ref: {args.fp32_ref}   split: {split}",
        ])
        return 1
    print("[gate] INT8 accuracy gate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
