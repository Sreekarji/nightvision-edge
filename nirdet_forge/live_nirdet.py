"""
live_nirdet.py — Raspberry Pi 5 runtime (NCNN INT8)
====================================================
    python live_nirdet.py --profile datasets/<n>.yaml \
                          --param nirdet-int8.param --bin nirdet-int8.bin \
                          --contract nirdet-int8.contract.json
    python live_nirdet.py --bench --param ... --bin ... --contract ...
    python live_nirdet.py --video clip.mp4 --score-thresh 0.42 ...

THREE THREADS: capture (picamera2 YUV420 Y plane, or cv2.VideoCapture),
inference (preprocess + NCNN INT8 + decode + NMS), display (IoU tracker +
OpenCV window). Both hand-offs are queue.Queue(maxsize=2) and capture DROPS a
frame rather than blocking, so a slow display never back-pressures the sensor.

NO THRESHOLD LITERAL IN THIS FILE. The score threshold comes from the dataset
profile (deploy_score_thresh, measured by evaluate.py) or from --score-thresh.

NO TORCH, DIRECTLY OR TRANSITIVELY
----------------------------------
This file imports config, preprocess, numpy, cv2, yaml and ncnn. It must NOT
import dataset.py (torch) — and therefore must NOT import dataset_profiles.py
either, because dataset_profiles does `from dataset import label_path_for,
read_yolo_labels`. The profile YAML is read directly by _apply_profile_lite
below: it is a flat yaml.safe_dump of the DatasetProfile dataclass, so the
fields this runtime needs are plain top-level keys. test_decode_contract T2
enforces the rule.

The profile is read WITHOUT the dataset freshness check: this machine is the
deployment target and does not host the dataset, so an unconditional
re-fingerprint would make the documented deployment command impossible to
run. What actually protects decode geometry is verify_contract(), which
compares the hashed deploy contract — canvas, strides, decode constants,
membership constant, blob layout AND the preprocessing chain.

FLAT-FIELD WARNING
------------------
config.Config._flat_field_digest hashes the flat-field FILE CONTENTS. If the
map is absent on the Pi it falls back to the basename and the contract hash
will NOT match the export machine. Copy the map to the Pi and point
--flat-field at it, or export with flat_field_path unset.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# OpenMP reads these at library init, so they MUST be set BEFORE ncnn (or
# anything that drags libgomp in) is imported. Cores 0-2 run inference; core 3
# is left to capture, display and the display server. The ncnn thread count is
# derived from the SAME number (F58) — the two used to disagree (3 vs 4).
_DEFAULT_THREADS = int(os.environ.get("NIRDET_THREADS", "3"))
os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

import numpy as np

import cv2

cv2.setNumThreads(1)

from config import (DECODE_OFFSET_BIAS, DECODE_OFFSET_SCALE,
                    REG_LOG_CLAMP_MAX, REG_LOG_CLAMP_MIN, STRIDES,
                    clamp_and_filter, get_config, np_sigmoid)
# F57: preprocess.py is numpy+cv2 ONLY — the Pi runtime must not import
# torch, and `from dataset import ...` pulls dataset.py (torch) in.
from preprocess import (apply_clahe, apply_flat_field, letterbox,
                        load_flat_field)

DEFAULT_INPUT_BLOB = "images"


# ===========================================================================
# profile (torch-free)
# ===========================================================================

# Exactly the DatasetProfile fields this runtime consumes. Anything else in
# the YAML (fingerprints, split stats, priors, baselines) is training-side.
_PROFILE_KEYS = (
    "name", "img_h", "img_w", "strides", "deploy_score_thresh",
    "clahe_enabled", "clahe_clip", "clahe_grid", "flat_field_path",
)


def _apply_profile_lite(cfg, path: str) -> dict:
    """
    Read a DatasetProfile YAML and transplant the DEPLOYMENT fields into cfg.

    Deliberately not DatasetProfile.load(...).apply(cfg, verify=False):
    importing dataset_profiles imports dataset, which imports torch, which
    this file is not allowed to do. The YAML is a flat asdict() dump, so the
    keys below are stable as long as the dataclass fields keep their names —
    and every key is looked up explicitly so a rename fails loudly here
    rather than silently deploying a default.
    """
    try:
        import yaml
    except ImportError as exc:                       # pragma: no cover
        raise SystemExit("--profile needs PyYAML: pip install pyyaml") from exc
    if not os.path.isfile(path):
        raise SystemExit(f"dataset profile not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        prof = yaml.safe_load(fh) or {}
    if not isinstance(prof, dict):
        raise SystemExit(f"{path} is not a YAML mapping")

    missing = [k for k in ("img_h", "img_w", "strides") if k not in prof]
    if missing:
        raise SystemExit(
            f"{path} is missing {missing}. This is not a DatasetProfile YAML, "
            f"or dataset_profiles.DatasetProfile has been renamed and this "
            f"runtime's _PROFILE_KEYS list needs updating alongside it.")

    cfg.data.img_h = int(prof["img_h"])
    cfg.data.img_w = int(prof["img_w"])
    cfg.model.strides = tuple(int(s) for s in prof["strides"])
    # Preprocessing is part of the HASHED contract: a model trained at
    # clahe_clip 2.0 and deployed at 3.0 decodes fine and loses accuracy
    # silently. Transplant it verbatim.
    if "clahe_enabled" in prof:
        cfg.aug.clahe_enabled = bool(prof["clahe_enabled"])
    if "clahe_clip" in prof:
        cfg.aug.clahe_clip = float(prof["clahe_clip"])
    if "clahe_grid" in prof:
        cfg.aug.clahe_grid = int(prof["clahe_grid"])
    if prof.get("flat_field_path"):
        cfg.aug.flat_field_path = str(prof["flat_field_path"])
    if prof.get("deploy_score_thresh") is not None:
        cfg.eval.deploy_score_thresh = float(prof["deploy_score_thresh"])
    cfg.data.profile_applied = True
    cfg.data.profile_name = str(prof.get("name", os.path.basename(path)))

    print(f"[profile] {cfg.data.profile_name}: canvas {cfg.data.img_h}x"
          f"{cfg.data.img_w}, strides {cfg.model.strides}, "
          f"clahe={cfg.aug.clahe_enabled}/{cfg.aug.clahe_clip}/"
          f"{cfg.aug.clahe_grid}, deploy_score_thresh="
          f"{cfg.eval.deploy_score_thresh}")
    return {k: prof.get(k) for k in _PROFILE_KEYS}


# ===========================================================================
# contract verification
# ===========================================================================

def load_contract(path: Optional[str]) -> Optional[dict]:
    """Read the .contract.json sidecar written beside every exported artefact
    by export_onnx.write_contract_sidecar / export_ncnn.py."""
    if not path:
        return None
    if not os.path.isfile(path):
        raise SystemExit(f"contract sidecar not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_contract(contract: Optional[dict], cfg, strides) -> None:
    """
    Refuse to decode a model whose contract does not match this runtime.

    The .param/.bin carry no geometry, decode or preprocessing information, so
    without this the runtime happily decodes any model with whatever constants
    it was built against — a half-stride box shift or a wrong exp range,
    visible only as a few lost mAP points that nobody can attribute.
    """
    if contract is None:
        print("[contract] WARNING no --contract given. The .param/.bin carry "
              "no geometry, so nothing verifies that this model was trained "
              "under the constants this runtime was built with. Pass the "
              "sidecar that export_ncnn.py wrote beside the artefacts.")
        return
    want = cfg.deploy_contract()
    got_hash = str(contract.get("hash", ""))
    if got_hash != want["hash"]:
        raise SystemExit(
            f"DEPLOY CONTRACT MISMATCH\n"
            f"  model sidecar : {got_hash or '<absent>'}\n"
            f"  this runtime  : {want['hash']}\n"
            f"The contract covers num_classes, canvas h/w, strides, the EAA "
            f"edge stride/pool factor, DECODE_OFFSET_SCALE/BIAS, "
            f"REG_LOG_CLAMP_MIN/MAX, MIN_BOX_PX, the blob layout AND the "
            f"CLAHE / flat-field preprocessing. Decoding across a mismatch "
            f"puts every box in the wrong place, gives it the wrong size "
            f"range, or feeds the network a distribution it was never "
            f"calibrated on.\n"
            f"COMMON CAUSE ON THE PI: the flat-field map is hashed by FILE "
            f"CONTENTS on the export machine and falls back to the basename "
            f"when the file is absent here. Copy the map over and pass "
            f"--flat-field, or re-export with flat_field_path unset.\n"
            f"  sidecar flat_field : {contract.get('flat_field')!r}\n"
            f"  runtime flat_field : {want.get('flat_field')!r}\n"
            f"Otherwise: re-export from the checkpoint that matches this "
            f"config, or deploy the config that trained the model.")
    for key, mine in (("img_h", int(cfg.data.img_h)),
                      ("img_w", int(cfg.data.img_w))):
        if key in contract and int(contract[key]) != mine:
            raise SystemExit(f"contract {key}={contract[key]} != runtime {mine}")
    c_str = [int(s) for s in contract.get("strides", [])]
    if c_str and c_str != [int(s) for s in strides]:
        raise SystemExit(f"contract strides {c_str} != runtime {list(strides)}")
    names = contract.get("blob_names")
    if names:
        want_names = [f"{p}{s}" for s in strides
                      for p in ("cls", "off", "size")]
        if list(names) != want_names:
            raise SystemExit(
                f"contract blob_names {list(names)} != runtime {want_names}")
    print(f"[contract] {got_hash} verified: canvas {contract.get('img_h')}x"
          f"{contract.get('img_w')}, strides {c_str or list(strides)}, "
          f"{contract.get('blobs_per_level', 3)} blobs/level, "
          f"clahe={contract.get('clahe_enabled')} "
          f"clip={contract.get('clahe_clip')} grid={contract.get('clahe_grid')}")


# ===========================================================================
# math helpers
# ===========================================================================

def greedy_nms(boxes_xyxy: np.ndarray, scores: np.ndarray,
               iou_thresh: float) -> np.ndarray:
    """
    Pure-NumPy greedy NMS. Returns kept indices, score-descending.

    Deliberately not cv2.dnn.NMSBoxes: its argument order, box format and
    return shape have all changed between OpenCV releases, and the Pi's
    OpenCV comes from whatever the distro ships.
    """
    n = boxes_xyxy.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    areas = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0, None) * \
        np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0, None)
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        y1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        x2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        y2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(aa + ab - inter, 1e-9))


# ===========================================================================
# decode — THE CONTRACT
# ===========================================================================

def decode_level(cls_blob: np.ndarray, off_blob: np.ndarray,
                 size_blob: np.ndarray, stride: int, img_h: int, img_w: int,
                 score_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    One level's three blobs -> (boxes xyxy px, scores), pre-NMS.

    Arithmetic identical to losses.AnchorGeometry.decode,
    head.PedestrianHead.forward, evaluate_onnx.decode_onnx_outputs and
    nirdet_pp.c. MEMBERSHIP identical too (config.clamp_and_filter): xyxy is
    CLAMPED to the canvas and THEN filtered, in that order. Clamping after the
    filter lets zero-area off-canvas boxes reach NMS, where IoU 0 against
    everything makes them unsuppressable.
    """
    cls_blob = np.asarray(cls_blob, dtype=np.float32)
    off_blob = np.asarray(off_blob, dtype=np.float32)
    size_blob = np.asarray(size_blob, dtype=np.float32)
    if cls_blob.ndim == 3:
        cls_blob = cls_blob[0]

    # sigmoid is monotonic: threshold the RAW LOGIT to skip expf() on ~3000
    # rejected cells per level per frame.
    _raw_st = float(score_thresh)
    st = min(max(_raw_st, 1e-6), 1.0 - 1e-6)
    if st != _raw_st and not getattr(decode_level, "_st_warned", False):
        print(f"[live] WARNING: score_thresh {_raw_st:.6f} is out of "
              f"(1e-6, 1-1e-6); clamped to {st:.7f}")
        decode_level._st_warned = True
    logit_th = float(np.log(st / (1.0 - st)))
    m = cls_blob >= logit_th
    if not bool(m.any()):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    rows, cols = np.nonzero(m)
    conf = np_sigmoid(cls_blob[rows, cols])
    fr = rows.astype(np.float32)
    fc = cols.astype(np.float32)

    cx = (DECODE_OFFSET_SCALE * np_sigmoid(off_blob[0][rows, cols])
          - DECODE_OFFSET_BIAS + fc) * float(stride)
    cy = (DECODE_OFFSET_SCALE * np_sigmoid(off_blob[1][rows, cols])
          - DECODE_OFFSET_BIAS + fr) * float(stride)
    bw = np.exp(np.clip(size_blob[0][rows, cols], REG_LOG_CLAMP_MIN,
                        REG_LOG_CLAMP_MAX)) * float(img_w)
    bh = np.exp(np.clip(size_blob[1][rows, cols], REG_LOG_CLAMP_MIN,
                        REG_LOG_CLAMP_MAX)) * float(img_h)

    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                     axis=1).astype(np.float32)
    return clamp_and_filter(boxes, conf.astype(np.float32), img_h, img_w)


def decode_frame(blobs: Dict[str, np.ndarray], strides, img_h: int, img_w: int,
                 score_thresh: float, iou_thresh: float,
                 max_det: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    All levels -> final (boxes xyxy px, scores), descending score.

    Order matches nirdet_pp.c and model.decode_predictions: per-level decode
    with clamp-then-drop membership, concatenate, NMS across levels, THEN
    truncate to max_det. Truncating before NMS would let a cluster of
    mutually suppressing duplicates consume the whole budget.
    """
    all_b: List[np.ndarray] = []
    all_s: List[np.ndarray] = []
    for s in strides:
        b, sc = decode_level(blobs[f"cls{s}"], blobs[f"off{s}"],
                             blobs[f"size{s}"], int(s), img_h, img_w,
                             score_thresh)
        if b.shape[0]:
            all_b.append(b)
            all_s.append(sc)
    if not all_b:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    boxes = np.concatenate(all_b, axis=0)
    scores = np.concatenate(all_s, axis=0)
    from config import PRE_NMS_TOPK
    if PRE_NMS_TOPK > 0 and scores.shape[0] > PRE_NMS_TOPK:
        top_idx = np.argsort(-scores)[:PRE_NMS_TOPK]
        boxes, scores = boxes[top_idx], scores[top_idx]
    keep = greedy_nms(boxes, scores, iou_thresh)[:int(max_det)]
    return boxes[keep], scores[keep]


# ===========================================================================
# IoU tracker
# ===========================================================================

@dataclass
class Track:
    box: np.ndarray
    score: float
    frame_count: int = 1
    missed: int = 0
    confirmed: bool = False
    tid: int = 0


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(T,4) xyxy, (D,4) xyxy -> (T,D) IoU matrix (F60)."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    xx1 = np.maximum(ax1[:, None], bx1[None, :])
    yy1 = np.maximum(ay1[:, None], by1[None, :])
    xx2 = np.minimum(ax2[:, None], bx2[None, :])
    yy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class IoUTracker:
    """
    Minimal IoU tracker: no motion model, no Kalman filter.

    A detection must persist for confirm_hits consecutive frames before it is
    displayed, which removes isolated single-frame false positives. The three
    thresholds are DEPLOYMENT TUNING and come from cfg.eval (F60), not from
    class constants: confirm_hits in particular trades latency for precision
    and depends on frame rate.

    ``confirmed`` is latching and is the field everything reads (F61): once a
    pedestrian has been confirmed, a temporary occlusion should not un-confirm
    them.
    """

    def __init__(self, iou_match: float = 0.40, confirm_hits: int = 3,
                 max_missed: int = 5) -> None:
        self.iou_match = float(iou_match)
        self.confirm_hits = int(confirm_hits)
        self.max_missed = int(max_missed)
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> List[Track]:
        # F60: vectorised pairwise IoU replaces the O(T x D) Python loop.
        n_det = int(boxes.shape[0])
        if n_det == 0:
            for t in self.tracks:
                t.missed += 1
            self.tracks = [t for t in self.tracks if t.missed < self.max_missed]
            return self.tracks

        if self.tracks:
            track_boxes = np.stack([t.box for t in self.tracks])   # (T, 4)
            M = _iou_matrix(track_boxes,
                            np.ascontiguousarray(boxes, dtype=np.float32))
            ti, di = np.nonzero(M >= self.iou_match)
            order = np.argsort(-M[ti, di])                        # greedy desc
            used_t: set = set()
            used_d: set = set()
            for idx in order:
                t_i, d_i = int(ti[idx]), int(di[idx])
                if t_i in used_t or d_i in used_d:
                    continue
                used_t.add(t_i)
                used_d.add(d_i)
                tr = self.tracks[t_i]
                tr.box = boxes[d_i].copy()
                tr.score = float(scores[d_i])
                tr.frame_count += 1
                tr.missed = 0
                if tr.frame_count >= self.confirm_hits:
                    tr.confirmed = True
            for t_i, tr in enumerate(self.tracks):
                if t_i not in used_t:
                    tr.missed += 1
                    if not tr.confirmed:
                        tr.frame_count = 0
            self.tracks = [t for t in self.tracks if t.missed < self.max_missed]
            new_det_idxs = [i for i in range(n_det) if i not in used_d]
        else:
            new_det_idxs = list(range(n_det))

        for i in new_det_idxs:
            self.tracks.append(Track(box=boxes[i].copy(),
                                     score=float(scores[i]),
                                     tid=self._next_id))
            self._next_id += 1
        return self.tracks

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.confirmed]


# ===========================================================================
# NCNN engine
# ===========================================================================

class NCNNEngine:
    def __init__(self, param: str, bin_path: str, strides: Tuple[int, ...],
                 threads: int = _DEFAULT_THREADS,
                 input_blob: str = DEFAULT_INPUT_BLOB) -> None:
        try:
            import ncnn
        except ImportError as exc:
            raise SystemExit("live_nirdet.py needs ncnn: pip install ncnn") from exc
        for p in (param, bin_path):
            if not os.path.isfile(p):
                raise SystemExit(f"NCNN model file not found: {p}")
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = int(threads)
        self.net.opt.use_packing_layout = True
        self.net.opt.use_int8_inference = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = True      # A76 has FEAT_FP16
        self.net.opt.lightmode = True
        # Persistent pool allocators: re-allocating activation buffers every
        # frame costs several ms on a bandwidth-limited board.
        self._blob_pool = ncnn.PoolAllocator()
        self._ws_pool = ncnn.PoolAllocator()
        self.net.opt.blob_allocator = self._blob_pool
        self.net.opt.workspace_allocator = self._ws_pool
        self.net.load_param(param)
        self.net.load_model(bin_path)
        self.strides = tuple(int(s) for s in strides)
        self.input_blob = str(input_blob)
        self.blob_names = [f"{p}{s}" for s in self.strides
                           for p in ("cls", "off", "size")]
        self._ncnn = ncnn
        print(f"[ncnn] {param} / {bin_path}, {threads} threads, INT8")
        print(f"[ncnn] input '{self.input_blob}', {len(self.blob_names)} "
              f"output blobs: {self.blob_names}")

    def infer(self, canvas_u8: np.ndarray) -> Dict[str, np.ndarray]:
        """
        canvas_u8: (H, W) uint8 -> dict of blob arrays (C, H, W).

        UINT8 IN (F62). norm=1/255 in the Mat makes the graph's runtime input
        float [0,1] — identical to the ORT feed and to preprocess_frame's
        output — so converting the canvas to float and back cost two full
        passes over 147k pixels per frame for nothing.
        """
        ncnn = self._ncnn
        u8 = np.ascontiguousarray(canvas_u8, dtype=np.uint8)
        mat = ncnn.Mat.from_pixels(u8, ncnn.Mat.PixelType.PIXEL_GRAY,
                                   u8.shape[1], u8.shape[0])
        mat.substract_mean_normalize([0.0], [1.0 / 255.0])
        ex = self.net.create_extractor()
        ex.input(self.input_blob, mat)
        out: Dict[str, np.ndarray] = {}
        for name in self.blob_names:
            ret, m = ex.extract(name)
            if ret != 0:
                raise RuntimeError(f"NCNN extract failed for blob '{name}'")
            a = np.array(m)
            if a.ndim == 2:
                a = a[None]
            out[name] = a.astype(np.float32)
        return out


# ===========================================================================
# preprocessing (uint8 canvas — the engine normalises)
# ===========================================================================

def preprocess_u8(frame_u8: np.ndarray, out_h: int, out_w: int,
                  clahe_enabled: bool, clahe_clip: float, clahe_grid: int,
                  flat_field: Optional[np.ndarray]
                  ) -> Tuple[np.ndarray, float, int, int]:
    """
    The deployment chain, stopping one step short of /255.

    preprocess.preprocess_frame returns float32 [0,1]; NCNNEngine.infer wants
    uint8 and applies 1/255 itself. Steps 2 and 3 run at RAW resolution
    because both are illumination corrections defined against the sensor's
    own geometry — reordering them past the letterbox changes the CLAHE tile
    grid and silently breaks the hashed preprocessing contract.
    """
    if frame_u8.ndim == 3:
        frame_u8 = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY)
    img = apply_flat_field(frame_u8, flat_field)
    if clahe_enabled:
        img = apply_clahe(img, clip=float(clahe_clip), grid=int(clahe_grid))
    canvas, scale, pad_x, pad_y = letterbox(img, out_h, out_w)
    return np.ascontiguousarray(canvas, dtype=np.uint8), scale, pad_x, pad_y


# ===========================================================================
# capture sources
# ===========================================================================

class VideoSource:
    """cv2.VideoCapture over a file, a device index or a GStreamer pipeline."""

    def __init__(self, spec: str) -> None:
        src = int(spec) if str(spec).isdigit() else str(spec)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open video source {spec!r}")
        self.name = f"cv2:{spec}"

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


class PiCameraSource:
    """
    picamera2, configured for YUV420 and read as the Y PLANE ONLY.

    The sensor is monochrome-relevant here (850 nm reflective NIR through an
    IR-pass filter), so the chroma planes are wasted bandwidth and a BGR
    conversion would cost a full colour-space pass per frame just to be
    averaged back down to grey.
    """

    def __init__(self, width: int = 1280, height: int = 720,
                 fps: int = 30) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise SystemExit(
                "no --video given and picamera2 is not installed. Install "
                "picamera2, or pass --video <file|index>.") from exc
        self.cam = Picamera2()
        cfgn = self.cam.create_video_configuration(
            main={"size": (int(width), int(height)), "format": "YUV420"},
            controls={"FrameRate": float(fps)})
        self.cam.configure(cfgn)
        self.cam.start()
        time.sleep(0.5)                      # AE/AGC settle
        self.h = int(height)
        self.name = f"picamera2:{width}x{height}@{fps}"

    def read(self) -> Optional[np.ndarray]:
        yuv = self.cam.capture_array("main")
        return yuv[: self.h, :]              # Y plane, no copy

    def close(self) -> None:
        self.cam.stop()


# ===========================================================================
# threads
# ===========================================================================

_STOP = object()


def capture_thread(src, q: "queue.Queue", stop: threading.Event,
                   stats: dict) -> None:
    """DROPS frames rather than blocking: a slow consumer must never
    back-pressure the sensor, or latency grows without bound."""
    try:
        while not stop.is_set():
            frame = src.read()
            if frame is None:
                break
            try:
                q.put_nowait(frame)
            except queue.Full:
                stats["dropped"] += 1
    finally:
        stop.set()
        try:
            q.put_nowait(_STOP)
        except queue.Full:
            pass


def inference_thread(engine: NCNNEngine, in_q: "queue.Queue",
                     out_q: "queue.Queue", stop: threading.Event,
                     pp: dict, dec: dict, stats: dict) -> None:
    try:
        while not stop.is_set():
            try:
                frame = in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is _STOP:
                break
            t0 = time.perf_counter()
            canvas, scale, pad_x, pad_y = preprocess_u8(frame, **pp)
            t1 = time.perf_counter()
            blobs = engine.infer(canvas)
            t2 = time.perf_counter()
            boxes, scores = decode_frame(blobs, **dec)
            t3 = time.perf_counter()

            stats["n"] += 1
            stats["t_pre"] += t1 - t0
            stats["t_net"] += t2 - t1
            stats["t_dec"] += t3 - t2
            try:
                out_q.put_nowait((canvas, boxes, scores))
            except queue.Full:
                stats["display_dropped"] += 1
    finally:
        stop.set()
        try:
            out_q.put_nowait(_STOP)
        except queue.Full:
            pass


def display_loop(out_q: "queue.Queue", stop: threading.Event,
                 tracker: IoUTracker, stats: dict, show: bool) -> None:
    """Runs on the MAIN thread: OpenCV HighGUI is not thread-safe and a
    window created off-main silently fails to receive events on some Pi
    display stacks."""
    last = time.perf_counter()
    fps = 0.0
    while not stop.is_set():
        try:
            item = out_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is _STOP:
            break
        canvas, boxes, scores = item
        tracks = tracker.update(boxes, scores)

        now = time.perf_counter()
        dt = now - last
        last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        if not show:
            continue
        vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for t in tracks:
            if not t.confirmed:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in t.box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"#{t.tid} {t.score:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                        cv2.LINE_AA)
        n_conf = sum(1 for t in tracks if t.confirmed)
        cv2.putText(vis, f"{fps:5.1f} fps  {n_conf} confirmed  "
                         f"drop {stats['dropped']}",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 255), 1, cv2.LINE_AA)
        cv2.imshow("nirdet", vis)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    stop.set()


# ===========================================================================
# bench
# ===========================================================================

def run_bench(engine: NCNNEngine, pp: dict, dec: dict, iters: int) -> int:
    """Single-threaded timing on a synthetic canvas: preprocess, network and
    decode reported separately, because only the network is INT8-bound and
    only preprocess scales with the raw sensor resolution."""
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, size=(720, 1280), dtype=np.uint8)
    t_pre = t_net = t_dec = 0.0
    n_box = 0
    for i in range(iters + 5):
        t0 = time.perf_counter()
        canvas, _, _, _ = preprocess_u8(raw, **pp)
        t1 = time.perf_counter()
        blobs = engine.infer(canvas)
        t2 = time.perf_counter()
        boxes, _ = decode_frame(blobs, **dec)
        t3 = time.perf_counter()
        if i >= 5:                      # discard warm-up
            t_pre += t1 - t0
            t_net += t2 - t1
            t_dec += t3 - t2
            n_box += int(boxes.shape[0])
    n = float(iters)
    tot = (t_pre + t_net + t_dec) / n * 1e3
    print(f"\n[bench] {iters} iterations, {engine.net.opt.num_threads} threads")
    print(f"  preprocess : {t_pre / n * 1e3:7.2f} ms")
    print(f"  network    : {t_net / n * 1e3:7.2f} ms")
    print(f"  decode+nms : {t_dec / n * 1e3:7.2f} ms")
    print(f"  total      : {tot:7.2f} ms  ({1000.0 / max(tot, 1e-6):.1f} fps)")
    print(f"  mean boxes : {n_box / n:.1f} (random input: expect ~0)")
    return 0


# ===========================================================================
# main
# ===========================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="NIRDet-Lite Pi 5 runtime (NCNN INT8)")
    ap.add_argument("--param", required=True, help="NCNN .param")
    ap.add_argument("--bin", dest="bin_path", required=True, help="NCNN .bin")
    ap.add_argument("--contract", default=None,
                    help=".contract.json sidecar written by export_ncnn.py")
    ap.add_argument("--profile", default=None,
                    help="dataset profile YAML (supplies canvas, strides, "
                         "CLAHE settings and deploy_score_thresh)")
    ap.add_argument("--video", default=None,
                    help="video file, device index, or GStreamer pipeline; "
                         "omit to use picamera2")
    ap.add_argument("--score-thresh", type=float, default=None,
                    help="overrides the profile's measured "
                         "deploy_score_thresh")
    ap.add_argument("--iou-thresh", type=float, default=None)
    ap.add_argument("--max-det", type=int, default=None)
    ap.add_argument("--flat-field", default=None,
                    help="flat-field map; must be the SAME FILE used at "
                         "export or the contract hash will not match")
    ap.add_argument("--threads", type=int, default=_DEFAULT_THREADS,
                    help=f"ncnn thread count. Must equal NIRDET_THREADS env var "
                         f"(currently {_DEFAULT_THREADS}). To change both at once: "
                         f"NIRDET_THREADS=4 python live_nirdet.py --threads 4 ...")
    ap.add_argument("--input-blob", default=DEFAULT_INPUT_BLOB)
    ap.add_argument("--bench", action="store_true",
                    help="synthetic timing loop, no camera, no window")
    ap.add_argument("--bench-iters", type=int, default=100)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--cam-fps", type=int, default=30)
    return ap.parse_args(argv)


def check_omp_threads(threads: int) -> None:
    """
    AGREE OR FAIL (F58).

    OMP_NUM_THREADS is read by libgomp at load time, which happened at the
    top of this module — long before argparse. If --threads disagrees with
    it, ncnn's own pool uses --threads while every OpenMP region inside ncnn
    uses the environment value, so the process oversubscribes the three
    inference cores and the benchmark measures scheduler contention. There is
    no way to fix it after the fact, so refuse to run.
    """
    env = os.environ.get("OMP_NUM_THREADS")
    if env is None:
        return
    if int(env) == int(threads):
        print(f"[threads] OMP_NUM_THREADS={env} == --threads={threads}")
        return
    raise SystemExit(
        f"THREAD COUNT DISAGREEMENT\n"
        f"  OMP_NUM_THREADS = {env}   (set at module import from NIRDET_THREADS)\n"
        f"  --threads       = {threads}\n"
        f"These must match. libgomp read OMP_NUM_THREADS at import time and\n"
        f"cannot be changed afterwards. Set both together:\n"
        f"  NIRDET_THREADS={threads} python live_nirdet.py --threads {threads} "
        f"{' '.join(sys.argv[1:])}")


def main(argv=None) -> int:
    args = parse_args(argv)
    check_omp_threads(args.threads)

    cfg = get_config()
    if args.profile:
        _apply_profile_lite(cfg, args.profile)
    else:
        print("[profile] WARNING no --profile: using config defaults for "
              "canvas, strides and preprocessing. The contract check below "
              "is the only thing that will catch a mismatch.")
    if args.flat_field:
        cfg.aug.flat_field_path = args.flat_field

    strides = tuple(int(s) for s in cfg.model.strides)
    contract = load_contract(args.contract)
    verify_contract(contract, cfg, strides)

    # THE one place a threshold is resolved. No literal exists in this file:
    # CLI beats profile, and absent both is a hard error, because an
    # invented default silently changes the operating point that evaluate.py
    # measured on the report split.
    score_thresh = args.score_thresh
    if score_thresh is None:
        score_thresh = cfg.eval.deploy_score_thresh
    if score_thresh is None:
        raise SystemExit(
            "no score threshold. It is a MEASURED property of the model and "
            "dataset (evaluate.py writes deploy_score_thresh into the "
            "dataset profile), not a constant. Pass --profile with a "
            "measured profile, or --score-thresh explicitly.")
    iou_thresh = (args.iou_thresh if args.iou_thresh is not None
                  else cfg.model.nms_iou_thresh)
    max_det = args.max_det if args.max_det is not None else cfg.model.max_det
    print(f"[decode] score_thresh {score_thresh:.4f}  iou {iou_thresh:.2f}  "
          f"max_det {max_det}")

    flat = load_flat_field(cfg.aug.flat_field_path)
    pp = dict(out_h=int(cfg.data.img_h), out_w=int(cfg.data.img_w),
              clahe_enabled=bool(cfg.aug.clahe_enabled),
              clahe_clip=float(cfg.aug.clahe_clip),
              clahe_grid=int(cfg.aug.clahe_grid),
              flat_field=flat)
    dec = dict(strides=strides, img_h=int(cfg.data.img_h),
               img_w=int(cfg.data.img_w), score_thresh=float(score_thresh),
               iou_thresh=float(iou_thresh), max_det=int(max_det))

    engine = NCNNEngine(args.param, args.bin_path, strides,
                        threads=int(args.threads), input_blob=args.input_blob)

    if args.bench:
        return run_bench(engine, pp, dec, int(args.bench_iters))

    src = (VideoSource(args.video) if args.video
           else PiCameraSource(args.cam_width, args.cam_height, args.cam_fps))
    print(f"[source] {src.name}")

    tracker = IoUTracker(iou_match=cfg.eval.track_iou_match,
                         confirm_hits=cfg.eval.track_confirm_hits,
                         max_missed=cfg.eval.track_max_missed)
    print(f"[track] iou_match {cfg.eval.track_iou_match} "
          f"confirm_hits {cfg.eval.track_confirm_hits} "
          f"max_missed {cfg.eval.track_max_missed}")

    cap_q: "queue.Queue" = queue.Queue(maxsize=2)
    out_q: "queue.Queue" = queue.Queue(maxsize=2)
    stop = threading.Event()
    stats = {"n": 0, "dropped": 0, "display_dropped": 0,
             "t_pre": 0.0, "t_net": 0.0, "t_dec": 0.0}

    t_cap = threading.Thread(target=capture_thread,
                             args=(src, cap_q, stop, stats), daemon=True)
    t_inf = threading.Thread(target=inference_thread,
                             args=(engine, cap_q, out_q, stop, pp, dec, stats),
                             daemon=True)
    t_cap.start()
    t_inf.start()
    try:
        display_loop(out_q, stop, tracker, stats, show=not args.no_display)
    except KeyboardInterrupt:
        print("\n[main] interrupted")
    finally:
        stop.set()
        t_cap.join(timeout=2.0)
        t_inf.join(timeout=2.0)
        src.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    n = max(stats["n"], 1)
    print(f"\n[stats] {stats['n']} frames inferred, "
          f"{stats['dropped']} capture drops, "
          f"{stats['display_dropped']} display drops")
    print(f"  preprocess {stats['t_pre'] / n * 1e3:.2f} ms  "
          f"network {stats['t_net'] / n * 1e3:.2f} ms  "
          f"decode {stats['t_dec'] / n * 1e3:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
