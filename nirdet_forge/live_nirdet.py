"""
live_nirdet.py — Raspberry Pi 5 runtime (NCNN INT8)
====================================================
    python live_nirdet.py --profile datasets/miniNIRPed_261.yaml \
                          --param nirdet.param --bin nirdet.bin
    python live_nirdet.py --bench --param nirdet.param --bin nirdet.bin
    python live_nirdet.py --video clip.mp4 --score-thresh 0.42 ...

THREE THREADS
-------------
    capture    picamera2 YUV420 -> Y plane (or cv2.VideoCapture for --video)
    inference  preprocessing + NCNN INT8 + decode + NMS
    display    IoU tracker + OpenCV window

Both hand-offs are queue.Queue(maxsize=2). Capture DROPS a frame rather than
blocking: on a live camera a stale frame is worse than no frame, and blocking
the capture thread makes the camera's own buffer queue grow until latency is
measured in seconds.

NO THRESHOLD LITERAL IN THIS FILE
---------------------------------
The score threshold is either read from a dataset profile
(deploy_score_thresh, measured by evaluate.py as the best-F1 point on the
report split) or given explicitly with --score-thresh. Neither present is an
argparse error. A hardcoded default would be a number tuned on whatever
dataset happened to be current when this file was written.

DECODE
------
The four decode constants are imported from config.py. They are identical in
losses.AnchorGeometry.decode, head.PedestrianHead.forward,
evaluate_onnx.decode_onnx_outputs and nirdet_pp.c, and
test_decode_contract.py greps this file to prove it.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# OpenMP reads these at library init, so they MUST be set BEFORE ncnn is
# imported anywhere in the process. Cores 0-2 run inference; core 3 is left
# to capture, display and the display server.
os.environ.setdefault("OMP_NUM_THREADS",
                      os.environ.get("NIRDET_THREADS", "3"))
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

import numpy as np

import cv2

cv2.setNumThreads(1)

from config import (DECODE_OFFSET_BIAS, DECODE_OFFSET_SCALE,
                    REG_LOG_CLAMP_MAX, REG_LOG_CLAMP_MIN, STRIDES, get_config)
from dataset import apply_clahe, apply_flat_field, letterbox, load_flat_field

BLOB_INPUT = "images"


# ===========================================================================
# math helpers
# ===========================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Two-branch overflow-safe sigmoid.

    The naive 1/(1+exp(-x)) overflows for x <= -750 and emits a
    RuntimeWarning well before that. INT8 logits dequantised near the ends of
    their range reach a few hundred, so this is not hypothetical.
    """
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def greedy_nms(boxes_xyxy: np.ndarray, scores: np.ndarray,
               iou_thresh: float) -> np.ndarray:
    """
    Pure-NumPy greedy NMS. Returns kept indices, score-descending.

    Deliberately not cv2.dnn.NMSBoxes: its argument order, its box format
    (xywh vs xyxy) and its return shape have all changed between OpenCV
    releases, and the Pi's OpenCV comes from whatever the distro ships.
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
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
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

        cls  (1, H, W)
        off  (2, H, W)   channel 0 = t_cx, channel 1 = t_cy
        size (2, H, W)   channel 0 = t_w,  channel 1 = t_h

    Formula, byte-identical to losses.AnchorGeometry.decode,
    head.PedestrianHead.forward and nirdet_pp.c:

        cx = (S * sigmoid(t_cx) - B + col) * stride
        cy = (S * sigmoid(t_cy) - B + row) * stride
        w  = exp(clamp(t_w, MIN, MAX)) * img_w
        h  = exp(clamp(t_h, MIN, MAX)) * img_h

    img_w and img_h are the FULL canvas extent and are level-independent, so
    the size regression decodes against the same reference at every stride.
    """
    cls_blob = np.asarray(cls_blob, dtype=np.float32)
    off_blob = np.asarray(off_blob, dtype=np.float32)
    size_blob = np.asarray(size_blob, dtype=np.float32)
    if cls_blob.ndim == 3:
        cls_blob = cls_blob[0]
    h, w = cls_blob.shape[-2:]

    # sigmoid is monotonic: threshold the RAW LOGIT to skip expf() on ~3000
    # rejected cells per level per frame.
    st = min(max(float(score_thresh), 1e-6), 1.0 - 1e-6)
    logit_th = float(np.log(st / (1.0 - st)))
    m = cls_blob >= logit_th
    if not bool(m.any()):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    rows, cols = np.nonzero(m)
    conf = _sigmoid(cls_blob[rows, cols])
    fr = rows.astype(np.float32)
    fc = cols.astype(np.float32)

    t_cx = off_blob[0][rows, cols]
    t_cy = off_blob[1][rows, cols]
    t_w = size_blob[0][rows, cols]
    t_h = size_blob[1][rows, cols]

    cx = (DECODE_OFFSET_SCALE * _sigmoid(t_cx) - DECODE_OFFSET_BIAS + fc) * float(stride)
    cy = (DECODE_OFFSET_SCALE * _sigmoid(t_cy) - DECODE_OFFSET_BIAS + fr) * float(stride)
    bw = np.exp(np.clip(t_w, REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX)) * float(img_w)
    bh = np.exp(np.clip(t_h, REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX)) * float(img_h)

    # Same degenerate-exp filter as model.decode_predictions and
    # nirdet_decode_level in nirdet_pp.c; without it the three decoders
    # disagree on which boxes exist.
    keep = (bw > 1.0) & (bh > 1.0)
    if not bool(keep.any()):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    cx, cy, bw, bh = cx[keep], cy[keep], bw[keep], bh[keep]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                     axis=1).astype(np.float32)
    return boxes, conf[keep].astype(np.float32)


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


class IoUTracker:
    """
    Minimal IoU tracker: no motion model, no Kalman filter.

    A detection must persist for CONFIRM_HITS consecutive frames before it is
    displayed, which removes the isolated single-frame false positives that a
    per-frame threshold cannot: a spurious hot reflection fires on one frame,
    a person does not. Unconfirmed tracks are drawn in grey so the behaviour
    is visible during development instead of silently hiding detections.
    """

    IOU_MATCH = 0.4
    CONFIRM_HITS = 3
    MAX_MISSED = 5

    def __init__(self) -> None:
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> List[Track]:
        n_det = int(boxes.shape[0])
        pairs: List[Tuple[float, int, int]] = []
        for ti, tr in enumerate(self.tracks):
            for di in range(n_det):
                v = iou_xyxy(tr.box, boxes[di])
                if v >= self.IOU_MATCH:
                    pairs.append((v, ti, di))
        pairs.sort(key=lambda p: -p[0])         # greedy, descending IoU

        used_t: set = set()
        used_d: set = set()
        for _v, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr = self.tracks[ti]
            tr.box = boxes[di].copy()
            tr.score = float(scores[di])
            tr.frame_count += 1
            tr.missed = 0
            if tr.frame_count >= self.CONFIRM_HITS:
                tr.confirmed = True

        alive: List[Track] = []
        for ti, tr in enumerate(self.tracks):
            if ti in used_t:
                alive.append(tr)
                continue
            tr.missed += 1
            if tr.missed < self.MAX_MISSED:
                alive.append(tr)
        self.tracks = alive

        for di in range(n_det):
            if di in used_d:
                continue
            self.tracks.append(Track(box=boxes[di].copy(),
                                     score=float(scores[di]),
                                     tid=self._next_id))
            self._next_id += 1

        return self.tracks

    def confirmed(self) -> List[Track]:
        return [t for t in self.tracks if t.frame_count >= self.CONFIRM_HITS]


# ===========================================================================
# NCNN engine
# ===========================================================================

class NCNNEngine:
    def __init__(self, param: str, bin_path: str, strides: Tuple[int, ...],
                 threads: int = 4) -> None:
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
        # INT8 on the Cortex-A76 SDOT path; the model must have been built
        # with ncnn2int8 from the same calibration table family used for the
        # ONNX QDQ export.
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
        self.blob_names = [f"{p}{s}" for s in self.strides
                           for p in ("cls", "off", "size")]
        self._ncnn = ncnn
        print(f"[ncnn] {param} / {bin_path}, {threads} threads, INT8")
        print(f"[ncnn] {len(self.blob_names)} output blobs: {self.blob_names}")

    def infer(self, canvas: np.ndarray) -> Dict[str, np.ndarray]:
        """canvas: (H, W) float32 in [0, 1] -> dict of blob arrays (C, H, W)."""
        ncnn = self._ncnn
        # from_pixels + substract_mean_normalize is the supported construction
        # path and folds the /255 into ncnn's packed layout; building a Mat
        # around a foreign float32 buffer relies on lifetime guarantees the
        # Python bindings do not make.
        u8 = np.ascontiguousarray(
            np.clip(canvas * 255.0, 0.0, 255.0).astype(np.uint8))
        mat = ncnn.Mat.from_pixels(u8, ncnn.Mat.PixelType.PIXEL_GRAY,
                                   u8.shape[1], u8.shape[0])
        mat.substract_mean_normalize([0.0], [1.0 / 255.0])
        ex = self.net.create_extractor()
        ex.input(BLOB_INPUT, mat)
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
# pipeline
# ===========================================================================

@dataclass
class Shared:
    cap_q: "queue.Queue" = field(
        default_factory=lambda: queue.Queue(maxsize=2))
    inf_q: "queue.Queue" = field(
        default_factory=lambda: queue.Queue(maxsize=2))
    stop: threading.Event = field(default_factory=threading.Event)
    dropped: int = 0
    frames: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _put_or_drop(q: "queue.Queue", item, shared: Shared) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        with shared.lock:
            shared.dropped += 1


def _capture_camera(shared: Shared, width: int, height: int) -> None:
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("[capture] picamera2 unavailable; use --video for testing")
        shared.stop.set()
        return
    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": (width, height), "format": "YUV420"},
        buffer_count=4,
        # queue=False returns the NEWEST frame rather than the oldest. With
        # the default queue, inference slower than capture makes the buffer
        # queue grow until latency is measured in seconds.
        queue=False)
    cam.configure(cfg)
    cam.start()
    print(f"[capture] IMX708 {width}x{height} YUV420 (Y plane only)")
    try:
        while not shared.stop.is_set():
            yuv = cam.capture_array("main")
            # YUV420 planar: the first `height` rows are the full-resolution
            # luma plane. The 850 nm NoIR sensor gives a single meaningful
            # channel, so chroma is discarded without conversion cost.
            y = np.ascontiguousarray(yuv[:height, :width])
            _put_or_drop(shared.cap_q, y, shared)
    finally:
        cam.stop()
        shared.stop.set()


def _capture_video(shared: Shared, path: str, loop: bool = True) -> None:
    """Camera-free fallback for development on the laptop."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[capture] could not open {path}")
        shared.stop.set()
        return
    print(f"[capture] video {path}")
    try:
        while not shared.stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _put_or_drop(shared.cap_q, np.ascontiguousarray(frame), shared)
    finally:
        cap.release()
        shared.stop.set()


def _inference(shared: Shared, engine: NCNNEngine, img_h: int, img_w: int,
               score_thresh: float, iou_thresh: float, max_det: int,
               clahe_enabled: bool, clahe_clip: float, clahe_grid: int,
               flat_field: Optional[np.ndarray]) -> None:
    while not shared.stop.is_set():
        try:
            raw = shared.cap_q.get(timeout=0.2)
        except queue.Empty:
            continue
        t0 = time.perf_counter()

        # PREPROCESSING CONTRACT: flat-field, then CLAHE, then letterbox,
        # then /255 — the same order and the same functions as dataset.py.
        frame = apply_flat_field(raw, flat_field)
        if clahe_enabled:
            frame = apply_clahe(frame, clahe_clip, clahe_grid)
        canvas_u8, scale, pad_x, pad_y = letterbox(frame, img_h, img_w)
        canvas = canvas_u8.astype(np.float32) / 255.0

        blobs = engine.infer(canvas)

        all_b: List[np.ndarray] = []
        all_s: List[np.ndarray] = []
        for s in engine.strides:
            b, sc = decode_level(blobs[f"cls{s}"], blobs[f"off{s}"],
                                 blobs[f"size{s}"], s, img_h, img_w,
                                 score_thresh)
            if b.shape[0]:
                all_b.append(b)
                all_s.append(sc)

        if all_b:
            boxes = np.concatenate(all_b, 0)
            scores = np.concatenate(all_s, 0)
            keep = greedy_nms(boxes, scores, iou_thresh)[:max_det]
            boxes, scores = boxes[keep], scores[keep]
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_w)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_h)
        else:
            boxes = np.zeros((0, 4), np.float32)
            scores = np.zeros((0,), np.float32)

        ms = (time.perf_counter() - t0) * 1000.0
        with shared.lock:
            shared.frames += 1
        _put_or_drop(shared.inf_q, (canvas_u8, boxes, scores, ms), shared)


def _display(shared: Shared, headless: bool = False) -> None:
    tracker = IoUTracker()
    ema_ms = None
    t_last = time.perf_counter()
    fps = 0.0
    while not shared.stop.is_set():
        try:
            canvas_u8, boxes, scores, ms = shared.inf_q.get(timeout=0.2)
        except queue.Empty:
            continue

        tracker.update(boxes, scores)
        ema_ms = ms if ema_ms is None else 0.9 * ema_ms + 0.1 * ms
        now = time.perf_counter()
        dt = now - t_last
        t_last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

        if headless:
            n_conf = len(tracker.confirmed())
            print(f"\r[live] {fps:5.1f} fps  {ema_ms:6.2f} ms  "
                  f"confirmed {n_conf:2d}  tracks {len(tracker.tracks):2d}  "
                  f"dropped {shared.dropped}", end="", flush=True)
            continue

        vis = cv2.cvtColor(canvas_u8, cv2.COLOR_GRAY2BGR)
        for tr in tracker.tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.box]
            if tr.frame_count >= IoUTracker.CONFIRM_HITS:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 230, 0), 2)
                cv2.putText(vis, f"#{tr.tid} {tr.score:.2f}",
                            (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 230, 0), 1)
            else:
                # Grey: seen but not yet confirmed. Visible on purpose.
                cv2.rectangle(vis, (x1, y1), (x2, y2), (130, 130, 130), 1)
        with shared.lock:
            dropped = shared.dropped
        cv2.putText(vis, f"{fps:.1f} fps  {ema_ms:.1f} ms  "
                         f"conf {len(tracker.confirmed())}  drop {dropped}",
                    (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow("NIRDet-Lite", vis)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            shared.stop.set()
    if not headless:
        cv2.destroyAllWindows()
    else:
        print()


# ===========================================================================
# bench
# ===========================================================================

def bench(engine: NCNNEngine, img_h: int, img_w: int, n: int = 100) -> None:
    """100 passes on a black frame. No camera, no display, no dataset."""
    canvas = np.zeros((img_h, img_w), dtype=np.float32)
    for _ in range(10):
        engine.infer(canvas)                 # warm the thread pool
    times: List[float] = []
    for _ in range(int(n)):
        t0 = time.perf_counter()
        engine.infer(canvas)
        times.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times)
    print(f"[bench] {n} passes on a {img_h}x{img_w} black frame")
    print(f"[bench] avg {a.mean():.2f} ms   min {a.min():.2f} ms   "
          f"max {a.max():.2f} ms   p95 {np.percentile(a, 95):.2f} ms")
    print(f"[bench] -> {1000.0 / a.mean():.1f} fps inference-only")


# ===========================================================================
# main
# ===========================================================================

def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(
        description="NIRDet-Lite live runtime (Raspberry Pi 5, NCNN INT8).")
    ap.add_argument("--param", default="nirdet.param")
    ap.add_argument("--bin", dest="bin_path", default="nirdet.bin")
    ap.add_argument("--profile", default=None,
                    help="dataset profile YAML; supplies deploy_score_thresh "
                         "as measured by evaluate.py")
    ap.add_argument("--score-thresh", type=float, default=None,
                    help="explicit threshold; overrides the profile")
    ap.add_argument("--iou-thresh", type=float,
                    default=float(cfg.model.nms_iou_thresh))
    ap.add_argument("--max-det", type=int, default=int(cfg.model.max_det))
    ap.add_argument("--flat-field", default=cfg.aug.flat_field_path)
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--video", default=None,
                    help="video file instead of the camera (testing)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--bench", action="store_true",
                    help="100 inference passes on a black frame; no camera")
    args = ap.parse_args()

    img_h, img_w = int(cfg.data.img_h), int(cfg.data.img_w)
    strides = tuple(cfg.model.strides) or STRIDES

    if args.bench:
        engine = NCNNEngine(args.param, args.bin_path, strides, args.threads)
        bench(engine, img_h, img_w, 100)
        return 0

    # ---- threshold resolution: no literal in this file ----
    score_thresh: Optional[float] = args.score_thresh
    if score_thresh is None and args.profile:
        from dataset_profiles import DatasetProfile
        prof = DatasetProfile.load(args.profile)
        prof.apply(cfg, verbose=False)
        score_thresh = float(prof.deploy_score_thresh)
        print(f"[thresh] deploy_score_thresh {score_thresh:.3f} from "
              f"{args.profile} (best-F1 point measured by evaluate.py on the "
              f"'{cfg.eval.report_split}' split)")
    if score_thresh is None:
        ap.error(
            "no score threshold. This file deliberately contains no hardcoded "
            "threshold, because the right value is a property of the dataset "
            "and the trained model, not of the runtime. Supply one of:\n"
            "  --profile <dataset.yaml>   read deploy_score_thresh, which "
            "evaluate.py measures as the best-F1 threshold on the report "
            "split and writes back into the profile;\n"
            "  --score-thresh <float>     an explicit value, for sweeping or "
            "for a deployment that wants a different precision/recall "
            "trade-off than max F1.")
    else:
        print(f"[thresh] using {score_thresh:.3f}")

    flat_field = load_flat_field(args.flat_field)
    if flat_field is not None:
        print(f"[prep] flat-field {args.flat_field} "
              f"(mean gain {float(flat_field.mean()):.3f})")
    clahe_enabled = bool(cfg.aug.clahe_enabled) and not args.no_clahe
    print(f"[prep] clahe={clahe_enabled} clip {cfg.aug.clahe_clip} "
          f"grid {cfg.aug.clahe_grid}; canvas {img_h}x{img_w}")

    engine = NCNNEngine(args.param, args.bin_path, strides, args.threads)
    shared = Shared()

    if args.video:
        t_cap = threading.Thread(target=_capture_video,
                                 args=(shared, args.video), daemon=True)
    else:
        t_cap = threading.Thread(
            target=_capture_camera,
            args=(shared, args.cam_width, args.cam_height), daemon=True)
    t_inf = threading.Thread(
        target=_inference,
        args=(shared, engine, img_h, img_w, float(score_thresh),
              float(args.iou_thresh), int(args.max_det), clahe_enabled,
              float(cfg.aug.clahe_clip), int(cfg.aug.clahe_grid), flat_field),
        daemon=True)

    t_cap.start()
    t_inf.start()
    try:
        _display(shared, headless=bool(args.headless))
    except KeyboardInterrupt:
        pass
    finally:
        shared.stop.set()
        t_cap.join(timeout=2.0)
        t_inf.join(timeout=2.0)

    print(f"[live] {shared.frames} frames inferred, "
          f"{shared.dropped} dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
