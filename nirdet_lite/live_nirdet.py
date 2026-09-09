#!/usr/bin/env python3
"""
live_nirdet.py — live annotated NIR feed on Raspberry Pi 5 (NCNN INT8)
=======================================================================
Three threads, because two is not enough: cv2.imshow on the Pi's display
stack costs 5-15 ms and would stall the inference thread.

    capture   picamera2, YUV420, Y-plane view (zero-copy grayscale)
    infer     NCNN INT8, 3 threads, decode + NMS
    display   overlay + FPS, owns the OpenCV window

Every queue has depth 1 with OVERWRITE semantics. With an unbounded queue and
inference slower than capture, latency grows without bound and the "live" feed
drifts seconds behind reality. Dropping stale frames pins end-to-end latency
at roughly one inference period, which is what live means.

Preprocessing parity
--------------------
CLAHE -> letterbox -> /255. ``_letterbox`` is a deliberate, commented
duplicate of dataset.letterbox so the Pi runtime needs no torch or
albumentations import; the two must stay identical.

Decode parity
-------------
``decode`` mirrors head.py exactly:
    cx = (OFF_S * sigmoid(t_cx) - OFF_B + col) * stride
    cy = (OFF_S * sigmoid(t_cy) - OFF_B + row) * stride
    w  = exp(clamp(t_w)) * img_w
    h  = exp(clamp(t_h)) * img_h
The score filter is applied to the RAW LOGIT via a precomputed logit
threshold, so sigmoid/exp run on the few surviving cells instead of all ~4800.

Setup
-----
    sudo apt install -y python3-picamera2 python3-opencv
    # build NCNN with -DNCNN_ARM82=ON -DNCNN_ARM82DOT=ON -DNCNN_PYTHON=ON
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    vcgencmd get_throttled          # must be 0x0 or your numbers are noise
    taskset -c 0-2 python3 live_nirdet.py
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Thread/affinity setup MUST happen before NCNN is imported: OpenMP reads
# these at library init. Cores 0-2 run inference, core 3 stays free for
# capture + display + the display server.
# ---------------------------------------------------------------------------
INFER_THREADS = int(os.environ.get("NIRDET_THREADS", "3"))
os.environ.setdefault("OMP_NUM_THREADS", str(INFER_THREADS))
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")
cv2.setNumThreads(1)

try:
    import ncnn
except ImportError:                                          # pragma: no cover
    raise SystemExit("ncnn python module not found. Build NCNN with "
                     "-DNCNN_PYTHON=ON -DNCNN_ARM82DOT=ON and install it.")

# ---------------------------------------------------------------------------
# DECODE CONTRACT — must equal config.py / head.py / losses.py / nirdet_pp.c
# ---------------------------------------------------------------------------
OFF_S = 2.0
OFF_B = 0.5
REG_CLAMP_MIN = -6.0
REG_CLAMP_MAX = 1.0

DEFAULT_STRIDES = (8, 16, 32)
DEFAULT_OUTPUTS = ("cls3", "reg3", "cls4", "reg4", "cls5", "reg5")


# ---------------------------------------------------------------------------
# preprocessing (duplicate of dataset.letterbox — keep in sync)
# ---------------------------------------------------------------------------

def _letterbox(img: np.ndarray, out_h: int, out_w: int,
               pad_value: int = 0) -> Tuple[np.ndarray, float, int, int]:
    h0, w0 = img.shape[:2]
    scale = min(out_w / float(w0), out_h / float(h0))
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((out_h, out_w), pad_value, dtype=np.uint8)
    left = (out_w - nw) // 2
    top = (out_h - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic, matching nirdet_pp.c's nirdet_sigmoidf().

    np.exp(-x) overflows for x < -88 in float32 and emits a RuntimeWarning
    every frame. t_cx / t_cy are never clamped before sigmoid, so an INT8
    outlier can reach this path.
    """
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0.0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


# ---------------------------------------------------------------------------
# decode — bit-for-bit head.py
# ---------------------------------------------------------------------------

def decode(cls_np: np.ndarray, reg_np: np.ndarray, stride: int,
           logit_th: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    cls_np : (1, H, W) or (H, W) raw logits
    reg_np : (4, H, W) raw (t_cx, t_cy, t_w, t_h) with the level scale folded in
    ->       (N, 4) xyxy pixels, (N,) confidences in [0, 1]

    Thresholding on the raw logit turns the hot loop into one comparison per
    cell instead of ~4800 exp() calls.
    """
    cls2d = cls_np[0] if cls_np.ndim == 3 else cls_np
    h, w = cls2d.shape

    ys, xs = np.nonzero(cls2d >= logit_th)
    if xs.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    conf = _sigmoid(cls2d[ys, xs].astype(np.float32))
    t = reg_np[:, ys, xs].astype(np.float32)          # (4, N)

    img_w = float(w * stride)
    img_h = float(h * stride)

    cx = (OFF_S * _sigmoid(t[0]) - OFF_B + xs.astype(np.float32)) * stride
    cy = (OFF_S * _sigmoid(t[1]) - OFF_B + ys.astype(np.float32)) * stride
    bw = np.exp(np.clip(t[2], REG_CLAMP_MIN, REG_CLAMP_MAX)) * img_w
    bh = np.exp(np.clip(t[3], REG_CLAMP_MIN, REG_CLAMP_MAX)) * img_h

    keep = (bw > 1.0) & (bh > 1.0)                    # drop degenerate exp boxes
    if not keep.any():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    cx, cy, bw, bh, conf = cx[keep], cy[keep], bw[keep], bh[keep], conf[keep]

    boxes = np.stack([cx - bw * 0.5, cy - bh * 0.5,
                      cx + bw * 0.5, cy + bh * 0.5], 1).astype(np.float32)
    np.clip(boxes[:, 0::2], 0.0, img_w, out=boxes[:, 0::2])
    np.clip(boxes[:, 1::2], 0.0, img_h, out=boxes[:, 1::2])
    return boxes, conf


def nms(boxes: np.ndarray, scores: np.ndarray, score_th: float,
        iou_th: float, max_det: int) -> Tuple[np.ndarray, np.ndarray]:
    if boxes.shape[0] == 0:
        return boxes, scores
    rects = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
             for x1, y1, x2, y2 in boxes]
    idx = cv2.dnn.NMSBoxes(rects, scores.tolist(), float(score_th),
                           float(iou_th))
    if idx is None or len(idx) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
    idx = np.asarray(idx).reshape(-1)[:max_det]
    return boxes[idx], scores[idx]


# ---------------------------------------------------------------------------
# queues with overwrite semantics
# ---------------------------------------------------------------------------

def put_latest(q: "queue.Queue", item) -> None:
    """Drop the stale frame, keep latency bounded."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

class NIRDetNCNN:
    def __init__(self, param: str, bin_: str, img_h: int, img_w: int,
                 strides: Tuple[int, ...], out_names: Tuple[str, ...],
                 threads: int = 3, int8: bool = True,
                 input_name: str = "img") -> None:
        for p in (param, bin_):
            if not os.path.exists(p):
                raise SystemExit(f"missing NCNN file: {p}")

        self.net = ncnn.Net()
        opt = self.net.opt
        opt.use_int8_inference = bool(int8)
        opt.use_fp16_packed = True
        opt.use_fp16_storage = True
        opt.use_fp16_arithmetic = True        # A76 has FEAT_FP16
        opt.use_packing_layout = True         # NC4HW4 tiling: cache friendly
        opt.use_winograd_convolution = True
        opt.use_sgemm_convolution = True
        opt.lightmode = True                  # free intermediates eagerly
        opt.num_threads = int(threads)

        # Persistent pool allocators: without them every frame re-allocates
        # and re-faults the activation buffers, which is several ms per frame
        # on a memory-bandwidth-limited board.
        self._blob_pool = ncnn.PoolAllocator()
        self._ws_pool = ncnn.PoolAllocator()
        opt.blob_allocator = self._blob_pool
        opt.workspace_allocator = self._ws_pool

        self.net.load_param(param)
        self.net.load_model(bin_)

        self.img_h, self.img_w = int(img_h), int(img_w)
        self.strides = tuple(int(s) for s in strides)
        self.out_names = tuple(out_names)
        self.input_name = input_name
        if len(self.out_names) != 2 * len(self.strides):
            raise SystemExit(f"{len(self.out_names)} output names for "
                             f"{len(self.strides)} strides; expected "
                             f"{2 * len(self.strides)}")

    def infer(self, canvas: np.ndarray, logit_th: float, score_th: float,
              iou_th: float, max_det: int
              ) -> Tuple[np.ndarray, np.ndarray, int]:
        """canvas: (img_h, img_w) uint8 letterboxed grayscale."""
        mat = ncnn.Mat.from_pixels(canvas, ncnn.Mat.PixelType.PIXEL_GRAY,
                                   self.img_w, self.img_h)
        # /255, no mean subtraction: exactly dataset.py's normalisation.
        mat.substract_mean_normalize([0.0], [1.0 / 255.0])

        ex = self.net.create_extractor()
        ex.input(self.input_name, mat)

        raw: List[np.ndarray] = []
        for name in self.out_names:
            ret, m = ex.extract(name)
            if ret != 0:
                raise RuntimeError(f"NCNN extract('{name}') failed with {ret}; "
                                   f"check the output blob names in the .param")
            raw.append(np.array(m))

        bs, ss = [], []
        for i, stride in enumerate(self.strides):
            b, s = decode(raw[2 * i], raw[2 * i + 1], stride, logit_th)
            if b.shape[0]:
                bs.append(b)
                ss.append(s)
        if not bs:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32), 0)

        boxes = np.concatenate(bs, 0)
        scores = np.concatenate(ss, 0)
        n_pre = int(boxes.shape[0])
        boxes, scores = nms(boxes, scores, score_th, iou_th, max_det)
        return boxes, scores, n_pre


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop = threading.Event()
        self.cap_q: "queue.Queue" = queue.Queue(maxsize=1)
        self.disp_q: "queue.Queue" = queue.Queue(maxsize=1)
        self.engine = NIRDetNCNN(
            args.param, args.bin, args.img_h, args.img_w,
            tuple(args.strides), tuple(args.outputs),
            threads=args.threads, int8=not args.no_int8,
            input_name=args.input_name,
        )
        self.logit_th = _logit(args.score_thresh)
        self.clahe = (cv2.createCLAHE(clipLimit=args.clahe_clip,
                                      tileGridSize=(args.clahe_grid,
                                                    args.clahe_grid))
                      if not args.no_clahe else None)
        self.dropped = 0
        self.cap_count = 0

    # ---------------- capture ----------------

    def capture_thread(self) -> None:
        """
        Captures at the MODEL resolution in YUV420 and takes the Y plane.

        The Pi 5 ISP does the downscale in hardware for free, and
        yuv[:H, :W] is a zero-copy grayscale view. Capturing at sensor
        resolution and doing cv2.resize + cvtColor in Python costs 10-25 ms
        per frame for nothing.
        """
        if self.args.video:
            self._capture_video()
            return
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("picamera2 not available; falling back to cv2.VideoCapture",
                  file=sys.stderr)
            self._capture_v4l2()
            return

        cam = Picamera2()
        cfg = cam.create_video_configuration(
            main={"size": (self.args.cam_w, self.args.cam_h),
                  "format": "YUV420"},
            buffer_count=4,
            queue=False,        # KEY: return the newest frame, not the oldest
            controls={"FrameDurationLimits": (int(1e6 / self.args.cam_fps),
                                              int(1e6 / self.args.cam_fps))},
        )
        cam.configure(cfg)
        cam.start()
        print(f"[capture] picamera2 {self.args.cam_w}x{self.args.cam_h} "
              f"YUV420 @ {self.args.cam_fps} fps (Y-plane, zero-copy)")
        try:
            while not self.stop.is_set():
                yuv = cam.capture_array("main")
                gray = yuv[:self.args.cam_h, :self.args.cam_w]
                self.cap_count += 1
                if self.cap_q.full():
                    self.dropped += 1
                put_latest(self.cap_q, gray.copy())
        finally:
            cam.stop()
            print("[capture] stopped")

    def _capture_v4l2(self) -> None:
        cap = cv2.VideoCapture(self.args.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.cam_h)
        cap.set(cv2.CAP_PROP_FPS, self.args.cam_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("[capture] cannot open camera", file=sys.stderr)
            self.stop.set()
            return
        try:
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if frame.ndim == 3 else frame)
                self.cap_count += 1
                if self.cap_q.full():
                    self.dropped += 1
                put_latest(self.cap_q, gray)
        finally:
            cap.release()

    def _capture_video(self) -> None:
        cap = cv2.VideoCapture(self.args.video)
        if not cap.isOpened():
            print(f"[capture] cannot open {self.args.video}", file=sys.stderr)
            self.stop.set()
            return
        try:
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.stop.set()
                    break
                gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if frame.ndim == 3 else frame)
                self.cap_count += 1
                put_latest(self.cap_q, gray)
                time.sleep(1.0 / max(self.args.cam_fps, 1))
        finally:
            cap.release()

    # ---------------- inference ----------------

    def infer_thread(self) -> None:
        a = self.args
        while not self.stop.is_set():
            try:
                gray = self.cap_q.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            if self.clahe is not None:
                gray = self.clahe.apply(gray)
            canvas, scale, pad_l, pad_t = _letterbox(gray, a.img_h, a.img_w)
            t_pre = (time.perf_counter() - t0) * 1e3

            t1 = time.perf_counter()
            boxes, scores, n_pre = self.engine.infer(
                canvas, self.logit_th, a.score_thresh, a.iou_thresh, a.max_det)
            t_inf = (time.perf_counter() - t1) * 1e3

            put_latest(self.disp_q, (canvas, boxes, scores,
                                     t_pre, t_inf, n_pre))
        print("[infer] stopped")

    # ---------------- display ----------------

    def display_loop(self) -> None:
        a = self.args
        fps_hist: deque = deque(maxlen=30)
        inf_hist: deque = deque(maxlen=30)
        last = time.perf_counter()
        frames = 0
        writer = None

        if not a.headless:
            cv2.namedWindow("NIRDet-Lite", cv2.WINDOW_AUTOSIZE)

        try:
            while not self.stop.is_set():
                try:
                    canvas, boxes, scores, t_pre, t_inf, n_pre = \
                        self.disp_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                now = time.perf_counter()
                dt = max(now - last, 1e-6)
                last = now
                fps_hist.append(1.0 / dt)
                inf_hist.append(t_inf)
                frames += 1

                vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
                for (x1, y1, x2, y2), s in zip(boxes.astype(int), scores):
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"{s:.2f}", (x1, max(y1 - 5, 11)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 0), 1, cv2.LINE_AA)

                fps = sum(fps_hist) / len(fps_hist)
                inf = sum(inf_hist) / len(inf_hist)
                cv2.putText(vis,
                            f"{fps:5.1f} FPS | infer {inf:5.1f} ms | "
                            f"pre {t_pre:4.1f} ms | {len(boxes)} det "
                            f"({n_pre} pre-NMS)",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(vis,
                            f"int8={not a.no_int8} thr={a.threads} "
                            f"conf>={a.score_thresh:.2f} "
                            f"drop={self.dropped}",
                            (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (0, 200, 200), 1, cv2.LINE_AA)

                if a.record:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            a.record, cv2.VideoWriter_fourcc(*"mp4v"),
                            max(int(round(fps)), 1),
                            (vis.shape[1], vis.shape[0]))
                    writer.write(vis)

                if not a.headless:
                    cv2.imshow("NIRDet-Lite", vis)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord('q'), 27):
                        self.stop.set()
                    elif k == ord('+'):
                        a.score_thresh = min(0.95, a.score_thresh + 0.05)
                        self.logit_th = _logit(a.score_thresh)
                    elif k == ord('-'):
                        a.score_thresh = max(0.05, a.score_thresh - 0.05)
                        self.logit_th = _logit(a.score_thresh)

                if a.bench_frames and frames >= a.bench_frames:
                    self.stop.set()
        finally:
            if writer is not None:
                writer.release()
            if not a.headless:
                cv2.destroyAllWindows()
            if fps_hist:
                print(f"\n[summary] frames {frames} | "
                      f"mean {sum(fps_hist) / len(fps_hist):.2f} FPS | "
                      f"mean infer {sum(inf_hist) / len(inf_hist):.2f} ms | "
                      f"captured {self.cap_count} | dropped {self.dropped}")
                print("[summary] reminder: run "
                      "`vcgencmd get_throttled` — anything other than 0x0 "
                      "means these numbers are thermally invalid.")

    # ---------------- run ----------------

    def run(self) -> int:
        threads = [
            threading.Thread(target=self.capture_thread, daemon=True,
                             name="capture"),
            threading.Thread(target=self.infer_thread, daemon=True,
                             name="infer"),
        ]
        for t in threads:
            t.start()
        try:
            self.display_loop()
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self.stop.set()
            for t in threads:
                t.join(timeout=2.0)
        return 0


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("NIRDet-Lite live NCNN pipeline")
    p.add_argument("--param", default="nirdet-int8.param")
    p.add_argument("--bin", default="nirdet-int8.bin")
    p.add_argument("--input-name", default="img")
    p.add_argument("--outputs", nargs="+", default=list(DEFAULT_OUTPUTS))
    p.add_argument("--strides", nargs="+", type=int,
                   default=list(DEFAULT_STRIDES))
    p.add_argument("--img-h", type=int, default=384)
    p.add_argument("--img-w", type=int, default=640)
    p.add_argument("--cam-h", type=int, default=720)
    p.add_argument("--cam-w", type=int, default=1280)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--video", default=None, help="file instead of a camera")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--score-thresh", type=float, default=0.4455,
                   help="detection confidence threshold; default matches "
                        "the best-F1 point from evaluate.py / miniNIRPed_261.yaml")
    p.add_argument("--iou-thresh", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=300)  # must match config.ModelCfg.max_det
    p.add_argument("--threads", type=int, default=INFER_THREADS)
    p.add_argument("--no-int8", action="store_true",
                   help="run the fp16 model (measure both: requantisation "
                        "overhead sometimes eats the SDOT gain)")
    p.add_argument("--no-clahe", action="store_true")
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--clahe-grid", type=int, default=8)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record", default=None)
    p.add_argument("--bench", dest="bench_frames", type=int, default=0,
                   help="stop after N displayed frames and print a summary")
    args = p.parse_args()

    # Warn if capture aspect ratio deviates from 16:9 training geometry.
    _train_ar = 16.0 / 9.0
    _cam_ar = args.cam_w / args.cam_h
    if abs(_cam_ar - _train_ar) > 0.05:
        print(
            f"  WARNING: camera {args.cam_w}x{args.cam_h} "
            f"(AR={_cam_ar:.3f}) differs from training geometry "
            f"16:9 (AR={_train_ar:.3f}). "
            f"Letterbox distribution will not match training. "
            f"Consider --cam-w 1280 --cam-h 720 or --cam-w 640 --cam-h 360."
        )
    return args


def main() -> int:
    a = parse_args()
    print("=" * 62)
    print("  NIRDet-Lite live pipeline")
    print("=" * 62)
    print(f"  model    : {a.param} / {a.bin}  int8={not a.no_int8}")
    print(f"  input    : {a.img_h}x{a.img_w}  strides {tuple(a.strides)}")
    print(f"  outputs  : {tuple(a.outputs)}")
    print(f"  threads  : {a.threads} (OMP_PROC_BIND="
          f"{os.environ.get('OMP_PROC_BIND')})")
    print(f"  conf/iou : {a.score_thresh} / {a.iou_thresh}   "
          f"(logit threshold {_logit(a.score_thresh):+.3f})")
    print(f"  clahe    : {not a.no_clahe}")
    print("  keys     : q/Esc quit, +/- adjust confidence")
    print("  tip      : taskset -c 0-2 python3 live_nirdet.py")
    print("=" * 62)
    return Pipeline(a).run()


if __name__ == "__main__":
    sys.exit(main())
