#!/usr/bin/env python
"""
quantize_qdq.py — ONNX -> INT8 QDQ per-channel (ST Edge AI / Neural-ART)
=========================================================================
ST's Neural-ART NPU accelerates exactly one scheme: 8-bit/8-bit, ss/sa,
per-channel, supplied either as a TFLite INT8 model or as an ONNX model in
QDQ (tensor-oriented Quantize/DeQuantize) form. Anything left in float32 is
emitted as a software kernel on the Cortex-M55. This script produces the
ONNX QDQ variant, which avoids the PyTorch->ONNX->TF->TFLite layout-transpose
pathologies entirely.

THE CRITICAL DETAIL
-------------------
The calibration reader below reproduces dataset.py's VALIDATION transform
byte-for-byte: optional CLAHE, then the deterministic letterbox
(aspect-preserving resize + centred zero pad), then /255 with NO mean
subtraction, then NCHW float32. Any mismatch here produces a 10+ point mAP
drop that looks exactly like a quantisation failure but is a preprocessing
failure. ``letterbox`` and ``apply_clahe`` are imported from dataset.py rather
than reimplemented, so they cannot drift.

Usage
-----
    python quantize_qdq.py --onnx nirdet-sim.onnx --data-root <root>
    python quantize_qdq.py --onnx nirdet-sim.onnx --data-root <root> \
        --method percentile --percentile 99.999 --exclude-first-conv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np

try:
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
        shape_inference,
    )
except ImportError as exc:                                   # pragma: no cover
    raise SystemExit("missing dependency: pip install onnx onnxruntime "
                     f"({exc})")

from config import get_config
from dataset import apply_clahe, letterbox        # single source of truth

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

CALIB_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "percentile": CalibrationMethod.Percentile,
    "entropy": CalibrationMethod.Entropy,
}


# ---------------------------------------------------------------------------
# preprocessing — identical to dataset.py's val path
# ---------------------------------------------------------------------------

def preprocess(path: str, out_h: int, out_w: int,
               use_clahe: bool, clahe_clip: float,
               clahe_grid: int) -> np.ndarray:
    """-> (1, 1, out_h, out_w) float32 in [0, 1]"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cv2 could not decode {path}")
    if use_clahe:
        img = apply_clahe(img, clahe_clip, clahe_grid)
    canvas, _, _, _ = letterbox(img, out_h, out_w, pad_value=0)
    return (canvas.astype(np.float32) / 255.0)[None, None]


class NIRCalibrationReader(CalibrationDataReader):
    """Streams calibration tensors one at a time (never loads all 300 at once)."""

    def __init__(self, files: List[str], input_name: str,
                 out_h: int, out_w: int, use_clahe: bool,
                 clahe_clip: float, clahe_grid: int) -> None:
        self.files = list(files)
        self.input_name = input_name
        self.out_h, self.out_w = int(out_h), int(out_w)
        self.use_clahe = bool(use_clahe)
        self.clahe_clip = float(clahe_clip)
        self.clahe_grid = int(clahe_grid)
        self._it: Iterator[str] = iter(self.files)
        self._n = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        path = next(self._it, None)
        if path is None:
            return None
        self._n += 1
        return {self.input_name: preprocess(
            path, self.out_h, self.out_w, self.use_clahe,
            self.clahe_clip, self.clahe_grid)}

    def rewind(self) -> None:
        self._it = iter(self.files)
        self._n = 0

    @property
    def consumed(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def collect_images(root: str, split: str, limit: int, seed: int) -> List[str]:
    d = Path(root) / "images" / split
    files: List[str] = []
    for ext in IMG_EXTS:
        files += [str(p) for p in sorted(d.glob(ext))]
    if not files:
        raise SystemExit(f"no calibration images under {d}")
    rng = np.random.default_rng(seed)
    if len(files) > limit:
        files = [files[i] for i in rng.permutation(len(files))[:limit]]
    return sorted(files)


def graph_io(path: str):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    i = sess.get_inputs()[0]
    return i.name, list(i.shape), [o.name for o in sess.get_outputs()]


def first_conv_name(path: str) -> Optional[str]:
    m = onnx.load(path)
    for n in m.graph.node:
        if n.op_type == "Conv":
            return n.name
    return None


def op_histogram(path: str) -> Dict[str, int]:
    m = onnx.load(path)
    h: Dict[str, int] = {}
    for n in m.graph.node:
        h[n.op_type] = h.get(n.op_type, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def compare_outputs(fp32_path: str, int8_path: str, sample: str,
                    input_name: str, out_h: int, out_w: int,
                    use_clahe: bool, clip: float, grid: int) -> None:
    """
    Sanity check on the CONFIDENCE logits: if the INT8 cls output has drifted
    badly, the model is broken before it ever reaches the board and it is much
    cheaper to find out here.
    """
    import onnxruntime as ort
    x = {input_name: preprocess(sample, out_h, out_w, use_clahe, clip, grid)}
    a = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    ra, rb = a.run(None, x), b.run(None, x)
    names = [o.name for o in a.get_outputs()]
    print("\n  fp32 vs INT8 on one calibration image")
    for n, u, v in zip(names, ra, rb):
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        mad = float(np.abs(u - v).max())
        rng = float(u.max() - u.min()) or 1.0
        num = float((u * v).sum())
        den = float(np.linalg.norm(u) * np.linalg.norm(v)) or 1.0
        print(f"    {n:<6} max|d| {mad:9.4f}  rel {mad / rng * 100:6.2f}%  "
              f"cos {num / den:.5f}")
    print("    (cosine < ~0.99 on a cls output means the INT8 model will "
          "lose real mAP: try --method percentile or --exclude-first-conv)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser("NIRDet-Lite INT8 QDQ quantisation")
    ap.add_argument("--onnx", default=cfg.export.onnx_sim_path,
                    help="simplified fp32 ONNX from export_onnx.py")
    ap.add_argument("--out", default=cfg.export.onnx_int8_path)
    ap.add_argument("--data-root", default=cfg.data.root)
    ap.add_argument("--profile", default=None,
                    help="dataset profile YAML (same as used for training); "
                         "verifies label fingerprint before calibration")
    ap.add_argument("--split", default="train",
                    help="calibrate on the TRAIN split (val stays untouched)")
    ap.add_argument("--img-h", type=int, default=cfg.data.img_h)
    ap.add_argument("--img-w", type=int, default=cfg.data.img_w)
    ap.add_argument("--num-images", type=int, default=cfg.export.calib_images)
    ap.add_argument("--seed", type=int, default=cfg.train.seed)
    ap.add_argument("--method", choices=tuple(CALIB_METHODS),
                    default=cfg.export.calib_method)
    ap.add_argument("--percentile", type=float,
                    default=cfg.export.calib_percentile)
    ap.add_argument("--per-channel", dest="per_channel", action="store_true",
                    default=True, help="ST's scheme (default, keep it on)")
    ap.add_argument("--per-tensor", dest="per_channel", action="store_false")
    ap.add_argument("--no-clahe", action="store_true",
                    help="only if dataset.py ran with clahe disabled")
    ap.add_argument("--exclude-first-conv", action="store_true",
                    help="keep the NIR stem in float; usual first fix when "
                         "INT8 costs more than ~2 mAP")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="extra node names to leave in float")
    args = ap.parse_args()

    profile = None
    if args.profile is not None:
        from dataset_profiles import DatasetProfile
        profile = DatasetProfile.load(Path(args.profile))
        profile.verify_fresh()
        profile.apply(cfg)
        # Cross-check --data-root against the profile root.
        if args.data_root != cfg.data.root:
            import os
            a = os.path.normcase(os.fspath(Path(args.data_root).resolve()))
            b = os.path.normcase(os.fspath(Path(cfg.data.root).resolve()))
            if a != b:
                raise RuntimeError(
                    f"--data-root '{args.data_root}' conflicts with "
                    f"profile root '{cfg.data.root}'"
                )
        print(f"  profile    : {args.profile} "
              f"(fingerprint OK, n_train={cfg.data.n_train})")

    src = Path(args.onnx)
    if not src.exists():
        raise SystemExit(f"{src} not found; run export_onnx.py first")

    use_clahe = (not args.no_clahe) and (cfg.aug.clahe_p > 0.0)

    print("=" * 62)
    print("  NIRDet-Lite INT8 QDQ quantisation")
    print("=" * 62)
    print(f"  source     : {src} ({src.stat().st_size / 1e6:.2f} MB)")

    in_name, in_shape, out_names = graph_io(str(src))
    print(f"  input      : {in_name} {in_shape}")
    print(f"  outputs    : {out_names}")
    if any(isinstance(d, str) for d in in_shape):
        raise SystemExit("the graph has a dynamic input dimension; re-export "
                         "with dynamic_axes=None and run onnxsim with "
                         "--overwrite-input-shape")
    if len(in_shape) == 4 and int(in_shape[2]) != args.img_h:
        print(f"  ! graph H={in_shape[2]} but --img-h={args.img_h}; "
              f"using the GRAPH shape for calibration")
        args.img_h, args.img_w = int(in_shape[2]), int(in_shape[3])

    print(f"  preprocess : CLAHE={use_clahe} -> letterbox "
          f"{args.img_h}x{args.img_w} (pad 0) -> /255, no mean subtraction")
    print(f"               (imported verbatim from dataset.py — the calibration"
          f" distribution cannot drift from training)")

    files = collect_images(args.data_root, args.split, args.num_images,
                           args.seed)
    print(f"  calib set  : {len(files)} images from "
          f"{Path(args.data_root) / 'images' / args.split}")

    probe = preprocess(files[0], args.img_h, args.img_w, use_clahe,
                       cfg.aug.clahe_clip, cfg.aug.clahe_grid)
    print(f"  probe      : shape {probe.shape} dtype {probe.dtype} "
          f"range [{probe.min():.4f}, {probe.max():.4f}]")

    pre = src.with_name(src.stem + "-pre.onnx")
    print(f"\n  shape inference / pre-process -> {pre}")
    shape_inference.quant_pre_process(str(src), str(pre),
                                      skip_symbolic_shape=False)

    exclude = list(args.exclude)
    if args.exclude_first_conv:
        fc = first_conv_name(str(pre))
        if fc:
            exclude.append(fc)
            print(f"  excluding first Conv from quantisation: {fc}")

    extra: Dict[str, object] = {
        "ActivationSymmetric": False,     # ss/sa: asymmetric activations
        "WeightSymmetric": True,          # symmetric per-channel weights
        "AddQDQPairToWeight": True,       # ST expects QDQ around weights
        "DedicatedQDQPair": False,
    }
    if args.method == "percentile":
        extra["CalibPercentile"] = float(args.percentile)
        extra["CalibMovingAverage"] = False

    reader = NIRCalibrationReader(files, in_name, args.img_h, args.img_w,
                                  use_clahe, cfg.aug.clahe_clip,
                                  cfg.aug.clahe_grid)

    print(f"\n  quantising  format=QDQ  act=int8  weight=int8  "
          f"per_channel={args.per_channel}  method={args.method}")
    if args.method == "percentile":
        print(f"              percentile={args.percentile}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(pre),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=bool(args.per_channel),
        reduce_range=False,
        calibrate_method=CALIB_METHODS[args.method],
        nodes_to_exclude=exclude or None,
        extra_options=extra,
    )
    print(f"  consumed {reader.consumed} calibration images")
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")

    print("\n  INT8 graph op histogram")
    for op, n in op_histogram(str(out)).items():
        print(f"    {op:<26} {n}")

    compare_outputs(str(pre), str(out), files[0], in_name,
                    args.img_h, args.img_w, use_clahe,
                    cfg.aug.clahe_clip, cfg.aug.clahe_grid)

    print("\n" + "=" * 62)
    print("  next steps")
    print("=" * 62)
    print("  1. Validate the INT8 model in onnxruntime over the FULL val set")
    print("     and recompute mAP50 with evaluate.py's torchmetrics call.")
    print("     Accept only if the drop is under ~2 points. If it is worse:")
    print("       --method percentile --percentile 99.999")
    print("       --exclude-first-conv        (single-channel NIR stem is the")
    print("                                    usual outlier)")
    print("       verify CLAHE parity between calibration and training")
    print("  2. Compile for Neural-ART:")
    print(f"     stedgeai analyze  --model {out} --target stm32n6 \\")
    print("         --st-neural-art default@user_neuralart.json \\")
    print("         --input-data-type uint8 --output ./st_ai_output")
    print(f"     stedgeai generate --model {out} --target stm32n6 \\")
    print("         --st-neural-art default@user_neuralart.json \\")
    print("         --input-data-type uint8 --output ./st_ai_output")
    print("  3. The analyze report is the go/no-go gate. Require:")
    print("       ~zero SW_FLOAT layers, internal RAM < ~1.5 MB,")
    print("       weights < ~3 MB. Any conv shown as SW means an unsupported")
    print("       neighbour is forcing a dequantise boundary.")
    print("  4. --input-data-type uint8 folds the quantise into layer 1, so")
    print("     DCMIPP can hand the NPU its 8-bit mono buffer with no CPU")
    print("     touch. Dequantise the outputs in nirdet_pp.c using the exact")
    print("     scale / zero-point from the generated header.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
