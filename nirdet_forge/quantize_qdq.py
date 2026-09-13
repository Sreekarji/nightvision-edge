"""
quantize_qdq.py — INT8 QDQ quantization with the training preprocessing
========================================================================
    python quantize_qdq.py --onnx nirdet-sim.onnx \
                           --profile datasets/miniNIRPed_261.yaml

WHY THE REGRESSION CONV WAS SPLIT
---------------------------------
The head used to emit reg (4 channels) = (t_cx, t_cy, t_w, t_h) as one
tensor, so in INT8 QDQ all four channels shared one quantisation scale. But
t_cx/t_cy are practically +/-6 after training while t_w/t_h are clamped to
[REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX] = [-6, 1]. The union forces the scale
to cover [-6, 6]: 12.0/255 = 0.047 per LSB in log space, versus 7.0/255 =
0.027 (2.7%) for the size branch alone. On a 29 px pedestrian at stride 8
that is +/-1.4 px versus +/-0.8 px of width per quantisation step.

With off_pred and size_pred as separate convolutions the quantiser assigns them
independent scale / zero-point pairs. This script PRINTS both and fails loudly
if they came out identical, because identical scales mean the split did not
survive to the QDQ graph and the whole exercise bought nothing.

PREPROCESSING PARITY
Calibration uses dataset.preprocess_frame — the same flat-field, the same
CLAHE switch, the same letterbox, the same /255 as training. Calibrating on a
different distribution is the single most common way to lose several mAP
points in INT8 and blame the quantiser.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config, get_config, validate_config
from dataset import (list_images, load_flat_field, preprocess_frame,
                     resolve_split_dirs)

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader,
                                          CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static,
                                          shape_inference)
except ImportError as exc:                                   # pragma: no cover
    raise SystemExit(
        "quantize_qdq.py needs onnx + onnxruntime: "
        "pip install onnx onnxruntime") from exc

import cv2


# ===========================================================================
# calibration reader
# ===========================================================================

class NIRCalibrationReader(CalibrationDataReader):
    """
    Feeds calibration tensors through the EXACT training preprocessing chain.

    Images are drawn from the training split and shuffled with a fixed seed
    so calibration is reproducible. The same flat-field, CLAHE switch,
    letterbox and /255 that training saw — using a different pipeline
    calibrates the wrong distribution.
    """

    def __init__(self, cfg: Config, n_images: int, seed: int = 42,
                 flat_field: Optional[np.ndarray] = None) -> None:
        self.cfg = cfg
        img_dir, _ = resolve_split_dirs(cfg.data.root, "train")
        paths = sorted(list_images(img_dir))
        rng = random.Random(seed)
        rng.shuffle(paths)
        self.paths = paths[:n_images]
        self.flat_field = flat_field
        self._idx = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._idx >= len(self.paths):
            return None
        p = self.paths[self._idx]
        self._idx += 1
        raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            return self.get_next()
        canvas = preprocess_frame(
            raw, self.cfg.data.img_h, self.cfg.data.img_w,
            flat_field=self.flat_field,
            apply_clahe=self.cfg.data.use_clahe,
        )
        t = canvas[np.newaxis, np.newaxis, :, :].astype(np.float32)
        return {"input": t}


# ===========================================================================
# scale sanity check
# ===========================================================================

def output_scales(model_path: str) -> Dict[str, float]:
    """
    Return {output_name: scale} for every DequantizeLinear in the graph.

    In a QDQ graph the model output is produced by a DequantizeLinear whose
    'scale' initialiser is the per-tensor quantisation scale. Extracting it
    here lets us verify that off_pred and size_pred got independent scales.
    """
    m = onnx.load(model_path)
    init_map = {i.name: i for i in m.graph.initializer}
    scales: Dict[str, float] = {}
    for node in m.graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        if len(node.input) < 2:
            continue
        scale_init = init_map.get(node.input[1])
        if scale_init is None:
            continue
        val = float(np.frombuffer(scale_init.raw_data, dtype=np.float32)[0])
        # Use the output name as key so callers can match to blob names.
        if node.output:
            scales[node.output[0]] = val
    return scales


# ===========================================================================
# main
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="INT8 QDQ quantization of the NIRDet-Lite ONNX graph.")
    ap.add_argument("--onnx", default=None,
                    help="fp32 ONNX (default: cfg.export.onnx_sim_path if it "
                         "exists, else cfg.export.onnx_path)")
    ap.add_argument("--out", default=None,
                    help="output path (default: cfg.export.onnx_int8_path)")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--calib-images", type=int, default=None)
    ap.add_argument("--method", default=None,
                    choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-first-conv", action="store_true",
                    help="keep the single-channel NIR stem in float. This is "
                         "the usual first fix when evaluate_onnx.py reports a "
                         "drop above cfg.eval.int8_max_map50_drop: the stem "
                         "sees a different input distribution from every "
                         "other conv and is the common calibration outlier.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="extra node names to leave in float32")
    args = ap.parse_args()

    cfg = get_config()
    if args.profile:
        from dataset_profiles import DatasetProfile
        DatasetProfile.load(args.profile).apply(cfg)
    validate_config(cfg)

    fp32 = args.onnx or (
        cfg.export.onnx_sim_path
        if os.path.isfile(cfg.export.onnx_sim_path)
        else cfg.export.onnx_path)
    if not os.path.isfile(fp32):
        raise SystemExit(f"FP32 ONNX not found: {fp32}\n"
                         "  Run: python export_onnx.py first")

    out_path = args.out or cfg.export.onnx_int8_path
    n_cal = args.calib_images or cfg.export.calib_images

    cal_method_map = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
        None: CalibrationMethod.MinMax,
    }
    cal = cal_method_map[args.method]

    flat_field = load_flat_field(cfg)
    reader = NIRCalibrationReader(cfg, n_cal, seed=args.seed,
                                  flat_field=flat_field)

    extra: Dict = {}
    if cal == CalibrationMethod.Percentile:
        extra["calibration_sampling_size"] = min(n_cal, 256)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)

    # Symbolic shape inference + graph cleanup. quantize_static assumes every
    # tensor has a known shape; skipping this silently leaves some activations
    # unquantised, which shows up as SW_FLOAT layers in the stedgeai report.
    pre = os.path.splitext(fp32)[0] + "-pre.onnx"
    shape_inference.quant_pre_process(fp32, pre, skip_symbolic_shape=False)
    print(f"[quant] pre-processed -> {pre}")

    exclude = list(args.exclude)
    if args.exclude_first_conv:
        _m = onnx.load(pre)
        for _n in _m.graph.node:
            if _n.op_type == "Conv":
                exclude.append(_n.name)
                print(f"[quant] excluding first Conv from quantisation: "
                      f"{_n.name}")
                break

    print(f"[quant] {fp32} -> {out_path}")
    print(f"[quant] {n_cal} calibration images, method={args.method or 'minmax'}")

    quantize_static(
        model_input=pre,
        model_output=out_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=cal,
        nodes_to_exclude=exclude or None,
        extra_options=extra,
    )

    # Verify off_pred and size_pred got independent QDQ scales.
    scales = output_scales(out_path)
    off8_scales = [v for k, v in scales.items() if "off" in k.lower()]
    size8_scales = [v for k, v in scales.items() if "size" in k.lower()]
    print("\n[quant] output QDQ scales:")
    for name, val in sorted(scales.items()):
        print(f"  {name}: {val:.6f}")
    if off8_scales and size8_scales:
        if abs(off8_scales[0] - size8_scales[0]) < 1e-9:
            raise RuntimeError(
                "off8 and size8 have IDENTICAL QDQ scales — the split "
                "did not survive to the QDQ graph. Check that the ONNX "
                "was exported with separate off_pred / size_pred convs "
                "(test_decode_contract.t7 must PASS before quantising).")
        print(f"\n[quant] off scale  {off8_scales[0]:.6f}  "
              f"size scale {size8_scales[0]:.6f}  (must differ)")

    print("=" * 62)
    print(f"  INT8 QDQ written to: {out_path}")
    print(f"  A sanity check, not a gate: evaluate_onnx.py is the gate.")
    print(f"  next: python evaluate_onnx.py --int8 {out_path}"
          + (f" --profile {args.profile}" if args.profile else ""))
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
