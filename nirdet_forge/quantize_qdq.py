"""
quantize_qdq.py — INT8 QDQ quantization with the training preprocessing
========================================================================
    python quantize_qdq.py --onnx nirdet-sim.onnx --profile datasets/<n>.yaml

The head emits off and size as separate convolutions so the quantiser gives
them independent scales: t_cx/t_cy are practically +/-6 while t_w/t_h are
clamped to [-6, 1], and a shared scale spanning [-6, 6] costs 4.7% of box
width per LSB instead of 2.7%. This script PRINTS both per stride. A MISSING
scale is fatal; IDENTICAL scales are only a WARNING and do NOT change the
exit code (F82: ORT can legitimately assign equal scales to very narrow
off/size distributions). evaluate_onnx.py --int8 is the gate, not this
script.

Calibration uses dataset.preprocess_frame — the same flat-field, CLAHE switch,
letterbox and /255 as training.
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
# graph_input_name lives in export_onnx (needs only `onnx`), so importing it
# does not drag the quantisation toolchain into fp32-only consumers. Re-exported
# here for backward compatibility.
from export_onnx import graph_input_name  # noqa: F401

try:
    import onnx
    from onnx import numpy_helper
    from onnxruntime.quantization import (CalibrationDataReader,
                                          CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static,
                                          shape_inference)
except ImportError as exc:                                   # pragma: no cover
    raise SystemExit(
        "quantize_qdq.py needs onnx + onnxruntime: "
        "pip install onnx onnxruntime") from exc

import cv2


def first_conv_names_on_input(model_path: str, input_name: str) -> List[str]:
    """
    Every Conv that directly consumes the graph input.

    After quant_pre_process the input feeds BOTH the NIR stem and
    eaa.edge_conv; both sit on the raw sensor distribution and both are
    calibration outliers, so excluding only one leaves the other quantised
    against a distribution it was never calibrated for.
    """
    m = onnx.load(model_path)
    names = [n.name for n in m.graph.node
             if n.op_type == "Conv" and n.input and n.input[0] == input_name]
    if not names:
        raise RuntimeError(
            f"no Conv consumes graph input '{input_name}' in {model_path} — "
            f"--exclude-first-conv found nothing to exclude")
    return names


# ===========================================================================
# calibration reader
# ===========================================================================

class NIRCalibrationReader(CalibrationDataReader):
    """Feeds calibration tensors through the EXACT training preprocessing."""

    def __init__(self, cfg: Config, n_images: int, input_name: str,
                 seed: int = 42,
                 flat_field: Optional[np.ndarray] = None) -> None:
        self.cfg = cfg
        self.input_name = input_name
        # F49: shared seeded selection with export_ncnn so ORT and NCNN
        # calibrate on identical images. `seed` is kept in the signature for
        # backward compatibility but the authority is cfg.export.calib_seed.
        from config import calibration_paths
        _cs = int(seed if seed is not None else cfg.export.calib_seed)
        self.paths = calibration_paths(cfg.data.root, "train",
                                       n=n_images, seed=_cs)
        self.flat_field = flat_field
        self._idx = 0
        if not self.paths:
            # quantize_static does NOT treat zero calibration data as an
            # error: it emits a graph with no measured activation ranges and
            # exits 0 — the worst possible INT8 failure mode (F89).
            raise SystemExit(
                f"no calibration images for the train split. Calibrating on zero "
                f"images produces a QDQ graph with no measured activation "
                f"ranges, which quantize_static does NOT treat as an error.")
        print(f"[calib] {len(self.paths)} images (seed={_cs})")

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        # Iterative, not recursive (F88): a directory of unreadable images
        # used to recurse once per file and would hit RecursionError at a
        # few thousand --calib-images.
        while self._idx < len(self.paths):
            p = self.paths[self._idx]
            self._idx += 1
            raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                print(f"  [calib] unreadable, skipped: {p}")
                continue
            canvas, _s, _px, _py = preprocess_frame(
                raw, self.cfg.data.img_h, self.cfg.data.img_w,
                clahe_enabled=self.cfg.aug.clahe_enabled,
                clahe_clip=self.cfg.aug.clahe_clip,
                clahe_grid=self.cfg.aug.clahe_grid,
                flat_field=self.flat_field)
            return {self.input_name:
                    canvas[np.newaxis, np.newaxis, :, :].astype(np.float32)}
        return None


# ===========================================================================
# scale extraction
# ===========================================================================

def output_scales(model_path: str) -> Dict[str, float]:
    """{DequantizeLinear output name: per-tensor scale}."""
    m = onnx.load(model_path)
    init_map = {i.name: i for i in m.graph.initializer}
    scales: Dict[str, float] = {}
    for node in m.graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 2:
            continue
        scale_init = init_map.get(node.input[1])
        if scale_init is None:
            continue
        val = float(numpy_helper.to_array(scale_init).reshape(-1)[0])
        if node.output:
            scales[node.output[0]] = val
    return scales


def _scale_for(scales: Dict[str, float], blob: str
               ) -> Optional[Tuple[str, float]]:
    """
    Suffix-tolerant lookup (F90).

    An exact-key match assumes ORT preserves graph output names through
    quantisation; some paths append a suffix, and the strict lookup then
    raised "the split did not survive" for a graph whose split is fine.
    """
    if blob in scales:
        return blob, scales[blob]
    cands = [k for k in scales
             if k == blob or k.startswith(blob + "_") or k.endswith("/" + blob)]
    if len(cands) == 1:
        return cands[0], scales[cands[0]]
    return None


def verify_exclusions(out_path: str, exclude: List[str]) -> List[str]:
    """
    Confirm ORT honoured nodes_to_exclude (F86).

    ORT silently ignores names that do not match the POST-preprocessing node
    names, so the primary documented remediation for a failed accuracy gate
    could be a no-op with no diagnostic.
    """
    if not exclude:
        return
    qm = onnx.load(out_path)
    dq_out = {o for n in qm.graph.node
              if n.op_type == "DequantizeLinear" for o in n.output}
    still = []
    for n in qm.graph.node:
        if n.name in exclude and n.input and n.input[0] in dq_out:
            still.append(n.name)
    return still


# ===========================================================================
# main
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="INT8 QDQ quantization of the NIRDet-Lite ONNX graph.")
    ap.add_argument("--onnx", default=None,
                    help="fp32 ONNX (default: cfg.export.onnx_sim_path)")
    ap.add_argument("--out", default=None,
                    help="output path (default: cfg.export.onnx_int8_path)")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--calib-images", type=int, default=None)
    ap.add_argument("--method", default=None,
                    choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg.export.calib_seed")
    ap.add_argument("--exclude-first-conv", action="store_true",
                    help="keep the convs that directly consume the graph "
                         "input (the NIR stem and eaa.edge_conv) in float32")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="extra node names to leave in float32")
    ap.add_argument("--keep-pre", action="store_true",
                    help="keep the intermediate -pre.onnx graph")
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
    # F09 (audit fix): the CLI overrides the config; the config is the
    # fallback. Reading only args.method made cfg.export.calib_method a
    # no-op field, and it silently defeated remediation step 2 that
    # evaluate_onnx.py prints when the INT8 gate fails.
    method = str(args.method or cfg.export.calib_method or "minmax").lower()
    _CAL_METHODS = {"minmax": CalibrationMethod.MinMax,
                    "percentile": CalibrationMethod.Percentile,
                    "entropy": CalibrationMethod.Entropy}
    if method not in _CAL_METHODS:
        raise SystemExit(
            f"unknown calibration method {method!r}; expected one of "
            f"{sorted(_CAL_METHODS)} (from --method or cfg.export.calib_method)")
    cal = _CAL_METHODS[method]

    flat_field = load_flat_field(cfg.aug.flat_field_path)

    extra: Dict = {}
    if cal == CalibrationMethod.Percentile:
        # ORT's own key is "CalibPercentile"; some versions VALIDATE
        # extra_options strictly, so passing both keys unconditionally could
        # make --method percentile fail outright. Try the modern key, fall
        # back on rejection (F87).
        extra["CalibPercentile"] = float(cfg.export.calib_percentile)
        print(f"[quant] percentile calibration: CalibPercentile="
              f"{extra['CalibPercentile']} "
              f"(cfg.export.calib_percentile)")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)

    pre = os.path.splitext(fp32)[0] + "-pre.onnx"
    shape_inference.quant_pre_process(fp32, pre, skip_symbolic_shape=False)
    print(f"[quant] pre-processed -> {pre}")

    in_name = graph_input_name(pre)
    print(f"[quant] graph input: {in_name!r}")

    reader = NIRCalibrationReader(cfg, n_cal, input_name=in_name,
                                  seed=args.seed, flat_field=flat_field)

    exclude = list(args.exclude)
    if args.exclude_first_conv:
        for conv_name in first_conv_names_on_input(pre, in_name):
            exclude.append(conv_name)
            print(f"[quant] excluding input Conv from quantisation: "
                  f"{conv_name}")

    print(f"[quant] {fp32} -> {out_path}")
    print(f"[quant] {n_cal} calibration images, method={method} "
          f"(source: {'--method' if args.method else 'cfg.export.calib_method'})")

    kwargs = dict(
        model_input=pre, model_output=out_path,
        calibration_data_reader=reader, quant_format=QuantFormat.QDQ,
        per_channel=True, weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8, calibrate_method=cal,
        nodes_to_exclude=exclude or None, extra_options=extra,
    )
    try:
        quantize_static(**kwargs)
    except (TypeError, ValueError) as exc:
        if "CalibPercentile" not in str(exc):
            raise
        print(f"[quant] this onnxruntime rejects CalibPercentile ({exc}); "
              f"retrying with the legacy 'percentile' key")
        reader._idx = 0
        kwargs["extra_options"] = {
            "percentile": float(cfg.export.calib_percentile)}
        quantize_static(**kwargs)

    still = verify_exclusions(out_path, exclude)
    if still:
        raise RuntimeError(
            f"nodes {sorted(set(still))} were requested excluded but appear "
            f"quantised in {out_path} (their data input is fed by a "
            f"DequantizeLinear). ORT matches node NAMES from the "
            f"pre-processed graph; re-read them with "
            f"first_conv_names_on_input(pre, in_name) and pass those exactly.")
    if exclude:
        print(f"[quant] verified {len(exclude)} node(s) left in float32")

    scales = output_scales(out_path)
    # F83: these are ACTIVATION scales (per-tensor), not weight
    # DQ scales. Per-channel weight scales are separate; do not read
    # this table as the weight quantisation.
    print("\n[quant] output QDQ activation scales (per-tensor):")
    for name, val in sorted(scales.items()):
        print(f"  {name}: {val:.6f}")

    checked = 0
    for s in cfg.model.strides:
        off_key, size_key = f"off{s}", f"size{s}"
        off = _scale_for(scales, off_key)
        size = _scale_for(scales, size_key)
        missing = [k for k, v in ((off_key, off), (size_key, size)) if v is None]
        if missing:
            raise RuntimeError(
                f"QDQ output scale missing for {missing}. The off/size split "
                f"did not survive to the QDQ graph. Check that the ONNX was "
                f"exported with separate off_pred / size_pred convs "
                f"(test_decode_contract t7 must PASS before quantising). "
                f"Present keys: {sorted(scales)}")
        (off_name, off_val), (size_name, size_val) = off, size
        if abs(off_val - size_val) < 1e-9:
            # F82: warn, do not abort. ORT may legitimately assign equal
            # scales for very narrow off/size distributions; evaluate_onnx.py
            # gates the real accuracy anyway.
            print(f"[quant] WARNING: {off_name} and {size_name} have identical "
                  f"QDQ scales ({off_val:.6f}). This usually means the off/size "
                  f"split did not survive to the QDQ graph. Verify with "
                  f"onnxruntime before proceeding.")
        checked += 1
        print(f"[quant] stride {s}: {off_name} scale {off_val:.6f}  "
              f"{size_name} scale {size_val:.6f}  (expected to differ)")
    if checked == 0:
        raise RuntimeError(
            f"off/size scale check ran on no strides — cfg.model.strides is "
            f"empty: {cfg.model.strides!r}")

    if args.keep_pre:
        print(f"[quant] kept intermediate graph: {pre}")
    else:
        os.remove(pre)
        print(f"[quant] removed intermediate graph: {pre}")

    # The INT8 graph must carry its own contract, like every other artefact.
    from export_onnx import write_contract_sidecar
    write_contract_sidecar(cfg, out_path)

    print("=" * 62)
    print(f"  INT8 QDQ written to: {out_path}")
    print("  A sanity check, not a gate: evaluate_onnx.py is the gate.")
    print(f"  next: python evaluate_onnx.py --int8 {out_path} "
          f"--fp32-ref {fp32}"
          + (f" --profile {args.profile}" if args.profile else ""))
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
