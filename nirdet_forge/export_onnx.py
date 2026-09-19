"""
export_onnx.py — static ONNX export + Neural-ART / NCNN graph hygiene checks
=============================================================================
    python export_onnx.py --checkpoint checkpoints/best.pth

Exports model.forward_raw only: NINE NCHW convolution outputs, three per
level. No grid, sigmoid, exp, clamp, reshape, concat or NMS — ST documents
detection post-processing including NMS as a HOST responsibility, and NCNN
INT8 is happiest with a graph that ends at the last convolution.

Most banned ops ARE hardware-mapped. They are banned because their PRESENCE
IS A SYMPTOM: a Gather means dynamic indexing leaked in, a Shape means a
dynamic shape leaked in, a Softmax means decode logic leaked in.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import (ST_FEATURE_WIDTH_CHANNEL_LIMIT, Config, get_config,
                    validate_config)
from model import NIRDet, build_nirdet

try:
    import onnx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("export_onnx.py needs onnx: pip install onnx") from exc


# ===========================================================================
# graph input name — lives HERE, not in quantize_qdq (F35)
# ===========================================================================

def graph_input_name(model_path: str) -> str:
    """
    The name of the single non-initializer graph input. Needs only `onnx`.

    It used to live in quantize_qdq.py, so evaluate_onnx.py's import of it
    executed that module's body — dragging onnxruntime.quantization into a
    plain fp32 evaluation and raising SystemExit (not ImportError) when it
    was absent, which no guarded import could catch.
    """
    m = onnx.load(model_path)
    inits = {i.name for i in m.graph.initializer}
    names = [i.name for i in m.graph.input if i.name not in inits]
    if len(names) != 1:
        raise RuntimeError(
            f"expected exactly 1 non-initializer graph input in "
            f"{model_path}, got: {names!r}")
    return names[0]


# ===========================================================================
# export wrapper
# ===========================================================================

class RawExportWrapper(nn.Module):
    """Flattens forward_raw's list-of-triples into a flat tuple, in
    head.output_names() order, so torch.onnx.export sees one tensor per name."""

    def __init__(self, model: NIRDet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        flat: List[torch.Tensor] = []
        for cls_l, off_l, size_l in self.model.forward_raw(x):
            flat += [cls_l, off_l, size_l]
        return tuple(flat)


# ===========================================================================
# graph checks
# ===========================================================================

ALLOWED_OPS = {
    "Conv", "BatchNormalization", "Relu", "Clip", "Add", "Mul", "Sub",
    "Concat", "Resize", "Abs", "AveragePool", "GlobalAveragePool", "Sigmoid",
    "Constant", "Identity", "Pad", "Div",
    "Exp", "Reshape", "Slice", "Transpose", "Sqrt", "Erf", "Pow",
    "ReduceMean", "Cast", "Squeeze", "Unsqueeze",
    # Present ONLY in the QDQ graph. run_checks is normally pointed at the
    # fp32 graph, but nothing prevents pointing it at the INT8 one, and
    # --strict-ops would then fail on ops that belong there (F50).
    "QuantizeLinear", "DequantizeLinear",
}

FORBIDDEN_OPS: Dict[str, str] = {
    "Gather": (
        "Gather is SW_INT on Neural-ART, but it is banned here for graph "
        "hygiene: in a pure convolution graph the only things that emit "
        "Gather are dynamic indexing and Shape->Gather->Concat size "
        "computation."),
    "ScatterND": (
        "Decode logic leaked into the graph. Nothing in forward_raw writes "
        "into a tensor by index."),
    "NonMaxSuppression": (
        "NMS must run on the host; an in-graph NMS makes the output shape "
        "dynamic, which breaks the fixed-size INT8 buffers on both targets."),
    "Shape": (
        "A dynamic shape leaked in — typically "
        "F.interpolate(size=x.shape[-2:]), which also drags in Gather."),
    "Loop": "Control flow cannot be statically traced or scheduled on the NPU.",
    "If": "Control flow cannot be statically traced or scheduled on the NPU.",
    "Softmax": (
        "Decode leaked in — with NUM_CLASSES=1 there is no axis to normalise. "
        "ST also maps Softmax on hardware only with --expand-softmax."),
    "InstanceNormalization": (
        "SW_FLOAT on Neural-ART. Every normalisation here is BatchNorm, which "
        "folds into the preceding convolution."),
    "Softplus": (
        "SW_FLOAT on Neural-ART. All activations are ReLU6 (Clip)."),
    "ReduceSum": (
        "SW_FLOAT on Neural-ART; unlike ReduceMean it is not convertible to "
        "GlobalAveragePool."),
}

ALLOWED_NOTE = (
    "Resize, Exp, Reshape, Slice, Transpose, Sqrt, Erf, Pow (constant "
    "exponent), ReduceMean (convertible to GlobalAveragePool), Clip, Concat, "
    "Add, Mul, Sub, Sigmoid, AveragePool and Conv are hardware-mapped and are "
    "NOT forbidden. Quantize/DequantizeLinear are expected in a QDQ graph."
)


def _initializer_names(graph) -> set:
    return {i.name for i in graph.initializer}


def _constant_output_names(graph) -> set:
    out = set()
    for n in graph.node:
        if n.op_type in ("Constant", "ConstantOfShape"):
            out.update(n.output)
    return out


def _attr(node, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.STRING:
                return a.s.decode("utf-8")
            if a.type == onnx.AttributeProto.INT:
                return int(a.i)
            if a.type == onnx.AttributeProto.FLOAT:
                return float(a.f)
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            return a
    return default


def _check_forbidden_ops(graph) -> List[str]:
    errs: List[str] = []
    consts = _initializer_names(graph) | _constant_output_names(graph)
    for node in graph.node:
        op = node.op_type
        if op in FORBIDDEN_OPS:
            errs.append(f"forbidden op {op} at node '{node.name or '<anon>'}': "
                        f"{FORBIDDEN_OPS[op]}")
        # F44: constant-operand rule applies to Pow (constant exponent) too.
        if op in ("Div", "Pow") and len(node.input) >= 2 \
                and node.input[1] not in consts:
            reason = ("ST maps Div on hardware only when the second operand is a "
                      "constant, so a runtime divisor becomes SW_INT and makes the "
                      "INT8 activation range depend on the scene. This is what "
                      "cfg.model.eaa_normalize_edges=True emits. Set it False and "
                      "stabilise brightness in preprocessing instead."
                      if op == "Div" else
                      "ST maps Pow on hardware only with a constant exponent")
            errs.append(
                f"forbidden {op} with non-constant second operand at "
                f"node '{node.name or '<anon>'}' ('{node.input[1]}'): {reason}")
    return errs


def _check_resize_nodes(graph) -> List[str]:
    """Every Resize must be nearest/asymmetric/floor — the only HW path."""
    errs: List[str] = []
    n_resize = 0
    for node in graph.node:
        if node.op_type != "Resize":
            continue
        n_resize += 1
        mode = _attr(node, "mode", "nearest")
        ctm = _attr(node, "coordinate_transformation_mode", "half_pixel")
        nm = _attr(node, "nearest_mode", "round_prefer_floor")
        tag = node.name or "<anon>"
        if mode != "nearest":
            errs.append(f"Resize '{tag}': mode='{mode}', expected 'nearest'")
        if ctm != "asymmetric":
            errs.append(f"Resize '{tag}': coordinate_transformation_mode="
                        f"'{ctm}', expected 'asymmetric'")
        if nm != "floor":
            errs.append(f"Resize '{tag}': nearest_mode='{nm}', expected 'floor'")
    expected_resize = 2
    if n_resize != expected_resize:
        errs.append(
            f"found {n_resize} Resize nodes, expected exactly {expected_resize} "
            f"(the two FPN top-down upsamples). A third usually means "
            f"EAA.apply_to took its upsample branch because "
            f"eaa_edge_stride * eaa_pool_factor no longer equals the finest "
            f"detection stride.")
    else:
        print(f"[check] Resize nodes: {n_resize} (expect 2: the two FPN "
              f"top-down upsamples) ✓")
    return errs


def _check_resize_scale_factor(graph) -> List[str]:
    """Every Resize must drive the ``scales`` input, never ``sizes``."""
    errs: List[str] = []
    consts = _initializer_names(graph) | _constant_output_names(graph)
    for node in graph.node:
        if node.op_type != "Resize":
            continue
        tag = node.name or "<anon>"
        ins = list(node.input)
        sizes = ins[3] if len(ins) >= 4 else ""
        scales = ins[2] if len(ins) >= 3 else ""
        if sizes:
            errs.append(
                f"Resize '{tag}' uses the 'sizes' input ('{sizes}'): that is "
                f"produced by Shape->Gather->Concat. Use scale_factor.")
        if not scales:
            errs.append(f"Resize '{tag}' has no 'scales' input")
        elif scales not in consts:
            errs.append(f"Resize '{tag}': 'scales' input '{scales}' is not a "
                        f"constant, so the scale is computed at runtime")
    return errs


def _report_unreviewed_ops(graph, strict: bool = True) -> List[str]:
    """F48: strict (fail the export) is the DEFAULT; --allow-unreviewed-ops
    downgrades to a warning."""
    seen = {n.op_type for n in graph.node}
    unknown = sorted(seen - ALLOWED_OPS - set(FORBIDDEN_OPS))
    errs: List[str] = []
    for op in unknown:
        msg = (f"unreviewed op '{op}': not in ALLOWED_OPS or FORBIDDEN_OPS. "
               f"Confirm it maps on Neural-ART and NCNN INT8, then add it to "
               f"one of the two lists.")
        if strict:
            errs.append(msg)
        else:
            print(f"[warn] {msg}")
    return errs


def _shape_map(model_proto) -> Dict[str, List[int]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model_proto)
    except Exception as _shape_exc:
        print(f"[export] WARNING: shape inference failed "
              f"({type(_shape_exc).__name__}: {_shape_exc}); "
              f"using un-inferred graph.")
        inferred = model_proto
    out: Dict[str, List[int]] = {}
    g = inferred.graph
    for coll in (g.input, g.output, g.value_info):
        for vi in coll:
            out[vi.name] = [int(d.dim_value) if d.dim_value > 0 else -1
                            for d in vi.type.tensor_type.shape.dim]
    for init in g.initializer:
        out[init.name] = list(init.dims)
    return out


def _is_constant_producer(name: str, graph) -> bool:
    """True if the tensor's producer node is a Constant (legit shape vectors)."""
    for node in graph.node:
        if name in node.output:
            return node.op_type == "Constant"
    return False


def _check_static_shapes(model_proto) -> List[str]:
    """
    No dynamic dimension may survive export, on the boundary OR INTERNALLY.

    Checking only graph.input/output missed a dynamic dim on an internal
    value_info, which is what a surviving Shape->Reshape produces after
    partial folding (F48).

    F47: the len(dims)==4 restriction is dropped — 1-D/2-D dynamic tensors
    (e.g. Shape->Concat size vectors) are caught too, unless their producer
    is a Constant node.
    """
    errs: List[str] = []
    g = model_proto.graph
    for vi in list(g.input) + list(g.output):
        dims = [d.dim_param if d.dim_param else int(d.dim_value)
                for d in vi.type.tensor_type.shape.dim]
        if any(isinstance(d, str) or d == 0 for d in dims):
            errs.append(
                f"'{vi.name}' has a dynamic dimension {dims}: the export must "
                f"be static (dynamic_axes=None), or the INT8 buffers on the "
                f"Pi and the N6 cannot be sized at compile time")
    init_names = _initializer_names(g)
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        name = vi.name
        if name in init_names:
            continue
        dims = [int(d.dim_value) if d.dim_value > 0 else -1
                for d in vi.type.tensor_type.shape.dim]
        if dims and any(int(d) <= 0 for d in dims) \
                and not _is_constant_producer(name, g):
            errs.append(
                f"internal tensor '{name}' has an unresolved shape {dims}; "
                f"the INT8 buffers on both targets cannot be sized")
    return errs


def _check_no_gather(graph) -> List[str]:
    hits = [n.name or "<anon>" for n in graph.node if n.op_type == "Gather"]
    if not hits:
        print("[check] Gather nodes: 0")
        return []
    return [f"found {len(hits)} Gather node(s): {hits[:8]}. With per-level "
            f"prediction convolutions replacing reg_level_scale, nothing in "
            f"this graph should emit Gather. Usual causes: a registered "
            f"selector buffer being indexed, or a Resize falling back to the "
            f"sizes input."]


def _check_feature_widths(model_proto) -> List[str]:
    """ST: feature WIDTH x input channels must be <= 2048 for 8-bit data.

    A WARNING, not an error: at 512x288 with a 64-channel head the stride-8
    head input is 64x64 = 4096, so the compiler will split it. Expected and
    cheaper than shrinking the canvas or the head.
    """
    shapes = _shape_map(model_proto)
    warns: List[str] = []
    for node in model_proto.graph.node:
        if node.op_type not in ("Conv", "AveragePool", "MaxPool",
                                "GlobalAveragePool") or not node.input:
            continue
        dims = shapes.get(node.input[0])
        if not dims or len(dims) != 4:
            continue
        c, w = dims[1], dims[3]
        if c <= 0 or w <= 0:
            continue
        prod = int(c) * int(w)
        if prod > ST_FEATURE_WIDTH_CHANNEL_LIMIT:
            warns.append(
                f"[warn] {node.name or node.op_type}: {w}x{c}={prod} > "
                f"{ST_FEATURE_WIDTH_CHANNEL_LIMIT}; Neural-ART compiler will "
                f"split into columns")
    return warns


def _macs(model_proto) -> int:
    shapes = _shape_map(model_proto)
    total = 0
    for node in model_proto.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        wdims = shapes.get(node.input[1])
        odims = shapes.get(node.output[0]) if node.output else None
        if not wdims or len(wdims) != 4 or not odims or len(odims) != 4:
            continue
        m, cg, kh, kw = wdims
        oh, ow = odims[2], odims[3]
        if min(oh, ow) <= 0:
            continue
        total += int(m) * int(cg) * int(kh) * int(kw) * int(oh) * int(ow)
    return total


def op_histogram(graph) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for n in graph.node:
        hist[n.op_type] = hist.get(n.op_type, 0) + 1
    return hist


# ===========================================================================
# export
# ===========================================================================

def write_contract_sidecar(cfg: Config, artefact_path: str) -> str:
    """
    The ONNX, .param and .bin artefacts carry no contract information, so
    live_nirdet.py and nirdet_pp.c cannot verify that the model they decode
    was trained under the constants they were compiled with.
    """
    import json
    contract = cfg.deploy_contract()
    contract["strides"] = [int(s) for s in cfg.model.strides]
    contract["blob_names"] = [f"{p}{s}" for s in cfg.model.strides
                              for p in ("cls", "off", "size")]
    contract["artefact"] = os.path.basename(artefact_path)
    path = os.path.splitext(artefact_path)[0] + ".contract.json"
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(contract, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    print(f"[export] contract sidecar -> {path} (hash {contract['hash']})")
    return path


def export(cfg: Config, checkpoint: Optional[str], device: torch.device,
           simplify: bool = True) -> Tuple[str, str, NIRDet]:
    model = build_nirdet(cfg).to(device).eval()

    if checkpoint:
        from evaluate import load_checkpoint_into
        load_checkpoint_into(model, checkpoint, cfg, device)
        model.eval()
    else:
        print("[warn] no --checkpoint: exporting randomly initialised "
              "weights. Useful for graph checks, useless for deployment.")

    names = model.head.output_names()
    expected = 3 * len(cfg.model.strides)
    if len(names) != expected:
        raise RuntimeError(f"head.output_names() returned {len(names)} names, "
                           f"expected {expected}")
    print(f"[export] output blobs ({len(names)}): {names}")

    wrapper = RawExportWrapper(model).eval()
    dummy = torch.zeros(1, 1, cfg.data.img_h, cfg.data.img_w, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.export.onnx_path)) or ".",
                exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, cfg.export.onnx_path,
        export_params=True, opset_version=int(cfg.export.opset),
        do_constant_folding=True,
        input_names=["images"], output_names=names,
        dynamic_axes=None,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"[export] {cfg.export.onnx_path} (opset {cfg.export.opset}, "
          f"static 1x1x{cfg.data.img_h}x{cfg.data.img_w})")
    write_contract_sidecar(cfg, cfg.export.onnx_path)

    sim_path = cfg.export.onnx_path
    if simplify:
        try:
            import onnxsim
            m = onnx.load(cfg.export.onnx_path)
            m_sim, ok = onnxsim.simplify(m)
            if not ok:
                raise RuntimeError("onnxsim reported failure")
            onnx.save(m_sim, cfg.export.onnx_sim_path)
            sim_path = cfg.export.onnx_sim_path
            print(f"[export] simplified -> {sim_path}")
        except ImportError:
            print("[export] onnxsim not installed; skipping simplification. "
                  "Copying raw ONNX to sim path so downstream commands work.")
            import shutil
            shutil.copyfile(cfg.export.onnx_path, cfg.export.onnx_sim_path)
            sim_path = cfg.export.onnx_sim_path
        except Exception as exc:
            # F43: copy the raw graph to the sim path so downstream commands
            # (which consume onnx_sim_path) keep working after a sim failure.
            print(f"[export] simplification failed ({type(exc).__name__}: {exc}); "
                  f"copying the raw graph to the sim path so downstream commands work")
            import shutil as _shutil
            _shutil.copyfile(cfg.export.onnx_path, cfg.export.onnx_sim_path)
            sim_path = cfg.export.onnx_sim_path

    # SIDECAR EVERY GRAPH THAT IS EMITTED (F47). Everything downstream
    # (quantize_qdq, evaluate_onnx, export_ncnn) consumes the SIM graph, and
    # only the raw one used to get a contract file.
    if os.path.abspath(sim_path) != os.path.abspath(cfg.export.onnx_path):
        write_contract_sidecar(cfg, sim_path)

    return cfg.export.onnx_path, sim_path, model


def run_checks(path: str, strict_ops: bool = True) -> int:
    """F48: strict unreviewed-op reporting is the default now."""
    m = onnx.load(path)
    onnx.checker.check_model(m)
    g = m.graph

    print("=" * 70)
    print(f"  graph checks: {path}")
    print("=" * 70)
    print(f"[note] {ALLOWED_NOTE}")
    if any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in g.node):
        print("[check] QDQ graph detected: the MAC and feature-width numbers "
              "below describe the quantised graph, not the fp32 one")

    errs: List[str] = []
    errs += _check_forbidden_ops(g)
    errs += _report_unreviewed_ops(g, strict=strict_ops)
    errs += _check_static_shapes(m)
    errs += _check_no_gather(g)
    errs += _check_resize_nodes(g)
    errs += _check_resize_scale_factor(g)

    for w in _check_feature_widths(m):
        print(w)

    hist = op_histogram(g)
    print(f"[graph] {len(g.node)} nodes, {len(hist)} op types:")
    for op in sorted(hist):
        print(f"    {op:<22} {hist[op]}")
    print(f"[graph] MACs (Conv only): {_macs(m) / 1e6:.2f} M")

    shapes = _shape_map(m)
    print("[graph] outputs:")
    for o in g.output:
        print(f"    {o.name:<10} {shapes.get(o.name)}")

    if errs:
        print("-" * 70)
        for e in errs:
            print(f"[ERROR] {e}")
        print("-" * 70)
        return len(errs)
    print("[check] all graph checks PASSED")
    return 0


def parity_check(cfg: Config, onnx_path: str, device: torch.device,
                 model: NIRDet, tol: float = 1e-5) -> bool:
    """
    RELATIVE tolerance (F49): an absolute 1e-5 on a logit of magnitude 8 is a
    relative 1.2e-6, tighter than fp32 accumulation-order differences between
    PyTorch and ORT for a 96-channel 3x3 conv, and would produce spurious
    FAILs on a trained checkpoint (the random-weight path passes only because
    its activations are small).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[parity] onnxruntime not installed; skipping parity check")
        return True

    model.eval()
    torch.manual_seed(0)
    # F45: probe with a REAL preprocessed frame when a dataset root is
    # available. Uniform noise is unrepresentative of the EAA sigmoid regime:
    # it drives the edge gate to a distribution the trained model never sees,
    # and a parity FAIL on noise is not diagnostic of anything.
    x = None
    if getattr(cfg.data, "root", None) and os.path.isdir(str(cfg.data.root)):
        try:
            import cv2 as _cv2
            from dataset import (list_images, load_flat_field,
                                 preprocess_frame, resolve_split_dirs)
            img_dir, _ = resolve_split_dirs(cfg.data.root, "train")
            _p = sorted(list_images(img_dir))[0]
            _raw = _cv2.imread(_p, _cv2.IMREAD_GRAYSCALE)
            if _raw is None:
                raise RuntimeError(f"unreadable probe image {_p}")
            _canvas, *_ = preprocess_frame(
                _raw, cfg.data.img_h, cfg.data.img_w,
                cfg.aug.clahe_enabled, cfg.aug.clahe_clip,
                cfg.aug.clahe_grid,
                load_flat_field(cfg.aug.flat_field_path))
            x = torch.from_numpy(_canvas)[None, None].to(device)
            print("[parity] probing on a real preprocessed frame")
        except Exception as exc:
            print(f"[parity] falling back to uniform noise ({exc})")
            x = torch.rand(1, 1, cfg.data.img_h, cfg.data.img_w, device=device)
    else:
        x = torch.rand(1, 1, cfg.data.img_h, cfg.data.img_w, device=device)

    with torch.no_grad():
        ref: List[torch.Tensor] = []
        for c, o, s in model.forward_raw(x):
            ref += [c, o, s]

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name:
                          x.detach().cpu().numpy().astype(np.float32)})

    names = model.head.output_names()
    print("[parity] |torch - onnx| per blob (abs / rel):")
    worst = 0.0
    ok = True
    for name, t, o in zip(names, ref, got):
        a = t.detach().cpu().numpy()
        b = np.asarray(o)
        d = float(np.abs(a - b).max())
        scale = max(1.0, float(np.abs(a).max()))
        rel = d / scale
        worst = max(worst, rel)
        good = rel < tol
        ok = ok and good
        print(f"    {name:<8} abs {d:.3e}  rel {rel:.3e}"
              f"{'' if good else '   <-- FAIL'}")
    print(f"[parity] worst relative {worst:.3e} (tolerance {tol:.0e}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Export NIRDet-Lite to ONNX.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--strict-ops", action="store_true",
                    help="deprecated no-op: unreviewed ops fail by default "
                         "(F48); use --allow-unreviewed-ops to downgrade")
    ap.add_argument("--allow-unreviewed-ops", action="store_true",
                    help="warn instead of failing on ops outside "
                         "ALLOWED_OPS / FORBIDDEN_OPS (F48 default is strict)")
    ap.add_argument("--device", default="cpu",
                    help="cpu is correct for export")
    args = ap.parse_args()

    cfg = get_config()
    if args.profile:
        from dataset_profiles import DatasetProfile
        DatasetProfile.load(args.profile).apply(cfg)
    else:
        # Recover priors from the checkpoint so export works without a profile.
        # The checkpoint carries a full cfg dict from the training run.
        try:
            ck = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)
            ck_cfg = ck.get("cfg") or {}
            model_cfg = ck_cfg.get("model") or {}
            if cfg.model.prior_w is None and "prior_w" in model_cfg:
                cfg.model.prior_w = float(model_cfg["prior_w"])
                cfg.model.prior_h = float(model_cfg["prior_h"])
                print(f"[export] priors recovered from checkpoint: "
                      f"prior_w={cfg.model.prior_w} prior_h={cfg.model.prior_h}")
        except Exception as exc:
            print(f"[export] WARNING could not recover priors from checkpoint "
                  f"({exc}). Pass --profile <yaml> if validate_config fails.")
    validate_config(cfg, verbose=False)

    device = torch.device(args.device)
    raw_path, sim_path, model = export(cfg, args.checkpoint, device,
                                       simplify=not args.no_simplify)
    n_err = run_checks(sim_path,
                       strict_ops=not bool(args.allow_unreviewed_ops))
    parity_ok = True
    if not args.no_parity:
        parity_ok = parity_check(cfg, sim_path, device, model)

    print("=" * 70)
    if n_err or not parity_ok:
        print(f"  EXPORT FAILED: {n_err} graph error(s), "
              f"parity {'ok' if parity_ok else 'FAILED'}")
        return 1
    print(f"  export OK: {sim_path}")
    print(f"  next: python quantize_qdq.py --onnx {sim_path}"
          + (f" --profile {args.profile}" if args.profile else ""))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
