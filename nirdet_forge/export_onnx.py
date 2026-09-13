"""
export_onnx.py — static ONNX export + Neural-ART / NCNN graph hygiene checks
=============================================================================
    python export_onnx.py --checkpoint checkpoints/best.pth

WHAT IS EXPORTED
----------------
model.forward_raw only: NINE NCHW convolution outputs, three per level.

    cls8  (1, 1, 36, 64)   off8  (1, 2, 36, 64)   size8  (1, 2, 36, 64)
    cls16 (1, 1, 18, 32)   off16 (1, 2, 18, 32)   size16 (1, 2, 18, 32)
    cls32 (1, 1,  9, 16)   off32 (1, 2,  9, 16)   size32 (1, 2,  9, 16)

No grid, no sigmoid, no exp, no clamp, no reshape, no concat, no NMS. ST
documents detection post-processing including NMS as a HOST responsibility
with no NPU support, and NCNN INT8 is likewise happiest with a graph that
ends at the last convolution. Decode lives in live_nirdet.py and nirdet_pp.c.

WHY THE FORBIDDEN-OP LIST IS NOT A HARDWARE-SUPPORT LIST
--------------------------------------------------------
Most of the banned ops ARE hardware-mapped on Neural-ART. They are banned
because their PRESENCE IS A SYMPTOM: in a graph that should be nine
convolution stacks and two Resizes, a Gather means dynamic indexing leaked
in, a Shape means a dynamic shape leaked in, and a Softmax or a ScatterND
means decode logic leaked in. Each entry below states the real reason.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import ST_FEATURE_WIDTH_CHANNEL_LIMIT, Config, get_config, validate_config
from model import NIRDet, build_nirdet

try:
    import onnx
    from onnx import numpy_helper
except ImportError as exc:  # pragma: no cover
    raise SystemExit("export_onnx.py needs onnx: pip install onnx") from exc


# ===========================================================================
# export wrapper
# ===========================================================================

class RawExportWrapper(nn.Module):
    """
    Flattens forward_raw's list-of-triples into a flat tuple, in
    head.output_names() order, so torch.onnx.export sees one tensor per name.

    A wrapper module (rather than exporting the bound method) keeps the export
    a normal nn.Module trace, which is what the ONNX exporter and every
    downstream simplifier expect.
    """

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

# Every op type this graph is KNOWN to contain. Anything outside both this set
# and FORBIDDEN_OPS is reported as unreviewed — a blacklist alone cannot catch
# an op nobody has thought about yet.
ALLOWED_OPS = {
    "Conv", "BatchNormalization", "Relu", "Clip", "Add", "Mul", "Sub",
    "Concat", "Resize", "Abs", "AveragePool", "GlobalAveragePool", "Sigmoid",
    "Constant", "Identity", "Pad", "Div",
}

# Op -> the ACTUAL reason it is banned here.
FORBIDDEN_OPS: Dict[str, str] = {
    "Gather": (
        "Gather is SW_INT on Neural-ART, but the reason it is banned here is "
        "graph hygiene: in a pure convolution graph the only things that emit "
        "Gather are dynamic indexing and Shape->Gather->Concat size "
        "computation. head.py's old reg_level_scale selector buffers emitted "
        "it; the per-level off_pred/size_pred convolutions that replaced them "
        "must not."),
    "ScatterND": (
        "Decode logic leaked into the graph. Nothing in forward_raw writes "
        "into a tensor by index; a ScatterND means box assembly or NMS "
        "bookkeeping was traced."),
    "NonMaxSuppression": (
        "NMS must run on the host. ST documents detection post-processing "
        "including NMS as a host responsibility with no NPU support, and an "
        "in-graph NMS makes the output shape dynamic, which breaks the "
        "fixed-size INT8 buffers on both targets."),
    "Shape": (
        "A dynamic shape leaked in. Every spatial dimension is known at "
        "export time; a Shape node means something read .shape at runtime "
        "(typically F.interpolate(size=x.shape[-2:])), which also drags in "
        "Gather and Concat."),
    "Loop": "Control flow cannot be statically traced or scheduled on the NPU.",
    "If": "Control flow cannot be statically traced or scheduled on the NPU.",
    "Softmax": (
        "Decode leaked in — nothing in forward_raw normalises over a class "
        "axis, and with NUM_CLASSES=1 there is no axis to normalise. "
        "Separately, ST maps Softmax on hardware only when the compiler is "
        "invoked with --expand-softmax, and SW_INT otherwise."),
    "InstanceNormalization": (
        "SW_FLOAT on Neural-ART with no exception. Every normalisation in "
        "this network is BatchNorm, which folds into the preceding "
        "convolution and disappears entirely."),
    "Softplus": (
        "SW_FLOAT on Neural-ART with no exception. All activations are ReLU6, "
        "which exports as Clip and is unconditionally hardware-mapped."),
    "ReduceSum": (
        "SW_FLOAT on Neural-ART. ReduceMean is hardware-mapped when it is "
        "convertible to GlobalAveragePool, but ReduceSum is not convertible "
        "and has no hardware path."),
}

# Explicitly allowed, so nobody 'tightens' the list later: these are all
# hardware-mapped on Neural-ART and expected in this graph.
ALLOWED_NOTE = (
    "Resize, Exp, Reshape, Slice, Transpose, Sqrt, Erf, Pow (constant "
    "exponent), ReduceMean (convertible to GlobalAveragePool), Clip, Concat, "
    "Add, Mul, Sub, Sigmoid, AveragePool and Conv are hardware-mapped and are "
    "NOT forbidden."
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
        if op == "Div":
            # Div with a CONSTANT divisor is hardware-mapped and fine (it is
            # how a fixed scale is folded). Div with a runtime tensor divisor
            # is SW_INT on the Cortex-M55 and makes the INT8 activation range
            # scene-dependent — that is exactly what
            # eaa_normalize_edges=True (a per-image e/e.mean()) produces.
            if len(node.input) >= 2 and node.input[1] not in consts:
                errs.append(
                    f"forbidden op Div with a NON-CONSTANT second operand at "
                    f"node '{node.name or '<anon>'}' (divisor "
                    f"'{node.input[1]}'): ST maps Div on hardware only when "
                    f"the second operand is a constant, so a runtime divisor "
                    f"becomes SW_INT on the Cortex-M55 and it makes the INT8 "
                    f"activation range depend on the scene. This is what "
                    f"cfg.model.eaa_normalize_edges=True emits (ReduceMean + "
                    f"Div). Set it False and stabilise brightness in "
                    f"preprocessing (flat-field + CLAHE) instead.")
    return errs


def _check_resize_nodes(graph) -> List[str]:
    """
    Every Resize must be nearest/asymmetric/floor.

    ST maps Resize Nearest on hardware only with
    coordinate_transformation_mode='asymmetric' and nearest_mode='floor'.
    PyTorch opset-12 mode="nearest" + scale_factor normally produces exactly
    that pair, but a torch version bump can silently change the defaults, and
    the failure mode is a half-pixel shift in the FPN upsample that shows up
    only as a couple of mAP points.
    """
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
            errs.append(
                f"Resize '{tag}': coordinate_transformation_mode='{ctm}', "
                f"expected 'asymmetric' — ST requires it for the Neural-ART "
                f"hardware Resize path")
        if nm != "floor":
            errs.append(
                f"Resize '{tag}': nearest_mode='{nm}', expected 'floor' — ST "
                f"requires it for the Neural-ART hardware Resize path")
    print(f"[check] Resize nodes: {n_resize} "
          f"(expect 2: the two FPN top-down upsamples)")
    return errs


def _check_resize_scale_factor(graph) -> List[str]:
    """
    Every Resize must drive the ``scales`` input, never ``sizes``.

    F.interpolate(size=x.shape[-2:]) emits Shape -> Gather -> Concat to build
    the sizes input. F.interpolate(scale_factor=2.0,
    recompute_scale_factor=False) emits a constant scales initializer and no
    Gather at all. The neck uses the latter; this check proves it survived
    the export.
    """
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
                f"produced by Shape->Gather->Concat, which injects Gather "
                f"into the graph. Use scale_factor, not size=x.shape[-2:].")
        if not scales:
            errs.append(f"Resize '{tag}' has no 'scales' input")
        elif scales not in consts:
            errs.append(
                f"Resize '{tag}': 'scales' input '{scales}' is not a "
                f"constant, so the scale is computed at runtime")
    return errs


def _check_unreviewed_ops(graph) -> List[str]:
    """Report op types in neither the allow list nor the deny list."""
    seen = {n.op_type for n in graph.node}
    unknown = sorted(seen - ALLOWED_OPS - set(FORBIDDEN_OPS))
    for op in unknown:
        print(f"[warn] unreviewed op '{op}': not in ALLOWED_OPS or "
              f"FORBIDDEN_OPS. Confirm it maps on Neural-ART and NCNN INT8, "
              f"then add it to one of the two lists.")
    return []


def _check_static_shapes(graph) -> List[str]:
    """No dynamic dimension may survive export: both targets need fixed buffers."""
    errs: List[str] = []
    for vi in list(graph.input) + list(graph.output):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_param if d.dim_param else int(d.dim_value))
        if any(isinstance(d, str) or d == 0 for d in dims):
            errs.append(
                f"'{vi.name}' has a dynamic dimension {dims}: the export must "
                f"be static (dynamic_axes=None), or the INT8 buffers on the "
                f"Pi and the N6 cannot be sized at compile time")
    return errs


def _check_no_gather(graph) -> List[str]:
    """
    Hard zero-Gather assertion, separate from the forbidden-op sweep so the
    message can be specific.

    head.py's reg_level_scale used two registered selector buffers plus four
    broadcast ops per level. Those are gone, replaced by per-level off_pred
    and size_pred convolutions whose biases absorb the per-level scale. If a
    Gather reappears, either those buffers came back or a Resize fell back to
    the sizes input.
    """
    hits = [n.name or "<anon>" for n in graph.node if n.op_type == "Gather"]
    if not hits:
        print("[check] Gather nodes: 0")
        return []
    return [f"found {len(hits)} Gather node(s): {hits[:8]}. With per-level "
            f"prediction convolutions replacing reg_level_scale, nothing in "
            f"this graph should emit Gather. The usual causes are (a) a "
            f"registered selector buffer being indexed, (b) "
            f"F.interpolate(size=x.shape[-2:]) emitting "
            f"Shape->Gather->Concat."]


def _shape_map(model_proto) -> Dict[str, List[int]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model_proto)
    except Exception:
        inferred = model_proto
    out: Dict[str, List[int]] = {}
    g = inferred.graph
    for coll in (g.input, g.output, g.value_info):
        for vi in coll:
            dims: List[int] = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(int(d.dim_value) if d.dim_value > 0 else -1)
            out[vi.name] = dims
    for init in g.initializer:
        out[init.name] = list(init.dims)
    return out


def _check_feature_widths(model_proto) -> List[str]:
    """
    ST Neural-ART: for 8-bit feature data, feature WIDTH times the used batch
    depth (input channels) must be <= 2048, else the compiler splits the
    operator into columns. Also the AveragePool line-buffer limit.

    This is a WARNING, not an error. At 512x288 with a 64-channel head the
    stride-8 head input is 64 wide x 64 ch = 4096, so the compiler will split
    it. That is expected, documented, and cheaper than shrinking the canvas or
    the head; we want it visible, not fatal.
    """
    shapes = _shape_map(model_proto)
    warns: List[str] = []
    for node in model_proto.graph.node:
        if node.op_type not in ("Conv", "AveragePool", "MaxPool",
                                "GlobalAveragePool"):
            continue
        if not node.input:
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

def export(cfg: Config, checkpoint: Optional[str], device: torch.device,
           simplify: bool = True) -> Tuple[str, str]:
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
                           f"expected {expected} (3 blobs x "
                           f"{len(cfg.model.strides)} levels)")
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
        # STATIC shapes on purpose: no dynamic_axes. A dynamic batch or
        # spatial axis reintroduces Shape/Gather and breaks the fixed INT8
        # buffers on both targets.
        dynamic_axes=None,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"[export] {cfg.export.onnx_path} (opset {cfg.export.opset}, "
          f"static 1x1x{cfg.data.img_h}x{cfg.data.img_w})")

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
            print("[export] onnxsim not installed; skipping simplification "
                  "(pip install onnxsim). Checks run on the raw graph.")
        except Exception as exc:
            print(f"[export] simplification failed ({exc}); "
                  f"checks run on the raw graph")

    return cfg.export.onnx_path, sim_path


def run_checks(path: str) -> int:
    m = onnx.load(path)
    onnx.checker.check_model(m)
    g = m.graph

    print("=" * 70)
    print(f"  graph checks: {path}")
    print("=" * 70)
    print(f"[note] {ALLOWED_NOTE}")

    errs: List[str] = []
    errs += _check_forbidden_ops(g)
    errs += _check_unreviewed_ops(g)
    errs += _check_static_shapes(g)
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
                 checkpoint: Optional[str], tol: float = 1e-5) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[parity] onnxruntime not installed; skipping parity check")
        return True

    model = build_nirdet(cfg).to(device).eval()
    if checkpoint:
        from evaluate import load_checkpoint_into
        load_checkpoint_into(model, checkpoint, cfg, device)
        model.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 1, cfg.data.img_h, cfg.data.img_w, device=device)

    with torch.no_grad():
        ref: List[torch.Tensor] = []
        for c, o, s in model.forward_raw(x):
            ref += [c, o, s]

    sess = ort.InferenceSession(onnx_path,
                                providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name:
                          x.detach().cpu().numpy().astype(np.float32)})

    names = model.head.output_names()
    print("[parity] max |torch - onnx| per blob:")
    worst = 0.0
    ok = True
    for name, t, o in zip(names, ref, got):
        d = float(np.abs(t.detach().cpu().numpy() - np.asarray(o)).max())
        worst = max(worst, d)
        flag = "" if d < tol else "   <-- FAIL"
        print(f"    {name:<8} {d:.3e}{flag}")
        ok = ok and d < tol
    print(f"[parity] worst {worst:.3e} (tolerance {tol:.0e}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Export NIRDet-Lite to ONNX.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--device", default="cpu",
                    help="cpu is correct for export; cuda only changes "
                         "numerics in the parity check")
    args = ap.parse_args()

    cfg = get_config()
    if args.profile:
        from dataset_profiles import DatasetProfile
        DatasetProfile.load(args.profile).apply(cfg)
    validate_config(cfg, verbose=False)

    device = torch.device(args.device)
    raw_path, sim_path = export(cfg, args.checkpoint, device,
                                simplify=not args.no_simplify)

    n_err = run_checks(sim_path)

    parity_ok = True
    if not args.no_parity:
        parity_ok = parity_check(cfg, sim_path, device, args.checkpoint)

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
