#!/usr/bin/env python
"""
export_onnx.py — NIRDet-Lite -> static ONNX (network only)
===========================================================
What is exported
----------------
    img (1, 1, H, W) float32 in [0, 1]
      -> cls3 (1, 1, H/8,  W/8)   raw logits
         reg3 (1, 4, H/8,  W/8)   raw (t_cx, t_cy, t_w, t_h), level scale folded in
         cls4 (1, 1, H/16, W/16)
         reg4 (1, 4, H/16, W/16)

What is NOT exported, and why
-----------------------------
Everything after the convolutions: the grid construction (arange/meshgrid),
sigmoid/exp decode, stack(dim=-1), reshape(B, H*W, 5), the score filter, and
NMS. Those produce Shape/Gather/Concat/NonZero debris that onnx2ncnn handles
badly and stedgeai handles worse, and ST documents detection post-processing
as explicitly unsupported (host M55 responsibility). The decode lives in
live_nirdet.py (Pi) and nirdet_pp.c (STM32), both bit-for-bit aligned with
head.py.

Guarantees
----------
  * batch 1, fully static shapes, dynamic_axes=None
  * opset 12 (11-13 is the sweet spot for onnx2ncnn; avoid 17+)
  * do_constant_folding=True
  * BatchNorm everywhere -> folds into the preceding conv downstream
  * Resize with a constant ``scales`` input (scale_factor, never size=)
  * an op-type whitelist check that fails loudly on export-hostile nodes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from config import get_config
from model import build_nirdet

# Ops that a clean NIRDet-Lite graph may contain.
ALLOWED_OPS = {
    "Conv", "BatchNormalization", "Relu", "Clip", "Add", "Mul", "Concat",
    "Resize", "Abs", "AveragePool", "Sigmoid", "Constant", "Identity", "Pad",
}
# Ops that are known to be software fallbacks on Neural-ART and/or to break
# NCNN INT8 fusion. Presence means the model surgery was not fully applied.
FORBIDDEN_OPS = {
    "ReduceMean": "GroupNorm or EAA normalize_edges is still present",
    "ReduceSum": "software float on Neural-ART",
    "Div": "Div with a runtime divisor is not HW-mapped (EAA normalize_edges)",
    "Sqrt": "software float (GroupNorm/InstanceNorm decomposition)",
    "Pow": "software float (GroupNorm decomposition)",
    "InstanceNormalization": "no hardware mapping",
    "Shape": "dynamic shape debris; use scale_factor, not size=",
    "Gather": "dynamic shape debris",
    "ScatterND": "in-place index_put_ in the head level-scale path; use pure Mul+Add with selector buffers",
    "Slice": "graph surgery incomplete: constant slicing debris; re-run simplify",
    "NonZero": "post-processing leaked into the graph",
    "TopK": "post-processing leaked into the graph",
    "NonMaxSuppression": "NMS must run on the host",
    "Exp": "decode leaked into the graph",
    "Reshape": "decode leaked into the graph",
    "Transpose": "layout debris; check for a stray permute",
    "Softplus": "software float",
    "Erf": "GELU/SiLU not expanded; switch to ReLU6",
}


class NIRDetExport(nn.Module):
    """
    Thin wrapper: trunk + head convolutions, nothing else.

    Returns a flat tuple so onnx output_names map 1:1 and the runtime can
    fetch blobs by name without unpacking nested structures.
    """

    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor):
        outs: List[torch.Tensor] = []
        for cls_logit, reg_raw in self.net.forward_raw(x):
            outs.append(cls_logit)
            outs.append(reg_raw)
        return tuple(outs)


def load_weights(net: nn.Module, ckpt_path: Path, prefer: str = "deploy") -> str:
    # weights_only=False is required because checkpoints store cfg.to_dict()
    # as a plain Python dict via pickle. If you switch to saving config as a
    # sidecar JSON, change this to weights_only=True.
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    order = ["deploy_state", "model_state", "model_state_dict", "state_dict"]
    if prefer == "live":
        order = ["model_state"] + [k for k in order if k != "model_state"]

    state, used = None, None
    if isinstance(ckpt, dict):
        for k in order:
            v = ckpt.get(k)
            if isinstance(v, dict) and v:
                state = v.get("weights", v) if "weights" in v else v
                used = k
                break
        if state is None and all(torch.is_tensor(t) for t in ckpt.values()):
            state, used = ckpt, "<bare state_dict>"
    if state is None:
        raise RuntimeError(f"no weights found in {ckpt_path}")

    inc = net.load_state_dict(state, strict=False)
    if inc.missing_keys:
        print(f"  ! missing keys   : {inc.missing_keys[:6]}")
    if inc.unexpected_keys:
        print(f"  ! unexpected keys: {inc.unexpected_keys[:6]}")
    return used


def check_graph(path: Path, strict: bool = True) -> bool:
    try:
        import onnx
    except ImportError:
        print("  ! onnx not installed; skipping graph audit "
              "(pip install onnx onnxsim)")
        return True

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"  op types ({len(ops)}): {ops}")

    bad = [(o, FORBIDDEN_OPS[o]) for o in ops if o in FORBIDDEN_OPS]
    unknown = [o for o in ops if o not in ALLOWED_OPS and o not in FORBIDDEN_OPS]

    for o, why in bad:
        print(f"  ! FORBIDDEN {o:<22} {why}")
    for o in unknown:
        print(f"  ? unreviewed {o:<21} verify it maps on your target")

    # Every Resize must have a constant scales/sizes input.
    inits = {i.name for i in m.graph.initializer}
    consts = {n.output[0] for n in m.graph.node if n.op_type == "Constant"}
    for n in m.graph.node:
        if n.op_type != "Resize":
            continue
        dyn = [i for i in n.input[1:] if i and i not in inits and i not in consts]
        if dyn:
            print(f"  ! Resize '{n.name}' has non-constant inputs {dyn}: "
                  f"use F.interpolate(scale_factor=...), not size=")
            bad.append(("Resize", "non-constant scales"))

    for vi in list(m.graph.input) + list(m.graph.output):
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in vi.type.tensor_type.shape.dim]
        print(f"  {vi.name:<6} {dims}")
        if any(isinstance(d, str) or d == 0 for d in dims):
            print(f"  ! '{vi.name}' has a dynamic dimension; both targets "
                  f"require fully static shapes")
            bad.append(("dynamic_shape", vi.name))

    if bad and strict:
        raise SystemExit("\nexport aborted: the graph contains export-hostile "
                         "nodes. Re-check that neck out_channels=64, the head "
                         "uses BatchNorm (not GroupNorm), activations are "
                         "ReLU6, EAA has normalize_edges=False and "
                         "padding_mode='zeros', and interpolation uses "
                         "scale_factor.")
    return not bad


def simplify(src: Path, dst: Path, h: int, w: int) -> bool:
    """onnxsim collapses the Pad/Shape debris and the Resize scales into
    constants. This is not optional."""
    try:
        import onnxsim  # noqa: F401
    except ImportError:
        print("  ! onnxsim not installed; skipping "
              "(pip install onnxsim) — STRONGLY recommended")
        return False
    cmd = [sys.executable, "-m", "onnxsim", str(src), str(dst),
           "--overwrite-input-shape", f"img:1,1,{h},{w}"]
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        return False
    return dst.exists()


def verify_parity(net: nn.Module, wrapper: nn.Module, onnx_path: Path,
                  h: int, w: int) -> None:
    """PyTorch vs onnxruntime on the same input. If this drifts, nothing
    downstream can be trusted."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("  ! onnxruntime not installed; skipping parity check")
        return

    torch.manual_seed(0)
    x = torch.rand(1, 1, h, w)
    with torch.no_grad():
        ref = [t.numpy() for t in wrapper(x)]

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    got = sess.run(None, {"img": x.numpy()})
    names = [o.name for o in sess.get_outputs()]

    worst = 0.0
    for n, a, b in zip(names, ref, got):
        d = float(np.abs(a - b).max())
        worst = max(worst, d)
        print(f"  {n:<6} shape {tuple(b.shape)}  max|dPyTorch-ORT| = {d:.3e}")
    if worst > 1e-4:
        raise SystemExit(f"parity failure: max abs diff {worst:.3e} > 1e-4")
    print(f"  parity OK (worst {worst:.3e})")


def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser("NIRDet-Lite ONNX export")
    ap.add_argument("--checkpoint",
                    default=str(Path(cfg.train.checkpoint_dir) / "best.pth"))
    ap.add_argument("--weights", choices=("deploy", "live"), default="deploy")
    ap.add_argument("--out", default=cfg.export.onnx_path)
    ap.add_argument("--sim-out", default=cfg.export.onnx_sim_path)
    ap.add_argument("--img-h", type=int, default=cfg.data.img_h)
    ap.add_argument("--img-w", type=int, default=cfg.data.img_w)
    ap.add_argument("--opset", type=int, default=cfg.export.opset)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--no-strict", action="store_true",
                    help="warn instead of aborting on forbidden ops")
    ap.add_argument("--random-weights", action="store_true",
                    help="export an untrained net (toolchain smoke test)")
    args = ap.parse_args()

    h, w = args.img_h, args.img_w
    for s in cfg.model.strides:
        if h % s or w % s:
            raise SystemExit(f"{h}x{w} is not divisible by stride {s}")

    print("=" * 62)
    print("  NIRDet-Lite ONNX export")
    print("=" * 62)

    net = build_nirdet(cfg)
    if args.random_weights:
        print("  weights: RANDOM (--random-weights)")
    else:
        used = load_weights(net, Path(args.checkpoint), args.weights)
        print(f"  checkpoint : {args.checkpoint}")
        print(f"  weights key: {used}")
    net.eval()

    # Fuse-friendly, deterministic state for tracing.
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

    print(f"  input      : (1, 1, {h}, {w}) float32 in [0,1]")
    print(f"  strides    : {cfg.model.strides}")
    print(f"  params     : {sum(p.numel() for p in net.parameters()):,}")

    wrapper = NIRDetExport(net).eval()
    dummy = torch.zeros(1, 1, h, w)

    out_names: List[str] = []
    for s in cfg.model.strides:
        out_names += [f"cls{s}", f"reg{s}"]
    # Backwards-compatible aliases used across the deployment files.
    alias = {"cls8": "cls3", "reg8": "reg3", "cls16": "cls4", "reg16": "reg4",
             "cls32": "cls5", "reg32": "reg5"}
    out_names = [alias.get(n, n) for n in out_names]
    print(f"  outputs    : {out_names}")

    with torch.no_grad():
        probe = wrapper(dummy)
    for n, t in zip(out_names, probe):
        print(f"     {n:<6} {tuple(t.shape)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n  exporting ...")
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        dynamo=False,
        opset_version=args.opset,
        input_names=["img"],
        output_names=out_names,
        dynamic_axes=None,               # fully static: both targets require it
        do_constant_folding=True,
        export_params=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    print("\n  graph audit (raw)")
    check_graph(out_path, strict=False)

    final = out_path
    if not args.no_simplify:
        print("\n  simplifying ...")
        try:
            import onnxsim  # noqa: F401
        except ImportError:
            raise SystemExit(
                "onnxsim is required for a safe export; "
                "install it with:  pip install onnxsim\n"
                "Or pass --no-simplify to skip (not recommended for deployment)."
            )
        sim = Path(args.sim_out)
        if simplify(out_path, sim, h, w):
            final = sim
            print(f"  wrote {sim} ({sim.stat().st_size / 1e6:.2f} MB)")
            print("\n  graph audit (simplified)")
            check_graph(sim, strict=not args.no_strict)

    # Strict audit always runs on the final graph, whether simplified or not.
    print("\n  graph audit (final — strict)")
    check_graph(final, strict=not args.no_strict)

    print("\n  parity check")
    verify_parity(net, wrapper, final, h, w)

    print("\n" + "=" * 62)
    print("  next steps")
    print("=" * 62)
    print("  Raspberry Pi 5 / NCNN INT8:")
    print(f"    onnx2ncnn {final} nirdet.param nirdet.bin")
    print("    ncnnoptimize nirdet.param nirdet.bin \\")
    print("        nirdet-opt.param nirdet-opt.bin 0")
    print("    find <train images> -name '*.jpg' | shuf -n 300 > calib.txt")
    print("    ncnn2table nirdet-opt.param nirdet-opt.bin calib.txt \\")
    print("        nirdet.table mean=0,0,0 norm=0.00392156862745,0,0 \\")
    print(f"        shape=[{w},{h},1] pixel=GRAY thread=4 method=kl")
    print("    ncnn2int8 nirdet-opt.param nirdet-opt.bin \\")
    print("        nirdet-int8.param nirdet-int8.bin nirdet.table")
    print("    # then read nirdet-opt.param: any GroupNorm / Reduction /")
    print("    # BinaryOp-with-tensor / Pooling(32x32) means fix the model,")
    print("    # not the toolchain.")
    print("\n  STM32N6570-DK:")
    print(f"    python quantize_qdq.py --onnx {final} \\")
    print(f"        --data-root <dataset root> --img-h {h} --img-w {w}")
    print(f"    stedgeai analyze --model {cfg.export.onnx_int8_path} \\")
    print("        --target stm32n6 \\")
    print("        --st-neural-art default@user_neuralart.json \\")
    print("        --input-data-type uint8")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
