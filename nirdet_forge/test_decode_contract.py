"""
test_decode_contract.py — the decode contract is sacred
========================================================
    python test_decode_contract.py
    python test_decode_contract.py --checkpoint checkpoints/best.pth

Five geometry constants (DECODE_OFFSET_SCALE, DECODE_OFFSET_BIAS,
REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX, STRIDES) appear in six decoders:
config.py (the source), head.py, losses.py, evaluate_onnx.py, live_nirdet.py
and nirdet_pp.c. Five of the six import them. nirdet_pp.c cannot, so this file
greps it.

A drift in any one of them shifts every box by up to a full stride or changes
the exp() range, and the symptom is a few lost mAP points with no traceback.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Dict, Optional

import numpy as np
import torch

from config import (DECODE_OFFSET_BIAS, DECODE_OFFSET_SCALE, NUM_CLASSES,
                    REG_LOG_CLAMP_MAX, REG_LOG_CLAMP_MIN, STRIDES, get_config)
from head import PedestrianHead
from losses import AnchorGeometry

HERE = os.path.dirname(os.path.abspath(__file__))
C_FILE = os.path.join(HERE, "nirdet_pp.c")
LIVE_FILE = os.path.join(HERE, "live_nirdet.py")

_failures: list = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _failures.append(msg)


# ===========================================================================
# T1: constants in nirdet_pp.c
# ===========================================================================

def _c_defines(path: str) -> Dict[str, float]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found; the C post-processor is "
                                f"part of the decode contract")
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    out: Dict[str, float] = {}
    for name in ("DECODE_OFFSET_SCALE", "DECODE_OFFSET_BIAS",
                 "REG_LOG_CLAMP_MIN", "REG_LOG_CLAMP_MAX"):
        m = re.search(rf"^\s*#define\s+{name}\s+(-?[0-9.]+)f?\s*$",
                      src, re.MULTILINE)
        if not m:
            raise AssertionError(f"#define {name} not found in {path}")
        out[name] = float(m.group(1))
    m = re.search(r"^\s*#define\s+NIRDET_NUM_CLASSES\s+(\d+)\s*$",
                  src, re.MULTILINE)
    out["NIRDET_NUM_CLASSES"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"^\s*#define\s+NIRDET_MAX_LEVELS\s+(\d+)\s*$",
                  src, re.MULTILINE)
    out["NIRDET_MAX_LEVELS"] = float(m.group(1)) if m else float("nan")
    return out


def t1_c_constants() -> None:
    print("\nT1  nirdet_pp.c #defines == config.py")
    d = _c_defines(C_FILE)
    check(d["DECODE_OFFSET_SCALE"] == DECODE_OFFSET_SCALE,
          f"DECODE_OFFSET_SCALE {d['DECODE_OFFSET_SCALE']} == "
          f"{DECODE_OFFSET_SCALE}")
    check(d["DECODE_OFFSET_BIAS"] == DECODE_OFFSET_BIAS,
          f"DECODE_OFFSET_BIAS {d['DECODE_OFFSET_BIAS']} == "
          f"{DECODE_OFFSET_BIAS}")
    check(d["REG_LOG_CLAMP_MIN"] == REG_LOG_CLAMP_MIN,
          f"REG_LOG_CLAMP_MIN {d['REG_LOG_CLAMP_MIN']} == {REG_LOG_CLAMP_MIN}")
    check(d["REG_LOG_CLAMP_MAX"] == REG_LOG_CLAMP_MAX,
          f"REG_LOG_CLAMP_MAX {d['REG_LOG_CLAMP_MAX']} == {REG_LOG_CLAMP_MAX}")
    check(d["NIRDET_NUM_CLASSES"] == float(NUM_CLASSES),
          f"NIRDET_NUM_CLASSES == NUM_CLASSES == {NUM_CLASSES}")
    check(d["NIRDET_MAX_LEVELS"] >= float(len(STRIDES)),
          f"NIRDET_MAX_LEVELS {d['NIRDET_MAX_LEVELS']:.0f} >= "
          f"{len(STRIDES)} levels")


# ===========================================================================
# T2: live_nirdet.py imports rather than hardcodes
# ===========================================================================

def t2_live_imports() -> None:
    print("\nT2  live_nirdet.py imports the constants (no literals)")
    if not os.path.isfile(LIVE_FILE):
        check(False, f"{LIVE_FILE} not found")
        return
    with open(LIVE_FILE, "r", encoding="utf-8") as fh:
        src = fh.read()
    for name in ("DECODE_OFFSET_SCALE", "DECODE_OFFSET_BIAS",
                 "REG_LOG_CLAMP_MIN", "REG_LOG_CLAMP_MAX"):
        imported = re.search(rf"\b{name}\b", src) is not None
        assigned = re.search(rf"^\s*{name}\s*=", src, re.MULTILINE) is not None
        check(imported and not assigned,
              f"{name} referenced and not redefined in live_nirdet.py")
    check(re.search(r"from\s+config\s+import", src) is not None,
          "live_nirdet.py imports from config")
    # No score-threshold literal: the value comes from the profile or the CLI.
    check(re.search(r"score[_-]thresh\w*\s*=\s*0\.\d+", src) is None,
          "no hardcoded score-threshold literal in live_nirdet.py")


# ===========================================================================
# T3: nine blobs, correct channel counts
# ===========================================================================

def t3_blob_layout() -> None:
    print("\nT3  three blobs per level: cls(1) / off(2) / size(2)")
    cfg = get_config()
    head = PedestrianHead(in_channels=(48, 64, 64),
                          strides=cfg.model.strides).eval()
    names = head.output_names()
    n_lvl = len(cfg.model.strides)
    check(len(names) == 3 * n_lvl,
          f"output_names() has {len(names)} entries (3 x {n_lvl})")
    expect = [f"{p}{s}" for s in cfg.model.strides
              for p in ("cls", "off", "size")]
    check(names == expect, f"names == {expect}")

    feats = [torch.zeros(1, c, cfg.data.img_h // s, cfg.data.img_w // s)
             for c, s in zip((48, 64, 64), cfg.model.strides)]
    raw = head.forward_raw(feats)
    check(len(raw) == n_lvl, f"forward_raw returns {len(raw)} levels")
    ok = True
    for lvl, (c, o, sz) in enumerate(raw):
        ok = ok and (c.shape[1] == NUM_CLASSES and o.shape[1] == 2
                     and sz.shape[1] == 2)
        print(f"        L{lvl}: cls {c.shape[1]}ch off {o.shape[1]}ch "
              f"size {sz.shape[1]}ch")
    check(ok, "channel counts are (1, 2, 2) at every level")

    contract = cfg.deploy_contract()
    check(int(contract["blobs_per_level"]) == 3,
          "deploy_contract blobs_per_level == 3")
    check(contract["blob_channels"] == {"cls": 1, "off": 2, "size": 2},
          f"deploy_contract blob_channels == {contract['blob_channels']}")


# ===========================================================================
# T4: numerical round trip, Python vs the C formula
# ===========================================================================

def _c_decode_numpy(t_cx, t_cy, t_w, t_h, cols, rows, stride, img_w, img_h):
    """Literal transcription of nirdet_decode_level's arithmetic.

    All intermediates in float32: cols/rows/stride are cast explicitly so
    NumPy 2's new int64+float32->float64 promotion does not cause a spurious
    ~1.2e-4 diff at large coordinates (2 float32 ulps at ~1000 px).
    """
    cols = np.asarray(cols, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.float32)
    stride = np.asarray(stride, dtype=np.float32)
    def sig(x):
        x = np.clip(x, -10.0, 10.0)
        return np.float32(1.0) / (np.float32(1.0) + np.exp(-np.asarray(x, np.float32)))
    cx = (DECODE_OFFSET_SCALE * sig(t_cx) - DECODE_OFFSET_BIAS + cols) * stride
    cy = (DECODE_OFFSET_SCALE * sig(t_cy) - DECODE_OFFSET_BIAS + rows) * stride
    bw = np.exp(np.clip(t_w, REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX)) * img_w
    bh = np.exp(np.clip(t_h, REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX)) * img_h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)


def t4_round_trip() -> None:
    print("\nT4  numerical round trip: losses.decode vs the C formula")
    cfg = get_config()
    h, w = cfg.data.img_h, cfg.data.img_w
    geom = AnchorGeometry(h, w, cfg.model.strides)

    torch.manual_seed(7)
    # Offsets in a realistic +/-6 band; sizes spanning and exceeding the clamp
    # so the clamp itself is exercised on both sides.
    raw = torch.empty(1, geom.num_cells, 4)
    raw[..., 0].uniform_(-6.0, 6.0)
    raw[..., 1].uniform_(-6.0, 6.0)
    raw[..., 2].uniform_(-8.0, 3.0)
    raw[..., 3].uniform_(-8.0, 3.0)

    py = geom.decode(raw)[0].numpy()

    r = raw[0].numpy()
    c_out = _c_decode_numpy(
        r[:, 0], r[:, 1], r[:, 2], r[:, 3],
        geom.cols.numpy(), geom.rows.numpy(),
        geom.strides_flat.numpy(), float(w), float(h))

    d = float(np.abs(py - c_out).max())
    print(f"        cells {geom.num_cells}, max |python - C| = {d:.3e}")
    check(d < 5e-4, f"max|diff| {d:.3e} < 5e-4  (tolerance covers float32 "
          f"exp() ulp differences; real contract drift is >=100 px)")

    # live_nirdet.decode_level, end to end on one level
    try:
        from live_nirdet import decode_level
        s = int(cfg.model.strides[0])
        gh, gw = h // s, w // s
        cls = np.full((1, gh, gw), 6.0, np.float32)
        off = np.zeros((2, gh, gw), np.float32)
        size = np.zeros((2, gh, gw), np.float32)
        size[0] = math.log(cfg.model.prior_w)
        size[1] = math.log(cfg.model.prior_h)
        boxes, scores = decode_level(cls, off, size, s, h, w, 0.5)
        check(boxes.shape[0] == gh * gw,
              f"decode_level returned all {gh * gw} cells above threshold")
        bw_med = float(np.median(boxes[:, 2] - boxes[:, 0]))
        bh_med = float(np.median(boxes[:, 3] - boxes[:, 1]))
        check(abs(bw_med - cfg.model.prior_w * w) < 0.5,
              f"decode_level width {bw_med:.2f} == prior_w*W "
              f"{cfg.model.prior_w * w:.2f}")
        check(abs(bh_med - cfg.model.prior_h * h) < 0.5,
              f"decode_level height {bh_med:.2f} == prior_h*H "
              f"{cfg.model.prior_h * h:.2f}")
    except ImportError as exc:
        print(f"  SKIP  live_nirdet not importable here ({exc})")

    # evaluate_onnx.decode_onnx_outputs
    try:
        from evaluate_onnx import decode_onnx_outputs
        outs = {}
        for s in cfg.model.strides:
            gh, gw = h // s, w // s
            outs[f"cls{s}"] = np.full((1, 1, gh, gw), -10.0, np.float32)
            outs[f"off{s}"] = np.zeros((1, 2, gh, gw), np.float32)
            outs[f"size{s}"] = np.zeros((1, 2, gh, gw), np.float32)
        s0 = int(cfg.model.strides[0])
        outs[f"cls{s0}"][0, 0, 2, 3] = 8.0
        outs[f"size{s0}"][0, 0, 2, 3] = math.log(cfg.model.prior_w)
        outs[f"size{s0}"][0, 1, 2, 3] = math.log(cfg.model.prior_h)
        b, sc = decode_onnx_outputs(outs, cfg, 0.5)
        check(b.shape[0] == 1, "decode_onnx_outputs found exactly 1 box")
        exp_cx = (DECODE_OFFSET_SCALE * 0.5 - DECODE_OFFSET_BIAS + 3) * s0
        got_cx = float((b[0, 0] + b[0, 2]) / 2)
        check(abs(got_cx - exp_cx) < 1e-3,
              f"decode_onnx_outputs cx {got_cx:.3f} == {exp_cx:.3f}")
    except ImportError as exc:
        print(f"  SKIP  evaluate_onnx not importable here ({exc})")


# ===========================================================================
# T5: head inference decode == losses decode
# ===========================================================================

def t5_head_vs_losses() -> None:
    print("\nT5  head.forward(training_mode=False) == AnchorGeometry.decode")
    cfg = get_config()
    h, w = cfg.data.img_h, cfg.data.img_w
    head = PedestrianHead(in_channels=(48, 64, 64),
                          strides=cfg.model.strides).eval()
    torch.manual_seed(3)
    feats = [torch.randn(1, c, h // s, w // s)
             for c, s in zip((48, 64, 64), cfg.model.strides)]
    with torch.no_grad():
        packed = head(feats, training_mode=True)
        dec = head(feats, training_mode=False)

    geom = AnchorGeometry(h, w, cfg.model.strides)
    raw = torch.cat([p[..., :4] for p in packed], dim=1)
    ref = geom.decode(raw)[0].numpy()

    flat = torch.cat(dec, dim=1)[0].numpy()
    cx, cy, bw, bh = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    got = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)

    d = float(np.abs(ref - got).max())
    print(f"        max |head - losses| = {d:.3e}")
    check(d < 1e-3, f"head and losses agree to {d:.3e}")


# ===========================================================================
# T6: checkpoint contract hash
# ===========================================================================

def t6_contract_hash(ckpt_path: Optional[str]) -> None:
    print("\nT6  checkpoint deploy_contract_hash")
    cfg = get_config()
    want = cfg.deploy_contract()["hash"]
    print(f"        current contract hash: {want}")
    if not ckpt_path:
        print("  SKIP  no --checkpoint given")
        return
    if not os.path.isfile(ckpt_path):
        check(False, f"checkpoint not found: {ckpt_path}")
        return
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    got = str(ck.get("deploy_contract_hash", ""))
    check(got == want,
          f"checkpoint hash {got or '<absent>'} == config hash {want}")
    if got != want:
        print("        The checkpoint was trained with different geometry "
              "(canvas, strides, or the decode constants). Every box it "
              "produces would be decoded against the wrong grid.")
    from train import CKPT_DEPLOY_KEY
    check(CKPT_DEPLOY_KEY in ck,
          f"checkpoint carries '{CKPT_DEPLOY_KEY}' (EMA deploy weights)")


# ===========================================================================
# T7: no removed head machinery
# ===========================================================================

def t7_no_reg_level_scale() -> None:
    print("\nT7  reg_level_scale and its selector buffers are gone")
    head = PedestrianHead(in_channels=(48, 64, 64))
    names = [n for n, _ in head.named_parameters()]
    bufs = [n for n, _ in head.named_buffers()]
    check(not any("reg_level_scale" in n for n in names + bufs),
          "no reg_level_scale parameter or buffer")
    check(not any(n.endswith("_w_sel") or n.endswith("_h_sel") for n in bufs),
          "no _w_sel / _h_sel selector buffers")
    check(hasattr(head, "off_pred") and hasattr(head, "size_pred"),
          "off_pred and size_pred exist (per-level, independent QDQ scales)")
    check(not hasattr(head, "reg_pred"), "no combined reg_pred")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("  decode contract tests")
    print("=" * 70)
    print(f"  S={DECODE_OFFSET_SCALE} B={DECODE_OFFSET_BIAS} "
          f"clamp=[{REG_LOG_CLAMP_MIN}, {REG_LOG_CLAMP_MAX}] "
          f"strides={STRIDES} num_classes={NUM_CLASSES}")

    t1_c_constants()
    t2_live_imports()
    t3_blob_layout()
    t4_round_trip()
    t5_head_vs_losses()
    t6_contract_hash(args.checkpoint)
    t7_no_reg_level_scale()

    print("=" * 70)
    if _failures:
        print(f"  {len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"    - {f}")
        print("=" * 70)
        return 1
    print("  all decode contract tests PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
