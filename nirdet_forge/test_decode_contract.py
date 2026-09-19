"""
test_decode_contract.py — the decode contract is sacred
========================================================
    python test_decode_contract.py
    python test_decode_contract.py --checkpoint checkpoints/best.pth

Four decode constants (DECODE_OFFSET_SCALE, DECODE_OFFSET_BIAS,
REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX) plus MIN_BOX_PX and NIRDET_NUM_CLASSES
appear in six decoders: config.py (the source), head.py, losses.py,
evaluate_onnx.py, live_nirdet.py and nirdet_pp.c.

Five of the six import them. nirdet_pp.c cannot — so it is now GENERATED from
config.py by gen_contract_c.py, and T8 proves the file on disk is what the
current config.py would emit. T1 keeps greping the values anyway: the grep is
cheap, it is readable in a failure report, and it still catches the case where
someone hand-edits the generated file and forgets to re-emit.

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

from config import (DECODE_OFFSET_BIAS, DECODE_OFFSET_SCALE, MIN_BOX_PX,
                    NUM_CLASSES, REG_LOG_CLAMP_MAX, REG_LOG_CLAMP_MIN,
                    STRIDES, get_config)
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
        raise FileNotFoundError(
            f"{path} not found. It is GENERATED: run "
            f"`python gen_contract_c.py --emit`.")
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    out: Dict[str, float] = {}
    for name in ("DECODE_OFFSET_SCALE", "DECODE_OFFSET_BIAS",
                 "REG_LOG_CLAMP_MIN", "REG_LOG_CLAMP_MAX",
                 "NIRDET_MIN_BOX_PX"):
        m = re.search(rf"^\s*#define\s+{name}\s+(-?[0-9.]+)f?\s*$",
                      src, re.MULTILINE)
        if not m:
            raise AssertionError(f"#define {name} not found in {path}")
        out[name] = float(m.group(1))
    for name in ("NIRDET_NUM_CLASSES", "NIRDET_MAX_LEVELS", "NIRDET_MAX_DET"):
        m = re.search(rf"^\s*#define\s+{name}\s+(\d+)\s*$", src, re.MULTILINE)
        out[name] = float(m.group(1)) if m else float("nan")
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
    # The MEMBERSHIP half of the contract, previously unrepresented in C.
    check(d["NIRDET_MIN_BOX_PX"] == MIN_BOX_PX,
          f"NIRDET_MIN_BOX_PX {d['NIRDET_MIN_BOX_PX']} == MIN_BOX_PX "
          f"{MIN_BOX_PX}")
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
    # The Pi runtime must stay torch-free. dataset.py and dataset_profiles.py
    # both pull torch in transitively, so importing EITHER is a failure.
    # Strip the module docstring so prose examples cannot trip the import check.
    code_src = re.sub(r'^""".*?"""', '', src, count=1, flags=re.DOTALL)
    for banned in (r"^\s*import\s+torch\b", r"^\s*from\s+torch\b",
                   r"^\s*from\s+dataset\b", r"^\s*import\s+dataset\b",
                   r"^\s*from\s+dataset_profiles\b",
                   r"^\s*import\s+dataset_profiles\b"):
        check(re.search(banned, code_src, re.MULTILINE) is None,
              f"live_nirdet.py does not match {banned!r} (torch-free rule)")


# ===========================================================================
# T3: nine blobs, correct channel counts
# ===========================================================================

def t3_blob_layout() -> None:
    print("\nT3  three blobs per level: cls(1) / off(2) / size(2)")
    cfg = get_config(model=dict(prior_w=0.05, prior_h=0.15))
    head = PedestrianHead(in_channels=(48, 64, 64),
                          strides=cfg.model.strides,
                          prior_w=cfg.model.prior_w,
                          prior_h=cfg.model.prior_h).eval()
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
    NumPy 2's int64+float32->float64 promotion does not cause a spurious
    ~1.2e-4 diff at large coordinates.
    """
    cols = np.asarray(cols, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.float32)
    stride = np.asarray(stride, dtype=np.float32)

    def sig(x):
        # Two-branch logistic, matching nirdet_sigmoid EXACTLY — no argument
        # clamp, because the C does not clamp either.
        x = np.asarray(x, dtype=np.float32)
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = np.float32(1.0) / (np.float32(1.0) + np.exp(-x[pos]))
        e = np.exp(x[~pos])
        out[~pos] = e / (np.float32(1.0) + e)
        return out

    cx = (DECODE_OFFSET_SCALE * sig(t_cx) - DECODE_OFFSET_BIAS + cols) * stride
    cy = (DECODE_OFFSET_SCALE * sig(t_cy) - DECODE_OFFSET_BIAS + rows) * stride
    bw = np.exp(np.clip(t_w, REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX)) * img_w
    bh = np.exp(np.clip(t_h, REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX)) * img_h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)


def t4_round_trip() -> None:
    print("\nT4  numerical round trip: losses.decode vs the C formula")
    cfg = get_config(model=dict(prior_w=0.05, prior_h=0.15))
    h, w = cfg.data.img_h, cfg.data.img_w
    geom = AnchorGeometry(h, w, cfg.model.strides)

    torch.manual_seed(7)
    raw = torch.empty(1, geom.num_cells, 4)
    raw[..., 0].uniform_(-14.0, 14.0)
    raw[..., 1].uniform_(-14.0, 14.0)
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
        check(b.shape[0] == 1, f"decode_onnx_outputs found {b.shape[0]} box")
    except ImportError as exc:
        print(f"  SKIP  evaluate_onnx not importable here ({exc})")


# ===========================================================================
# T5: head inference decode == losses decode
# ===========================================================================

def t5_head_vs_losses() -> None:
    print("\nT5  head.forward(training_mode=False) == AnchorGeometry.decode")
    cfg = get_config(model=dict(prior_w=0.05, prior_h=0.15))
    h, w = cfg.data.img_h, cfg.data.img_w
    head = PedestrianHead(in_channels=(48, 64, 64),
                          strides=cfg.model.strides,
                          prior_w=cfg.model.prior_w,
                          prior_h=cfg.model.prior_h).eval()
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
    cfg = get_config(model=dict(prior_w=0.05, prior_h=0.15))
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
    # From CONFIG, not from train (changed): config.py already owns these
    # keys and documents why — `from train import ...` while train.py is the
    # __main__ module imports a SECOND copy of train.py under the name
    # `train`, re-running its module body inside the test process.
    from config import CKPT_DEPLOY_KEY
    check(CKPT_DEPLOY_KEY in ck,
          f"checkpoint carries '{CKPT_DEPLOY_KEY}' (EMA deploy weights)")


# ===========================================================================
# T7: no removed head machinery
# ===========================================================================

def t7_no_reg_level_scale() -> None:
    print("\nT7  reg_level_scale and its selector buffers are gone")
    cfg = get_config(model=dict(prior_w=0.05, prior_h=0.15))
    head = PedestrianHead(in_channels=(48, 64, 64),
                          prior_w=cfg.model.prior_w, prior_h=cfg.model.prior_h)
    names = [n for n, _ in head.named_parameters()]
    bufs = [n for n, _ in head.named_buffers()]
    check(not any("reg_level_scale" in n for n in names + bufs),
          "no reg_level_scale parameter or buffer")
    check(not any(n.endswith("_w_sel") or n.endswith("_h_sel") for n in bufs),
          "no _w_sel / _h_sel selector buffers")
    check(hasattr(head, "off_pred") and hasattr(head, "size_pred"),
          "off_pred and size_pred exist (per-level, independent QDQ scales)")
    check(not hasattr(head, "reg_pred"), "no combined reg_pred")


# ===========================================================================
# T8: nirdet_pp.c is generated, current, and stamped
# ===========================================================================

def t8_c_is_generated() -> None:
    """
    T1 proves five values match. T8 proves the WHOLE FILE — including the
    membership order and the max_det placement, which no grep can see — is
    what gen_contract_c.py would emit from the current config.py.
    """
    print("\nT8  nirdet_pp.c is generated from config.py and up to date")
    try:
        import gen_contract_c
    except ImportError as exc:
        check(False, f"gen_contract_c.py not importable ({exc})")
        return

    with open(C_FILE, "r", encoding="utf-8") as fh:
        src = fh.read()

    check("GENERATED BY gen_contract_c.py" in src,
          "nirdet_pp.c carries the generated-file provenance banner")

    stamp = gen_contract_c.contract_stamp(gen_contract_c.contract_values())
    m = re.search(r'#define\s+NIRDET_CONTRACT_STAMP\s+"([0-9a-f]+)"', src)
    check(m is not None, "NIRDET_CONTRACT_STAMP #define is present")
    if m:
        check(m.group(1) == stamp,
              f"NIRDET_CONTRACT_STAMP {m.group(1)} == config-derived {stamp}")

    ok, report = gen_contract_c.check(C_FILE)
    check(ok, "nirdet_pp.c is byte-identical to gen_contract_c.render()")
    if not ok:
        print("        " + report.replace("\n", "\n        ")[:1500])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("  decode contract tests")
    print("=" * 70)
    print(f"  S={DECODE_OFFSET_SCALE} B={DECODE_OFFSET_BIAS} "
          f"clamp=[{REG_LOG_CLAMP_MIN}, {REG_LOG_CLAMP_MAX}] "
          f"min_box_px={MIN_BOX_PX} strides={STRIDES} "
          f"num_classes={NUM_CLASSES}")

    t1_c_constants()
    t2_live_imports()
    t3_blob_layout()
    t4_round_trip()
    t5_head_vs_losses()
    t6_contract_hash(args.checkpoint)
    t7_no_reg_level_scale()
    t8_c_is_generated()

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
