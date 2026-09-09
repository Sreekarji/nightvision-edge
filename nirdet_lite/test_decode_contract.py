"""
test_decode_contract.py — single-source check for the box decode contract
==========================================================================

The decode constants live in config.py and are hand-mirrored in
live_nirdet.py (Python floats) and nirdet_pp.c (C float literals):

    config.py        DECODE_OFFSET_SCALE, DECODE_OFFSET_BIAS,
                     REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX
    live_nirdet.py   OFF_S, OFF_B, REG_CLAMP_MIN, REG_CLAMP_MAX
    nirdet_pp.c      NIRDET_OFF_S, NIRDET_OFF_B,
                     NIRDET_REG_CLAMP_MIN, NIRDET_REG_CLAMP_MAX

A silent drift in any copy corrupts every box decoded on the device, and no
training-time metric would explain why. This file pins every copy to config:

  1. the four constants import from config.py,
  2. live_nirdet.py's ``NAME = <float>`` copies match config exactly,
  3. nirdet_pp.c's ``#define NAME <float>f`` copies match config exactly,
  4. head.py's inference decode agrees with a reference re-implementation of
     the C decode formula to within 1e-3 on random raw logits (the random
     range deliberately crosses both reg clamps and saturates the sigmoid),
  5. log(prior_w) / log(prior_h) sit strictly inside the reg clamp.

Run after changing ANY of: config.py, head.py, losses.py, live_nirdet.py,
nirdet_pp.c.

    python test_decode_contract.py          # or: pytest test_decode_contract.py
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DECODE_OFFSET_BIAS,
    DECODE_OFFSET_SCALE,
    ModelCfg,
    REG_LOG_CLAMP_MAX,
    REG_LOG_CLAMP_MIN,
)
from head import PedestrianHead

_DIR = Path(__file__).resolve().parent
_TOL = 1e-3


# ===========================================================================
# 1. config exports the contract
# ===========================================================================

def test_config_constants_exist():
    for v in (DECODE_OFFSET_SCALE, DECODE_OFFSET_BIAS,
              REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX):
        assert isinstance(v, float), f"decode constant {v} is not a float"
    # internal consistency, mirrors the definition in config.py
    assert DECODE_OFFSET_BIAS == (DECODE_OFFSET_SCALE - 1.0) / 2.0
    assert REG_LOG_CLAMP_MIN < REG_LOG_CLAMP_MAX


# ===========================================================================
# text extraction
# ===========================================================================

_LIVE_NAMES = ("OFF_S", "OFF_B", "REG_CLAMP_MIN", "REG_CLAMP_MAX")
_C_NAMES = ("NIRDET_OFF_S", "NIRDET_OFF_B",
            "NIRDET_REG_CLAMP_MIN", "NIRDET_REG_CLAMP_MAX")

# NAME = <float>  (optional trailing comment)
_LIVE_RE = re.compile(
    r"^\s*(OFF_S|OFF_B|REG_CLAMP_MIN|REG_CLAMP_MAX)\s*=\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:#.*)?$",
    re.MULTILINE,
)

# #define NAME <float>f  (optional surrounding parens, trailing comment)
_C_RE = re.compile(
    r"^\s*#define\s+(NIRDET_OFF_S|NIRDET_OFF_B|NIRDET_REG_CLAMP_MIN|"
    r"NIRDET_REG_CLAMP_MAX)\s+\(?\s*([+-]?(?:\d+\.?\d*|\.\d+))f\s*\)?",
    re.MULTILINE,
)


def _extract(pattern, text, path, names):
    matches = list(pattern.finditer(text))
    found = {m.group(1): float(m.group(2)) for m in matches}
    assert len(matches) == len(names), (
        f"{path}: expected exactly {len(names)} constant definitions, "
        f"got {len(matches)} (duplicate or missing?)")
    assert set(found) == set(names), (
        f"{path}: found {sorted(found)}, expected {sorted(names)}")
    return found


# ===========================================================================
# 2. live_nirdet.py == config
# ===========================================================================

def test_live_nirdet_matches_config():
    text = (_DIR / "live_nirdet.py").read_text(encoding="utf-8")
    found = _extract(_LIVE_RE, text, "live_nirdet.py", _LIVE_NAMES)
    expected = {
        "OFF_S": DECODE_OFFSET_SCALE,
        "OFF_B": DECODE_OFFSET_BIAS,
        "REG_CLAMP_MIN": REG_LOG_CLAMP_MIN,
        "REG_CLAMP_MAX": REG_LOG_CLAMP_MAX,
    }
    for name, want in expected.items():
        assert found[name] == want, (
            f"live_nirdet.py {name} = {found[name]} != config {want}")


# ===========================================================================
# 3. nirdet_pp.c == config
# ===========================================================================

def test_nirdet_pp_c_matches_config():
    text = (_DIR / "nirdet_pp.c").read_text(encoding="utf-8")
    found = _extract(_C_RE, text, "nirdet_pp.c", _C_NAMES)
    expected = {
        "NIRDET_OFF_S": DECODE_OFFSET_SCALE,
        "NIRDET_OFF_B": DECODE_OFFSET_BIAS,
        "NIRDET_REG_CLAMP_MIN": REG_LOG_CLAMP_MIN,
        "NIRDET_REG_CLAMP_MAX": REG_LOG_CLAMP_MAX,
    }
    for name, want in expected.items():
        assert found[name] == want, (
            f"nirdet_pp.c {name} = {found[name]} != config {want}")


# ===========================================================================
# 4. head.py decode == reference re-implementation of the C formula
# ===========================================================================

class _FixedRegPred(nn.Module):
    """Stands in for reg_pred, returning pre-generated logit maps in order."""

    def __init__(self, maps):
        super().__init__()
        self.maps = [m.clone() for m in maps]
        self.i = 0

    def forward(self, x):
        t = self.maps[self.i % len(self.maps)]
        self.i += 1
        return t


def _ref_decode(t_cx, t_cy, t_w, t_h, cols, rows, stride, img_w, img_h):
    """Reference re-implementation of nirdet_pp.c's float32 decode path."""
    sx = 1.0 / (1.0 + np.exp(-t_cx))
    sy = 1.0 / (1.0 + np.exp(-t_cy))
    tw = np.clip(t_w, REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX)
    th = np.clip(t_h, REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX)
    cx = (DECODE_OFFSET_SCALE * sx - DECODE_OFFSET_BIAS + cols) * stride
    cy = (DECODE_OFFSET_SCALE * sy - DECODE_OFFSET_BIAS + rows) * stride
    bw = np.exp(tw) * img_w
    bh = np.exp(th) * img_h
    return cx, cy, bw, bh


def test_head_decode_matches_c_reference():
    torch.manual_seed(0)
    head = PedestrianHead(strides=(8, 16, 32)).eval()

    # 384x640 input / strides (8, 16, 32)
    grids = ((48, 80), (24, 40), (12, 20))
    strides = head.strides

    # Random raw logits well past both reg clamps and into both sigmoid
    # saturation regimes, so a missing clamp or wrong scale/bias cannot
    # slip through the 1e-3 comparison.
    g = torch.Generator().manual_seed(1234)
    maps = []
    for (h, w) in grids:
        m = torch.empty(1, 4, h, w)
        m[:, 0:2] = torch.empty(1, 2, h, w).uniform_(-6.0, 6.0, generator=g)
        m[:, 2:4] = torch.empty(1, 2, h, w).uniform_(-10.0, 3.0, generator=g)
        maps.append(m)
    head.reg_pred = _FixedRegPred(maps)

    feats = [torch.randn(1, 64, h, w, generator=g) for (h, w) in grids]
    with torch.no_grad():
        raw = head(feats, training_mode=True)    # (B, H*W, 5) raw logits
        dec = head(feats, training_mode=False)   # (B, H*W, 5) decoded px

    # forward_raw folds reg_level_scale into the w/h logits; the C side
    # receives those already-scaled values, so the reference adds it too.
    lvl_scale = head.reg_level_scale.detach().numpy()      # (3, 2), zeros here

    n_clamp_lo = n_clamp_hi = 0
    for lvl, ((h, w), s) in enumerate(zip(grids, strides)):
        t = raw[lvl][0].numpy().astype(np.float64)         # (H*W, 5)
        d = dec[lvl][0].numpy()                            # (H*W, 5)
        hw = h * w
        cols = (np.arange(hw) % w).astype(np.float64)
        rows = (np.arange(hw) // w).astype(np.float64)

        t_cx, t_cy = t[:, 0], t[:, 1]
        t_w = t[:, 2] + float(lvl_scale[lvl, 0])
        t_h = t[:, 3] + float(lvl_scale[lvl, 1])

        n_clamp_lo += int(np.sum(t_w < REG_LOG_CLAMP_MIN)
                          + np.sum(t_h < REG_LOG_CLAMP_MIN))
        n_clamp_hi += int(np.sum(t_w > REG_LOG_CLAMP_MAX)
                          + np.sum(t_h > REG_LOG_CLAMP_MAX))

        ref = _ref_decode(t_cx, t_cy, t_w, t_h, cols, rows, s, w * s, h * s)
        for name, ref_v, got_v in zip(("cx", "cy", "w", "h"), ref,
                                      (d[:, 0], d[:, 1], d[:, 2], d[:, 3])):
            err = float(np.max(np.abs(ref_v - got_v)))
            assert err < _TOL, (
                f"level {lvl} ({h}x{w}, stride {s}): {name} max err "
                f"{err:.3e} >= {_TOL} — head.py decode and the nirdet_pp.c "
                f"formula disagree")

    # the random logits must actually have exercised both clamp boundaries
    assert n_clamp_lo > 0 and n_clamp_hi > 0, (
        "test logits never crossed the reg clamp; widen the ranges")


# ===========================================================================
# 5. priors sit strictly inside the reg clamp
# ===========================================================================

def test_priors_inside_reg_clamp():
    m = ModelCfg()
    for label, prior in (("prior_w", m.prior_w), ("prior_h", m.prior_h)):
        v = math.log(prior)
        assert REG_LOG_CLAMP_MIN < v < REG_LOG_CLAMP_MAX, (
            f"log({label}) = {v:.3f} not strictly inside "
            f"({REG_LOG_CLAMP_MIN}, {REG_LOG_CLAMP_MAX})")


if __name__ == "__main__":
    test_config_constants_exist()
    print("PASS  config.py exports the decode contract")
    test_live_nirdet_matches_config()
    print("PASS  live_nirdet.py constants == config.py")
    test_nirdet_pp_c_matches_config()
    print("PASS  nirdet_pp.c #defines == config.py")
    test_head_decode_matches_c_reference()
    print(f"PASS  head.py decode == C reference within {_TOL} (clamps exercised)")
    test_priors_inside_reg_clamp()
    print("PASS  log(prior_w) / log(prior_h) strictly inside reg clamp")
    print("decode contract: all checks passed")
