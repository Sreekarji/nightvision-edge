"""
test_losses.py — loss, assigner and geometry invariants
========================================================
    python test_losses.py

T5 pins the canvas change: 512x288 gives 3024 cells, not the 5040 of the old
384x640 canvas. T7 pins the cold-start assertion, which is the one guard
standing between a broken size-prior init and a run that trains a
classification-only model with a perfectly plausible loss curve.
"""

from __future__ import annotations

import math
import sys

import torch

from config import get_config
from losses import (AnchorGeometry, ColdStartError, NIRDetLoss,
                    cxcywh_to_xyxy)

_failures: list = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _failures.append(msg)


def _dummy(b: int, geom: AnchorGeometry, seed: int = 0):
    torch.manual_seed(seed)
    return [torch.randn(b, h * w, 5) for h, w in geom.grid_sizes]


def t1_empty_gt() -> None:
    print("\nT1  empty ground truth")
    lf = NIRDetLoss()
    out = lf(_dummy(2, lf.geom, 0), [[], []])
    check(bool(torch.isfinite(out["total"])), "total is finite")
    check(float(out["reg"]) == 0.0, "regression term is exactly 0")
    check(float(out["n_pos"]) == 0.0, "no positives")
    print(f"        total {float(out['total']):.6f} "
          f"cls {float(out['cls']):.6f}")


def t2_direction() -> None:
    print("\nT2  matched predictions lower the loss")
    lf = NIRDetLoss()
    lf.set_epoch(lf.total_epochs)
    gt = torch.tensor([[0.35, 0.5, 0.05, 0.18]])
    rand = _dummy(1, lf.geom, 7)
    out_rand = lf(rand, [[gt[0]]])

    good = [p.clone() for p in rand]
    from config import DECODE_OFFSET_BIAS as B, DECODE_OFFSET_SCALE as S
    for lvl, (h, w) in enumerate(lf.geom.grid_sizes):
        col = min(int(gt[0, 0] * w), w - 1)
        row = min(int(gt[0, 1] * h), h - 1)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if not (0 <= r < h and 0 <= c < w):
                    continue
                i = r * w + c
                tx = torch.tensor((float(gt[0, 0]) * w - c + B) / S)
                ty = torch.tensor((float(gt[0, 1]) * h - r + B) / S)
                good[lvl][0, i, 0] = torch.logit(tx.clamp(1e-4, 1 - 1e-4))
                good[lvl][0, i, 1] = torch.logit(ty.clamp(1e-4, 1 - 1e-4))
                good[lvl][0, i, 2] = math.log(float(gt[0, 2]))
                good[lvl][0, i, 3] = math.log(float(gt[0, 3]))
                good[lvl][0, i, 4] = 6.0
    out_good = lf(good, [[gt[0]]])
    print(f"        random {float(out_rand['total']):.6f} -> "
          f"matched {float(out_good['total']):.6f} "
          f"(iou {float(out_good['iou']):.3f}, "
          f"n_pos {int(out_good['n_pos'])})")
    check(float(out_good["total"]) < float(out_rand["total"]),
          "matched loss < random loss")
    check(float(out_good["iou"]) > 0.8, "matched IoU > 0.8")


def t3_finite() -> None:
    print("\nT3  NaN/Inf guard over 25 random batches")
    lf = NIRDetLoss(assert_cold_start=False)
    bad = []
    for t in range(25):
        lf.set_epoch(t)
        torch.manual_seed(t * 991)
        preds = _dummy(4, lf.geom, t)
        gt = []
        for _ in range(4):
            k = int(torch.randint(0, 5, ()))
            if k == 0:
                gt.append([])
            else:
                bx = torch.rand(k, 4)
                bx[:, 2] = bx[:, 2] * 0.25 + 0.02
                bx[:, 3] = bx[:, 3] * 0.35 + 0.05
                bx[:, 0] = bx[:, 0] * 0.8 + 0.1
                bx[:, 1] = bx[:, 1] * 0.8 + 0.1
                gt.append(bx)
        out = lf(preds, gt)
        for k, v in out.items():
            if not bool(torch.isfinite(v).all()):
                bad.append(f"trial {t}: {k}")
    check(not bad, f"all outputs finite (offenders: {bad[:3]})")


def t4_collision() -> None:
    print("\nT4  collision resolution: smallest area wins")
    lf = NIRDetLoss()
    lf.set_epoch(lf.total_epochs)
    W, H = float(lf.geom.img_w), float(lf.geom.img_h)
    big = torch.tensor([0.5, 0.5, 0.40, 0.80])
    small = torch.tensor([0.5, 0.5, 0.05, 0.16])
    preds = _dummy(1, lf.geom, 3)
    _, boxes, _, _nf = lf.assigner(
        torch.sigmoid(torch.cat([p[..., 4] for p in preds], 1)),
        lf.geom.to(torch.device("cpu")).decode(
            torch.cat([p[..., :4] for p in preds], 1)),
        cxcywh_to_xyxy(torch.stack([big, small]).unsqueeze(0) *
                       torch.tensor([W, H, W, H])),
        torch.ones(1, 2), lf.geom.centers_px())
    areas = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    # Find cells assigned to EITHER GT (positive cells only).
    # At least one cell must be assigned to the small GT.
    small_area = 0.05 * W * 0.16 * H
    big_area = 0.40 * W * 0.80 * H
    pos_areas = areas[0][areas[0] > 0]
    n_small = int((pos_areas < (small_area + big_area) / 2).sum())
    print(f"        small GT area {small_area:.0f}, big GT area {big_area:.0f}, "
          f"cells assigned to small GT: {n_small}")
    check(n_small >= 1,
          f"at least one cell assigned to the smaller GT ({n_small} found)")
    check(float(pos_areas.min()) < (small_area + big_area) / 2,
          "minimum assigned area is the small GT area, not the big GT area")


def t5_geometry() -> None:
    print("\nT5  geometry is derived from resolution; 512x288 -> 3024 cells")
    cfg = get_config(model=dict(prior_w=0.05, prior_h=0.15))
    for (h, w) in ((288, 512), (384, 640), (256, 416), (288, 1024)):
        g = AnchorGeometry(h, w, (8, 16, 32))
        expect = ((h // 8) * (w // 8) + (h // 16) * (w // 16)
                  + (h // 32) * (w // 32))
        check(g.num_cells == expect,
              f"{h}x{w}: {g.num_cells} cells, grids {g.grid_sizes}")
    g = AnchorGeometry(288, 512, (8, 16, 32))
    check(g.num_cells == 3024,
          f"288x512 -> {g.num_cells} cells (was 5040 at 384x640)")
    check(g.grid_sizes == ((36, 64), (18, 32), (9, 16)),
          f"grids {g.grid_sizes}")
    check((cfg.data.img_h, cfg.data.img_w) == (288, 512) and
          cfg.num_cells == 3024,
          f"config agrees: {cfg.num_cells} cells")


def t6_per_level() -> None:
    print("\nT6  per-level positive counts sum to n_pos")
    lf = NIRDetLoss()
    lf.set_epoch(lf.total_epochs)
    gt = [torch.tensor([[0.5, 0.5, 0.06, 0.20],
                        [0.2, 0.7, 0.30, 0.60]])]
    out = lf(_dummy(1, lf.geom, 11), gt)
    keys = sorted(k for k in out if k.startswith("n_pos_l"))
    check(len(keys) == len(lf.geom.strides),
          f"{len(keys)} per-level keys for {len(lf.geom.strides)} levels")
    print("        " + "  ".join(f"{k}={float(out[k]):.0f}" for k in keys))
    tot = sum(float(out[k]) for k in keys)
    check(abs(tot - float(out["n_pos"])) < 1e-6,
          f"sum {tot:.0f} == n_pos {float(out['n_pos']):.0f}")


def t7_cold_start() -> None:
    print("\nT7  cold-start assertion")
    cfg = get_config(model=dict(prior_w=0.05, prior_h=0.15))
    lf = NIRDetLoss()
    lf.set_epoch(0)
    preds = [torch.zeros(1, h * w, 5) for h, w in lf.geom.grid_sizes]
    for p in preds:
        p[..., 2] = math.log(cfg.model.prior_w)
        p[..., 3] = math.log(cfg.model.prior_h)
        p[..., 4] = -4.595                        # logit(0.01)
    out = lf(preds, [torch.tensor([[0.5, 0.5, cfg.model.prior_w,
                                    cfg.model.prior_h]])])
    check(int(out["n_pos"]) > 0,
          f"prior-initialised size logits give n_pos={int(out['n_pos'])} "
          f"and do NOT raise")

    lf2 = NIRDetLoss()
    lf2.set_epoch(0)
    bad = [torch.zeros(1, h * w, 5) for h, w in lf2.geom.grid_sizes]
    for p in bad:
        p[..., 2] = -6.0                          # 1.3 px wide boxes
        p[..., 3] = -6.0
        p[..., 4] = -4.595
    raised = False
    msg = ""
    try:
        lf2(bad, [torch.tensor([[0.5, 0.5, cfg.model.prior_w,
                                 cfg.model.prior_h]])])
    except ColdStartError as exc:
        raised = True
        msg = str(exc)
    check(raised, "a broken size init raises ColdStartError")
    check("size_pred.bias" in msg,
          "the error names head.size_pred.bias as the first thing to check")


def t9_fallback_assignment() -> None:
    print("\nT9  sub-stride GT box receives a fallback assignment")
    lf = NIRDetLoss()
    lf.set_epoch(0)

    # Construct a GT box that is guaranteed to contain NO stride-8 cell centre.
    # At 512x288, stride-8 cell centres are at (4, 12, 20, ...) horizontally.
    # A 3px-wide box at x=8..11 contains no centre.
    W, H = float(lf.geom.img_w), float(lf.geom.img_h)
    tiny_cx = 9.5 / W      # centre at pixel 9.5
    tiny_w  = 3.0 / W      # 3 px wide: centres at 4 and 12 are outside
    tiny_cy = 0.5
    tiny_h  = 0.10
    gt = [torch.tensor([[tiny_cx, tiny_cy, tiny_w, tiny_h]])]
    preds = _dummy(1, lf.geom, 0)
    out = lf(preds, gt)
    check(int(out["n_pos"]) >= 1,
          f"sub-stride GT gets n_pos={int(out['n_pos'])} >= 1 (fallback works)")
    check(int(out["n_gt_fallback"]) >= 1,
          f"n_gt_fallback={int(out['n_gt_fallback'])} >= 1")


def t8_tal_settings() -> None:
    print("\nT8  TAL alignment exponents")
    cfg = get_config(model=dict(prior_w=0.05, prior_h=0.15))
    lf = NIRDetLoss()
    check(lf.assigner.alpha == 0.5,
          f"tal_alpha == 0.5 (got {lf.assigner.alpha}) — the YOLOv8 setting. "
          f"At 1.0 the classification score dominates t = cls^a * iou^b, and "
          f"on a single-class head initialised to a 0.01 prior those early "
          f"scores are almost pure noise.")
    check(lf.assigner.beta == 6.0, f"tal_beta == 6.0 (got {lf.assigner.beta})")
    check(cfg.loss.tal_alpha == 0.5 and cfg.loss.tal_beta == 6.0,
          "config.LossCfg agrees")
    check(lf.qfl.alpha < 0.0,
          f"QFL alpha weighting disabled ({lf.qfl.alpha}) — reference GFL and "
          f"YOLOv8 use none")


def main() -> int:
    print("=" * 70)
    print("  losses.py tests")
    print("=" * 70)
    t1_empty_gt()
    t2_direction()
    t3_finite()
    t4_collision()
    t5_geometry()
    t6_per_level()
    t7_cold_start()
    t8_tal_settings()
    t9_fallback_assignment()
    print("=" * 70)
    if _failures:
        print(f"  {len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  all loss tests PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
