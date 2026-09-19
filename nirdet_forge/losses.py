"""
losses.py — NIRDet-Lite loss (single class)
============================================
    L = lambda_cls * QFL(conf, target_score) + lambda_reg * CIoU(pos)

Both terms are normalised exactly once, by target_scores.sum() (the YOLOv8/GFL
convention). Task-Aligned Assignment replaces static centre-cell assignment.
CIoU is computed entirely in PIXEL space. GT collisions are resolved by
smallest-area-wins, not by array-overwrite order.

DESIGN POINTS
-------------
1. tal_alpha 0.5 (the YOLOv8 setting). t = cls^alpha * iou^beta; at alpha=1.0
   the classification score carries as much weight as in a many-class
   detector, but this head is single-class and initialised to a 0.01 prior.

2. COLD-START ASSERTION. At epoch 0 the only reason TAL finds any positive is
   that head.size_pred.bias is seeded to (log prior_w, log prior_h). Break
   that init and the assigner returns zero positives for the ENTIRE run while
   the loss curve still looks plausible.

3. NEAREST-CENTRE FALLBACK. Strict centre-in-box sampling gives zero
   candidates to any GT that contains no cell centre — at stride 8 the centres
   are 8 px apart, so a box narrower than one stride in either axis is never
   learned, silently. Every real GT now gets at least its nearest centre, and
   the frequency is reported as n_gt_fallback.

4. NO PER-STEP HOST SYNCS in the assigner (the collision resolver already
   avoids one deliberately; the early-out and the cold-start counter used to
   add two back).

5. PER-LEVEL POSITIVE COUNTS, so a dead stride-32 level is visible.

Decode is bit-for-bit identical to head.py, live_nirdet.decode_level,
evaluate_onnx.decode_onnx_outputs and nirdet_pp.c.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DECODE_OFFSET_BIAS as _OFF_B,
    DECODE_OFFSET_SCALE as _OFF_S,
    REG_LOG_CLAMP_MAX,
    REG_LOG_CLAMP_MIN,
)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# geometry helpers (all pixel space)
# ---------------------------------------------------------------------------

def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w * 0.5, cy - h * 0.5,
                        cx + w * 0.5, cy + h * 0.5], dim=-1)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a (..., M, 4) xyxy, b (..., N, 4) xyxy -> (..., M, N)."""
    a = a.unsqueeze(-2)                            # (..., M, 1, 4)
    b = b.unsqueeze(-3)                            # (..., 1, N, 4)
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + _EPS)


def ciou(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor
         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Element-wise CIoU in PIXEL space.

        L = 1 - IoU + rho^2/c^2 + alpha * v
        v = (4/pi^2) * (atan(wg/hg) - atan(wp/hp))^2
        alpha = v / ((1 - IoU) + v)     [detached, per Zheng et al. 2020]
    """
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    pw = (px2 - px1).clamp(min=_EPS)
    ph = (py2 - py1).clamp(min=_EPS)
    gw = (gx2 - gx1).clamp(min=_EPS)
    gh = (gy2 - gy1).clamp(min=_EPS)

    iw = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0)
    ih = (torch.minimum(py2, gy2) - torch.maximum(py1, gy1)).clamp(min=0)
    inter = iw * ih
    union = pw * ph + gw * gh - inter + _EPS
    iou_v = (inter / union).clamp(0.0, 1.0)

    pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    gcx, gcy = (gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    ex1 = torch.minimum(px1, gx1)
    ey1 = torch.minimum(py1, gy1)
    ex2 = torch.maximum(px2, gx2)
    ey2 = torch.maximum(py2, gy2)
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + _EPS

    v = (4.0 / (math.pi ** 2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / ((1.0 - iou_v) + v + _EPS)

    return 1.0 - iou_v + rho2 / c2 + alpha * v, iou_v.detach()


# ---------------------------------------------------------------------------
# anchor geometry
# ---------------------------------------------------------------------------

class AnchorGeometry:
    """
    Flat cell geometry for the concatenated multi-level prediction tensor.

    Everything is derived from (img_h, img_w, strides): nothing is hardcoded.
    """

    def __init__(self, img_h: int, img_w: int,
                 strides: Sequence[int] = (8, 16, 32)) -> None:
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.strides = tuple(int(s) for s in strides)
        for s in self.strides:
            if self.img_h % s or self.img_w % s:
                raise ValueError(f"{img_h}x{img_w} not divisible by stride {s}")

        self.grid_sizes: Tuple[Tuple[int, int], ...] = tuple(
            (self.img_h // s, self.img_w // s) for s in self.strides
        )
        self.level_sizes: Tuple[int, ...] = tuple(h * w for h, w in self.grid_sizes)
        self.num_cells: int = int(sum(self.level_sizes))

        cols, rows, strd = [], [], []
        for (h, w), s in zip(self.grid_sizes, self.strides):
            gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                                    torch.arange(w, dtype=torch.float32),
                                    indexing="ij")
            cols.append(gx.reshape(-1))
            rows.append(gy.reshape(-1))
            strd.append(torch.full((h * w,), float(s)))
        self._cols = torch.cat(cols)
        self._rows = torch.cat(rows)
        self._strides_flat = torch.cat(strd)
        self._centers_px = torch.stack(
            [(self._cols + 0.5) * self._strides_flat,
             (self._rows + 0.5) * self._strides_flat], dim=-1)
        self._cached_device = None

    def to(self, device: torch.device) -> "AnchorGeometry":
        if self._cached_device != device:
            self._cols = self._cols.to(device)
            self._rows = self._rows.to(device)
            self._strides_flat = self._strides_flat.to(device)
            self._centers_px = self._centers_px.to(device)
            self._cached_device = device
        return self

    @property
    def cols(self) -> torch.Tensor:
        return self._cols

    @property
    def rows(self) -> torch.Tensor:
        return self._rows

    @property
    def strides_flat(self) -> torch.Tensor:
        return self._strides_flat

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        """
        raw (B, N, 4) = (t_cx, t_cy, t_w, t_h) -> (B, N, 4) xyxy PIXELS.

        Bit-for-bit identical to head.py's inference decode,
        live_nirdet.decode_level and nirdet_pp.c. No clamp and no membership
        filter here on purpose: the loss must see every cell's prediction,
        including boxes that reach outside the canvas, or the regression
        gradient for edge objects disappears. The MEMBERSHIP contract
        (config.clamp_and_filter) applies to INFERENCE decoders only.
        """
        cols = self._cols.view(1, -1)
        rows = self._rows.view(1, -1)
        st = self._strides_flat.view(1, -1)

        cx = (_OFF_S * torch.sigmoid(raw[..., 0]) - _OFF_B + cols) * st
        cy = (_OFF_S * torch.sigmoid(raw[..., 1]) - _OFF_B + rows) * st
        w = torch.exp(raw[..., 2].clamp(REG_LOG_CLAMP_MIN,
                                        REG_LOG_CLAMP_MAX)) * self.img_w
        h = torch.exp(raw[..., 3].clamp(REG_LOG_CLAMP_MIN,
                                        REG_LOG_CLAMP_MAX)) * self.img_h
        return cxcywh_to_xyxy(torch.stack([cx, cy, w, h], dim=-1))

    def centers_px(self) -> torch.Tensor:
        """(N, 2) cell-centre coordinates in pixels (cached in __init__)."""
        return self._centers_px


# ---------------------------------------------------------------------------
# Task-Aligned Assigner
# ---------------------------------------------------------------------------

class TaskAlignedAssigner(nn.Module):
    """
    TOOD / YOLOv8-style dynamic assignment, single class.

    For each GT:
        t = cls_score^alpha * iou^beta over candidate cells
        keep top-k by t
    Conflicts -> smallest GT area wins. Confidence target = t normalised per
    GT and rescaled by that GT's best IoU.
    """

    def __init__(self, topk: int = 10, alpha: float = 0.5,
                 beta: float = 6.0, min_pos_target: float = 0.10,
                 level_sizes: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._min_pos_target = float(min_pos_target)
        # F65: the nearest-centre fallback is restricted to level 0. NIRDetLoss
        # wires this from self.geom.level_sizes.
        self._level_sizes_hint = (tuple(int(x) for x in level_sizes)
                                  if level_sizes is not None else None)

    @staticmethod
    def _centers_in_boxes(centers: torch.Tensor, gt_xyxy: torch.Tensor,
                          gt_mask: torch.Tensor, level_sizes=None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        centers (N,2), gt (B,M,4), gt_mask (B,M) -> (candidate mask (B,M,N),
        fallback flag (B,M), inside (B,M,N)).

        Strict centre-in-box, PLUS a nearest-centre fallback for any real GT
        that contains no cell centre at all (F66). At stride 8 the centres are
        8 px apart, so a box narrower than one stride in either axis can fall
        entirely between them and would otherwise receive zero candidates at
        EVERY level — i.e. never be learned, with no diagnostic. A far-field
        NIR pedestrian set has a non-trivial fraction of such boxes.

        F65: the fallback is restricted to level_sizes[0] (the stride-8
        level); a sub-stride box at stride 16 or 32 is truly undetectable at
        that resolution and rescuing it there injects noise.
        """
        cx = centers[:, 0].view(1, 1, -1)
        cy = centers[:, 1].view(1, 1, -1)
        x1, y1 = gt_xyxy[..., 0:1], gt_xyxy[..., 1:2]
        x2, y2 = gt_xyxy[..., 2:3], gt_xyxy[..., 3:4]
        inside = ((cx > x1) & (cx < x2) & (cy > y1) & (cy < y2)).to(gt_xyxy.dtype)

        gcx = (x1 + x2) * 0.5
        gcy = (y1 + y2) * 0.5
        d2 = (cx - gcx) ** 2 + (cy - gcy) ** 2                  # (B, M, N)

        # F02 (audit fix): restrict the fallback search to level 0 BEFORE the
        # argmin. Masking a GLOBAL argmin afterwards deletes the rescue
        # outright whenever the globally nearest centre belongs to a coarser
        # level — and it often does: stride-16/32 cell centres lie exactly on
        # stride-8 cell corners, so a GT centre near one of them is closer to
        # a coarse cell than to any stride-8 cell. The GT would then receive
        # zero candidates while n_gt_fallback still reported it as rescued.
        # d2 is freshly allocated above and not aliased, so the in-place
        # write is safe and costs no extra (B, M, N) allocation.
        if level_sizes is not None:
            n0 = int(level_sizes[0])
            if 0 < n0 < d2.shape[-1]:
                d2[..., n0:] = float("inf")

        nearest = torch.zeros_like(inside)
        nearest.scatter_(-1, d2.argmin(dim=-1, keepdim=True), 1.0)

        empty = (inside.sum(dim=-1, keepdim=True) <= 0) & \
                (gt_mask.unsqueeze(-1) > 0)
        # Return `inside` separately so the caller can identify fallback cells
        # and bypass the cand > _EPS filter for them (F63).
        return torch.where(empty, nearest, inside), empty.squeeze(-1), inside

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,   # (B, N)   sigmoid confidences
        pd_xyxy: torch.Tensor,     # (B, N, 4) pixels
        gt_xyxy: torch.Tensor,     # (B, M, 4) pixels
        gt_mask: torch.Tensor,     # (B, M)   1 = real GT, 0 = padding
        centers: torch.Tensor,     # (N, 2)   pixels
    ):
        """
        Returns:
            fg_mask       (B, N)     1.0 for positive cells
            target_boxes  (B, N, 4)  xyxy pixels (garbage where fg_mask == 0)
            target_scores (B, N)     soft confidence target in [0, 1]
            n_fallback    scalar     GTs that needed the nearest-centre rescue
        """
        b, n = pd_scores.shape
        m = gt_xyxy.shape[1]
        zeros_n = pd_scores.new_zeros(b, n)
        zero_scalar = pd_scores.new_zeros(())

        # F68: memory guard on the (B, M, N) candidate tensor before any of
        # the big allocations below.
        _MAX_CELLS_MEM = 50_000_000   # ~200 MB at float32 for a (B, M, N) tensor
        if b * m * n > _MAX_CELLS_MEM:
            raise RuntimeError(
                f"assigner would allocate a ({b}, {m}, {n}) candidate tensor "
                f"({b*m*n:,} elements ≈ {b*m*n*4//1024//1024} MB). "
                f"Reduce max GT boxes per image or batch size. "
                f"The most crowded frame has {m} boxes.")

        # NO `gt_mask.sum() == 0` early-out (F65): evaluating that as a Python
        # bool forces a device->host sync on EVERY training step, which is the
        # exact cost the collision resolver below deliberately avoids. An
        # all-zero gt_mask needs no special case anyway — gtm multiplies align
        # and in_box to zero, cand is all-zero, topk_mask is killed by
        # (cand > _EPS), and fg_mask/target_scores come out zero.
        if m == 0:
            return zeros_n, pd_scores.new_zeros(b, n, 4), zeros_n, zero_scalar

        gtm = gt_mask.unsqueeze(-1).to(pd_scores.dtype)        # (B, M, 1)

        ious = pairwise_iou(gt_xyxy, pd_xyxy).clamp(min=0)     # (B, M, N)
        scores = pd_scores.unsqueeze(1).expand(b, m, n)        # (B, M, N)
        # (ious + 1e-4) instead of ious.clamp(min=1e-9): a genuine zero IoU
        # gives (1e-9)^6 = 1e-54, which underflows to 0.0 in float32. The
        # resulting all-zero cand row means topk returns indices 0..k-1 by
        # tie-break order, and the fallback cell at index ~1200 is never
        # selected. Adding 1e-4 keeps the ordering intact without underflow.
        align = scores.clamp(min=_EPS).pow(self.alpha) * \
            (ious + 1e-4).pow(self.beta)
        align = align * gtm

        in_box, fell_back, inside = self._centers_in_boxes(
            centers, gt_xyxy, gt_mask,
            level_sizes=self._level_sizes_hint)
        cand = align * in_box * gtm

        k = min(self.topk, n)
        _, topk_idx = cand.topk(k, dim=-1)                     # (B, M, k)
        topk_mask = torch.zeros_like(cand)
        topk_mask.scatter_(-1, topk_idx, 1.0)
        # A zero-valued top-k slot is not a real candidate. But fallback
        # candidates (the nearest centre for GTs that have no cell centre
        # inside them) have align ≈ 0 at epoch 0 because iou^6 is tiny for a
        # sub-stride box, and the (cand > _EPS) filter used to remove exactly
        # the fallback cells _centers_in_boxes just rescued (F63). Preserve
        # any cell that was the nearest-centre fallback.
        fallback_cells = (in_box > 0) & \
                         (inside.sum(dim=-1, keepdim=True) == 0) & \
                         (gt_mask.unsqueeze(-1) > 0)
        # OR the fallback one-hot DIRECTLY into topk_mask — not into the
        # filter applied to it. The previous form multiplied an already-zero
        # topk_mask by 1, leaving the fallback cell unselected. The fallback
        # cell must be unconditionally in topk_mask regardless of whether
        # topk happened to select it.
        topk_mask = (topk_mask * (cand > _EPS).to(cand.dtype)) \
                    + fallback_cells.to(cand.dtype)
        topk_mask = topk_mask.clamp(max=1.0)

        mask_pos = topk_mask * in_box * gtm                    # (B, M, N)

        # ---- smallest-area-wins conflict resolution ----
        gt_area = ((gt_xyxy[..., 2] - gt_xyxy[..., 0]).clamp(min=0) *
                   (gt_xyxy[..., 3] - gt_xyxy[..., 1]).clamp(min=0))   # (B, M)
        fg_count = mask_pos.sum(dim=1)                          # (B, N)
        # Always resolve collisions — never gate this on
        # bool((fg_count > 1).any()), which forces a host sync. torch.where on
        # the broadcast form avoids the (B,M,N) .clone() + masked_fill_ pair
        # and the +inf sentinel, which could propagate NaN through a later
        # multiply-by-zero (F69).
        big = torch.finfo(mask_pos.dtype).max
        areas = torch.where(
            mask_pos > 0,
            gt_area.unsqueeze(-1).expand(b, m, n),
            torch.full((), big, device=mask_pos.device, dtype=mask_pos.dtype))
        best_gt = areas.argmin(dim=1)                           # (B, N)
        onehot = F.one_hot(best_gt, m).permute(0, 2, 1).to(mask_pos.dtype)
        multi = (fg_count > 1).unsqueeze(1).expand(b, m, n)
        mask_pos = torch.where(multi, onehot, mask_pos)
        fg_count = mask_pos.sum(dim=1)

        fg_mask = fg_count.clamp(max=1.0)                       # (B, N)
        target_gt_idx = mask_pos.argmax(dim=1)                  # (B, N)

        bidx = torch.arange(b, device=pd_scores.device).unsqueeze(-1)
        target_boxes = gt_xyxy[bidx, target_gt_idx]             # (B, N, 4)

        # ---- alignment-normalised confidence target ----
        align_pos = align * mask_pos
        max_align = align_pos.amax(dim=-1, keepdim=True)        # (B, M, 1)
        max_iou = (ious * mask_pos).amax(dim=-1, keepdim=True)  # (B, M, 1)
        # F64: when max_align is near zero (sub-stride box at epoch 0), dividing
        # by (max_align + _EPS) collapses the confidence target and regression
        # weight to zero, stopping gradient flow for exactly the boxes the
        # fallback was added for. Use a floor so fallback assignments get a
        # meaningful target.
        safe_max_align = torch.where(
            max_align > _EPS,
            max_align,
            torch.ones_like(max_align))
        norm = align_pos * max_iou / safe_max_align
        target_scores = norm.amax(dim=1) * fg_mask              # (B, N)

        # F64: ensure every positive cell (including fallback rescues) gets at
        # least tal_min_pos_target as its confidence target so it receives
        # gradient. This value comes from config.LossCfg.tal_min_pos_target.
        min_target = getattr(self, "_min_pos_target", 0.10)
        target_scores = torch.where(
            fg_mask > 0,
            target_scores.clamp(min=min_target),
            target_scores)

        return (fg_mask, target_boxes, target_scores.clamp(0.0, 1.0),
                fell_back.to(pd_scores.dtype).sum())


# ---------------------------------------------------------------------------
# Quality Focal Loss
# ---------------------------------------------------------------------------

class QualityFocalLoss(nn.Module):
    """QFL(sigma(x), y) = |y - sigma(x)|^beta * BCE(x, y), y continuous."""

    def __init__(self, beta: float = 2.0, alpha: float = -1.0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                 reduction="none")
        mod = (targets - p).abs().detach().pow(self.beta)
        if self.alpha < 0.0:
            # Reference QFL (Li et al., GFL 2020) and YOLOv8 use no alpha:
            # |y - p|^beta already down-weights easy negatives, and on a
            # single-class detector a 0.25 alpha would down-weight foreground
            # 3:1 for no reason.
            return mod * bce
        fg = (targets > 0).to(logits.dtype)
        w = self.alpha * fg + (1.0 - self.alpha) * (1.0 - fg)
        return mod * bce * w


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class ColdStartError(RuntimeError):
    """Task-Aligned Assignment produced no positives on the first real batch."""


class NIRDetLoss(nn.Module):
    """
    forward(predictions, gt_batch) -> dict with keys
        total, cls, reg, iou, n_pos, norm, norm_floored, n_gt_fallback,
        n_pos_l0 ... (one per level)

    predictions : list of (B, H_i*W_i, 5) RAW logits, level order == strides.
                  Column order is (t_cx, t_cy, t_w, t_h, conf_logit).
    gt_batch    : list of length B; each entry is a (N_i, 4) or (N_i, 5)
                  tensor of normalised boxes, a list of such rows, or []/None.
    """

    def __init__(
        self,
        img_h: int = 288,
        img_w: int = 512,
        strides: Sequence[int] = (8, 16, 32),
        lambda_cls: float = 1.0,
        lambda_reg: float = 2.5,
        qfl_beta: float = 2.0,
        qfl_alpha: float = -1.0,
        tal_topk: int = 10,
        tal_alpha: float = 0.5,
        tal_beta: float = 6.0,
        ramp_frac: float = 0.15,
        total_epochs: int = 100,
        assert_cold_start: bool = True,
        tal_min_pos_target: float = 0.10,
    ) -> None:
        super().__init__()
        self.geom = AnchorGeometry(img_h, img_w, strides)
        self.lambda_cls = float(lambda_cls)
        self.lambda_reg = float(lambda_reg)
        self.ramp_frac = float(ramp_frac)
        self.total_epochs = int(max(total_epochs, 1))
        self._epoch = 0

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, alpha=tal_alpha, beta=tal_beta,
            min_pos_target=tal_min_pos_target,
            # F65: restrict the nearest-centre fallback to the stride-8 level.
            level_sizes=self.geom.level_sizes)
        self.qfl = QualityFocalLoss(beta=qfl_beta, alpha=qfl_alpha)

        self._assert_cold_start = bool(assert_cold_start)
        self._cold_start_checked = False

    # ------------------------------------------------------------------ #

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def grid_sizes(self) -> Tuple[Tuple[int, int], ...]:
        return self.geom.grid_sizes

    @property
    def level_sizes(self) -> Tuple[int, ...]:
        return self.geom.level_sizes

    def _soft_ramp(self) -> float:
        """0.0 -> hard targets (1.0); 1.0 -> alignment-normalised targets."""
        end = max(1, int(self.ramp_frac * self.total_epochs))
        return float(min(1.0, self._epoch / end))

    # ------------------------------------------------------------------ #

    def _pad_gt(self, gt_batch, batch: int,
                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (B, M, 4) xyxy pixels, (B, M) validity mask."""
        per_img: List[torch.Tensor] = []
        for i in range(batch):
            g = gt_batch[i] if gt_batch is not None and i < len(gt_batch) else None
            if g is None:
                t = torch.zeros((0, 4), device=device)
            elif isinstance(g, (list, tuple)):
                t = (torch.stack([x.reshape(-1) for x in g], 0).to(device)
                     if len(g) else torch.zeros((0, 4), device=device))
            else:
                t = g.to(device).reshape(-1, g.shape[-1]) if g.numel() \
                    else torch.zeros((0, 4), device=device)
            if t.shape[-1] == 5:
                t = t[:, 1:5]          # single class: drop the class column
            elif t.shape[-1] != 4:
                raise ValueError(f"gt entry has last dim {t.shape[-1]}, want 4/5")
            per_img.append(t.float())

        m = max((t.shape[0] for t in per_img), default=0)
        boxes = torch.zeros(batch, max(m, 1), 4, device=device)
        mask = torch.zeros(batch, max(m, 1), device=device)
        scale = torch.tensor([self.geom.img_w, self.geom.img_h,
                              self.geom.img_w, self.geom.img_h], device=device)
        for i, t in enumerate(per_img):
            if t.shape[0]:
                boxes[i, : t.shape[0]] = cxcywh_to_xyxy(t * scale)
                mask[i, : t.shape[0]] = 1.0
        if m == 0:
            mask.zero_()
        return boxes, mask

    # ------------------------------------------------------------------ #

    def _check_cold_start(self, n_pos: int, n_gt: int) -> None:
        """
        Fail loudly if TAL finds no positive on the first batch that has
        ground truth at epoch 0.

        F03 (audit fix): the guard stays ARMED across GT-less batches. The
        previous version set _cold_start_checked = True BEFORE testing
        n_gt == 0, so an epoch-0 batch that happens to contain no ground
        truth — routine on a NIR pedestrian set with empty frames — marked
        the guard satisfied and disabled it for the whole run. The extra cost
        is one int(gt_mask.sum().item()) host sync per empty batch at epoch 0
        only, which is bounded and paid at most a handful of times.
        """
        if n_gt == 0:
            return                       # not the batch this guard is about
        self._cold_start_checked = True
        if n_pos > 0:
            return
        raise ColdStartError(
            f"Task-Aligned Assignment produced 0 positives on the first "
            f"epoch-0 batch, which had {n_gt} ground-truth box(es).\n"
            f"alignment metric t = cls^{self.assigner.alpha} * iou^{self.assigner.beta}, topk={self.assigner.topk}\n"
            f"At epoch 0 every cls score is ~prior_prob, so the iou term is "
            f"the only thing keeping t above zero, and it is only "
            f"non-negligible because head.size_pred.bias is initialised to "
            f"(log prior_w, log prior_h). Check, in order:\n"
            f" 1. head._init_predictions still writes log(prior_w/h) into "
            f"size_pred[lvl].bias\n"
            f" 2. cfg.model.prior_w / prior_h came from the active dataset "
            f"profile and are canvas-space for {self.geom.img_h}x"
            f"{self.geom.img_w}\n"
            f" 3. loss.tal_beta ({self.assigner.beta}) has not been raised\n"
            f" 4. the GT boxes are normalised (cx, cy, w, h) in [0, 1]\n"
            f"Training with zero positives produces a plausible-looking "
            f"classification-only loss curve and a useless model, so this is "
            f"a hard error. Pass assert_cold_start=False only if you know "
            f"exactly why you want it off.")

    # ------------------------------------------------------------------ #

    def forward(self, predictions: List[torch.Tensor],
        gt_batch) -> Dict[str, torch.Tensor]:
        if len(predictions) != len(self.geom.strides):
            raise ValueError(f"expected {len(self.geom.strides)} prediction "
                             f"levels, got {len(predictions)}")

        device = predictions[0].device
        self.geom.to(device)

        flat = torch.cat([p.reshape(p.shape[0], -1, 5) for p in predictions],
                         dim=1).float()                  # (B, N, 5)
        b, n, _ = flat.shape
        if n != self.geom.num_cells:
            raise ValueError(f"prediction has {n} cells, geometry expects "
                             f"{self.geom.num_cells}")

        raw_box = flat[..., :4]
        cls_logits = flat[..., 4]

        pd_xyxy = self.geom.decode(raw_box)              # (B, N, 4) px
        pd_scores = torch.sigmoid(cls_logits.detach())

        gt_xyxy, gt_mask = self._pad_gt(gt_batch, b, device)

        fg_mask, tgt_boxes, tgt_scores, n_fallback = self.assigner(
            pd_scores, pd_xyxy.detach(), gt_xyxy, gt_mask,
            self.geom.centers_px(),
        )

        # Hard -> soft ramp: at epoch 0 the IoUs are low, so pure alignment
        # targets would keep confidence pinned near zero.
        lam = self._soft_ramp()
        tgt_scores = (1.0 - lam) * fg_mask + lam * tgt_scores

        # ---- single normalisation constant for BOTH terms ----
        # The clamp(min=1) floor is the YOLOv8/GFL convention. When norm_raw < 1
        # (few positives, early training), the floor rescales BOTH loss terms by
        # 1/norm_raw, which inflates the effective learning rate proportionally.
        # This is reported via norm_floored; consecutive floors abort the run (F66).
        norm_raw = tgt_scores.sum()
        norm = norm_raw.clamp(min=1.0)

        # F66: per-epoch floored-batch counter, consumed and reset by
        # train_one_epoch() in train.py.
        if not hasattr(self, "_norm_floor_count"):
            self._norm_floor_count = 0
        if bool(norm_raw.detach() < 1.0):
            self._norm_floor_count += 1

        cls_loss = self.qfl(cls_logits, tgt_scores).sum() / norm

        pos = fg_mask > 0
        n_pos = int(pos.sum())

        per_level: Dict[str, torch.Tensor] = {}
        start = 0
        for lvl, size in enumerate(self.geom.level_sizes):
            per_level[f"n_pos_l{lvl}"] = fg_mask[:, start:start + size].sum().detach()
            start += size

        # Gate BEFORE the .item() so the host sync is paid once per run.
        if self._assert_cold_start and not self._cold_start_checked \
                and self._epoch == 0:
            self._check_cold_start(n_pos, int(gt_mask.sum().item()))

        if n_pos > 0:
            reg_raw, iou_val = ciou(pd_xyxy[pos], tgt_boxes[pos])
            w = tgt_scores[pos]
            reg_loss = (reg_raw * w).sum() / norm
            mean_iou = iou_val.mean()
        else:
            # reg_loss stays graph-connected (zero gradient); mean_iou is a
            # plain zero — a flat.sum()*0.0 there cost a full reduction over
            # every cell purely to log a zero (F68).
            reg_loss = flat.sum() * 0.0
            mean_iou = torch.zeros((), device=device)

        total = self.lambda_cls * cls_loss + self.lambda_reg * reg_loss

        out = {
            # NOT cast to the prediction dtype: train.py computes this loss
            # OUTSIDE autocast so the CIoU / alignment arithmetic stays fp32.
            "total": total,
            "cls": cls_loss.detach(),
            "reg": reg_loss.detach(),
            "iou": mean_iou.detach(),
            "n_pos": torch.tensor(float(n_pos), device=device),
            "norm": norm.detach(),
            "norm_floored": torch.tensor(
                float(bool(norm_raw.detach() < 1.0)), device=device),
            "n_gt_fallback": n_fallback.detach(),
        }
        out.update(per_level)
        return out
if __name__ == "__main__":
    # The self-tests live in test_losses.py — ONE home, no drifting duplicate
    # (F71). This entry point just runs them.
    import test_losses
    raise SystemExit(test_losses.main())
