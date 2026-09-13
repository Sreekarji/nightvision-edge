"""
losses.py — NIRDet-Lite loss (single class)
============================================
    L = lambda_cls * QFL(conf, target_score) + lambda_reg * CIoU(pos)

Both terms are normalised exactly once, by target_scores.sum() (the YOLOv8/GFL
convention). Task-Aligned Assignment replaces static centre-cell assignment.
CIoU is computed entirely in PIXEL space, so x- and y-distances are not
differentially compressed by the canvas aspect ratio. GT collisions are
resolved by smallest-area-wins, not by array-overwrite order.

WHAT CHANGED IN THIS REVISION
-----------------------------
1. tal_alpha 1.0 -> 0.5 (the YOLOv8 setting). The alignment metric is
   t = cls^alpha * iou^beta. At alpha=1.0 the classification score carries as
   much weight as it does in a many-class detector, but this head is
   single-class and initialised to a 0.01 prior, so early cls scores are
   almost pure noise and weighting them fully makes epoch-0 assignment noisier
   than it needs to be. beta stays 6.0.

2. COLD-START ASSERTION. At epoch 0 the only reason TAL finds any positive is
   that head.size_pred.bias is seeded to (log prior_w, log prior_h), which
   makes the predicted boxes overlap the ground truth enough for iou^beta to
   be non-negligible. If that init is ever changed, or tal_beta is raised, the
   assigner silently returns zero positives for the ENTIRE run and the loss
   still looks plausible because the classification term keeps decreasing
   toward all-background. _assert_cold_start() makes that failure loud on the
   first batch that has at least one ground-truth box.

3. DEAD top-k LINE REMOVED. The old code computed
       topk_mask *= (topk_val > _EPS).sum(-1, keepdim=True).clamp(max=1.0)
   which is a per-GT scalar meaning "this GT had at least one real
   candidate". The very next line applies (cand > _EPS) per CELL, which is
   strictly stronger and already implies it. The removed line cost a reduction
   over N cells on every training step for nothing.

4. PER-LEVEL POSITIVE COUNTS. forward() now returns n_pos_l0/l1/l2 alongside
   the loss terms. This is the only way to see whether stride 32 is earning
   its keep or is a dead level, which is what the --p5-ablate flag in train.py
   is for.

Decode is bit-for-bit identical to head.py, live_nirdet.decode and
nirdet_pp.c, and the split of the head's regression conv into off_pred and
size_pred does not change it: the packed (B, N, 5) tensor keeps the order
(t_cx, t_cy, t_w, t_h, conf).
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

    Returns (loss, iou.detach()).
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

    Everything is derived from (img_h, img_w, strides): nothing is hardcoded,
    so the 384x640 -> 512x288 canvas change needs no edit here.
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
        self._cached_device: Optional[torch.device] = None

    @property
    def scale_ranges(self) -> Tuple[Tuple[float, float], ...]:
        """
        Derived size bracket per level, kept for diagnostics and for optional
        candidate prefiltering. TAL does the actual routing, so these are
        advisory: level i nominally owns objects with max(w,h) in
        [stride*2, stride*16], with deliberate overlap.
        """
        out = []
        for i, s in enumerate(self.strides):
            lo = 0.0 if i == 0 else float(s) * 2.0
            hi = 1e9 if i == len(self.strides) - 1 else float(s) * 16.0
            out.append((lo, hi))
        return tuple(out)

    def to(self, device: torch.device) -> "AnchorGeometry":
        if self._cached_device != device:
            self._cols = self._cols.to(device)
            self._rows = self._rows.to(device)
            self._strides_flat = self._strides_flat.to(device)
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

        Bit-for-bit identical to head.py's inference decode, live_nirdet.decode
        and nirdet_pp.c.
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
        """(N, 2) cell-centre coordinates in pixels."""
        return torch.stack([(self._cols + 0.5) * self._strides_flat,
                            (self._rows + 0.5) * self._strides_flat], dim=-1)


# ---------------------------------------------------------------------------
# Task-Aligned Assigner
# ---------------------------------------------------------------------------

class TaskAlignedAssigner(nn.Module):
    """
    TOOD / YOLOv8-style dynamic assignment, single class.

    For each GT:
        t = cls_score^alpha * iou^beta over cells whose centre is inside the box
        keep top-k by t
    Conflicts (one cell claimed by several GTs) -> smallest GT area wins.
    Confidence target = t normalised per GT and rescaled by that GT's best IoU.
    """

    def __init__(self, topk: int = 10, alpha: float = 0.5,
                 beta: float = 6.0) -> None:
        super().__init__()
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @staticmethod
    def _centers_in_boxes(centers: torch.Tensor,
                          gt_xyxy: torch.Tensor) -> torch.Tensor:
        """centers (N,2), gt (B,M,4) -> (B,M,N) bool."""
        cx = centers[:, 0].view(1, 1, -1)
        cy = centers[:, 1].view(1, 1, -1)
        x1 = gt_xyxy[..., 0:1]
        y1 = gt_xyxy[..., 1:2]
        x2 = gt_xyxy[..., 2:3]
        y2 = gt_xyxy[..., 3:4]
        return (cx > x1) & (cx < x2) & (cy > y1) & (cy < y2)

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,   # (B, N)   sigmoid confidences
        pd_xyxy: torch.Tensor,     # (B, N, 4) pixels
        gt_xyxy: torch.Tensor,     # (B, M, 4) pixels
        gt_mask: torch.Tensor,     # (B, M)   1 = real GT, 0 = padding
        centers: torch.Tensor,     # (N, 2)   pixels
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fg_mask       (B, N)     1.0 for positive cells
            target_boxes  (B, N, 4)  xyxy pixels (garbage where fg_mask == 0)
            target_scores (B, N)     soft confidence target in [0, 1]
        """
        b, n = pd_scores.shape
        m = gt_xyxy.shape[1]
        zeros_n = pd_scores.new_zeros(b, n)

        if m == 0 or gt_mask.sum() == 0:
            return zeros_n, pd_scores.new_zeros(b, n, 4), zeros_n

        gtm = gt_mask.unsqueeze(-1).to(pd_scores.dtype)        # (B, M, 1)

        ious = pairwise_iou(gt_xyxy, pd_xyxy).clamp(min=0)     # (B, M, N)
        scores = pd_scores.unsqueeze(1).expand(b, m, n)        # (B, M, N)
        align = scores.clamp(min=_EPS).pow(self.alpha) * \
            ious.clamp(min=_EPS).pow(self.beta)
        align = align * gtm

        in_box = self._centers_in_boxes(centers, gt_xyxy).to(pd_scores.dtype)
        cand = align * in_box * gtm

        k = min(self.topk, n)
        _, topk_idx = cand.topk(k, dim=-1)                     # (B, M, k)
        topk_mask = torch.zeros_like(cand)
        topk_mask.scatter_(-1, topk_idx, 1.0)
        # A zero-valued top-k slot is not a real candidate. This per-CELL test
        # is strictly stronger than the per-GT "had any candidate" reduction
        # that used to sit above it, so that line is gone.
        topk_mask = topk_mask * (cand > _EPS).to(cand.dtype)

        mask_pos = topk_mask * in_box * gtm                    # (B, M, N)

        # ---- smallest-area-wins conflict resolution ----
        gt_area = ((gt_xyxy[..., 2] - gt_xyxy[..., 0]).clamp(min=0) *
                   (gt_xyxy[..., 3] - gt_xyxy[..., 1]).clamp(min=0))   # (B, M)
        fg_count = mask_pos.sum(dim=1)                          # (B, N)
        # Always resolve collisions — never gate this on
        # bool((fg_count > 1).any()), which forces a GPU->CPU sync on every
        # training step. With no collisions `multi` is all false and
        # torch.where returns mask_pos unchanged.
        areas = gt_area.unsqueeze(-1).expand(b, m, n).clone()
        areas = areas.masked_fill(mask_pos <= 0, float("inf"))
        best_gt = areas.argmin(dim=1)                       # (B, N)
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
        norm = align_pos * max_iou / (max_align + _EPS)
        target_scores = norm.amax(dim=1) * fg_mask              # (B, N)

        return fg_mask, target_boxes, target_scores.clamp(0.0, 1.0)


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
        total, cls, reg, iou, n_pos, n_pos_l0, n_pos_l1, ... (one per level)

    predictions : list of (B, H_i*W_i, 5) RAW logits, level order == strides.
                  Column order is (t_cx, t_cy, t_w, t_h, conf_logit); the head
                  produces this by concatenating its separate off and size
                  convolutions, so the split for INT8 is invisible here.
    gt_batch    : list of length B; each entry is either
                    a (N_i, 4) tensor of normalised (cx, cy, w, h),
                    a (N_i, 5) tensor of (cls, cx, cy, w, h), or
                    a list of (4,) / (5,) tensors, or [] / None.
                  Single class, so the cls column is read and discarded.
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
    ) -> None:
        super().__init__()
        self.geom = AnchorGeometry(img_h, img_w, strides)
        self.lambda_cls = float(lambda_cls)
        self.lambda_reg = float(lambda_reg)
        self.ramp_frac = float(ramp_frac)
        self.total_epochs = int(max(total_epochs, 1))
        self._epoch = 0

        self.assigner = TaskAlignedAssigner(topk=tal_topk, alpha=tal_alpha,
                                            beta=tal_beta)
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

    @property
    def scale_ranges(self) -> Tuple[Tuple[float, float], ...]:
        return self.geom.scale_ranges

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

        Why this exists: at epoch 0 the classification scores are all ~0.01
        (the focal prior) and the ONLY thing making iou^tal_beta non-negligible
        is head.size_pred.bias being seeded to (log prior_w, log prior_h) so
        the predicted boxes are already roughly pedestrian-shaped. Break that
        init, or raise tal_beta, and the assigner returns zero positives for
        every step of the run. The regression term then contributes nothing,
        the classification term happily converges to all-background, and the
        loss curve looks entirely normal. This has to be an exception, not a
        warning.
        """
        if not self._assert_cold_start or self._cold_start_checked:
            return
        if self._epoch != 0 or n_gt == 0:
            return
        self._cold_start_checked = True
        if n_pos > 0:
            return
        raise ColdStartError(
            f"Task-Aligned Assignment produced 0 positives on the first "
            f"epoch-0 batch, which had {n_gt} ground-truth box(es).\n"
            f"  alignment metric t = cls^{self.assigner.alpha} * "
            f"iou^{self.assigner.beta}, topk={self.assigner.topk}\n"
            f"At epoch 0 every cls score is ~prior_prob, so the iou term is "
            f"the only thing keeping t above zero, and the iou term is only "
            f"non-negligible because head.size_pred.bias is initialised to "
            f"(log prior_w, log prior_h). Check, in order:\n"
            f"  1. head._init_predictions still writes log(prior_w/h) into "
            f"size_pred[lvl].bias\n"
            f"  2. cfg.model.prior_w / prior_h came from the active dataset "
            f"profile and are canvas-space for {self.geom.img_h}x"
            f"{self.geom.img_w}\n"
            f"  3. loss.tal_beta ({self.assigner.beta}) has not been raised\n"
            f"  4. the GT boxes are normalised (cx, cy, w, h) in [0, 1], not "
            f"pixels\n"
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
        dtype = predictions[0].dtype
        self.geom.to(device)

        flat = torch.cat([p.reshape(p.shape[0], -1, 5) for p in predictions],
                         dim=1).float()                  # (B, N, 5)
        b, n, _ = flat.shape
        if n != self.geom.num_cells:
            raise ValueError(f"prediction has {n} cells, geometry expects "
                             f"{self.geom.num_cells}: input resolution and "
                             f"strides disagree")

        raw_box = flat[..., :4]
        cls_logits = flat[..., 4]

        pd_xyxy = self.geom.decode(raw_box)              # (B, N, 4) px
        pd_scores = torch.sigmoid(cls_logits.detach())

        gt_xyxy, gt_mask = self._pad_gt(gt_batch, b, device)

        fg_mask, tgt_boxes, tgt_scores = self.assigner(
            pd_scores, pd_xyxy.detach(), gt_xyxy, gt_mask,
            self.geom.centers_px(),
        )

        # Hard -> soft ramp: at epoch 0 the IoUs are low, so pure alignment
        # targets would keep confidence pinned near zero.
        lam = self._soft_ramp()
        tgt_scores = (1.0 - lam) * fg_mask + lam * tgt_scores

        # ---- single normalisation constant for BOTH terms ----
        norm = tgt_scores.sum().clamp(min=1.0)

        cls_loss = self.qfl(cls_logits, tgt_scores).sum() / norm

        pos = fg_mask > 0
        n_pos = int(pos.sum())

        # ---- per-level positive counts ----
        # The only way to tell whether stride 32 is earning its keep. If
        # n_pos_l2 stays near zero across training, P5 is a dead level and
        # --p5-ablate should confirm it costs nothing to remove.
        per_level: Dict[str, torch.Tensor] = {}
        start = 0
        for lvl, size in enumerate(self.geom.level_sizes):
            cnt = fg_mask[:, start:start + size].sum()
            per_level[f"n_pos_l{lvl}"] = cnt.detach()
            start += size

        self._check_cold_start(n_pos, int(gt_mask.sum().item()))

        if n_pos > 0:
            reg_raw, iou_val = ciou(pd_xyxy[pos], tgt_boxes[pos])
            w = tgt_scores[pos]
            reg_loss = (reg_raw * w).sum() / norm
            mean_iou = iou_val.mean()
        else:
            reg_loss = flat.sum() * 0.0
            mean_iou = flat.sum() * 0.0

        total = self.lambda_cls * cls_loss + self.lambda_reg * reg_loss

        out = {
            "total": total.to(dtype) if dtype.is_floating_point else total,
            "cls": cls_loss.detach(),
            "reg": reg_loss.detach(),
            "iou": mean_iou.detach(),
            "n_pos": torch.tensor(float(n_pos), device=device),
        }
        out.update(per_level)
        return out


# ---------------------------------------------------------------------------
# self-tests:  python losses.py
# ---------------------------------------------------------------------------

def _dummy(b: int, geom: AnchorGeometry, seed: int = 0) -> List[torch.Tensor]:
    torch.manual_seed(seed)
    return [torch.randn(b, h * w, 5) for h, w in geom.grid_sizes]


def _t_empty() -> None:
    print("T1 empty GT")
    lf = NIRDetLoss()
    out = lf(_dummy(2, lf.geom, 0), [[], []])
    assert torch.isfinite(out["total"]), "non-finite total"
    assert float(out["reg"]) == 0.0, f"reg should be 0, got {float(out['reg'])}"
    # cold start must NOT fire: there is no ground truth to assign
    print(f"   total={float(out['total']):.6f} cls={float(out['cls']):.6f} PASS")


def _t_direction() -> None:
    print("T2 loss direction")
    lf = NIRDetLoss()
    lf.set_epoch(lf.total_epochs)                     # fully soft targets
    gt = torch.tensor([[0.35, 0.5, 0.05, 0.18]])
    gtb = [[gt[0]]]
    rand = _dummy(1, lf.geom, 7)
    out_rand = lf(rand, gtb)

    good = [p.clone() for p in rand]
    for lvl, (h, w) in enumerate(lf.geom.grid_sizes):
        col = min(int(gt[0, 0] * w), w - 1)
        row = min(int(gt[0, 1] * h), h - 1)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if not (0 <= r < h and 0 <= c < w):
                    continue
                i = r * w + c
                tx = (gt[0, 0] * w - c + _OFF_B) / _OFF_S
                ty = (gt[0, 1] * h - r + _OFF_B) / _OFF_S
                tx = float(torch.logit(tx.clamp(1e-4, 1 - 1e-4)))
                ty = float(torch.logit(ty.clamp(1e-4, 1 - 1e-4)))
                good[lvl][0, i, 0] = tx
                good[lvl][0, i, 1] = ty
                good[lvl][0, i, 2] = math.log(float(gt[0, 2]))
                good[lvl][0, i, 3] = math.log(float(gt[0, 3]))
                good[lvl][0, i, 4] = 6.0
    out_good = lf(good, gtb)
    print(f"   random={float(out_rand['total']):.6f} "
          f"matched={float(out_good['total']):.6f} "
          f"iou={float(out_good['iou']):.3f} n_pos={int(out_good['n_pos'])}")
    assert float(out_good["total"]) < float(out_rand["total"]), "no decrease"
    print("   PASS")


def _t_nan() -> None:
    print("T3 NaN/Inf guard, 25 random batches")
    lf = NIRDetLoss(assert_cold_start=False)   # random logits, no prior init
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
            assert torch.isfinite(v).all(), f"trial {t}: {k} non-finite"
    print("   PASS")


def _t_collision() -> None:
    print("T4 collision -> smallest area wins")
    lf = NIRDetLoss()
    lf.set_epoch(lf.total_epochs)
    big = torch.tensor([0.5, 0.5, 0.40, 0.80])
    small = torch.tensor([0.5, 0.5, 0.05, 0.16])
    preds = _dummy(1, lf.geom, 3)
    W, H = float(lf.geom.img_w), float(lf.geom.img_h)
    _, boxes, _ = lf.assigner(
        torch.sigmoid(torch.cat([p[..., 4] for p in preds], 1)),
        lf.geom.to(torch.device("cpu")).decode(
            torch.cat([p[..., :4] for p in preds], 1)),
        cxcywh_to_xyxy(torch.stack([big, small]).unsqueeze(0) *
                       torch.tensor([W, H, W, H])),
        torch.ones(1, 2),
        lf.geom.centers_px(),
    )
    areas = ((boxes[..., 2] - boxes[..., 0]) *
             (boxes[..., 3] - boxes[..., 1]))
    small_area = 0.05 * W * 0.16 * H
    centre_cell_area = float(areas[0, areas[0] > 0].min())
    assert abs(centre_cell_area - small_area) < 1.0, \
        f"expected smallest-area GT ({small_area:.0f}), got {centre_cell_area:.0f}"
    print(f"   contested cells resolved to area {centre_cell_area:.0f} PASS")


def _t_resolution() -> None:
    print("T5 geometry derived from resolution")
    for (h, w) in ((288, 512), (384, 640), (256, 416), (288, 1024)):
        g = AnchorGeometry(h, w, (8, 16, 32))
        expect = ((h // 8) * (w // 8) + (h // 16) * (w // 16)
                  + (h // 32) * (w // 32))
        assert g.num_cells == expect
        print(f"   {h}x{w}: grids {g.grid_sizes} cells {g.num_cells}")
    g = AnchorGeometry(288, 512, (8, 16, 32))
    assert g.num_cells == 3024, g.num_cells
    print("   512x288 -> 3024 cells (was 5040 at 640x384)  PASS")


def _t_per_level() -> None:
    print("T6 per-level positive counts")
    lf = NIRDetLoss()
    lf.set_epoch(lf.total_epochs)
    gt = [torch.tensor([[0.5, 0.5, 0.06, 0.20],
                        [0.2, 0.7, 0.30, 0.60]])]
    out = lf(_dummy(1, lf.geom, 11), gt)
    keys = [k for k in out if k.startswith("n_pos_l")]
    assert len(keys) == len(lf.geom.strides), keys
    tot = sum(float(out[k]) for k in sorted(keys))
    print("   " + "  ".join(f"{k}={float(out[k]):.0f}" for k in sorted(keys)))
    assert abs(tot - float(out["n_pos"])) < 1e-6, \
        f"per-level sum {tot} != n_pos {float(out['n_pos'])}"
    print("   per-level counts sum to n_pos  PASS")


def _t_cold_start() -> None:
    print("T7 cold-start assertion")
    # Prior-initialised size logits: must NOT raise.
    lf = NIRDetLoss()
    lf.set_epoch(0)
    preds = [torch.zeros(1, h * w, 5) for h, w in lf.geom.grid_sizes]
    for p in preds:
        p[..., 2] = math.log(0.046094)
        p[..., 3] = math.log(0.179167)
        p[..., 4] = -4.595                     # logit(0.01)
    out = lf(preds, [torch.tensor([[0.5, 0.5, 0.046094, 0.179167]])])
    assert int(out["n_pos"]) > 0, "prior-init produced no positives"
    print(f"   prior-init: n_pos={int(out['n_pos'])} (no raise)  PASS")

    # Broken size init (boxes 1 px wide): must raise.
    lf2 = NIRDetLoss()
    lf2.set_epoch(0)
    bad = [torch.zeros(1, h * w, 5) for h, w in lf2.geom.grid_sizes]
    for p in bad:
        p[..., 2] = -6.0
        p[..., 3] = -6.0
        p[..., 4] = -4.595
    try:
        lf2(bad, [torch.tensor([[0.5, 0.5, 0.046094, 0.179167]])])
        print("   FAIL: broken size init did not raise")
        raise AssertionError("cold-start assertion did not fire")
    except ColdStartError as e:
        assert "size_pred.bias" in str(e)
        print("   broken size init raised ColdStartError  PASS")


def _t_tal_alpha() -> None:
    print("T8 tal_alpha default")
    lf = NIRDetLoss()
    assert lf.assigner.alpha == 0.5, lf.assigner.alpha
    assert lf.assigner.beta == 6.0, lf.assigner.beta
    print(f"   alpha={lf.assigner.alpha} beta={lf.assigner.beta} "
          f"(YOLOv8 setting)  PASS")


if __name__ == "__main__":
    print("=" * 62)
    print("  losses.py self-tests")
    print("=" * 62)
    _t_empty()
    _t_direction()
    _t_nan()
    _t_collision()
    _t_resolution()
    _t_per_level()
    _t_cold_start()
    _t_tal_alpha()
    print("=" * 62)
    print("  all tests PASSED")
    print("=" * 62)
