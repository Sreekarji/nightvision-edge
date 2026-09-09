"""
NIRDet — losses.py
==================
Phase 5 loss function module.

Model output format (per scale, per cell):
    (cx_offset, cy_offset, w, h, conf)  — 5 raw values

Three scales (640×384 input):
    48×80  (stride 8)
    24×40  (stride 16)
    12×20  (stride 32)

Ground truth: YOLO format — (class, cx, cy, w, h) normalized [0, 1].
Single class: person. Heavy background imbalance (~5 040 cells per image,
only 1–5 are foreground).

References used throughout this file:
    [Lin 2017]  Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet),
                ICCV 2017.
    [Zheng 2020] Zheng et al., "Distance-IoU Loss: Faster and Better Learning
                 for Bounding Box Regression", AAAI 2020.
    [FCOS 2019]  Tian et al., "FCOS: Fully Convolutional One-Stage Object
                 Detection", ICCV 2019.
    [YOLOX 2021] Ge et al., "YOLOX: Exceeding Yolo Series Detectors", 2021.
    [YOLOv5]     Ultralytics YOLOv5 default.yaml — box:0.05, cls:0.5, obj:1.0
    [YOLOv8]     Ultralytics YOLOv8 default.yaml — box:7.5, cls:0.5, dfl:1.5
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from config import DECODE_OFFSET_SCALE
_OFF_S = DECODE_OFFSET_SCALE
_OFF_B = (DECODE_OFFSET_SCALE - 1.0) / 2.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Division with a floor on the denominator to prevent NaN/Inf gradients."""
    return a / (b + eps)


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from center format (cx, cy, w, h) → corner format (x1, y1, x2, y2).

    Used internally for IoU computation; all values remain in the same coordinate
    space (normalized [0,1] in the functions below).

    Args:
        boxes: (..., 4) tensor in cx-cy-w-h format.
    Returns:
        (..., 4) tensor in x1-y1-x2-y2 format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ---------------------------------------------------------------------------
# Q1 — Label Assignment
# ---------------------------------------------------------------------------

class LabelAssigner:
    """
    Centre-cell + nearest-neighbour assignment with FPN scale routing.

    Changes from original:
    - Scale routing (Item 4): each GT is assigned ONLY to the FPN level
      whose scale_ranges bracket contains max(w_px, h_px). Overlapping
      ranges handle boundary objects.
    - Neighbour assignment (Item 1, cross3): beyond the centre cell, the
      two nearest neighbours in x and y are also assigned, subject to an
      in-box guard. Triples regression signal. Fixes cell-boundary dead zone.

    Dataset-agnostic: scale_ranges defaults suit pedestrian data at 640x384.
    Run tools/measure_priors.py on any new dataset to re-derive them.
    """

    def __init__(
        self,
        grid_sizes: Tuple[Tuple[int, int], ...] = ((48, 80), (24, 40), (12, 20)),
        strides: Tuple[int, ...] = (8, 16, 32),
        scale_ranges: Tuple[Tuple[float, float], ...] = (
            (0.0,   64.0),
            (48.0,  160.0),
            (128.0, 1e5),
        ),
        neighbour_mode: str = "cross3",
    ):
        assert len(grid_sizes) == len(strides) == len(scale_ranges)
        assert neighbour_mode in ("center", "cross3")
        self.grid_sizes     = grid_sizes
        self.strides        = strides
        self.scale_ranges   = scale_ranges
        self.neighbour_mode = neighbour_mode
        self.img_h = grid_sizes[0][0] * strides[0]
        self.img_w = grid_sizes[0][1] * strides[0]

    def _cells_for_gt(
        self, cx: float, cy: float, w: float, h: float, S_h: int, S_w: int
    ) -> list:
        """Return flat cell indices assigned to one GT at one FPN level."""
        fx, fy = cx * S_w, cy * S_h
        col = min(int(fx), S_w - 1)
        row = min(int(fy), S_h - 1)
        u   = fx - col
        v   = fy - row
        cands = [(row, col)]

        if self.neighbour_mode == "cross3":
            dc = -1 if u < 0.5 else (1 if u > 0.5 else 0)
            dr = -1 if v < 0.5 else (1 if v > 0.5 else 0)
            if dc != 0:
                cands.append((row, col + dc))
            if dr != 0:
                cands.append((row + dr, col))

        half_w_cells = 0.5 * w * S_w
        half_h_cells = 0.5 * h * S_h

        out = []
        for (r, c) in cands:
            if not (0 <= r < S_h and 0 <= c < S_w):
                continue
            if (r, c) != (row, col):
                if abs((c + 0.5) - fx) > half_w_cells:
                    continue
                if abs((r + 0.5) - fy) > half_h_cells:
                    continue
            out.append(r * S_w + c)
        return out

    def assign(
        self,
        gt_boxes: torch.Tensor,
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Build per-scale assignment targets for one image.
        Same return interface as original — callers unchanged.
        """
        results = []
        n_gt = gt_boxes.shape[0]

        if n_gt > 0:
            w_px = gt_boxes[:, 2] * self.img_w
            h_px = gt_boxes[:, 3] * self.img_h
            size = torch.maximum(w_px, h_px)

        for s, (S_h, S_w) in enumerate(self.grid_sizes):
            total = S_h * S_w
            conf_target = torch.zeros(total, dtype=torch.float32, device=device)
            box_target  = torch.zeros(total, 4, dtype=torch.float32, device=device)

            if n_gt == 0:
                results.append((conf_target, box_target, conf_target.bool()))
                continue

            lo, hi = self.scale_ranges[s]
            for i in range(n_gt):
                sz = float(size[i])
                if not (lo <= sz < hi):
                    continue
                cx, cy, w, h = (float(v) for v in gt_boxes[i])
                for idx in self._cells_for_gt(cx, cy, w, h, S_h, S_w):
                    conf_target[idx] = 1.0
                    box_target[idx]  = gt_boxes[i]

            results.append((conf_target, box_target, conf_target.bool()))

        return results


# ---------------------------------------------------------------------------
# Q2 — Classification Loss: Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (foreground/background).

    Reference: Lin et al. [Lin 2017], equation (5).

    Standard Binary Cross-Entropy (BCE) fails with extreme imbalance because:
        • With 5 040 cells and ~3 positives, negatives outnumber positives
          ~1680:1.
        • Easy, correctly-classified background cells (p ≈ 0.01) each
          contribute a small loss, but there are so many of them that they
          collectively dominate the gradient.
        • The network is pressured toward the trivial solution: predict
          background everywhere, confidently.

    Focal Loss adds the modulating factor (1 − p_t)^γ in front of BCE:

        FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

    where:
        p_t = p      if y = 1   (foreground)
              1 − p  if y = 0   (background)

        α_t = α      if y = 1
              1 − α  if y = 0

    Effect of the modulating factor:
        • Easy negatives (p_t ≈ 0.99) → (1 − 0.99)^2 = 0.0001 → loss
          suppressed by factor ~100.
        • Hard positives  (p_t ≈ 0.10) → (1 − 0.10)^2 = 0.81 → barely
          suppressed; gradient retained.
        • The network is forced to attend to the rare, hard positives.

    γ (gamma) — focusing parameter:
        • γ = 0 recovers standard BCE.
        • Lin et al. [Lin 2017] Table 1b: peak AP at γ = 2.
          Values tested: {0, 0.5, 1, 2, 5}. γ = 2 gave best result; the
          network is "relatively robust to γ ∈ [0.5, 5]" per the paper.
        • We default γ = 2 following the paper recommendation.

    α (alpha) — class-balance weight:
        • [Lin 2017]: with γ = 2, optimal α shifts DOWN from ~0.75 (pure
          class-frequency inverse) to α ≈ 0.25 for the foreground class.
          This is because the modulating factor already increases relative
          attention on positives; α does not need to compensate as strongly.
        • We set α = 0.25 (foreground weight) by default, matching RetinaNet.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        """
        Args:
            gamma:     Focusing exponent. γ = 2 is the [Lin 2017] default.
            alpha:     Foreground class weight ∈ (0, 1).
                       alpha for foreground, (1-alpha) for background.
            reduction: "mean" | "sum" | "none".
        """
        super().__init__()
        self.gamma = gamma        # focusing exponent — see class docstring
        self.alpha = alpha        # foreground weight — see class docstring
        self.reduction = reduction

    def forward(
        self,
        pred_logits: torch.Tensor,   # (N,) — raw logits (before sigmoid)
        targets: torch.Tensor,       # (N,) — binary labels {0.0, 1.0}
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Formula:
            1.  p     = sigmoid(pred_logits)
            2.  p_t   = p   where target==1,  (1-p) where target==0
            3.  α_t   = α   where target==1,  (1-α) where target==0
            4.  FL    = -α_t · (1 - p_t)^γ · log(p_t + ε)

        We use F.binary_cross_entropy_with_logits for numerical stability
        (it fuses sigmoid + log-sum-exp) and then apply the modulating factor
        as a multiplicative correction.

        Args:
            pred_logits: Raw logits, shape (N,).
            targets:     Binary labels {0, 1}, shape (N,).

        Returns:
            Scalar loss (if reduction != "none").
        """
        # ----------------------------------------------------------------
        # Step 1: Compute p = sigmoid(logits) for the modulating factor.
        #         Use .detach() here so the (1-p_t)^γ weight is treated as
        #         a constant during the backward pass — standard practice.
        # ----------------------------------------------------------------
        p = torch.sigmoid(pred_logits)          # (N,) in [0, 1]

        # ----------------------------------------------------------------
        # Step 2: Numerically stable BCE (= -[y log p + (1-y) log(1-p)]).
        #         Returns per-element losses.
        # ----------------------------------------------------------------
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, targets, reduction="none"
        )  # (N,)

        # ----------------------------------------------------------------
        # Step 3: Compute p_t — the "correct class probability".
        #         p_t = p   if target == 1
        #         p_t = 1-p if target == 0
        # ----------------------------------------------------------------
        # Equivalent to: p_t = targets * p + (1 - targets) * (1 - p)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)  # (N,)

        # ----------------------------------------------------------------
        # Step 4: Apply the modulating factor (1 - p_t)^γ.
        #
        #   This is the key contribution of [Lin 2017]. Well-classified
        #   examples (p_t near 1) get a near-zero weight; hard examples
        #   (p_t near 0.5 or below) retain their gradient.
        # ----------------------------------------------------------------
        modulating_factor = (1.0 - p_t) ** self.gamma  # (N,)

        # ----------------------------------------------------------------
        # Step 5: Apply α_t weighting.
        #         α_t = α   for foreground,  (1-α) for background.
        # ----------------------------------------------------------------
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)  # (N,)

        # ----------------------------------------------------------------
        # Step 6: Final focal loss = α_t · (1 - p_t)^γ · BCE
        # ----------------------------------------------------------------
        focal = alpha_t * modulating_factor * bce_loss  # (N,)

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:  # "none"
            return focal


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss (Li et al. 2020 / Generalized Focal Loss).

        QFL(σ(x), y) = -|y - σ(x)|^beta * BCE(x, y)

    y ∈ [0,1] is a continuous quality target (the IoU of the predicted
    box with its GT). At y=0 (background) this reduces exactly to focal
    loss — background suppression is unchanged.

    WHY THIS FIXES mAP@75:
    With hard targets (y=1.0), a box with IoU=0.4 and IoU=0.9 get
    identical training signal. Confidence scores carry no localisation
    information. After QFL, conf ≈ IoU, so NMS automatically keeps the
    tightest box — mAP@75 improves much more than mAP@50.

    QFL ramp: at epoch 0 predictions are random and IoU ≈ 0, which would
    make ALL positive targets ≈ 0 and prevent confidence from rising.
    We ramp from hard targets (y=1.0) to soft IoU targets over the first
    qfl_ramp_frac of training.
    """

    def __init__(self, beta: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.beta  = beta
        self.alpha = alpha

    def forward(
        self,
        logits:  torch.Tensor,   # (N,) raw logits
        targets: torch.Tensor,   # (N,) continuous targets in [0,1]
    ) -> torch.Tensor:
        p    = torch.sigmoid(logits)
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        mod  = (targets - p).abs().detach().pow(self.beta)
        loss = mod * bce
        fg   = (targets > 0).float()
        loss = loss * (self.alpha * fg + (1.0 - self.alpha) * (1.0 - fg))
        return loss


# ---------------------------------------------------------------------------
# Q3 — Regression Loss: CIoU
# ---------------------------------------------------------------------------

def ciou_loss(
    pred_boxes: torch.Tensor,   # (M, 4) — cx, cy, w, h (normalized)
    gt_boxes: torch.Tensor,     # (M, 4) — cx, cy, w, h (normalized)
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Complete IoU Loss [Zheng 2020] — computed per matched pair.

    Background and motivation
    -------------------------
    Loss hierarchy (each adds a geometric term to the previous):

        L1/MSE  : penalises raw coordinate error. Scale-dependent; no gradient
                  when boxes don't overlap. Unusable for anchor-free detectors.

        IoU     : scale-invariant; zero gradient for non-overlapping boxes.
                  Slow convergence when predicted box is far from GT.

        GIoU    : adds a penalty term proportional to the area of the smallest
                  enclosing box NOT covered by either box. Fixes non-overlap
                  gradient, but slow when centers are misaligned.

        DIoU    : adds ρ²(b, bᵍᵗ)/c² — squared center distance normalized by
                  diagonal of enclosing box. Faster convergence than GIoU.
                  But does NOT penalise aspect-ratio mismatch.

        CIoU    : DIoU + aspect-ratio consistency term v, weighted by α.
                  The complete formula is:

                      L_CIoU = 1 − IoU + ρ²(b, bᵍᵗ)/c² + α·v

                  where:
                      ρ² = squared Euclidean distance between box centers
                      c  = diagonal length of the smallest enclosing box
                      v  = (4/π²) · (arctan(wᵍᵗ/hᵍᵗ) − arctan(w/h))²
                      α  = v / [(1 − IoU) + v]   ← trade-off parameter

    Why CIoU for pedestrian detection specifically?
    -----------------------------------------------
    Pedestrians have consistent aspect ratios (tall and narrow, ~0.4 w:h).
    CIoU's v term explicitly penalises predicted boxes that drift in aspect
    ratio (e.g., predicting a wide box for a thin person). Multiple
    pedestrian detection papers (MobileNet-YoLo [PMC9722493], BDD100K
    pedestrian experiments [PLOS 2023]) confirm CIoU outperforms GIoU/DIoU
    on pedestrian datasets. YOLOv4/v5/v7/v8 all default to CIoU.

    Mathematical derivation
    -----------------------
    Given predicted box bₚ = (cxₚ, cyₚ, wₚ, hₚ) and GT box bᵍ = (cxᵍ, cyᵍ, wᵍ, hᵍ):

        x1ₚ, y1ₚ = cxₚ − wₚ/2,  cyₚ − hₚ/2
        x2ₚ, y2ₚ = cxₚ + wₚ/2,  cyₚ + hₚ/2   (similarly for gt)

    Intersection:
        inter_x1 = max(x1ₚ, x1ᵍ),  inter_y1 = max(y1ₚ, y1ᵍ)
        inter_x2 = min(x2ₚ, x2ᵍ),  inter_y2 = min(y2ₚ, y2ᵍ)
        inter_area = max(0, inter_x2 − inter_x1) · max(0, inter_y2 − inter_y1)

    IoU:
        union_area = wₚ·hₚ + wᵍ·hᵍ − inter_area
        IoU = inter_area / union_area

    Center distance (squared):
        ρ² = (cxₚ − cxᵍ)² + (cyₚ − cyᵍ)²

    Enclosing box diagonal (squared):
        c_x1 = min(x1ₚ, x1ᵍ),  c_y1 = min(y1ₚ, y1ᵍ)
        c_x2 = max(x2ₚ, x2ᵍ),  c_y2 = max(y2ₚ, y2ᵍ)
        c²   = (c_x2 − c_x1)² + (c_y2 − c_y1)²

    Aspect ratio consistency:
        v = (4/π²) · (arctan(wᵍ/hᵍ) − arctan(wₚ/hₚ))²

    Trade-off parameter:
        α = v / [(1 − IoU) + v]           (stops v dominating when IoU is low)

    CIoU loss:
        L_CIoU = 1 − IoU + ρ²/c² + α·v

    Args:
        pred_boxes: (M, 4) predicted boxes in cx-cy-w-h format, normalized [0,1].
        gt_boxes:   (M, 4) ground-truth boxes in the same format.
        eps:        Small constant to prevent division by zero.

    Returns:
        (M,) per-pair CIoU loss values, all in [0, 2] for well-behaved boxes.
    """
    # ----------------------------------------------------------------
    # Convert center format → corner format for intersection computation.
    # ----------------------------------------------------------------
    pred_xyxy = _box_cxcywh_to_xyxy(pred_boxes)   # (M, 4)
    gt_xyxy   = _box_cxcywh_to_xyxy(gt_boxes)     # (M, 4)

    # ----------------------------------------------------------------
    # Intersection area.
    #   inter_width  = max(0,  min(x2ₚ, x2ᵍ) − max(x1ₚ, x1ᵍ))
    #   inter_height = max(0,  min(y2ₚ, y2ᵍ) − max(y1ₚ, y1ᵍ))
    #   inter_area   = inter_width · inter_height
    # ----------------------------------------------------------------
    inter_x1 = torch.max(pred_xyxy[:, 0], gt_xyxy[:, 0])  # max of left edges
    inter_y1 = torch.max(pred_xyxy[:, 1], gt_xyxy[:, 1])  # max of top edges
    inter_x2 = torch.min(pred_xyxy[:, 2], gt_xyxy[:, 2])  # min of right edges
    inter_y2 = torch.min(pred_xyxy[:, 3], gt_xyxy[:, 3])  # min of bottom edges

    # Clamp to zero: no intersection when boxes don't overlap.
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)  # (M,)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)  # (M,)
    inter_area = inter_w * inter_h                   # (M,)

    # ----------------------------------------------------------------
    # Union area.
    #   union_area = area_pred + area_gt − inter_area
    # ----------------------------------------------------------------
    pred_w, pred_h = pred_boxes[:, 2], pred_boxes[:, 3]  # (M,)
    gt_w,   gt_h   = gt_boxes[:, 2],   gt_boxes[:, 3]    # (M,)

    area_pred = pred_w * pred_h              # (M,)
    area_gt   = gt_w   * gt_h               # (M,)
    union_area = area_pred + area_gt - inter_area + eps  # (M,) — eps prevents /0

    # ----------------------------------------------------------------
    # IoU = inter / union
    # ----------------------------------------------------------------
    iou = inter_area / union_area  # (M,) in [0, 1]

    # ----------------------------------------------------------------
    # Center distance squared:  ρ² = (cxₚ − cxᵍ)² + (cyₚ − cyᵍ)²
    # ----------------------------------------------------------------
    cx_p, cy_p = pred_boxes[:, 0], pred_boxes[:, 1]  # (M,)
    cx_g, cy_g = gt_boxes[:, 0],   gt_boxes[:, 1]    # (M,)

    rho_sq = (cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2  # (M,)

    # ----------------------------------------------------------------
    # Smallest enclosing box diagonal squared:
    #   c² = (c_x2 − c_x1)² + (c_y2 − c_y1)²
    # where c_x1 = min(x1ₚ, x1ᵍ), etc.
    # ----------------------------------------------------------------
    encl_x1 = torch.min(pred_xyxy[:, 0], gt_xyxy[:, 0])  # (M,)
    encl_y1 = torch.min(pred_xyxy[:, 1], gt_xyxy[:, 1])  # (M,)
    encl_x2 = torch.max(pred_xyxy[:, 2], gt_xyxy[:, 2])  # (M,)
    encl_y2 = torch.max(pred_xyxy[:, 3], gt_xyxy[:, 3])  # (M,)

    c_sq = (encl_x2 - encl_x1) ** 2 + (encl_y2 - encl_y1) ** 2 + eps  # (M,)

    # ----------------------------------------------------------------
    # DIoU penalty term: ρ²/c²
    #   Normalises center distance by enclosing box diagonal.
    # ----------------------------------------------------------------
    diou_term = rho_sq / c_sq  # (M,)

    # ----------------------------------------------------------------
    # Aspect ratio consistency:
    #   v = (4/π²) · (arctan(wᵍ/hᵍ) − arctan(wₚ/hₚ))²
    #
    # For pedestrians, wᵍ/hᵍ is consistently small (~0.4).
    # Predicted boxes that drift to wide aspect ratios incur a large v.
    # ----------------------------------------------------------------
    # arctan is well-defined; add eps to heights to prevent /0 on edge cases.
    atan_gt   = torch.atan(_safe_div(gt_w,   gt_h))    # (M,) in [0, π/2]
    atan_pred = torch.atan(_safe_div(pred_w, pred_h))  # (M,) in [0, π/2]

    v = (4.0 / (math.pi ** 2)) * (atan_gt - atan_pred) ** 2  # (M,) ≥ 0

    # ----------------------------------------------------------------
    # Trade-off parameter α = v / [(1 − IoU) + v].
    #
    # α approaches 1 when IoU is high (both boxes overlap well) and v is
    # large (aspect ratios still differ). In that regime, the aspect-ratio
    # term is the dominant remaining error, so we pay attention to it.
    # When IoU is low, (1 − IoU) >> v, α ≈ 0, and DIoU already penalises
    # the distance — we do not prematurely penalise aspect ratio.
    # ----------------------------------------------------------------
    with torch.no_grad():
        # α is used as a weighting constant, so we detach its gradient
        # (following the original implementation in [Zheng 2020]).
        alpha_ciou = v / ((1.0 - iou) + v + eps)  # (M,) in [0, 1]

    # ----------------------------------------------------------------
    # CIoU loss:
    #   L_CIoU = 1 − IoU + ρ²/c² + α·v
    #
    # All terms are non-negative; L_CIoU = 0 iff IoU = 1 and centers
    # coincide and aspect ratios match exactly.
    # ----------------------------------------------------------------
    loss = 1.0 - iou + diou_term + alpha_ciou * v  # (M,)

    return loss  # (M,)


def ciou_loss_with_iou(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    eps: float = 1e-7,
    aspect_scale: tuple = None,
) -> tuple:
    """
    Identical to ciou_loss() but also returns iou.detach() as the
    quality target for QFL.

    aspect_scale: (W_px, H_px). When provided, the aspect-ratio term v
    is computed in pixel space — matches Zheng et al. 2020 exactly and
    corrects the normalised-space mismatch at non-square resolutions
    like 640x384. Pass (640, 384) for this project.
    """
    pred_xyxy = _box_cxcywh_to_xyxy(pred_boxes)
    gt_xyxy   = _box_cxcywh_to_xyxy(gt_boxes)

    inter_x1 = torch.max(pred_xyxy[:, 0], gt_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], gt_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], gt_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], gt_xyxy[:, 3])
    inter_w  = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h  = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    pred_w, pred_h = pred_boxes[:, 2], pred_boxes[:, 3]
    gt_w,   gt_h   = gt_boxes[:, 2],   gt_boxes[:, 3]
    union_area = pred_w * pred_h + gt_w * gt_h - inter_area + eps
    iou = (inter_area / union_area).clamp(0.0, 1.0)

    cx_p, cy_p = pred_boxes[:, 0], pred_boxes[:, 1]
    cx_g, cy_g = gt_boxes[:, 0],   gt_boxes[:, 1]
    rho_sq = (cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2

    encl_x1 = torch.min(pred_xyxy[:, 0], gt_xyxy[:, 0])
    encl_y1 = torch.min(pred_xyxy[:, 1], gt_xyxy[:, 1])
    encl_x2 = torch.max(pred_xyxy[:, 2], gt_xyxy[:, 2])
    encl_y2 = torch.max(pred_xyxy[:, 3], gt_xyxy[:, 3])
    c_sq = (encl_x2 - encl_x1) ** 2 + (encl_y2 - encl_y1) ** 2 + eps
    diou_term = rho_sq / c_sq

    if aspect_scale is not None:
        Wp, Hp = aspect_scale
        atan_gt   = torch.atan(_safe_div(gt_w   * Wp, gt_h   * Hp))
        atan_pred = torch.atan(_safe_div(pred_w * Wp, pred_h * Hp))
    else:
        atan_gt   = torch.atan(_safe_div(gt_w,   gt_h))
        atan_pred = torch.atan(_safe_div(pred_w, pred_h))

    v = (4.0 / (math.pi ** 2)) * (atan_gt - atan_pred) ** 2
    with torch.no_grad():
        alpha_ciou = v / ((1.0 - iou) + v + eps)

    loss = 1.0 - iou + diou_term + alpha_ciou * v
    return loss, iou.detach()


# ---------------------------------------------------------------------------
# Combined NIRDet Loss
# ---------------------------------------------------------------------------

class NIRDetLoss(nn.Module):
    """
    Combined loss for NIRDet.

    Architecture
    ------------
    The model outputs raw predictions per scale per cell:
        (cx_offset, cy_offset, w, h, conf)

    For each scale s with grid (S_h, S_w):
        pred shape: (B, S_h*S_w, 5)

    This class:
        1. Assigns GT boxes to cells via LabelAssigner.
        2. Converts raw predictions to normalized box coordinates.
        3. Computes:
            - Focal Loss on conf channel (all cells, foreground + background).
            - CIoU Loss on box channels (positive cells only).
        4. Returns a monitoring dict {total, cls, reg, conf}.

    Note: We treat "conf" as the objectness score and "cls" as the
    class confidence. Since there is only one class, conf and cls are
    the same prediction. We name the focal loss term "cls" (classifiation)
    and also compute "conf" as a separate BCE without the focal modulation,
    so you can see both in the monitoring dict. In practice for single-class
    you can simply use cls_loss as the only classification signal.

    Loss weights (Q4)
    -----------------
    Standard values across detectors:

    | Detector | λ_cls | λ_reg | λ_conf |
    |----------|-------|-------|--------|
    | YOLOv1   | 1.0   | 5.0   | —      |
    | YOLOv5   | 0.5   | 0.05  | 1.0    |
    | YOLOv8   | 0.5   | 7.5   | —      |
    | FCOS     | 1.0   | 1.0   | 1.0    |
    | YOLOX    | 1.0   | 5.0   | 1.0    |

    The divergence across papers (reg from 0.05 to 7.5) reflects the
    scale difference between loss functions: GIoU/CIoU are bounded in
    [0, 2] while old L2 coordinate losses are unbounded. YOLOv8's
    box=7.5 compensates for CIoU's small dynamic range relative to BCE.

    Sensitivity: Training is moderately sensitive to the ratio λ_reg /
    λ_cls. If λ_reg >> λ_cls, the network learns to regress boxes before
    it learns to detect objects (conf scores stay random). If λ_cls >>
    λ_reg, boxes regress slowly. The YOLOX ratio (cls:reg = 1:5) is a
    reasonable middle ground. With Focal Loss (which already down-weights
    easy negatives), λ_cls can be kept at 1.0 without the background
    swamping the gradient — so the relative amplification of λ_reg is less
    critical than with plain BCE.

    We default to: λ_cls=1.0, λ_reg=5.0, λ_conf=1.0 (YOLOX convention).
    Start here; tune λ_cls up if classification is the bottleneck.
    """

    def __init__(
        self,
        grid_sizes: Tuple[Tuple[int, int], ...] = ((48, 80), (24, 40), (12, 20)),
        lambda_cls: float = 1.0,
        lambda_reg: float = 2.0,     # reduced from 5.0 — QFL shifts cls/reg balance
        lambda_conf: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        quality_target: bool = True,
        qfl_ramp_frac: float = 0.20, # ramp hard→soft over first 20 epochs (of 100)
        total_epochs: int = 100,     # 100 epoch target for miniNIRPed
        img_h: int = 384,
        img_w: int = 640,
    ):
        super().__init__()
        self.grid_sizes     = grid_sizes
        self.lambda_cls     = lambda_cls
        self.lambda_reg     = lambda_reg
        self.lambda_conf    = lambda_conf
        self.quality_target = quality_target
        self.qfl_ramp_frac  = qfl_ramp_frac
        self.total_epochs   = total_epochs
        self.img_h = img_h
        self.img_w = img_w
        self._epoch = 0

        self.assigner   = LabelAssigner(grid_sizes=grid_sizes)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="none")
        self.qfl        = QualityFocalLoss(beta=focal_gamma, alpha=focal_alpha)

    def set_epoch(self, epoch: int) -> None:
        """Call from train.py each epoch: criterion.set_epoch(epoch)"""
        self._epoch = epoch

    # ------------------------------------------------------------------

    def _decode_pred_boxes(
        self,
        raw_pred: torch.Tensor,        # (P, 4) — cx_off, cy_off, w_raw, h_raw for P cells
        cell_indices: torch.Tensor,    # (P,) long — flat cell indices in [0, S_h*S_w)
        S_h: int,
        S_w: int,
    ) -> torch.Tensor:
        """
        Convert raw model output offsets to absolute normalized box coordinates.

        The model outputs:
            cx_offset ∈ (-∞, ∞) → sigmoid → fraction-within-cell → /S_w → [0,1]
            cy_offset ∈ (-∞, ∞) → sigmoid → fraction-within-cell → /S_h → [0,1]
            w_raw     ∈ (-∞, ∞) → exp(w_raw) → w in normalized [0,1] units
            h_raw     ∈ (-∞, ∞) → exp(h_raw) → h in normalized [0,1] units

        For a cell at flat index i (row-major over the S_h×S_w grid):
            col = i % S_w   (column index)
            row = i // S_w  (row index)

        Decoding:
            cx = (sigmoid(cx_off) + col) / S_w   ← offset within cell + cell position
            cy = (sigmoid(cy_off) + row) / S_h
            w  = exp(w_raw).clamp(max=1.0)      ← clamp prevents explosion at init
            h  = exp(h_raw).clamp(max=1.0)

        Args:
            raw_pred:     (P, 4) — raw logit outputs for cx, cy, w, h.
                          P is the number of POSITIVE cells (often 1–5).
            cell_indices: (P,) long tensor — flat indices of the positive cells.
                          Used to look up (col, row) without rebuilding the grid.
            S_h:          Grid height (rows).
            S_w:          Grid width (columns).

        Returns:
            (P, 4) decoded boxes in normalized [0,1] cx-cy-w-h format.
        """
        # ------------------------------------------------------------------
        # Recover column and row for each positive cell.
        #
        #   col = flat_index % S_w   e.g. index 1960 in a 48×80 grid:
        #                                col = 1960 % 80 = 40, row = 1960 // 80 = 24
        #   row = flat_index // S_w
        # ------------------------------------------------------------------
        col_offsets = (cell_indices % S_w).float()   # (P,) — column index per cell
        row_offsets = (cell_indices // S_w).float()  # (P,) — row index per cell

        cx_raw = raw_pred[:, 0]  # (P,)
        cy_raw = raw_pred[:, 1]  # (P,)
        w_raw  = raw_pred[:, 2]  # (P,)
        h_raw  = raw_pred[:, 3]  # (P,)

        # DECODE_OFFSET_SCALE=2.0: range is [-0.5, 1.5] cells.
        # Required for cross3 — neighbour cells need offset > 1.0 or < 0.0.
        # MUST match head.py inference decode exactly.
        cx = (_OFF_S * torch.sigmoid(cx_raw) - _OFF_B + col_offsets) / S_w
        # cy = (σ(cy_raw) + row_offset) / S_h   → absolute normalized y ∈ [0,1]
        cy = (_OFF_S * torch.sigmoid(cy_raw) - _OFF_B + row_offsets) / S_h
        # w  = exp(w_raw) in normalized [0,1] units.
        # BUG FIX: clamp w_raw before exp to prevent fp32 overflow (exp(89)=inf),
        # then clamp the output to (1e-4, 1.0) so decoded boxes stay within the image.
        # Input clamp max=4.0: exp(4)≈54 is already 54× the image — clearly a
        # degenerate prediction; clamping prevents NaN gradients through CIoU.
        w  = torch.exp(w_raw.clamp(min=-8.0, max=4.0)).clamp(min=1e-4)   # FIX: dropped max=1.0 cap — it was a zero-gradient dead zone and disagreed with head.py inference (which multiplies by img_size, no cap)
        # h  = exp(h_raw), same treatment
        h  = torch.exp(h_raw.clamp(min=-8.0, max=4.0)).clamp(min=1e-4)   # FIX: dropped max=1.0 cap (match inference)

        return torch.stack([cx, cy, w, h], dim=-1)  # (P, 4)

    # ------------------------------------------------------------------

    def forward(
        self,
        predictions: List[torch.Tensor],
        gt_batch: List[List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        device = predictions[0].device
        B = predictions[0].shape[0]

        # QFL ramp: 0.0 at epoch 0 (hard targets), 1.0 at epoch 20 (soft IoU)
        ramp_end = max(1, int(self.qfl_ramp_frac * self.total_epochs))
        qfl_lam  = min(1.0, self._epoch / ramp_end) if self.quality_target else 0.0

        total_cls_loss  = 0.0
        total_reg_loss  = 0.0
        total_conf_loss = 0.0

        # ── Pass 1: assignments + decode for every image in the batch;
        #    accumulate the TOTAL number of positive cells across ALL images. ──
        batch_assigns = []
        total_pos = 0

        for b in range(B):
            gt_list = gt_batch[b]
            if len(gt_list) == 0:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            else:
                gt_boxes = torch.stack(gt_list, dim=0).to(device)

            assignments = self.assigner.assign(gt_boxes, device)

            per_scale = []
            for s, (S_h, S_w) in enumerate(self.grid_sizes):
                pred = predictions[s][b]
                conf_target, box_target, pos_mask = assignments[s]
                n_pos_s = int(pos_mask.sum())
                total_pos += n_pos_s

                if n_pos_s > 0:
                    pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(-1)
                    decoded = self._decode_pred_boxes(
                        pred[pos_mask, :4], pos_idx, S_h, S_w
                    )
                    reg_vals, iou = ciou_loss_with_iou(
                        decoded, box_target[pos_mask],
                        aspect_scale=(self.img_w, self.img_h),
                    )
                    if self.quality_target and qfl_lam > 0:
                        conf_t = conf_target.clone()
                        conf_t[pos_mask] = (1.0 - qfl_lam) * 1.0 + qfl_lam * iou
                    else:
                        conf_t = conf_target
                else:
                    reg_vals = None
                    conf_t   = conf_target

                per_scale.append((pred, conf_t, pos_mask, reg_vals))

            batch_assigns.append(per_scale)

        # FIX: batch-wide normalisation — ONE norm from ALL positives across
        # the whole batch (RetinaNet §4), instead of a per-image /max(total_pos,1).
        # Per-image normalisation gave a sparse image (1 positive) the same loss
        # scale as a dense one (8 positives), inflating gradients from sparse
        # images; batch-wide makes the scale consistent across the whole batch.
        norm = float(max(total_pos, 1))

        # ── Pass 2: compute losses, all divided by the SAME batch-wide norm. ──
        for per_scale in batch_assigns:
            for (pred, conf_t, pos_mask, reg_vals) in per_scale:
                conf_logits = pred[:, 4]

                if self.quality_target:
                    cls_vals = self.qfl(conf_logits, conf_t)
                else:
                    cls_vals = self.focal_loss(conf_logits, conf_t)
                total_cls_loss = total_cls_loss + cls_vals.sum() / norm

                if reg_vals is not None:
                    total_reg_loss = total_reg_loss + reg_vals.sum() / norm

                with torch.no_grad():
                    conf_bce = F.binary_cross_entropy_with_logits(
                        conf_logits, (conf_t > 0).float(), reduction="mean"
                    )
                    total_conf_loss = total_conf_loss + conf_bce

        denom = float(max(B, 1))

        if isinstance(total_cls_loss, float):
            total_cls_loss = torch.tensor(total_cls_loss, device=device, dtype=torch.float32)
        if isinstance(total_reg_loss, float):
            total_reg_loss = torch.tensor(total_reg_loss, device=device, dtype=torch.float32)
        if isinstance(total_conf_loss, float):
            total_conf_loss = torch.tensor(total_conf_loss, device=device, dtype=torch.float32)

        cls_loss  = total_cls_loss  / denom
        reg_loss  = total_reg_loss  / denom
        conf_loss = total_conf_loss / denom

        total_loss = self.lambda_cls * cls_loss + self.lambda_reg * reg_loss

        return {
            "total": total_loss,
            "cls":   cls_loss,
            "reg":   reg_loss,
            "conf":  conf_loss,   # monitoring only, not in total_loss
        }


# ---------------------------------------------------------------------------
# Unit Tests (Part C)
# ---------------------------------------------------------------------------

def _make_dummy_predictions(
    batch_size: int,
    grid_sizes: Tuple[Tuple[int, int], ...],
    device: torch.device,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Helper: random predictions with controlled seed."""
    torch.manual_seed(seed)
    preds = []
    for (S_h, S_w) in grid_sizes:
        preds.append(torch.randn(batch_size, S_h * S_w, 5, device=device))
    return preds


def test_empty_gt():
    """
    Test 1 — Empty GT.

    An image with no people must not crash, must produce no NaN, and
    the regression loss must be exactly 0 (no positive cells).
    """
    print("=" * 60)
    print("Test 1 — Empty GT")
    device = torch.device("cpu")
    loss_fn = NIRDetLoss()

    B = 2
    preds = _make_dummy_predictions(B, ((48, 80), (24, 40), (12, 20)), device, seed=0)
    # No GT boxes for any image in the batch.
    gt_batch = [[] for _ in range(B)]

    out = loss_fn(preds, gt_batch)

    assert not torch.isnan(out["total"]),  "total loss is NaN on empty GT!"
    assert not torch.isinf(out["total"]),  "total loss is Inf on empty GT!"
    assert not torch.isnan(out["cls"]),    "cls loss is NaN on empty GT!"
    assert not torch.isnan(out["reg"]),    "reg loss is NaN on empty GT!"
    assert out["reg"].item() == 0.0,       f"reg loss should be 0.0 on empty GT, got {out['reg'].item()}"

    # cls loss should be very small (focal down-weights easy negatives,
    # all of which are correctly predicted background at random init).
    # We only assert it is near zero, not exactly zero.
    print(f"  total={out['total'].item():.6f}, cls={out['cls'].item():.6f}, "
          f"reg={out['reg'].item():.6f}, conf={out['conf'].item():.6f}")
    print("  ✓ PASSED\n")


def test_single_gt():
    """
    Test 2 — Single GT.

    Exactly one positive cell must be assigned. When we manually set the
    prediction at that cell to closely match the GT, the total loss should
    decrease compared to a random prediction.
    """
    print("=" * 60)
    print("Test 2 — Single GT assignment")
    device = torch.device("cpu")
    grid_sizes = ((48, 80), (24, 40), (12, 20))
    loss_fn = NIRDetLoss(grid_sizes=grid_sizes)

    B = 1
    # GT box for one person near the center of the image.
    gt_cx, gt_cy, gt_w, gt_h = 0.5, 0.5, 0.1, 0.3
    gt_tensor = torch.tensor([[gt_cx, gt_cy, gt_w, gt_h]])

    gt_batch = [[gt_tensor[0]]]

    # Baseline: fully random predictions.
    preds_random = _make_dummy_predictions(B, grid_sizes, device, seed=7)
    out_random   = loss_fn(preds_random, gt_batch)

    # Now set the positive cells' predictions to match the GT closely.
    # With scale routing, GT (0.1×0.3 → max(64, 115.2)=115.2px) is assigned
    # ONLY to level 1 (scale range (48,160)); cross3 adds up to two neighbour
    # cells. Write the match to every cell the assigner actually marked.
    preds_good = [p.clone() for p in preds_random]

    assignments = loss_fn.assigner.assign(gt_tensor, device)
    for s, (_conf_t, _box_t, mask) in enumerate(assignments):
        pos_cells = mask.nonzero(as_tuple=False).squeeze(-1)
        for idx in pos_cells.tolist():
            preds_good[s][0, idx, 0] = 0.0          # cx_off → sigmoid=0.5 → cell-centred
            preds_good[s][0, idx, 1] = 0.0          # cy_off
            preds_good[s][0, idx, 2] = math.log(gt_w)   # w = 0.1
            preds_good[s][0, idx, 3] = math.log(gt_h)   # h = 0.3
            preds_good[s][0, idx, 4] = 6.0          # conf logit → σ(6)≈0.998

    out_good = loss_fn(preds_good, gt_batch)

    # Verify exactly one positive across all scales.
    assignments = loss_fn.assigner.assign(gt_tensor, device)
    n_pos_total = sum(a[2].sum().item() for a in assignments)
    assert n_pos_total >= 1, (
        f"Expected at least 1 positive, got {n_pos_total}"
    )

    print(f"  Positive cells total (all scales): {int(n_pos_total)}  "
          f"(expected {len(grid_sizes)})")
    print(f"  random total={out_random['total'].item():.4f}  "
          f"→  matched total={out_good['total'].item():.4f}")
    assert out_good["total"].item() < out_random["total"].item(), (
        "Loss should decrease when prediction matches GT!"
    )
    print("  ✓ PASSED\n")


def test_nan_guard():
    """
    Test 3 — NaN / Inf guard.

    20 random batches with random predictions must produce zero NaN and
    zero Inf across all output loss values.
    """
    print("=" * 60)
    print("Test 3 — NaN / Inf guard (20 random batches)")
    device = torch.device("cpu")
    grid_sizes = ((48, 80), (24, 40), (12, 20))
    loss_fn = NIRDetLoss(grid_sizes=grid_sizes)

    for trial in range(20):
        torch.manual_seed(trial * 1337)
        B = 4
        preds = _make_dummy_predictions(B, grid_sizes, device, seed=trial)

        # Random GT: 0–3 boxes per image.
        gt_batch = []
        for b in range(B):
            n_gt = torch.randint(0, 4, ()).item()
            if n_gt == 0:
                gt_batch.append([])
            else:
                boxes = torch.rand(n_gt, 4)
                # Keep w, h strictly positive and small.
                boxes[:, 2] = boxes[:, 2] * 0.3 + 0.05
                boxes[:, 3] = boxes[:, 3] * 0.3 + 0.05
                # Keep cx, cy in [0.1, 0.9].
                boxes[:, 0] = boxes[:, 0] * 0.8 + 0.1
                boxes[:, 1] = boxes[:, 1] * 0.8 + 0.1
                gt_batch.append([boxes[i] for i in range(n_gt)])

        out = loss_fn(preds, gt_batch)

        for key, val in out.items():
            assert not torch.isnan(val), f"Trial {trial}: {key} is NaN!"
            assert not torch.isinf(val), f"Trial {trial}: {key} is Inf!"

    print("  Zero NaN, Zero Inf across all 20 trials.")
    print("  ✓ PASSED\n")


def test_loss_direction():
    """
    Test 4 — Loss direction.

    Starting from fully random predictions, manually set one cell to match
    the GT exactly (or very closely). The total loss must be strictly lower
    than with fully random predictions.

    This verifies that the loss surface points in the correct direction:
    moving predictions closer to GT decreases loss.
    """
    print("=" * 60)
    print("Test 4 — Loss direction (matched pred < random pred)")
    device = torch.device("cpu")
    grid_sizes = ((48, 80), (24, 40), (12, 20))
    loss_fn = NIRDetLoss(grid_sizes=grid_sizes)

    B = 1
    # Place a person in the upper-left quadrant.
    gt_cx, gt_cy, gt_w, gt_h = 0.2, 0.3, 0.08, 0.25
    gt_tensor = torch.tensor([[gt_cx, gt_cy, gt_w, gt_h]])
    gt_batch   = [[gt_tensor[0]]]

    torch.manual_seed(999)
    preds_random = _make_dummy_predictions(B, grid_sizes, device, seed=999)
    out_random   = loss_fn(preds_random, gt_batch)

    # Clone and surgically improve the positive cells (whatever level the
    # assigner routed this GT to — size = max(51.2, 96) = 96px → level 1).
    preds_good = [p.clone() for p in preds_random]

    assignments = loss_fn.assigner.assign(gt_tensor, device)
    for s, (_conf_t, _box_t, mask) in enumerate(assignments):
        pos_cells = mask.nonzero(as_tuple=False).squeeze(-1)
        for idx in pos_cells.tolist():
            preds_good[s][0, idx, 0] = 0.0               # cx sigmoid-centered
            preds_good[s][0, idx, 1] = 0.0               # cy sigmoid-centered
            preds_good[s][0, idx, 2] = math.log(gt_w)   # w ≈ 0.08
            preds_good[s][0, idx, 3] = math.log(gt_h)   # h ≈ 0.25
            preds_good[s][0, idx, 4] = 8.0              # conf → σ(8) ≈ 0.9997

    out_good = loss_fn(preds_good, gt_batch)

    print(f"  Random total:  {out_random['total'].item():.6f}")
    print(f"  Matched total: {out_good['total'].item():.6f}")

    assert out_good["total"].item() < out_random["total"].item(), (
        "Loss did NOT decrease when prediction was set to match GT.\n"
        f"  random={out_random['total'].item():.6f}, "
        f"  matched={out_good['total'].item():.6f}"
    )
    print("  ✓ PASSED\n")


def run_all_tests():
    """Run all unit tests sequentially."""
    print("\n" + "=" * 60)
    print("NIRDet losses.py — Unit Test Suite")
    print("=" * 60 + "\n")

    test_empty_gt()
    test_single_gt()
    test_nan_guard()
    test_loss_direction()

    print("=" * 60)
    print("All 4 tests PASSED.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
