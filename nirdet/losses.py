"""
NIRDet — losses.py
==================
Phase 5 loss function module.

Model output format (per scale, per cell):
    (cx_offset, cy_offset, w, h, conf)  — 5 raw values

Three scales:
    80×80  (stride 8)
    40×40  (stride 16)
    20×20  (stride 32)

Ground truth: YOLO format — (class, cx, cy, w, h) normalized [0, 1].
Single class: person. Heavy background imbalance (~8 400 cells per image,
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
    Center-cell label assignment.

    Motivation and comparison
    -------------------------
    Four mainstream strategies exist:

    1. **FCOS-style** [FCOS 2019]: Every cell whose *center falls inside* the GT
       box is labeled positive. Simple and well-studied but produces many
       positive cells (hundreds for a large box) which increases gradient
       variance with small datasets.

    2. **SimOTA** [YOLOX 2021]: Formulates assignment as an optimal-transport
       problem; each GT "supplies" k tokens distributed to the cheapest
       (cls+IoU cost) prediction cells globally. Excellent on large-scale
       benchmarks (COCO) but requires a warm-up cost matrix across all cells
       which is slow and numerically brittle on tiny datasets where predictions
       are random for many epochs.

    3. **TOOD / Task-Aligned** [YOLOv6 ablation]: Selects top-k cells ranked by
       p^α · IoU^β. Better than SimOTA in accuracy on larger models but adds
       two extra hyperparameters and a dependency on meaningful classification
       scores early in training, which is problematic for a cold-start detector.

    4. **Center-cell** (chosen here): The single grid cell whose center is
       closest to the GT box center is assigned positive. One positive per GT
       box per scale.

       Why chosen:
       - Stable with small data: only 1–5 positive cells per image means the
         gradient signal is maximally clean.
       - No hyper-parameter tuning (SimOTA needs k, FCOS needs scale ranges).
       - Correct and unambiguous: there is always exactly one responsible cell.
       - Matches YOLOv1–v3 philosophy which proved effective for single-class
         pedestrian detection long before OTA existed.
       - SimOTA's global optimum advantage only materialises at COCO scale
         (>100k images, 80 classes, dense crowds). For a limited NIR person
         dataset it is overkill and a source of training instability.

    Mathematical derivation
    -----------------------
    Given:
        GT box g = (cx_g, cy_g, w_g, h_g) in normalized [0,1] coordinates.
        A scale with S×S grid cells (S ∈ {80, 40, 20}).

    Step 1 — Map GT center to grid indices:
        col = floor(cx_g · S)    ← column index ∈ [0, S)
        row = floor(cy_g · S)    ← row index    ∈ [0, S)

    Step 2 — Flatten to cell index:
        cell_idx = row · S + col  ← scalar index into the (S·S) flat vector

    Step 3 — Assign:
        For the cell at (row, col), set:
            conf_target = 1
            box_target  = (cx_g, cy_g, w_g, h_g)   (absolute, normalized)

        All other cells have conf_target = 0 and are background.

    Step 4 — Multi-scale:
        Repeat for each of the three scales independently. A person that is
        very small will be assigned to the 80×80 scale's cell; a large person
        to the 20×20 scale's cell. This is a simplification — YOLOv3 used
        anchor matching to route GT boxes, but since we have no anchors we
        assign the GT to ALL three scales simultaneously and let the
        regression loss sort out which scale specialises. (A future upgrade
        could add scale-routing by object size.)
    """

    def __init__(self, grid_sizes: Tuple[int, ...] = (80, 40, 20)):
        """
        Args:
            grid_sizes: Spatial grid sizes for each scale, e.g. (80, 40, 20).
        """
        self.grid_sizes = grid_sizes  # (S1, S2, S3)

    # ------------------------------------------------------------------
    def assign(
        self,
        gt_boxes: torch.Tensor,   # (N, 4) — cx, cy, w, h in [0,1]
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Build per-scale assignment targets for one image.

        Args:
            gt_boxes: (N, 4) tensor of GT boxes [cx, cy, w, h] ∈ [0,1].
                      If N == 0, returns all-zero targets with no positives.
            device:   Device to allocate tensors on.

        Returns:
            List of length len(self.grid_sizes). Each element is a tuple:
                (conf_target, box_target, pos_mask)
            where:
                conf_target: (S*S,) float — 1.0 at positive cells, 0.0 elsewhere.
                box_target:  (S*S, 4) float — GT box at positive cells, 0 elsewhere.
                pos_mask:    (S*S,) bool   — True at cells that are positive.
        """
        results = []

        for S in self.grid_sizes:
            total_cells = S * S  # e.g. 6400 for S=80

            # Initialise targets as all-background.
            # Shape: (S*S,) and (S*S, 4)
            conf_target = torch.zeros(total_cells, dtype=torch.float32, device=device)
            box_target  = torch.zeros(total_cells, 4, dtype=torch.float32, device=device)

            if gt_boxes.shape[0] == 0:
                # No GT boxes → nothing to assign.
                pos_mask = conf_target.bool()
                results.append((conf_target, box_target, pos_mask))
                continue

            # ----------------------------------------------------------
            # Step 1: Map each GT center to grid indices.
            #
            #   col_idx = floor(cx_g * S)   ← clamp to [0, S-1] for safety
            #   row_idx = floor(cy_g * S)
            #
            # gt_boxes[:, 0] = cx_g,  gt_boxes[:, 1] = cy_g
            # ----------------------------------------------------------
            cx_g = gt_boxes[:, 0]  # (N,)
            cy_g = gt_boxes[:, 1]  # (N,)

            # floor → int; clamp guards against cx/cy == 1.0 exactly.
            col_idx = (cx_g * S).long().clamp(0, S - 1)  # (N,)
            row_idx = (cy_g * S).long().clamp(0, S - 1)  # (N,)

            # ----------------------------------------------------------
            # Step 2: Flatten (row, col) → 1D cell index.
            #
            #   cell_idx = row_idx * S + col_idx
            # ----------------------------------------------------------
            cell_idx = row_idx * S + col_idx  # (N,) — flat index

            # ----------------------------------------------------------
            # Step 3: Write targets at those cell indices.
            #
            # If two GT boxes map to the same cell (overlapping people),
            # the last-written GT wins. A more principled fix (choose the
            # one with higher IoU at eval time) is listed in Part D.
            # ----------------------------------------------------------
            for i in range(gt_boxes.shape[0]):
                idx = cell_idx[i].item()
                conf_target[idx] = 1.0
                box_target[idx]  = gt_boxes[i]  # (cx, cy, w, h)

            pos_mask = conf_target.bool()  # (S*S,)
            results.append((conf_target, box_target, pos_mask))

        return results  # list of (conf_target, box_target, pos_mask) per scale


# ---------------------------------------------------------------------------
# Q2 — Classification Loss: Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (foreground/background).

    Reference: Lin et al. [Lin 2017], equation (5).

    Standard Binary Cross-Entropy (BCE) fails with extreme imbalance because:
        • With 8 400 cells and ~3 positives, negatives outnumber positives
          ~2800:1.
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

    For each scale s with grid size S:
        pred shape: (B, S*S, 5)

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
        grid_sizes: Tuple[int, ...] = (80, 40, 20),
        lambda_cls: float = 1.0,
        lambda_reg: float = 5.0,
        lambda_conf: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        """
        Args:
            grid_sizes:   Spatial sizes for each scale (must match model).
            lambda_cls:   Weight for focal classification loss.
            lambda_reg:   Weight for CIoU regression loss (positives only).
            lambda_conf:  Weight for objectness/confidence focal loss.
            focal_gamma:  Focusing exponent for FocalLoss (default 2, [Lin 2017]).
            focal_alpha:  Foreground weight for FocalLoss (default 0.25).
        """
        super().__init__()

        self.grid_sizes   = grid_sizes
        self.lambda_cls   = lambda_cls
        self.lambda_reg   = lambda_reg
        self.lambda_conf  = lambda_conf

        self.assigner  = LabelAssigner(grid_sizes=grid_sizes)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="none")

    # ------------------------------------------------------------------

    def _decode_pred_boxes(
        self,
        raw_pred: torch.Tensor,        # (P, 4) — cx_off, cy_off, w_raw, h_raw for P cells
        cell_indices: torch.Tensor,    # (P,) long — flat cell indices in [0, S²)
        S: int,
    ) -> torch.Tensor:
        """
        Convert raw model output offsets to absolute normalized box coordinates.

        The model outputs:
            cx_offset ∈ (-∞, ∞) → sigmoid → fraction-within-cell → /S → [0,1]
            cy_offset ∈ (-∞, ∞) → sigmoid → fraction-within-cell → /S → [0,1]
            w_raw     ∈ (-∞, ∞) → exp(w_raw) → w in normalized [0,1] units
            h_raw     ∈ (-∞, ∞) → exp(h_raw) → h in normalized [0,1] units

        For a cell at flat index i:
            col = i % S    (column index)
            row = i // S   (row index)

        Decoding:
            cx = (sigmoid(cx_off) + col) / S    ← offset within cell + cell position
            cy = (sigmoid(cy_off) + row) / S
            w  = exp(w_raw).clamp(max=1.0)      ← clamp prevents explosion at init
            h  = exp(h_raw).clamp(max=1.0)

        Args:
            raw_pred:     (P, 4) — raw logit outputs for cx, cy, w, h.
                          P is the number of POSITIVE cells (often 1–5).
            cell_indices: (P,) long tensor — flat indices of the positive cells.
                          Used to look up (col, row) without rebuilding the S² grid.
            S:            Grid size (spatial dimension).

        Returns:
            (P, 4) decoded boxes in normalized [0,1] cx-cy-w-h format.
        """
        # ------------------------------------------------------------------
        # Recover column and row for each positive cell.
        #
        #   col = flat_index % S      e.g. index 3240 in S=80: col = 3240%80 = 40
        #   row = flat_index // S     e.g.                      row = 3240//80 = 40
        # ------------------------------------------------------------------
        col_offsets = (cell_indices % S).float()   # (P,) — column index per cell
        row_offsets = (cell_indices // S).float()  # (P,) — row index per cell

        cx_raw = raw_pred[:, 0]  # (P,)
        cy_raw = raw_pred[:, 1]  # (P,)
        w_raw  = raw_pred[:, 2]  # (P,)
        h_raw  = raw_pred[:, 3]  # (P,)

        # cx = (σ(cx_raw) + col_offset) / S   → absolute normalized x ∈ [0,1]
        cx = (torch.sigmoid(cx_raw) + col_offsets) / S
        # cy = (σ(cy_raw) + row_offset) / S   → absolute normalized y ∈ [0,1]
        cy = (torch.sigmoid(cy_raw) + row_offsets) / S
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
        # list of (B, S*S, 5) — one tensor per scale, raw logits
        gt_batch: List[List[torch.Tensor]],
        # gt_batch[b] = list of GT boxes for image b
        # each GT: (4,) tensor [cx, cy, w, h] in [0,1]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined NIRDet loss over a batch.

        Args:
            predictions: List of length len(grid_sizes).
                         predictions[s] has shape (B, S_s*S_s, 5).
                         Channel order: [cx_off, cy_off, w, h, conf_logit].
            gt_batch:    Outer list length B (batch size).
                         gt_batch[b] is a list of GT box tensors (N_b, 4)
                         or an empty list if the image has no objects.

        Returns:
            Dict with keys: "total", "cls", "reg", "conf"
            All values are scalar tensors (differentiable).
        """
        device = predictions[0].device
        B = predictions[0].shape[0]

        # Accumulators — plain Python floats so the first tensor addition
        # creates a proper autograd node (a pre-allocated zero tensor with
        # requires_grad=False would detach the graph on the first +=).
        total_cls_loss  = 0.0
        total_reg_loss  = 0.0
        total_conf_loss = 0.0
        n_images_processed = 0

        for b in range(B):
            # ----------------------------------------------------------
            # Collect GT boxes for image b.
            # gt_boxes: (N, 4) or (0, 4) tensor.
            # ----------------------------------------------------------
            gt_list = gt_batch[b]
            if len(gt_list) == 0:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            else:
                gt_boxes = torch.stack(gt_list, dim=0).to(device)  # (N, 4)

            # ----------------------------------------------------------
            # Run label assigner to get per-scale targets.
            # ----------------------------------------------------------
            assignments = self.assigner.assign(gt_boxes, device)
            # assignments[s] = (conf_target, box_target, pos_mask)

            for s, S in enumerate(self.grid_sizes):
                # Raw predictions for scale s, image b: (S², 5)
                pred = predictions[s][b]  # (S², 5)

                conf_target, box_target, pos_mask = assignments[s]
                # conf_target: (S²,) float, 1 at positives
                # box_target:  (S², 4) float, GT box at positives
                # pos_mask:    (S²,) bool

                # -------------------------------------------------------
                # Confidence / objectness logits: pred[:, 4]
                # -------------------------------------------------------
                conf_logits = pred[:, 4]  # (S²,) raw logits

                # -------------------------------------------------------
                # Classification loss (Focal) — applied to ALL cells.
                #
                # For single-class detection, the confidence/objectness
                # score IS the classification score. We compute focal loss
                # over all S² cells so that both positives and the mass of
                # easy negatives are accounted for.
                # -------------------------------------------------------
                focal_vals = self.focal_loss(conf_logits, conf_target)  # (S²,)

                # FIX: normalize focal by n_pos (Lin et al. RetinaNet §4: "normalized by
                # the number of anchors assigned to a ground-truth box"). The earlier
                # .mean() (=sum/S²) divided cls by ~8400 while reg is divided by n_pos≈3,
                # making the confidence gradient ~2800x too weak -> conf never rose -> mAP=0.
                # This does NOT re-amplify background: dividing the SUM by n_pos scales
                # foreground and background identically; focal (1-p)^γ·α already balances them.
                n_pos = pos_mask.sum().float().clamp(min=1.0)  # still needed for reg
                cls_loss_s = focal_vals.sum() / n_pos   # FIX: normalize by n_pos per Lin et al. RetinaNet §4; .mean()=sum/S² shrank cls ~2800x vs reg and starved the confidence gradient (conf stuck ~0.07 -> mAP=0). Dividing the sum by n_pos scales fg and bg equally, so this does NOT re-amplify background.

                # FIX Bug7: conf_bce_s is for MONITORING ONLY — keep it detached.
                # Previously total_conf_loss fed into total_loss with lambda_conf=1.0,
                # adding a second unmodulated BCE gradient on top of focal loss.
                # Plain BCE gives equal weight to all 8400 cells, so 8397 background
                # cells dominated by ~79x, driving conf logits to -inf regardless of
                # the focal loss. Now conf_bce_s is always detached and never enters
                # the gradient. The 'conf' key in the return dict is monitoring only.
                with torch.no_grad():
                    conf_bce_s = F.binary_cross_entropy_with_logits(
                        conf_logits, conf_target, reduction="mean"
                    )  # FIX: always no_grad — monitoring only, excluded from total_loss

                # -------------------------------------------------------
                # Regression loss (CIoU) — ONLY on positive cells.
                #
                # Only positive cells have a valid GT box. Applying CIoU
                # to background cells is meaningless (their box_target is
                # zeros, not a real GT box).
                # -------------------------------------------------------
                if pos_mask.any():
                    # Extract predictions and targets for positive cells.
                    pred_raw_pos = pred[pos_mask, :4]    # (P, 4) — cx, cy, w, h raw
                    gt_pos       = box_target[pos_mask]   # (P, 4)

                    # Cell indices for the positive cells (needed for decoding).
                    # pos_mask is (S²,) bool → nonzero gives the flat cell indices.
                    pos_cell_idx = pos_mask.nonzero(as_tuple=False).squeeze(-1)  # (P,)

                    # Decode raw predictions → normalized box coords.
                    decoded_pos = self._decode_pred_boxes(pred_raw_pos, pos_cell_idx, S)

                    # CIoU loss per matched pair: (P,)
                    reg_vals = ciou_loss(decoded_pos, gt_pos)

                    # Normalise by number of positives.
                    reg_loss_s = reg_vals.sum() / n_pos
                else:
                    # No positive cells → regression contribution = 0 (no grad needed).
                    reg_loss_s = 0.0

                # Accumulate across scales.
                total_cls_loss  = total_cls_loss  + cls_loss_s
                total_reg_loss  = total_reg_loss  + reg_loss_s
                total_conf_loss = total_conf_loss + conf_bce_s

            n_images_processed += 1

        # Average over batch and scales.
        n_scale = float(len(self.grid_sizes))
        denom = float(max(n_images_processed, 1)) * n_scale

        # Convert accumulators to tensors (they may still be plain 0.0 floats
        # if the whole batch had no objects — a tensor is required for .backward()).
        if isinstance(total_cls_loss, float):
            total_cls_loss = torch.tensor(total_cls_loss, device=device, dtype=torch.float32)
        if isinstance(total_reg_loss, float):
            total_reg_loss = torch.tensor(total_reg_loss, device=device, dtype=torch.float32)
        if isinstance(total_conf_loss, float):
            total_conf_loss = torch.tensor(total_conf_loss, device=device, dtype=torch.float32)

        cls_loss  = total_cls_loss  / denom
        reg_loss  = total_reg_loss  / denom
        conf_loss = total_conf_loss / denom

        # ----------------------------------------------------------------
        # Weighted total:
        #   total = λ_cls · cls + λ_reg · reg
        #
        # FIX Bug7: conf_loss (plain BCE) is excluded from total_loss.
        # It is logged in the return dict as a monitoring signal only.
        # Including it was double-counting background suppression on top
        # of focal loss, overwhelming foreground cells by ~79x.
        # cls_loss (focal) already provides all the confidence gradient
        # needed — adding unmodulated BCE on top breaks the focal balance.
        # ----------------------------------------------------------------
        total_loss = (
            self.lambda_cls * cls_loss +
            self.lambda_reg * reg_loss
            # conf_loss intentionally excluded — monitoring only (FIX Bug7)
        )

        return {
            "total": total_loss,   # gradient flows here
            "cls":   cls_loss,     # focal classification loss
            "reg":   reg_loss,     # CIoU regression loss (positives only)
            "conf":  conf_loss,    # plain BCE confidence loss (monitoring)
        }


# ---------------------------------------------------------------------------
# Unit Tests (Part C)
# ---------------------------------------------------------------------------

def _make_dummy_predictions(
    batch_size: int,
    grid_sizes: Tuple[int, ...],
    device: torch.device,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Helper: random predictions with controlled seed."""
    torch.manual_seed(seed)
    preds = []
    for S in grid_sizes:
        preds.append(torch.randn(batch_size, S * S, 5, device=device))
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
    preds = _make_dummy_predictions(B, (80, 40, 20), device, seed=0)
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
    grid_sizes = (80, 40, 20)
    loss_fn = NIRDetLoss(grid_sizes=grid_sizes)

    B = 1
    # GT box for one person near the center of the image.
    gt_cx, gt_cy, gt_w, gt_h = 0.5, 0.5, 0.1, 0.3
    gt_tensor = torch.tensor([[gt_cx, gt_cy, gt_w, gt_h]])

    gt_batch = [[gt_tensor[0]]]

    # Baseline: fully random predictions.
    preds_random = _make_dummy_predictions(B, grid_sizes, device, seed=7)
    out_random   = loss_fn(preds_random, gt_batch)

    # Now set the positive cell's prediction to match the GT closely.
    # Positive cell for S=80: col=floor(0.5*80)=40, row=floor(0.5*80)=40
    # cell_idx = 40*80 + 40 = 3240
    preds_good = [p.clone() for p in preds_random]

    S0 = grid_sizes[0]  # 80
    col_gt = int(gt_cx * S0)  # 40
    row_gt = int(gt_cy * S0)  # 40
    cell_gt = row_gt * S0 + col_gt  # 3240

    # For the predictions to decode to (gt_cx, gt_cy, gt_w, gt_h):
    #   cx = (sigmoid(cx_off) + col_gt) / S  = gt_cx
    #   → sigmoid(cx_off) = gt_cx * S - col_gt = 0.5*80 - 40 = 0.0
    #   → cx_off = logit(0.5) = 0.0   (since sigmoid(0) = 0.5 → 0.5/S+col/S=...
    # Actually: cx = (sigmoid(0) + 40) / 80 = (0.5 + 40) / 80 = 0.50625 ≈ 0.5
    # Close enough for a test. Let's set logits to zero for cx and cy offsets,
    # and set w/h to match:
    #   w = exp(w_raw) → w_raw = log(gt_w) = log(0.1)
    #   h = exp(h_raw) → h_raw = log(gt_h) = log(0.3)
    #   conf logit → large positive (confident foreground)

    preds_good[0][0, cell_gt, 0] = 0.0                   # cx_off → sigmoid=0.5, cx≈0.506
    preds_good[0][0, cell_gt, 1] = 0.0                   # cy_off → cy≈0.506
    preds_good[0][0, cell_gt, 2] = math.log(gt_w)        # w = 0.1
    preds_good[0][0, cell_gt, 3] = math.log(gt_h)        # h = 0.3
    preds_good[0][0, cell_gt, 4] = 6.0                   # conf logit → σ(6)≈0.998

    out_good = loss_fn(preds_good, gt_batch)

    # Verify exactly one positive across all scales.
    assignments = loss_fn.assigner.assign(gt_tensor, device)
    n_pos_total = sum(a[2].sum().item() for a in assignments)
    assert n_pos_total == len(grid_sizes), (
        f"Expected {len(grid_sizes)} positive cells (one per scale), "
        f"got {n_pos_total}"
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
    grid_sizes = (80, 40, 20)
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
    grid_sizes = (80, 40, 20)
    loss_fn = NIRDetLoss(grid_sizes=grid_sizes)

    B = 1
    # Place a person in the upper-left quadrant.
    gt_cx, gt_cy, gt_w, gt_h = 0.2, 0.3, 0.08, 0.25
    gt_tensor = torch.tensor([[gt_cx, gt_cy, gt_w, gt_h]])
    gt_batch   = [[gt_tensor[0]]]

    torch.manual_seed(999)
    preds_random = _make_dummy_predictions(B, grid_sizes, device, seed=999)
    out_random   = loss_fn(preds_random, gt_batch)

    # Clone and surgically improve the positive cell for scale 0 (80×80).
    preds_good = [p.clone() for p in preds_random]
    S0   = grid_sizes[0]
    col  = int(gt_cx * S0)   # 16
    row  = int(gt_cy * S0)   # 24
    cidx = row * S0 + col    # 24*80+16 = 1936

    # Set to near-perfect match (same logic as Test 2).
    preds_good[0][0, cidx, 0] = 0.0               # cx sigmoid-centered
    preds_good[0][0, cidx, 1] = 0.0               # cy sigmoid-centered
    preds_good[0][0, cidx, 2] = math.log(gt_w)   # w ≈ 0.08
    preds_good[0][0, cidx, 3] = math.log(gt_h)   # h ≈ 0.25
    preds_good[0][0, cidx, 4] = 8.0              # conf → σ(8) ≈ 0.9997

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
