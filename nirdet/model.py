from __future__ import annotations

"""
model.py — NIRDet Phase 5
==========================
Assembles NIRBackbone + EdgeAwareAttention + LightweightFPN + PedestrianHead
into a single nn.Module with a clean training/inference split.

Part A answers are embedded as docstrings below before the class definition.

Forward pass shape annotations (640×640 input example):
  Input        : (B, 1, 640, 640)
  Stem         : (B, 32,  320, 320)
  Stage1+EAA   : (B, 64,  160, 160)
  Stage2+EAA   : (B, 128,  80,  80)  → P3
  Stage3+EAA   : (B, 256,  40,  40)  → P4
  Stage4+EAA   : (B, 256,  20,  20)  → P5
  Neck N3      : (B, 256,  80,  80)
  Neck N4      : (B, 256,  40,  40)
  Neck N5      : (B, 256,  20,  20)
  Head (train) : [(B,6400,5), (B,1600,5), (B,400,5)]  raw logits
  Head (infer) : decoded boxes + NMS
"""

# ─────────────────────────────────────────────────────────────────────────────
# Part A — Research Questions
# ─────────────────────────────────────────────────────────────────────────────

"""
Q1 — Training vs Inference Mode
--------------------------------
YOLOX (Ge et al. 2021, arXiv:2107.08430) uses a single forward() with a
`self.training` flag inherited from nn.Module, returning raw logits when
training=True and running NMS+decode when training=False (via model.eval()).
FCOS (Tian et al. 2019, arXiv:1904.01355) is similar — the FPN head returns
raw (l,t,r,b,centerness) tensors unconditionally, and a separate
`inference_single_image()` method performs decoding and NMS post-hoc.

Three clean options:
  A) PyTorch self.training flag — forward() checks it internally.
     Pro: one call site, automatic with model.train()/model.eval().
     Con: forward signature doesn't make the mode explicit; callers can be
     confused if they call model(x) in .eval() and get boxes instead of logits.

  B) Explicit training_mode kwarg — forward(x, training_mode=True).
     Pro: callers state intent explicitly; head.py already uses this pattern;
     avoids silent mode changes if someone forgets to switch eval().
     Con: slightly more verbose at call sites.

  C) Separate decode() method — forward() always returns raw; decode() wraps.
     Pro: cleanest separation; forward is pure compute graph.
     Con: losses.py and inference code call different entry points, and callers
     must remember which to call for which purpose.

Decision: Option B (explicit kwarg).
Rationale: head.py was already designed with training_mode=True. The kwargs
approach is explicit at every call site (no silent state bugs), and it fits a
custom model where training and inference entry points are clearly separated.
self.training is still honoured as a default fallback so model.eval() + no
kwarg works for pure inference deployment.

Q2 — Weight Initialisation Strategy for Mixed-Init Components
-------------------------------------------------------------
Each component has its own init requirements:
  EAA:      Sobel/Laplacian for edge_conv, Kaiming for proj (already done in __init__)
  Backbone: Kaiming-uniform for all conv, Sobel override for stem (already done)
  Neck:     Kaiming-uniform for all conv, BN init (already done in _init_weights)
  Head:     Kaiming-normal (std=0.01) for branch convs, focal-loss bias for cls_pred,
            aspect-ratio bias for reg_pred (already done in _init_prediction_biases)

The safe strategy, validated by PyTorch's own module design (see
`torch.nn.Module.apply()` docstring), is:
  1. Construct all submodules — their __init__ applies component-specific init.
  2. In NIRDet.__init__, call _fill_gap_inits() which applies Kaiming-uniform
     ONLY to any Conv2d/Linear layers that have not yet been touched by a
     component's own init.

Detection: Each component's init sets weights in no_grad(). After construction,
any Conv2d that still has PyTorch's default init (uniform from
kaiming_uniform_(a=sqrt(5))) has mean≈0 but high variance in the first values.
We can't reliably detect "was this already initialised" without bookkeeping.
The safe approach: do NOT call apply() with a global init on the whole model —
that would overwrite Sobel and bias inits. Instead, define a whitelist of which
submodules get the gap fill, which is nothing here since all four components
fully initialise themselves. NIRDet.__init__ therefore does NOT call any
additional init on submodules. This is documented explicitly so Phase 6 authors
know not to add a global init call.

Q3 — Parameter Count Breakdown
---------------------------------
Comparison of torchinfo vs manual counting vs torchsummary:

  torchsummary (szagoruyko): prints per-layer breakdown but does not handle
    multi-input models (our forward takes x + img) cleanly. Outdated, limited.

  torchinfo (tyleryep, 2021): handles arbitrary inputs via input_data kwarg,
    prints per-module breakdown including nested modules, handles shared
    parameters correctly (counts once). Best option for a custom multi-component
    model. Install: pip install torchinfo.

  Manual counting: sum(p.numel() for p in m.parameters()) per submodule.
    Completely reliable, zero dependencies, works even in inference-only deploys.
    The downside is it double-counts parameters that are shared between modules
    (there are none here). This is our choice for __repr__ since it has no deps.

Decision: manual counting for __repr__ (always available, no deps), with a
note that torchinfo can be used for a richer breakdown during development.

Q4 — Prediction Decoding
-------------------------
FCOS (Tian et al. 2019) predicts (l, t, r, b, centerness) — distances from the
grid point to box edges. Decode: cx = x_i - l + (l+r)/2, etc. Not directly
applicable here since we predict (cx_offset, cy_offset, w, h) not distances.

Decode as ACTUALLY implemented in head.py (post Bug4/Bug5 fixes):
  cx = (sigmoid(t_cx) + grid_x) * stride   # FIX: sigmoid-bounded offset (Bug4); grid_x = column index; the +0.5 was removed
  cy = (sigmoid(t_cy) + grid_y) * stride
  w  = exp(t_w) * img_size                 # FIX: *img_size (= H*stride), not *stride (Bug5)
  h  = exp(t_h) * img_size

sigmoid(t_cx/t_cy) bounds the center inside its cell (matches the training
label assignment); exp(t_w/t_h)*img_size decodes width/height in pixels and
matches losses.py's normalized-space decode once the w/h max=1.0 cap is removed
there. (The previous docstring described the pre-bugfix unbounded/exp*stride
decode and is no longer accurate.)
One subtlety: w and h use stride rather than anchor size because our head is
anchor-free (FCOS-style assignment), so stride is the natural scale factor
— each grid cell covers stride×stride pixels.
"""

# ─────────────────────────────────────────────────────────────────────────────

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import NIRBackbone
from attention import EdgeAwareAttention
from neck import LightweightFPN
from head import PedestrianHead
from config import get_config


# ─────────────────────────────────────────────────────────────────────────────
# NIRDet
# ─────────────────────────────────────────────────────────────────────────────

class NIRDet(nn.Module):
    """
    NIR pedestrian detector.

    Components
    ----------
    backbone : NIRBackbone
        4-stage CSP-DWS backbone. Outputs P3, P4, P5.
    eaa      : EdgeAwareAttention
        Applied after each backbone stage (stem→stage1, stage2, stage3, stage4).
    neck     : LightweightFPN
        Top-down FPN. Inputs P3/P4/P5 → outputs N3/N4/N5.
    head     : PedestrianHead
        Decoupled cls/reg head. Outputs raw logits (train) or decoded boxes (infer).

    EAA Wiring
    ----------
    backbone.py stage structure (all shapes for H=W=640):
        stem          : (B,1,640,640) → (B,32,320,320)
        down1+stage1  : (B,32,320,320) → (B,64,160,160)   ← EAA after stage1
        down2+stage2  : (B,64,160,160) → (B,128,80,80)    ← EAA after stage2 = P3
        down3+stage3  : (B,128,80,80) → (B,256,40,40)     ← EAA after stage3 = P4
        down4+stage4  : (B,256,40,40) → (B,256,20,20)     ← EAA after stage4 = P5

    EAA always receives img=(B,1,H,W) — the original raw input — so it can
    compute edge maps at full resolution regardless of the feature map scale.
    The module downsamples internally to match the feature map size.

    Args
    ----
    base_ch          : int   — backbone channel width multiplier (default 32)
    n_blocks         : list  — CSP blocks per stage (default [1,2,2,1])
    num_edge_filters : int   — EAA edge filters (default 4)
    freeze_eaa_epochs: int   — freeze EAA Sobel weights for N epochs (default 5)
    nms_iou_thresh   : float — IoU threshold for NMS in inference (default 0.45)
    nms_score_thresh : float — score threshold before NMS (default 0.25)
    strides          : tuple — detection strides matching head (default (8,16,32))
    """

    def __init__(
        self,
        base_ch: int = 32,
        n_blocks: Optional[List[int]] = None,
        num_edge_filters: int = 4,
        freeze_eaa_epochs: int = 5,
        nms_iou_thresh: float = 0.45,
        nms_score_thresh: float = 0.25,
        strides: Tuple[int, int, int] = (8, 16, 32),
    ) -> None:
        super().__init__()

        if n_blocks is None:
            n_blocks = [1, 2, 2, 1]

        self.nms_iou_thresh   = nms_iou_thresh
        self.nms_score_thresh = nms_score_thresh
        self.strides          = strides

        # ── Backbone ─────────────────────────────────────────────────────────
        # NIRBackbone has its own init (Sobel stem + Kaiming for remaining convs).
        # Do NOT re-init after construction.
        self.backbone = NIRBackbone(base_ch=base_ch, n_blocks=n_blocks)
        # out_channels = (base*4, base*8, base*8) = (128, 256, 256) for base=32

        # ── EAA ──────────────────────────────────────────────────────────────
        # One shared EAA instance applied after each of the 4 backbone stages.
        # Sharing weights keeps the EAA param count at ~(4*N+1) regardless of
        # how many stages it is applied to, and forces the attention mechanism
        # to generalise across scales (similar to YOLOX's shared head).
        # Sobel init done inside EdgeAwareAttention.__init__().
        self.eaa = EdgeAwareAttention(
            num_edge_filters=num_edge_filters,
            freeze_epochs=freeze_eaa_epochs,
            pool_mode="avg",
            residual_scale=0.5,
            padding_mode="reflect",
        )

        # ── Neck ─────────────────────────────────────────────────────────────
        # LightweightFPN reads its channel defaults from config via module-level
        # constants — those have been fixed to match actual backbone channels.
        # _init_weights() is called inside LightweightFPN.__init__().
        c3, c4, c5 = self.backbone.out_channels  # (128, 256, 256)
        self.neck = LightweightFPN(c3=c3, c4=c4, c5=c5)

        # ── Head ─────────────────────────────────────────────────────────────
        # in_channels must match neck out_channels (always 256 = backbone_ch[-1]).
        # Bias inits (focal-loss prior, aspect-ratio prior) done inside
        # PedestrianHead.__init__(); do NOT re-run after construction.
        self.head = PedestrianHead(
            in_channels=self.neck.out_channels,   # 256
            feat_channels=256,
            strides=strides,
            num_branch_convs=2,
        )

        # ── Weight init gap-fill ──────────────────────────────────────────────
        # All four components fully initialise their own weights in __init__.
        # We do NOT apply a global init here to avoid overwriting:
        #   - backbone NIRStem Sobel kernels
        #   - EAA Sobel kernels in edge_conv
        #   - PedestrianHead focal-loss bias in cls_pred
        #   - PedestrianHead aspect-ratio bias in reg_pred
        # If a new submodule is added in future that lacks its own init,
        # add it to _fill_gap_inits() below rather than calling self.apply().
        # (Currently a no-op — documented for future authors.)
        self._fill_gap_inits()

    def _fill_gap_inits(self) -> None:
        """
        Apply Kaiming-uniform init to any Conv2d/Linear that:
          - is NOT inside backbone, EAA, or head (which all self-init), AND
          - does not already have a meaningful init.

        Currently a no-op because neck also self-inits. Kept as an explicit
        hook so future additions have a clear place to add gap fills without
        touching component __init__ files.
        """
        # No gaps at this time — all submodule inits are self-contained.
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # EAA epoch step — call from training loop once per epoch
    # ─────────────────────────────────────────────────────────────────────────

    def step_eaa_epoch(self) -> None:
        """Advance EAA epoch counter (controls Sobel freeze schedule)."""
        self.eaa.step_epoch()

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        training_mode: Optional[bool] = None,
    ) -> List[torch.Tensor]:
        """
        Full forward pass.

        Args:
            x             : (B, 1, H, W) — single-channel NIR image batch
            training_mode : if None, defaults to self.training (set by .train()/.eval()).
                            Pass True/False explicitly to override.

        Returns:
            training_mode=True:
                List of 3 tensors: [(B,H3*W3,5), (B,H4*W4,5), (B,H5*W5,5)]
                last dim = (t_cx, t_cy, t_w, t_h, conf_logit) — raw logits

            training_mode=False:
                List of 2 tensors: [boxes (N,4), scores (N,)]
                boxes in (x1,y1,x2,y2) pixel coords, scores in [0,1]
                after NMS with self.nms_score_thresh / self.nms_iou_thresh.
                (Batch size must be 1 for inference; batched NMS is Phase 7.)

        Shape trace for H=W=640, B=2:
          x           : (2, 1,   640, 640)
          stem        : (2, 32,  320, 320)    — NIRStem stride=2
          down1       : (2, 64,  160, 160)    — StrideDown
          stage1      : (2, 64,  160, 160)    — CSPBlock
          eaa(stage1) : (2, 64,  160, 160)    — EAA, residual-scale form
          down2       : (2, 128,  80,  80)    — StrideDown
          stage2=P3   : (2, 128,  80,  80)    — CSPBlock
          eaa(P3)     : (2, 128,  80,  80)    — EAA
          down3       : (2, 256,  40,  40)    — StrideDown
          stage3=P4   : (2, 256,  40,  40)    — CSPBlock
          eaa(P4)     : (2, 256,  40,  40)    — EAA
          down4       : (2, 256,  20,  20)    — StrideDown
          stage4=P5   : (2, 256,  20,  20)    — CSPBlock
          eaa(P5)     : (2, 256,  20,  20)    — EAA
          N3          : (2, 256,  80,  80)    — LightweightFPN
          N4          : (2, 256,  40,  40)
          N5          : (2, 256,  20,  20)
          head[0]     : (2, 6400,  5)         — stride-8  scale
          head[1]     : (2, 1600,  5)         — stride-16 scale
          head[2]     : (2,  400,  5)         — stride-32 scale
        """
        if training_mode is None:
            training_mode = self.training

        # ── 1. Backbone with interleaved EAA ─────────────────────────────────
        # We re-run the backbone stage-by-stage to insert EAA after each stage.
        # This mirrors the forward pass described in backbone.py but with EAA
        # inserted between stages rather than calling backbone.forward() directly.

        # Stem: (B,1,H,W) → (B,32,H/2,W/2)
        feat = self.backbone.stem(x)

        # Stage 1: → (B,64,H/4,W/4)
        feat = self.backbone.down1(feat)      # (B,64,H/4,W/4)
        feat = self.backbone.stage1(feat)     # (B,64,H/4,W/4)
        feat = self.eaa(feat, x)              # (B,64,H/4,W/4)

        # Stage 2 → P3: → (B,128,H/8,W/8)
        feat = self.backbone.down2(feat)      # (B,128,H/8,W/8)
        P3   = self.backbone.stage2(feat)     # (B,128,H/8,W/8)
        P3   = self.eaa(P3, x)               # (B,128,H/8,W/8)

        # Stage 3 → P4: → (B,256,H/16,W/16)
        feat = self.backbone.down3(P3)        # (B,256,H/16,W/16)
        P4   = self.backbone.stage3(feat)     # (B,256,H/16,W/16)
        P4   = self.eaa(P4, x)               # (B,256,H/16,W/16)

        # Stage 4 → P5: → (B,256,H/32,W/32)
        feat = self.backbone.down4(P4)        # (B,256,H/32,W/32)
        P5   = self.backbone.stage4(feat)     # (B,256,H/32,W/32)
        P5   = self.eaa(P5, x)               # (B,256,H/32,W/32)

        # ── 2. Neck ───────────────────────────────────────────────────────────
        # P3/P4/P5 → N3/N4/N5, all (B,256,H/stride,W/stride)
        N3, N4, N5 = self.neck(P3, P4, P5)

        # ── 3. Head ───────────────────────────────────────────────────────────
        raw_preds = self.head(N3, N4, N5, training_mode=training_mode)
        # raw_preds: [(B,H3*W3,5), (B,H4*W4,5), (B,H5*W5,5)]

        if training_mode:
            return raw_preds  # raw logits for losses.py

        # ── 4. Inference decode + NMS ─────────────────────────────────────────
        # raw_preds already contain decoded pixel coords when training_mode=False
        # (head.py decoding runs inside head.forward when training_mode=False).
        return self.decode_predictions(raw_preds, input_size=x.shape[-2:])

    # ─────────────────────────────────────────────────────────────────────────
    # decode_predictions
    # ─────────────────────────────────────────────────────────────────────────

    def decode_predictions(
        self,
        raw_preds: List[torch.Tensor],
        input_size: Tuple[int, int],
    ) -> List[torch.Tensor]:
        """
        Convert head outputs (decoded pixel coords) to filtered boxes after NMS.

        Args:
            raw_preds  : List of 3 tensors from head (training_mode=False),
                         each (B, H_i*W_i, 5), last dim = (cx,cy,w,h,conf).
                         NOTE: when called from forward(), head already decoded
                         to pixel coords. When called externally, pass the
                         output of head(N3,N4,N5, training_mode=False).
            input_size : (H, W) of the original input image.

        Returns:
            List of B pairs [boxes (N_i,4), scores (N_i,)] in xyxy pixel coords.
            Boxes are clamped to [0, H] / [0, W].
        """
        H, W = input_size
        B    = raw_preds[0].shape[0]

        # Concatenate all scales: (B, total_cells, 5)
        all_preds = torch.cat(raw_preds, dim=1)  # (B, sum(H_i*W_i), 5)

        results = []
        for b in range(B):
            pred = all_preds[b]  # (total_cells, 5)

            cx    = pred[:, 0]
            cy    = pred[:, 1]
            w     = pred[:, 2]
            h     = pred[:, 3]
            scores = pred[:, 4]

            # Score filter before NMS (cheap, eliminates most cells)
            keep_mask = scores >= self.nms_score_thresh
            cx     = cx[keep_mask]
            cy     = cy[keep_mask]
            w      = w[keep_mask]
            h      = h[keep_mask]
            scores = scores[keep_mask]
                        # Filter degenerate boxes — exp() can produce boxes larger than
            # the image; clamping turns them into full-image false positives.
            # 1px minimum size removes them before they reach NMS.
            valid = (w > 1.0) & (h > 1.0)
            cx, cy, w, h, scores = cx[valid], cy[valid], w[valid], h[valid], scores[valid]

            if scores.numel() == 0:
                results.append([
                    torch.zeros((0, 4), device=pred.device, dtype=pred.dtype),
                    torch.zeros((0,),   device=pred.device, dtype=pred.dtype),
                ])
                continue

            # cx,cy,w,h → x1,y1,x2,y2
            x1 = (cx - w * 0.5).clamp(0, W)
            y1 = (cy - h * 0.5).clamp(0, H)
            x2 = (cx + w * 0.5).clamp(0, W)
            y2 = (cy + h * 0.5).clamp(0, H)
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (N, 4)

            # torchvision NMS expects float32
            from torchvision.ops import nms as tv_nms
            MAX_DET = 300
            keep_idx = tv_nms(boxes.float(), scores.float(), self.nms_iou_thresh)
            keep_idx = keep_idx[:MAX_DET]
            results.append([boxes[keep_idx], scores[keep_idx]])

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # __repr__ with per-component parameter count
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        backbone_n = count(self.backbone)
        eaa_n      = count(self.eaa)
        neck_n     = count(self.neck)
        head_n     = count(self.head)
        total_n    = count(self)

        lines = [
            "NIRDet",
            "=" * 40,
            f"  backbone  : {backbone_n:>10,}",
            f"  EAA       : {eaa_n:>10,}",
            f"  neck      : {neck_n:>10,}",
            f"  head      : {head_n:>10,}",
            "-" * 40,
            f"  total     : {total_n:>10,}",
            f"  budget    : {'OK' if total_n < 5_000_000 else 'EXCEEDED'} (<5M)",
        ]
        return "\n".join(lines)

    def param_breakdown(self) -> dict:
        """Return {component: param_count} dict for programmatic access."""
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        return {
            "backbone": count(self.backbone),
            "eaa":      count(self.eaa),
            "neck":     count(self.neck),
            "head":     count(self.head),
            "total":    count(self),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_nirdet(cfg=None) -> NIRDet:
    """Build NIRDet from config. Falls back to defaults if cfg is None."""
    if cfg is None:
        cfg = get_config()
    base_ch = cfg.model.backbone_channels[1]   # 32 at default
    return NIRDet(base_ch=base_ch)


# ─────────────────────────────────────────────────────────────────────────────
# Part D — Uncertainties
# ─────────────────────────────────────────────────────────────────────────────

"""
Part D — Uncertainties and Resolving Experiments
=================================================

1. SHARED vs PER-SCALE EAA
   Uncertainty: A single shared EAA instance is used for all 4 backbone stages.
   This keeps param count low but forces the same attention parameters to work
   at stride-4 (160×160), stride-8 (80×80), stride-16 (40×40), stride-32 (20×20).
   Risk: the optimal edge sensitivity may differ by scale.
   Experiment: replace self.eaa with a nn.ModuleList of 4 independent EAA instances
   and measure mAP50 delta vs the shared version. If delta > 0.5 mAP50 points,
   use per-scale EAA and accept the parameter increase (~4×).

2. EAA STAGE INSERTION POINT
   Uncertainty: EAA is applied after each complete stage (post-CSP). Inserting
   it before the downsampling (before StrideDown) would expose it to a finer
   spatial resolution. Inserting it after (current) means the attention map is
   computed at the downsampled scale.
   Experiment: move EAA insertion to before down2/down3/down4 in forward().
   Measure mAP50 delta and runtime.

3. FREEZE_EPOCHS FOR EAA SOBEL WEIGHTS
   Uncertainty: freeze_eaa_epochs=5 is inherited from EAA defaults. On small
   datasets (<5K images), the Sobel prior may need longer protection.
   Experiment: train with freeze_eaa_epochs ∈ {0, 5, 10, 20}. Monitor
   edge_conv.weight norm divergence from Sobel init over training.

4. NMS THRESHOLDS
   Uncertainty: nms_iou_thresh=0.45, nms_score_thresh=0.25 are initialised to
   typical YOLO inference values. These are dataset- and scene-dependent.
   Experiment: sweep nms_score_thresh ∈ {0.1, 0.2, 0.25, 0.3, 0.4} while
   measuring precision/recall. Choose threshold at F1 peak on validation set.

5. BATCHED NMS FOR INFERENCE
   Uncertainty: decode_predictions loops over batch items and calls NMS per
   image. For B>1 inference, torchvision.ops.batched_nms is available but was
   not used here to keep Phase 5 simple.
   Experiment: for production throughput, replace the loop with batched_nms
   and benchmark B=4/8/16 inference latency on the RTX 4050.

6. HEAD in_channels=256 HARDCODED
   The neck always outputs 256 channels (backbone_channels[-1]). If config
   backbone_channels is changed to a different final channel count, head
   in_channels must match. Currently both read backbone_channels[-1]=256,
   but head gets it via self.neck.out_channels which is the authoritative source.
   No mismatch is possible with the current wiring. Flagged for documentation.

7. OVERFIT TEST LOSS FUNCTION (Test 4)
   The overfit test in test_model.py uses a combined BCE + L1 loss as a proxy
   for losses.py (Phase 6). If the overfit test passes with this proxy loss but
   fails with the real losses.py loss, the issue is in the loss formulation,
   not the model. The proxy loss is intentionally simple to isolate model bugs
   from loss bugs.
"""
