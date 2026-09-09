"""
head.py — NIRDet PedestrianHead

Single-class (person) detection head consumed by the neck, which outputs:
    N3: (B, 256, H/8,  W/8)
    N4: (B, 256, H/16, W/16)
    N5: (B, 256, H/32, W/32)

Design summary (see Part A of the response this file shipped with for the
full justification and what is/isn't backed by a citable ablation):

  - Grid-based, dense-positive-assignment (FCOS-style) point detector, but
    decoding DIRECTLY to (cx, cy, w, h) rather than FCOS's (l, t, r, b)
    distances, because direct w/h regression is the natural place to bake
    in a pedestrian aspect-ratio prior (Q1/Q2).
  - Decoupled cls/reg branches, YOLOX-style (Q3).
  - No centerness branch (Q4) — left as a documented, easy extension point.
  - Aspect-ratio bias init on the w/h output channels, extrapolating FCOS's
    classification-bias-encodes-a-prior trick (Q2) onto regression, which
    is NOT something the original FCOS code does — flagged explicitly.

Every line below is annotated with its tensor shape.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from config import DECODE_OFFSET_SCALE
_OFF_S = DECODE_OFFSET_SCALE
_OFF_B = (DECODE_OFFSET_SCALE - 1.0) / 2.0


def _init_conv_kaiming_normal(conv: nn.Conv2d, std: float = 0.01) -> None:
    """Match FCOS's own init convention: normal(std=0.01) weights, zero bias.
    (Verified against tianzhi0549/FCOS fcos.py and the torchvision port.)
    """
    nn.init.normal_(conv.weight, std=std)
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)


class ConvGNReLU(nn.Module):
    """3x3 conv -> GroupNorm -> ReLU. Spatial size (H, W) is unchanged."""

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(num_groups, channels)
        self.act = nn.ReLU(inplace=True)
        _init_conv_kaiming_normal(self.conv, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.conv(x)  # (B, C, H, W) -- 3x3, padding=1 => spatial size preserved
        x = self.gn(x)    # (B, C, H, W) -- GroupNorm normalizes within channel groups, shape unchanged
        x = self.act(x)   # (B, C, H, W) -- ReLU is elementwise, shape unchanged
        return x


class PedestrianHead(nn.Module):
    """Single-class, decoupled, aspect-ratio-aware detection head.

    forward(N3, N4, N5) -> List[Tensor], one tensor per input scale, each of
    shape (B, H_i * W_i, 5) where the last dim is (cx, cy, w, h, conf), with
    conf already passed through sigmoid and cx/cy/w/h already decoded to
    input-image pixel coordinates (not raw network offsets).
    """

    def __init__(
        self,
        in_channels: int = 256,
        feat_channels: int = 256,
        strides: Tuple[int, int, int] = (8, 16, 32),
        num_branch_convs: int = 2,
        prior_prob: float = 0.01,
        prior_w: float = 0.0461,    # median w at 640×384 pipeline (948 train labels, Step-1 measure)
        prior_h: float = 0.1680,    # median h at 640×384 pipeline (948 train labels, Step-1 measure)
    ):
        super().__init__()
        self.strides = strides
        self.prior_w = prior_w      # FIX: store prior_w instead of aspect_ratio_prior
        self.prior_h = prior_h      # FIX: store prior_h instead of aspect_ratio_prior

        # --- Shared stem: 1x1 conv to (re)project neck features into the
        # head's working channel width. Weight-shared across N3/N4/N5, the
        # same way YOLOX shares its decoupled-head weights across FPN levels
        # (scale differences are handled later by the stride-based decode,
        # not by having separate weights per level). ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, feat_channels),
            nn.ReLU(inplace=True),
        )
        _init_conv_kaiming_normal(self.stem[0], std=0.01)

        # --- Decoupled branches (Q3) ---
        self.cls_branch = nn.Sequential(
            *[ConvGNReLU(feat_channels) for _ in range(num_branch_convs)]
        )
        self.reg_branch = nn.Sequential(
            *[ConvGNReLU(feat_channels) for _ in range(num_branch_convs)]
        )

        # --- Prediction convs ---
        self.cls_pred = nn.Conv2d(feat_channels, 1, kernel_size=1)  # -> 1 channel: person confidence logit
        self.reg_pred = nn.Conv2d(feat_channels, 4, kernel_size=1)  # -> 4 channels: (t_cx, t_cy, t_w, t_h)

        # Per-level learnable regression scale (additive in log-space).
        # Without this, exp(t_w)*W*stride = 640px at EVERY level even after
        # scale routing — levels cannot specialise output magnitude.
        # Init to 0: exp(0)=1, no change at init. Training adjusts.
        self.reg_level_scale = nn.Parameter(torch.zeros(len(strides), 2))

        self._init_prediction_biases(prior_prob, prior_w, prior_h)

    def _init_prediction_biases(self, prior_prob: float, prior_w: float, prior_h: float) -> None:
        # cls_pred: weight std=0.01 (FCOS convention), bias = focal-loss prior.
        # Confirmed against FCOS source: bias_value = -log((1 - prior_prob) / prior_prob)
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

        # reg_pred: weight std=0.01 (FCOS convention — FCOS itself zeros the
        # bias here; we deviate on the bias only, per the Q2 extrapolation).
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)

        # FIX: Use absolute priors for w/h instead of aspect-ratio prior.
        # w_init = exp(log(prior_w)) = prior_w = 0.0461 (median w over 948 train labels,
        #          measured through the 640×384 val pipeline)
        # h_init = exp(log(prior_h)) = prior_h = 0.1680 (median h, same measurement)
        with torch.no_grad():
            # reg_pred.bias layout: [t_cx, t_cy, t_w, t_h]
            self.reg_pred.bias[0] = 0.0                 # t_cx: no prior, center offset starts at 0
            self.reg_pred.bias[1] = 0.0                 # t_cy: no prior, center offset starts at 0
            self.reg_pred.bias[2] = math.log(prior_w)   # FIX: t_w = log(0.0461) = -3.08 → exp = 0.0461
            self.reg_pred.bias[3] = math.log(prior_h)   # FIX: t_h = log(0.1680) = -1.78 → exp = 0.1680

    @staticmethod
    def _make_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Returns (H, W, 2) grid of (grid_x, grid_y) integer coordinates.
        ys = torch.arange(h, device=device, dtype=dtype)  # (H,)
        xs = torch.arange(w, device=device, dtype=dtype)  # (W,)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # each (H, W)
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2) -- last dim = (x, y)
        return grid

    def forward(
        self,
        N3: torch.Tensor,
        N4: torch.Tensor,
        N5: torch.Tensor,
        training_mode: bool = True,
    ) -> List[torch.Tensor]:
        """
        Args:
            N3: (B, 256, H/8,  W/8)
            N4: (B, 256, H/16, W/16)
            N5: (B, 256, H/32, W/32)
            training_mode: if True, return raw logits for loss computation;
                           if False, return decoded pixel coordinates for inference.

        Returns:
            List of 3 tensors (one per scale, same order as input):

            training_mode=True  — raw logits, no sigmoid, no decode:
              [ (B, H3*W3, 5), (B, H4*W4, 5), (B, H5*W5, 5) ]
              last dim = (t_cx, t_cy, t_w, t_h, conf_logit)
              t_cx/t_cy: raw offset logits (unbounded)
              t_w/t_h:   raw log-scale logits (unbounded)
              conf_logit: raw confidence logit (pre-sigmoid)

            training_mode=False — decoded pixel coords, conf sigmoided:
              [ (B, H3*W3, 5), (B, H4*W4, 5), (B, H5*W5, 5) ]
              last dim = (cx, cy, w, h, conf) in input-image pixel coordinates
        """
        features = [N3, N4, N5]
        outputs: List[torch.Tensor] = []

        for lvl, (feat, stride) in enumerate(zip(features, self.strides)):
            B, C, H, W = feat.shape  # e.g. (1, 256, 80, 80) for N3 at 640x640 input

            stem_out = self.stem(feat)  # (B, 256, H, W) -- 1x1 conv keeps spatial size

            cls_feat = self.cls_branch(stem_out)  # (B, 256, H, W)
            reg_feat = self.reg_branch(stem_out)  # (B, 256, H, W)

            conf_logit = self.cls_pred(cls_feat)  # (B, 1, H, W)
            reg_raw = self.reg_pred(reg_feat)      # (B, 4, H, W) -- channels: t_cx, t_cy, t_w, t_h

            t_cx = reg_raw[:, 0, :, :]  # (B, H, W)
            t_cy = reg_raw[:, 1, :, :]  # (B, H, W)
            t_w  = reg_raw[:, 2, :, :] + self.reg_level_scale[lvl, 0]  # (B, H, W) -- level-scaled log-w
            t_h  = reg_raw[:, 3, :, :] + self.reg_level_scale[lvl, 1]  # (B, H, W) -- level-scaled log-h

            if training_mode:
                # Return raw logits — losses.py applies sigmoid/exp internally.
                conf_2d = conf_logit[:, 0, :, :]           # (B, H, W) raw logit
                level_pred = torch.stack(
                    [t_cx, t_cy, t_w, t_h, conf_2d], dim=-1
                )                                           # (B, H, W, 5)
                level_pred = level_pred.reshape(B, H * W, 5)  # (B, H*W, 5)
            else:
                # Decode to input-image pixel coordinates (inference path).
                conf = torch.sigmoid(conf_logit)  # (B, 1, H, W) -- values in [0, 1]

                # FIX (bf16): build the grid in torch.float32 explicitly, NOT
                # feat.dtype.  Under AMP autocast, feat is bf16; a bf16 grid
                # quantises the integer cell coordinates and loses precision in
                # the (sigmoid + grid) * stride decode.  Float32 grid keeps the
                # integer cell offsets exact regardless of autocast state.
                grid = self._make_grid(H, W, feat.device, torch.float32)  # (H, W, 2) -- (x, y)
                grid_x = grid[..., 0].unsqueeze(0)  # (1, H, W)
                grid_y = grid[..., 1].unsqueeze(0)  # (1, H, W)

                # FIX Bug4: apply sigmoid to t_cx/t_cy to match losses.py training decode:
                #   cx_norm = (sigmoid(t_cx) + col) / S  →  cx_px = cx_norm * S * stride
                #   The +0.5 is removed: sigmoid(0)=0.5 already places the default
                #   prediction at the cell centre; raw t_cx is unbounded and must NOT
                #   be added directly (it can be ±5, blowing the center far outside the cell).
                # MUST match losses._decode_pred_boxes exactly.
                cx = (_OFF_S * torch.sigmoid(t_cx) - _OFF_B + grid_x) * stride  # (B, H, W) -- center x in pixels
                cy = (_OFF_S * torch.sigmoid(t_cy) - _OFF_B + grid_y) * stride  # (B, H, W) -- center y in pixels
                # FIX Bug5: losses.py decodes w_norm = exp(t_w) in [0,1] space;
                #   to convert to pixels: w_px = w_norm * img_w_px, h_px = h_norm * img_h_px.
                #   The original code used only * stride, giving w ~80x too small
                #   (e.g. exp(log(0.04)) * 8 = 0.32 px instead of 0.04 * 640 = 25.6 px).
                # FIX (640×384): the old code scaled BOTH w and h by H*stride (the image
                #   height).  w must be scaled by the image WIDTH (W*stride) — at non-square
                #   input, H*stride made inferred widths 384/640 = 0.6x too narrow.
                # Clamp raw logits before exp to prevent fp32 overflow (exp(89)=inf).
                # max=4.0 → exp(4)*640 ≈ 34560 px wide / exp(4)*384 ≈ 20736 px tall
                #   (huge but finite).  The NMS score threshold will discard degenerate
                #   boxes before they matter.
                img_w_px = float(W * stride)  # full image width in px at this scale
                img_h_px = float(H * stride)  # full image height in px at this scale
                w  = torch.exp(t_w.clamp(min=-8.0, max=4.0)) * img_w_px  # pixels
                h  = torch.exp(t_h.clamp(min=-8.0, max=4.0)) * img_h_px  # pixels

                conf_2d = conf[:, 0, :, :]  # (B, H, W)

                level_pred = torch.stack([cx, cy, w, h, conf_2d], dim=-1)  # (B, H, W, 5)
                level_pred = level_pred.reshape(B, H * W, 5)               # (B, H*W, 5)

            outputs.append(level_pred)

        return outputs  # [(B, H3*W3, 5), (B, H4*W4, 5), (B, H5*W5, 5)]
