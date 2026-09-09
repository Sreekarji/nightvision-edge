"""
head.py — PedestrianHead (NIRDet-Lite)
=======================================
Single-class, decoupled, weight-shared across levels.

Fixes relative to the original:
  * GroupNorm -> BatchNorm everywhere. GN cannot be folded into the preceding
    convolution (its statistics are computed at runtime), which cost most of
    the INT8 benefit under NCNN and had no hardware mapping at all on
    Neural-ART. At batch 8 there is no reason to prefer GN.
  * Branch convs are depthwise-separable, one per branch, at 64 channels
    (was two dense 3x3 at 256 -> 12.2 GMAC, 76% of the whole model).
  * strides cover three levels (8, 16, 32) including P5.
  * The exp() clamp is tightened to config's (-6, 1) for INT8 sanity.
  * The inference decode is unchanged in *semantics* and stays bit-for-bit
    aligned with losses._decode_boxes, live_nirdet.decode, and nirdet_pp.c.

Decode contract (identical in all four places)
----------------------------------------------
    cx_px = (OFF_S * sigmoid(t_cx) - OFF_B + col) * stride
    cy_px = (OFF_S * sigmoid(t_cy) - OFF_B + row) * stride
    w_px  = exp(clamp(t_w)) * img_w
    h_px  = exp(clamp(t_h)) * img_h

Note img_w = W * stride and img_h = H * stride are level-independent, so w/h
are decoded against the full image extent at every level.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from config import (
    DECODE_OFFSET_BIAS as _OFF_B,
    DECODE_OFFSET_SCALE as _OFF_S,
    REG_LOG_CLAMP_MAX,
    REG_LOG_CLAMP_MIN,
)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1,
                 p: int = 1, groups: int = 1, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=p, groups=groups, bias=False,
                              padding_mode="zeros")
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWSBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, both BN + ReLU6."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.dw = ConvBNAct(ch, ch, k=3, s=1, p=1, groups=ch)
        self.pw = ConvBNAct(ch, ch, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class PedestrianHead(nn.Module):
    """
    forward(feats, training_mode) -> list of (B, H_i*W_i, 5) tensors.

      training_mode=True : (t_cx, t_cy, t_w, t_h, conf_logit)  raw
      training_mode=False: (cx, cy, w, h, conf)  pixels + sigmoid
    """

    def __init__(
        self,
        in_channels: int = 64,
        feat_channels: int = 64,
        strides: Tuple[int, ...] = (8, 16, 32),
        num_branch_convs: int = 1,
        use_stem: bool = True,
        prior_prob: float = 0.01,
        prior_w: float = 0.0461,
        prior_h: float = 0.1680,
    ) -> None:
        super().__init__()
        self.strides = tuple(int(s) for s in strides)
        self.n_levels = len(self.strides)
        self.feat_channels = int(feat_channels)
        self.prior_w = float(prior_w)
        self.prior_h = float(prior_h)

        if use_stem or in_channels != feat_channels:
            self.stem: nn.Module = ConvBNAct(in_channels, feat_channels,
                                             k=1, s=1, p=0)
        else:
            self.stem = nn.Identity()

        self.cls_branch = nn.Sequential(
            *[DWSBlock(feat_channels) for _ in range(num_branch_convs)]
        )
        self.reg_branch = nn.Sequential(
            *[DWSBlock(feat_channels) for _ in range(num_branch_convs)]
        )

        self.cls_pred = nn.Conv2d(feat_channels, 1, kernel_size=1)
        self.reg_pred = nn.Conv2d(feat_channels, 4, kernel_size=1)

        # Per-level additive log-space scale: without it every level decodes
        # w/h against the same image extent and cannot specialise magnitude.
        self.reg_level_scale = nn.Parameter(torch.zeros(self.n_levels, 2))

        # Channel-selector buffers for export-safe level scale application.
        self.register_buffer("_w_sel",
            torch.tensor([0., 0., 1., 0.]).view(1, 4, 1, 1))
        self.register_buffer("_h_sel",
            torch.tensor([0., 0., 0., 1.]).view(1, 4, 1, 1))

        self._init_predictions(prior_prob, prior_w, prior_h)

    # ------------------------------------------------------------------ #

    def _init_predictions(self, prior_prob: float,
                          prior_w: float, prior_h: float) -> None:
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.constant_(self.cls_pred.bias,
                          -math.log((1.0 - prior_prob) / prior_prob))

        nn.init.normal_(self.reg_pred.weight, std=0.01)
        with torch.no_grad():
            self.reg_pred.bias.zero_()
            self.reg_pred.bias[2] = math.log(prior_w)   # exp -> prior_w
            self.reg_pred.bias[3] = math.log(prior_h)

    @staticmethod
    def _grid(h: int, w: int, device: torch.device
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cell indices as float32 (never autocast dtype: bf16 quantises them)."""
        ys = torch.arange(h, device=device, dtype=torch.float32)
        xs = torch.arange(w, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return gx.unsqueeze(0), gy.unsqueeze(0)        # (1, H, W) each

    # ------------------------------------------------------------------ #

    def forward_raw(self, feats: List[torch.Tensor]
                    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns per level (cls_logit (B,1,H,W), reg_raw (B,4,H,W)) with the
        per-level log scale already folded into reg_raw. This is the entry
        point used by export_onnx.py: NCHW convolution outputs only, no
        grid construction, no exp, no reshape.
        """
        if len(feats) != self.n_levels:
            raise ValueError(f"expected {self.n_levels} feature maps, "
                             f"got {len(feats)}")
        out = []
        for lvl, feat in enumerate(feats):
            s = self.stem(feat)
            cls_logit = self.cls_pred(self.cls_branch(s))
            reg_raw = self.reg_pred(self.reg_branch(s))
            # Pure Mul+Add — both in ALLOWED_OPS, no Gather/ScatterND in the trace.
            sw = self.reg_level_scale[lvl, 0].float()
            sh = self.reg_level_scale[lvl, 1].float()
            reg = reg_raw + self._w_sel.to(reg_raw.device) * sw \
                  + self._h_sel.to(reg_raw.device) * sh
            out.append((cls_logit, reg))
        return out

    def forward(self, feats: List[torch.Tensor],
                training_mode: bool = True) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []

        for lvl, (cls_logit, reg_raw) in enumerate(self.forward_raw(feats)):
            b, _, h, w = cls_logit.shape
            stride = self.strides[lvl]

            t_cx = reg_raw[:, 0]
            t_cy = reg_raw[:, 1]
            t_w = reg_raw[:, 2]
            t_h = reg_raw[:, 3]
            conf = cls_logit[:, 0]

            if training_mode:
                level = torch.stack([t_cx, t_cy, t_w, t_h, conf], dim=-1)
            else:
                gx, gy = self._grid(h, w, cls_logit.device)
                img_w = float(w * stride)
                img_h = float(h * stride)

                cx = (_OFF_S * torch.sigmoid(t_cx.float()) - _OFF_B + gx) * stride
                cy = (_OFF_S * torch.sigmoid(t_cy.float()) - _OFF_B + gy) * stride
                bw = torch.exp(t_w.float().clamp(REG_LOG_CLAMP_MIN,
                                                 REG_LOG_CLAMP_MAX)) * img_w
                bh = torch.exp(t_h.float().clamp(REG_LOG_CLAMP_MIN,
                                                 REG_LOG_CLAMP_MAX)) * img_h
                level = torch.stack(
                    [cx, cy, bw, bh, torch.sigmoid(conf.float())], dim=-1
                )

            outputs.append(level.reshape(b, h * w, 5))

        return outputs


if __name__ == "__main__":
    head = PedestrianHead().eval()
    f = [torch.zeros(1, 64, 48, 80), torch.zeros(1, 64, 24, 40), torch.zeros(1, 64, 12, 20)]
    raw = head(f, training_mode=True)
    dec = head(f, training_mode=False)
    print([tuple(t.shape) for t in raw])
    print("w mean px", float(dec[0][..., 2].mean()),
          "(expect ~", 0.0461 * 640, ")")
    print("h mean px", float(dec[0][..., 3].mean()),
          "(expect ~", 0.1680 * 384, ")")
    print("params", f"{sum(p.numel() for p in head.parameters()):,}")
