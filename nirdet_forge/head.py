"""
head.py — PedestrianHead (NIRDet-Lite)
=======================================
Single-class, decoupled cls/reg, branch weights shared across levels with
PER-LEVEL BatchNorm, and the regression output SPLIT into a 2-channel offset
blob and a 2-channel size blob so INT8 gives them independent scales
(t_cx/t_cy are unbounded; t_w/t_h are clamped to [-6, 1]).

Decode contract (identical in config.py, losses.py, live_nirdet.py,
evaluate_onnx.py, nirdet_pp.c):
    cx_px = (OFF_S * sigmoid(t_cx) - OFF_B + col) * stride
    cy_px = (OFF_S * sigmoid(t_cy) - OFF_B + row) * stride
    w_px  = exp(clamp(t_w, MIN, MAX)) * img_w
    h_px  = exp(clamp(t_h, MIN, MAX)) * img_h
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn

from config import (
    BF16_EXACT_INT_MAX,
    DECODE_OFFSET_BIAS as _OFF_B,
    DECODE_OFFSET_SCALE as _OFF_S,
    NUM_CLASSES,
    REG_LOG_CLAMP_MAX,
    REG_LOG_CLAMP_MIN,
)


class SharedConvPerLevelBN(nn.Module):
    """One shared 3x3 convolution, one BatchNorm PER LEVEL, ReLU6.

    At export each level is traced separately, so each level's BN folds into
    a private copy of the shared weights: three plain HW-mapped Conv nodes.
    """

    def __init__(self, ch: int, n_levels: int, k: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=k, stride=1,
                              padding=k // 2, bias=False, padding_mode="zeros")
        self.bns = nn.ModuleList([nn.BatchNorm2d(ch) for _ in range(n_levels)])
        self.act = nn.ReLU6(inplace=True)
        # Kaiming-for-ReLU slightly over-scales for ReLU6 (which clips the
        # tail), but the following BatchNorm absorbs it exactly (F54).
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor, level: int) -> torch.Tensor:
        if not (0 <= level < len(self.bns)):
            raise IndexError(
                f"level {level} out of range for {len(self.bns)} per-level "
                f"BatchNorms (per_level_bn misconfigured)")      # F53
        return self.act(self.bns[level](self.conv(x)))


class SharedBranch(nn.Module):
    """A stack of SharedConvPerLevelBN blocks."""

    def __init__(self, ch: int, n_convs: int, n_levels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [SharedConvPerLevelBN(ch, n_levels) for _ in range(max(0, n_convs))]
        )

    def forward(self, x: torch.Tensor, level: int) -> torch.Tensor:
        for b in self.blocks:
            x = b(x, level)
        return x


class ConvBNAct(nn.Module):
    """Plain Conv->BN->ReLU6, used for the per-level 1x1 stems."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1,
                 p: int = 0, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=p, bias=False, padding_mode="zeros")
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# F55: dedupe set for the in_channels truncation NOTE — one print per
# distinct message per process, not one per model construction.
_TRUNCATION_WARNED: set = set()


class PedestrianHead(nn.Module):
    """
    forward_raw(feats) -> [(cls, off, size), ...]   NCHW, the exported graph
    forward(feats, training_mode)                   -> [(B, H*W, 5)]
        training_mode=True  -> raw (t_cx, t_cy, t_w, t_h, logit)
        training_mode=False -> decoded (cx, cy, w, h, conf) in pixels
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]] = (48, 64, 64),
        feat_channels: int = 64,
        strides: Tuple[int, ...] = (8, 16, 32),
        num_branch_convs: int = 1,
        use_stem: bool = True,
        per_level_bn: bool = True,
        prior_prob: float = 0.01,
        *,
        # REQUIRED: canvas-space dataset measurements from
        # DatasetProfile.apply() -> cfg.model.prior_w / prior_h. size_pred.bias
        # is seeded from them and that seed is the only reason TAL finds a
        # positive at epoch 0. Keyword-only so they cannot be passed by
        # accident.
        prior_w: float,
        prior_h: float,
    ) -> None:
        super().__init__()
        self.strides = tuple(int(s) for s in strides)
        self.n_levels = len(self.strides)
        self.feat_channels = int(feat_channels)
        self.per_level_bn = bool(per_level_bn)
        self.prior_w = float(prior_w)
        self.prior_h = float(prior_h)

        if isinstance(in_channels, int):
            in_chs = [int(in_channels)] * self.n_levels
        else:
            in_chs = [int(c) for c in in_channels]
        if len(in_chs) < self.n_levels:
            raise ValueError(f"in_channels has {len(in_chs)} entries for "
                             f"{self.n_levels} levels")
        if len(in_chs) > self.n_levels:
            # Operator-facing message. F55: printed ONCE per distinct message
            # per process — warnings.warn collapses per location, and a bare
            # print repeats for every model built after the first.
            _msg = (f"[head] NOTE received {len(in_chs)} in_channels for "
                    f"{self.n_levels} levels; truncating to {self.n_levels}. "
                    f"Expected under --p5-ablate; otherwise check the neck config.")
            if _msg not in _TRUNCATION_WARNED:
                print(_msg)
                _TRUNCATION_WARNED.add(_msg)
        in_chs = in_chs[:self.n_levels]
        self.in_channels = tuple(in_chs)

        # Per-level 1x1 stems: the neck emits three different widths.
        stems: List[nn.Module] = []
        for c in self.in_channels:
            if use_stem or c != self.feat_channels:
                stems.append(ConvBNAct(c, self.feat_channels, k=1, s=1, p=0))
            else:
                stems.append(nn.Identity())
        self.stems = nn.ModuleList(stems)

        n_bn = self.n_levels if self.per_level_bn else 1
        self.cls_branch = SharedBranch(self.feat_channels, num_branch_convs, n_bn)
        self.reg_branch = SharedBranch(self.feat_channels, num_branch_convs, n_bn)

        self.cls_pred = nn.ModuleList([
            nn.Conv2d(self.feat_channels, NUM_CLASSES, kernel_size=1)
            for _ in range(self.n_levels)])
        self.off_pred = nn.ModuleList([
            nn.Conv2d(self.feat_channels, 2, kernel_size=1)
            for _ in range(self.n_levels)])
        self.size_pred = nn.ModuleList([
            nn.Conv2d(self.feat_channels, 2, kernel_size=1)
            for _ in range(self.n_levels)])

        self._init_predictions(prior_prob, prior_w, prior_h)

    # ------------------------------------------------------------------ #

    def _init_predictions(self, prior_prob: float,
                          prior_w: float, prior_h: float) -> None:
        cls_bias = -math.log((1.0 - prior_prob) / prior_prob)
        if not (0.0 < prior_w < 1.0 and 0.0 < prior_h < 1.0):
            raise ValueError(f"prior_w/prior_h must be in (0,1), got "
                             f"{prior_w}/{prior_h}")
        log_w = math.log(prior_w)
        log_h = math.log(prior_h)
        for label, lg in (("prior_w", log_w), ("prior_h", log_h)):
            if not (REG_LOG_CLAMP_MIN < lg < REG_LOG_CLAMP_MAX):
                raise ValueError(
                    f"log({label})={lg:.3f} outside the reg clamp "
                    f"({REG_LOG_CLAMP_MIN}, {REG_LOG_CLAMP_MAX}); the size "
                    f"bias would be clipped and TAL would cold-start with "
                    f"zero positives")

        with torch.no_grad():
            for lvl in range(self.n_levels):
                nn.init.normal_(self.cls_pred[lvl].weight, std=0.01)
                nn.init.constant_(self.cls_pred[lvl].bias, cls_bias)
                nn.init.normal_(self.off_pred[lvl].weight, std=0.01)
                self.off_pred[lvl].bias.zero_()      # sigmoid(0) -> cell centre
                nn.init.normal_(self.size_pred[lvl].weight, std=0.01)
                # LOAD-BEARING: the only reason TAL finds a positive at epoch 0.
                self.size_pred[lvl].bias[0] = log_w
                self.size_pred[lvl].bias[1] = log_h

    # ------------------------------------------------------------------ #

    _GRID_CACHE: dict = {}

    @classmethod
    def _grid(cls, h: int, w: int, device: torch.device
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cell indices as float32, NEVER the autocast dtype. Cached per
        (h, w, device) (F53).

        bf16 has 8 significand bits, so integers are exact only to 256. A
        column index of 257 would round to 256 and shift a decoded box by a
        full stride.
        """
        if w > BF16_EXACT_INT_MAX:
            raise RuntimeError(
                f"grid has {w} columns, above the bf16 exact-integer limit of "
                f"{BF16_EXACT_INT_MAX}. Canvas width must stay <= "
                f"{BF16_EXACT_INT_MAX * 8} px at stride 8.")
        if h > BF16_EXACT_INT_MAX:
            raise RuntimeError(f"grid has {h} rows, above {BF16_EXACT_INT_MAX}")
        key = (int(h), int(w), str(device))
        hit = cls._GRID_CACHE.get(key)
        if hit is None:
            ys = torch.arange(h, device=device, dtype=torch.float32)
            xs = torch.arange(w, device=device, dtype=torch.float32)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            hit = (gx.unsqueeze(0), gy.unsqueeze(0))        # (1, H, W) each
            cls._GRID_CACHE[key] = hit
        return hit

    # ------------------------------------------------------------------ #

    def forward_raw(self, feats: List[torch.Tensor]
                    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Per level (cls (B,1,H,W), off (B,2,H,W), size (B,2,H,W)).

        NCHW convolution outputs only: no grid, sigmoid, exp, clamp, reshape
        or concat. Everything after this is host post-processing.
        """
        if len(feats) != self.n_levels:
            raise ValueError(f"expected {self.n_levels} feature maps, "
                             f"got {len(feats)}")
        out = []
        for lvl, feat in enumerate(feats):
            if feat.shape[1] != self.in_channels[lvl]:
                raise ValueError(
                    f"level {lvl} feature has {feat.shape[1]} channels, head "
                    f"stem expects {self.in_channels[lvl]}")
            # F56: spatial halving contract between consecutive levels.
            if lvl > 0:
                prev_h, prev_w = out[-1][0].shape[-2], out[-1][0].shape[-1]
                r = self.strides[lvl] // self.strides[lvl - 1]
                exp_h, exp_w = feat.shape[-2] * r, feat.shape[-1] * r
                if (exp_h, exp_w) != (prev_h, prev_w):
                    raise ValueError(
                        f"level {lvl} feature is {feat.shape[-2]}×{feat.shape[-1]}, "
                        f"expected half of level {lvl-1} "
                        f"({prev_h}×{prev_w} → {exp_h}×{exp_w}). "
                        f"Feature pyramid strides do not match self.strides={self.strides}.")
            s = self.stems[lvl](feat)
            bn_idx = lvl if self.per_level_bn else 0
            cls_logit = self.cls_pred[lvl](self.cls_branch(s, bn_idx))
            reg_feat = self.reg_branch(s, bn_idx)
            out.append((cls_logit, self.off_pred[lvl](reg_feat),
                        self.size_pred[lvl](reg_feat)))
        return out

    def forward(self, feats: List[torch.Tensor],
                training_mode: bool = True) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for lvl, (cls_logit, off_raw, size_raw) in enumerate(
                self.forward_raw(feats)):
            b, _, h, w = cls_logit.shape
            stride = self.strides[lvl]
            t_cx, t_cy = off_raw[:, 0], off_raw[:, 1]
            t_w, t_h = size_raw[:, 0], size_raw[:, 1]
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
                    [cx, cy, bw, bh, torch.sigmoid(conf.float())], dim=-1)
            outputs.append(level.reshape(b, h * w, 5))
        return outputs

    def output_names(self) -> List[str]:
        """ONNX output blob names, in forward_raw order. Three per level."""
        names: List[str] = []
        for s in self.strides:
            names += [f"cls{s}", f"off{s}", f"size{s}"]
        return names


if __name__ == "__main__":
    # SYNTHETIC priors — deliberately NOT any dataset's measurement (F55).
    PW, PH = 0.05, 0.15
    head = PedestrianHead(in_channels=(48, 64, 64), strides=(8, 16, 32),
                          prior_w=PW, prior_h=PH).eval()
    f = [torch.zeros(1, 48, 36, 64),
         torch.zeros(1, 64, 18, 32),
         torch.zeros(1, 64, 9, 16)]

    raw = head.forward_raw(f)
    print("blobs per level:", len(raw[0]), "(must be 3)")
    assert len(raw[0]) == 3
    for lvl, (c, o, s) in enumerate(raw):
        print(f"  L{lvl}: cls {tuple(c.shape)} off {tuple(o.shape)} "
              f"size {tuple(s.shape)}")
    print("output names:", head.output_names())

    dec = head(f, training_mode=False)
    print(f"w mean px {float(dec[0][..., 2].mean()):.2f} (expect ~{PW * 512:.2f})")
    print(f"h mean px {float(dec[0][..., 3].mean()):.2f} (expect ~{PH * 288:.2f})")
    print("has reg_level_scale:",
          any("reg_level_scale" in n for n, _ in head.named_parameters()))
    print("params", f"{sum(p.numel() for p in head.parameters()):,}")
    try:
        PedestrianHead._grid(36, 300, torch.device("cpu"))
        print("grid guard: FAILED to trip")
    except RuntimeError:
        print("grid guard tripped as expected at 300 columns")
