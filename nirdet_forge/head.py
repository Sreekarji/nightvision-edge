"""
head.py — PedestrianHead (NIRDet-Lite)
=======================================
Single-class (person), decoupled classification / regression, branch weights
shared across levels with PER-LEVEL BatchNorm, and the regression output SPLIT
into a 2-channel offset blob and a 2-channel size blob.

THREE BLOBS PER LEVEL: cls (1ch), off (2ch), size (2ch)
-------------------------------------------------------
The old head emitted reg (4ch) = (t_cx, t_cy, t_w, t_h) as one tensor, so in
INT8 QDQ all four channels shared one quantisation scale. But t_cx/t_cy are
unbounded (practically +/-6 after training) while t_w/t_h are clamped to
[-6, 1]. The union forces the scale to cover [-6, 6]:

    shared scale : 12.0 / 255 = 0.047 per LSB in log space -> 4.7% width error
    split scale  :  7.0 / 255 = 0.027 per LSB               -> 2.7% width error

On a 29 px pedestrian at stride 8 that is the difference between +/-1.4 px and
+/-0.8 px of width per quantisation step. Splitting the conv costs nothing in
fp32 and nothing in parameters, and the two blobs get independent scale /
zero-point pairs from the ONNX quantiser. nirdet_pp.c consumes three blobs per
level with three scale/zp pairs.

PER-LEVEL BATCHNORM
-------------------
Branch conv WEIGHTS are shared across levels — that is deliberate FCOS/YOLOX
practice and the reason the head is cheap. The BatchNorm statistics were also
shared, which is not defensible: a stride-8 activation distribution and a
stride-32 activation distribution differ materially and one set of running
statistics fits neither. Each shared nn.Conv2d now has an nn.ModuleList of
BatchNorm2d, one entry per level. Cost: 2 * 64 * 3 * 4 bytes of parameters
plus the same again in buffers, about 1 KB.

reg_level_scale IS GONE
-----------------------
It existed only because reg_pred was shared across levels, and it was applied
through two registered selector buffers plus four broadcast ops per level —
four extra ONNX nodes per level, and the .to(device) calls on already-
registered buffers were no-ops that could emit stray Cast/Identity nodes.
With per-level off_pred / size_pred the per-level scale folds into each conv's
own bias for free.

PER-LEVEL STEMS
---------------
The neck no longer emits a uniform width (N3 is 48, N4/N5 are 64), so the 1x1
stem is per-level. That is the correct place for the reprojection anyway: it is
the cheapest layer in the head.

Decode contract (identical in config.py, losses.py, live_nirdet.py, nirdet_pp.c)
--------------------------------------------------------------------------------
    cx_px = (OFF_S * sigmoid(t_cx) - OFF_B + col) * stride
    cy_px = (OFF_S * sigmoid(t_cy) - OFF_B + row) * stride
    w_px  = exp(clamp(t_w, MIN, MAX)) * img_w
    h_px  = exp(clamp(t_h, MIN, MAX)) * img_h

img_w = W * stride and img_h = H * stride are level-independent, so w/h decode
against the full image extent at every level.
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
    """
    One shared 3x3 convolution, one BatchNorm PER LEVEL, ReLU6.

    forward(x, level) selects the BN. The convolution weight is a single
    tensor, so the parameter count is unchanged from a shared ConvBNAct; only
    the normalisation is specialised.

    At export every level is traced separately, so each level's BN folds into
    a private copy of the shared conv weights. That is the intended outcome:
    three folded convolutions with identical weights but level-appropriate
    scale and shift, all of them plain HW-mapped Conv nodes.
    """

    def __init__(self, ch: int, n_levels: int, k: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=k, stride=1,
                              padding=k // 2, bias=False,
                              padding_mode="zeros")
        self.bns = nn.ModuleList([nn.BatchNorm2d(ch) for _ in range(n_levels)])
        self.act = nn.ReLU6(inplace=True)
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor, level: int) -> torch.Tensor:
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


class PedestrianHead(nn.Module):
    """
    forward_raw(feats) -> [(cls, off, size), ...]      NCHW, the exported graph
    forward(feats, training_mode)
        training_mode=True  -> [(B, H*W, 5)]  raw (t_cx, t_cy, t_w, t_h, logit)
        training_mode=False -> [(B, H*W, 5)]  decoded (cx, cy, w, h, conf) px

    The (B, H*W, 5) packing order is (t_cx, t_cy, t_w, t_h, conf), which is
    what losses.py indexes. Splitting the convolution does not change that
    contract: off and size are concatenated back in the same order.
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
        prior_w: float = 0.046094,
        prior_h: float = 0.179167,
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
                             f"{self.n_levels} levels — need at least one per level")
        in_chs = in_chs[:self.n_levels]
        self.in_channels = tuple(in_chs)

        # ---- per-level 1x1 stems ----
        # Per-level because the neck emits three different widths (N3 is
        # narrowed to 48). A 1x1 is the cheapest layer here, so this is the
        # right place for the reprojection.
        stems: List[nn.Module] = []
        for c in self.in_channels:
            if use_stem or c != self.feat_channels:
                stems.append(ConvBNAct(c, self.feat_channels, k=1, s=1, p=0))
            else:
                stems.append(nn.Identity())
        self.stems = nn.ModuleList(stems)

        # ---- shared branch weights, per-level BN ----
        n_bn = self.n_levels if self.per_level_bn else 1
        self.cls_branch = SharedBranch(self.feat_channels, num_branch_convs, n_bn)
        self.reg_branch = SharedBranch(self.feat_channels, num_branch_convs, n_bn)

        # ---- per-level prediction convs ----
        # Single class: cls_pred emits exactly NUM_CLASSES == 1 channel.
        # off_pred and size_pred are separate so the INT8 quantiser assigns
        # them independent scales. Per-level so the old reg_level_scale folds
        # into each conv's bias.
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
    # init
    # ------------------------------------------------------------------ #

    def _init_predictions(self, prior_prob: float,
                          prior_w: float, prior_h: float) -> None:
        cls_bias = -math.log((1.0 - prior_prob) / prior_prob)
        log_w = math.log(prior_w)
        log_h = math.log(prior_h)
        if not (REG_LOG_CLAMP_MIN < log_w < REG_LOG_CLAMP_MAX):
            raise ValueError(
                f"log(prior_w)={log_w:.3f} outside the reg clamp "
                f"({REG_LOG_CLAMP_MIN}, {REG_LOG_CLAMP_MAX}); the size bias "
                f"would be clipped and TAL would cold-start with zero "
                f"positives")
        if not (REG_LOG_CLAMP_MIN < log_h < REG_LOG_CLAMP_MAX):
            raise ValueError(
                f"log(prior_h)={log_h:.3f} outside the reg clamp "
                f"({REG_LOG_CLAMP_MIN}, {REG_LOG_CLAMP_MAX})")

        with torch.no_grad():
            for lvl in range(self.n_levels):
                nn.init.normal_(self.cls_pred[lvl].weight, std=0.01)
                nn.init.constant_(self.cls_pred[lvl].bias, cls_bias)

                nn.init.normal_(self.off_pred[lvl].weight, std=0.01)
                self.off_pred[lvl].bias.zero_()      # sigmoid(0) -> cell centre

                nn.init.normal_(self.size_pred[lvl].weight, std=0.01)
                # exp(bias) == prior. THIS INITIALISATION IS LOAD-BEARING:
                # it is the only reason Task-Aligned Assignment finds any
                # positive at epoch 0 (the predicted boxes must overlap the
                # ground truth enough for iou^tal_beta to be non-negligible).
                # NIRDetLoss asserts n_pos > 0 on the first batch so a change
                # here cannot silently produce a classification-only run.
                self.size_pred[lvl].bias[0] = log_w
                self.size_pred[lvl].bias[1] = log_h

    # ------------------------------------------------------------------ #
    # grid
    # ------------------------------------------------------------------ #

    @staticmethod
    def _grid(h: int, w: int, device: torch.device
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cell indices as float32, NEVER the autocast dtype.

        bf16 carries 8 bits of significand, so integers are exactly
        representable only up to 2^8 = 256. A column index of 257 rounds to
        256 in bf16, which would shift a decoded box by a full stride. This
        function therefore pins dtype=torch.float32 regardless of any
        surrounding autocast context.

        The dangerous configuration is a canvas wider than
        stride_min * 256 = 8 * 256 = 2048 px, at which point the grid would
        exceed 256 columns. Current canvas: 512 wide -> 64 columns at stride 8,
        with 4x headroom. Native 1280 capture -> 160 columns, also safe. The
        check below fails loudly rather than silently mis-decoding, because a
        half-stride box shift is very hard to attribute from a mAP number.
        """
        if w > BF16_EXACT_INT_MAX:
            raise RuntimeError(
                f"grid has {w} columns, above the bf16 exact-integer limit of "
                f"{BF16_EXACT_INT_MAX}. Cell indices are built in float32 "
                f"here so this forward pass is still correct, but any code "
                f"path that lets the grid inherit a bf16 autocast dtype would "
                f"quantise column indices and shift decoded boxes by up to a "
                f"full stride. Canvas width must stay <= "
                f"{BF16_EXACT_INT_MAX * 8} px at stride 8.")
        if h > BF16_EXACT_INT_MAX:
            raise RuntimeError(
                f"grid has {h} rows, above the bf16 exact-integer limit of "
                f"{BF16_EXACT_INT_MAX}")
        ys = torch.arange(h, device=device, dtype=torch.float32)
        xs = torch.arange(w, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return gx.unsqueeze(0), gy.unsqueeze(0)        # (1, H, W) each

    # ------------------------------------------------------------------ #
    # raw forward — this is what export_onnx.py traces
    # ------------------------------------------------------------------ #

    def forward_raw(self, feats: List[torch.Tensor]
                    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns per level (cls_logit (B,1,H,W), off_raw (B,2,H,W),
        size_raw (B,2,H,W)).

        NCHW convolution outputs only: no grid construction, no sigmoid, no
        exp, no clamp, no reshape, no concat. Everything after this is host
        post-processing (live_nirdet.py on the Pi, nirdet_pp.c on the M55),
        which is also what ST requires — they document detection
        post-processing including NMS as a HOST responsibility with no NPU
        support.
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
            s = self.stems[lvl](feat)
            bn_idx = lvl if self.per_level_bn else 0
            cls_logit = self.cls_pred[lvl](self.cls_branch(s, bn_idx))
            reg_feat = self.reg_branch(s, bn_idx)
            off_raw = self.off_pred[lvl](reg_feat)
            size_raw = self.size_pred[lvl](reg_feat)
            out.append((cls_logit, off_raw, size_raw))
        return out

    # ------------------------------------------------------------------ #
    # packed forward — training and host-side inference
    # ------------------------------------------------------------------ #

    def forward(self, feats: List[torch.Tensor],
                training_mode: bool = True) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []

        for lvl, (cls_logit, off_raw, size_raw) in enumerate(
                self.forward_raw(feats)):
            b, _, h, w = cls_logit.shape
            stride = self.strides[lvl]

            t_cx = off_raw[:, 0]
            t_cy = off_raw[:, 1]
            t_w = size_raw[:, 0]
            t_h = size_raw[:, 1]
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

    # ------------------------------------------------------------------ #

    def output_names(self) -> List[str]:
        """ONNX output blob names, in forward_raw order. Three per level."""
        names: List[str] = []
        for s in self.strides:
            names += [f"cls{s}", f"off{s}", f"size{s}"]
        return names


if __name__ == "__main__":
    head = PedestrianHead(in_channels=(48, 64, 64), strides=(8, 16, 32)).eval()
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

    packed = head(f, training_mode=True)
    dec = head(f, training_mode=False)
    print("packed:", [tuple(t.shape) for t in packed],
          " (expect (1,2304,5) (1,576,5) (1,144,5))")

    pw, ph = 0.046094, 0.179167
    print(f"w mean px {float(dec[0][..., 2].mean()):.2f} "
          f"(expect ~{pw * 512:.2f})")
    print(f"h mean px {float(dec[0][..., 3].mean()):.2f} "
          f"(expect ~{ph * 288:.2f})")

    print("has reg_level_scale:",
          any("reg_level_scale" in n for n, _ in head.named_parameters()))
    print("has selector buffers:",
          any(n.endswith("_w_sel") or n.endswith("_h_sel")
              for n, _ in head.named_buffers()))
    print("params", f"{sum(p.numel() for p in head.parameters()):,}")

    # bf16 grid guard
    try:
        PedestrianHead._grid(36, 300, torch.device("cpu"))
        print("grid guard: FAILED to trip")
    except RuntimeError as e:
        print("grid guard tripped as expected at 300 columns")
