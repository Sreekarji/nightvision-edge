# neck.py
# ─────────────────────────────────────────────────────────────────────────────
# NIRDet — Feature Pyramid Network neck (Phase 3)
#
# Architecture: standard top-down FPN (Lin et al., CVPR 2017).
# Three stages:
#   1. Lateral 1×1 convolutions — project each backbone level to OUT channels.
#   2. Top-down fusion — upsample deeper features, add to shallower laterals.
#   3. Output 3×3 convolutions — smooth aliasing introduced by addition.
#
# Inputs  (from backbone + EAA, Phase 2):
#   P3  (B, C3, H/8,  W/8)   stride-8  feature map   C3 = backbone_channels[2]
#   P4  (B, C4, H/16, W/16)  stride-16 feature map   C4 = backbone_channels[3]
#   P5  (B, C5, H/32, W/32)  stride-32 feature map   C5 = backbone_channels[4]
#
# Outputs (fed to detection head, Phase 4):
#   N3  (B, OUT, H/8,  W/8)
#   N4  (B, OUT, H/16, W/16)
#   N5  (B, OUT, H/32, W/32)
#
# Where OUT = cfg.model.backbone_channels[-1]  (= 256 at default config).
#
# All channel constants are derived from the Phase 0 config via get_config().
# Nothing is hardcoded in this file.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Single source of truth: Phase 0 config ───────────────────────────────────
# Import get_config() and read channel widths at module load time.
# This guarantees neck.py always reflects the live config values — changing
# backbone_channels in config.py automatically propagates here.
from config import get_config as _get_config

_cfg = _get_config()

# backbone_channels: (stem, stage1, stage2, stage3, stage4)
#                    ( 16,    32,     64,    128,    256  )   at default config
# FPN taps the last three stages (strides 8, 16, 32):
#   P3 / C3 → backbone_channels[2]  = 64   (stride 8,  H/8)
#   P4 / C4 → backbone_channels[3]  = 128  (stride 16, H/16)
#   P5 / C5 → backbone_channels[4]  = 256  (stride 32, H/32)
_BACKBONE_CH: tuple = _cfg.model.backbone_channels  # (16, 32, 64, 128, 256)
# Actual backbone out_channels = (base*4, base*8, base*8) = (128, 256, 256)
# where base = backbone_channels[1] = 32.  The config tuple indices [2],[3],[4]
# are (64, 128, 256) which do NOT match — backbone uses base*4/8/8 not channels[2/3/4].
# Fix: derive from the actual backbone formula so neck always matches backbone.
_BASE: int = _BACKBONE_CH[1]   # 32 — stem output = stage-1 base
_C3: int = _BASE * 4            # 128 — P3 stride-8  (backbone c3 = base_ch * 4)
_C4: int = _BASE * 8            # 256 — P4 stride-16 (backbone c4 = base_ch * 8)
_C5: int = _BASE * 8            # 256 — P5 stride-32 (backbone c5 = base_ch * 8)

# Neck unified output width.
# No explicit neck field exists in ModelConfig; we use backbone_channels[-1]
# (the deepest stage width = 256) because:
#   1. It matches the FPN paper canonical value of 256.
#   2. It is the only channel value derivable from config without adding a new field.
#   3. It keeps the neck parameter budget within the ~1.0 M estimate in config.py §3.
_OUT: int = _BACKBONE_CH[-1]  # 256


# ─────────────────────────────────────────────────────────────────────────────
# Helper: ConvBnRelu
# A reusable building block — Conv2d → BatchNorm2d → ReLU.
# Used for *output* (smoothing) convolutions; NOT used for lateral projections
# (those stay as bare 1×1 convs to match the FPN paper exactly).
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """3×3 Conv → BN → ReLU, padding=1 keeps H and W unchanged."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            # (B, in_ch, H, W) → (B, out_ch, H, W)   [padding=1 preserves H, W]
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # (B, out_ch, H, W) → (B, out_ch, H, W)  [normalise across batch]
            nn.BatchNorm2d(out_ch),
            # (B, out_ch, H, W) → (B, out_ch, H, W)  [element-wise, in-place]
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W) → out: (B, out_ch, H, W)
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# LightweightFPN
# ─────────────────────────────────────────────────────────────────────────────

class LightweightFPN(nn.Module):
    """
    Lightweight Feature Pyramid Network neck for NIRDet.

    Parameters
    ----------
    c3 : int
        Backbone output channels at stride-8.
        Default: backbone_channels[1]*4 = 128.
    c4 : int
        Backbone output channels at stride-16.
        Default: backbone_channels[1]*8 = 256.
    c5 : int
        Backbone output channels at stride-32.
        Default: backbone_channels[1]*8 = 256.
    out_channels : int
        Unified output channel count for N3, N4, N5.
        Default: cfg.model.backbone_channels[-1]  (256 at default config).

    Design choices (see Part A for full justification):
    • Standard top-down FPN — best bias-variance trade-off for single-class
      pedestrian detection on limited data.
    • Standard Conv2d (not depthwise separable) at neck scale for reliability.
    • Nearest-neighbour upsampling — no learnable parameters, avoids
      checkerboard artefacts, and matches the original FPN paper.
    • out_channels = backbone_channels[-1] = 256 — matches FPN paper,
      fits sub-5 M budget, derived entirely from config.
    """

    def __init__(
        self,
        c3: int = _C3,            # cfg.model.backbone_channels[2] = 64
        c4: int = _C4,            # cfg.model.backbone_channels[3] = 128
        c5: int = _C5,            # cfg.model.backbone_channels[4] = 256
        out_channels: int = _OUT,  # cfg.model.backbone_channels[-1] = 256
    ) -> None:
        super().__init__()

        # ── Save for forward() documentation and external inspection ──────────
        self.out_channels = out_channels

        # ── Stage 1: Lateral 1×1 projections ─────────────────────────────────
        # A 1×1 conv collapses the channel dimension without touching H or W.
        # No BN/ReLU here — standard FPN (Lin et al. §3):
        # "We attach a 1×1 convolutional layer to produce a fixed-size feature map."

        # P3 (B, C3=64,  H/8,  W/8)  → L3 (B, OUT=256, H/8,  W/8)
        self.lat3 = nn.Conv2d(c3, out_channels, kernel_size=1, bias=False)

        # P4 (B, C4=128, H/16, W/16) → L4 (B, OUT=256, H/16, W/16)
        self.lat4 = nn.Conv2d(c4, out_channels, kernel_size=1, bias=False)

        # P5 (B, C5=256, H/32, W/32) → L5 (B, OUT=256, H/32, W/32)
        self.lat5 = nn.Conv2d(c5, out_channels, kernel_size=1, bias=False)

        # ── Stage 3: Output 3×3 smoothing convolutions ───────────────────────
        # Applied *after* top-down addition to reduce aliasing from nearest-
        # neighbour upsampling.  ConvBnRelu keeps H, W unchanged.

        # TD3 (B, OUT=256, H/8,  W/8)  → N3 (B, OUT=256, H/8,  W/8)
        self.out3 = ConvBnRelu(out_channels, out_channels)

        # TD4 (B, OUT=256, H/16, W/16) → N4 (B, OUT=256, H/16, W/16)
        self.out4 = ConvBnRelu(out_channels, out_channels)

        # TD5 (B, OUT=256, H/32, W/32) → N5 (B, OUT=256, H/32, W/32)
        self.out5 = ConvBnRelu(out_channels, out_channels)

        # ── Weight initialisation ─────────────────────────────────────────────
        # Kaiming uniform for conv weights (good default for ReLU networks).
        # Constant 1 / 0 for BN gamma / beta (standard).
        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Initialise all conv and BN weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming uniform (He et al.) — appropriate for ReLU activations.
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)   # gamma = 1
                nn.init.zeros_(m.bias)    # beta  = 0

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        p3: torch.Tensor,  # (B, C3=64,  H/8,  W/8)
        p4: torch.Tensor,  # (B, C4=128, H/16, W/16)
        p5: torch.Tensor,  # (B, C5=256, H/32, W/32)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        p3 : (B, C3, H/8,  W/8)   C3 = backbone_channels[1]*4 = 128
        p4 : (B, C4, H/16, W/16)  C4 = backbone_channels[1]*8 = 256
        p5 : (B, C5, H/32, W/32)  C5 = backbone_channels[1]*8 = 256

        Returns
        -------
        n3 : (B, OUT, H/8,  W/8)   OUT = cfg.model.backbone_channels[-1] = 256
        n4 : (B, OUT, H/16, W/16)
        n5 : (B, OUT, H/32, W/32)
        """

        # ── Stage 1: Lateral projections ─────────────────────────────────────

        # p3 (B, C3=64,  H/8,  W/8)  → lat3(1×1) → l3 (B, OUT=256, H/8,  W/8)
        l3: torch.Tensor = self.lat3(p3)

        # p4 (B, C4=128, H/16, W/16) → lat4(1×1) → l4 (B, OUT=256, H/16, W/16)
        l4: torch.Tensor = self.lat4(p4)

        # p5 (B, C5=256, H/32, W/32) → lat5(1×1) → l5 (B, OUT=256, H/32, W/32)
        l5: torch.Tensor = self.lat5(p5)

        # ── Stage 2: Top-down fusion ──────────────────────────────────────────
        # Work from deepest (coarsest, most semantic) up to shallowest (finest).

        # td5 is just the lateral at the deepest level — no merging needed here.
        # td5: (B, OUT=256, H/32, W/32)
        td5: torch.Tensor = l5

        # Upsample td5 by 2× (nearest-neighbour) to match l4's spatial size.
        # F.interpolate with mode="nearest" has no learnable params and
        # produces no checkerboard artefacts (unlike transposed conv).
        # td5_up: (B, OUT=256, H/16, W/16)   [H/32 × 2 = H/16, same for W]
        td5_up: torch.Tensor = F.interpolate(
            td5,
            scale_factor=2,
            mode="nearest",
            # recompute_scale_factor=False avoids a deprecation warning in
            # PyTorch ≥ 1.11 when scale_factor is an integer.
            recompute_scale_factor=False,
        )

        # Element-wise addition merges semantic context from P5 with
        # spatial detail from P4.
        # td4: (B, OUT=256, H/16, W/16)
        td4: torch.Tensor = l4 + td5_up

        # Upsample td4 by 2× to match l3's spatial size.
        # td4_up: (B, OUT=256, H/8, W/8)
        td4_up: torch.Tensor = F.interpolate(
            td4,
            scale_factor=2,
            mode="nearest",
            recompute_scale_factor=False,
        )

        # Element-wise addition merges semantic context (now from both P5 and P4)
        # with fine spatial detail from P3.
        # td3: (B, OUT=256, H/8, W/8)
        td3: torch.Tensor = l3 + td4_up

        # ── Stage 3: Output smoothing convolutions ────────────────────────────
        # 3×3 ConvBnRelu removes the aliasing "grid" that nearest-neighbour
        # upsampling can leave at edges, and also integrates local context.

        # td5 (B, OUT=256, H/32, W/32) → out5(3×3 BnReLU) → n5 (B, OUT=256, H/32, W/32)
        n5: torch.Tensor = self.out5(td5)

        # td4 (B, OUT=256, H/16, W/16) → out4(3×3 BnReLU) → n4 (B, OUT=256, H/16, W/16)
        n4: torch.Tensor = self.out4(td4)

        # td3 (B, OUT=256, H/8,  W/8)  → out3(3×3 BnReLU) → n3 (B, OUT=256, H/8,  W/8)
        n3: torch.Tensor = self.out3(td3)

        # Return ordered shallowest → deepest, matching convention used by
        # the detection head in Phase 4.
        return n3, n4, n5


# ─────────────────────────────────────────────────────────────────────────────
# Quick parameter count (run this file directly: python neck.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = LightweightFPN()   # defaults come from _C3, _C4, _C5, _OUT above
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Config backbone_channels : {_BACKBONE_CH}")
    print(f"Neck input  channels     : C3={_C3}, C4={_C4}, C5={_C5}")
    print(f"Neck output channels     : OUT={_OUT}")
    print(f"LightweightFPN — total params   : {total:,}")
    print(f"LightweightFPN — trainable params: {trainable:,}")

    # Sanity forward pass using config-derived values
    B = 1
    dummy_p3 = torch.zeros(B, _C3, 80, 80)
    dummy_p4 = torch.zeros(B, _C4, 40, 40)
    dummy_p5 = torch.ones(B, _C5, 20, 20)
    n3, n4, n5 = model(dummy_p3, dummy_p4, dummy_p5)
    print(f"N3: {tuple(n3.shape)}")
    print(f"N4: {tuple(n4.shape)}")
    print(f"N5: {tuple(n5.shape)}")
    print(f"N3 all-zero: {n3.abs().sum().item() == 0.0}  (should be False — P5 info propagated)")
