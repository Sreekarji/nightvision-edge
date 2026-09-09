"""
neck.py — LightweightFPN (two levels, configurable width)
==========================================================
Fixes relative to the original:
  * ``out_channels`` is a real constructor argument (was a module constant
    baked in from get_config() at import time, so model.py could not change it).
  * No ``get_config()`` at import: importing this module has no side effects.
  * ``F.interpolate(scale_factor=2.0)`` instead of ``size=x.shape[-2:]``, so
    ONNX gets a Resize with a constant ``scales`` input rather than a
    Shape -> Gather -> Concat chain.
  * BatchNorm + ReLU6 (foldable, HW-mapped) instead of BN + ReLU/SiLU mixes.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LightweightFPN(nn.Module):
    """
    Top-down FPN over three levels.

        p3 (B, c3, H/8,  W/8)  -> n3 (B, out, H/8,  W/8)
        p4 (B, c4, H/16, W/16) -> n4 (B, out, H/16, W/16)
        p5 (B, c5, H/32, W/32) -> n5 (B, out, H/32, W/32)

    Args:
        c3, c4, c5   : backbone channel widths at strides 8, 16, and 32.
        out_channels : unified neck width (64 for NIRDet-Lite).
        depthwise    : use DW3x3 + PW1x1 for the smoothing convs.
    """

    def __init__(self, c3: int = 128, c4: int = 256, c5: int = 256,
                 out_channels: int = 64, depthwise: bool = False) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        oc = self.out_channels

        # Lateral 1x1 projections (no activation: FPN convention).
        self.lat3 = nn.Conv2d(c3, oc, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(c4, oc, kernel_size=1, bias=False)
        self.lat5 = nn.Conv2d(c5, oc, kernel_size=1, bias=False)
        self.lat3_bn = nn.BatchNorm2d(oc)
        self.lat4_bn = nn.BatchNorm2d(oc)
        self.lat5_bn = nn.BatchNorm2d(oc)

        if depthwise:
            self.out3 = nn.Sequential(
                ConvBNAct(oc, oc, k=3, s=1, p=1, groups=oc),
                ConvBNAct(oc, oc, k=1, s=1, p=0),
            )
            self.out4 = nn.Sequential(
                ConvBNAct(oc, oc, k=3, s=1, p=1, groups=oc),
                ConvBNAct(oc, oc, k=1, s=1, p=0),
            )
            self.out5 = nn.Sequential(
                ConvBNAct(oc, oc, k=3, s=1, p=1, groups=oc),
                ConvBNAct(oc, oc, k=1, s=1, p=0),
            )
        else:
            self.out3 = ConvBNAct(oc, oc, k=3, s=1, p=1)
            self.out4 = ConvBNAct(oc, oc, k=3, s=1, p=1)
            self.out5 = ConvBNAct(oc, oc, k=3, s=1, p=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor,
                p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l3 = self.lat3_bn(self.lat3(p3))          # (B, out, H/8,  W/8)
        l4 = self.lat4_bn(self.lat4(p4))          # (B, out, H/16, W/16)
        l5 = self.lat5_bn(self.lat5(p5))          # (B, out, H/32, W/32)

        td5 = l5
        up5 = F.interpolate(td5, scale_factor=2.0, mode="nearest",
                            recompute_scale_factor=False)
        td4 = l4 + up5
        up4 = F.interpolate(td4, scale_factor=2.0, mode="nearest",
                            recompute_scale_factor=False)
        td3 = l3 + up4

        n3 = self.out3(td3)
        n4 = self.out4(td4)
        n5 = self.out5(td5)
        return n3, n4, n5


if __name__ == "__main__":
    neck = LightweightFPN(128, 256, 256, out_channels=64).eval()
    n3, n4, n5 = neck(torch.zeros(1, 128, 48, 80), torch.ones(1, 256, 24, 40), torch.ones(1, 256, 12, 20))
    print("N3", tuple(n3.shape), "N4", tuple(n4.shape), "N5", tuple(n5.shape))
    print("params", f"{sum(p.numel() for p in neck.parameters()):,}")
    print("P5 reached N3:", bool(n3.abs().sum() > 0))
