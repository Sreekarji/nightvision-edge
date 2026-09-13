"""
neck.py — LightweightFPN + one-level PAN + optional selective kernel
=====================================================================
    p3 (B, 96, H/8,  W/8)  -> n3 (B, 48, H/8,  W/8)
    p4 (B, 96, H/16, W/16) -> n4 (B, 64, H/16, W/16)
    p5 (B, 96, H/32, W/32) -> n5 (B, 64, H/32, W/32)

WHY out3 IS 48 AND THE OTHERS ARE 64
------------------------------------
out3 is a dense 3x3 over the largest feature map in the neck. At 64->64 on a
36x64 grid it was the single most expensive convolution outside the backbone,
and larger than the entire head. The 1x1 lateral in front of it had already
compressed 96 channels to 64, so it was never carrying 64 channels' worth of
independent information. 48 = 2*24, which is an ideal output-channel multiple
for the Neural-ART 1x1 path, and it also lowers the head's stride-8
feature-width x channel product.

ONE-LEVEL BOTTOM-UP PAN
-----------------------
Top-down alone propagates semantics downward but never returns fine spatial
detail upward. The dataset's pedestrian height distribution is wide (canvas p5
around 0.07 to p95 around 0.55 of frame height), so the large-object levels
genuinely want stride-8 detail. A single stride-2 3x3 from the fused stride-8
tensor into the stride-8->16 path is standard FPN->PAN practice and is the
cheapest possible version of it. Toggle with neck_pan.

SELECTIVE KERNEL (optional, off by default)
-------------------------------------------
After IPD-Net (Zhou et al., "IPD-Net: Infrared Pedestrian Detection Network
via Adaptive Feature Extraction and Coordinate Information Fusion", Sensors
2022, 22(22):8966), which added a selective-kernel component to a YOLOv5 base
and reported a 3.6% mAP50 improvement over that baseline on the ZUT infrared
dataset. Two parallel receptive fields (3x3 and 3x3 dilated by 2, which is
cheaper than 3x3 + 5x5 and avoids ST's decomposition of kernels wider than 3),
global-average-pooled and reduced to per-channel mixing weights.

One deliberate deviation from the SK paper: the two-way mix is produced by a
SIGMOID gate a and (1 - a), not by a softmax over two branches. ST maps
Softmax on hardware only when the NPU compiler is invoked with
--expand-softmax, and SW_INT otherwise; Sigmoid and Sub are both
unconditionally HW. GlobalAveragePool is HW with a limited window. The gate is
mathematically equivalent to a 2-way softmax up to a reparameterisation of the
logits, so nothing is lost.

EXPORT NOTES
------------
* F.interpolate(scale_factor=2.0, mode="nearest", recompute_scale_factor=False)
  rather than size=x.shape[-2:]. The latter emits Shape -> Gather -> Concat,
  and ST maps Gather as SW_INT. At opset 12 this produces a Resize with
  coordinate_transformation_mode='asymmetric' and nearest_mode='floor', which
  are exactly the two attributes ST requires for the HW path — asserted by
  export_onnx._check_resize_nodes().
* BatchNorm + ReLU6 throughout, both foldable and both HW.
* No get_config() at import: importing this module has no side effects.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1,
                 p: int = None, d: int = 1, act: bool = True) -> None:
        super().__init__()
        if p is None:
            p = d * (k - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=p, dilation=d, bias=False,
                              padding_mode="zeros")
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SelectiveKernel(nn.Module):
    """
    Two-receptive-field channel gating.

        u1 = ConvBNAct(3x3, d=1)(x)
        u2 = ConvBNAct(3x3, d=2)(x)
        s  = GlobalAveragePool(u1 + u2)          # (B, C, 1, 1)
        z  = ReLU6(fc_down(s))                   # (B, C/r, 1, 1)
        a  = sigmoid(fc_up(z))                   # (B, C, 1, 1)
        y  = a * u1 + (1 - a) * u2

    Channel broadcasting of a (B, C, 1, 1) tensor against (B, C, H, W) is the
    best-performing broadcast case on the Neural-ART arithmetic unit for
    C <= 512.
    """

    def __init__(self, ch: int, reduction: int = 4, dilation: int = 2) -> None:
        super().__init__()
        mid = max(8, int(ch) // int(reduction))
        self.branch_a = ConvBNAct(ch, ch, k=3, s=1, d=1)
        self.branch_b = ConvBNAct(ch, ch, k=3, s=1, d=int(dilation))
        self.fc_down = nn.Conv2d(ch, mid, kernel_size=1, bias=True)
        self.fc_up = nn.Conv2d(mid, ch, kernel_size=1, bias=True)
        self.act = nn.ReLU6(inplace=True)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fc_down.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc_down.bias)
            # Zero-init the gate so the module starts as a plain 0.5/0.5
            # average of the two branches and cannot destabilise early
            # training; it learns its way to a real selection.
            nn.init.zeros_(self.fc_up.weight)
            nn.init.zeros_(self.fc_up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u1 = self.branch_a(x)
        u2 = self.branch_b(x)
        s = F.adaptive_avg_pool2d(u1 + u2, 1)       # -> GlobalAveragePool
        a = torch.sigmoid(self.fc_up(self.act(self.fc_down(s))))
        return a * u1 + (1.0 - a) * u2


class LightweightFPN(nn.Module):
    """
    Top-down FPN over three levels, optional one-level bottom-up PAN,
    optional selective-kernel smoothing.

    Args:
        c3, c4, c5      : backbone widths at strides 8, 16, 32 (96/96/96).
        out_channels    : common fusion width (64).
        out3_channels   : narrowed stride-8 output width (48).
        pan             : add the bottom-up N3 -> N4 path.
        sk              : use SelectiveKernel for the smoothing convs.

    out_channels_per_level is a TUPLE, because the three outputs are no longer
    the same width. The head builds a per-level 1x1 stem from it.
    """

    def __init__(self, c3: int = 96, c4: int = 96, c5: int = 96,
                 out_channels: int = 64, out3_channels: int = 48,
                 pan: bool = True, sk: bool = False,
                 sk_dilation: int = 2) -> None:
        super().__init__()
        oc = int(out_channels)
        oc3 = int(out3_channels)
        self.out_channels = oc
        self.out3_channels = oc3
        self.use_pan = bool(pan)
        self.use_sk = bool(sk)

        # Lateral 1x1 projections, no activation (FPN convention). BN keeps
        # the three levels on a common scale before addition.
        self.lat3 = nn.Conv2d(c3, oc, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(c4, oc, kernel_size=1, bias=False)
        self.lat5 = nn.Conv2d(c5, oc, kernel_size=1, bias=False)
        self.lat3_bn = nn.BatchNorm2d(oc)
        self.lat4_bn = nn.BatchNorm2d(oc)
        self.lat5_bn = nn.BatchNorm2d(oc)

        # One-level bottom-up PAN: stride-2 dense 3x3, fused width -> fused
        # width, injected into the stride-16 tensor before smoothing.
        self.pan3to4 = StrideDownPAN(oc, oc) if self.use_pan else None

        def smooth(in_ch: int, out_ch: int) -> nn.Module:
            if self.use_sk:
                if in_ch != out_ch:
                    # SK mixes two same-width branches, so project first.
                    return nn.Sequential(
                        ConvBNAct(in_ch, out_ch, k=1),
                        SelectiveKernel(out_ch, dilation=sk_dilation),
                    )
                return SelectiveKernel(out_ch, dilation=sk_dilation)
            return ConvBNAct(in_ch, out_ch, k=3, s=1)

        self.out3 = smooth(oc, oc3)       # narrowed: the expensive one
        self.out4 = smooth(oc, oc)
        self.out5 = smooth(oc, oc)

        self._init_weights()

    @property
    def out_channels_per_level(self) -> Tuple[int, int, int]:
        return (self.out3_channels, self.out_channels, self.out_channels)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # SelectiveKernel.fc_up is deliberately zero-initialised;
                # do not overwrite it.
                if getattr(m, "_sk_gate", False):
                    continue
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Re-apply the SK gate init, which _init_weights above would clobber.
        for m in self.modules():
            if isinstance(m, SelectiveKernel):
                with torch.no_grad():
                    nn.init.kaiming_uniform_(m.fc_down.weight,
                                             nonlinearity="relu")
                    nn.init.zeros_(m.fc_down.bias)
                    nn.init.zeros_(m.fc_up.weight)
                    nn.init.zeros_(m.fc_up.bias)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l3 = self.lat3_bn(self.lat3(p3))          # (B, 64, H/8,  W/8)
        l4 = self.lat4_bn(self.lat4(p4))          # (B, 64, H/16, W/16)
        l5 = self.lat5_bn(self.lat5(p5))          # (B, 64, H/32, W/32)

        # ---- top-down ----
        td5 = l5
        up5 = F.interpolate(td5, scale_factor=2.0, mode="nearest",
                            recompute_scale_factor=False)
        td4 = l4 + up5
        up4 = F.interpolate(td4, scale_factor=2.0, mode="nearest",
                            recompute_scale_factor=False)
        td3 = l3 + up4

        # ---- one-level bottom-up ----
        # Source is td3 (the fused 64-channel tensor), not n3, so the PAN path
        # is independent of out3's narrowing and stays 64->64.
        if self.pan3to4 is not None:
            td4 = td4 + self.pan3to4(td3)

        n3 = self.out3(td3)                       # (B, 48, H/8,  W/8)
        n4 = self.out4(td4)                       # (B, 64, H/16, W/16)
        n5 = self.out5(td5)                       # (B, 64, H/32, W/32)
        return n3, n4, n5


class StrideDownPAN(nn.Module):
    """Dense strided 3x3 for the bottom-up path. Named separately so the PAN
    contribution is legible in a parameter breakdown and in the ONNX graph."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = ConvBNAct(in_ch, out_ch, k=3, s=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


if __name__ == "__main__":
    for sk in (False, True):
        neck = LightweightFPN(96, 96, 96, out_channels=64, out3_channels=48,
                              pan=True, sk=sk).eval()
        n3, n4, n5 = neck(torch.randn(1, 96, 36, 64),
                          torch.randn(1, 96, 18, 32),
                          torch.randn(1, 96, 9, 16))
        print(f"sk={sk}  N3 {tuple(n3.shape)}  N4 {tuple(n4.shape)}  "
              f"N5 {tuple(n5.shape)}")
        print(f"         widths {neck.out_channels_per_level}  "
              f"params {sum(p.numel() for p in neck.parameters()):,}")
    neck = LightweightFPN().eval()
    n3, _, _ = neck(torch.ones(1, 96, 36, 64), torch.ones(1, 96, 18, 32),
                    torch.ones(1, 96, 9, 16))
    print("P5 reaches N3:", bool(n3.abs().sum() > 0))
    n_dw = sum(1 for m in neck.modules()
               if isinstance(m, nn.Conv2d) and m.groups > 1)
    print("grouped/depthwise convs:", n_dw, "(must be 0)")
    assert n_dw == 0
