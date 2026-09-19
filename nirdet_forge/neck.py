"""
neck.py — LightweightFPN + one-level PAN + optional selective kernel
=====================================================================
    p3 (B,96,H/8, W/8)  -> n3 (B,48,H/8, W/8)
    p4 (B,96,H/16,W/16) -> n4 (B,64,H/16,W/16)
    p5 (B,96,H/32,W/32) -> n5 (B,64,H/32,W/32)

out3 is 48 because that 3x3 over the largest map was the most expensive conv
outside the backbone, and the 1x1 lateral in front of it had already
compressed 96 -> 64. 48 = 2*24, an ideal ST output-channel multiple.

The one-level bottom-up PAN (N3 -> N4) returns fine spatial detail upward.
SelectiveKernel (IPD-Net, Sensors 2022 22(22):8966) is off by default; its
two-way mix uses a SIGMOID gate rather than a softmax, because ST maps
Softmax on hardware only with --expand-softmax.

Export: F.interpolate(scale_factor=..., recompute_scale_factor=False) so the
Resize carries constant scales and no Shape->Gather->Concat.

RANGE LEDGER (FPN fusions)
--------------------------
td4 = l4 + up5 and td3 = l3 + up4 are BatchNorm'd additions. Without a
post-fusion clamp their range is unbounded before out3/out4. With
fuse_clip=True (the default) both are clamped to [-6, 6] — ONNX Clip,
HW-mapped. This is a trained-numerics decision set here, before the first
training run.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1,
                 p: Optional[int] = None, d: int = 1, act: bool = True) -> None:
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
    u1 = ConvBNAct(3x3, d=1)(x);  u2 = ConvBNAct(3x3, d=2)(x)
    a  = sigmoid(fc_up(ReLU6(fc_down(GlobalAvgPool(u1 + u2)))))
    y  = a * u1 + (1 - a) * u2
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
            # average and learns its way to a real selection.
            nn.init.zeros_(self.fc_up.weight)
            nn.init.zeros_(self.fc_up.bias)
        self.fc_up._sk_gate = True     # LightweightFPN._init_weights skips it

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u1 = self.branch_a(x)
        u2 = self.branch_b(x)
        s = F.adaptive_avg_pool2d(u1 + u2, 1)       # -> GlobalAveragePool
        a = torch.sigmoid(self.fc_up(self.act(self.fc_down(s))))
        return a * u1 + (1.0 - a) * u2


class StrideDownPAN(nn.Module):
    """Dense strided 3x3 for the bottom-up path, named so the PAN
    contribution is legible in a parameter breakdown and in the ONNX graph."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = ConvBNAct(in_ch, out_ch, k=3, s=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightFPN(nn.Module):
    """Top-down FPN, optional one-level bottom-up PAN, optional SK smoothing.

    out_channels_per_level is a TUPLE: the three outputs are not the same
    width, and the head builds a per-level 1x1 stem from it.
    """

    def __init__(self, c3: int = 96, c4: int = 96, c5: int = 96,
                 out_channels: int = 64, out3_channels: int = 48,
                 pan: bool = True, sk: bool = False,
                 sk_dilation: int = 2,
                 fuse_clip: bool = True) -> None:
        super().__init__()
        oc = int(out_channels)
        oc3 = int(out3_channels)
        self.out_channels = oc
        self.out3_channels = oc3
        self.use_pan = bool(pan)
        self.use_sk = bool(sk)
        self.sk_dilation = int(sk_dilation)
        # F77: clamp each top-down fusion result to [-6, 6] (ONNX Clip).
        self.fuse_clip = bool(fuse_clip)

        # Lateral 1x1 projections, no activation (FPN convention). BN keeps
        # the three levels on a common scale before addition.
        self.lat3 = nn.Conv2d(c3, oc, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(c4, oc, kernel_size=1, bias=False)
        self.lat5 = nn.Conv2d(c5, oc, kernel_size=1, bias=False)
        for lat in (self.lat3, self.lat4, self.lat5):
            # Marked so _init_weights uses Xavier, not Kaiming-for-ReLU: these
            # are LINEAR projections with no activation after them, and
            # Kaiming over-scales a linear projection by sqrt(2) (F77).
            lat._linear_proj = True
        self.lat3_bn = nn.BatchNorm2d(oc)
        self.lat4_bn = nn.BatchNorm2d(oc)
        self.lat5_bn = nn.BatchNorm2d(oc)

        self.pan3to4 = StrideDownPAN(oc, oc) if self.use_pan else None

        self.out3 = self._smooth(oc, oc3)   # narrowed: the expensive one
        self.out4 = self._smooth(oc, oc)
        self.out5 = self._smooth(oc, oc)

        self._init_weights()

    def _smooth(self, in_ch: int, out_ch: int) -> nn.Module:
        """Smoothing conv for one level (a method, not a closure over the
        constructor's locals, so the sk path is traceable — F79)."""
        if self.use_sk:
            if in_ch != out_ch:
                # SK mixes two same-width branches, so project first.
                return nn.Sequential(
                    ConvBNAct(in_ch, out_ch, k=1),
                    SelectiveKernel(out_ch, dilation=self.sk_dilation),
                )
            return SelectiveKernel(out_ch, dilation=self.sk_dilation)
        return ConvBNAct(in_ch, out_ch, k=3, s=1)

    @property
    def out_channels_per_level(self) -> Tuple[int, int, int]:
        return (self.out3_channels, self.out_channels, self.out_channels)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if getattr(m, "_sk_gate", False):
                    continue                       # deliberately zero-init
                if getattr(m, "_linear_proj", False):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    continue
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l3 = self.lat3_bn(self.lat3(p3))
        l4 = self.lat4_bn(self.lat4(p4))
        l5 = self.lat5_bn(self.lat5(p5))

        # ---- top-down (RANGE LEDGER: see the module docstring) ----
        td5 = l5
        up5 = F.interpolate(td5, scale_factor=2.0, mode="nearest",
                            recompute_scale_factor=False)
        td4 = l4 + up5
        if self.fuse_clip:
            td4 = torch.clamp(td4, -6.0, 6.0)   # ONNX Clip, HW-mapped
        up4 = F.interpolate(td4, scale_factor=2.0, mode="nearest",
                            recompute_scale_factor=False)
        td3 = l3 + up4
        if self.fuse_clip:
            td3 = torch.clamp(td3, -6.0, 6.0)

        # ---- one-level bottom-up ----
        # Source is td3 (the fused 64-channel tensor), not n3, so the PAN path
        # is independent of out3's narrowing and stays 64->64.
        if self.pan3to4 is not None:
            td4 = td4 + self.pan3to4(td3)
            if self.fuse_clip:
                # F06 (audit fix): the PAN join is a FUSION too and must be in
                # the range ledger. pan3to4 ends in ReLU6 ([0,6]) and is added
                # to a td4 already clamped to [-6,6], so without this re-clamp
                # the tensor entering out4 spans [-6,12] — double the
                # documented range — and the INT8 per-tensor scale is sized
                # from that wider range. ONNX Clip, HW-mapped.
                td4 = torch.clamp(td4, -6.0, 6.0)

        return self.out3(td3), self.out4(td4), self.out5(td5)


if __name__ == "__main__":
    for sk in (False, True):
        neck = LightweightFPN(96, 96, 96, out_channels=64, out3_channels=48,
                              pan=True, sk=sk).eval()
        n3, n4, n5 = neck(torch.randn(1, 96, 36, 64),
                          torch.randn(1, 96, 18, 32),
                          torch.randn(1, 96, 9, 16))
        print(f"sk={sk}  N3 {tuple(n3.shape)}  N4 {tuple(n4.shape)}  "
              f"N5 {tuple(n5.shape)}   widths "
              f"{neck.out_channels_per_level}  "
              f"params {sum(p.numel() for p in neck.parameters()):,}")

    # Connectivity by GRADIENT (F80). The old all-ones forward test could not
    # fail: with untrained BN running stats the bias terms alone make n3
    # non-zero regardless of whether P5 reaches it.
    neck = LightweightFPN().train()
    p5 = torch.randn(1, 96, 9, 16, requires_grad=True)
    n3, _, _ = neck(torch.zeros(1, 96, 36, 64),
                    torch.zeros(1, 96, 18, 32), p5)
    n3.sum().backward()
    print("P5 reaches N3:",
          bool(p5.grad is not None and float(p5.grad.abs().sum()) > 0))
    n_dw = sum(1 for m in neck.modules()
               if isinstance(m, nn.Conv2d) and m.groups > 1)
    print("grouped/depthwise convs:", n_dw, "(must be 0)")
    assert n_dw == 0
