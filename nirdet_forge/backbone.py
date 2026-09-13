"""
backbone.py — NIRBackbone (dense convolutions only)
====================================================
    input (B,1,288,512)
      stem        3x3 s2, Sobel/Laplacian-seeded  -> (B, 16, 144, 256)
      widen1      3x3 s2 -> 48                    -> (B, 48,  72, 128)
      stage1      dense residual x1               -> (B, 48,  72, 128)
      down2       3x3 s2 -> 96                    -> (B, 96,  36,  64)  = P3
      stage2      dense residual x2
      down3       3x3 s2 -> 96                    -> (B, 96,  18,  32)  = P4
      stage3      dense residual x2
      down4       3x3 s2 -> 96                    -> (B, 96,   9,  16)  = P5
      stage4      dense residual x1, dilation 2

NO DEPTHWISE SEPARABLE CONVOLUTIONS
-----------------------------------
ST's Neural-ART table does list DEPTHWISE_CONV_2D as hardware-mapped, so the
problem is not support, it is occupancy. ST's own kernel guidance states that
a 1x1 kernel is handled efficiently only when the input channel count is
N*(72..128) and the output channel count is M*(16..24), best M*24. A 64-channel
pointwise convolution therefore leaves the four CONV accelerators substantially
idle, and a 3x3 with groups=channels gives the array almost nothing to
parallelise across. STResNet/STYOLO (arXiv:2601.05364), benchmarked on STM32N6
silicon, rejected depthwise separable convolutions, fire modules, channel
shuffle and squeeze-excitation as "often unsupported or inefficient on MCU/NPU
hardware" and built "exclusively" from standard 3x3 and 1x1, reporting lower
RAM (1.39 vs 2.01 MB) and lower latency (21.3 vs 22.4 ms) than MobileNetV2-1.0
at comparable accuracy. On the Pi 5 Cortex-A76 the argument is different but
points the same way: INT8 dense GEMM exploits SDOT, depthwise kernels are
memory-bandwidth bound.

So: every residual block is a plain ConvBNAct(ch, ch, k=3, p=1), every
downsample is a strided dense 3x3, and base_ch drops 32 -> 24 to pay for it.

STAGE WIDTH: base_ch * 4 = 96 EVERYWHERE
----------------------------------------
Stages 3 and 4 used to emit 256 channels that the neck immediately projected
to 64 with a 1x1 — three quarters of that computation was discarded. 96 is
inside ST's 72..128 input-channel window for 1x1 kernels and is 4*24, an ideal
output-channel multiple. It is also non-prime and a multiple of 8 (ST: "avoid
using prime numbers as number of kernels/channels").

STEM PROJECTION (STResNet/STYOLO section 5.3)
---------------------------------------------
The stem emits only ``stem_ch`` channels on the stride-2 map — the largest
tensor in the network — and width is recovered on the stride-4 map. STYOLO
measured peak RAM 4.26 -> 2.46 MB (-42%) and latency 47.32 -> 42.99 ms for
30.54 -> 30.12 mAP from exactly this change. The N6 falls off a cliff past
~4 MB because it starts using external memory.

ACTIVATION / PADDING
--------------------
ReLU6 (ONNX Clip), which ST maps on hardware unconditionally. SiLU is not
supported as a single ONNX operator at all; only the expanded form is, behind
a compiler flag. Padding is zeros: ST maps Pad "partial - according to the
parameters", and Pad <= 2 costs nothing, but reflect is the case that falls
back to software.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """Conv -> BatchNorm -> ReLU6. BN folds into the conv at export.

    BatchNorm, never GroupNorm: Neural-ART has no GroupNorm mapping (it
    decomposes into ReduceMean/Sqrt/Div, and Div with a runtime divisor is
    SW_INT), and GN also blocks NCNN's BN-fusion pass.
    """

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


class DenseResBlock(nn.Module):
    """
    Dense 3x3 residual block. This is the direct replacement for the old
    _DWSResBlock: one full 3x3 convolution, no depthwise/pointwise split.

    ``dilation`` is used only by the deepest stage.
    """

    def __init__(self, ch: int, shortcut: bool = True, dilation: int = 1) -> None:
        super().__init__()
        self.conv = ConvBNAct(ch, ch, k=3, s=1, d=int(dilation))
        self.shortcut = bool(shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return x + out if self.shortcut else out


class CSPBlock(nn.Module):
    """
    x -> cv1(1x1, out/2) -> [dense 3x3 res] * n --.
      -> cv2(1x1, out/2) ------------------------ concat -> cv3(1x1, out)

    ``dilate_last`` applies the dilation to the FINAL residual block only,
    which is how the stride-32 stage gets its enlarged receptive field for
    free. Concat is HW-mapped on Neural-ART.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1,
                 shortcut: bool = True, dilate_last: int = 1) -> None:
        super().__init__()
        if out_ch % 2:
            raise ValueError(f"CSP out_ch must be even, got {out_ch}")
        hidden = out_ch // 2
        self.cv1 = ConvBNAct(in_ch, hidden, k=1)
        self.cv2 = ConvBNAct(in_ch, hidden, k=1)
        self.cv3 = ConvBNAct(2 * hidden, out_ch, k=1)
        blocks: List[nn.Module] = []
        for i in range(n):
            d = int(dilate_last) if (i == n - 1) else 1
            blocks.append(DenseResBlock(hidden, shortcut, dilation=d))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.blocks(self.cv1(x))
        bypass = self.cv2(x)
        return self.cv3(torch.cat([main, bypass], dim=1))


class StrideDown(nn.Module):
    """Learnable 2x downsample: a single dense strided 3x3.

    ST: stride 2 is one of the three efficient horizontal strides, and a 3x3
    kernel is the recommended height/width. The previous implementation was
    a strided depthwise 3x3 followed by a pointwise 1x1.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = ConvBNAct(in_ch, out_ch, k=3, s=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------------

class NIRStem(nn.Module):
    """
    Single-channel NIR entry conv, stride 2, zeros padding, NARROW output.

    ``out_ch`` defaults to 16 (config.ModelCfg.stem_ch), not 32: this is the
    largest activation tensor in the network and the STYOLO projection result
    says the width is better spent one stage later.

    The first ``n_edge_init`` filters are seeded with unit-L2 Sobel /
    Laplacian kernels — 850 nm reflective NIR is an edge-rich, texture-poor
    modality, so oriented gradients are the right inductive bias. The rest are
    Kaiming-uniform. ``imagenet_slots`` reports which filter indices are free
    for channel-summed ImageNet transfer (see train.imagenet_stem_init).
    """

    def __init__(self, out_ch: int = 16, n_edge_init: int = 6) -> None:
        super().__init__()
        if n_edge_init > out_ch:
            raise ValueError(f"n_edge_init={n_edge_init} > out_ch={out_ch}")
        self.out_ch = int(out_ch)
        self.n_edge_init = int(n_edge_init)

        self.conv = nn.Conv2d(1, self.out_ch, kernel_size=3, stride=2,
                              padding=1, bias=False, padding_mode="zeros")
        self.bn = nn.BatchNorm2d(self.out_ch)
        self.act = nn.ReLU6(inplace=True)
        self._init_edge_kernels()

    @staticmethod
    def _edge_templates() -> List[torch.Tensor]:
        Gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        Gy = Gx.T.contiguous()
        G45 = torch.tensor([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]])
        G135 = torch.tensor([[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]])
        Lap = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]])
        LapD = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]])
        out = []
        for t in (Gx, Gy, G45, G135, Lap, LapD):
            n = t.norm()
            out.append(t / n if float(n) > 0 else t)
        return out

    def _init_edge_kernels(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
            for i, t in enumerate(self._edge_templates()[: self.n_edge_init]):
                self.conv.weight[i, 0] = t

    @property
    def imagenet_slots(self) -> Tuple[int, int]:
        """[start, end) filter indices that may be overwritten by transfer."""
        return (self.n_edge_init, self.out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class NIRBackbone(nn.Module):
    """
    Returns (P3, P4, P5) at strides 8, 16, 32.

    out_channels = (base_ch*4, base_ch*4, base_ch*4) = (96, 96, 96) at
    base_ch=24. All three are equal on purpose: the neck projects every level
    to a common width anyway, so a wider deep stage was paying for channels
    that a 1x1 immediately discarded.
    """

    def __init__(self, base_ch: int = 24,
                 stem_ch: int = 16,
                 n_blocks: Optional[Tuple[int, ...]] = None,
                 n_edge_init: int = 6,
                 p5_dilation: int = 2) -> None:
        super().__init__()
        n_blocks = tuple(n_blocks) if n_blocks is not None else (1, 2, 2)
        if len(n_blocks) != 3:
            raise ValueError("n_blocks must have 3 entries (P2, P3, P4)")
        if base_ch % 8:
            raise ValueError(f"base_ch should be a multiple of 8 for clean "
                             f"channel splitting on Neural-ART, got {base_ch}")

        c_stem = int(stem_ch)          # 16  stride 2   (narrow: peak RAM)
        c2 = base_ch * 2               # 48  stride 4   (width recovered here)
        c_out = base_ch * 4            # 96  strides 8/16/32
        self.base_ch = int(base_ch)
        self.p5_dilation = int(p5_dilation)

        self.stem = NIRStem(out_ch=c_stem, n_edge_init=n_edge_init)

        # STYOLO projection: recover width on the already-downsampled map.
        self.widen1 = StrideDown(c_stem, c2)
        self.stage1 = CSPBlock(c2, c2, n=n_blocks[0])

        self.down2 = StrideDown(c2, c_out)
        self.stage2 = CSPBlock(c_out, c_out, n=n_blocks[1])

        self.down3 = StrideDown(c_out, c_out)
        self.stage3 = CSPBlock(c_out, c_out, n=n_blocks[2])

        self.down4 = StrideDown(c_out, c_out)
        # P5 is a semantic context level, not a counting level: depth 1, but
        # the single block is dilated so it sees roughly twice the spatial
        # extent. Free in parameters and MACs; ST supports dilation and only
        # warns against LARGE factors.
        self.stage4 = CSPBlock(c_out, c_out, n=1,
                               dilate_last=self.p5_dilation)

        self.out_channels: Tuple[int, int, int] = (c_out, c_out, c_out)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)                       # (B, 16, H/2,  W/2)
        x = self.stage1(self.widen1(x))        # (B, 48, H/4,  W/4)
        p3 = self.stage2(self.down2(x))        # (B, 96, H/8,  W/8)
        p4 = self.stage3(self.down3(p3))       # (B, 96, H/16, W/16)
        p5 = self.stage4(self.down4(p4))       # (B, 96, H/32, W/32)
        return p3, p4, p5


if __name__ == "__main__":
    m = NIRBackbone().eval()
    p3, p4, p5 = m(torch.zeros(1, 1, 288, 512))
    print("P3", tuple(p3.shape), "P4", tuple(p4.shape), "P5", tuple(p5.shape))
    print("expect (1,96,36,64) (1,96,18,32) (1,96,9,16)")
    print("out_channels", m.out_channels)
    print("params", f"{sum(p.numel() for p in m.parameters()):,}")
    n_dw = sum(1 for mod in m.modules()
               if isinstance(mod, nn.Conv2d) and mod.groups > 1)
    print("grouped/depthwise convs:", n_dw, "(must be 0)")
    assert n_dw == 0
