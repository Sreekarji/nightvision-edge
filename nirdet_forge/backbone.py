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
Not a support question — ST lists DEPTHWISE_CONV_2D as hardware-mapped — but
an occupancy one. A 3x3 with groups=channels gives the four CONV accelerators
almost nothing to parallelise across, and on the Pi's Cortex-A76 depthwise
kernels are memory-bandwidth bound while dense INT8 GEMM exploits SDOT.
STResNet/STYOLO (arXiv:2601.05364) rejected depthwise separable convolutions
and built exclusively from standard 3x3 and 1x1, reporting lower RAM
(1.39 vs 2.01 MB) and lower latency (21.3 vs 22.4 ms) than MobileNetV2-1.0.

STAGE WIDTH: base_ch * 4 = 96 EVERYWHERE
----------------------------------------
96 is inside ST's 72..128 input-channel window for 1x1 kernels, is 4*24 (an
ideal output-channel multiple), non-prime and a multiple of 8.

STEM PROJECTION (STResNet/STYOLO section 5.3)
---------------------------------------------
The stem emits only ``stem_ch`` channels on the stride-2 map — the largest
tensor in the network — and width is recovered on the stride-4 map. STYOLO
measured peak RAM 4.26 -> 2.46 MB (-42%) from exactly this change.

ACTIVATION / PADDING
--------------------
ReLU6 (ONNX Clip), unconditionally HW-mapped. Padding is zeros.

ACTIVATION RANGE (resolved)
---------------------------
DenseResBlock.forward clamps x + out to [0, 6], preserving the ReLU6
guarantee through residual connections. A two-block stage therefore also
stays in [0, 6]. See attention.py RANGE LEDGER for how this propagates
through EAA and the FPN.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from attention import _edge_templates_canonical


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """Conv -> BatchNorm -> ReLU6. BN folds into the conv at export.

    BatchNorm, never GroupNorm: Neural-ART has no GroupNorm mapping, and GN
    also blocks NCNN's BN-fusion pass.
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
    Dense 3x3 residual block: one full 3x3 convolution, no depthwise split.

    ``dilation`` is used only by the deepest stage.

    Post-add clamp to [0, 6] ensures the residual output stays within the
    ReLU6 contract. ONNX Clip, HW-mapped.
    """

    def __init__(self, ch: int, shortcut: bool = True, dilation: int = 1) -> None:
        super().__init__()
        self.conv = ConvBNAct(ch, ch, k=3, s=1, d=int(dilation))
        self.shortcut = bool(shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if not self.shortcut:
            return out
        # RANGE LEDGER: both x and out are in [0,6] after ReLU6. The unclamped
        # add gives [0,12]; a two-block stage reaches [0,18], costing ~1.6 bits
        # of INT8 resolution and compounding with EAA (attention.py range ledger).
        # Clamp here, once, before the INT8 activation range is measured.
        # ONNX Clip is unconditionally HW-mapped on Neural-ART and NCNN.
        return torch.clamp(x + out, 0.0, 6.0)


class CSPBlock(nn.Module):
    """
    x -> cv1(1x1, out/2) -> [dense 3x3 res] * n --.
      -> cv2(1x1, out/2) ------------------------ concat -> cv3(1x1, out)

    ``dilate_last`` applies the dilation to the FINAL residual block only.
    Concat is HW-mapped on Neural-ART.
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
    """Learnable 2x downsample: a single dense strided 3x3."""

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

    The first ``n_edge_init`` filters are seeded with Sobel / Laplacian
    kernels — 850 nm reflective NIR is an edge-rich, texture-poor modality, so
    oriented gradients are the right inductive bias. The rest are
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

    def _init_edge_kernels(self) -> None:
        """
        Kaiming the whole tensor, then overwrite the edge slots — RESCALED to
        the std of the slots they replace (F6).

        The templates are unit-L2 (L2 = 1.0) while a Kaiming slot with
        a=sqrt(5) has L2 ~= 0.57, so seeding them raw pushed the edge filters
        into the stem BatchNorm roughly 1.7x hotter than the random ones.
        This is the same normalisation train.imagenet_stem_init already
        applies to its own slots.
        """
        templates = _edge_templates_canonical()
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
            ref_std = float(self.conv.weight.std())
            for i, t in enumerate(templates[: self.n_edge_init]):
                t_std = float(t.std())
                scale = (ref_std / t_std) if t_std > 1e-12 else 1.0
                self.conv.weight[i, 0] = t[0] * scale

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

    out_channels = (base_ch*4,) * 3 = (96, 96, 96) at base_ch=24. All three
    are equal on purpose: the neck projects every level to a common width
    anyway, so a wider deep stage was paying for channels that a 1x1
    immediately discarded.
    """

    def __init__(self, base_ch: int = 24,
                 stem_ch: int = 16,
                 n_blocks: Optional[Tuple[int, ...]] = None,
                 n_edge_init: int = 6,
                 p5_dilation: int = 2) -> None:
        super().__init__()
        n_blocks = tuple(n_blocks) if n_blocks is not None else (1, 2, 2, 1)
        if len(n_blocks) not in (3, 4):
            raise ValueError(
                "n_blocks must have 3 or 4 entries: (stride-4, stride-8, stride-16) "
                "plus an optional 4th for the stride-32 stage (default 1). "
                "The stride-32 stage is fixed at depth 1 when 3 entries are given.")
        if len(n_blocks) == 3:
            n_blocks = n_blocks + (1,)
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
        # extent. Free in parameters and MACs.
        self.stage4 = CSPBlock(c_out, c_out, n=n_blocks[3],
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

    # F6: the seeded edge filters must enter the BatchNorm at the same
    # magnitude as the Kaiming slots they sit beside.
    s = NIRStem(out_ch=16, n_edge_init=6)
    edge_std = float(s.conv.weight[:6].std())
    rand_std = float(s.conv.weight[6:].std())
    ratio = edge_std / rand_std
    print(f"stem std: edge slots {edge_std:.5f} vs random slots "
          f"{rand_std:.5f}  ratio {ratio:.3f} (want ~1.0)")
    assert 0.8 < ratio < 1.25, \
        f"edge/random std ratio {ratio:.3f} outside [0.8, 1.25]; " \
        f"NIRStem._init_edge_kernels rescaling is broken"
    # Also verify every template is zero-mean (non-zero-mean templates like a
    # box filter would make the ratio test pass trivially).
    for i, t in enumerate(_edge_templates_canonical()[: s.n_edge_init]):
        mean_abs = float(t.mean().abs())
        assert mean_abs < 1e-5, \
            f"template {i} has non-zero mean {mean_abs:.2e}; " \
            f"the std ratio test is not sensitive to mean shifts"
    print("stem std assertions passed")
