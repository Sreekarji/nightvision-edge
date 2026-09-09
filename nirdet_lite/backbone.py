"""
backbone.py — NIRDet-Lite backbone
===================================
Three output-bearing stages including P5 (stride 32).

  input (B,1,H,W)
    stem       3x3 s2  zeros-pad, Sobel/Laplacian-initialised  -> (B, 32, H/2,  W/2)
    down1+csp                                                   -> (B, 64, H/4,  W/4)
    down2+csp                                            = P3   -> (B,128, H/8,  W/8)
    down3+csp                                            = P4   -> (B,256, H/16, W/16)
    down4+csp                                            = P5   -> (B,256, H/32, W/32)

Export-relevant deltas from the original file
---------------------------------------------
  * SiLU -> ReLU6      (ReLU6 maps to Neural-ART / NCNN INT8 with no flags;
                        the SiLU advantage is inside noise on 261 images)
  * reflect -> zeros   (Pad is only partially HW-mapped on Neural-ART)
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
    """Conv -> BatchNorm -> ReLU6. BN folds into the conv at export."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1,
                 p: int = 0, groups: int = 1, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=p, groups=groups, bias=False,
                              padding_mode="zeros")
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWSConv(nn.Module):
    """Depthwise 3x3 (+BN+ReLU6) then pointwise 1x1 (+BN+ReLU6)."""

    def __init__(self, in_ch: int, out_ch: Optional[int] = None,
                 stride: int = 1) -> None:
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.dw = ConvBNAct(in_ch, in_ch, k=3, s=stride, p=1, groups=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class _DWSResBlock(nn.Module):
    def __init__(self, ch: int, shortcut: bool = True) -> None:
        super().__init__()
        self.dws = DWSConv(ch, ch, stride=1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dws(x)
        return x + out if self.shortcut else out


class CSPBlock(nn.Module):
    """
    x -> cv1(1x1, out/2) -> [DWS res] * n --.
      -> cv2(1x1, out/2) ------------------ concat -> cv3(1x1, out)
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1,
                 shortcut: bool = True) -> None:
        super().__init__()
        hidden = out_ch // 2
        self.cv1 = ConvBNAct(in_ch, hidden, k=1)
        self.cv2 = ConvBNAct(in_ch, hidden, k=1)
        self.cv3 = ConvBNAct(2 * hidden, out_ch, k=1)
        self.blocks = nn.Sequential(
            *[_DWSResBlock(hidden, shortcut) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.blocks(self.cv1(x))
        bypass = self.cv2(x)
        return self.cv3(torch.cat([main, bypass], dim=1))


class StrideDown(nn.Module):
    """Learnable 2x downsample: strided DW 3x3 then PW 1x1."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = DWSConv(in_ch, out_ch, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------------

class NIRStem(nn.Module):
    """
    Single-channel NIR entry conv, stride 2, zeros padding.

    The first ``n_edge_init`` filters are seeded with unit-L2 Sobel /
    Laplacian kernels; the rest are Kaiming-uniform. ``imagenet_slots``
    reports the filter indices that are free for channel-summed ImageNet
    transfer (see train.imagenet_stem_init).
    """

    def __init__(self, out_ch: int = 32, n_edge_init: int = 6) -> None:
        super().__init__()
        if n_edge_init > out_ch:
            raise ValueError(f"n_edge_init={n_edge_init} > out_ch={out_ch}")
        self.out_ch = out_ch
        self.n_edge_init = n_edge_init

        self.conv = nn.Conv2d(1, out_ch, kernel_size=3, stride=2, padding=1,
                              bias=False, padding_mode="zeros")
        self.bn = nn.BatchNorm2d(out_ch)
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
    Returns (P3, P4, P5) at strides 8, 16, and 32.

    out_channels = (base_ch * 4, base_ch * 8, base_ch * 8) = (128, 256, 256) at base_ch=32.
    """

    def __init__(self, base_ch: int = 32,
                 n_blocks: Optional[Tuple[int, ...]] = None,
                 n_edge_init: int = 6) -> None:
        super().__init__()
        n_blocks = tuple(n_blocks) if n_blocks is not None else (1, 2, 2)
        if len(n_blocks) != 3:
            raise ValueError("n_blocks must have 3 entries (P2, P3, P4)")
        # stage4 always has n=1 (P5 is a semantic context level, not a counting level)

        c1 = base_ch          # 32  stride 2
        c2 = base_ch * 2      # 64  stride 4
        c3 = base_ch * 4      # 128 stride 8   -> P3
        c4 = base_ch * 8      # 256 stride 16  -> P4
        c5 = base_ch * 8      # 256 stride 32  -> P5

        self.stem = NIRStem(out_ch=c1, n_edge_init=n_edge_init)

        self.down1 = StrideDown(c1, c2)
        self.stage1 = CSPBlock(c2, c2, n=n_blocks[0])

        self.down2 = StrideDown(c2, c3)
        self.stage2 = CSPBlock(c3, c3, n=n_blocks[1])

        self.down3 = StrideDown(c3, c4)
        self.stage3 = CSPBlock(c4, c4, n=n_blocks[2])

        self.down4 = StrideDown(c4, c5)
        self.stage4 = CSPBlock(c5, c5, n=1)

        self.out_channels: Tuple[int, int, int] = (c3, c4, c5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)                       # (B, 32, H/2,  W/2)
        x = self.stage1(self.down1(x))         # (B, 64, H/4,  W/4)
        p3 = self.stage2(self.down2(x))        # (B,128, H/8,  W/8)
        p4 = self.stage3(self.down3(p3))       # (B,256, H/16, W/16)
        p5 = self.stage4(self.down4(p4))       # (B,256, H/32, W/32)
        return p3, p4, p5


if __name__ == "__main__":
    m = NIRBackbone().eval()
    p3, p4, p5 = m(torch.zeros(1, 1, 384, 640))
    print("P3", tuple(p3.shape), "P4", tuple(p4.shape), "P5", tuple(p5.shape))
    print("params", f"{sum(p.numel() for p in m.parameters()):,}")
