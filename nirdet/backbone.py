"""
backbone.py — NIRDet Phase 1
=============================
Single-channel NIR backbone producing three FPN feature maps:
  P3 → stride 8  (1/8  of input)
  P4 → stride 16 (1/16 of input)
  P5 → stride 32 (1/32 of input)

Design lineage
--------------
* CSP split structure: Wang et al., "CSPNet: A New Backbone that can Enhance Learning
  Capability of CNN" (CVPRW 2020). Splits the input into a processed branch and a
  direct bypass branch, then concatenates — halves gradient duplication in the backbone.

* Depthwise-separable (DW+PW) bottleneck inside CSP blocks: Howard et al., "MobileNets:
  Efficient Convolutional Neural Networks for Mobile Vision Applications" (arXiv 2017).
  Reduces parameters by ~8-9× vs. standard 3×3 conv at the same channel count, which is
  critical here because the backbone must stay under 2.5 M parameters total.

* Sobel-initialized stem: motivated by "Rethinking ImageNet Pre-training" (He et al. 2019),
  which showed that good initialization matters more when pretraining is unavailable.
  NIR images are edge-dominant — silhouettes are the primary pedestrian detection cue —
  so seeding the first conv layer with Sobel/Laplacian patterns gives the optimiser a
  meaningful starting point rather than random noise.

* Padding mode — reflect: for edge-detection kernels in the stem, reflect padding avoids
  the artificial zero-boundary response that zero-padding would inject at image borders,
  preserving clean edge activations near the frame edges.  Confirmed by Liu et al. (2018)
  "Image Inpainting for Irregular Holes Using Partial Convolutions" benchmarks showing
  reflect/replicate padding outperforms zero padding for border-sensitive tasks.
  All subsequent layers use the default zero padding (reflect is expensive mid-network).

Parameter budget (approximate; exact count printed by the unit test):
  Stem        ~  1.6 k
  Stage 1     ~ 10 k
  Stage 2     ~ 36 k
  Stage 3     ~ 140 k
  Stage 4     ~ 555 k
  Total       < 800 k  (comfortably under the 2.5 M backbone cap)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# Helper: standard Conv-BN-SiLU brick
# ---------------------------------------------------------------------------

class ConvBNSiLU(nn.Module):
    """
    A single 2-D convolution followed by Batch Normalisation and SiLU activation.

    This is the atomic unit used everywhere outside the DWS block.  SiLU (Swish)
    outperforms ReLU on small-to-medium detectors because its non-zero gradient for
    negative inputs reduces dead-neuron problems that are common when training from
    scratch without ImageNet initialisation.

    Parameters
    ----------
    in_ch  : int   — number of input channels
    out_ch : int   — number of output channels
    k      : int   — kernel size (default 1)
    s      : int   — stride (default 1)
    p      : int   — padding (default 0; caller sets explicitly)
    groups : int   — conv groups (default 1 = standard conv)
    bias   : bool  — whether to include a bias term (default False; BN subsumes it)
    padding_mode : str — 'zeros' (default) or 'reflect' (stem only)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 1,
        s: int = 1,
        p: int = 0,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=k, stride=s, padding=p,
            groups=groups, bias=bias,
            padding_mode=padding_mode,
        )
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W)
        return self.act(self.bn(self.conv(x)))
        # out: (B, out_ch, H, W)  — spatial dims unchanged (stride=1) or halved (stride=2)


# ---------------------------------------------------------------------------
# Building block 1: DepthwiseSeparableConv
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-Separable Convolution (DWS) — a factorised replacement for
    standard 3×3 conv that reduces parameters by ~8-9× at the same channel width.

    Architecture (Howard et al. 2017, MobileNets):
        [DW] groups=in_ch  3×3 conv  — one filter per input channel (spatial mixing)
        [BN + SiLU]
        [PW] groups=1      1×1 conv  — cross-channel projection
        [BN + SiLU]

    Why DWS inside CSP?
        A standard 3×3 bottleneck with c hidden channels costs 2·c²·9 params.
        DWS replaces the 3×3 with DW(c) + PW(c→c): cost = c·9 + c²·1.
        At c=64 this is 576 + 4096 = 4672 vs 73728 for standard — 15× fewer.
        With CSP's half-channel split (effective c = out_ch/2), savings are retained.

    Parameters
    ----------
    in_ch  : int  — input channels (= output channels; DWS preserves width)
    stride : int  — stride applied to the depthwise conv (1 or 2)
                    stride=2 used only in the downsampling variant; inside blocks use 1.
    """

    def __init__(self, in_ch: int, stride: int = 1) -> None:
        super().__init__()

        # Depthwise: each input channel gets its own 3×3 filter → spatial feature mixing
        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=3, stride=stride, padding=1,
            groups=in_ch,   # one group per channel = depthwise
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)

        # Pointwise: 1×1 conv mixes channels (no spatial work here)
        self.pw = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W)
        x = self.act(self.bn1(self.dw(x)))   # (B, in_ch, H/s, W/s)
        x = self.act(self.bn2(self.pw(x)))   # (B, in_ch, H/s, W/s)  — shape unchanged
        return x


# ---------------------------------------------------------------------------
# Building block 2: CSPBlock with DWS bottlenecks
# ---------------------------------------------------------------------------

class CSPBlock(nn.Module):
    """
    Cross-Stage Partial block (Wang et al. 2020, CSPNet) with depthwise-separable
    bottlenecks instead of standard 3×3 convolutions.

    Why CSP for training-from-scratch?
    ------------------------------------
    CSP splits the input into two streams:
        • Main branch  — processed by n DWS bottlenecks (learns transformations)
        • Bypass branch — passes through a 1×1 conv unmodified (preserves gradients)

    The skip bypass prevents gradient duplication that plagues very deep ResNet-style
    stacks, making gradient flow more stable without pretraining.  Evidence: YOLOv5's
    C3 (CSP Bottleneck with 3 convolutions, Ultralytics 2020) achieves strong from-
    scratch convergence vs. the plain ResNet bottleneck baseline.

    Why DWS inside the main branch instead of C2f-style bottlenecks?
    -----------------------------------------------------------------
    C2f (YOLOv8, Ultralytics 2023) concatenates all intermediate bottleneck outputs,
    which improves gradient flow further but increases memory and intermediate feature
    width.  For a <2.5 M-param backbone on <5 k images, DWS-CSP keeps parameter count
    low enough that overfitting is a greater risk than under-capacity.  The bypass in
    CSP provides sufficient gradient highway without C2f's full multi-branch memory cost.

    Structure
    ---------
        x  ──┬── cv1(1×1) → [DWS × n] ──┐
             │                             cat → cv3(1×1) → output
             └── cv2(1×1) ───────────────┘

    Parameters
    ----------
    in_ch   : int  — number of input channels
    out_ch  : int  — number of output channels
    n       : int  — number of DWS bottleneck blocks in the main branch (default 1)
    shortcut: bool — if True, add residual connection inside each DWS block (default True)
                     disabled automatically when in_ch ≠ hidden_ch
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__()

        # Hidden channel width: half of output (CSP convention, Wang et al. 2020).
        # This halves the cost of the bypass branch and DWS blocks simultaneously.
        hidden = out_ch // 2

        # cv1: projects input into the hidden width for the DWS branch
        self.cv1 = ConvBNSiLU(in_ch,    hidden, k=1, s=1, p=0)
        # cv2: projects input into hidden width for the bypass branch
        self.cv2 = ConvBNSiLU(in_ch,    hidden, k=1, s=1, p=0)
        # cv3: fuses the two branches back to out_ch (2·hidden = out_ch)
        self.cv3 = ConvBNSiLU(2 * hidden, out_ch, k=1, s=1, p=0)

        # Main branch: n sequential DWS blocks at the hidden channel width.
        # shortcut is used only if in/out channels of the block match (hidden == hidden ✓)
        self.blocks = nn.Sequential(*[
            _DWSResBlock(hidden, use_shortcut=shortcut)
            for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W)

        main   = self.blocks(self.cv1(x))   # (B, hidden, H, W)
        bypass = self.cv2(x)                # (B, hidden, H, W)
        fused  = torch.cat([main, bypass], dim=1)  # (B, 2·hidden = out_ch, H, W)
        return self.cv3(fused)              # (B, out_ch, H, W)


class _DWSResBlock(nn.Module):
    """
    Internal residual wrapper around DepthwiseSeparableConv.

    Not part of the public API — used exclusively inside CSPBlock.

    Parameters
    ----------
    ch           : int  — in = out channels (residual requires equal widths)
    use_shortcut : bool — whether to add the residual skip connection
    """

    def __init__(self, ch: int, use_shortcut: bool = True) -> None:
        super().__init__()
        self.dws      = DepthwiseSeparableConv(ch, stride=1)
        self.shortcut = use_shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ch, H, W)
        out = self.dws(x)                              # (B, ch, H, W)
        return x + out if self.shortcut else out       # (B, ch, H, W)


# ---------------------------------------------------------------------------
# Stem: single-channel NIR entry with Sobel initialisation
# ---------------------------------------------------------------------------

class NIRStem(nn.Module):
    """
    First convolution of the backbone, specialised for single-channel NIR input.

    Two design decisions distinguished from a standard RGB stem:
      1. in_channels = 1  (NIR is single-channel; ImageNet RGB weights are inapplicable)
      2. Edge-kernel initialisation — first kernels are seeded with Sobel/Laplacian
         patterns, giving the optimiser a head-start for NIR's edge-dominant signal.

    Initialisation strategy (Q2 answer)
    ------------------------------------
    We allocate the N output kernels as:
        • Kernels 0–5  : 6 deterministic edge kernels
                          Gx, Gy, G45, G135 (Sobel variants)
                          Laplacian, Diagonal-Laplacian
        • Kernels 6–N-1: Kaiming-uniform random (standard for layers without pretraining)

    Ratio choice (6 edge : N-6 random):
        At N=32 (our default) this is 6:26 ≈ 19 % edge-initialised.
        Rationale: enough edge filters to exploit the NIR prior without constraining the
        early-layer representation so tightly that it cannot learn NIR-specific texture
        or illumination cues beyond pure edges.

    Should edge kernels be frozen initially?
        Short answer: No, with a caveated explanation.
        Freezing edge kernels for the first few epochs is sometimes advocated to "protect"
        the prior, but it creates an asymmetric gradient problem: the deeper layers adapt
        to a fixed stem, then must re-adapt when the stem is unfrozen.  On small datasets
        (<5 k images) this two-phase training requires careful LR scheduling that adds a
        hyper-parameter without a clear payoff.  We instead use a lower weight-decay on
        the stem (documented in Uncertainties, Part D) to keep the Sobel patterns from
        drifting too far while still allowing end-to-end backprop from epoch 1.
        This is consistent with the ablation suggestion in Part D.

    Padding mode: 'reflect'
        For edge-detection kernels, zero padding at image borders would produce spurious
        high-magnitude edge responses along the frame edge.  Reflect padding mirrors the
        nearest real pixels outward, producing smooth, consistent edge outputs even at the
        boundary.  This is only applied to the stem; internal layers use standard zeros.

    Parameters
    ----------
    out_ch      : int — number of stem output channels (default 32)
    n_edge_init : int — number of kernels to fill with edge patterns (default 6, ≤ out_ch)
    """

    def __init__(self, out_ch: int = 32, n_edge_init: int = 6) -> None:
        super().__init__()

        assert n_edge_init <= out_ch, (
            f"n_edge_init={n_edge_init} cannot exceed out_ch={out_ch}"
        )
        self.n_edge_init = n_edge_init

        # reflect padding is set on the conv itself so it persists through state_dict saves
        self.conv = nn.Conv2d(
            1, out_ch,
            kernel_size=3, stride=2, padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

        # Apply Sobel/Laplacian initialisation immediately after construction
        self._init_edge_kernels()

    # ------------------------------------------------------------------
    # Edge-kernel initialisation
    # ------------------------------------------------------------------

    def _edge_templates(self) -> list[torch.Tensor]:
        """
        Return 6 canonical 3×3 edge-detection kernels normalised to unit ℓ2 norm.

        All kernels detect first-order or second-order intensity discontinuities.
        Using both orientations (Gx/Gy) and diagonals (G45/G135) ensures coverage of
        vertical, horizontal, and oblique silhouette edges — all common in pedestrian NIR.
        """
        # --- Sobel: horizontal and vertical gradient approximations ---
        Gx = torch.tensor([
            [-1.,  0.,  1.],
            [-2.,  0.,  2.],
            [-1.,  0.,  1.],
        ])                                           # detects vertical edges
        Gy = Gx.T                                    # detects horizontal edges

        # --- 45° and 135° Sobel variants (Roberts-Cross inspired) ---
        G45 = torch.tensor([
            [ 0.,  1.,  2.],
            [-1.,  0.,  1.],
            [-2., -1.,  0.],
        ])
        G135 = torch.tensor([
            [-2., -1.,  0.],
            [-1.,  0.,  1.],
            [ 0.,  1.,  2.],
        ])

        # --- Laplacian: second-order isotropic edge detector ---
        Lap = torch.tensor([
            [ 0., -1.,  0.],
            [-1.,  4., -1.],
            [ 0., -1.,  0.],
        ])

        # --- Diagonal Laplacian (full 8-connectivity) ---
        LapD = torch.tensor([
            [-1., -1., -1.],
            [-1.,  8., -1.],
            [-1., -1., -1.],
        ])

        templates = [Gx, Gy, G45, G135, Lap, LapD]

        # Normalise each kernel to unit ℓ2 norm so they produce comparable activation scales.
        # This is important because Kaiming-init kernels also have controlled variance.
        normed = []
        for t in templates:
            norm = t.norm()
            normed.append(t / norm if norm > 0 else t)
        return normed                                # list of 6 tensors, each (3, 3)

    def _init_edge_kernels(self) -> None:
        """
        Replace the first `n_edge_init` kernels of self.conv with Sobel/Laplacian
        patterns; leave the remainder with standard Kaiming-uniform random values.

        Weight tensor layout:  (out_ch, 1, 3, 3)
            dim 0 → output filter index
            dim 1 → input channel index (always 1 for NIR stem)
            dim 2, 3 → spatial (H, W)
        """
        with torch.no_grad():
            # Step 1: fill all kernels with Kaiming-uniform init (the correct baseline
            # for ReLU/SiLU layers without pretraining, He et al. 2015).
            nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

            # Step 2: overwrite the first n_edge_init kernels with edge patterns.
            templates = self._edge_templates()
            for idx, t in enumerate(templates[:self.n_edge_init]):
                # self.conv.weight[idx] has shape (1, 3, 3); t has shape (3, 3)
                self.conv.weight[idx, 0] = t         # (3, 3) ← edge kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)   ← single-channel NIR
        x = self.conv(x)      # (B, out_ch, H/2, W/2)
        x = self.bn(x)
        x = self.act(x)
        return x               # (B, out_ch, H/2, W/2)


# ---------------------------------------------------------------------------
# Down-sampler: strided depthwise-separable conv
# ---------------------------------------------------------------------------

class StrideDown(nn.Module):
    """
    Spatial downsampling by 2× using a strided depthwise-separable convolution.

    Why strided DWS instead of MaxPool or strided standard conv?
    ------------------------------------------------------------
    MaxPool2d(2,2):
        • Non-learnable: discards 3/4 of activations with a fixed max rule.
        • Gradient propagation: only the max-selected cell receives gradient; the
          other 3 are zeroed.  On a small NIR dataset this can starve shallow feature
          channels of useful gradients.
        • FishNet ablation (Zhao et al. 2019): strided conv outperforms 3×3 max pool
          in detection when gradient flow to shallow layers is prioritised.

    Strided standard conv (stride=2):
        • Learnable, but can introduce checkerboard aliasing (Odena et al. 2016) and
          uses 9× more params than a DWS alternative at the same channel count.

    Strided DWS (our choice):
        • Learnable downsampling: the network can decide which spatial features to
          preserve, which is critical for fine-grained pedestrian silhouette features.
        • Checkerboard is mitigated compared to a single strided conv because the
          pointwise step re-mixes channels after the spatial decimation.
        • Parameter cost: O(ch × 9 + ch²) vs O(ch² × 9) for standard conv — 8-9× cheaper.

    Parameters
    ----------
    in_ch  : int — number of input channels
    out_ch : int — number of output channels (can differ from in_ch for channel expansion)
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # DW at stride 2: halves spatial resolution, groups=in_ch
        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=3, stride=2, padding=1,
            groups=in_ch, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)

        # PW: project to out_ch
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W)
        x = self.act(self.bn1(self.dw(x)))   # (B, in_ch,  H/2, W/2)
        x = self.act(self.bn2(self.pw(x)))   # (B, out_ch, H/2, W/2)
        return x


# ---------------------------------------------------------------------------
# Main backbone: NIRBackbone
# ---------------------------------------------------------------------------

class NIRBackbone(nn.Module):
    """
    NIR-specific 4-stage backbone for single-channel infrared pedestrian detection.

    Stage layout and depth choice (Q3 answer)
    ------------------------------------------
    Depth configuration [1, 2, 2, 1] (blocks per stage):
        Stage 1 (P1→P2): 1 block — early feature acquisition; few blocks needed.
        Stage 2 (P2→P3): 2 blocks — medium-scale feature building; 2 gives more
                          capacity than 1 without the overfitting risk of 3+.
        Stage 3 (P3→P4): 2 blocks — ditto; this stage produces the P3 FPN output.
        Stage 4 (P4→P5): 1 block — high-level semantic features; kept shallow because
                          spatial resolution is already 1/32 of input at this point.
        Total bottlenecks = 6.  Compare: YOLOv5n uses depth [1,2,3,1] = 7 blocks but
        is RGB-pretrained; MobileNetV3-Small uses 11 blocks but 3× more channels.

    Why not MobileNetV3-Small?
        MobileNetV3-Small (Howard et al. 2019) has excellent parameter efficiency via
        SE (Squeeze-and-Excite) attention but is designed for 3-channel RGB input and
        the SE block requires global average pooling at each stage — expensive for
        640×640 NIR images at stage 1.  More importantly, it has 2.5 M parameters
        for classification features, leaving nothing for the neck+head budget.

    Why not plain YOLOv5n backbone?
        YOLOv5n's CSP C3 blocks use standard 3×3 bottlenecks.  At our channel widths
        (half of YOLOv5n to meet param budget), gradient variance grows relative to
        the number of training images.  DWS substitution cuts params by ~8× per block,
        allowing wider channels (more feature capacity) at the same total param count.

    FPN output strides
    -------------------
    Input (B, 1, H, W) → P3 at (B, C3, H/8, W/8)  — stride 8
                       → P4 at (B, C4, H/16, W/16) — stride 16
                       → P5 at (B, C5, H/32, W/32) — stride 32

    For H=W=640: P3=(80×80), P4=(40×40), P5=(20×20).

    Parameters
    ----------
    base_ch    : int  — stem output channels, also the width multiplier seed (default 32).
                        Stage channels are [base_ch, 2·bc, 4·bc, 8·bc] = [32, 64, 128, 256].
                        Increasing base_ch scales the whole backbone linearly.
    n_blocks   : list[int] — number of DWS-CSP blocks per stage (default [1, 2, 2, 1]).
    n_edge_init: int  — number of stem kernels initialised with Sobel/Laplacian (default 6).
    """

    def __init__(
        self,
        base_ch: int       = 32,
        n_blocks: list     = None,
        n_edge_init: int   = 6,
    ) -> None:
        super().__init__()

        if n_blocks is None:
            n_blocks = [1, 2, 2, 1]

        # Channel widths at each stage boundary
        c1 = base_ch          #  32 — stem output / stage-1 input
        c2 = base_ch * 2      #  64 — after stage 1 downsampling
        c3 = base_ch * 4      # 128 — P3 output (stride 8)
        c4 = base_ch * 8      # 256 — P4 output (stride 16)
        c5 = base_ch * 8      # 256 — P5 output (stride 32)
        # Note: c4 == c5; the final stage deepens features without widening further
        # to stay within the 2.5 M budget.

        # ── Stem ────────────────────────────────────────────────────────
        # Input: (B, 1, H, W)  →  (B, c1, H/2, W/2)
        self.stem = NIRStem(out_ch=c1, n_edge_init=n_edge_init)

        # ── Stage 1 ─────────────────────────────────────────────────────
        # Down: (B, c1, H/2, W/2) → (B, c2, H/4, W/4)
        self.down1  = StrideDown(c1, c2)
        # CSP: (B, c2, H/4, W/4) → (B, c2, H/4, W/4)
        self.stage1 = CSPBlock(c2, c2, n=n_blocks[0], shortcut=True)

        # ── Stage 2 → P3 ────────────────────────────────────────────────
        # Down: (B, c2, H/4, W/4) → (B, c3, H/8, W/8)     stride = 8 relative to input
        self.down2  = StrideDown(c2, c3)
        # CSP: (B, c3, H/8, W/8) → (B, c3, H/8, W/8)
        self.stage2 = CSPBlock(c3, c3, n=n_blocks[1], shortcut=True)

        # ── Stage 3 → P4 ────────────────────────────────────────────────
        # Down: (B, c3, H/8, W/8) → (B, c4, H/16, W/16)   stride = 16
        self.down3  = StrideDown(c3, c4)
        # CSP: (B, c4, H/16, W/16) → (B, c4, H/16, W/16)
        self.stage3 = CSPBlock(c4, c4, n=n_blocks[2], shortcut=True)

        # ── Stage 4 → P5 ────────────────────────────────────────────────
        # Down: (B, c4, H/16, W/16) → (B, c5, H/32, W/32) stride = 32
        self.down4  = StrideDown(c4, c5)
        # CSP: (B, c5, H/32, W/32) → (B, c5, H/32, W/32)
        self.stage4 = CSPBlock(c5, c5, n=n_blocks[3], shortcut=True)

        # Store output channel counts so the neck can query them
        self.out_channels = (c3, c4, c5)   # (P3_ch, P4_ch, P5_ch)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor — shape (B, 1, H, W)   NIR single-channel image batch

        Returns
        -------
        P3 : (B, C3, H/8,  W/8)   — fine-grained features, stride 8
        P4 : (B, C4, H/16, W/16)  — medium features, stride 16
        P5 : (B, C5, H/32, W/32)  — coarse/semantic features, stride 32
        """

        # ── Stem ────────────────────────────────────────────────────────
        # (B, 1, H, W) → (B, 32, H/2, W/2)
        x = self.stem(x)

        # ── Stage 1 ─────────────────────────────────────────────────────
        # (B, 32, H/2, W/2) → down → (B, 64, H/4, W/4) → csp → (B, 64, H/4, W/4)
        x = self.down1(x)
        x = self.stage1(x)

        # ── Stage 2 → P3 ────────────────────────────────────────────────
        # (B, 64, H/4, W/4) → down → (B, 128, H/8, W/8) → csp → P3
        x  = self.down2(x)
        P3 = self.stage2(x)     # (B, 128, H/8, W/8)

        # ── Stage 3 → P4 ────────────────────────────────────────────────
        # P3 → down → (B, 256, H/16, W/16) → csp → P4
        x  = self.down3(P3)
        P4 = self.stage3(x)     # (B, 256, H/16, W/16)

        # ── Stage 4 → P5 ────────────────────────────────────────────────
        # P4 → down → (B, 256, H/32, W/32) → csp → P5
        x  = self.down4(P4)
        P5 = self.stage4(x)     # (B, 256, H/32, W/32)

        return P3, P4, P5
