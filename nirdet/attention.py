"""
attention.py — Edge-Aware Attention (EAA) for NIRDet
=====================================================
Computes a spatial attention map directly from the raw NIR image using
Sobel/Laplacian-initialized learnable filters, then uses that map to
modulate backbone feature maps at any FPN scale (P3, P4, P5).

Design decisions are documented in Part A of the project notes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Kernel constants  (shape for Conv2d(1, N, 3, padding=1): (N, 1, 3, 3))
# ---------------------------------------------------------------------------

# Classic Sobel — unnormalized.  Max absolute value = 4 (corner of Sx / Sy).
# Division by 4 maps every weight to [-1, 1].
_SOBEL_X = torch.tensor(
    [[[[ 1.,  0., -1.],
       [ 2.,  0., -2.],
       [ 1.,  0., -1.]]]]
) / 4.0   # shape (1, 1, 3, 3), range [-1, 1]

_SOBEL_Y = torch.tensor(
    [[[[ 1.,  2.,  1.],
       [ 0.,  0.,  0.],
       [-1., -2., -1.]]]]
) / 4.0   # shape (1, 1, 3, 3), range [-1, 1]

# 8-connected Laplacian — max abs = 8, so /8 → [-1, 1].
_LAPLACIAN = torch.tensor(
    [[[[ 1.,  1.,  1.],
       [ 1., -8.,  1.],
       [ 1.,  1.,  1.]]]]
) / 8.0   # shape (1, 1, 3, 3), range [-1, 1]

# 45° diagonal Prewitt-style complement (optional fourth filter)
_DIAG_POS = torch.tensor(
    [[[[ 0.,  1.,  2.],
       [-1.,  0.,  1.],
       [-2., -1.,  0.]]]]
) / 2.0   # rough normalisation

_DIAG_NEG = torch.tensor(
    [[[[-2., -1.,  0.],
       [-1.,  0.,  1.],
       [ 0.,  1.,  2.]]]]
) / 2.0


class EdgeAwareAttention(nn.Module):
    """
    Edge-Aware Attention (EAA) module for NIRDet.

    Args:
        num_edge_filters (int): Number of learnable edge kernels N.
                                 Default 4 (Sx, Sy, L, diagonal pair).
                                 Must be >= 2.
        freeze_epochs (int): Keep Sobel weights frozen for this many epochs
                              then unfreeze.  0 = always trainable from start.
                              Call ``step_epoch()`` each epoch from the training
                              loop.  Default 5.
        pool_mode (str): 'avg' | 'max' for edge map downsampling.
                          'avg' is recommended (see Part A Q2).
        residual_scale (float | None): If not None, uses the additive-residual
                                        form  F*(1 + residual_scale*A)  instead
                                        of pure multiplicative  F*A.
                                        None = pure multiplicative.
                                        Recommended: 0.5 for body-context tasks.
        padding_mode (str): 'reflect' | 'replicate' | 'zeros'.
                             'reflect' recommended for boundary edges.
    """

    def __init__(
        self,
        num_edge_filters: int = 4,
        freeze_epochs: int = 5,
        pool_mode: str = "avg",
        residual_scale: Optional[float] = 0.5,
        padding_mode: str = "reflect",
    ):
        super().__init__()

        if num_edge_filters < 2:
            raise ValueError("num_edge_filters must be >= 2 (at least Sx and Sy)")
        if pool_mode not in ("avg", "max"):
            raise ValueError("pool_mode must be 'avg' or 'max'")
        if padding_mode not in ("reflect", "replicate", "zeros"):
            raise ValueError("padding_mode must be 'reflect', 'replicate', or 'zeros'")

        self.N = num_edge_filters
        self.freeze_epochs = freeze_epochs
        self.pool_mode = pool_mode
        self.residual_scale = residual_scale
        self.padding_mode = padding_mode

        self._current_epoch: int = 0

        # ── Learnable edge conv  (1, H, W) → (N, H, W) ─────────────────────
        # bias=False: bias would shift all activations uniformly and break the
        # zero-output-on-flat-region guarantee before sigmoid.
        # We do NOT use Conv2d's built-in padding here because different
        # padding modes (reflect etc.) are set per-call in forward() via
        # F.pad + padding=0.
        self.edge_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.N,
            kernel_size=3,
            stride=1,
            padding=0,   # handled manually via F.pad for configurable mode
            bias=False,
        )

        # ── 1×1 projection: N edge response channels → 1 attention weight ───
        # Includes bias so the sigmoid can be shifted away from 0.5 during
        # training; this lets flat regions suppress (< 0.5) and edge regions
        # amplify (> 0.5) asymmetrically.
        self.proj = nn.Conv2d(
            in_channels=self.N,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # ── Initialise weights ───────────────────────────────────────────────
        self._init_sobel_weights()

        # ── Register epoch tracking buffer (not a parameter) ─────────────────
        self.register_buffer("_epoch_buf", torch.zeros(1, dtype=torch.long))

        # FIX: apply the freeze state at construction (epoch 0). Without this,
        # _update_grad_state() only ran on the first step_epoch(), so the Sobel
        # edge_conv kernels were trainable during epoch 0 (default requires_grad
        # =True), silently violating the curriculum-freeze contract.
        self._update_grad_state()

    # ------------------------------------------------------------------ #
    #  Private: weight initialisation                                      #
    # ------------------------------------------------------------------ #

    def _init_sobel_weights(self) -> None:
        """
        Fill ``edge_conv.weight`` (shape: N, 1, 3, 3) with canonical
        Sobel / Laplacian / diagonal kernels, all normalised to [-1, 1].

        Allocation strategy for N kernels:
          idx 0  → Sobel-X   (vertical edge detector)
          idx 1  → Sobel-Y   (horizontal edge detector)
          idx 2  → Laplacian (omnidirectional, if N >= 3)
          idx 3  → Diag+     (45° edge, if N >= 4)
          idx 4  → Diag-     (135° edge, if N >= 5)
          idx 5+ → Kaiming-uniform random (general learnable)
        """
        kernels = [_SOBEL_X, _SOBEL_Y, _LAPLACIAN, _DIAG_POS, _DIAG_NEG]

        with torch.no_grad():
            # First, initialise everything with Kaiming uniform as fallback
            nn.init.kaiming_uniform_(self.edge_conv.weight, a=0.01)

            for i in range(min(self.N, len(kernels))):
                # kernels[i] has shape (1, 1, 3, 3); select row i in weight
                self.edge_conv.weight[i] = kernels[i]

        # proj: initialise weights to be positive so that Sobel-detected edges
        # (always non-negative after abs() in forward) map to attention > 0.5.
        # Negative proj weights would cause edges to suppress rather than amplify,
        # breaking the edge-sensitivity guarantee before training begins.
        # We use kaiming_uniform then take abs() to preserve the magnitude distribution
        # while enforcing the correct sign prior.
        nn.init.kaiming_uniform_(self.proj.weight, a=0.01)
        with torch.no_grad():
            self.proj.weight.abs_()   # all weights become non-negative
        nn.init.constant_(self.proj.bias, 0.0)

    # ------------------------------------------------------------------ #
    #  Epoch control — call from training loop                             #
    # ------------------------------------------------------------------ #

    def step_epoch(self) -> None:
        """
        Advance the internal epoch counter and toggle Sobel-weight freezing.

        Call once per epoch::

            for epoch in range(num_epochs):
                model.eaa.step_epoch()
                train_one_epoch(...)
        """
        self._current_epoch += 1
        self._epoch_buf[0] = self._current_epoch
        self._update_grad_state()

    def _update_grad_state(self) -> None:
        frozen = (
            self.freeze_epochs > 0
            and self._current_epoch <= self.freeze_epochs
        )
        for p in self.edge_conv.parameters():
            p.requires_grad = not frozen

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, feat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat : backbone feature map — (B, C, H', W')
                   e.g. P3=(B,C3,H/8,W/8), P4=(B,C4,H/16,W/16), P5=(B,C5,H/32,W/32)
            img  : raw NIR image       — (B, 1, H, W)

        Returns:
            Attended feature map       — (B, C, H', W')   same shape as feat

        Shape trace (example: img=640×640, P3 feat=80×80):
          img   : (B, 1, 640, 640)
          padded: (B, 1, 642, 642)   — reflect-pad 1 px each side
          e_map : (B, N, 640, 640)   — edge_conv
          e_mag : (B, N, 640, 640)   — abs() edge magnitude (always ≥ 0)
          e_ds  : (B, N, H', W')     — adaptive pool to feature-map size
          a_raw : (B, 1, H', W')     — 1×1 projection
          attn  : (B, 1, H', W')     — sigmoid → [0, 1]
          out   : (B, C, H', W')     — feat * attn  [or residual form]
        """
        # ── shapes ─────────────────────────────────────────────────────────
        B, C, Hf, Wf = feat.shape   # feature-map spatial size
        _, _, H, W   = img.shape    # full NIR image spatial size

        # ── 1. Pad image for 'valid' edge conv with configurable padding mode
        #       padding=1 each side preserves H×W after 3×3 conv
        img_pad = F.pad(img, (1, 1, 1, 1), mode=self.padding_mode)
        # img_pad : (B, 1, H+2, W+2)

        # ── 2. Learnable edge convolution (padding=0 because we pre-padded)
        e_map = self.edge_conv(img_pad)
        # e_map : (B, N, H, W)

        # ── 3. Absolute edge magnitude — makes the map direction-agnostic
        #       and always non-negative, which is necessary before pooling
        #       and sigmoid to ensure attention ∈ [0, 1]
        e_mag = torch.abs(e_map)
        # e_mag : (B, N, H, W)

        # ── 4. Downsample edge magnitude map to feature-map resolution
        #       adaptive_avg_pool2d chosen over bilinear (see Part A Q2):
        #         • avg pooling: each output cell = mean of input region,
        #           preserving relative edge density — strong edges stay
        #           strong relative to flat regions.
        #         • max pooling: preserves peak activations (better for
        #           point-like edges) but can amplify noise; useful for
        #           very fine structure.  Selectable via pool_mode.
        if self.pool_mode == "avg":
            e_ds = F.adaptive_avg_pool2d(e_mag, (Hf, Wf))
        else:
            e_ds = F.adaptive_max_pool2d(e_mag, (Hf, Wf))
        # e_ds : (B, N, H', W')

        # ── 5. 1×1 projection: N channels → 1 raw attention logit
        a_raw = self.proj(e_ds)
        # a_raw : (B, 1, H', W')

        # ── 6. Sigmoid → attention weights in [0, 1]
        attn = torch.sigmoid(a_raw)
        # attn : (B, 1, H', W')

        # ── 7. Apply attention to feature map
        #       Broadcast over C channels automatically (1 → C)
        if self.residual_scale is not None:
            # Additive-residual: F * (1 + α*A)
            #   • At A=0  → output = F   (no suppression of flat regions)
            #   • At A=1  → output = F*(1+α)  (amplification capped)
            #   • Preserves body-mass context even in low-edge regions
            out = feat * (1.0 + self.residual_scale * attn)
        else:
            # Pure multiplicative: F * A
            #   • At A≈0  → flat regions fully suppressed
            #   • At A≈1  → edge regions pass through unchanged
            out = feat * attn
        # out : (B, C, H', W')

        return out


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_eaa(
    num_edge_filters: int = 4,
    freeze_epochs: int = 5,
    pool_mode: str = "avg",
    residual_scale: Optional[float] = 0.5,
    padding_mode: str = "reflect",
) -> EdgeAwareAttention:
    """Build and return an EAA module with project-standard defaults."""
    return EdgeAwareAttention(
        num_edge_filters=num_edge_filters,
        freeze_epochs=freeze_epochs,
        pool_mode=pool_mode,
        residual_scale=residual_scale,
        padding_mode=padding_mode,
    )
