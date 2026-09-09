"""
attention.py — Edge-Aware Attention (EAA), export-safe rewrite
===============================================================
What changed and why
--------------------
1. ``normalize_edges`` defaults to False. The old per-image
   ``e / e.mean((2,3))`` emitted ReduceMean + Div-by-runtime-tensor: not
   mappable on Neural-ART (HW Div needs a constant divisor) and it made the
   INT8 activation range scene-dependent. Brightness stabilisation now
   belongs in preprocessing (CLAHE in dataset.py / live_nirdet.py).

2. Padding mode is ``zeros``. Reflect Pad is only partially HW-mapped.

3. The edge magnitude map is produced ONCE at stride 8 by three stacked
   ``avg_pool2d(kernel=2, stride=2)`` calls. Every pooling kernel is <= 3, so
   nothing is decomposed by the Neural-ART compiler. The old code ran
   ``adaptive_avg_pool2d`` with implicit 4x / 8x / 16x / 32x windows.

4. ``residual_scale`` defaults to 2.0 -> gain in [1.0, 3.0]. At 0.5 the gain
   was confined to [1.0, 1.5] immediately before a BatchNorm whose whole job
   is absorbing exactly that kind of rescaling, which made the module close to
   inert.

5. ``proj`` is initialised with unit-sum weights and bias -0.1, derived for
   *unnormalised* |Sobel| responses on [0,1] NIR (typical 0.05-0.30). The old
   bias of -1.0 assumed mean-normalised inputs and would have pinned the
   sigmoid near 0.27 once normalisation was removed.

6. ``_current_epoch`` is restored from the ``_epoch_buf`` buffer inside
   ``_load_from_state_dict``, so resuming at epoch 40 no longer re-freezes the
   Sobel kernels for another 5 epochs.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 3x3 kernels normalised into [-1, 1]; shape (1, 1, 3, 3) each.
_SOBEL_X = torch.tensor([[[[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]]]) / 4.0
_SOBEL_Y = torch.tensor([[[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]]]) / 4.0
_LAPLACIAN = torch.tensor([[[[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]]]) / 8.0
_DIAG_POS = torch.tensor([[[[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]]]]) / 4.0
_DIAG_NEG = torch.tensor([[[[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]]]]) / 4.0

_TEMPLATES: List[torch.Tensor] = [
    _SOBEL_X, _SOBEL_Y, _LAPLACIAN, _DIAG_POS, _DIAG_NEG,
]


def _pool_by_factor(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Average-pool by an integer factor using only kernels of size 2 or 3.

    Neural-ART decomposes AveragePool windows larger than 3, and NCNN's
    pooling fast paths are tuned for 2x2/3x3. Factorising 8 -> 2*2*2 keeps
    every kernel inside the hardware-friendly range.
    """
    if factor <= 1:
        return x
    f = factor
    while f % 2 == 0:
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        f //= 2
    while f % 3 == 0:
        x = F.avg_pool2d(x, kernel_size=3, stride=3)
        f //= 3
    if f != 1:
        raise ValueError(
            f"pool factor {factor} is not 2^a*3^b; choose strides whose "
            f"ratios factorise into 2s and 3s"
        )
    return x


class EdgeAwareAttention(nn.Module):
    """
    Spatial gate derived from learnable, Sobel-initialised edge filters.

    Usage (once per forward, reused across levels)::

        e8 = eaa.compute_edge_magnitude(img)     # (B, N, H/8, W/8)
        p3 = eaa.apply_to(p3, e8)                # ratio 1 -> no pooling
        p4 = eaa.apply_to(p4, e8)                # ratio 2 -> avg_pool2d(2)
    """

    def __init__(
        self,
        num_edge_filters: int = 4,
        freeze_epochs: int = 5,
        residual_scale: Optional[float] = 2.0,
        normalize_edges: bool = False,
        padding_mode: str = "zeros",
        edge_stride: int = 8,
    ) -> None:
        super().__init__()
        if num_edge_filters < 2:
            raise ValueError("num_edge_filters must be >= 2")
        if padding_mode not in ("zeros", "replicate", "reflect"):
            raise ValueError(f"bad padding_mode '{padding_mode}'")
        if edge_stride < 1:
            raise ValueError("edge_stride must be >= 1")

        self.N = int(num_edge_filters)
        self.freeze_epochs = int(freeze_epochs)
        self.residual_scale = residual_scale
        self.normalize_edges = bool(normalize_edges)
        self.padding_mode = padding_mode
        self.edge_stride = int(edge_stride)

        self._current_epoch: int = 0

        self.edge_conv = nn.Conv2d(
            1, self.N, kernel_size=3, stride=1, padding=1,
            bias=False, padding_mode=padding_mode,
        )
        self.proj = nn.Conv2d(self.N, 1, kernel_size=1, bias=True)

        self._init_weights()
        self.register_buffer("_epoch_buf", torch.zeros(1, dtype=torch.long))
        self._update_grad_state()

    # ------------------------------------------------------------------ #
    # init
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.edge_conv.weight, a=0.01)
            for i in range(min(self.N, len(_TEMPLATES))):
                self.edge_conv.weight[i] = _TEMPLATES[i][0]

            # Unit-sum positive projection: with |edge| in ~[0, 0.4] on [0,1]
            # NIR, a_raw = sum(w * e) - 0.1 straddles zero, so flat regions
            # sit just below sigmoid 0.5 and edges rise above it.
            self.proj.weight.fill_(1.0 / float(self.N))
            self.proj.bias.fill_(-0.1)

    # ------------------------------------------------------------------ #
    # freeze schedule
    # ------------------------------------------------------------------ #

    def step_epoch(self) -> None:
        self._current_epoch += 1
        self._epoch_buf.fill_(self._current_epoch)
        self._update_grad_state()

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)
        self._epoch_buf.fill_(self._current_epoch)
        self._update_grad_state()

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def edges_frozen(self) -> bool:
        return self.freeze_epochs > 0 and self._current_epoch < self.freeze_epochs

    def _update_grad_state(self) -> None:
        frozen = self.edges_frozen
        for p in self.edge_conv.parameters():
            p.requires_grad_(not frozen)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Restore the epoch counter and freeze state on resume."""
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        key = prefix + "_epoch_buf"
        if key in state_dict:
            try:
                self._current_epoch = int(state_dict[key].reshape(-1)[0].item())
            except Exception:
                self._current_epoch = 0
        else:
            self._current_epoch = int(self._epoch_buf.reshape(-1)[0].item())
        self._epoch_buf.fill_(self._current_epoch)
        self._update_grad_state()

    # ------------------------------------------------------------------ #
    # edge magnitude
    # ------------------------------------------------------------------ #

    def compute_edge_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        """
        img : (B, 1, H, W) in [0, 1]
        ->    (B, N, H/edge_stride, W/edge_stride), all values >= 0

        The convolution runs at full resolution (edges are a high-frequency
        cue) and the result is immediately reduced to ``edge_stride`` with
        2x2 average pools, so downstream levels only ever pool by a further
        factor of 2 or 3.
        """
        e = torch.abs(self.edge_conv(img))
        e = _pool_by_factor(e, self.edge_stride)
        if self.normalize_edges:
            # Kept for research parity only. Do NOT enable for export.
            e = e / (e.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return e

    # ------------------------------------------------------------------ #
    # apply
    # ------------------------------------------------------------------ #

    def apply_to(self, feat: torch.Tensor, e_map: torch.Tensor) -> torch.Tensor:
        """
        feat  : (B, C, Hf, Wf)
        e_map : (B, N, He, We) from compute_edge_magnitude
        ->      (B, C, Hf, Wf)
        """
        hf, wf = feat.shape[-2], feat.shape[-1]
        he, we = e_map.shape[-2], e_map.shape[-1]

        if (he, we) != (hf, wf):
            if he >= hf and wf > 0 and he % hf == 0 and we % wf == 0:
                fh, fw = he // hf, we // wf
                if fh != fw:
                    raise ValueError(
                        f"anisotropic pool ratio {fh}x{fw} between edge map "
                        f"{he}x{we} and feature {hf}x{wf}"
                    )
                e_map = _pool_by_factor(e_map, fh)
            else:
                # Feature is finer than the edge map (e.g. a stride-4 level).
                # Constant scale_factor keeps Resize export-friendly.
                if hf % he or wf % we:
                    raise ValueError(
                        f"cannot align edge map {he}x{we} to feature {hf}x{wf}"
                    )
                e_map = F.interpolate(
                    e_map, scale_factor=float(hf // he),
                    mode="nearest", recompute_scale_factor=False,
                )

        attn = torch.sigmoid(self.proj(e_map))          # (B, 1, Hf, Wf)
        if self.residual_scale is not None:
            return feat * (1.0 + float(self.residual_scale) * attn)
        return feat * attn

    def forward(self, feat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Convenience path; recomputes the edge map. Prefer the two-step API."""
        return self.apply_to(feat, self.compute_edge_magnitude(img))


def build_eaa(
    num_edge_filters: int = 4,
    freeze_epochs: int = 5,
    residual_scale: Optional[float] = 2.0,
    normalize_edges: bool = False,
    padding_mode: str = "zeros",
    edge_stride: int = 8,
) -> EdgeAwareAttention:
    return EdgeAwareAttention(
        num_edge_filters=num_edge_filters,
        freeze_epochs=freeze_epochs,
        residual_scale=residual_scale,
        normalize_edges=normalize_edges,
        padding_mode=padding_mode,
        edge_stride=edge_stride,
    )


if __name__ == "__main__":
    eaa = build_eaa()
    img = torch.rand(2, 1, 384, 640)
    e8 = eaa.compute_edge_magnitude(img)
    print("e8", tuple(e8.shape))
    print("P3", tuple(eaa.apply_to(torch.randn(2, 128, 48, 80), e8).shape))
    print("P4", tuple(eaa.apply_to(torch.randn(2, 256, 24, 40), e8).shape))
    print("params", sum(p.numel() for p in eaa.parameters()))
