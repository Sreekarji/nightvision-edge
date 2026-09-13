"""
attention.py — Edge-Aware Attention (EAA), export-safe and numerically active
=============================================================================

WHY THE OLD MODULE WAS INERT
----------------------------
``compute_edge_magnitude`` ran the Sobel convolution at full resolution and
then average-pooled by 8, i.e. it averaged |edge| over a 64-pixel area. A
pedestrian silhouette is a SPARSE edge inside that window, so the pooled value
is diluted by edge density and typically lands in 0.01-0.05 on [0,1] NIR. With
unit-sum ``proj`` weights and bias -0.1 that gives

    a_raw = sum(w * e) - 0.1  ~=  0.03 - 0.1  =  -0.07    everywhere
    sigmoid(a_raw)            ~=  0.48                     everywhere
    gain = 1 + 2.0 * 0.48     ~=  1.96                     everywhere

which is a near-uniform rescale. The BatchNorm immediately downstream exists
to absorb exactly that kind of rescaling, so the module contributed nothing.

TWO FIXES
---------
1. STRIDE-2 EDGE CONV, POOL BY 4 (was: full-res conv, pool by 8).
   The convolution now runs on the stride-2 map and is reduced by 4 to reach
   stride 8. Three consequences:
     * peak activation for this branch drops 4x, which matters because the
       STM32N6 degrades sharply past ~4 MB when it spills to external memory;
     * the constant-padded canvas border no longer smears a full-resolution
       zero edge into the first pooled row/column;
     * every pooling kernel stays at 2 (ST: AveragePool kernels must be 1-3,
       larger windows are decomposed by the compiler), and at 512 canvas width
       the pooled map is 256 wide x 4 ch = 1024, inside ST's 2048
       width x channel pooling line-buffer limit.

2. ``calibrate_bias(images)``.
   The projection bias is MEASURED from the real pooled edge statistics and
   set to -(mean + 0.5*std), so the sigmoid straddles a useful part of its
   range instead of collapsing to 0.48. It cannot be hardcoded: it depends on
   the sensor, the 850 nm illuminator beam profile, the flat-field map and the
   CLAHE settings. train.py calls this on the first real batch, BEFORE the
   edge kernels unfreeze, and writes the result back to
   cfg.model.eaa_proj_bias so it lands in the checkpoint.

OTHER EXPORT NOTES
------------------
* ``normalize_edges`` defaults False. Per ST's operator table, Div maps to
  hardware only when the second operand is a constant; a per-image
  ``e / e.mean()`` divisor is a runtime tensor, so it becomes SW_INT on the
  Cortex-M55 and it makes the INT8 activation range scene-dependent.
  Brightness stabilisation belongs in preprocessing (flat-field + CLAHE).
* Padding mode is ``zeros``: ST maps Pad "partial - according to the
  parameters", and reflect is the case that falls back.
* ``residual_scale`` 2.0 -> gain in [1.0, 3.0]. At 0.5 the gain was confined
  to [1.0, 1.5] immediately before a BatchNorm.
* ``_current_epoch`` is restored from ``_epoch_buf`` in
  ``_load_from_state_dict``, so resuming at epoch 40 does not re-freeze the
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

    ST Neural-ART decomposes AveragePool windows with height or width above 3,
    and NCNN's pooling fast paths are tuned for 2x2/3x3. Factorising 4 -> 2*2
    keeps every kernel inside the hardware-friendly range.
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
            f"ratios factorise into 2s and 3s")
    return x


class EdgeAwareAttention(nn.Module):
    """
    Spatial gate derived from learnable, Sobel-initialised edge filters.

    Usage (once per forward, reused across levels)::

        e8 = eaa.compute_edge_magnitude(img)     # (B, N, H/8, W/8)
        p3 = eaa.apply_to(p3, e8)                # ratio 1 -> no pooling
        p4 = eaa.apply_to(p4, e8)                # ratio 2 -> avg_pool2d(2)
        p5 = eaa.apply_to(p5, e8)                # ratio 4 -> avg_pool2d(2) x2

    The total edge stride is ``edge_stride * pool_factor`` and must equal the
    finest detection stride (8).
    """

    def __init__(
        self,
        num_edge_filters: int = 4,
        freeze_epochs: int = 5,
        residual_scale: Optional[float] = 2.0,
        normalize_edges: bool = False,
        padding_mode: str = "zeros",
        edge_stride: int = 2,
        pool_factor: int = 4,
        proj_bias: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_edge_filters < 2:
            raise ValueError("num_edge_filters must be >= 2")
        if padding_mode not in ("zeros", "replicate", "reflect"):
            raise ValueError(f"bad padding_mode '{padding_mode}'")
        if edge_stride not in (1, 2):
            raise ValueError("edge_stride must be 1 or 2 (ST: horizontal "
                             "strides other than 1/2/4 lose data reuse)")
        if pool_factor < 1:
            raise ValueError("pool_factor must be >= 1")

        self.N = int(num_edge_filters)
        self.freeze_epochs = int(freeze_epochs)
        self.residual_scale = residual_scale
        self.normalize_edges = bool(normalize_edges)
        self.padding_mode = padding_mode
        self.edge_stride = int(edge_stride)
        self.pool_factor = int(pool_factor)

        self._current_epoch: int = 0

        self.edge_conv = nn.Conv2d(
            1, self.N, kernel_size=3, stride=self.edge_stride, padding=1,
            bias=False, padding_mode=padding_mode,
        )
        self.proj = nn.Conv2d(self.N, 1, kernel_size=1, bias=True)

        self._init_weights(proj_bias)
        self.register_buffer("_epoch_buf", torch.zeros(1, dtype=torch.long))
        # 0 = bias is the uncalibrated default, 1 = calibrate_bias() has run.
        # Buffered so a resumed run does not silently recalibrate.
        self.register_buffer("_calibrated", torch.zeros(1, dtype=torch.long))
        if proj_bias is not None:
            self._calibrated.fill_(1)
        self._update_grad_state()

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #

    @property
    def total_edge_stride(self) -> int:
        return self.edge_stride * self.pool_factor

    @property
    def is_calibrated(self) -> bool:
        return bool(int(self._calibrated.reshape(-1)[0].item()))

    @property
    def proj_bias_value(self) -> float:
        return float(self.proj.bias.detach().reshape(-1)[0].item())

    # ------------------------------------------------------------------ #
    # init
    # ------------------------------------------------------------------ #

    def _init_weights(self, proj_bias: Optional[float]) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.edge_conv.weight, a=0.01)
            for i in range(min(self.N, len(_TEMPLATES))):
                self.edge_conv.weight[i] = _TEMPLATES[i][0]

            # Unit-sum positive projection: proj then computes the mean edge
            # response across filters, which is exactly the statistic
            # calibrate_bias() measures. The bias is a PLACEHOLDER until
            # calibration; -0.1 is not a meaningful value for the stride-2 /
            # pool-4 edge distribution and is only here so an uncalibrated
            # forward pass does not produce a degenerate gate.
            self.proj.weight.fill_(1.0 / float(self.N))
            self.proj.bias.fill_(-0.1 if proj_bias is None else float(proj_bias))

    # ------------------------------------------------------------------ #
    # bias calibration
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def calibrate_bias(self, img: torch.Tensor,
                       std_mult: float = 0.5,
                       verbose: bool = True) -> float:
        """
        Measure the real pooled edge statistics and set proj.bias so the
        sigmoid spans a useful range.

            a_raw(x) = (W * e)(x) + b          with sum(W) = 1
            b        = -(mean(W*e) + std_mult * std(W*e))

        With std_mult = 0.5 the flat majority of the frame sits BELOW
        sigmoid 0.5 (attenuated) and pixels more than half a standard
        deviation above the mean edge response sit above it (amplified). That
        is a gate; a constant 0.48 is not.

        Call on a real batch (after flat-field + CLAHE + letterbox, i.e. the
        exact deployment preprocessing), before the edge kernels unfreeze.

        img : (B, 1, H, W) in [0, 1]
        ->    the bias value that was set
        """
        if img.dim() != 4 or img.shape[1] != 1:
            raise ValueError(f"expected (B, 1, H, W), got {tuple(img.shape)}")
        if img.shape[0] < 1:
            raise ValueError("need at least one image to calibrate")

        was_training = self.training
        self.eval()
        try:
            e = self._edge_maps(img.to(self.proj.weight.device,
                                       dtype=self.proj.weight.dtype))
            # Pre-bias projection response. Done with the real weights so a
            # later change to the proj init cannot desynchronise the two.
            s = F.conv2d(e, self.proj.weight, bias=None)       # (B, 1, H/8, W/8)
            mean = float(s.mean().item())
            std = float(s.std(unbiased=False).item())
            bias = -(mean + float(std_mult) * std)
            self.proj.bias.fill_(bias)
            self._calibrated.fill_(1)

            if verbose:
                gate_flat = torch.sigmoid(torch.tensor(mean + bias))
                gate_edge = torch.sigmoid(torch.tensor(mean + 2.0 * std + bias))
                rs = 1.0 if self.residual_scale is None else float(self.residual_scale)
                print(f"[eaa] calibrated on {img.shape[0]} image(s): "
                      f"edge response mean {mean:.5f} std {std:.5f} -> "
                      f"proj.bias {bias:+.5f}")
                print(f"[eaa]   gain at mean response      : "
                      f"{1.0 + rs * float(gate_flat):.3f}")
                print(f"[eaa]   gain at mean + 2 sigma      : "
                      f"{1.0 + rs * float(gate_edge):.3f}")
                if std < 1e-6:
                    print("[eaa]   ! edge response has ~zero variance; check "
                          "that the calibration batch is real imagery and "
                          "that flat-field/CLAHE ran")
            return bias
        finally:
            if was_training:
                self.train()

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

    def _edge_maps(self, img: torch.Tensor) -> torch.Tensor:
        """|edge| at stride ``edge_stride``, pooled to ``total_edge_stride``."""
        e = torch.abs(self.edge_conv(img))
        return _pool_by_factor(e, self.pool_factor)

    def compute_edge_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        """
        img : (B, 1, H, W) in [0, 1]
        ->    (B, N, H/total_edge_stride, W/total_edge_stride), all >= 0

        The convolution runs at stride 2 (edges are a high-frequency cue, but
        the full-resolution tensor is the peak-RAM contributor and its first
        row/column is contaminated by the constant letterbox pad), and the
        result is reduced to stride 8 with two 2x2 average pools.
        """
        e = self._edge_maps(img)
        if self.normalize_edges:
            # Research parity only. Do NOT enable for export: the divisor is a
            # runtime tensor, which ST maps as SW_INT.
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
                        f"{he}x{we} and feature {hf}x{wf}")
                e_map = _pool_by_factor(e_map, fh)
            else:
                # Feature finer than the edge map. Constant scale_factor keeps
                # the Resize export-friendly; ST maps Resize Nearest on HW
                # only with coordinate_transformation_mode='asymmetric' and
                # nearest_mode='floor', which is what opset-12
                # mode="nearest" + scale_factor produces.
                if hf % he or wf % we:
                    raise ValueError(
                        f"cannot align edge map {he}x{we} to feature {hf}x{wf}")
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
    edge_stride: int = 2,
    pool_factor: int = 4,
    proj_bias: Optional[float] = None,
) -> EdgeAwareAttention:
    return EdgeAwareAttention(
        num_edge_filters=num_edge_filters,
        freeze_epochs=freeze_epochs,
        residual_scale=residual_scale,
        normalize_edges=normalize_edges,
        padding_mode=padding_mode,
        edge_stride=edge_stride,
        pool_factor=pool_factor,
        proj_bias=proj_bias,
    )


if __name__ == "__main__":
    eaa = build_eaa()
    # Synthetic frame with sparse structure, so the calibration numbers are
    # not those of uniform noise.
    img = torch.zeros(4, 1, 288, 512)
    img += 0.15
    img[:, :, 100:200, 120:140] = 0.75          # a few bright vertical bars
    img[:, :, 90:210, 300:316] = 0.65
    img += 0.02 * torch.randn_like(img)
    img.clamp_(0.0, 1.0)

    print("total edge stride:", eaa.total_edge_stride)
    print("calibrated before:", eaa.is_calibrated)
    eaa.calibrate_bias(img)
    print("calibrated after :", eaa.is_calibrated,
          f"bias {eaa.proj_bias_value:+.5f}")

    e8 = eaa.compute_edge_magnitude(img)
    print("e8", tuple(e8.shape), "(expect 4x4x36x64)")
    gate = torch.sigmoid(eaa.proj(e8))
    print(f"gate min/mean/max: {float(gate.min()):.3f} / "
          f"{float(gate.mean()):.3f} / {float(gate.max()):.3f}")
    print("P3", tuple(eaa.apply_to(torch.randn(4, 48, 36, 64), e8).shape))
    print("P4", tuple(eaa.apply_to(torch.randn(4, 64, 18, 32), e8).shape))
    print("P5", tuple(eaa.apply_to(torch.randn(4, 64, 9, 16), e8).shape))
    print("params", sum(p.numel() for p in eaa.parameters()))
