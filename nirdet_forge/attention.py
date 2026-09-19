"""
attention.py — Edge-Aware Attention (EAA), export-safe and numerically active
=============================================================================

RANGE LEDGER
------------
* DenseResBlock output (backbone F5/F6): [0, 12] per stage; [0, 18] after a
  two-block stage. Clamped to [0, 6] by backbone.DenseResBlock.forward.
* EAA apply_to output: feat*(0.5+attn) clamped to [0,6] (this file,
  apply_to). Range-preserving: gate is monotone across the full [0,6]
  input range. At attn=0.5 (calibrated mean) it is the identity.
* FPN fusions (neck F78): td4 = l4+up5, td3 = l3+up4 are unactivated.
  Clamped by neck.LightweightFPN when fuse_clip=True (neck.py).
All three clamps are ONNX Clip nodes, unconditionally HW-mapped on
Neural-ART and on the Pi's NCNN fast path.

WHY THE OLD MODULE WAS INERT
----------------------------
``compute_edge_magnitude`` ran the Sobel convolution at full resolution and
then average-pooled by 8, i.e. it averaged |edge| over a 64-pixel area. A
pedestrian silhouette is a SPARSE edge inside that window, so the pooled value
is diluted and typically lands in 0.01-0.05. With unit-sum ``proj`` weights and
bias -0.1 the gate collapsed to sigmoid(-0.07) ~= 0.48 everywhere — a
near-uniform rescale that the downstream BatchNorm absorbs entirely.

TWO FIXES
---------
1. STRIDE-2 EDGE CONV, POOL BY 4 (was: full-res conv, pool by 8). Peak
   activation for this branch drops 4x, the constant-padded canvas border no
   longer smears into the first pooled row/column, and every pooling kernel
   stays at 2 (ST: AveragePool kernels must be 1-3).
2. ``calibrate_bias(images)``. The projection bias is MEASURED from the real
   pooled edge statistics and set to -(mean + 0.5*std). It cannot be
   hardcoded: it depends on the sensor, the 850 nm illuminator beam profile,
   the flat-field map and the CLAHE settings.

OTHER EXPORT NOTES
------------------
* ``normalize_edges`` defaults False: a per-image ``e / e.mean()`` divisor is
  a runtime tensor, so ST maps it SW_INT.
* Padding mode is ``zeros``; reflect Pad falls back to software.
* ``_current_epoch`` is restored from ``_epoch_buf`` in
  ``_load_from_state_dict``, so resuming at epoch 40 does not re-freeze the
  Sobel kernels for another 5 epochs.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 3x3 seed kernels, unit-L2, shape (1, 3, 3) each.
def _edge_templates_canonical() -> list:
    """
    Canonical edge templates shared by NIRStem (backbone.py) and EAA
    (attention.py). Each is a (1,3,3) float32 tensor, rescaled to unit-L2
    so callers can normalise at their own call site.

    Sign convention: positive response on a bright-left / dark-right edge
    (Sobel-x). NIRStem takes abs() via BN; EAA takes abs() explicitly.
    Both are followed by BN, so the sign is absorbed.
    """
    Gx  = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]) / 4.0
    Gy  = torch.tensor([[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]) / 4.0
    G45 = torch.tensor([[[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]]]) / 4.0
    G135= torch.tensor([[[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]]]) / 4.0
    Lap = torch.tensor([[[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]]) / 4.0
    LapD= torch.tensor([[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]) / 8.0
    out = []
    for t in (Gx, Gy, G45, G135, Lap, LapD):
        n = t.norm()
        out.append(t / n if float(n) > 1e-12 else t)
    return out


# Seed templates, in priority order. At default N=4 ALL four slots are seeded
# (there are exactly 4 templates used); none stay Kaiming-uniform at N=4.
# Each template is rescaled to the std of the Kaiming-initialised slots it
# replaces so the seeded and random filters enter the pipeline at equal
# magnitude. At N=6 the last two slots stay Kaiming-uniform.
_TEMPLATES: List[torch.Tensor] = [t.unsqueeze(0) for t in _edge_templates_canonical()]


def _pool_by_factor(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Average-pool by an integer factor using only kernels of size 2 or 3.

    ST Neural-ART decomposes AveragePool windows above 3, and NCNN's pooling
    fast paths are tuned for 2x2/3x3.
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

    ``edge_stride * pool_factor`` must equal the finest detection stride (8).
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
        self.register_buffer("_calibrated", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_gate_span", torch.zeros(1))
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
        """
        The projection bias only. After calibrate_bias() this reflects ONLY
        the bias, not the gain: calibration also scales proj.weight by k, so
        the full gate span lives in the ``_gate_span`` buffer.
        """
        return float(self.proj.bias.detach().reshape(-1)[0].item())

    # ------------------------------------------------------------------ #
    # init
    # ------------------------------------------------------------------ #

    def _init_weights(self, proj_bias: Optional[float]) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.edge_conv.weight, a=0.01)
            ref_std = float(self.edge_conv.weight.std())
            for i in range(min(self.N, len(_TEMPLATES))):
                t = _TEMPLATES[i][0]
                t_std = float(t.std())
                scale = (ref_std / t_std) if t_std > 1e-12 else 1.0
                self.edge_conv.weight[i] = t * scale

            # Initial unit-sum projection: proj starts as a cross-filter mean.
            # calibrate_bias() later scales proj.weight by k = logits_per_sigma/std,
            # breaking the unit-sum property ON PURPOSE to make the gate spatially
            # active. After calibration, train.py's weight-decay exemption for
            # eaa.proj.weight is LOAD-BEARING: decay would pull k back toward 1/N
            # and re-inert the gate.
            self.proj.weight.fill_(1.0 / float(self.N))
            self.proj.bias.fill_(-0.1 if proj_bias is None else float(proj_bias))

    # ------------------------------------------------------------------ #
    # bias calibration
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def calibrate_bias(self, img: torch.Tensor,
                       std_mult: float = 0.5,
                       logits_per_sigma: float = 2.0,
                       min_gate_span: float = 0.25,
                       verbose: bool = True) -> float:
        """
        Calibrate BOTH the gain (proj.weight scale) and the bias so the gate
        spans a useful range over the real edge-response distribution.

            k = logits_per_sigma / std(s)     # scale so ±1σ = ±logits_per_sigma
            proj.weight *= k                  # unit-sum property broken ON PURPOSE
            proj.bias   = -k*(mean(s) + std_mult*std(s))

        After calibration the gate spans ≈ sigmoid(-logits_per_sigma – std_mult)
        to sigmoid(+logits_per_sigma*(2 – std_mult)), which is a real modulation
        the downstream BatchNorm cannot absorb.

        train.py's weight-decay exemption for eaa.proj.weight is LOAD-BEARING
        after this call: decay would pull k back toward 1/N and re-inert the gate.
        """
        if img.dim() != 4 or img.shape[1] != 1:
            raise ValueError(f"expected (B, 1, H, W), got {tuple(img.shape)}")
        if img.shape[0] < 1:
            raise ValueError("need at least one image to calibrate")
        if self.normalize_edges:
            raise RuntimeError(
                "calibrate_bias measures the UNNORMALISED pooled edge statistic; "
                "with normalize_edges=True the runtime divides by a per-image mean "
                "(≈ ×33), so the calibrated bias would be off by ~1/mean and the "
                "gate would saturate to 1.0 everywhere. "
                "Set normalize_edges=False (it is SW_INT on Neural-ART anyway).")

        was_training = self.training
        self.eval()
        try:
            e = self._edge_maps(img.to(self.proj.weight.device,
                                       dtype=self.proj.weight.dtype))
            s = F.conv2d(e, self.proj.weight, bias=None)   # measure BEFORE scaling
            mean = float(s.mean().item())
            std  = float(s.std(unbiased=False).item())
            if std < 1e-6:
                raise RuntimeError(
                    "edge response has ~zero variance; check that the calibration "
                    "batch is real imagery and that flat-field/CLAHE ran")
            # Scale proj.weight so that ±1σ of the edge response maps to
            # ±logits_per_sigma logits — making the gate spatially active.
            k = float(logits_per_sigma) / std
            self.proj.weight.mul_(k)          # breaks unit-sum ON PURPOSE
            bias = -k * (mean + float(std_mult) * std)
            self.proj.bias.fill_(bias)

            # Verify the resulting gate span; hard-fail if still degenerate.
            lo = float(torch.sigmoid(torch.tensor(k * (mean - 2 * std) + bias)))
            hi = float(torch.sigmoid(torch.tensor(k * (mean + 2 * std) + bias)))
            span = hi - lo
            if span < float(min_gate_span):
                raise RuntimeError(
                    f"EAA gate spans only {span:.4f} over ±2σ of the measured "
                    f"edge response (need ≥ {min_gate_span}); the gate is still a "
                    f"near-uniform rescale that the next BatchNorm will absorb. "
                    f"Raise logits_per_sigma or set eaa_residual_scale=None.")
            self._gate_span.fill_(span)
            self._calibrated.fill_(1)

            if verbose:
                rs = 1.0 if self.residual_scale is None else float(self.residual_scale)
                print(f"[eaa] calibrated on {img.shape[0]} image(s): "
                      f"mean {mean:.5f}  std {std:.5f}  k={k:.2f}  "
                      f"proj.bias {bias:+.5f}  gate_span {span:.4f}")
                print(f"[eaa]   gain at mean - 2σ : {0.5 + lo:.3f}")
                print(f"[eaa]   gain at mean + 2σ : {0.5 + hi:.3f}")

            # F11 (audit, diagnostic only — no numerics change): k and bias
            # are fitted to the UNPOOLED (stride-8) edge statistic, but
            # apply_to average-pools the edge map by 2 (P4) and 4 (P5) before
            # proj. Pooling shrinks the std, so the gate span is strictly
            # smaller at the coarse levels and min_gate_span above only
            # validates level 0. Measure and warn — a collapsed P5 gate is
            # otherwise invisible.
            for r in (2, 4):
                e_r = _pool_by_factor(e, r)
                s_r = F.conv2d(e_r, self.proj.weight, bias=self.proj.bias)
                m_r = float(s_r.mean().item())
                sd_r = float(s_r.std(unbiased=False).item())
                lo_r = float(torch.sigmoid(torch.tensor(m_r - 2.0 * sd_r)))
                hi_r = float(torch.sigmoid(torch.tensor(m_r + 2.0 * sd_r)))
                span_r = hi_r - lo_r
                if verbose or span_r < float(min_gate_span):
                    print(f"[eaa]   pool x{r}: gate span {span_r:.4f} "
                          f"[{'OK' if span_r >= float(min_gate_span) else 'WEAK'}]"
                          f"  (logit sigma {sd_r:.3f})")
                if span_r < float(min_gate_span):
                    print(f"[eaa]   WARNING the gate at pool factor {r} is a "
                          f"near-uniform rescale the following BatchNorm will "
                          f"absorb; EAA contributes little at that level. "
                          f"Raise logits_per_sigma or document EAA as "
                          f"effectively stride-8-only.")
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
            except Exception as exc:
                # NEVER silent (F4): a reset here re-freezes the Sobel kernels
                # for another freeze_epochs with no diagnostic.
                print(f"[eaa] WARNING could not restore the epoch counter from "
                      f"'{key}' ({type(exc).__name__}: {exc}); assuming 0, so "
                      f"the edge kernels will be frozen for "
                      f"{self.freeze_epochs} more epochs")
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

        The two resampling cases are split EXPLICITLY (F1). The old combined
        condition sent a non-integral DOWN ratio (e.g. edge 37x64 -> feature
        18x32) into the upsample branch, where it failed with "cannot align",
        naming the wrong problem.
        """
        hf, wf = feat.shape[-2], feat.shape[-1]
        he, we = e_map.shape[-2], e_map.shape[-1]

        if (he, we) != (hf, wf):
            if hf <= 0 or wf <= 0 or he <= 0 or we <= 0:
                raise ValueError(
                    f"degenerate spatial size: edge map {he}x{we}, "
                    f"feature {hf}x{wf}")
            if he > hf or we > wf:
                # Edge map coarser-or-equal in at least one axis -> POOL.
                if he % hf or we % wf:
                    raise ValueError(
                        f"non-integral pool ratio from edge map {he}x{we} to "
                        f"feature {hf}x{wf}")
                fh, fw = he // hf, we // wf
                if fh != fw:
                    raise ValueError(
                        f"anisotropic pool ratio {fh}x{fw} between edge map "
                        f"{he}x{we} and feature {hf}x{wf}")
                e_map = _pool_by_factor(e_map, fh)
            else:
                # Feature finer than the edge map -> UPSAMPLE. Constant
                # scale_factor keeps the Resize export-friendly; ST maps
                # Resize Nearest on HW only with
                # coordinate_transformation_mode='asymmetric' and
                # nearest_mode='floor', which opset-12 mode="nearest" +
                # scale_factor produces.
                if hf % he or wf % we:
                    raise ValueError(
                        f"non-integral upsample ratio from edge map {he}x{we} "
                        f"to feature {hf}x{wf}")
                fh, fw = hf // he, wf // we
                if fh != fw:
                    raise ValueError(
                        f"anisotropic upsample ratio {fh}x{fw} between edge "
                        f"map {he}x{we} and feature {hf}x{wf}")
                e_map = F.interpolate(
                    e_map, scale_factor=float(fh),
                    mode="nearest", recompute_scale_factor=False,
                )

        attn = torch.sigmoid(self.proj(e_map))          # (B, 1, Hf, Wf)
        if self.residual_scale is not None:
            # RANGE-PRESERVING GATE (replaces feat*(1+2*attn) clamped to 6).
            #
            # The old form: feat*(1+2*attn) clamped to 6. For feat in [0,6]
            # and attn in (0,1) the multiplier is in (1,3), so any feat >= 2
            # is partially clipped and any feat >= 6 is fully clipped — the
            # gate contributes NOTHING at the strongest activations.
            #
            # New form: feat * (0.5 + attn).
            #   attn=0  -> feat * 0.5  (suppress)
            #   attn=0.5 -> feat * 1.0  (identity, the calibrated mean)
            #   attn=1  -> feat * 1.5  (amplify)
            # Output range is [0, 9] before the clamp, so feat=6 contributes
            # gate * 6 with gate in (0.5, 1.5) — monotone across the full
            # input range. ONNX Clip node, HW-mapped on Neural-ART and NCNN.
            #
            # TRAINED-NUMERICS DECISION: must be set before the first training
            # run. The weight-decay exemption for eaa.proj.weight in train.py
            # is still load-bearing.
            out = feat * (0.5 + attn)
            return torch.clamp(out, 0.0, 6.0)   # ONNX Clip, HW-mapped
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
    img = torch.zeros(4, 1, 288, 512)
    img += 0.15
    img[:, :, 100:200, 120:140] = 0.75          # a few bright vertical bars
    img[:, :, 90:210, 300:316] = 0.65
    img += 0.02 * torch.randn_like(img)
    img.clamp_(0.0, 1.0)

    print("total edge stride:", eaa.total_edge_stride)
    print("seeded templates :", min(eaa.N, len(_TEMPLATES)), "of",
          len(_TEMPLATES), "(the rest stay Kaiming-uniform)")
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
