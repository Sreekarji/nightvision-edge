"""
model.py — NIRDet-Lite
=======================
    input (B,1,288,512)
      EAA.compute_edge_magnitude  -> e8 (B,4,36,64)      [once per forward]
      backbone                    -> P3 (B,96,36,64)
                                     P4 (B,96,18,32)
                                     P5 (B,96, 9,16)
      EAA.apply_to(P3, e8)  ratio 1
      EAA.apply_to(P4, e8)  ratio 2 -> avg_pool2d(2)
      EAA.apply_to(P5, e8)  ratio 4 -> avg_pool2d(2) x2
      LightweightFPN(+PAN)        -> N3 (B,48,36,64)
                                     N4 (B,64,18,32)
                                     N5 (B,64, 9,16)
      PedestrianHead(64)          -> per level cls(1) / off(2) / size(2)

Single class throughout: the head emits one confidence channel, decode is
single-class, and decode_predictions returns boxes and scores with no label
tensor because there is nothing to label.

Training vs inference is an explicit ``training_mode`` kwarg (defaulting to
self.training), so a forgotten .eval() cannot silently change the return type.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision.ops import nms as tv_nms

from attention import EdgeAwareAttention
from backbone import NIRBackbone
from head import PedestrianHead
from neck import LightweightFPN


class NIRDet(nn.Module):
    def __init__(
        self,
        base_ch: int = 24,
        stem_ch: int = 16,
        n_blocks: Optional[Sequence[int]] = (1, 2, 2),
        n_edge_init: int = 6,
        p5_dilation: int = 2,
        neck_channels: int = 64,
        neck_out3_channels: int = 48,
        neck_pan: bool = True,
        neck_sk: bool = False,
        head_channels: int = 64,
        head_branch_convs: int = 1,
        head_use_stem: bool = True,
        head_per_level_bn: bool = True,
        strides: Tuple[int, ...] = (8, 16, 32),
        num_edge_filters: int = 4,
        freeze_eaa_epochs: int = 5,
        eaa_residual_scale: Optional[float] = 2.0,
        eaa_normalize_edges: bool = False,
        eaa_padding_mode: str = "zeros",
        eaa_edge_stride: int = 2,
        eaa_pool_factor: int = 4,
        eaa_proj_bias: Optional[float] = None,
        prior_w: float = 0.046094,
        prior_h: float = 0.179167,
        prior_prob: float = 0.01,
        nms_iou_thresh: float = 0.45,
        nms_score_thresh: float = 0.25,
        max_det: int = 300,
    ) -> None:
        super().__init__()

        self.strides = tuple(int(s) for s in strides)
        self.nms_iou_thresh = float(nms_iou_thresh)
        self.nms_score_thresh = float(nms_score_thresh)
        self.max_det = int(max_det)

        self.backbone = NIRBackbone(
            base_ch=base_ch,
            stem_ch=stem_ch,
            n_blocks=tuple(n_blocks) if n_blocks else (1, 2, 2),
            n_edge_init=n_edge_init,
            p5_dilation=p5_dilation,
        )
        c3, c4, c5 = self.backbone.out_channels

        # The edge map is produced at the finest detection stride so every
        # per-level reduction is a factor of 2 or 4 (kernel <= 3 on all
        # targets). edge_stride * pool_factor must equal strides[0].
        total = int(eaa_edge_stride) * int(eaa_pool_factor)
        if total != self.strides[0]:
            raise ValueError(
                f"eaa_edge_stride * eaa_pool_factor = {total} but the finest "
                f"detection stride is {self.strides[0]}; the edge map would "
                f"not align with P3")
        self.eaa = EdgeAwareAttention(
            num_edge_filters=num_edge_filters,
            freeze_epochs=freeze_eaa_epochs,
            residual_scale=eaa_residual_scale,
            normalize_edges=eaa_normalize_edges,
            padding_mode=eaa_padding_mode,
            edge_stride=eaa_edge_stride,
            pool_factor=eaa_pool_factor,
            proj_bias=eaa_proj_bias,
        )

        self.neck = LightweightFPN(
            c3=c3, c4=c4, c5=c5,
            out_channels=neck_channels,
            out3_channels=neck_out3_channels,
            pan=neck_pan,
            sk=neck_sk,
        )

        self.head = PedestrianHead(
            in_channels=self.neck.out_channels_per_level,
            feat_channels=head_channels,
            strides=self.strides,
            num_branch_convs=head_branch_convs,
            use_stem=head_use_stem,
            per_level_bn=head_per_level_bn,
            prior_prob=prior_prob,
            prior_w=prior_w,
            prior_h=prior_h,
        )

        # Every submodule initialises itself (Sobel stem, EAA kernels, focal
        # bias, box priors, zero-init SK gate, Kaiming neck). Do NOT call
        # self.apply(init_fn): it would erase all five.

    # ------------------------------------------------------------------ #
    # EAA schedule + calibration
    # ------------------------------------------------------------------ #

    def step_eaa_epoch(self) -> None:
        self.eaa.step_epoch()

    def set_eaa_epoch(self, epoch: int) -> None:
        self.eaa.set_epoch(epoch)

    @torch.no_grad()
    def calibrate_eaa(self, images: torch.Tensor,
                      force: bool = False, verbose: bool = True
                      ) -> Optional[float]:
        """
        Measure the real edge statistics and set the EAA projection bias.

        Call from train.py on the FIRST real batch, before epoch 0 and
        therefore before the edge kernels unfreeze. ``images`` must have gone
        through the full deployment preprocessing (flat-field, CLAHE,
        letterbox, /255), because the bias depends on all of it.

        Returns the bias that was set, or None if already calibrated and
        force is False (so a resumed run keeps the checkpointed value).
        """
        if self.eaa.is_calibrated and not force:
            if verbose:
                print(f"[eaa] already calibrated "
                      f"(bias {self.eaa.proj_bias_value:+.5f}); skipping. "
                      f"Pass force=True to recalibrate.")
            return None
        return self.eaa.calibrate_bias(images, verbose=verbose)

    # ------------------------------------------------------------------ #
    # feature trunk (shared by training, inference and export)
    # ------------------------------------------------------------------ #

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns exactly len(self.strides) feature maps.

        The backbone and neck are structurally three-level (8/16/32). Slicing
        by stride here makes strides=(8, 16) a legal config — the --p5-ablate
        experiment — instead of a first-batch ValueError inside the head.
        """
        e8 = self.eaa.compute_edge_magnitude(x)
        p3, p4, p5 = self.backbone(x)
        p3 = self.eaa.apply_to(p3, e8)
        p4 = self.eaa.apply_to(p4, e8)
        p5 = self.eaa.apply_to(p5, e8)
        n3, n4, n5 = self.neck(p3, p4, p5)
        feat_by_stride = {8: n3, 16: n4, 32: n5}
        return [feat_by_stride[s] for s in self.strides]

    def forward_raw(self, x: torch.Tensor
                    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """NCHW conv outputs only — exactly the exported graph.
        Three blobs per level: cls (1ch), off (2ch), size (2ch)."""
        return self.head.forward_raw(self.forward_features(x))

    def forward(self, x: torch.Tensor,
                training_mode: Optional[bool] = None):
        if training_mode is None:
            training_mode = self.training

        preds = self.head(self.forward_features(x), training_mode=training_mode)
        if training_mode:
            return preds
        return self.decode_predictions(preds, input_size=x.shape[-2:])

    # ------------------------------------------------------------------ #
    # post-processing (kept OUT of the exported graph)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def decode_predictions(
        self,
        preds: List[torch.Tensor],
        input_size: Tuple[int, int],
        score_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
        max_det: Optional[int] = None,
    ) -> List[List[torch.Tensor]]:
        """
        preds : per level (B, HW, 5), already decoded to pixels by the head.
        ->      B items, each [boxes (N,4) xyxy px, scores (N,)]

        Single class, so there is no per-class NMS and no label tensor.
        Thresholds are arguments; the module attributes are only defaults, so
        callers never need to mutate module state.
        """
        h, w = int(input_size[0]), int(input_size[1])
        st = self.nms_score_thresh if score_thresh is None else float(score_thresh)
        it = self.nms_iou_thresh if iou_thresh is None else float(iou_thresh)
        md = self.max_det if max_det is None else int(max_det)

        flat = torch.cat(preds, dim=1)               # (B, N, 5)
        results: List[List[torch.Tensor]] = []

        for b in range(flat.shape[0]):
            p = flat[b]
            scores = p[:, 4]
            keep = scores >= st
            p, scores = p[keep], scores[keep]

            if p.numel():
                bw, bh = p[:, 2], p[:, 3]
                valid = (bw > 1.0) & (bh > 1.0)      # drop degenerate exp boxes
                p, scores = p[valid], scores[valid]

            if p.numel() == 0:
                results.append([
                    torch.zeros((0, 4), device=flat.device, dtype=flat.dtype),
                    torch.zeros((0,), device=flat.device, dtype=flat.dtype),
                ])
                continue

            cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
            boxes = torch.stack([
                (cx - bw * 0.5).clamp(0, w),
                (cy - bh * 0.5).clamp(0, h),
                (cx + bw * 0.5).clamp(0, w),
                (cy + bh * 0.5).clamp(0, h),
            ], dim=-1)

            idx = tv_nms(boxes.float(), scores.float(), it)[:md]
            results.append([boxes[idx], scores[idx]])

        return results

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #

    def param_breakdown(self) -> dict:
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        return {
            "backbone": n(self.backbone),
            "eaa": n(self.eaa),
            "neck": n(self.neck),
            "head": n(self.head),
            "total": n(self),
        }

    def count_grouped_convs(self) -> int:
        """Depthwise / grouped convolutions. Must be 0: see backbone.py."""
        return sum(1 for m in self.modules()
                   if isinstance(m, nn.Conv2d) and m.groups > 1)

    def __repr__(self) -> str:
        b = self.param_breakdown()
        lines = ["NIRDet-Lite (single class: person)", "=" * 46]
        for k in ("backbone", "eaa", "neck", "head"):
            lines.append(f"  {k:<9}: {b[k]:>10,}")
        lines += [
            "-" * 46,
            f"  {'total':<9}: {b['total']:>10,}",
            f"  strides   : {self.strides}",
            f"  neck ch   : {self.neck.out_channels_per_level} "
            f"(pan={self.neck.use_pan} sk={self.neck.use_sk})",
            f"  head ch   : {self.head.feat_channels} "
            f"(per-level BN={self.head.per_level_bn})",
            f"  blobs/lvl : 3 (cls/off/size)",
            f"  eaa       : stride {self.eaa.edge_stride} x pool "
            f"{self.eaa.pool_factor} = {self.eaa.total_edge_stride}, "
            f"calibrated={self.eaa.is_calibrated}",
            f"  grouped convs: {self.count_grouped_convs()} (must be 0)",
        ]
        return "\n".join(lines)


def build_nirdet(cfg=None) -> NIRDet:
    """Build from config.py. Imported lazily so importing model.py is cheap."""
    if cfg is None:
        from config import get_config
        cfg = get_config()
    m = cfg.model
    return NIRDet(
        base_ch=m.base_ch,
        stem_ch=m.stem_ch,
        n_blocks=m.n_blocks,
        n_edge_init=m.n_edge_init,
        p5_dilation=m.backbone_p5_dilation,
        neck_channels=m.neck_channels,
        neck_out3_channels=m.neck_out3_channels,
        neck_pan=m.neck_pan,
        neck_sk=m.neck_sk,
        head_channels=m.head_channels,
        head_branch_convs=m.head_branch_convs,
        head_use_stem=m.head_use_stem,
        head_per_level_bn=m.head_per_level_bn,
        strides=m.strides,
        num_edge_filters=m.eaa_filters,
        freeze_eaa_epochs=m.eaa_freeze_epochs,
        eaa_residual_scale=m.eaa_residual_scale,
        eaa_normalize_edges=m.eaa_normalize_edges,
        eaa_padding_mode=m.eaa_padding_mode,
        eaa_edge_stride=m.eaa_edge_stride,
        eaa_pool_factor=m.eaa_pool_factor,
        eaa_proj_bias=m.eaa_proj_bias,
        prior_w=m.prior_w,
        prior_h=m.prior_h,
        prior_prob=m.prior_prob,
        nms_iou_thresh=m.nms_iou_thresh,
        nms_score_thresh=m.nms_score_thresh,
        max_det=m.max_det,
    )


if __name__ == "__main__":
    from config import get_config

    cfg = get_config()
    net = build_nirdet(cfg)
    print(net)
    assert net.count_grouped_convs() == 0, "depthwise conv survived"

    x = torch.zeros(2, 1, cfg.data.img_h, cfg.data.img_w)

    net.train()
    tr = net(x, training_mode=True)
    print("\ntrain packed:", [tuple(t.shape) for t in tr])

    raw = net.forward_raw(x)
    print("raw blobs/level:", len(raw[0]))
    for lvl, (c, o, s) in enumerate(raw):
        print(f"  L{lvl}: cls {tuple(c.shape)} off {tuple(o.shape)} "
              f"size {tuple(s.shape)}")

    net.eval()
    with torch.no_grad():
        inf = net(x, training_mode=False)
    print("infer: boxes", tuple(inf[0][0].shape),
          "scores", tuple(inf[0][1].shape))

    print("\nEAA calibration:")
    img = torch.full((4, 1, cfg.data.img_h, cfg.data.img_w), 0.15)
    img[:, :, 80:180, 100:120] = 0.8
    img += 0.02 * torch.randn_like(img)
    net.calibrate_eaa(img.clamp(0, 1))
    print("second call (should skip):", net.calibrate_eaa(img.clamp(0, 1)))

    print("\np5 ablation (strides 8,16):")
    ab = build_nirdet(get_config(model=dict(strides=(8, 16))))
    print("  levels:", len(ab(x, training_mode=True)))
