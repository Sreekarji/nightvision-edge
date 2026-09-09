"""
model.py — NIRDet-Lite
=======================
    input (B,1,H,W)
      EAA.compute_edge_magnitude -> e8 (B,4,H/8,W/8)             [once per forward]
      backbone                   -> P3 (B,128,H/8,  W/8)
                                    P4 (B,256,H/16, W/16)
                                    P5 (B,256,H/32, W/32)
      EAA.apply_to(P3, e8)  ratio 1
      EAA.apply_to(P4, e8)  ratio 2 -> avg_pool2d(2)
      EAA.apply_to(P5, e8)  ratio 4 -> avg_pool2d(2) x2
      LightweightFPN(out=64)     -> N3 (B,64,H/8,W/8), N4 (B,64,H/16,W/16), N5 (B,64,H/32,W/32)
      PedestrianHead(64, 1 DWS conv per branch, strides (8,16,32))

Budget at 384x640 (recomputed):
      backbone 0.64 GMAC | neck 0.18 GMAC | head 0.07 GMAC | EAA <0.01 GMAC
      total   ~0.90 GMAC, ~0.72 M params   (was 16.0 GMAC, 4.85 M)

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
        base_ch: int = 32,
        n_blocks: Optional[Sequence[int]] = (1, 2, 2),
        n_edge_init: int = 6,
        neck_channels: int = 64,
        head_channels: int = 64,
        head_branch_convs: int = 1,
        head_use_stem: bool = True,
        strides: Tuple[int, ...] = (8, 16, 32),
        num_edge_filters: int = 4,
        freeze_eaa_epochs: int = 5,
        eaa_residual_scale: Optional[float] = 2.0,
        eaa_normalize_edges: bool = False,
        eaa_padding_mode: str = "zeros",
        prior_w: float = 0.0461,
        prior_h: float = 0.1680,
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
            n_blocks=tuple(n_blocks) if n_blocks else (1, 2, 2),
            n_edge_init=n_edge_init,
        )
        c3, c4, c5 = self.backbone.out_channels

        # Edge map is produced at the finest detection stride so that every
        # per-level reduction is a factor of 2 (kernel <= 3 on all targets).
        self.eaa = EdgeAwareAttention(
            num_edge_filters=num_edge_filters,
            freeze_epochs=freeze_eaa_epochs,
            residual_scale=eaa_residual_scale,
            normalize_edges=eaa_normalize_edges,
            padding_mode=eaa_padding_mode,
            edge_stride=self.strides[0],
        )

        self.neck = LightweightFPN(c3=c3, c4=c4, c5=c5, out_channels=neck_channels)

        self.head = PedestrianHead(
            in_channels=self.neck.out_channels,
            feat_channels=head_channels,
            strides=self.strides,
            num_branch_convs=head_branch_convs,
            use_stem=head_use_stem,
            prior_prob=prior_prob,
            prior_w=prior_w,
            prior_h=prior_h,
        )

        # Every submodule initialises itself (Sobel stem, EAA kernels, focal
        # bias, box priors, Kaiming neck). Do NOT call self.apply(init_fn):
        # it would erase all four.

    # ------------------------------------------------------------------ #
    # EAA schedule
    # ------------------------------------------------------------------ #

    def step_eaa_epoch(self) -> None:
        self.eaa.step_epoch()

    def set_eaa_epoch(self, epoch: int) -> None:
        self.eaa.set_epoch(epoch)

    # ------------------------------------------------------------------ #
    # feature trunk (shared by training, inference and export)
    # ------------------------------------------------------------------ #

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns exactly len(self.strides) feature maps.

        The backbone and neck are structurally three-level (8/16/32).
        Slicing here makes strides=(8, 16) a legal config instead of a
        first-batch ValueError inside PedestrianHead.
        """
        e8 = self.eaa.compute_edge_magnitude(x)
        p3, p4, p5 = self.backbone(x)
        p3 = self.eaa.apply_to(p3, e8)
        p4 = self.eaa.apply_to(p4, e8)
        p5 = self.eaa.apply_to(p5, e8)
        n3, n4, n5 = self.neck(p3, p4, p5)
        _FEAT_BY_STRIDE = {8: n3, 16: n4, 32: n5}
        return [_FEAT_BY_STRIDE[s] for s in self.strides]

    def forward_raw(self, x: torch.Tensor
                    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """NCHW conv outputs only — the exported graph."""
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
        preds : per level (B, HW, 5) already decoded to pixels by the head.
        ->      B items, each [boxes (N,4) xyxy px, scores (N,)]

        Thresholds are arguments; ``self.nms_score_thresh`` is only the
        default so callers never need to mutate module state.
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
                valid = (bw > 1.0) & (bh > 1.0)      # drop degenerate exp() boxes
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

    def __repr__(self) -> str:
        b = self.param_breakdown()
        lines = ["NIRDet-Lite", "=" * 42]
        for k in ("backbone", "eaa", "neck", "head"):
            lines.append(f"  {k:<9}: {b[k]:>10,}")
        lines += [
            "-" * 42,
            f"  {'total':<9}: {b['total']:>10,}",
            f"  strides  : {self.strides}",
            f"  neck ch  : {self.neck.out_channels}",
            f"  head ch  : {self.head.feat_channels}",
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
        n_blocks=m.n_blocks,
        n_edge_init=m.n_edge_init,
        neck_channels=m.neck_channels,
        head_channels=m.head_channels,
        head_branch_convs=m.head_branch_convs,
        head_use_stem=m.head_use_stem,
        strides=m.strides,
        num_edge_filters=m.eaa_filters,
        freeze_eaa_epochs=m.eaa_freeze_epochs,
        eaa_residual_scale=m.eaa_residual_scale,
        eaa_normalize_edges=m.eaa_normalize_edges,
        eaa_padding_mode=m.eaa_padding_mode,
        prior_w=m.prior_w,
        prior_h=m.prior_h,
        prior_prob=m.prior_prob,
        nms_iou_thresh=m.nms_iou_thresh,
        nms_score_thresh=m.nms_score_thresh,
        max_det=m.max_det,
    )


if __name__ == "__main__":
    net = build_nirdet()
    print(net)
    x = torch.zeros(2, 1, 384, 640)
    net.train()
    tr = net(x, training_mode=True)
    print("train :", [tuple(t.shape) for t in tr])
    net.eval()
    with torch.no_grad():
        inf = net(x, training_mode=False)
    print("infer : boxes", tuple(inf[0][0].shape),
          "scores", tuple(inf[0][1].shape))
