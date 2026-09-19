"""
test_model.py — architecture invariants
========================================
    python test_model.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from config import NUM_CLASSES, get_config
from model import build_nirdet

_failures: list = []


def _synthetic_cfg():
    """Priors are UNSET by default and that is fatal (model.build_nirdet).
    These are SYNTHETIC demo values — the same 0.05/0.15 convention as
    model.py's __main__ — deliberately not any dataset measurement."""
    return get_config(model=dict(prior_w=0.05, prior_h=0.15))


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _failures.append(msg)


def t1_no_grouped_convs() -> None:
    print("\nT1  zero grouped / depthwise convolutions")
    net = build_nirdet(_synthetic_cfg())
    n = net.count_grouped_convs()
    # Not a support question: ST lists DEPTHWISE_CONV_2D as HW-mapped. It is
    # an occupancy question. A 3x3 with groups=channels gives the four CONV
    # accelerators almost nothing to parallelise across, and on the Pi's
    # Cortex-A76 depthwise kernels are memory-bandwidth bound while dense INT8
    # GEMM exploits SDOT.
    check(n == 0, f"count_grouped_convs() == 0 (got {n})")
    offenders = [name for name, m in net.named_modules()
                 if isinstance(m, nn.Conv2d) and m.groups > 1]
    check(not offenders, f"no grouped Conv2d modules (found {offenders[:4]})")


def t2_three_blobs() -> None:
    print("\nT2  forward_raw gives 3 levels x 3 blobs")
    cfg = _synthetic_cfg()
    net = build_nirdet(cfg).eval()
    x = torch.zeros(1, 1, cfg.data.img_h, cfg.data.img_w)
    with torch.no_grad():
        raw = net.forward_raw(x)
    check(len(raw) == len(cfg.model.strides),
          f"{len(raw)} levels == {len(cfg.model.strides)}")
    ok = True
    for lvl, (c, o, s) in enumerate(raw):
        stride = cfg.model.strides[lvl]
        eh, ew = cfg.data.img_h // stride, cfg.data.img_w // stride
        good = (len(raw[lvl]) == 3 and c.shape[1] == NUM_CLASSES
                and o.shape[1] == 2 and s.shape[1] == 2
                and tuple(c.shape[2:]) == (eh, ew)
                and tuple(o.shape[2:]) == (eh, ew)
                and tuple(s.shape[2:]) == (eh, ew))
        ok = ok and good
        print(f"        L{lvl} s{stride}: cls {tuple(c.shape)} "
              f"off {tuple(o.shape)} size {tuple(s.shape)}")
    check(ok, "every level emits cls(1ch) / off(2ch) / size(2ch) at H/s x W/s")
    check(len(net.head.output_names()) == 3 * len(cfg.model.strides),
          f"output_names() has {len(net.head.output_names())} entries")


def t3_packed_shapes() -> None:
    print("\nT3  packed training shapes at 288x512")
    cfg = _synthetic_cfg()
    check((cfg.data.img_h, cfg.data.img_w) == (288, 512),
          f"canvas is {cfg.data.img_h}x{cfg.data.img_w}")
    net = build_nirdet(cfg)
    net.train()
    x = torch.zeros(1, 1, cfg.data.img_h, cfg.data.img_w)
    out = net(x, training_mode=True)
    got = [tuple(t.shape) for t in out]
    want = [(1, 2304, 5), (1, 576, 5), (1, 144, 5)]
    print(f"        got  {got}")
    print(f"        want {want}   (64*36 + 32*18 + 16*9 = 3024 cells)")
    check(got == want, "packed shapes match")
    check(sum(s[1] for s in got) == 3024, "total cells == 3024")


def t4_eaa_calibration() -> None:
    print("\nT4  EAA calibration")
    cfg = _synthetic_cfg()
    net = build_nirdet(cfg)
    check(not net.eaa.is_calibrated,
          "a fresh model is NOT calibrated (eaa_proj_bias is None)")

    img = torch.full((4, 1, cfg.data.img_h, cfg.data.img_w), 0.15)
    img[:, :, 80:180, 100:120] = 0.80          # sparse vertical structure
    img[:, :, 60:200, 300:316] = 0.70
    img = (img + 0.02 * torch.randn_like(img)).clamp(0, 1)

    bias = net.calibrate_eaa(img, verbose=False)
    check(bias is not None, "calibrate_eaa returned a bias")
    check(net.eaa.is_calibrated, "model.eaa.is_calibrated is True")
    check(abs(net.eaa.proj_bias_value - float(bias)) < 1e-6,
          f"proj.bias == returned bias ({net.eaa.proj_bias_value:+.5f})")

    # The gate must actually vary spatially; a constant ~0.48 everywhere is
    # exactly the inert behaviour the calibration exists to remove.
    e8 = net.eaa.compute_edge_magnitude(img)
    check(tuple(e8.shape[1:]) == (cfg.model.eaa_filters,
                                  cfg.data.img_h // 8, cfg.data.img_w // 8),
          f"edge map is {tuple(e8.shape)} (stride "
          f"{net.eaa.total_edge_stride})")
    gate = torch.sigmoid(net.eaa.proj(e8))
    spread = float(gate.max() - gate.min())
    print(f"        gate min/mean/max "
          f"{float(gate.min()):.3f}/{float(gate.mean()):.3f}/"
          f"{float(gate.max()):.3f}  spread {spread:.3f}")
    check(spread > 0.02, "the calibrated gate varies spatially (not a "
                         "near-uniform rescale)")
    check(float(net.eaa._gate_span) >= 0.25,
          f"_gate_span {float(net.eaa._gate_span):.4f} >= 0.25 "
          f"(calibration achieved a real spatial modulation)")

    second = net.calibrate_eaa(img, verbose=False)
    check(second is None, "a second calibrate_eaa call is a no-op "
                          "(force=True required)")


def t5_p5_ablation() -> None:
    print("\nT5  P5 ablation: strides=(8, 16)")
    cfg = get_config(model=dict(strides=(8, 16), prior_w=0.05, prior_h=0.15))
    net = build_nirdet(cfg)
    net.train()
    x = torch.zeros(1, 1, cfg.data.img_h, cfg.data.img_w)
    out = net(x, training_mode=True)
    check(len(out) == 2, f"{len(out)} prediction levels (expect 2)")
    feats = net.forward_features(x)
    check(len(feats) == 2, f"{len(feats)} feature maps (expect 2)")
    with torch.no_grad():
        raw = net.forward_raw(x)
    check(len(raw) == 2 and all(len(r) == 3 for r in raw),
          "2 levels x 3 blobs")
    check(net.head.output_names() == ["cls8", "off8", "size8",
                                      "cls16", "off16", "size16"],
          f"output_names() == {net.head.output_names()}")


def t6_inference_path() -> None:
    print("\nT6  inference decode returns boxes + scores only")
    cfg = _synthetic_cfg()
    net = build_nirdet(cfg).eval()
    x = torch.rand(2, 1, cfg.data.img_h, cfg.data.img_w)
    with torch.no_grad():
        res = net(x, training_mode=False, score_thresh=0.05)
    check(len(res) == 2, "one result per batch item")
    check(len(res[0]) == 2,
          "each result is [boxes, scores] — single class, so no labels")
    b, s = res[0]
    check(b.ndim == 2 and b.shape[-1] == 4, f"boxes {tuple(b.shape)} is (N,4)")
    check(s.ndim == 1, f"scores {tuple(s.shape)} is (N,)")


def t7_param_breakdown() -> None:
    print("\nT7  param_breakdown")
    net = build_nirdet(_synthetic_cfg())
    pb = net.param_breakdown()
    for k in ("backbone", "eaa", "neck", "head", "total"):
        check(k in pb, f"breakdown has '{k}' = {pb.get(k, 0):,}")
    check(pb["total"] == pb["backbone"] + pb["eaa"] + pb["neck"] + pb["head"],
          "sections sum to the total")
    check(net.neck.out_channels_per_level == (48, 64, 64),
          f"neck widths {net.neck.out_channels_per_level} == (48, 64, 64)")
    check(net.backbone.out_channels == (96, 96, 96),
          f"backbone widths {net.backbone.out_channels} == (96, 96, 96)")


def main() -> int:
    print("=" * 70)
    print("  model architecture tests")
    print("=" * 70)
    t1_no_grouped_convs()
    t2_three_blobs()
    t3_packed_shapes()
    t4_eaa_calibration()
    t5_p5_ablation()
    t6_inference_path()
    t7_param_breakdown()
    print("=" * 70)
    if _failures:
        print(f"  {len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  all model tests PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
