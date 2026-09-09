"""
config.py — NIRDet-Lite SINGLE SOURCE OF TRUTH
===============================================
Every other module reads its defaults from here. There are no competing
declarations of input resolution, batch size, epochs, LR, or thresholds
anywhere else in the codebase.

Rules enforced by this file:
  * No function calls at import time (safe to import from any module).
  * Note: not every field is consumed by every module in the current codebase.
    Fields are kept for completeness and future use.
  * Deployment-critical constants (DECODE_OFFSET_SCALE, clamps, priors)
    live here so head.py / losses.py / export / C post-processing agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


# ===========================================================================
# DECODE CONTRACT — must be identical in head.py, losses.py, live_nirdet.py,
# and nirdet_pp.c. Changing any of these invalidates trained checkpoints.
# ===========================================================================

DECODE_OFFSET_SCALE: float = 2.0                 # offset range [-0.5, 1.5] cells
DECODE_OFFSET_BIAS: float = (DECODE_OFFSET_SCALE - 1.0) / 2.0   # 0.5

# exp() clamp on the w/h regression logits.
# Tightened from (-8, 4) to (-6, 1) for INT8: at img_w=640 this spans
# exp(-6)*640 = 1.6 px .. exp(1)*640 = 1740 px, which covers every plausible
# pedestrian while shrinking the quantisation dynamic range by ~250x.
REG_LOG_CLAMP_MIN: float = -6.0
REG_LOG_CLAMP_MAX: float = 1.0


# ===========================================================================
# SUB-CONFIGS
# ===========================================================================

@dataclass
class DataCfg:
    root: str = r"C:\projects\nightvision\data\raw\miniNIRPed"
    img_h: int = 384
    img_w: int = 640
    num_channels: int = 1
    num_classes: int = 1
    class_names: Tuple[str, ...] = ("person",)
    num_workers: int = 4
    pin_memory: bool = True
    # Split sizes. 0 = unprofiled; dataset_profiles.DatasetProfile.apply()
    # sets these when train.py / evaluate.py run with --profile, and
    # train.py cross-checks them against the built loaders so image files
    # added/removed after profiling cannot pass next to a valid fingerprint.
    n_train: int = 0
    n_val: int = 0


@dataclass
class ModelCfg:
    # Backbone
    base_ch: int = 32                      # stem width; stages = 2x, 4x, 8x
    n_blocks: Tuple[int, ...] = (1, 2, 2)  # CSP depth per stage (P2, P3, P4)
    n_edge_init: int = 6                   # Sobel/Laplacian kernels in the stem

    # Neck / head widths (NIRDet-Lite)
    neck_channels: int = 64
    head_channels: int = 64
    head_branch_convs: int = 1
    head_use_stem: bool = True

    # Detection levels — P3 (stride 8), P4 (stride 16), P5 (stride 32)
    strides: Tuple[int, ...] = (8, 16, 32)

    # Edge-Aware Attention
    eaa_filters: int = 4
    eaa_freeze_epochs: int = 5
    eaa_residual_scale: float = 2.0        # gain in [1.0, 3.0]
    eaa_normalize_edges: bool = False      # runtime Div is NPU-hostile
    eaa_padding_mode: str = "zeros"        # reflect Pad is SW on Neural-ART

    # Box priors (median normalised w/h over 948 miniNIRPed train labels,
    # measured in letterbox-canvas space — see dataset_profiles.py). When
    # running with --profile these are overridden (and freshness-verified)
    # by dataset_profiles.DatasetProfile.apply().
    prior_w: float = 0.0461
    prior_h: float = 0.1680
    prior_prob: float = 0.01               # focal-loss cls bias prior

    # Inference post-processing
    nms_iou_thresh: float = 0.45
    nms_score_thresh: float = 0.25
    max_det: int = 300


@dataclass
class LossCfg:
    lambda_cls: float = 1.0
    lambda_reg: float = 2.5
    qfl_beta: float = 2.0
    qfl_alpha: float = -1.0         # < 0 disables RetinaNet-style alpha weighting.
                                    # Reference QFL/GFL and YOLOv8 use no alpha;
                                    # 0.25 down-weights foreground 3:1 on a
                                    # single-class detector.
    # Task-Aligned Assignment
    tal_topk: int = 10
    tal_alpha: float = 1.0
    tal_beta: float = 6.0
    # Soft-target ramp: hard 1.0 -> alignment-normalised score
    ramp_frac: float = 0.15


@dataclass
class AugCfg:
    hflip_p: float = 0.5
    affine_p: float = 0.7
    affine_scale: Tuple[float, float] = (0.8, 1.2)
    affine_translate: float = 0.08
    affine_rotate: float = 5.0
    affine_shear: float = 2.0
    brightness_contrast_p: float = 0.5
    brightness_limit: float = 0.3
    contrast_limit: float = 0.3
    clahe_p: float = 0.3
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    gauss_noise_p: float = 0.3
    motion_blur_p: float = 0.2
    motion_blur_limit: int = 5
    downscale_p: float = 0.2
    downscale_range: Tuple[float, float] = (0.5, 0.9)
    cutout_p: float = 0.15
    # Box-level copy-paste (needs no segmentation masks)
    copy_paste_p: float = 0.5
    copy_paste_max_objs: int = 3
    copy_paste_scale: Tuple[float, float] = (0.7, 1.3)
    copy_paste_max_iou: float = 0.25
    copy_paste_feather: int = 3


@dataclass
class TrainCfg:
    checkpoint_dir: str = r"C:\projects\nightvision\nirdet_lite\checkpoints"
    epochs: int = 150
    batch_size: int = 8
    optimizer: str = "adamw"               # "adamw" | "sgd"
    lr_peak: float = 8e-4
    lr_start: float = 1e-5
    lr_min: float = 1e-6
    warmup_steps: int = 300                # RESCALED at runtime by train.py
                                           # (see the warmup block there);
                                           # kept as a config default only.
    weight_decay: float = 5e-4
    momentum: float = 0.937
    amp_mode: str = "bf16"                 # "bf16" | "fp16" | "fp32"
    grad_clip_norm: float = 10.0
    val_interval: int = 1
    seed: int = 42
    use_ema: bool = True
    ema_decay: float = 0.995
    ema_tau_steps: int = 160
    es_enabled: bool = False
    es_patience: int = 20
    # Channel-summed ImageNet stem transfer
    imagenet_stem_init: bool = True
    imagenet_arch: str = "mobilenet_v3_small"


@dataclass
class EvalCfg:
    eval_score_thresh: float = 0.05         # integrate the full PR curve
    deploy_score_thresh: float = 0.30       # re-derive from the PR curve
    baseline_map50: float = 0.735           # YOLO11n fine-tune reference
    out_dir: str = "eval_outputs"
    max_hard_cases: int = 12
    bootstrap_n: int = 200                  # 0 disables CI computation
    bootstrap_seed: int = 1234


@dataclass
class ExportCfg:
    onnx_path: str = "nirdet.onnx"
    onnx_sim_path: str = "nirdet-sim.onnx"
    onnx_int8_path: str = "nirdet-int8-qdq.onnx"
    opset: int = 12
    calib_images: int = 300
    calib_method: str = "minmax"            # "minmax" | "percentile" | "entropy"
    calib_percentile: float = 99.999


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    aug: AugCfg = field(default_factory=AugCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    export: ExportCfg = field(default_factory=ExportCfg)

    # ---- derived, never stored twice ----
    @property
    def grid_sizes(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (self.data.img_h // s, self.data.img_w // s)
            for s in self.model.strides
        )

    def to_dict(self) -> dict:
        return asdict(self)


def get_config(**overrides) -> Config:
    """
    cfg = get_config()
    cfg = get_config(train=dict(epochs=200, batch_size=4))
    """
    cfg = Config()
    for section, kv in overrides.items():
        if not hasattr(cfg, section):
            raise ValueError(f"no config section '{section}'")
        sec = getattr(cfg, section)
        if not isinstance(kv, dict):
            raise ValueError(f"override for '{section}' must be a dict")
        for k, v in kv.items():
            if not hasattr(sec, k):
                raise ValueError(f"no field '{k}' in {type(sec).__name__}")
            setattr(sec, k, v)
    return cfg


def validate_config(cfg: Config, verbose: bool = True) -> bool:
    """Range checks only. No unreachable placeholder sentinels."""
    errors, warns = [], []

    h, w = cfg.data.img_h, cfg.data.img_w
    for s in cfg.model.strides:
        if h % s or w % s:
            errors.append(f"input {h}x{w} is not divisible by stride {s}")
    if not set(cfg.model.strides) <= {8, 16, 32}:
        errors.append(f"model.strides must be a subset of {{8, 16, 32}}; "
                      f"got {cfg.model.strides}")
    if cfg.model.strides != tuple(sorted(cfg.model.strides)):
        errors.append("model.strides must be ascending")
    if cfg.model.neck_channels != cfg.model.head_channels:
        warns.append("neck_channels != head_channels: head.stem will reproject")
    if cfg.train.batch_size < 4:
        warns.append("batch_size < 4 makes BatchNorm statistics unreliable")
    if not (0.0 < cfg.model.prior_w < 1.0 and 0.0 < cfg.model.prior_h < 1.0):
        errors.append("prior_w / prior_h must be normalised into (0, 1)")
    if cfg.eval.eval_score_thresh >= cfg.eval.deploy_score_thresh:
        warns.append("eval_score_thresh >= deploy_score_thresh (unusual)")
    if cfg.train.amp_mode not in ("bf16", "fp16", "fp32"):
        errors.append(f"unknown amp_mode '{cfg.train.amp_mode}'")
    if cfg.model.eaa_padding_mode != "zeros":
        warns.append("eaa_padding_mode != 'zeros' will fall back to software "
                     "Pad on Neural-ART and complicate NCNN INT8")
    if cfg.model.eaa_normalize_edges:
        warns.append("eaa_normalize_edges=True emits ReduceMean+Div with a "
                     "runtime divisor: unsupported on Neural-ART, unstable "
                     "for INT8 calibration")

    if verbose:
        print("=" * 62)
        print("  NIRDet-Lite config validation")
        print("=" * 62)
        print(f"  input        : {cfg.data.num_channels}ch x {h}x{w}")
        print(f"  strides      : {cfg.model.strides}  grids {cfg.grid_sizes}")
        print(f"  neck/head ch : {cfg.model.neck_channels}/{cfg.model.head_channels}")
        print(f"  epochs/batch : {cfg.train.epochs}/{cfg.train.batch_size}")
        print(f"  optim        : {cfg.train.optimizer} lr={cfg.train.lr_peak}")
        for m in warns:
            print(f"  WARN  {m}")
        for m in errors:
            print(f"  ERROR {m}")
        print("=" * 62)

    if errors:
        raise ValueError(f"config invalid: {errors[0]}")
    return True


if __name__ == "__main__":
    validate_config(get_config())
