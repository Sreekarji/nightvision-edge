"""
config.py — NIRDet-Lite SINGLE SOURCE OF TRUTH
===============================================
Single-class (person) NIR pedestrian detector for Raspberry Pi 5 (NCNN INT8)
and STM32N6570-DK (Neural-ART NPU).

CANVAS: 512x288 (16:9)
----------------------
The IMX708 sensor and the NIRPed source imagery are both 1280x720. The old
384x640 canvas is 5:3, so every frame carried 12 rows of constant padding top
and bottom and the train/deploy input distributions differed by construction.
At 512x288 the letterbox scale is min(512/1280, 288/720) = 0.4 in BOTH axes,
so padding is exactly zero and canvas-space box statistics equal raw
label-space statistics. All three strides divide evenly:
    288/8=36  288/16=18  288/32=9
    512/8=64  512/16=32  512/32=16
Grid cells: 64*36 + 32*18 + 16*9 = 3024 (was 5040 at 384x640).

Rules enforced by this file:
  * No function calls at import time (safe to import from any module).
  * Deployment-critical constants live here so head.py / losses.py /
    live_nirdet.py / nirdet_pp.c agree. deploy_contract() hashes them.
  * Dataset-dependent values (priors, deploy threshold, copy-paste
    probability, EMA tau, EAA projection bias) are DERIVED at runtime, never
    hardcoded. The helpers at the bottom express that dependence in code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


# ===========================================================================
# DECODE CONTRACT — must be identical in head.py, losses.py, live_nirdet.py,
# and nirdet_pp.c. Changing any of these invalidates trained checkpoints.
# test_decode_contract.py enforces it.
# ===========================================================================

DECODE_OFFSET_SCALE: float = 2.0                 # offset range [-0.5, 1.5] cells
DECODE_OFFSET_BIAS: float = (DECODE_OFFSET_SCALE - 1.0) / 2.0   # 0.5

# exp() clamp on the w/h regression logits. With the regression head split
# into off_pred / size_pred, the SIZE branch gets its own QDQ scale and this
# clamp is the only thing setting its dynamic range:
#   exp(-6)*512 = 1.3 px .. exp(1)*512 = 1392 px
# One int8 LSB over [-6, 1] is 7/255 = 0.027 in log space (2.7% of width),
# versus 0.047 when offsets and sizes shared a scale over [-6, 6].
REG_LOG_CLAMP_MIN: float = -6.0
REG_LOG_CLAMP_MAX: float = 1.0

# Single class. Not a tunable. Not an expansion point.
NUM_CLASSES: int = 1

# Detection levels.
STRIDES: Tuple[int, ...] = (8, 16, 32)

# bf16 grid-index safety limit. bf16 carries 8 bits of significand, so
# integers are exact only up to 2^8 = 256. Cell column indices are built in
# float32 in head._grid() precisely so autocast cannot quantise them, but a
# canvas wider than stride*256 = 2048 px at stride 8 would make the indices
# unrepresentable in bf16 if that ever regressed. head._grid() asserts it.
BF16_EXACT_INT_MAX: int = 256
MAX_SAFE_CANVAS_WIDTH: int = BF16_EXACT_INT_MAX * 8      # 2048

# ST Neural-ART: "for 8-bit feature data, feature width multiplied by the used
# batch depth (input channels) must be <= 2048", else the compiler splits the
# operator into columns. Also the pooling line-buffer limit.
ST_FEATURE_WIDTH_CHANNEL_LIMIT: int = 2048


# ===========================================================================
# SUB-CONFIGS
# ===========================================================================

@dataclass
class DataCfg:
    root: str = r"C:\projects\nightvision\data\raw\miniNIRPed"
    img_h: int = 288
    img_w: int = 512
    num_channels: int = 1
    num_classes: int = NUM_CLASSES
    class_names: Tuple[str, ...] = ("person",)
    num_workers: int = 4
    pin_memory: bool = True
    # Split sizes. 0 = unprofiled; DatasetProfile.apply() fills these.
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    # Total train-split box count, from the profile. Drives copy_paste_p.
    n_train_boxes: int = 0


@dataclass
class ModelCfg:
    # ---- Backbone ----
    # base_ch 24: stages output base_ch*4 = 96 channels, which sits inside
    # ST's N*(72..128) input-channel window for 1x1 kernels and is 4*24, the
    # ideal output-channel multiple. 24 also keeps every stage width a
    # non-prime multiple of 8 (ST: "avoid prime numbers as channel counts").
    base_ch: int = 24
    # STYOLO RAM-efficient projection: the stem emits a NARROW tensor on the
    # largest (stride-2) map and width is recovered on the already-downsampled
    # stride-4 map. STResNet/STYOLO measured 4.26 -> 2.46 MB peak RAM (-42%)
    # on real STM32N6 silicon for 30.54 -> 30.12 mAP. The N6 degrades sharply
    # past 4 MB because it spills to external memory.
    stem_ch: int = 16
    n_blocks: Tuple[int, ...] = (1, 2, 2)  # dense residual depth (P2, P3, P4)
    n_edge_init: int = 6                   # Sobel/Laplacian kernels in the stem
    # Dilation on the LAST residual block of the stride-32 stage only.
    # Enlarges the receptive field for global context (the consistent finding
    # in the IR-detection literature for separating low-contrast targets from
    # clutter) at zero parameter and zero MAC cost. ST warns against LARGE
    # dilation factors; 2 is the smallest useful value.
    backbone_p5_dilation: int = 2

    # ---- Neck ----
    neck_channels: int = 64
    # out3 is a dense 3x3 over the largest feature map and was ~15% of total
    # model compute, larger than the whole head. The 1x1 lateral was already
    # the information bottleneck, so narrowing the smoothing conv costs little.
    # 48 = 2*24, an ideal ST output-channel multiple.
    neck_out3_channels: int = 48
    # One-level bottom-up PAN (N3 -> N4) so large objects get fine spatial
    # detail injected back up. Standard FPN->PAN practice.
    neck_pan: bool = True
    # Selective-kernel channel weighting over two receptive fields, after
    # IPD-Net (Zhou et al., Sensors 2022, 22(22):8966), which reported +3.6%
    # mAP50 over its YOLOv5 baseline on the ZUT infrared set from a
    # combination of adaptive feature extraction and coordinate fusion.
    # OFF by default so it can be ablated cleanly against this codebase.
    neck_sk: bool = False

    # ---- Head ----
    head_channels: int = 64
    head_branch_convs: int = 1
    head_use_stem: bool = True
    # Branch conv WEIGHTS are shared across levels (FCOS/YOLOX practice), but
    # BatchNorm statistics are not: a stride-8 and a stride-32 feature map have
    # materially different activation distributions and one set of running
    # stats fits neither. ~1 KB of extra parameters.
    head_per_level_bn: bool = True

    strides: Tuple[int, ...] = STRIDES

    # ---- Edge-Aware Attention ----
    eaa_filters: int = 4
    eaa_freeze_epochs: int = 5
    eaa_residual_scale: float = 2.0        # gain in [1.0, 3.0]
    eaa_normalize_edges: bool = False      # runtime Div is SW_INT on Neural-ART
    eaa_padding_mode: str = "zeros"        # reflect Pad is only partially HW
    # Edge conv runs at stride 2, then pools by 4, reaching stride 8. Running
    # it at full resolution and pooling by 8 (the old path) held a
    # full-resolution N-channel tensor live and smeared the canvas border into
    # the first row/column of the pooled map.
    eaa_edge_stride: int = 2
    eaa_pool_factor: int = 4
    # proj bias is CALIBRATED on real data by EdgeAwareAttention.calibrate_bias
    # and written back here so it lands in the checkpoint. It cannot be
    # hardcoded: it depends on the sensor, the 850 nm illuminator, the
    # flat-field map and the CLAHE settings. None = not yet calibrated.
    eaa_proj_bias: Optional[float] = None

    # ---- Box priors ----
    # Canvas-space medians, overridden (and freshness-verified) by
    # DatasetProfile.apply(). At 512x288 the letterbox scale is 0.4 in both
    # axes for 1280x720 sources, so canvas medians equal raw label medians.
    prior_w: float = 0.046094
    prior_h: float = 0.179167
    prior_prob: float = 0.01               # focal-loss cls bias prior

    # ---- Inference post-processing (host side, never in the ONNX graph) ----
    nms_iou_thresh: float = 0.45
    nms_score_thresh: float = 0.25
    max_det: int = 300


@dataclass
class LossCfg:
    lambda_cls: float = 1.0
    lambda_reg: float = 2.5
    qfl_beta: float = 2.0
    qfl_alpha: float = -1.0         # < 0 disables alpha weighting (GFL/YOLOv8)
    # Task-Aligned Assignment. alpha=0.5 is the YOLOv8 setting: at alpha=1.0
    # the classification score dominates the alignment metric, and on a
    # single-class head initialised to a 0.01 prior that makes early
    # assignment noisier than it needs to be.
    tal_topk: int = 10
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
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
    # Preprocessing SWITCH, not a probability. CLAHE is applied to both splits
    # and at deployment; it is what let EAA drop its runtime mean-normalisation.
    clahe_enabled: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    # Flat-field correction. The 850 nm illuminator beam profile is not
    # uniform; the periphery is systematically darker than the centre and
    # CLAHE is tile-local, so it only partially compensates for that global
    # gradient. Commercial NIR systems apply flat-field correction before any
    # enhancement. Capture the map once from a uniform surface. None = off.
    # Part of the preprocessing contract: dataset.py, live_nirdet.py and
    # quantize_qdq.py must all apply it identically.
    flat_field_path: Optional[str] = None
    gauss_noise_p: float = 0.3
    motion_blur_p: float = 0.2
    motion_blur_limit: int = 5
    downscale_p: float = 0.2
    downscale_range: Tuple[float, float] = (0.5, 0.9)
    cutout_p: float = 0.15
    # Box-level copy-paste. copy_paste_p is DERIVED from the active dataset's
    # box count by copy_paste_p_for(); this field is only the cap.
    copy_paste_p_max: float = 0.5
    copy_paste_max_objs: int = 3
    copy_paste_scale: Tuple[float, float] = (0.7, 1.3)
    copy_paste_max_iou: float = 0.25
    copy_paste_feather: int = 3


@dataclass
class TrainCfg:
    checkpoint_dir: str = r"C:\projects\nightvision\nirdet_lite\checkpoints"
    epochs: int = 100
    batch_size: int = 8
    optimizer: str = "adamw"               # "adamw" | "sgd"
    lr_peak: float = 8e-4
    lr_start: float = 1e-5
    lr_min: float = 1e-6
    warmup_steps: int = 300                # RESCALED at runtime by train.py
    weight_decay: float = 5e-4
    momentum: float = 0.937
    amp_mode: str = "bf16"                 # "bf16" | "fp16" | "fp32"
    grad_clip_norm: float = 10.0
    val_interval: int = 1
    seed: int = 42
    use_ema: bool = True
    ema_decay: float = 0.995
    # DERIVED from steps_per_epoch by ema_tau_for(); this is the fallback.
    ema_tau_steps: int = 160
    es_enabled: bool = False
    es_patience: int = 20
    imagenet_stem_init: bool = True
    imagenet_arch: str = "mobilenet_v3_small"
    # Layer-wise LR scaling. STResNet/STYOLO report this single change taking
    # STYOLO-Nano from 21.32 to 26.25 mAP on MS-COCO with no architectural
    # change: 0.2x on the (pretrained-or-edge-seeded) backbone to preserve its
    # representations, 0.8x on the neck, 1.0x on the randomly-initialised head
    # which must adapt fastest.
    lr_scale_backbone: float = 0.2
    lr_scale_neck: float = 0.8
    lr_scale_head: float = 1.0


@dataclass
class EvalCfg:
    eval_score_thresh: float = 0.05         # integrate the full PR curve
    deploy_score_thresh: float = 0.30       # fallback only; profile overrides
    baseline_map50: float = 0.735           # YOLO11n fine-tune reference
    out_dir: str = "eval_outputs"
    max_hard_cases: int = 12
    bootstrap_n: int = 200                  # 0 disables CI computation
    bootstrap_seed: int = 1234
    # Checkpoint selection happens on one split, the headline number is
    # reported on another. Selecting best.pth on val across ~100 evaluations
    # and then reporting val mAP makes that number optimistically biased by
    # construction. validate_config() errors if these are equal.
    select_split: str = "val"
    report_split: str = "test"
    # evaluate_onnx.py exits non-zero if INT8 costs more than this.
    int8_max_map50_drop: float = 0.02


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

    @property
    def num_cells(self) -> int:
        return sum(h * w for h, w in self.grid_sizes)

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- deployment contract ----
    def deploy_contract(self) -> dict:
        """
        Everything the on-device decoders (live_nirdet.py, nirdet_pp.c) must
        agree with, plus a stable hash. The hash is written into every
        checkpoint; test_decode_contract.py asserts the checkpoint's hash
        matches the current config, so a geometry change cannot silently
        invalidate an exported model.
        """
        payload = {
            "num_classes": NUM_CLASSES,
            "img_h": int(self.data.img_h),
            "img_w": int(self.data.img_w),
            "strides": [int(s) for s in self.model.strides],
            "decode_offset_scale": float(DECODE_OFFSET_SCALE),
            "decode_offset_bias": float(DECODE_OFFSET_BIAS),
            "reg_log_clamp_min": float(REG_LOG_CLAMP_MIN),
            "reg_log_clamp_max": float(REG_LOG_CLAMP_MAX),
            # 3 blobs per level: cls (1ch), off (2ch), size (2ch).
            "blobs_per_level": 3,
            "blob_channels": {"cls": 1, "off": 2, "size": 2},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["hash"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return payload


# ===========================================================================
# DATASET-DERIVED HELPERS — the dependence is in code, not in a magic number
# ===========================================================================

def copy_paste_p_for(n_train_boxes: int) -> float:
    """
    Box-level copy-paste attacks positive scarcity. Its usefulness scales
    INVERSELY with how many positives the dataset already has, so the
    probability must be a function of the active dataset, not a constant.

        <=  1k boxes (miniNIRPed, 948) -> 0.50  (cap)
        ~  10k boxes                   -> 0.25
        >= 100k boxes (NIRPed, 146k)   -> 0.05  (floor)

    Log-linear interpolation between the 1k and 100k anchors.
    """
    n = max(int(n_train_boxes), 1)
    lo_n, lo_p = 1_000.0, 0.50
    hi_n, hi_p = 100_000.0, 0.05
    if n <= lo_n:
        return lo_p
    if n >= hi_n:
        return hi_p
    import math
    t = (math.log(n) - math.log(lo_n)) / (math.log(hi_n) - math.log(lo_n))
    return float(lo_p + t * (hi_p - lo_p))


def ema_tau_for(steps_per_epoch: int, target_epochs: float = 5.0) -> int:
    """
    EMA time constant in STEPS, sized so the average converges in roughly
    ``target_epochs`` regardless of dataset size.

    d_eff(step) = decay * (1 - exp(-step / tau)), so tau is the step count at
    which the effective decay reaches ~63% of its asymptote. A hardcoded tau
    is a hidden dataset dependence: 160 steps is ~5 epochs at 261 images /
    batch 8, but under one epoch at 146k images.
    """
    return max(1, int(round(max(1, int(steps_per_epoch)) * float(target_epochs))))


# ===========================================================================
# construction + validation
# ===========================================================================

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

    if cfg.data.num_classes != 1:
        errors.append(f"single-class detector: num_classes must be 1, got "
                      f"{cfg.data.num_classes}")

    if w > MAX_SAFE_CANVAS_WIDTH:
        errors.append(
            f"canvas width {w} exceeds {MAX_SAFE_CANVAS_WIDTH}: at stride "
            f"{min(cfg.model.strides)} the grid would have more than "
            f"{BF16_EXACT_INT_MAX} columns, and bf16 cannot represent those "
            f"integer cell indices exactly")

    # Split reporting must not reuse the selection split.
    if cfg.eval.select_split == cfg.eval.report_split:
        errors.append(
            f"eval.select_split and eval.report_split are both "
            f"'{cfg.eval.select_split}': best.pth is chosen by maximising "
            f"mAP on the selection split, so reporting the headline number "
            f"on that same split is optimistically biased. Use "
            f"select_split='val', report_split='test'.")

    if cfg.train.batch_size < 4:
        warns.append("batch_size < 4 makes BatchNorm statistics unreliable")
    if not (0.0 < cfg.model.prior_w < 1.0 and 0.0 < cfg.model.prior_h < 1.0):
        errors.append("prior_w / prior_h must be normalised into (0, 1)")

    import math
    for label, prior in (("prior_w", cfg.model.prior_w),
                         ("prior_h", cfg.model.prior_h)):
        lg = math.log(prior)
        if not (REG_LOG_CLAMP_MIN < lg < REG_LOG_CLAMP_MAX):
            errors.append(
                f"log({label}) = {lg:.3f} is not strictly inside the reg "
                f"clamp ({REG_LOG_CLAMP_MIN}, {REG_LOG_CLAMP_MAX}); the head "
                f"bias init would be clipped and TAL would cold-start with "
                f"zero positives")

    if cfg.eval.eval_score_thresh >= cfg.eval.deploy_score_thresh:
        warns.append("eval_score_thresh >= deploy_score_thresh (unusual)")
    if cfg.train.amp_mode not in ("bf16", "fp16", "fp32"):
        errors.append(f"unknown amp_mode '{cfg.train.amp_mode}'")
    if cfg.model.eaa_padding_mode != "zeros":
        warns.append("eaa_padding_mode != 'zeros' falls back to software Pad "
                     "on Neural-ART and complicates NCNN INT8")
    if cfg.model.eaa_normalize_edges:
        warns.append("eaa_normalize_edges=True emits ReduceMean + Div with a "
                     "runtime divisor; ST maps Div on HW only when the second "
                     "operand is constant, so this becomes SW_INT")
    if cfg.model.eaa_proj_bias is None:
        warns.append("model.eaa_proj_bias is None: EAA has not been "
                     "calibrated. train.py calls eaa.calibrate_bias() on the "
                     "first real batch; without it the spatial gate is a "
                     "near-uniform rescale that the following BatchNorm "
                     "absorbs entirely")

    if cfg.model.backbone_p5_dilation > 3:
        warns.append(f"backbone_p5_dilation={cfg.model.backbone_p5_dilation}: "
                     f"ST advises against large dilation factors")

    # ST feature-width x input-channel limit, checked on the widest head input.
    widest = w // min(cfg.model.strides)
    prod = widest * cfg.model.head_channels
    if prod > ST_FEATURE_WIDTH_CHANNEL_LIMIT:
        warns.append(
            f"head stride-{min(cfg.model.strides)} input is {widest} wide x "
            f"{cfg.model.head_channels} ch = {prod} > "
            f"{ST_FEATURE_WIDTH_CHANNEL_LIMIT}: the Neural-ART compiler will "
            f"split this operator into columns (export_onnx.py reports every "
            f"such node)")

    if verbose:
        c = cfg.deploy_contract()
        print("=" * 62)
        print("  NIRDet-Lite config validation")
        print("=" * 62)
        print(f"  input        : {cfg.data.num_channels}ch x {h}x{w}"
              f"  (AR {w / h:.4f})")
        print(f"  classes      : {cfg.data.num_classes} ({', '.join(cfg.data.class_names)})")
        print(f"  strides      : {cfg.model.strides}  grids {cfg.grid_sizes}"
              f"  cells {cfg.num_cells}")
        print(f"  backbone     : stem {cfg.model.stem_ch} -> base {cfg.model.base_ch}"
              f" -> stages {cfg.model.base_ch * 4}"
              f"  p5_dilation {cfg.model.backbone_p5_dilation}")
        print(f"  neck         : {cfg.model.neck_channels} "
              f"(out3 {cfg.model.neck_out3_channels}) "
              f"pan={cfg.model.neck_pan} sk={cfg.model.neck_sk}")
        print(f"  head         : {cfg.model.head_channels} ch, "
              f"per-level BN={cfg.model.head_per_level_bn}, "
              f"blobs/level=3 (cls/off/size)")
        print(f"  priors       : w {cfg.model.prior_w} h {cfg.model.prior_h}")
        print(f"  epochs/batch : {cfg.train.epochs}/{cfg.train.batch_size}")
        print(f"  optim        : {cfg.train.optimizer} lr={cfg.train.lr_peak} "
              f"scales b/n/h {cfg.train.lr_scale_backbone}/"
              f"{cfg.train.lr_scale_neck}/{cfg.train.lr_scale_head}")
        print(f"  eval splits  : select '{cfg.eval.select_split}' -> "
              f"report '{cfg.eval.report_split}'")
        print(f"  contract     : {c['hash']}")
        for m in warns:
            print(f"  WARN  {m}")
        for m in errors:
            print(f"  ERROR {m}")
        print("=" * 62)

    if errors:
        raise ValueError(f"config invalid: {errors[0]}")
    return True


if __name__ == "__main__":
    cfg = get_config()
    validate_config(cfg)
    print("\ndeploy contract:")
    for k, v in cfg.deploy_contract().items():
        print(f"  {k:<22} {v}")
    print("\ncopy_paste_p_for:")
    for n in (948, 5_000, 20_000, 146_000):
        print(f"  {n:>7} boxes -> {copy_paste_p_for(n):.3f}")
    print("\nema_tau_for:")
    for spe in (32, 100, 1_000, 18_000):
        print(f"  {spe:>6} steps/epoch -> tau {ema_tau_for(spe)}")
