"""
config.py — NIRDet-Lite SINGLE SOURCE OF TRUTH
===============================================
Single-class (person) NIR pedestrian detector for Raspberry Pi 5 (NCNN INT8)
and STM32N6570-DK (Neural-ART NPU).

CANVAS: 512x288 (16:9)
----------------------
The IMX708 sensor and the NIRPed source imagery are both 1280x720. At 512x288
the letterbox scale is min(512/1280, 288/720) = 0.4 in BOTH axes, so padding is
exactly zero and canvas-space box statistics equal raw label-space statistics.
    288/8=36  288/16=18  288/32=9      512/8=64  512/16=32  512/32=16
Grid cells: 64*36 + 32*18 + 16*9 = 3024.

Rules enforced by this file:
  * No function calls at import time (safe to import from any module).
  * Deployment-critical constants live here so head.py / losses.py /
    live_nirdet.py / nirdet_pp.c agree. deploy_contract() hashes them.
  * Dataset-dependent values (priors, deploy threshold, copy-paste
    probability, EMA tau, EAA projection bias) are DERIVED at runtime, never
    hardcoded. Priors are UNSET by default and their absence is FATAL (F9).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import numpy as np


# ===========================================================================
# DECODE CONTRACT — must be identical in head.py, losses.py, live_nirdet.py,
# evaluate_onnx.py and nirdet_pp.c. test_decode_contract.py enforces it.
# ===========================================================================

DECODE_OFFSET_SCALE: float = 2.0                 # offset range [-0.5, 1.5] cells
DECODE_OFFSET_BIAS: float = (DECODE_OFFSET_SCALE - 1.0) / 2.0   # 0.5

# exp() clamp on the w/h regression logits. With the regression head split
# into off_pred / size_pred, the SIZE branch gets its own QDQ scale and this
# clamp is the only thing setting its dynamic range:
#   exp(-6)*512 = 1.3 px .. exp(1)*512 = 1392 px
REG_LOG_CLAMP_MIN: float = -6.0
REG_LOG_CLAMP_MAX: float = 1.0

# Single class. Not a tunable. Not an expansion point.
NUM_CLASSES: int = 1

# Detection levels.
STRIDES: Tuple[int, ...] = (8, 16, 32)


# ===========================================================================
# POST-DECODE CONTRACT — the MEMBERSHIP half of the decode contract (F36,
# F59, F74, F81, F103).
#
# The arithmetic half (OFFSET_SCALE/BIAS, the log clamp) says WHERE a box is.
# This says WHICH boxes exist. Four decoders implement it — model.py,
# live_nirdet.py, evaluate_onnx.py and nirdet_pp.c — and they must agree on
# the ORDER as well as the values, because NMS is order-sensitive: two boxes
# that overlap only outside the canvas are suppressed if you clamp first and
# kept if you clamp last.
#
# CANONICAL ORDER:
#   1. decode to cx, cy, w, h in canvas pixels
#   2. convert to xyxy
#   3. CLAMP xyxy to [0, img_w] x [0, img_h]
#   4. DROP boxes whose CLAMPED extent is <= MIN_BOX_PX in either axis. A box
#      decoded entirely off-canvas collapses to zero area, and a zero-area box
#      has IoU 0 against everything, so greedy NMS can never suppress it and
#      it consumes a max_det slot forever.
#   5. NMS
# ===========================================================================

MIN_BOX_PX: float = 1.0

# The C post-processor's static heap and suppression buffer capacity.
# gen_contract_c.py reads this. cfg.model.max_det must be <= this value;
# validate_config enforces it. Changing this requires recompiling nirdet_pp.c
# (python gen_contract_c.py --emit) and re-validating the firmware ABI.
NIRDET_MAX_DET: int = 300

# Maximum candidates fed to NMS across all levels. The C heap enforces this
# before NMS; the Python decoders must match. A crowded frame at
# eval_score_thresh=0.05 easily exceeds 300 raw candidates, so without this
# the accuracy gate measures a different result than the device produces.
# 0 = disabled (Python-only paths). Must be >= NIRDET_MAX_DET or 0.
PRE_NMS_TOPK: int = NIRDET_MAX_DET   # 300 — matches the C heap exactly


def np_sigmoid(x):
    """
    Numerically stable sigmoid. Accepts scalars, 0-d arrays, and arrays.

    F11: scalar/0-d input (the deprecated indexing case) returns a float.
    ARRAY input always returns an array — including size-1 arrays: the
    decoders (live_nirdet.decode_level, evaluate_onnx) slice this result and
    call .astype() on it, so collapsing a 1-element array to a float breaks
    the decode contract (test_decode_contract T4 catches exactly that).
    """
    x_arr = np.asarray(x, dtype=np.float32)
    scalar_in = x_arr.ndim == 0
    x_arr = np.atleast_1d(x_arr)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_neg   = np.exp(x_arr[~pos])
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return float(out.reshape(-1)[0]) if scalar_in else out


def clamp_and_filter(boxes: np.ndarray, scores: np.ndarray,
                     img_h: int, img_w: int,
                     min_px: float = MIN_BOX_PX):
    """
    Steps 3 and 4 of the post-decode contract, in that order.

    Filtering on the PRE-clamp extent (what model.decode_predictions used to
    do) admits boxes that are 2 px wide on paper and 0 px wide after clamping.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.shape[0] == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
    b = boxes.astype(np.float32, copy=True)
    b[:, 0::2] = np.clip(b[:, 0::2], 0.0, float(img_w))
    b[:, 1::2] = np.clip(b[:, 1::2], 0.0, float(img_h))
    keep = ((b[:, 2] - b[:, 0]) > float(min_px)) & \
           ((b[:, 3] - b[:, 1]) > float(min_px))
    if not bool(keep.any()):
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
    return b[keep], scores[keep].astype(np.float32)


def clamp_and_filter_torch(boxes: "torch.Tensor", scores: "torch.Tensor",
                            img_h: int, img_w: int,
                            min_box_px: float = MIN_BOX_PX
                            ) -> "Tuple[torch.Tensor, torch.Tensor]":
    """
    Torch version of clamp_and_filter (mirrors the numpy path exactly).
    Clamped XYXY to canvas, then drop boxes whose width OR height <= min_box_px.
    This is the canonical membership contract; model.decode_predictions and
    test_decode_contract.py both call it.
    """
    import torch as _torch
    h, w = int(img_h), int(img_w)
    clamped = _torch.stack([
        boxes[:, 0].clamp(0, w),
        boxes[:, 1].clamp(0, h),
        boxes[:, 2].clamp(0, w),
        boxes[:, 3].clamp(0, h),
    ], dim=-1)
    ok = ((clamped[:, 2] - clamped[:, 0]) > min_box_px) & \
         ((clamped[:, 3] - clamped[:, 1]) > min_box_px)
    return clamped[ok], scores[ok]


# ===========================================================================
# CHECKPOINT KEYS (F29, F107)
# These used to live in train.py, so evaluate.py's `from train import ...`
# imported a SECOND copy of train.py under the name `train` whenever train.py
# was the __main__ module. They are pure constants; they belong here.
# ===========================================================================

CKPT_LIVE_KEY: str = "model"
CKPT_EMA_KEY: str = "model_ema"
CKPT_DEPLOY_KEY: str = "deploy_state_dict"


# bf16 grid-index safety limit. bf16 carries 8 bits of significand, so
# integers are exact only up to 2^8 = 256.
BF16_EXACT_INT_MAX: int = 256
MAX_SAFE_CANVAS_WIDTH: int = BF16_EXACT_INT_MAX * 8      # 2048

# ST Neural-ART: "for 8-bit feature data, feature width multiplied by the used
# batch depth (input channels) must be <= 2048".
ST_FEATURE_WIDTH_CHANNEL_LIMIT: int = 2048

# ONE fallback deploy threshold literal for the whole project (F10).
# EvalCfg and dataset_profiles.DatasetProfile both use THIS name.
# None means "not yet measured by evaluate.py". live_nirdet.py refuses to
# start without a measured threshold from the dataset profile or an explicit
# --score-thresh flag. 0.30 was a placeholder; dataset profiles now carry
# deploy_score_thresh measured on the report split.
DEFAULT_DEPLOY_SCORE_THRESH: Optional[float] = None


# ===========================================================================
# SUB-CONFIGS
# ===========================================================================

@dataclass
class DataCfg:
    # UNSET by default: DatasetProfile.apply() supplies the real root.
    root: Optional[str] = None
    img_h: int = 288
    img_w: int = 512
    num_channels: int = 1
    num_classes: int = NUM_CLASSES
    class_names: Tuple[str, ...] = ("person",)
    num_workers: int = 4
    pin_memory: bool = True
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_train_boxes: int = 0
    # Fraction of TRAIN images that letterbox with ZERO padding. Supplied by
    # DatasetProfile.apply from the measured split statistics (F08); 1.0 (the
    # default) means "no padding, or not yet measured".
    train_zero_pad_fraction: float = 1.0
    profile_applied: bool = False
    profile_name: str = ""


@dataclass
class ModelCfg:
    # ---- Backbone ----
    base_ch: int = 24
    stem_ch: int = 16
    n_blocks: Tuple[int, ...] = (1, 2, 2)  # stride-4, stride-8, stride-16 stages
    n_edge_init: int = 6
    backbone_p5_dilation: int = 2

    # ---- Neck ----
    neck_channels: int = 64
    neck_out3_channels: int = 48
    neck_pan: bool = True
    neck_sk: bool = False
    neck_fuse_clip: bool = True        # F77: clamp FPN fusions to [-6, 6]

    # ---- Head ----
    head_channels: int = 64
    head_branch_convs: int = 1
    head_use_stem: bool = True
    head_per_level_bn: bool = True

    strides: Tuple[int, ...] = STRIDES

    # ---- Edge-Aware Attention ----
    eaa_filters: int = 4
    eaa_freeze_epochs: int = 5
    eaa_residual_scale: float = 2.0
    eaa_normalize_edges: bool = False
    eaa_padding_mode: str = "zeros"
    eaa_edge_stride: int = 2
    eaa_pool_factor: int = 4
    eaa_proj_bias: Optional[float] = None

    # ---- Box priors ----
    # UNSET (F9). These are CANVAS-SPACE DATASET MEASUREMENTS, not constants.
    # head.size_pred.bias is seeded from them and that seed is the only reason
    # Task-Aligned Assignment finds a positive at epoch 0. A hardcoded
    # fallback from one dataset produces, on any other dataset, a plausible
    # loss curve with a useless regression branch — so their absence is a
    # validate_config ERROR, not a warning.
    prior_w: Optional[float] = None
    prior_h: Optional[float] = None
    prior_prob: float = 0.01               # focal-loss cls bias prior

    # ---- Inference post-processing (host side, never in the ONNX graph) ----
    nms_iou_thresh: float = 0.45
    # None ALWAYS. There is no default score threshold anywhere in this
    # project: the right value is cfg.eval.deploy_score_thresh, measured by
    # evaluate.py on the report split. Callers pass it explicitly.
    nms_score_thresh: Optional[float] = None
    max_det: int = 300


@dataclass
class LossCfg:
    lambda_cls: float = 1.0
    lambda_reg: float = 2.5
    qfl_beta: float = 2.0
    qfl_alpha: float = -1.0         # < 0 disables alpha weighting (GFL/YOLOv8)
    tal_topk: int = 10
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    ramp_frac: float = 0.15
    tal_min_pos_target: float = 0.10       # F64: floor on positive conf targets
    norm_floor_abort_thresh: int = 50      # F66: per-epoch norm-floor abort limit


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
    # Preprocessing SWITCH, not a probability. Applied to both splits and at
    # deployment; it is part of the hashed deploy contract.
    clahe_enabled: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    flat_field_path: Optional[str] = None
    gauss_noise_p: float = 0.3
    motion_blur_p: float = 0.2
    motion_blur_limit: int = 5
    downscale_p: float = 0.2
    downscale_range: Tuple[float, float] = (0.5, 0.9)
    cutout_p: float = 0.15
    copy_paste_p_max: float = 0.5
    copy_paste_max_objs: int = 3
    copy_paste_scale: Tuple[float, float] = (0.7, 1.3)
    copy_paste_max_iou: float = 0.25
    copy_paste_feather: int = 3
    # Horizon row (canvas px) for the copy-paste size/row prior (F05):
    # apparent height ~ (row - h0), so a patch scaled by f re-enters at
    # h0 + f*(src_row - h0). 0.0 = pure proportionality (horizon at the top
    # edge); set from the mounting geometry when it is known.
    copy_paste_horizon_row: float = 0.0


@dataclass
class TrainCfg:
    # Relative, resolved once in main() — symmetric with the export/eval
    # artefact paths so a run started from a different CWD does not scatter
    # checkpoints and eval outputs into different trees (F12).
    checkpoint_dir: str = "checkpoints"
    epochs: int = 100
    batch_size: int = 8
    optimizer: str = "adamw"               # "adamw" | "sgd"
    lr_peak: float = 8e-4
    lr_start: float = 1e-5
    lr_min: float = 1e-6
    warmup_epochs_frac: float = 0.03
    # Fraction of total training epochs used for LR warmup.
    # train.py derives warmup_steps = round(warmup_epochs_frac * epochs *
    # steps_per_epoch), capped at 10% of the total run so a small dataset
    # does not spend a disproportionate time below peak LR.
    weight_decay: float = 5e-4
    momentum: float = 0.937
    amp_mode: str = "bf16"                 # "bf16" | "fp16" | "fp32"
    grad_clip_norm: float = 10.0
    val_interval: int = 1
    seed: int = 42
    use_ema: bool = True
    ema_decay: float = 0.995
    es_enabled: bool = False
    es_patience: int = 20
    imagenet_stem_init: bool = True
    imagenet_arch: str = "mobilenet_v3_small"
    lr_scale_backbone: float = 0.2
    lr_scale_neck: float = 0.8
    lr_scale_head: float = 1.0
    # >= 32 unaugmented frames (F118): the EAA bias is a mean/std of a
    # scene-dependent statistic that is frozen into the checkpoint AND into
    # the INT8 activation ranges. One batch of 8 is too thin an estimate.
    eaa_calib_frames: int = 32


@dataclass
class EvalCfg:
    eval_score_thresh: float = 0.05         # integrate the full PR curve
    deploy_score_thresh: Optional[float] = DEFAULT_DEPLOY_SCORE_THRESH
    baseline_map50: Optional[float] = None
    baseline_source: str = ""               # provenance string, printed as-is
    out_dir: str = "eval_outputs"
    max_hard_cases: int = 12
    bootstrap_n: int = 200                  # 0 disables CI computation
    bootstrap_seed: int = 1234
    select_split: str = "val"
    report_split: str = "test"
    int8_max_map50_drop: float = 0.02
    # ---- deployment tracker (live_nirdet.py) ----
    # Deployment-tuning numbers belong beside deploy_score_thresh, not as
    # class constants inside the runtime (F60). confirm_hits in particular
    # trades latency for precision and depends on the frame rate.
    track_iou_match: float = 0.40
    track_confirm_hits: int = 3
    track_max_missed: int = 5


@dataclass
class ExportCfg:
    onnx_path: str = "nirdet.onnx"
    onnx_sim_path: str = "nirdet-sim.onnx"
    onnx_int8_path: str = "nirdet-int8-qdq.onnx"
    opset: int = 12
    calib_images: int = 300
    calib_method: str = "minmax"            # "minmax" | "percentile" | "entropy"
    calib_percentile: float = 99.999
    calib_seed: int = 42                    # F49: shared ORT/NCNN calibration seed


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
    @staticmethod
    def _flat_field_digest(path: str) -> Optional[str]:
        """SHA-256 of the flat-field file contents, prefixed with basename.
        Falls back to basename-only with a warning if the file is unreadable
        (e.g. on the Pi where the training map may not be present)."""
        if not path:
            return None
        import hashlib
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return f"{os.path.basename(path)}:{h.hexdigest()[:16]}"
        except OSError as exc:
            print(f"[contract] WARNING: could not hash flat-field '{path}' "
                  f"({exc}); using basename only. Contract will not detect a "
                  f"map swap if two maps share the same filename.")
            return os.path.basename(path)

    def deploy_contract(self) -> dict:
        """
        Everything the on-device decoders must agree with, plus a stable hash.

        The payload covers GEOMETRY, the DECODE CONSTANTS **and the
        PREPROCESSING CHAIN** (F8). CLAHE and the flat-field map are not
        cosmetic: EdgeAwareAttention.calibrate_bias measures a statistic of
        the CLAHE'd, flat-fielded canvas and bakes it into proj.bias, and the
        INT8 activation ranges are calibrated on that same distribution. A
        model trained at clahe_clip=2.0 and deployed at 3.0 decodes fine,
        passes every geometry check, and loses accuracy silently.
        """
        payload = {
            "num_classes": NUM_CLASSES,
            "img_h": int(self.data.img_h),
            "img_w": int(self.data.img_w),
            "strides": [int(s) for s in self.model.strides],
            "eaa_edge_stride": int(self.model.eaa_edge_stride),
            "eaa_pool_factor": int(self.model.eaa_pool_factor),
            "decode_offset_scale": float(DECODE_OFFSET_SCALE),
            "decode_offset_bias": float(DECODE_OFFSET_BIAS),
            "reg_log_clamp_min": float(REG_LOG_CLAMP_MIN),
            "reg_log_clamp_max": float(REG_LOG_CLAMP_MAX),
            "min_box_px": float(MIN_BOX_PX),
            "max_det": int(self.model.max_det),
            "pre_nms_topk": int(PRE_NMS_TOPK),
            # 3 blobs per level: cls (1ch), off (2ch), size (2ch).
            "blobs_per_level": 3,
            "blob_channels": {"cls": 1, "off": 2, "size": 2},
            "input_blob": "images",
            # ---- preprocessing contract ----
            "clahe_enabled": bool(self.aug.clahe_enabled),
            "clahe_clip": float(self.aug.clahe_clip),
            "clahe_grid": int(self.aug.clahe_grid),
            "flat_field": self._flat_field_digest(self.aug.flat_field_path),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["hash"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return payload


# ===========================================================================
# DATASET-DERIVED HELPERS — the dependence is in code, not in a magic number
# ===========================================================================

def copy_paste_p_for(n_train_boxes: int) -> float:
    """
    Box-level copy-paste attacks positive scarcity, so its usefulness scales
    INVERSELY with how many positives the dataset already has.

        <=   1k boxes -> 0.50  (cap)
        ~   10k boxes -> 0.25
        >= 100k boxes -> 0.05  (floor)

    Log-linear interpolation between the 1k and 100k anchors.
    """
    n = max(int(n_train_boxes), 1)
    lo_n, lo_p = 1_000.0, 0.50
    hi_n, hi_p = 100_000.0, 0.05
    if n <= lo_n:
        return lo_p
    if n >= hi_n:
        return hi_p
    t = (math.log(n) - math.log(lo_n)) / (math.log(hi_n) - math.log(lo_n))
    return float(lo_p + t * (hi_p - lo_p))


def ema_tau_for(steps_per_epoch: int, target_epochs: float = 5.0) -> int:
    """
    EMA time constant in STEPS, sized so the average converges in roughly
    ``target_epochs`` regardless of dataset size. A hardcoded tau is a hidden
    dataset dependence.
    """
    return max(1, int(round(max(1, int(steps_per_epoch)) * float(target_epochs))))


def calibration_paths(data_root: str, split: str = "train",
                      n: Optional[int] = None,
                      seed: int = 42) -> list:
    """
    Seeded-random selection of calibration image paths.
    Used by BOTH quantize_qdq.NIRCalibrationReader and
    export_ncnn.build_calibration_images so ORT and NCNN calibrate on
    identical images. seed comes from cfg.export.calib_seed.
    """
    import random as _random
    from dataset import list_images, resolve_split_dirs
    img_dir, _ = resolve_split_dirs(data_root, split)
    paths = sorted(list_images(img_dir))
    rng = _random.Random(int(seed))
    rng.shuffle(paths)
    return paths if n is None else paths[:int(n)]


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
    if h > MAX_SAFE_CANVAS_WIDTH:
        errors.append(
            f"canvas height {h} exceeds {MAX_SAFE_CANVAS_WIDTH}: head._grid() "
            f"rejects grids above {BF16_EXACT_INT_MAX} rows")

    # EAA edge geometry. NIRDet.__init__ enforces this too, but only AFTER
    # deploy_contract() has already hashed an inconsistent pair (F13).
    tot = int(cfg.model.eaa_edge_stride) * int(cfg.model.eaa_pool_factor)
    if cfg.model.strides and tot != int(cfg.model.strides[0]):
        errors.append(
            f"eaa_edge_stride * eaa_pool_factor = {tot} != finest stride "
            f"{cfg.model.strides[0]}: the edge map would not align with P3, "
            f"and deploy_contract() would hash a model that cannot be built")

    # max_det must fit in the C's static NIRDET_MAX_DET buffer. A value above
    # 300 makes every Python decoder work while nirdet_postprocess returns -1
    # on device with no other diagnostic.
    if int(cfg.model.max_det) > NIRDET_MAX_DET:
        errors.append(
            f"cfg.model.max_det={cfg.model.max_det} exceeds NIRDET_MAX_DET="
            f"{NIRDET_MAX_DET}. nirdet_postprocess will return -1 on device. "
            f"Lower max_det or rebuild the C with a larger NIRDET_MAX_DET and "
            f"python gen_contract_c.py --emit.")
    if int(cfg.model.max_det) <= 0:
        errors.append(f"cfg.model.max_det must be > 0, got {cfg.model.max_det}")

    # F12: CSPBlock splits channel counts in half and raises on odd values —
    # catch it here, before deploy_contract() is hashed.
    for ch_name in ("neck_channels", "neck_out3_channels", "head_channels"):
        v = int(getattr(cfg.model, ch_name))
        if v % 2:
            errors.append(
                f"model.{ch_name}={v} must be even — CSPBlock.__init__ splits "
                f"it in half (hidden = out_ch // 2) and raises ValueError on odd "
                f"values. This is caught here, before deploy_contract() is hashed.")
        elif v % 8:
            warns.append(
                f"model.{ch_name}={v} is not a multiple of 8; Neural-ART and "
                f"NCNN prefer 8-aligned channel counts for optimal vectorisation.")

    # F21 / F08 (audit fix): warn when the TRAIN split actually letterboxes
    # with padding at this canvas. Padding is filled with 0.0, which is
    # indistinguishable from a genuinely black sensor region and creates a
    # hard edge at the boundary that EAA's edge conv responds to as a
    # pedestrian silhouette. The previous version probed cfg.data.src_img_h /
    # src_img_w — fields that exist on no dataclass and that no code path
    # ever sets, so the branch could never fire. This reads the MEASURED
    # zero_pad_fraction transported by DatasetProfile.apply.
    if getattr(cfg.data, "profile_applied", False) and cfg.data.n_train:
        zpf = float(getattr(cfg.data, "train_zero_pad_fraction", 1.0))
        if zpf < 1.0:
            warns.append(
                f"only {zpf * 100.0:.1f}% of train images letterbox with zero "
                f"padding at canvas {cfg.data.img_h}x{cfg.data.img_w}; the "
                f"remaining {100.0 * (1.0 - zpf):.1f}% are padded with 0.0, "
                f"which is indistinguishable from a black sensor region and "
                f"creates a spurious edge at the pad boundary that EAA's edge "
                f"conv responds to. Canvas-space box statistics (prior_w / "
                f"prior_h) also stop equalling raw label-space statistics.")

    if cfg.eval.select_split == cfg.eval.report_split:
        errors.append(
            f"eval.select_split and eval.report_split are both "
            f"'{cfg.eval.select_split}': best.pth is chosen by maximising "
            f"mAP on the selection split, so reporting the headline number "
            f"on that same split is optimistically biased. Use "
            f"select_split='val', report_split='test'.")

    if cfg.train.batch_size < 4:
        warns.append("batch_size < 4 makes BatchNorm statistics unreliable")

    # ---- priors: unset is FATAL (F9) ----
    if cfg.model.prior_w is None or cfg.model.prior_h is None:
        errors.append(
            "model.prior_w / prior_h are unset. They are CANVAS-SPACE DATASET "
            "MEASUREMENTS that seed head.size_pred.bias, which is the only "
            "reason Task-Aligned Assignment finds a positive at epoch 0. "
            "Supply a dataset profile:\n"
            "  python dataset_profiles.py --root <path> --out datasets/<n>.yaml\n"
            "  python train.py --profile datasets/<n>.yaml")
    else:
        if not (0.0 < cfg.model.prior_w < 1.0 and 0.0 < cfg.model.prior_h < 1.0):
            errors.append("prior_w / prior_h must be normalised into (0, 1)")
        else:
            for label, prior in (("prior_w", cfg.model.prior_w),
                                 ("prior_h", cfg.model.prior_h)):
                lg = math.log(prior)
                if not (REG_LOG_CLAMP_MIN < lg < REG_LOG_CLAMP_MAX):
                    errors.append(
                        f"log({label}) = {lg:.3f} is not strictly inside the "
                        f"reg clamp ({REG_LOG_CLAMP_MIN}, "
                        f"{REG_LOG_CLAMP_MAX}); the head bias init would be "
                        f"clipped and TAL would cold-start with zero positives")

    if cfg.eval.deploy_score_thresh is not None \
            and cfg.eval.eval_score_thresh >= cfg.eval.deploy_score_thresh:
        warns.append("eval_score_thresh >= deploy_score_thresh (unusual)")
    if cfg.eval.deploy_score_thresh is None:
        warns.append("eval.deploy_score_thresh is unset (None): no dataset "
                     "profile has supplied a measured threshold yet")
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
                     "calibrated. train.py calls eaa.calibrate_bias() before "
                     "epoch 0; without it the spatial gate is a near-uniform "
                     "rescale that the following BatchNorm absorbs entirely")

    if cfg.model.backbone_p5_dilation > 3:
        warns.append(f"backbone_p5_dilation={cfg.model.backbone_p5_dilation}: "
                     f"ST advises against large dilation factors")

    widest = w // min(cfg.model.strides)
    prod = widest * cfg.model.head_channels
    if prod > ST_FEATURE_WIDTH_CHANNEL_LIMIT:
        warns.append(
            f"head stride-{min(cfg.model.strides)} input is {widest} wide x "
            f"{cfg.model.head_channels} ch = {prod} > "
            f"{ST_FEATURE_WIDTH_CHANNEL_LIMIT}: the Neural-ART compiler will "
            f"split this operator into columns (export_onnx.py reports every "
            f"such node)")

    if not cfg.data.root:
        errors.append(
            "data.root is unset. Supply a dataset profile:\n"
            "  python dataset_profiles.py --root <path> --out datasets/<n>.yaml\n"
            "  python train.py --profile datasets/<n>.yaml")
    if not cfg.data.profile_applied:
        warns.append("no dataset profile applied")

    if verbose:
        c = cfg.deploy_contract()
        pw = "unset" if cfg.model.prior_w is None else f"{cfg.model.prior_w:.6f}"
        ph = "unset" if cfg.model.prior_h is None else f"{cfg.model.prior_h:.6f}"
        print("=" * 62)
        print("  NIRDet-Lite config validation")
        print("=" * 62)
        print(f"  input        : {cfg.data.num_channels}ch x {h}x{w}"
              f"  (AR {w / h:.4f})")
        print(f"  classes      : {cfg.data.num_classes} "
              f"({', '.join(cfg.data.class_names)})")
        print(f"  strides      : {cfg.model.strides}  grids {cfg.grid_sizes}"
              f"  cells {cfg.num_cells}")
        print(f"  backbone     : stem {cfg.model.stem_ch} -> base "
              f"{cfg.model.base_ch} -> stages {cfg.model.base_ch * 4}"
              f"  p5_dilation {cfg.model.backbone_p5_dilation}")
        print(f"  neck         : {cfg.model.neck_channels} "
              f"(out3 {cfg.model.neck_out3_channels}) "
              f"pan={cfg.model.neck_pan} sk={cfg.model.neck_sk}")
        print(f"  head         : {cfg.model.head_channels} ch, "
              f"per-level BN={cfg.model.head_per_level_bn}, "
              f"blobs/level=3 (cls/off/size)")
        print(f"  priors       : w {pw}  h {ph}")
        print(f"  preprocess   : clahe={cfg.aug.clahe_enabled} "
              f"clip={cfg.aug.clahe_clip} grid={cfg.aug.clahe_grid} "
              f"flat_field={cfg.aug.flat_field_path!r}")
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
    # Priors are unset by default and that is now fatal; supply synthetic
    # values so this demo can print a contract. These are NOT measurements.
    cfg.model.prior_w, cfg.model.prior_h = 0.05, 0.15
    cfg.data.root = "."
    validate_config(cfg)
    print("\ndeploy contract:")
    for k, v in cfg.deploy_contract().items():
        print(f"  {k:<22} {v}")
    print("\ncopy_paste_p_for:")
    for n in (500, 1_000, 5_000, 20_000, 146_000):
        print(f"  {n:>7} boxes -> {copy_paste_p_for(n):.3f}")
    print("\nema_tau_for:")
    for spe in (32, 100, 1_000, 18_000):
        print(f"  {spe:>6} steps/epoch -> tau {ema_tau_for(spe)}")
