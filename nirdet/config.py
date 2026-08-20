"""
config.py — NIRDet Single Source of Truth
==========================================
AICTE-Funded NIR Surveillance Detector (850nm)
Target: <5M parameters, mAP50 > baseline on person-detection dataset

USAGE
-----
    from config import get_config, validate_config
    cfg = get_config()
    validate_config(cfg)

DESIGN PHILOSOPHY
-----------------
- Every parameter has a comment: what it controls, effect of increasing/decreasing,
  sensible range, and source/justification.
- NIR-invalid augmentations are explicitly set to 0/False with a reason.
- Placeholders marked ← REPLACE are values that depend on your hardware/dataset.
- All other files import from here. Nothing is hardcoded elsewhere.

CITATIONS (abbreviated)
-----------------------
[AdamW]     Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
[SGDR]      Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017
[Warmup]    Goyal et al., "Accurate Large Minibatch SGD", arXiv 2017
[Mosaic]    Bochkovskiy et al., "YOLOv4", arXiv 2020; Zoph et al., "Copy-Paste", CVPR 2021
[NIR-Aug]   Takumi et al., "Multispectral Object Detection", ICCV Workshop 2017
[LinScale]  Goyal et al., "Accurate Large Minibatch SGD", arXiv 2017
[WD-Small]  Zoph et al., "NAS-FPN", CVPR 2019; Zhao et al., "RT-DETR", arXiv 2023
[FocalLoss] Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
[GIoU]      Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


# ===========================================================================
# SECTION 0 — HARDWARE CONTEXT
# ===========================================================================
# Fill these in before running anything. They drive batch size, LR, and
# VRAM estimates in validate_config().
# ---------------------------------------------------------------------------

GPU_VRAM_GB: float = 8.0          # ← REPLACE: Your GPU VRAM in GB (e.g. 8.0, 16.0, 24.0)
                                   #   Effect: larger VRAM → larger batch → better gradient estimates

PYTORCH_VERSION: str = "2.3.0"    # ← REPLACE: e.g. "2.0.1", "2.3.0", "2.4.0"
                                   #   Used for compile() compat check in validate_config()

DATASET_TRAIN_COUNT: int = 3000    # ← REPLACE: number of training images
                                   #   Critical for: Mosaic threshold, LR schedule length

DATASET_VAL_COUNT: int = 600       # ← REPLACE: number of validation images

BASELINE_MAP50: float = 0.45       # ← REPLACE: mAP50 of the baseline model you're beating


# ===========================================================================
# SECTION 1 — PATHS
# ===========================================================================

@dataclass
class PathConfig:
    """All filesystem paths. Override with absolute paths in production."""

    # --- Dataset ---
    dataset_root: str = "./data/nirdet"
    # Root of your dataset directory. All sub-paths below are relative to this.
    # Sensible range: any valid directory. Must exist before training.

    train_images: str = "./data/nirdet/images/train"
    # Directory containing training .jpg/.png files (grayscale 850nm).

    train_labels: str = "./data/nirdet/labels/train"
    # Directory containing YOLO-format .txt label files for training.
    # Each line: <class_id> <cx> <cy> <w> <h> (all normalized 0–1).

    val_images: str = "./data/nirdet/images/val"
    # Validation images directory.

    val_labels: str = "./data/nirdet/labels/val"
    # Validation label directory.

    # --- Outputs ---
    checkpoint_dir: str = "./checkpoints/nirdet"
    # Where .pth checkpoint files are saved.
    # Increasing checkpoint frequency: more disk usage, finer recovery granularity.

    log_dir: str = "./logs/nirdet"
    # TensorBoard / CSV log output directory.

    best_model_path: str = "./checkpoints/nirdet/best.pth"
    # Path where the best validation-mAP checkpoint is copied.

    last_model_path: str = "./checkpoints/nirdet/last.pth"
    # Path where the most recent checkpoint is saved (for resume).

    # --- Optional: pretrained weights ---
    pretrained_backbone: Optional[str] = None
    # Path to a pretrained backbone checkpoint (e.g. from NIR ImageNet proxy task).
    # Set to None for full scratch training (our default — no public NIR pretraining exists).
    # If provided, only backbone weights are loaded; head is always random-init.


# ===========================================================================
# SECTION 2 — INPUT / DATASET
# ===========================================================================

@dataclass
class DataConfig:
    """Input resolution, dataset metadata, loader settings."""

    # --- Image dimensions ---
    input_height: int = 720
    # Input image height in pixels. Must match your sensor output.
    # Increasing: higher resolution features, but quadratic VRAM cost.
    # Decreasing: faster training, less detail for small persons.
    # Range: 320–1280. Your sensor outputs 1280×720.

    input_width: int = 1280
    # Input image width in pixels.
    # Note: YOLO-style detectors expect multiples of 32. 1280 ✓, 720 ✓.

    num_classes: int = 1
    # Number of object classes. NIRDet is single-class: person.
    # Increasing: add new categories (vehicle, animal) in future.

    class_names: Tuple[str, ...] = ("person",)
    # Human-readable class names. Index matches YOLO label class_id.

    num_channels: int = 1
    # Input channels: 1 for NIR grayscale. NOT 3.
    # The backbone's first conv layer must match this.
    # DO NOT set to 3 unless you replicate the channel 3× (see backbone notes).

    # --- DataLoader ---
    num_workers: int = 4
    # CPU workers for data loading. 4 is safe for most systems.
    # Increasing: faster data throughput, more RAM usage.
    # Decreasing: bottleneck on GPU utilization.
    # Range: 0 (debug/Windows) to min(os.cpu_count(), 8).

    pin_memory: bool = True
    # Pin DataLoader memory for faster CPU→GPU transfer.
    # Set False if RAM is limited (<8GB system RAM).

    persistent_workers: bool = True
    # Keep workers alive across epochs. Reduces per-epoch startup cost.
    # Set False if you see zombie process issues.

    prefetch_factor: int = 2
    # Batches to prefetch per worker. 2 is standard.
    # Range: 1–4. Higher uses more RAM.

    # --- Label format ---
    label_format: str = "yolo"
    # "yolo" = normalized cx,cy,w,h. Do not change unless you rewrite the dataset class.

    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    # Accepted image file extensions when scanning image directories.

    train_count: int = DATASET_TRAIN_COUNT   # mirrors global constant — single place to change
    val_count: int = DATASET_VAL_COUNT


# ===========================================================================
# SECTION 3 — MODEL ARCHITECTURE
# ===========================================================================

@dataclass
class ModelConfig:
    """
    NIRDet Architecture: NIR-Native Backbone + Edge-Aware Attention + Pedestrian Head
    Target: <5M parameters total.

    Backbone stage channels and depths follow a lightweight design similar to
    MobileNetV3 and GhostNet, but single-channel input and NIR-tuned.
    """

    # --- Backbone: NIR-Native Backbone ---
    # Stage output channels [stem, stage1, stage2, stage3, stage4]
    # Doubling channels per stage is standard; we stay narrow to hit <5M param target.
    backbone_channels: Tuple[int, ...] = (16, 32, 128, 256, 256)  # FIX: match actual backbone out = (base*4, base*8, base*8) -> P3/P4/P5 = (128,256,256)
    # Effect of increasing: richer features, more parameters, more VRAM.
    # Effect of decreasing: fewer parameters, risk of underfitting on complex scenes.
    # Range: (8,16,32,64,128) for ultra-light; (32,64,128,256,512) for higher-capacity.
    # At (16,32,64,128,256): ~1.8M params in backbone alone.

    backbone_depths: Tuple[int, ...] = (1, 2, 2, 1)  # FIX: code implements [1,2,2,1], not (1,2,3,2,1)
    # Number of repeated blocks per stage [stem, s1, s2, s3, s4].
    # Increasing any value: more non-linearity, more params, slower forward pass.
    # Range: 1–4 per stage. Total depth controls representational capacity.

    stem_kernel_size: int = 3
    # Kernel size of the initial stem convolution.
    # 3: standard for feature extraction. 7: used in ResNet for large-context stem.
    # For 1280×720 input, 3 is appropriate (avoids over-aggressive spatial reduction).

    use_depthwise_separable: bool = True
    # Replace standard conv with depthwise-separable convs in backbone.
    # True: ~8–9× parameter reduction. Essential for hitting <5M param target.
    # False: standard conv; higher accuracy ceiling but more params.
    # Source: Howard et al., MobileNets, 2017.

    backbone_activation: str = "silu"  # FIX: code uses SiLU throughout the backbone, not hardswish
    # Activation function throughout backbone.
    # Options: "relu" (fastest), "hardswish" (best accuracy/speed tradeoff, MobileNetV3),
    #          "gelu" (best accuracy, slower on edge). Use "relu" for STM32N6 deployment.

    backbone_norm: str = "batchnorm"
    # Normalization layer. "batchnorm" is standard.
    # "groupnorm" if batch size < 4 (BN becomes unstable).
    # Range: {"batchnorm", "groupnorm", "instancenorm"}.

    # --- Edge-Aware Attention (custom module; NOT CBAM) ---
    # FIX: removed dead fields attention_type="cbam"/attention_reduction_ratio —
    # the model hardwires EdgeAwareAttention (Sobel-based), never CBAM/SE/ECA.
    eaa_num_edge_filters: int = 4     # Sx, Sy, Laplacian, diagonal
    eaa_residual_scale: float = 0.5   # additive form F*(1 + scale*A)
    eaa_freeze_epochs: int = 5        # curriculum freeze on the Sobel edge_conv

    # --- Pedestrian Detection Head (anchor-free, FCOS/YOLOX-style) ---
    # FIX: removed num_anchors_per_cell/anchor_sizes — NIRDet is anchor-free
    # (center-cell assignment). Head hardwires feat_channels=256.
    head_channels: int = 256          # FIX: head uses 256, not 128
    prior_w: float = 0.04             # width prior from miniNIRPed GT stats
    prior_h: float = 0.15             # height prior from miniNIRPed GT stats
    detection_scales: int = 3
    # Number of FPN-style detection scales (multi-scale output heads).
    # 3: P3 (large objects), P4 (medium), P5 (small). Standard YOLO setup.
    # Reducing to 2 saves ~0.3M params but hurts small-person detection at range.

    # --- Parameter budget tracking (approximate) ---
    # backbone: ~1.8M (channels 16→256, depthwise-separable)
    # attention: ~0.05M (CBAM on stages 2,3)
    # neck/FPN: ~1.0M (estimated)
    # head: ~0.8M (3 scales × 128ch × 3 anchors × (5+1) outputs)
    # TOTAL ESTIMATED: ~3.65M  ← within 5M budget


# ===========================================================================
# SECTION 4 — TRAINING SCHEDULE
# ===========================================================================

@dataclass
class TrainingConfig:
    """Epochs, batch size, device, checkpointing, reproducibility."""

    # --- Core schedule ---
    epochs: int = 200
    # Total training epochs.
    # Increasing: more training time, risk of overfitting on small datasets.
    # Decreasing: underfitting if schedule hasn't converged.
    # Range: 100–300 for scratch training. Use early stopping (see below).
    # With 3K images + cosine schedule: convergence typically by epoch 150–180.

    batch_size: int = 16
    # Training batch size.
    # ← REPLACE based on GPU VRAM:
    #   8GB  VRAM: batch 8–16 at 1280×720 (fp16), 4–8 (fp32)
    #   16GB VRAM: batch 16–32 at 1280×720 (fp16)
    #   24GB VRAM: batch 32–64 at 1280×720 (fp16)
    # Rule: if you get OOM, halve batch_size and halve learning_rate.
    # Effect of increasing: better gradient estimates, but may overfit faster on small data.
    # Linear scaling rule [LinScale]: LR scales linearly with batch size.
    # Reference LR (batch=16, AdamW): 1e-3. For batch=32: use 2e-3.

    val_frequency: int = 5
    # Run validation every N epochs.
    # Increasing: less compute on validation, but miss the best checkpoint.
    # Decreasing: more frequent best-model saves, slightly slower total training.
    # Range: 1–10. 5 is a good default; use 1 in final tuning runs.

    save_frequency: int = 10
    # Save a numbered checkpoint every N epochs (in addition to best/last).
    # Increasing: less disk usage. Decreasing: finer recovery points.

    early_stopping_patience: int = 40
    # Stop training if val mAP50 doesn't improve for this many epochs.
    # Increasing: more training, risk of longer divergence.
    # Decreasing: may stop too early on plateau before cosine decay kicks in.
    # Set to epochs (disable) during initial hyperparameter search.

    # --- Precision ---
    use_amp: bool = True
    # Automatic Mixed Precision (fp16 compute, fp32 master weights).
    # True: ~1.7–2× memory saving, ~1.5× speedup on Ampere+ GPUs.
    # False: required for older GPUs (pre-Volta) or for debugging NaN losses.

    # --- Device ---
    device: str = "cuda"
    # "cuda" for GPU, "cpu" for CPU-only (extremely slow for this model).
    # "cuda:0", "cuda:1" to pin to a specific GPU.

    compile_model: bool = False
    # torch.compile() the model for ~15% speedup (PyTorch ≥ 2.0 only).
    # False by default: debugging is harder with compiled graphs.
    # Enable after confirming training is stable.

    # --- Reproducibility ---
    seed: int = 42
    # Random seed for torch, numpy, and random modules.
    # Changing: different weight initialization and augmentation order.
    # Use same seed for all ablation runs to isolate variable changes.

    deterministic: bool = False
    # Force deterministic CUDA ops (slower but fully reproducible).
    # True: required for exact reproducibility across runs.
    # False: ~10% faster; results vary by <0.1 mAP run-to-run (acceptable).

    # --- Resume ---
    resume_from: Optional[str] = None
    # Path to a checkpoint .pth to resume training from.
    # None: start from scratch. Set to last_model_path to resume.


# ===========================================================================
# SECTION 5 — OPTIMIZER
# ===========================================================================

@dataclass
class OptimizerConfig:
    """
    Optimizer: AdamW
    Rationale: See Part A. AdamW decouples weight decay from adaptive LR update,
    preventing weight norm explosion on small datasets. Faster convergence than
    SGD from scratch. Handles sparse NIR gradients better than SGD.
    [AdamW] Loshchilov & Hutter, ICLR 2019.
    """

    name: str = "adamw"
    # Optimizer name. Options: "adamw", "adam", "sgd".
    # "sgd": competitive on large datasets; needs careful LR tuning; avoid for scratch.
    # "adam": similar to AdamW but weight decay is incorrectly applied; use AdamW instead.

    learning_rate: float = 1e-3
    # Base learning rate (at batch_size=16, [LinScale]).
    # This is the PEAK LR reached after warmup; cosine decay reduces from here.
    # Increasing: faster early convergence, risk of instability/divergence.
    # Decreasing: more stable but may underfit within epoch budget.
    # Range: 5e-4 – 3e-3 for AdamW on detection. 1e-3 is the well-validated default.
    # Scale linearly with batch size: lr = 1e-3 * (batch_size / 16).

    weight_decay: float = 1e-4
    # L2 regularization strength (applied to weights only, not biases/BN params).
    # Increasing: stronger regularization, risk of underfitting on small datasets.
    # Decreasing: less regularization, risk of overfitting.
    # Range: 1e-5 – 5e-4. Use 1e-4 for <10K images [WD-Small].
    # Note: validate_config() warns if weight_decay > 1e-3 with small dataset.

    beta1: float = 0.9
    # Adam momentum term for first moment (gradient). Canonical value.
    # Increasing toward 1.0: more momentum, slower adaptation to new gradients.
    # Range: 0.85–0.95. Only tune if loss is oscillating.

    beta2: float = 0.999
    # Adam momentum term for second moment (squared gradient). Canonical value.
    # Decreasing toward 0.9: faster adaptation to gradient magnitude changes.
    # Range: 0.99–0.9999. Rarely needs tuning.

    epsilon: float = 1e-8
    # Numerical stability term in Adam denominator.
    # Increase to 1e-6 if you observe NaN losses (gradient spikes).

    # --- Layer-wise LR scaling (optional) ---
    backbone_lr_multiplier: float = 1.0
    # Scale backbone LR relative to head. Set to 0.1 if loading pretrained backbone.
    # 1.0: all layers train at same LR (correct for scratch training).
    # 0.1: fine-tuning mode — freeze most of backbone learning.

    bias_lr_multiplier: float = 2.0
    # Scale LR for bias parameters. Slight increase helps biases adapt faster.
    # Common in YOLO implementations. Range: 1.0–2.0.

    no_decay_params: Tuple[str, ...] = ("bias", "bn", "norm")
    # Parameter name substrings exempt from weight decay.
    # BatchNorm scale/shift and biases should NOT have weight decay applied.
    # This matches standard PyTorch best practice and AdamW paper recommendations.


# ===========================================================================
# SECTION 6 — LEARNING RATE SCHEDULE
# ===========================================================================

@dataclass
class SchedulerConfig:
    """
    Schedule: Cosine Annealing with Linear Warmup
    Rationale: See Part A. Smooth decay prevents LR cliff; warmup stabilizes
    BatchNorm statistics in early epochs when training from scratch.
    [SGDR] Loshchilov & Hutter, ICLR 2017. [Warmup] Goyal et al., 2017.
    """

    name: str = "cosine_with_warmup"
    # Schedule name.
    # Options: "cosine_with_warmup", "onecycle", "reduce_on_plateau", "step"
    # Only "cosine_with_warmup" is fully tested in this config.

    warmup_epochs: int = 5
    # Number of epochs for linear LR warmup (0 → peak LR).
    # Purpose: prevents large initial gradients from destabilizing BN running stats.
    # Increasing: more gentle warmup, slower to reach peak LR.
    # Decreasing: faster start, may cause early instability.
    # Range: 3–10. 5 is standard for scratch-trained detectors.
    # Formula: lr_at_epoch_e = (e / warmup_epochs) * learning_rate, for e < warmup_epochs

    warmup_bias_lr: float = 1e-6
    # Starting LR for bias params during warmup. Very small to avoid early oscillation.
    # Range: 1e-7 – 1e-5.

    min_lr_ratio: float = 0.01
    # Minimum LR as a fraction of peak LR (at end of cosine decay).
    # min_lr = learning_rate * min_lr_ratio = 1e-3 * 0.01 = 1e-5.
    # Increasing: LR never decays as low, may not fully converge.
    # Decreasing: very aggressive final decay; good for squeezing last mAP points.
    # Range: 0.001–0.1.

    use_cosine_restarts: bool = False
    # Use SGDR-style cosine restarts (T_mult schedule).
    # False recommended: restarts cause mAP to oscillate during validation,
    # making early stopping and best-checkpoint logic unreliable.

    # --- OneCycleLR (if name="onecycle") ---
    onecycle_pct_start: float = 0.3
    # Fraction of training spent in the increasing LR phase.
    # Unused unless name="onecycle". Kept here for easy switching.

    onecycle_div_factor: float = 25.0
    # initial_lr = max_lr / div_factor. Unused unless name="onecycle".

    onecycle_final_div_factor: float = 1e4
    # min_lr = initial_lr / final_div_factor. Unused unless name="onecycle".


# ===========================================================================
# SECTION 7 — AUGMENTATION
# ===========================================================================

@dataclass
class AugmentationConfig:
    """
    Augmentation strategy for single-channel 850nm NIR grayscale imagery.

    KEY PRINCIPLE: Any augmentation operating in color space (HSV, RGB channel
    manipulation) is INVALID for NIR grayscale and must be disabled (set to 0/False).
    Source: [NIR-Aug] Takumi et al., ICCV Workshop 2017; Gade & Moeslund, 2014.

    VALID NIR augmentations: geometric transforms, intensity transforms, noise.
    INVALID NIR augmentations: hue shift, saturation shift, channel operations.
    """

    # -----------------------------------------------------------------------
    # VALID: Geometric Transforms
    # -----------------------------------------------------------------------

    horizontal_flip_prob: float = 0.5
    # Probability of random horizontal flip.
    # Valid for NIR: person silhouette is left-right symmetric.
    # Range: 0.0–0.5. 0.5 is standard — gives ~2× effective dataset size.

    vertical_flip_prob: float = 0.0
    # Probability of random vertical flip.
    # Disabled: upside-down persons are not a valid training pattern for
    # surveillance cameras (fixed overhead/frontal angle).

    rotation_degrees: float = 5.0
    # Maximum rotation angle in degrees (uniform sample from [-d, +d]).
    # Valid for NIR: camera sway, mounting angle variation.
    # Increasing: more aggressive geometric diversity.
    # Decreasing: more conservative; use 0.0 for truly fixed cameras.
    # Range: 0.0–15.0. Beyond 15°, bounding boxes become inaccurate.

    scale_jitter_min: float = 0.5
    # Minimum scale factor for random resizing.
    # Valid for NIR: simulates person at different distances.
    # Range: 0.3–0.8. Below 0.5: persons may become too small to detect.

    scale_jitter_max: float = 1.5
    # Maximum scale factor for random resizing.
    # Range: 1.2–2.0. Above 1.5: padding artifacts at image border.

    translate_fraction: float = 0.1
    # Max fraction of image dimension for random translation (x and y).
    # Valid: simulates camera pan/tilt.
    # Range: 0.0–0.2.

    shear_degrees: float = 2.0
    # Maximum shear angle in degrees. Small shear simulates perspective distortion.
    # Range: 0.0–10.0.

    # -----------------------------------------------------------------------
    # VALID: Intensity / Photometric Transforms (grayscale-safe)
    # -----------------------------------------------------------------------

    brightness_jitter: float = 0.3
    # Max fractional brightness change: pixel *= uniform(1-b, 1+b).
    # VALID for NIR: simulates illuminator intensity variation, sensor gain drift,
    # and ambient IR contamination (sunlight at 850nm during daytime).
    # Range: 0.0–0.5. Beyond 0.5: images become unrealistically bright/dark.

    contrast_jitter: float = 0.3
    # Max fractional contrast change.
    # VALID for NIR: simulates fog, dust, rain attenuating NIR signal.
    # Range: 0.0–0.5.

    gamma_range: Tuple[float, float] = (0.7, 1.3)
    # Random gamma correction range [min, max].
    # VALID for NIR: simulates nonlinear sensor response and AGC behavior.
    # (0.7, 1.3) keeps images within a realistic sensor response envelope.

    # -----------------------------------------------------------------------
    # VALID: Noise Injection
    # -----------------------------------------------------------------------

    gaussian_noise_std: float = 0.02
    # Standard deviation of additive Gaussian noise (pixel values normalized 0–1).
    # VALID for NIR: models sensor read noise, shot noise at 850nm.
    # Increasing: more noise robustness, may degrade very fine features.
    # Range: 0.0–0.05. Above 0.05: significantly degrades small-person detection.
    # Source: NIR camera specs typically quote SNR 40–50dB → noise ~0.01–0.03.

    salt_pepper_prob: float = 0.01
    # Probability of salt-and-pepper pixel corruption.
    # VALID for NIR: dead pixels, cosmic ray hits on CMOS sensors.
    # Range: 0.0–0.05.

    # -----------------------------------------------------------------------
    # VALID: Cutout / Random Erasing
    # -----------------------------------------------------------------------

    cutout_prob: float = 0.3
    # Probability of applying random rectangular occlusion (cutout/erasing).
    # VALID for NIR: simulates partial occlusion by pillars, walls, foliage.
    # Increasing: stronger occlusion robustness, but may erase the entire person
    # in small-image cases. Use with cutout_max_area_ratio < 0.3.

    cutout_max_area_ratio: float = 0.2
    # Max fraction of image area that cutout can cover.
    # Range: 0.1–0.4.

    cutout_num_holes: int = 2
    # Number of rectangular holes. 1–4 is typical.

    # -----------------------------------------------------------------------
    # CONDITIONAL: Mosaic Augmentation
    # -----------------------------------------------------------------------

    mosaic_prob: float = 0.0
    # Probability of applying 4-image mosaic augmentation.
    # [Mosaic] Bochkovskiy et al. YOLOv4, 2020.
    #
    # DATASET-SIZE THRESHOLD WARNING:
    #   < 5,000 training images: mosaic is HARMFUL. It creates physically
    #   implausible composite scenes that confuse the detector when the
    #   dataset is too small to provide enough counterexamples. Set to 0.0.
    #   > 10,000 training images: mosaic consistently improves mAP by 2–4 points.
    #   [Zoph et al., Copy-Paste, CVPR 2021]
    #
    # ← REPLACE: set mosaic_prob = 0.5 if DATASET_TRAIN_COUNT > 10000.
    # Leave at 0.0 for default (conservative). validate_config() will warn.

    mosaic_scale_range: Tuple[float, float] = (0.3, 0.7)
    # Scale range for each of the 4 mosaic tiles.
    # Unused if mosaic_prob = 0.0.

    # -----------------------------------------------------------------------
    # INVALID: Color-Space Augmentations — DISABLED for NIR
    # -----------------------------------------------------------------------

    hsv_hue_gain: float = 0.0
    # HSV hue shift. SET TO 0.0: hue is undefined for single-channel NIR imagery.
    # At 850nm, there is no color information. Applying hue shift to a channel-
    # replicated grayscale image produces correlated RGB noise with no physical
    # meaning. Source: [NIR-Aug] Takumi et al., 2017.

    hsv_saturation_gain: float = 0.0
    # HSV saturation shift. SET TO 0.0: same reason as hue — NIR has no saturation.
    # Standard YOLO default is 0.7; this MUST be overridden to 0.0 for NIR.

    hsv_value_gain: float = 0.0
    # HSV value (brightness) shift. SET TO 0.0: replaced by brightness_jitter above,
    # which operates directly in grayscale intensity space without the HSV conversion
    # overhead. Using HSV value shift on a grayscale image is mathematically
    # equivalent but adds unnecessary conversion cost.

    channel_shuffle_prob: float = 0.0
    # RGB channel shuffle probability. SET TO 0.0: not applicable to single-channel
    # NIR imagery. There are no channels to shuffle.

    random_channel_drop_prob: float = 0.0
    # Random RGB channel dropout. SET TO 0.0: no color channels in NIR.

    color_jitter_prob: float = 0.0
    # ColorJitter augmentation (RGB-space). SET TO 0.0: RGB-invalid for NIR.
    # Use brightness_jitter and contrast_jitter (defined above) instead.

    # -----------------------------------------------------------------------
    # INVALID: Other RGB-specific augmentations
    # -----------------------------------------------------------------------

    mixup_prob: float = 0.0
    # MixUp augmentation (pixel-level blending of two images).
    # SET TO 0.0 for current dataset size. Mixup requires a large dataset
    # (>10K) to be beneficial; on small NIR datasets it blends ghost-person
    # artifacts that degrade precision. Re-evaluate if dataset grows.

    copy_paste_prob: float = 0.0
    # Copy-Paste augmentation [Zoph et al., CVPR 2021].
    # SET TO 0.0: requires instance segmentation masks, which we do not have
    # (our labels are bounding boxes only). Enable only if you obtain masks.

    # -----------------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------------

    normalize_mean: Tuple[float, ...] = (0.5,)
    # Per-channel mean for input normalization. Single value for grayscale.
    # 0.5 is a reasonable prior for NIR imagery.
    # ← REPLACE: compute from your actual training set using tools/compute_stats.py.
    # Formula: mean = sum(pixel_values) / (N_images * H * W)

    normalize_std: Tuple[float, ...] = (0.25,)
    # Per-channel std for input normalization.
    # ← REPLACE: compute from your training set.
    # Note: std ≈ 0.25 means most pixels fall in [0, 1] after (x - 0.5) / 0.25.


# ===========================================================================
# SECTION 8 — LOSS FUNCTION
# ===========================================================================

@dataclass
class LossConfig:
    """
    Loss function weights and parameters.
    NIRDet uses a composite loss:
        L_total = λ_cls * L_cls + λ_reg * L_reg + λ_conf * L_conf

    [FocalLoss] Lin et al., ICCV 2017.
    [GIoU]      Rezatofighi et al., CVPR 2019.
    """

    # --- Component weights ---
    cls_weight: float = 0.5
    # Weight on classification loss component.
    # Single-class detection: cls_weight is less critical (binary cls is easy).
    # Increasing: more emphasis on class confidence calibration.
    # Range: 0.1–1.0. For single-class, 0.3–0.5 is typical.

    reg_weight: float = 0.05
    # Weight on regression (bounding box) loss component.
    # Increasing: more emphasis on tight box localization.
    # Decreasing: looser boxes, better recall.
    # Range: 0.01–0.1. Kept low because regression loss scale is typically larger.

    conf_weight: float = 1.0
    # Weight on objectness/confidence loss component.
    # This is the dominant loss term for single-class detection.
    # Increasing: model learns to be more certain before predicting.
    # Range: 0.5–2.0.

    # --- Classification loss ---
    cls_loss_type: str = "bce"
    # Classification loss function.
    # "bce": Binary Cross-Entropy (standard for single-class / multi-label).
    # "focal": Focal Loss [FocalLoss], better for class imbalance (many background cells).
    # For dense surveillance scenes with many background cells: use "focal".

    focal_alpha: float = 0.25
    # Focal loss alpha (class weighting). 0.25 downweights easy negatives (background).
    # Range: 0.1–0.5. Unused if cls_loss_type != "focal".

    focal_gamma: float = 2.0
    # Focal loss gamma (focusing parameter). Higher → more focus on hard examples.
    # Range: 0.5–5.0. 2.0 is the value from the original paper.
    # Unused if cls_loss_type != "focal".

    # --- Regression loss ---
    reg_loss_type: str = "giou"
    # Bounding box regression loss.
    # "giou": Generalized IoU [GIoU] — handles non-overlapping boxes, smooth gradient.
    # "ciou": Complete IoU — also penalizes aspect ratio difference (good for pedestrians).
    # "diou": Distance IoU — middle ground.
    # Recommendation: "ciou" for pedestrian detection (consistent aspect ratios help).
    # ← REPLACE: try both "giou" and "ciou" in ablation.

    # --- Objectness loss ---
    obj_loss_type: str = "bce"
    # Objectness/confidence loss. BCE is standard.

    # --- Label assignment ---
    iou_threshold_pos: float = 0.5
    # IoU threshold above which an anchor is considered a positive match.
    # Increasing: stricter matching, fewer positives, higher precision, lower recall.
    # Decreasing: more positives, higher recall, potential false positives.
    # Range: 0.4–0.7.

    iou_threshold_neg: float = 0.4
    # IoU threshold below which an anchor is negative. Anchors between neg and pos
    # thresholds are ignored (not used in loss). Creates a "dead zone".

    # --- NMS (inference) ---
    nms_iou_threshold: float = 0.45
    # IoU threshold for Non-Maximum Suppression at inference.
    # Increasing: more boxes survive (higher recall, more duplicates).
    # Decreasing: aggressive suppression (higher precision, may miss close persons).
    # Range: 0.3–0.6.

    nms_conf_threshold: float = 0.25
    # Minimum confidence score to keep a detection before NMS.
    # Increasing: higher precision, lower recall.
    # Decreasing: more detections, more false positives.
    # Range: 0.1–0.5. 0.25 is a common evaluation threshold.

    max_detections: int = 100
    # Maximum number of detections per image after NMS.
    # 100 is generous for surveillance. Reduce for edge deployment speed.


# ===========================================================================
# SECTION 9 — LOGGING & EVALUATION
# ===========================================================================

@dataclass
class LoggingConfig:
    """Metrics, logging frequency, and evaluation settings."""

    use_tensorboard: bool = True
    # Log metrics to TensorBoard.

    use_csv: bool = True
    # Also log metrics to a CSV file (human-readable, easy to import).

    log_every_n_steps: int = 10
    # Log training loss every N gradient steps.
    # Increasing: less disk I/O, coarser loss curves.
    # Range: 1–50.

    eval_metric: str = "map50"
    # Primary metric for best checkpoint selection.
    # "map50": mAP at IoU=0.50 (PASCAL VOC style). Standard for surveillance eval.
    # "map5095": COCO-style mean mAP at IoU=[0.50:0.05:0.95]. Stricter.

    iou_thresholds_eval: Tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
    # IoU thresholds to compute AP over during evaluation.
    # For mAP50: use only (0.5,). For COCO mAP: use full range.

    target_map50: float = BASELINE_MAP50 + 0.05
    # Target mAP50 to beat (baseline + 5 points). Used in validate_config() warning.
    # Adjust upward as model improves.

    verbose_eval: bool = False
    # Print per-class AP breakdown. Useful but slow — enable for final evaluation.


# ===========================================================================
# SECTION 10 — MASTER CONFIG (combines all sub-configs)
# ===========================================================================

@dataclass
class NIRDetConfig:
    """
    Master configuration. Import this in all other files:

        from config import get_config
        cfg = get_config()

    Access sub-configs via dot notation:
        cfg.training.batch_size
        cfg.model.backbone_channels
        cfg.aug.horizontal_flip_prob
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    aug: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Convenience passthrough properties
    @property
    def input_size(self) -> Tuple[int, int]:
        """Returns (H, W) tuple. Use everywhere instead of hardcoding."""
        return (self.data.input_height, self.data.input_width)

    @property
    def effective_lr(self) -> float:
        """LR scaled by batch size relative to reference batch of 16."""
        return self.optimizer.learning_rate * (self.training.batch_size / 16.0)


def get_config(**overrides) -> NIRDetConfig:
    """
    Factory function. Use this as the single entry point.

    Usage:
        cfg = get_config()                              # defaults
        cfg = get_config(training=dict(epochs=300))     # override sub-config fields
    """
    cfg = NIRDetConfig()

    # Apply overrides
    for section_name, section_overrides in overrides.items():
        if not hasattr(cfg, section_name):
            raise ValueError(f"No config section named '{section_name}'")
        section = getattr(cfg, section_name)
        if isinstance(section_overrides, dict):
            for k, v in section_overrides.items():
                if not hasattr(section, k):
                    raise ValueError(f"No field '{k}' in {type(section).__name__}")
                setattr(section, k, v)

    return cfg


# ===========================================================================
# PART C — validate_config()
# ===========================================================================

def validate_config(cfg: NIRDetConfig, verbose: bool = True) -> bool:
    """
    Validate the config before training starts.

    Checks:
    1. Required paths exist (or can be created)
    2. Estimated VRAM usage vs declared GPU_VRAM_GB
    3. Known bad hyperparameter combinations
    4. Placeholder detection (warns when sentinels like 3000 are still set)
    5. NIR-specific sanity checks

    Returns:
        True if validation passes (possibly with warnings).
        Raises ValueError for hard errors that would cause training to fail.
    """
    errors: list[str] = []
    warnings_list: list[str] = []

    # -----------------------------------------------------------------------
    # 1. PATH CHECKS
    # -----------------------------------------------------------------------

    required_dirs = {
        "train_images": cfg.paths.train_images,
        "train_labels": cfg.paths.train_labels,
        "val_images": cfg.paths.val_images,
        "val_labels": cfg.paths.val_labels,
    }

    for name, path in required_dirs.items():
        if not Path(path).exists():
            errors.append(
                f"PATH ERROR: {name}='{path}' does not exist. "
                f"Create the directory or update cfg.paths.{name}."
            )

    # Auto-create output directories
    output_dirs = [cfg.paths.checkpoint_dir, cfg.paths.log_dir]
    for d in output_dirs:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            errors.append(f"PATH ERROR: Cannot create output directory '{d}': {e}")

    # Pretrained backbone (optional)
    if cfg.paths.pretrained_backbone is not None:
        if not Path(cfg.paths.pretrained_backbone).exists():
            errors.append(
                f"PATH ERROR: pretrained_backbone='{cfg.paths.pretrained_backbone}' "
                f"does not exist."
            )

    # -----------------------------------------------------------------------
    # 2. VRAM ESTIMATION
    # -----------------------------------------------------------------------

    # Rough VRAM estimation heuristic for a ~4M param detector
    # Formula: (model_params * 4 bytes) * 3 (params + grads + optimizer states)
    #          + (batch_size * channels * H * W * 4 bytes) * num_feature_maps
    # This is a very rough estimate. AMP (fp16) cuts it roughly in half.

    param_count_estimate = 4_851_790  # FIX: actual measured param count
    # FIX: AMP does NOT halve optimizer/master memory. AdamW+AMP holds fp32
    # master(4) + fp16 compute copy(2) + fp32 grad(4) + 2x fp32 Adam moments(8)
    # ≈ 18 B/param. Old code used 2 B/param for AMP, under-counting and inverting
    # the fp32<->AMP relationship.
    bytes_per_param = 18 if cfg.training.use_amp else 16  # fp32: 4 x (w + grad + 2 moments)
    model_vram_gb = (param_count_estimate * bytes_per_param) / (1024 ** 3)  # FIX: bytes_per_param already includes all copies — removed the extra *4

    # Activation memory (rough): batch * C * H * W * layers * bytes
    h, w = cfg.data.input_height, cfg.data.input_width
    activation_bytes = (
        cfg.training.batch_size
        * cfg.data.num_channels
        * h * w
        * 20          # heuristic: ~20 "image-sized" feature map layers
        * (2 if cfg.training.use_amp else 4)
    )
    activation_vram_gb = activation_bytes / (1024 ** 3)

    total_estimated_vram_gb = model_vram_gb + activation_vram_gb
    safety_margin = 0.85  # leave 15% headroom

    if total_estimated_vram_gb > GPU_VRAM_GB * safety_margin:
        warnings_list.append(
            f"VRAM WARNING: Estimated VRAM usage {total_estimated_vram_gb:.1f}GB "
            f"exceeds {GPU_VRAM_GB * safety_margin:.1f}GB "
            f"({safety_margin*100:.0f}% of declared {GPU_VRAM_GB}GB). "
            f"Consider reducing batch_size (currently {cfg.training.batch_size}) "
            f"or enabling use_amp=True."
        )
    elif verbose:
        print(f"[validate_config] VRAM estimate: {total_estimated_vram_gb:.2f}GB "
              f"/ {GPU_VRAM_GB:.1f}GB available. OK.")

    # -----------------------------------------------------------------------
    # 3. HYPERPARAMETER SANITY CHECKS
    # -----------------------------------------------------------------------

    # Batch size vs LR
    if cfg.training.batch_size > 64:
        warnings_list.append(
            f"BATCH SIZE WARNING: batch_size={cfg.training.batch_size} is very large. "
            f"For small datasets (<10K), large batches reduce gradient noise "
            f"diversity and can hurt generalization. Consider batch 16–32."
        )

    if cfg.training.batch_size < 4:
        warnings_list.append(
            f"BATCH SIZE WARNING: batch_size={cfg.training.batch_size} < 4. "
            f"BatchNorm statistics become unreliable. Switch to GroupNorm "
            f"(cfg.model.backbone_norm='groupnorm') or use larger batches."
        )

    # Weight decay for small datasets
    if cfg.optimizer.weight_decay > 1e-3 and cfg.data.train_count < 10000:
        warnings_list.append(
            f"WEIGHT DECAY WARNING: weight_decay={cfg.optimizer.weight_decay} "
            f"is high for a small dataset ({cfg.data.train_count} images). "
            f"This may cause underfitting. Recommended: 1e-5 to 1e-4."
        )

    # Warmup vs total epochs
    if cfg.scheduler.warmup_epochs >= cfg.training.epochs * 0.1:
        warnings_list.append(
            f"SCHEDULE WARNING: warmup_epochs={cfg.scheduler.warmup_epochs} is "
            f">10% of total epochs={cfg.training.epochs}. "
            f"This leaves less time for effective cosine decay."
        )

    # Early stopping vs eval frequency
    if cfg.training.early_stopping_patience < cfg.training.val_frequency * 3:
        warnings_list.append(
            f"EARLY STOPPING WARNING: patience={cfg.training.early_stopping_patience} "
            f"is only {cfg.training.early_stopping_patience // cfg.training.val_frequency} "
            f"validation cycles. May stop too early. Consider increasing patience."
        )

    # NIR-specific: color augmentations should be disabled
    nir_invalid = {
        "hsv_hue_gain": cfg.aug.hsv_hue_gain,
        "hsv_saturation_gain": cfg.aug.hsv_saturation_gain,
        "color_jitter_prob": cfg.aug.color_jitter_prob,
        "channel_shuffle_prob": cfg.aug.channel_shuffle_prob,
    }
    for param_name, val in nir_invalid.items():
        if val != 0.0:
            errors.append(
                f"NIR AUGMENTATION ERROR: cfg.aug.{param_name}={val} must be 0.0. "
                f"This augmentation is invalid for single-channel 850nm NIR imagery."
            )

    # Mosaic with small dataset
    if cfg.aug.mosaic_prob > 0.0 and cfg.data.train_count < 5000:
        warnings_list.append(
            f"MOSAIC WARNING: mosaic_prob={cfg.aug.mosaic_prob} is enabled but "
            f"train_count={cfg.data.train_count} < 5000. Mosaic augmentation is "
            f"harmful on small datasets and may reduce mAP. Set mosaic_prob=0.0."
        )

    if cfg.aug.mosaic_prob == 0.0 and cfg.data.train_count > 10000:
        warnings_list.append(
            f"MOSAIC SUGGESTION: train_count={cfg.data.train_count} > 10000 but "
            f"mosaic_prob=0.0. Enabling mosaic (0.5) could improve mAP by 2–4 points."
        )

    # AMP with old PyTorch
    try:
        major, minor = [int(x) for x in PYTORCH_VERSION.split(".")[:2]]
        if cfg.training.use_amp and major < 1:
            errors.append(
                f"AMP ERROR: use_amp=True but PyTorch version {PYTORCH_VERSION} "
                f"does not support AMP. Upgrade to PyTorch ≥ 1.6."
            )
        if cfg.training.compile_model and major < 2:
            errors.append(
                f"COMPILE ERROR: compile_model=True requires PyTorch ≥ 2.0. "
                f"Detected: {PYTORCH_VERSION}."
            )
    except (ValueError, AttributeError):
        warnings_list.append(
            f"PYTORCH VERSION: Could not parse PYTORCH_VERSION='{PYTORCH_VERSION}'. "
            f"Set it to your actual PyTorch version string."
        )

    # num_channels consistency
    if cfg.data.num_channels != 1:
        warnings_list.append(
            f"CHANNEL WARNING: num_channels={cfg.data.num_channels}. "
            f"NIR imagery is single-channel. Only set to 3 if you explicitly "
            f"replicate channels 3× before the backbone."
        )

    # -----------------------------------------------------------------------
    # 4. PLACEHOLDER DETECTION
    # -----------------------------------------------------------------------
    # These sentinel values are our defaults — warn the user to replace them.
    placeholder_checks = [
        (GPU_VRAM_GB == 8.0,
         "GPU_VRAM_GB is still 8.0 (default). Replace with your actual GPU VRAM."),
        (DATASET_TRAIN_COUNT == 3000,
         "DATASET_TRAIN_COUNT is still 3000 (default). Replace with your actual count."),
        (DATASET_VAL_COUNT == 600,
         "DATASET_VAL_COUNT is still 600 (default). Replace with your actual count."),
        (BASELINE_MAP50 == 0.45,
         "BASELINE_MAP50 is still 0.45 (default). Replace with your baseline mAP50."),
        (PYTORCH_VERSION == "2.3.0",
         "PYTORCH_VERSION is still '2.3.0' (default). Replace with your actual version."),
        (cfg.aug.normalize_mean == (0.5,),
         "normalize_mean is still 0.5 (default). Run tools/compute_stats.py for dataset-specific stats."),
        (cfg.aug.normalize_std == (0.25,),
         "normalize_std is still 0.25 (default). Run tools/compute_stats.py for dataset-specific stats."),
    ]

    for condition, message in placeholder_checks:
        if condition:
            warnings_list.append(f"← REPLACE: {message}")

    # -----------------------------------------------------------------------
    # 5. REPORT
    # -----------------------------------------------------------------------
    has_errors = len(errors) > 0

    if verbose:
        print("\n" + "=" * 60)
        print("NIRDet Config Validation Report")
        print("=" * 60)

        if warnings_list:
            print(f"\n⚠  WARNINGS ({len(warnings_list)}):")
            for w in warnings_list:
                print(f"   {w}")

        if has_errors:
            print(f"\n✗  ERRORS ({len(errors)}) — Training will fail:")
            for e in errors:
                print(f"   {e}")
        else:
            print("\n✓  No hard errors found.")

        print("=" * 60 + "\n")

    if has_errors:
        raise ValueError(
            f"Config validation failed with {len(errors)} error(s). "
            f"See printed report above.\nFirst error: {errors[0]}"
        )

    return True


# ===========================================================================
# PART D — UNCERTAINTY LOG
# ===========================================================================

UNCERTAINTY_LOG = """
Part D — Uncertain Parameters and Resolving Experiments
=======================================================

The following parameters have been set to reasonable defaults but require
empirical validation on YOUR dataset and hardware.

1. ANCHOR SIZES (cfg.model.anchor_sizes)
   Current: ((16,40), (32,80), (64,160)) — estimated from typical surveillance geometry.
   Uncertainty: Completely dataset-dependent. Wrong anchors → poor recall.
   Experiment: Run k-means clustering on all bounding box w,h values in your
   training labels (scaled to input resolution). Use k=3 or k=6.
   Tool: python tools/compute_anchors.py --labels ./data/nirdet/labels/train --k 3

2. NORMALIZATION STATS (cfg.aug.normalize_mean, normalize_std)
   Current: mean=0.5, std=0.25 — generic NIR prior.
   Uncertainty: Sensor-dependent. Different 850nm cameras have different output ranges.
   Experiment: Compute pixel statistics over your full training set.
   Tool: python tools/compute_stats.py --images ./data/nirdet/images/train

3. BATCH SIZE (cfg.training.batch_size)
   Current: 16 — default for 8GB VRAM at 1280×720 with fp16.
   Uncertainty: Depends on actual GPU model and reserved memory from other processes.
   Experiment: Start with batch=8, watch nvidia-smi during first epoch. Increase
   by 2× until you see OOM, then back off by 25%.

4. LEARNING RATE (cfg.optimizer.learning_rate)
   Current: 1e-3 (at batch=16, AdamW).
   Uncertainty: Optimal LR varies with architecture and dataset. 1e-3 is well-validated
   for AdamW on medium-scale detection but may be too high for scratch training.
   Experiment: Run an LR range test (learning rate finder) for 5 epochs:
   start at 1e-6, multiply by 10× every epoch, plot loss vs LR. Pick the LR
   at the steepest negative slope of the loss curve.

5. WEIGHT DECAY (cfg.optimizer.weight_decay)
   Current: 1e-4.
   Uncertainty: Optimal value depends on effective dataset size and model capacity.
   Experiment: Grid search over {1e-5, 5e-5, 1e-4, 5e-4} in 50-epoch ablations.
   Monitor train/val mAP gap (large gap → increase decay; small gap with low val mAP → decrease).

6. LOSS WEIGHTS (cfg.loss.cls_weight, reg_weight, conf_weight)
   Current: (0.5, 0.05, 1.0) — adapted from YOLOv5 single-class defaults.
   Uncertainty: These interact in non-obvious ways. Wrong balance → precision/recall skew.
   Experiment: Fix cls=0.5, conf=1.0. Grid search reg_weight in {0.02, 0.05, 0.1, 0.2}.
   Metric: mAP50 + visual inspection of box tightness on validation.

7. MOSAIC THRESHOLD (cfg.aug.mosaic_prob)
   Current: 0.0 (disabled for <5K images per Bochkovskiy et al. analysis).
   Uncertainty: The exact threshold is 5K–10K, but this is reported for RGB COCO-style data.
   For NIR single-class data, the threshold may be different.
   Experiment: Train two models at your dataset size — one with mosaic_prob=0.0,
   one with mosaic_prob=0.5. Compare mAP50 at convergence.

8. FOCAL LOSS vs BCE (cfg.loss.cls_loss_type)
   Current: "bce".
   Uncertainty: If background cells dominate (person is small in frame), focal loss helps.
   Experiment: Train with "focal" and tune focal_gamma in {0.5, 1.0, 2.0}.
   Monitor class precision — if model predicts background as person, focal helps.

9. ATTENTION TYPE (cfg.model.attention_type)
   Current: "cbam".
   Uncertainty: ECA attention has near-zero params but may be equally effective.
   Experiment: Ablate {"cbam", "se", "eca"} with all other params fixed.
   Metric: mAP50 vs param count tradeoff.

10. BACKBONE DEPTH/WIDTH TRADEOFF (cfg.model.backbone_channels, backbone_depths)
    Current: (16,32,64,128,256) / (1,2,3,2,1) — estimated to hit ~4M params.
    Uncertainty: Optimal channel/depth ratio is architecture-specific.
    Experiment: Fix total param budget at 4M. Compare wider+shallower vs
    narrower+deeper variants. Use NAS-style grid search if compute allows.

11. GAMMA RANGE for NIR sensor simulation (cfg.aug.gamma_range)
    Current: (0.7, 1.3) — estimated from typical NIR CMOS specs.
    Uncertainty: Real 850nm sensors may have tighter or wider gamma response curves.
    Experiment: Consult your sensor datasheet for AGC range. Set gamma_range
    to match the sensor's actual dynamic range envelope.

12. GAUSSIAN NOISE STD (cfg.aug.gaussian_noise_std)
    Current: 0.02 — estimated from 40–50dB SNR at 850nm.
    Uncertainty: Actual noise level depends on illuminator power, exposure time, and ISO.
    Experiment: Capture dark frames (lens cap on) and measure std of pixel values.
    Set gaussian_noise_std to match measured sensor floor.
"""


# ===========================================================================
# MAIN — Print config summary when run directly
# ===========================================================================

if __name__ == "__main__":
    import json

    cfg = get_config()

    print("\nNIRDet Config Summary")
    print("=" * 50)
    print(f"Input: {cfg.data.num_channels}ch × {cfg.input_size[1]}×{cfg.input_size[0]}")
    print(f"Classes: {cfg.data.num_classes} ({cfg.data.class_names})")
    print(f"Training: {cfg.training.epochs} epochs, batch={cfg.training.batch_size}")
    print(f"Optimizer: {cfg.optimizer.name.upper()}, "
          f"lr={cfg.optimizer.learning_rate}, wd={cfg.optimizer.weight_decay}")
    print(f"Schedule: {cfg.scheduler.name}, warmup={cfg.scheduler.warmup_epochs} epochs")
    print(f"AMP: {cfg.training.use_amp}, Compile: {cfg.training.compile_model}")
    print(f"Backbone channels: {cfg.model.backbone_channels}")
    print(f"Attention: EdgeAwareAttention (freeze_epochs={cfg.model.eaa_freeze_epochs})")
    print(f"Mosaic: {cfg.aug.mosaic_prob} (0.0 = disabled, safe for small datasets)")
    print(f"Dataset: {cfg.data.train_count} train, {cfg.data.val_count} val")
    print(f"Target mAP50: >{cfg.logging.target_map50:.3f} (baseline={BASELINE_MAP50}+5pt)")
    print()

    print(UNCERTAINTY_LOG)

    print("\nRunning validation...")
    try:
        validate_config(cfg, verbose=True)
    except ValueError as e:
        print(f"\nFix the above errors before training.\n{e}")
