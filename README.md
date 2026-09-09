# NIRDet — Lightweight NIR Pedestrian Detector Family

A family of single-class pedestrian detectors built from scratch in PyTorch for narrowband 850nm near-infrared surveillance imagery, designed for edge deployment.

**Deployment targets:** Raspberry Pi 5 (NCNN INT8) · STM32N6570-DK (ST Neural-ART NPU)

---

## Models

| Model | Params | mAP50 | Input | Status |
|-------|--------|-------|-------|--------|
| NIRDet | 4.85M | 0.5951 | 640×384 | Reference |
| NIRDet-Lite | 0.72M | — | 384×640 | Training |

NIRDet-Lite is the successor: a ground-up export-clean rewrite targeting INT8 deployment on both Pi 5 via NCNN and STM32N6570-DK via ST Neural-ART NPU. The original NIRDet serves as the reference implementation and performance baseline.

---

## NIRDet (Reference — `nirdet/`)

> mAP50=0.5951 · 89 FPS · 4.85M params · PyTorch 2.6

### Architecture

**Input:** 640×384 px, single-channel NIR greyscale, 850nm narrowband

| Module | Description |
|--------|-------------|
| NIRBackbone | MobileNet-style depthwise separable stages. Stem seeded with Sobel-X, Sobel-Y, Laplacian, and diagonal edge kernels (6 of 32 first-layer filters). Outputs P3 (128ch, stride-8), P4 (256ch, stride-16), P5 (256ch, stride-32). |
| EdgeAwareAttention (EAA) | Spatial attention computed from the raw NIR image via Sobel/Laplacian-initialized learnable filters (N=4). Curriculum freeze: edge_conv frozen epochs 1–5, unfrozen 6+. Residual form: F × (1 + 0.5 × A). Applied independently to P3, P4, P5 before the FPN neck. |
| LightweightFPN | Standard top-down FPN. Lateral 1×1 projections → top-down addition → 3×3 smoothing. 256ch at all three scales. Nearest-neighbour upsampling. |
| PedestrianHead | FCOS-style dense grid assignment. Decoupled cls/reg branches, weight-shared across scales. Strides: (8, 16, 32) → 3,840 + 960 + 240 = 5,040 total cells. Priors: prior_w=0.0461, prior_h=0.1680. |
| Loss | CIoU regression (λ_reg=5.0) + Focal classification (γ=2, α=0.25, λ_cls=1.0) + confidence BCE. Center-cell assignment. |

### Dataset — miniNIRPed

Narrowband 850nm active-illumination NIR pedestrian dataset.

| Split | Images | Annotated |
|-------|--------|-----------|
| Train | 261 | 254 |
| Val | 160 | 157 |

Single class: pedestrian. Severe class imbalance (background >> foreground cells).

### Training Results

| Metric | 640×640 | 416×416 | 640×384 ★ |
|--------|---------|---------|-----------|
| mAP50 | 0.5846 | 0.5401 | **0.5951** |
| mAP@75 | 0.2127 | — | 0.2465 |
| mAR@300 | 0.4084 | — | 0.4215 |
| mAP small | 0.0571 | — | 0.0707 |
| mAP medium | 0.3377 | — | 0.3344 |
| mAP large | 0.3804 | — | 0.4209 |
| Best epoch | 75 | 85 | 85 |
| GPU latency (RTX 4050) | 29.3 ms / 34 FPS | — | 11.3 ms / 89 FPS |
| CPU latency @640×384 | 249.8 ms / 4.0 FPS | 102.8 ms | 110.2 ms / 9.1 FPS |
| CPU latency @192×320 | — | — | 43.7 ms / 22.9 FPS |
| Est. Pi 5 FPS | 0.80–1.33 | 1.3–2.2 | 1.82–3.03 |

★ 640×384 is the final geometry. Eliminates 3,360 dead grid cells vs 640×640 with identical live-cell count (4,880). 40% fewer pixels → 2.6× GPU speedup, 2.27× CPU speedup.

### Evaluation (Phase 9, epoch 85)

| Metric | Value |
|-------|-------|
| mAP50 | 0.5951 (YOLO11n baseline: 0.7350 — pretrained on COCO) |
| mAP@75 | 0.2465 |
| Best F1 threshold | 0.301 (P=0.680, R=0.550, F1=0.608) |

`nirdet/eval_outputs/`: `pr_curve.png` · `per_image_results.csv` · `hard_cases/` · `summary.json`

### Development Phases

| Phase | Status | File | Tests |
|-------|--------|------|-------|
| 0 | ✅ Config | config.py | — |
| 1 | ✅ Backbone | backbone.py | test_backbone.py (4/4) |
| 2 | ✅ Attention | attention.py | test_attention.py (6/6) |
| 3 | ✅ Neck | neck.py | test_neck.py (3/3) |
| 4 | ✅ Head | head.py | test_head.py (5/5) |
| 5 | ✅ Model assembly | model.py | test_model.py |
| 6 | ✅ Loss design | losses.py | Self-tests |
| 7 | ✅ Dataset | dataset.py | test_dataset.py (4/4) |
| 8 | ✅ Training | train.py | — |
| 9 | ✅ Evaluation | evaluate.py | — |

### Key Engineering Decisions

**Dead-cell elimination (640×384):** removes 3,360 inert grid cells while keeping all 4,880 live detection cells.

**Head w-scale fix:** `w = exp(t_w) × (W × stride)`, `h = exp(t_h) × (H × stride)`. Previously both used H×stride — widths were 0.60× too narrow. Drove mAP@75: 0.2127 → 0.2465.

**Prior measurement:** median normalised w/h measured through the actual 640×384 augmentation pipeline on 948 training labels.

**RandomResizedCrop ratio=(1.2, 2.2):** default (0.75, 1.33) caused 3.9% silent box dropout.

**EAA curriculum freeze:** edge_conv frozen epochs 1–5 to let backbone stabilise.

**Overfit gate:** mAP50 > 0.90 on 10 images required before full training. Achieved 0.9560 @ epoch 100.

### Quick Start

```bash
# Overfit sanity check (should exceed mAP50 = 0.90 by epoch 100)
python nirdet/train.py --overfit-test

# Full training
python nirdet/train.py

# Evaluation
python nirdet/evaluate.py --checkpoint nirdet/checkpoints/best.pth
```

---

## NIRDet-Lite (Successor — `nirdet_lite/`)

> 0.72M params · INT8-ready · Pi 5 (NCNN) · STM32N6570-DK (Neural-ART)

A ground-up rewrite of NIRDet designed for INT8 export from the start. Every architectural and training decision is constrained by what the two deployment targets can actually execute in hardware.

### What changed from NIRDet

| Concern | NIRDet | NIRDet-Lite |
|---------|--------|-------------|
| Params | 4.85M | 0.72M |
| Activations | ReLU / SiLU mix | ReLU6 throughout (HW-mapped on both targets) |
| Normalisation | GroupNorm in head | BatchNorm everywhere (folds into conv at export) |
| Padding | reflect | zeros (reflect Pad is SW-only on Neural-ART) |
| Head branch convs | 2× dense 3×3 @ 256ch | 1× DWS @ 64ch |
| Level scale | `index_put_` → ScatterND | Pure Mul+Add with selector buffers (ONNX-clean) |
| Neck upsampling | `size=` → Shape/Gather debris | `scale_factor=` → constant Resize |
| Post-processing | In graph | Host-only: `live_nirdet.py` (Pi 5) · `nirdet_pp.c` (STM32) |
| Export | — | `export_onnx.py` → opset 12, 9 op types, strict audit |
| Quantisation | — | `quantize_qdq.py` → QDQ ONNX for stedgeai |
| Dataset profiles | — | `dataset_profiles.py` — fingerprint-verified label/image state |

### Architecture

**Input:** 384×640 px · 1ch NIR · `[0, 1]` float32

Backbone → EAA → LightweightFPN → PedestrianHead, identical structure to NIRDet but with the hardware constraints above applied throughout. Three detection levels: stride 8 (48×80), stride 16 (24×40), stride 32 (12×20) → 5,040 total cells.

**Exported graph** (9 op types post-simplify): `Abs · Add · AveragePool · Clip · Concat · Conv · Mul · Resize · Sigmoid` — no GroupNorm, no ScatterND, no Gather, no dynamic shapes.

### Decode contract

Identical in all four implementations — `config.py`, `head.py`, `live_nirdet.py`, `nirdet_pp.c`:

```
cx = (2.0 × sigmoid(t_cx) − 0.5 + col) × stride
cy = (2.0 × sigmoid(t_cy) − 0.5 + row) × stride
w  = exp(clamp(t_w, −6, 1)) × img_w
h  = exp(clamp(t_h, −6, 1)) × img_h
```

Verified by `test_decode_contract.py` — all four copies cross-checked against each other on every run.

### Export pipeline

```bash
# 1. Train
python nirdet_lite/train.py --profile nirdet_lite/datasets/miniNIRPed_261.yaml

# 2. Export to ONNX (opset 12, simplified, strict audit)
python nirdet_lite/export_onnx.py --checkpoint nirdet_lite/checkpoints/best.pth

# 3. INT8 quantisation (QDQ — for stedgeai / STM32N6570-DK)
python nirdet_lite/quantize_qdq.py --onnx nirdet-sim.onnx \
    --profile nirdet_lite/datasets/miniNIRPed_261.yaml

# 4. Pi 5 — NCNN INT8
onnx2ncnn nirdet-sim.onnx nirdet.param nirdet.bin
# (see export_onnx.py output for full ncnn2int8 pipeline)

# 5. Live inference (Pi 5)
python nirdet_lite/live_nirdet.py --model nirdet-int8.param
```

### Test suite

```bash
python nirdet_lite/test_head.py
python nirdet_lite/test_model.py
python nirdet_lite/test_losses.py
python nirdet_lite/test_decode_contract.py
python nirdet_lite/test_dataset_profiles.py
```

---

## Requirements

```
Python 3.11
PyTorch 2.6.0+cu124
CUDA 12.4 (CPU-only works, no code changes)
OpenCV, numpy, tqdm, pycocotools, onnx, onnxsim, onnxruntime
```

## Dataset

miniNIRPed — 850nm active-illumination NIR pedestrian dataset, not included in this repo. Place under `data/raw/miniNIRPed/` (gitignored). Profile: `nirdet_lite/datasets/miniNIRPed_261.yaml`.
