# NIRDet — Lightweight NIR Pedestrian Detector

`mAP50=0.5951` `89 FPS` `4.85M params` `PyTorch 2.6`

A single-class pedestrian detection model built from scratch in PyTorch for narrowband 850nm near-infrared surveillance imagery, designed for Raspberry Pi 5 edge deployment.

Existing detectors (YOLO11n baseline: mAP50=0.7350) rely on large-scale RGB pretraining. NIRDet is trained from scratch on NIR-specific data with an architecture tailored to narrow-spectrum pedestrian priors and the compute constraints of edge deployment.

---

## Architecture

```
Input: 640×384 px, single-channel NIR greyscale, 850nm narrowband
Params: 4,851,754 trainable
AMP:   bf16
```

### Components

| Module | Description |
|--------|-------------|
| **NIRBackbone** | MobileNet-style depthwise separable stages. Stem seeded with Sobel-X, Sobel-Y, Laplacian, and diagonal edge kernels (6 of 32 first-layer filters). Outputs P3 (128ch, stride-8), P4 (256ch, stride-16), P5 (256ch, stride-32). |
| **EdgeAwareAttention (EAA)** | Spatial attention computed from the raw NIR image via Sobel/Laplacian-initialized learnable filters (N=4). Curriculum freeze: edge_conv frozen epochs 1–5, unfrozen 6+. Residual form: `F × (1 + 0.5 × A)` — preserves body context in flat (low-edge) regions. Applied independently to P3, P4, P5 before the FPN neck. |
| **LightweightFPN** | Standard top-down FPN (Lin et al. CVPR 2017). Lateral 1×1 projections → top-down addition → 3×3 smoothing. Unified output: 256ch at all three scales (N3, N4, N5). Nearest-neighbour upsampling (no checkerboard artefacts). |
| **PedestrianHead** | FCOS-style dense grid assignment. Decoupled cls/reg branches (YOLOX-style), weight-shared across scales. Decodes: `cx = (sigmoid(t_cx) + col) × stride`, `cy = (sigmoid(t_cy) + row) × stride`, `w = exp(t_w) × W×stride`, `h = exp(t_h) × H×stride`. Strides: (8, 16, 32) → 3,840 + 960 + 240 = 5,040 total cells. Priors measured from 948 training labels through the 640×384 pipeline (median): prior_w=0.0461, prior_h=0.1680. |
| **Loss** | CIoU regression (λ_reg=5.0) + Focal classification (γ=2, α=0.25, λ_cls=1.0) + confidence BCE (monitoring only). Center-cell label assignment (1 positive per GT per scale). Grid sizes: ((48,80), (24,40), (12,20)). |

---

## Dataset — miniNIRPed

Narrowband 850nm active-illumination NIR pedestrian dataset.

| Split | Images | Annotated |
|-------|--------|-----------|
| Train | 261 | 254 |
| Val   | 160 | 157 |

Single class: pedestrian (class 0). Severe class imbalance (background >> foreground cells).

---

## Training Results

| Metric                  | 640×640 | 416×416 | **640×384 ★** |
|-------------------------|---------|---------|--------------|
| mAP50                   | 0.5846  | 0.5401  | **0.5951**   |
| mAP@75                  | 0.2127  | —       | **0.2465**   |
| mAR@300                 | 0.4084  | —       | **0.4215**   |
| mAP small               | 0.0571  | —       | 0.0707       |
| mAP medium              | 0.3377  | —       | 0.3344       |
| mAP large               | 0.3804  | —       | **0.4209**   |
| Best epoch              | 75      | 85      | 85           |
| GPU latency (RTX 4050)  | 29.3 ms / 34 FPS | — | **11.3 ms / 89 FPS** |
| CPU latency @640×384    | 249.8 ms / 4.0 FPS | 102.8 ms | **110.2 ms / 9.1 FPS** |
| CPU latency @192×320    | —       | —       | **43.7 ms / 22.9 FPS** |
| Est. Pi 5 FPS           | 0.80–1.33 | 1.3–2.2 | **1.82–3.03** |

**★ 640×384 is the current best and final geometry.**

Why 640×384 over 640×640:
- 640×640 had 3,360 dead grid cells (rows below 384 carry no pedestrian labels)
- 640×384 eliminates all dead cells with identical live-cell count (4,880)
- 40% fewer pixels → 2.6× GPU speedup, 2.27× CPU speedup

---

## Evaluation (Phase 9, epoch 85 checkpoint)

| Metric | Value |
|--------|-------|
| mAP50 | 0.5951 (YOLO11n baseline: 0.7350 — pretrained on COCO) |
| mAP@75 | 0.2465 |
| Best F1 threshold | 0.301 (P=0.680, R=0.550, F1=0.608) |
| Hard cases | 4 images (truncated-edge / headlight-obscured — irreducible with centre-cell assignment) |

`eval_outputs/`:
- `pr_curve.png` — precision-recall curve
- `per_image_results.csv` — per-image breakdown (160 rows)
- `hard_cases/` — 4 visualised failure images
- `summary.json` — machine-readable full summary

---

## Development Phases

| Phase | Status | File | Tests |
|-------|--------|------|-------|
| 0  | ✅ Config | `config.py` | — |
| 1  | ✅ Backbone | `backbone.py` | `test_backbone.py` (4/4) |
| 2  | ✅ Attention | `attention.py` | `test_attention.py` (6/6) |
| 3  | ✅ Neck | `neck.py` | `test_neck.py` (3/3) |
| 4  | ✅ Head | `head.py` | `test_head.py` (5/5) |
| 5  | ✅ Model assembly | `model.py` | `test_model.py` |
| 6  | ✅ Loss design | `losses.py` (CIoU + Focal + BCE) | Self-tests |
| 7  | ✅ Dataset | `dataset.py` | `test_dataset.py` (4/4) |
| 8  | ✅ Training | `train.py` (overfit + full, auto-evaluate on finish) | — |
| 9  | ✅ Evaluation | `evaluate.py` (7-step COCO pipeline) | — |

---

## Key Engineering Decisions

1. **Dead-cell elimination (640×384):** removes 3,360 inert grid cells while keeping all 4,880 live detection cells — verified by assign/decode trace.

2. **Head w-scale fix:** `w = exp(t_w) × (W × stride)`, `h = exp(t_h) × (H × stride)`. Previously both used H × stride — widths were 0.60× too narrow at 640×384. Directly drove mAP@75: 0.2127 → 0.2465 (16% improvement in box tightness).

3. **Prior measurement:** median normalized w/h measured through the actual 640×384 augmentation pipeline on 948 training labels (not mean, not raw).

4. **RandomResizedCrop ratio=(1.2, 2.2):** default (0.75, 1.33) caused 3.9% silent box dropout by squashing pedestrian aspect ratios toward square.

5. **EAA curriculum freeze:** edge_conv frozen epochs 1–5 to let backbone stabilise before the attention module adapts.

6. **Overfit gate:** mAP50 > 0.90 on 10 images within 300 epochs required before full training. Achieved 0.9560 @ epoch 100, 1.0000 @ epoch 150.

---

## Requirements

- Python 3.11.13
- PyTorch 2.6.0+cu124
- CUDA 12.4 (CPU-only also works, no code changes needed)
- OpenCV, numpy, tqdm, pycocotools

---

## Quick Start

```bash
# Run overfit sanity check (should exceed mAP50 = 0.90 by epoch 100)
python nirdet/train.py --overfit-test

# Full training — auto-runs evaluate.py at completion
python nirdet/train.py

# Standalone evaluation on any checkpoint
python nirdet/evaluate.py --checkpoint nirdet/checkpoints/best.pth
```

---

## Next Steps

Pending on-device access:
1. Export: best.pth → ONNX → NCNN
2. INT8 quantization (miniNIRPed calibration set)
3. On-device latency measurement (Pi 5 ARM Cortex-A76)
4. Target: ≥ 5 FPS — motivated by 192×320 CPU result (43.7 ms / 22.9 FPS on x86 laptop, suggesting Pi 5 + NCNN + INT8 is in range)