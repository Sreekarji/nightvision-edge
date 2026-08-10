# Night Vision Surveillance System — Edge Intelligence

VCE ECE Department Project | Batch 2028

## Overview
Portable, offline night vision surveillance system using:
- Raspberry Pi 5 + NoIR Camera Module (850nm)
- 850nm IR LED Array
- YOLO11n pedestrian detection — fully on-device, no internet required

## Current Status — Stage 2 Complete

### Model
- Architecture: YOLO11n (2.6M parameters, 5.4MB)
- Dataset: miniNIRPed (850nm NIR, 261 training images)
- Training: 100 epochs, 320x320, RTX 4050

### Results
| Metric | Value |
|---|---|
| mAP50 | 0.735 |
| mAP50-95 | 0.458 |
| Precision | 0.783 |
| Recall | 0.651 |
| Inference (GPU) | ~12ms/image |

### Why miniNIRPed
miniNIRPed uses a narrowband NIR camera with active NIR illumination —
the same 850nm physics as our NoIR + LED setup. It is the only public
dataset that matches our exact camera modality.

## Pipeline
COCO pretrained weights
    → Stage 2: fine-tune on miniNIRPed (done)
    → Stage 3: fine-tune on custom Pi 5 captures (pending)
    → Export to NCNN INT8 for Pi 5 deployment

## Deployment Target
- Board: Raspberry Pi 5 (Cortex-A76, 2.4GHz)
- Runtime: NCNN INT8
- Target: 30+ FPS at 320x320

## Team
Pranay, Chandana, Sreekar
