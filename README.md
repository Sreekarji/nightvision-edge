# NIRDet-Forge — NIR pedestrian detection for edge devices

![Model](https://img.shields.io/badge/model-NIRDet--Forge-1e40af)
![Input](https://img.shields.io/badge/input-850%20nm%20single--channel%20NIR-596579)
![Canvas](https://img.shields.io/badge/canvas-512%C3%97288-2878b5)
![Targets](https://img.shields.io/badge/targets-Pi%205%20%2B%20STM32N6570--DK-008f68)
![License](https://img.shields.io/badge/license-MIT%20(project%20code)%20%2B%20third--party%20terms-blue)

NIRDet began as a single-class near-infrared pedestrian detector. NIRDet-Lite reworked the model around export-friendly INT8 deployment. **NIRDet-Forge carries that lineage forward as the primary implementation in this repository**, with a Raspberry Pi 5/NCNN INT8 path and an STM32N6570-DK/ST Neural-ART project. The reported Forge checkpoint has about 630K trainable parameters and test mAP50 `0.7836`. Forge defines the shared model geometry and decode contract; the checked-in STM32 CubeIDE application is still an integration skeleton, not a complete camera-to-detection demo.

## Project history

- **NIRDet (`nirdet/`)** — the original reference implementation and its historical evaluation output.
- **NIRDet-Lite (`nirdet_lite/`)** — an INT8-oriented predecessor with its own training, export, runtime and tests. The existing project notes also describe Lite as targeting both edge platforms; Forge is presented here as the current, consolidated pipeline, not as the first dual-target attempt.
- **NIRDet-Forge (`nirdet_forge/`)** — the current focus: a 512×288 model, profile-derived dataset values, shared preprocessing, raw multi-level outputs and separate NCNN and Neural-ART deployment paths.

## Repository layout

| Path | Contents |
|---|---|
| `nirdet/` | Original reference model, tests, prior-measurement utility and historical evaluation snapshot. |
| `nirdet_lite/` | Predecessor model, profile tooling, export/runtime code and tests. |
| `nirdet_forge/` | Primary implementation: model, dataset pipeline, training, evaluation, ONNX/NCNN export, quantization, live runtime and C postprocessor. |
| `nirdet_forge/stedgeai_npu_out/` | ST Neural-ART output. The `network*` set comes from an `analyze` invocation; the `nirdet*` set comes from `generate`. They are separate outputs, not interchangeable duplicates. |
| `stm32_cubeide/` | STM32N6570-DK CubeIDE root project with separate `Appli/` and `FSBL/` projects. The application combines generated Neural-ART code with the C postprocessor; see the deployment notes below. |
| `tools/` | NIRPed split and COCO-to-YOLO preparation tools. |
| `yolo_nirped/` | YOLO baseline dataset configuration and frozen stage-two dependencies. |

The NIRPed images/labels and training checkpoints are not tracked. The curated Forge report `nirdet_forge/eval_outputs/summary.json` is tracked on upstream `main`; other local files under `datasets/`, `checkpoints/` and `eval_outputs/` may be ignored. The checkpoint and per-run files described later are local artifacts, not part of the tracked repository.

## NIRDet-Forge architecture

Forge is a single-class (`person`) detector. The model input is `(B, 1, 288, 512)`. Its three detection levels are P3/P4/P5 at strides 8/16/32, with spatial maps 36×64, 18×32 and 9×16 respectively (3,024 cells in total).

| Component | Design |
|---|---|
| Backbone | Dense convolutions, BatchNorm, zero padding and ReLU6. The backbone avoids depthwise-separable convolutions; the source rationale is accelerator/Pi execution efficiency. |
| Edge-Aware Attention | Four edge filters; defaults freeze the edge filters for the first five epochs. The default edge path uses stride 2 and a pool factor of 4. Its calibrated gate is `feat * (0.5 + attn)`, clamped to `[0, 6]`. |
| Neck | Top-down FPN with optional one-level bottom-up PAN. Default output widths are 48 channels at P3 and 64 at P4/P5. |
| Head | Each detection level emits separate class, offset and size blobs: `cls8/off8/size8`, `cls16/off16/size16`, `cls32/off32/size32`. The three levels carry 1, 2 and 2 channels respectively. |
| Loss and assignment | Quality Focal Loss plus pixel-space CIoU, with Task-Aligned Assignment. Defaults include `alpha=0.5`, `beta=6`, `topk=10`, nearest-centre fallback for boxes with no centre inside, and smallest-area-wins collision resolution. |
| Dataset-specific values | Box priors and the deployment score threshold are profile/evaluation values, not Forge-wide defaults. Missing priors are fatal; there is no default deployment score threshold. |

`eaa_residual_scale=2.0` is retained for configuration compatibility: a non-`None` value selects the residual-gate path; it is **not** a 2× multiplier. ONNX export uses opset 13 and runs static-shape and operator checks. The graph emits raw head blobs; box decoding and NMS stay outside the graph.

## Results

### STM32N6 Neural-ART compiler reports

The checked-in `nirdet_generate_report.txt` is from ST Edge AI Core `4.0.1-20581`, Neural-ART compiler `1.1.3-275`, and a `generate --target stm32n6 --st-neural-art` invocation. These are compiler mapping and memory estimates, **not** measured board latency or accuracy.

| Generated `nirdet` item | Value |
|---|---:|
| Compiler schedule | 81 reported epochs: 72 HW, 5 SW, 4 Hybrid |
| Weights | 789,297 B |
| Activations | 2,383,056 B |
| Total FLASH | 956,028 B |
| Total RAM | 2,384,869 B |

The separate `network_analyze_report.txt` is from an `analyze --input-data-type uint8` run. It reports 80 epochs (71 HW, 5 SW, 4 Hybrid), 955,519 B FLASH and the same 2,384,869 B RAM. Its `network_c_info.json` input is unsigned int8 with zero point 0; the generated `nirdet_c_info.json` input is signed int8 with zero point −128. Use the `nirdet*` generate outputs for the CubeIDE application contract; do not mix the analyze input format or report totals into that contract.

The compiler figures above are mapping and memory estimates, not accuracy or on-device throughput claims.

### NIRDet-Forge evaluation

The curated report `nirdet_forge/eval_outputs/summary.json` records test metrics for run `run_20260925_153959` on 2,061 test images with 6,140 ground-truth boxes. The test mAP50 is `0.7835813164710999`; the interval endpoints are `0.7680213809013366` and `0.7949684143066407`. The JSON labels this a “95% bootstrap confidence interval,” but also states that the coverage level is inferred rather than stated; the source records 97 bootstrap resamples. The best validation mAP50 is `0.9037` at displayed epoch 78, while the final validation mAP50 is `0.9021` at epoch 100. The report warns that validation and test results come from separate artefacts and are not directly comparable without care.

| Metric | Value |
|---|---:|
| Test mAP50 | `0.7835813164710999` |
| Test mAP50 interval | `[0.7680213809013366, 0.7949684143066407]` (95% label; coverage inferred) |
| Test mAP75 | `0.6226414442062378` |
| AP-small / AP-medium / AP-large | `0.4676576554775238` / `0.589597761631012` / `0.17258669435977936` |
| Best / final validation mAP50 | `0.9037` at displayed epoch 78 / `0.9021` at epoch 100 |
| Deploy score threshold | `0.43` |
| F1 / precision / recall at the best operating point | `0.7744449478930675` / `0.8729315628192033` / `0.6959283387622149` |
| TP / FP / FN at threshold `0.43` | `4273` / `622` / `1867` |

The report's `map50_95` field is the sentinel `-1.0`, not an mAP50–95 result. The threshold `0.43` is measured in the dataset profile and evaluation report; it is not a project-wide default.

### Local checkpoint snapshot (untracked run artifacts)

A separate read-only audit of the local run artifacts reports 630,459 trainable parameters (rounded to 630K). The model state dictionary has 635,090 total tensor elements; this includes 4,592 BatchNorm running-statistic values and 39 `num_batches_tracked` counters. The checkpoint stores `epoch=77` using zero-based indexing, which is displayed as epoch 78 in the training log; its `best_map50` is `0.9037208557128906`. The run trained the configured 100 epochs with batch size 8, AdamW, peak learning rate `0.0008`, BF16 autocast, seed 42 and EMA decay `0.995`. Early stopping was disabled (`es_enabled=False`), so the patience counter reaching `22/20` did not stop the run. The measured deploy threshold `0.43` is in the profile/report; the checkpoint config itself has `deploy_score_thresh=None`.

These checkpoint, history and log details are local run artifacts, not tracked model files in the repository. The compiler figures above should not be read as an accuracy or throughput claim.

### Predecessor reference (not Forge)

The tracked `nirdet/eval_outputs/summary.json` records an NIRDet epoch-50 snapshot with 160 validation images and mAP50 `0.6161`. The previous README also gives an epoch-85 NIRDet result of `0.5951`; those are different historical snapshots and are not combined here.

## Quickstart

Forge expects a YOLO-format dataset with train and validation splits; evaluation defaults to the report split `test`. Supported layouts include `images/train`, `images/val` and `images/test` alongside the matching directories under `labels/`, or `train/images`, `val/images` and `test/images` alongside the matching label directories. The dataset is not included in this repository.

The profile command writes a YAML file under `datasets/`, using the dataset-root directory name for its generated filename. The following Bash flow derives that name and prompts for paths instead of assuming a local dataset or checkpoint path:

```bash
cd nirdet_forge
read -r -p 'YOLO-format dataset root: ' DATA_ROOT
python dataset_profiles.py --root "$DATA_ROOT"
PROFILE="datasets/$(basename "$DATA_ROOT").yaml"

python train.py --profile "$PROFILE"
read -r -p 'Best checkpoint path printed by training: ' CHECKPOINT
python evaluate.py --checkpoint "$CHECKPOINT" --profile "$PROFILE"
python export_onnx.py --checkpoint "$CHECKPOINT" --profile "$PROFILE"
```

Evaluation measures the best-F1 deployment threshold on the report split and writes it back to the profile. If the dataset has no `test` split, select an available split explicitly, for example with `--split val`. Always pass the trained checkpoint to export: exporting ONNX without `--checkpoint` creates randomly initialized weights and is only useful for graph checks.

Training and model tests require PyTorch. Other stages use NumPy/OpenCV, PyYAML, TorchMetrics/pycocotools, ONNX/ONNX Runtime or NCNN as indicated by the invoked script. Forge does not provide a pinned dependency lockfile.

## Deployment: Raspberry Pi 5 and STM32N6570-DK

### Raspberry Pi 5 — NCNN INT8

From `nirdet_forge/`, convert the simplified FP32 graph. The exporter calibrates NCNN using the profile’s preprocessing and calibration-image selection, then writes the model files and deployment-contract sidecar:

```bash
python export_ncnn.py --onnx nirdet-sim.onnx --profile "$PROFILE"
```

Copy `nirdet-int8.param`, `nirdet-int8.bin`, `nirdet-int8.contract.json` and the profile to the Pi. If the profile uses a flat-field map, copy the same map and set its path on the Pi; its contents are part of the contract. Start camera inference with:

```bash
python live_nirdet.py \
  --param nirdet-int8.param \
  --bin nirdet-int8.bin \
  --contract nirdet-int8.contract.json \
  --profile "$PROFILE"
```

The runtime is Torch-free and accepts Picamera2 input or a video source. The NCNN export script invokes `onnx2ncnn`, `ncnnoptimize`, `ncnn2table` and `ncnn2int8` as external tools. On Linux/Pi, make those tools available on `PATH`; the Windows/WSL fallback in the script contains a machine-specific NCNN build path.

### STM32N6570-DK — ST Neural-ART and CubeIDE

#### Export and accuracy gate

Generate the QDQ graph and run the paired INT8-versus-FP32 accuracy gate before using the ST compiler:

```bash
python quantize_qdq.py --profile "$PROFILE"
python evaluate_onnx.py \
  --int8 nirdet-int8-qdq.onnx \
  --fp32-ref nirdet-sim.onnx \
  --profile "$PROFILE"
```

The gate allows at most a `0.02` mAP50 drop from this model’s FP32 graph. The checked-in reports record separate `analyze` and `generate` runs:

```bash
stedgeai analyze \
  --target stm32n6 \
  --model nirdet-int8-qdq.onnx \
  --st-neural-art \
  --input-data-type uint8 \
  --output stedgeai_npu_out

stedgeai generate \
  --target stm32n6 \
  --model nirdet-int8-qdq.onnx \
  --st-neural-art \
  --output stedgeai_npu_out \
  --name nirdet
```

Use the **`nirdet*` generate outputs** for the STM32 application contract. The `network*` analyze outputs are a separate compiler run: their input is U8 with zero point 0, while the generated `nirdet` application input is S8 with zero point −128. See `nirdet_forge/stedgeai_npu_out/network_analyze_report.txt`, `network_c_info.json`, `nirdet_generate_report.txt` and `nirdet_c_info.json`.

#### CubeIDE project

The GitHub project tree contains `stm32_cubeide/Appli/` and `stm32_cubeide/FSBL/` for the **STM32N6570-DK / STM32N657X0HxQ** target. Its `stm32_cubeide/README.md` specifies STM32CubeIDE **v2.2.0**, ST Edge AI Core **v4.0.1** installed at `C:\ST\4.0`, and the STM32N6570-DK applications tree at `C:\ST\4.0\Projects\STM32N6570-DK`. The Appli project uses absolute include paths to those locations; update them in `Appli/.cproject` if your installation differs. It has Debug and Release configurations and links `NetworkRuntime1201_CM55_GCC.a` in the Debug settings. The committed Release settings do not explicitly repeat the same ST AI include and library paths, so validate that configuration in CubeIDE before relying on it. The ARM GCC version is not pinned in the repository.

The CubeIDE project enables the compiler’s CMSE/secure-mode option, but that alone does not establish a working TrustZone handoff. The project metadata is mixed, and the checked-in FSBL source still contains a placeholder rather than a completed boot transfer; see the status notes below.

#### Generated input, outputs and decoder settings

The generated input contract is one **S8** tensor `[1, 1, 288, 512]`, 147,456 bytes, 32-byte aligned, with scale `0.00392156886` and zero point `−128`. The intended firmware input preparation is described in `Appli/Src/main.c` as CLAHE, letterboxing and conversion to that quantization; it is not implemented there.

All nine outputs are signed int8 and 32-byte aligned. The names, shapes, sizes and quantization values used by the current STM32 postprocessor are:

| Level | Classification blob | Offset blob | Size blob |
|---|---|---|---|
| Stride 8, grid 36×64 | `Quantize_221_out_0`, `[1,1,36,64]`, 2,304 B; scale `0.0550706312`, zp `81` | `Quantize_229_out_0`, `[1,2,36,64]`, 4,608 B; scale `0.1828420013`, zp `2` | `Quantize_227_out_0`, `[1,2,36,64]`, 4,608 B; scale `0.0178692229`, zp `119` |
| Stride 16, grid 18×32 | `Quantize_200_out_0`, `[1,1,18,32]`, 576 B; scale `0.0441131219`, zp `75` | `Quantize_208_out_0`, `[1,2,18,32]`, 1,152 B; scale `0.1466677934`, zp `−4` | `Quantize_206_out_0`, `[1,2,18,32]`, 1,152 B; scale `0.0160245188`, zp `124` |
| Stride 32, grid 9×16 | `Quantize_152_out_0`, `[1,1,9,16]`, 144 B; scale `0.0266222004`, zp `127` | `Quantize_160_out_0`, `[1,2,9,16]`, 288 B; scale `0.0328494124`, zp `43` | `Quantize_158_out_0`, `[1,2,9,16]`, 288 B; scale `0.0154807130`, zp `127` |

The C decoder contract is stamped `ed9bd4eb8e4f1922`; it defines one class, three levels and a 300-detection maximum. The CubeIDE `main.c` currently hard-codes canvas `512×288`, score threshold `0.430`, NMS IoU threshold `0.45` and `max_det=300`; it does not load the score threshold from the dataset profile. Keep these values synchronized with the model/profile used for evaluation. The C postprocessor expects contiguous CHW blobs, with offset channels `(t_cx,t_cy)` and size channels `(t_w,t_h)`.

After changing decode constants, regenerate and check the C contract from `nirdet_forge/`:

```bash
python gen_contract_c.py --check
```

#### Linker regions and NPU memory

`Appli/STM32N657X0HXQ_LRUN.ld` defines `ROM` at `0x34000400` (511K) and `RAM` at `0x34080000` (1536K). The FSBL linker script defines `ROM` at `0x34180400` (255K) and `RAM` at `0x341C0000` (256K). Neural-ART separately maps weights to OctoFlash at `0x71000000` and activations across CPU RAM2 and NPU RAM3–RAM6. The generate report shows about 770.798 kB of weights in OctoFlash and about 2.273 MB of activations; HyperRAM is unused by this compile. The generated `nirdet.c` references the OctoFlash weight base, so the weight image must be provisioned in the expected external-memory location; the CubeIDE application/FSBL sources do not show that programming step.

#### Current firmware status and integration gaps

These describe the source as checked in; this README update does not change the firmware:

- `Appli/Src/main.c` initializes the STAI network, binds buffers, runs synchronous inference and calls `nirdet_postprocess`. It zeroes the input buffer, leaves camera-frame preprocessing as a TODO, and discards the detection count; GPIO/UART/display output is also a TODO. The HAL configuration comments out the DCMI, DCMIPP and CSI modules.
- **Check the STAI context allocation before execution.** `main.c` declares a single `stai_network` object (its comment says the type is `unsigned char`), while the generated header exposes `STAI_NIRDET_CONTEXT_SIZE` and `STAI_NIRDET_CONTEXT_ALIGNMENT`; init casts the pointer to `_stai_aton_context`. Confirm the storage size and alignment before execution.
- **Check I/O buffer ownership before execution.** The generated `nirdet.h` says there are no user-allocated I/Os and the generated buffer metadata marks its input/output buffers as not user-allocated at fixed NPU RAM addresses. The CubeIDE `main.c` passes separate static arrays to the STAI setters, whose generated implementation validates pointers for non-user-allocated buffers. The return values from the init/setter/run calls are not checked. This is a source-level interface mismatch to resolve or verify in CubeIDE/on hardware.
- `FSBL/Src/main.c` warns that secure/non-secure regions are not initialized, comments that the non-secure stack and jump must be added, then loops forever. It does not currently launch the application.
- The top-level [LICENSE](LICENSE) applies the MIT license to project-owned code only. ST-generated material remains under its original terms, including SLA0104 (Rev1/June 2024) for the Neural-ART output; see the repository license notices and `nirdet_forge/stedgeai_npu_out/LICENSE.txt`.

## Decode contract

Each level emits class logits, two offset logits and two size logits. Decoding is in canvas pixels, with `row` and `col` denoting the grid cell and `stride` the level stride:

```text
cx = (2 * sigmoid(t_cx) - 0.5 + col) * stride
cy = (2 * sigmoid(t_cy) - 0.5 + row) * stride
w  = exp(clamp(t_w, -6, 1)) * img_w
h  = exp(clamp(t_h, -6, 1)) * img_h
```

The canonical membership order is:

1. Decode to centre/size and convert to `xyxy`.
2. Clamp coordinates to the canvas.
3. Drop boxes whose clamped width or height is at most 1 pixel.
4. Apply NMS (default IoU threshold `0.45`).
5. Limit final detections to `max_det` (default `300`). The pre-NMS candidate cap is also `300`.

The score threshold is measured on the report split and stored in the dataset profile. Forge does not silently substitute a default; live inference requires that measured profile value or an explicit `--score-thresh`. The contract sidecar hashes geometry, decode constants and preprocessing settings.

The STM32 C postprocessor consumes contiguous CHW int8 blobs, each with its own scale and zero-point. NCNN `Mat` uses a padded channel step, so its data must be repacked before calling the C helper. The C self-test’s `0.30f` score threshold is synthetic test data, not a deployment threshold.

## Test suites

Forge’s Python checks are standalone scripts, run from `nirdet_forge/`:

```bash
python test_dataset.py
python test_dataset_profiles.py
python test_decode_contract.py
python test_losses.py
python test_model.py
python gen_contract_c.py --check
cc -DNIRDET_PP_SELFTEST -O2 nirdet_pp.c -lm -o /tmp/nirdet_pp_selftest
/tmp/nirdet_pp_selftest
```

The dataset and profile suites include synthetic/temp-directory checks; the decode suite checks consistency across the Python and generated-C contract. The ONNX INT8 evaluation gate is described in the deployment section.

## License

The repository-level [LICENSE](LICENSE) is the MIT License for this project’s own code. It explicitly excludes ST-generated or ST-distributed material, which remains subject to its original terms. In particular, the Neural-ART generated output is licensed by STMicroelectronics under SLA0104 (Rev1/June 2024); refer to the license notice shipped with that output and to the relevant ST source-file headers. Do not interpret the MIT notice as relicensing ST material, datasets or untracked checkpoints.
