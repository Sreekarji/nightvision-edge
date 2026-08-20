"""
test_model.py — NIRDet Phase 5 unit tests
==========================================
Tests 1-3 use synthetic data. Test 4 (overfit) requires 4 real images +
YOLO-format .txt labels. Instructions for Test 4 are in the docstring below.

Usage:
    # Tests 1-3 (no real data required):
    python test_model.py

    # Test 4 (overfit) — provide 4 real images + labels:
    python test_model.py --overfit \
        --images img1.png img2.png img3.png img4.png \
        --labels img1.txt img2.txt img3.txt img4.txt \
        --input-size 640
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NIRDet, build_nirdet


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _green(s): return f"\033[92m{s}\033[0m"
def _red(s):   return f"\033[91m{s}\033[0m"
def _bold(s):  return f"\033[1m{s}\033[0m"

def _pass(name): print(f"  {_green('PASS')} {name}")
def _fail(name, reason): print(f"  {_red('FAIL')} {name}: {reason}"); return False


def _assert(cond, name, reason=""):
    if not cond:
        _fail(name, reason)
        return False
    _pass(name)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Training forward pass
# ─────────────────────────────────────────────────────────────────────────────

def test1_training_forward() -> bool:
    print(_bold("\nTest 1 — Training forward pass"))
    ok = True

    model = NIRDet()
    model.train()

    x = torch.zeros(2, 1, 640, 640)
    with torch.no_grad():
        preds = model(x, training_mode=True)

    # Must return a list of 3 tensors
    ok &= _assert(isinstance(preds, list) and len(preds) == 3,
                  "returns list of 3 tensors", f"got {type(preds)}")

    expected_shapes = [(2, 6400, 5), (2, 1600, 5), (2, 400, 5)]
    for i, (pred, exp) in enumerate(zip(preds, expected_shapes)):
        ok &= _assert(
            tuple(pred.shape) == exp,
            f"  preds[{i}] shape = {exp}",
            f"got {tuple(pred.shape)}"
        )

    # Shapes: 640/8=80 → 80*80=6400; 640/16=40 → 1600; 640/32=20 → 400
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Inference forward pass
# ─────────────────────────────────────────────────────────────────────────────

def test2_inference_forward() -> bool:
    print(_bold("\nTest 2 — Inference forward pass"))
    ok = True

    try:
        import torchvision
    except ImportError:
        print("  SKIP — torchvision not installed (required for NMS in inference mode)")
        print("         pip install torchvision")
        return True  # not a model bug; dependency missing

    model = NIRDet()
    model.eval()

    x = torch.zeros(1, 1, 640, 640)
    with torch.no_grad():
        results = model(x, training_mode=False)

    ok &= _assert(isinstance(results, list) and len(results) == 1,
                  "returns list of length 1 (batch=1)", f"got {len(results)}")

    boxes, scores = results[0]

    ok &= _assert(boxes.ndim == 2 and boxes.shape[1] == 4,
                  "boxes shape = (N, 4)", f"got {tuple(boxes.shape)}")
    ok &= _assert(scores.ndim == 1,
                  "scores shape = (N,)", f"got {tuple(scores.shape)}")
    ok &= _assert(boxes.shape[0] == scores.shape[0],
                  "boxes and scores have same N",
                  f"boxes N={boxes.shape[0]}, scores N={scores.shape[0]}")

    if scores.numel() > 0:
        ok &= _assert(
            scores.min().item() >= 0.0 and scores.max().item() <= 1.0,
            "all scores in [0, 1]",
            f"min={scores.min().item():.4f}, max={scores.max().item():.4f}"
        )
        ok &= _assert(
            boxes[:, 0].min().item() >= 0 and boxes[:, 2].max().item() <= 640,
            "box x-coords within [0, 640]",
            f"x1_min={boxes[:,0].min().item():.1f}, x2_max={boxes[:,2].max().item():.1f}"
        )
        ok &= _assert(
            boxes[:, 1].min().item() >= 0 and boxes[:, 3].max().item() <= 640,
            "box y-coords within [0, 640]",
            f"y1_min={boxes[:,1].min().item():.1f}, y2_max={boxes[:,3].max().item():.1f}"
        )
    else:
        # Zero detections on a blank image is expected; not a failure.
        _pass("no detections on blank image (expected — score_thresh filters all)")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Parameter count
# ─────────────────────────────────────────────────────────────────────────────

def test3_param_count() -> bool:
    print(_bold("\nTest 3 — Parameter count"))
    ok = True

    model = NIRDet()
    breakdown = model.param_breakdown()

    print(f"\n  {'Component':<12} {'Params':>12}")
    print(f"  {'-'*26}")
    for k in ("backbone", "eaa", "neck", "head"):
        print(f"  {k:<12} {breakdown[k]:>12,}")
    print(f"  {'='*26}")
    print(f"  {'total':<12} {breakdown['total']:>12,}")
    print()

    ok &= _assert(
        breakdown["total"] < 5_000_000,
        "total < 5,000,000",
        f"got {breakdown['total']:,}"
    )
    # Sanity: each component has at least some params
    for k in ("backbone", "eaa", "neck", "head"):
        ok &= _assert(breakdown[k] > 0, f"{k} has params", f"got {breakdown[k]}")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Overfit test (most critical)
# ─────────────────────────────────────────────────────────────────────────────

def _load_yolo_labels(
    label_path: str, img_h: int, img_w: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse a YOLO-format .txt file and return (boxes_xyxy, cls_ids).
    YOLO format: class cx_norm cy_norm w_norm h_norm (all normalized to [0,1]).
    Returns boxes in absolute pixel coords (x1,y1,x2,y2) and class tensor.
    """
    boxes, cls_ids = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx_n, cy_n, w_n, h_n = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            cx = cx_n * img_w;  cy = cy_n * img_h
            w  = w_n  * img_w;  h  = h_n  * img_h
            x1 = cx - w / 2;   y1 = cy - h / 2
            x2 = cx + w / 2;   y2 = cy + h / 2
            boxes.append([x1, y1, x2, y2])
            cls_ids.append(cls_id)
    if not boxes:
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(cls_ids, dtype=torch.long)


def _proxy_loss(
    preds: List[torch.Tensor],
    targets: List[dict],
    input_h: int,
    input_w: int,
    strides: Tuple[int, int, int] = (8, 16, 32),
) -> torch.Tensor:
    """
    Minimal proxy loss for the overfit test.
    This is NOT the Phase 6 loss — it is a simplified version that:
      - assigns GT boxes to the closest grid cell on each scale
      - computes BCE on confidence logit
      - computes L1 on (t_cx, t_cy, t_w, t_h) offsets for assigned cells

    Purpose: verify the model can overfit (gradients flow, predictions change).
    Phase 6 will replace this with a proper focal + GIoU + centerness loss.

    Args:
        preds   : list of (B, H_i*W_i, 5) raw logits from head
        targets : list of B dicts, each {"boxes": (N,4) xyxy, "cls": (N,)}
        input_h, input_w: image spatial dims
        strides : (8, 16, 32) matching head strides
    """
    B = preds[0].shape[0]
    device = preds[0].device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for scale_idx, (pred, stride) in enumerate(zip(preds, strides)):
        # pred: (B, H*W, 5) → last dim = (t_cx, t_cy, t_w, t_h, conf_logit)
        feat_h = input_h // stride
        feat_w = input_w // stride
        num_cells = feat_h * feat_w

        # Build confidence targets: (B, H*W)
        conf_target = torch.zeros(B, num_cells, device=device)
        reg_target  = torch.zeros(B, num_cells, 4, device=device)
        reg_mask    = torch.zeros(B, num_cells, device=device, dtype=torch.bool)

        for b, tgt in enumerate(targets):
            boxes = tgt["boxes"].to(device)   # (N, 4) xyxy absolute pixels
            if boxes.numel() == 0:
                continue

            # GT center in grid coordinates
            cx_abs = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy_abs = (boxes[:, 1] + boxes[:, 3]) * 0.5
            w_abs  = boxes[:, 2] - boxes[:, 0]
            h_abs  = boxes[:, 3] - boxes[:, 1]

            gx = (cx_abs / stride).long().clamp(0, feat_w - 1)  # (N,)
            gy = (cy_abs / stride).long().clamp(0, feat_h - 1)  # (N,)
            cell_idx = gy * feat_w + gx                          # (N,) flat index

            # Regression target offsets in the YOLOX formulation
            t_cx = cx_abs / stride - (gx.float() + 0.5)  # (N,)
            t_cy = cy_abs / stride - (gy.float() + 0.5)  # (N,)
            t_w  = torch.log(w_abs / stride + 1e-6)       # (N,)
            t_h  = torch.log(h_abs / stride + 1e-6)       # (N,)

            for i, cidx in enumerate(cell_idx):
                conf_target[b, cidx] = 1.0
                reg_target[b, cidx]  = torch.tensor(
                    [t_cx[i], t_cy[i], t_w[i], t_h[i]]
                )
                reg_mask[b, cidx] = True

        # Confidence BCE loss (all cells)
        conf_logit = pred[..., 4]   # (B, H*W)
        conf_loss  = F.binary_cross_entropy_with_logits(
            conf_logit, conf_target, reduction="mean"
        )

        # Regression L1 loss (positive cells only)
        if reg_mask.any():
            pred_reg    = pred[..., :4][reg_mask]   # (num_pos, 4)
            target_reg  = reg_target[reg_mask]       # (num_pos, 4)
            reg_loss = F.l1_loss(pred_reg, target_reg, reduction="mean")
        else:
            reg_loss = torch.tensor(0.0, device=device)

        total_loss = total_loss + conf_loss + 0.05 * reg_loss

    return total_loss


def test4_overfit(
    image_paths: List[str],
    label_paths: List[str],
    input_size: int = 640,
    steps: int = 100,
    lr: float = 1e-3,
    print_every: int = 10,
) -> bool:
    """
    Test 4 — Overfit on 4 real images.

    Criteria:
        - Loss at step 100 < loss at step 1 (must decrease)
        - Loss is printed every 10 steps for manual inspection
        - If loss does not decrease by step 30: STOP and report

    Returns True if loss trends downward, False otherwise.
    """
    print(_bold("\nTest 4 — Overfit test (most critical)"))

    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("  REQUIRES: pip install Pillow numpy")
        return False

    if len(image_paths) != 4 or len(label_paths) != 4:
        print(f"  ERROR: need exactly 4 images and 4 labels, "
              f"got {len(image_paths)} images, {len(label_paths)} labels")
        return False

    # ── Load images ───────────────────────────────────────────────────────────
    imgs = []
    targets = []
    for img_path, lbl_path in zip(image_paths, label_paths):
        img = Image.open(img_path).convert("L")  # force grayscale
        img = img.resize((input_size, input_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]
        imgs.append(torch.tensor(arr).unsqueeze(0).unsqueeze(0))  # (1,1,H,W)

        boxes, cls_ids = _load_yolo_labels(lbl_path, input_size, input_size)
        targets.append({"boxes": boxes, "cls": cls_ids})

    # Stack into batch (4, 1, H, W)
    x = torch.cat(imgs, dim=0)  # (4, 1, input_size, input_size)
    print(f"  Loaded 4 images: {x.shape}, "
          f"total GT boxes: {sum(len(t['boxes']) for t in targets)}")

    # ── Model + optimizer ─────────────────────────────────────────────────────
    model = NIRDet()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ── Training loop ─────────────────────────────────────────────────────────
    losses = []
    stopped_early = False
    t0 = time.time()

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        preds = model(x, training_mode=True)

        loss = _proxy_loss(preds, targets, input_size, input_size)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % print_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  step {step:>4d} / {steps}  loss = {loss_val:.6f}  "
                  f"({elapsed:.1f}s elapsed)")

        # Early stop check: if loss hasn't moved at all by step 30, something
        # is broken (NaN, exploding grad, frozen params).
        if step == 30:
            first, last = losses[0], losses[-1]
            if last >= first * 0.999 or (not torch.isfinite(torch.tensor(last))):
                print(f"\n  {_red('STOPPING EARLY')} — loss has not decreased by step 30.")
                print(f"  Initial: {first:.6f}, Step 30: {last:.6f}")
                print("  Likely causes:")
                print("    1. Learning rate too low — try lr=5e-3")
                print("    2. Label/image mismatch — verify .txt matches image content")
                print("    3. All GT boxes outside image bounds after resize")
                print("    4. NaN in loss — check for zero-width/height boxes in labels")
                stopped_early = True
                break

    if stopped_early:
        return False

    first_loss = losses[0]
    final_loss = losses[-1]
    improved   = final_loss < first_loss

    print(f"\n  Initial loss : {first_loss:.6f}")
    print(f"  Final loss   : {final_loss:.6f}")
    print(f"  Decrease     : {(first_loss - final_loss):.6f} "
          f"({(1 - final_loss/first_loss)*100:.1f}%)")

    result = _assert(
        improved,
        "loss trends downward",
        f"initial={first_loss:.6f}, final={final_loss:.6f} — no decrease detected. "
        "Do NOT proceed to Phase 6 until this passes."
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIRDet Phase 5 unit tests")
    parser.add_argument("--overfit", action="store_true",
                        help="Run Test 4 (overfit). Requires --images and --labels.")
    parser.add_argument("--images", nargs="+", default=[],
                        help="4 image paths for Test 4")
    parser.add_argument("--labels", nargs="+", default=[],
                        help="4 YOLO .txt label paths for Test 4")
    parser.add_argument("--input-size", type=int, default=640,
                        help="Resize all images to this square size for Test 4 (default 640)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Gradient steps for Test 4 (default 100)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="AdamW lr for Test 4 (default 1e-3)")
    args = parser.parse_args()

    print(_bold("NIRDet Phase 5 — Unit Tests"))
    print("=" * 50)

    results = {}
    results["test1"] = test1_training_forward()
    results["test2"] = test2_inference_forward()
    results["test3"] = test3_param_count()

    if args.overfit:
        results["test4"] = test4_overfit(
            image_paths=args.images,
            label_paths=args.labels,
            input_size=args.input_size,
            steps=args.steps,
            lr=args.lr,
        )
    else:
        print(_bold("\nTest 4 — Overfit test"))
        print("  SKIPPED — pass --overfit with --images and --labels to run.")
        print("  MANDATORY before Phase 6: this test must pass on real data.")
        results["test4"] = None

    # ── Summary ───────────────────────────────────────────────────────────────
    print(_bold("\n" + "=" * 50))
    print(_bold("Summary"))
    all_passed = True
    for name, result in results.items():
        if result is None:
            print(f"  {name}: SKIPPED")
        elif result:
            print(f"  {name}: {_green('PASS')}")
        else:
            print(f"  {name}: {_red('FAIL')}")
            all_passed = False

    if all_passed and results["test4"] is not False:
        if results["test4"] is None:
            print(_bold("\nTests 1-3 passed. Run Test 4 before proceeding to Phase 6."))
        else:
            print(_bold(f"\n{_green('All tests passed.')} Ready for Phase 6."))
    else:
        print(_bold(f"\n{_red('One or more tests failed.')} Fix issues before Phase 6."))
        sys.exit(1)


if __name__ == "__main__":
    main()
