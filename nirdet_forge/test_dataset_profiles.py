"""
test_dataset_profiles.py — profile module invariants
=====================================================
    python test_dataset_profiles.py
    python test_dataset_profiles.py --profile datasets/miniNIRPed_261.yaml

The tests that need no data build a synthetic profile in a temp directory, so
this file runs on a machine with no dataset mounted. If --profile is given,
the real YAML is additionally checked against the live config.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from config import STRIDES, get_config
from dataset_profiles import (CanvasMismatchError, DatasetProfile,
                              canvas_fingerprint, check_canvas,
                              label_fingerprint)

_failures: list = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _failures.append(msg)


def _synthetic(img_h: int = 288, img_w: int = 512,
               strides=STRIDES) -> DatasetProfile:
    return DatasetProfile(
        name="synthetic",
        root="/nonexistent",
        img_h=img_h, img_w=img_w, strides=list(strides),
        canvas_fingerprint=canvas_fingerprint(img_h, img_w, strides),
        label_sha256="deadbeef",
        n_train=261, n_val=160, n_test=165,
        n_train_boxes=948, n_val_boxes=600, n_test_boxes=610,
        prior_w=0.046094, prior_h=0.179167,
        deploy_score_thresh=0.42,
        splits={"train": {"n_images": 261, "n_boxes": 948}},
    )


def t1_fingerprint_sensitivity() -> None:
    print("\nT1  canvas fingerprint responds to every geometry field")
    base = canvas_fingerprint(288, 512, STRIDES)
    check(canvas_fingerprint(288, 512, STRIDES) == base,
          "fingerprint is deterministic")
    check(canvas_fingerprint(384, 512, STRIDES) != base,
          "changing img_h changes the fingerprint")
    check(canvas_fingerprint(288, 640, STRIDES) != base,
          "changing img_w changes the fingerprint")
    check(canvas_fingerprint(288, 512, (8, 16)) != base,
          "dropping a stride changes the fingerprint (the --p5-ablate case)")
    check(canvas_fingerprint(512, 288, STRIDES) != base,
          "transposing h and w changes the fingerprint")
    check(len(base) == 64, f"SHA-256 hex digest, {len(base)} chars")


def t2_priors_in_range() -> None:
    print("\nT2  priors are normalised and inside the regression clamp")
    from config import REG_LOG_CLAMP_MAX, REG_LOG_CLAMP_MIN
    import math
    p = _synthetic()
    check(0.0 < p.prior_w < 1.0, f"prior_w {p.prior_w} in (0, 1)")
    check(0.0 < p.prior_h < 1.0, f"prior_h {p.prior_h} in (0, 1)")
    for label, v in (("prior_w", p.prior_w), ("prior_h", p.prior_h)):
        lg = math.log(v)
        check(REG_LOG_CLAMP_MIN < lg < REG_LOG_CLAMP_MAX,
              f"log({label}) = {lg:.3f} strictly inside "
              f"({REG_LOG_CLAMP_MIN}, {REG_LOG_CLAMP_MAX}) — otherwise "
              f"head.size_pred.bias would be clipped and TAL would cold-start "
              f"with zero positives")


def t3_apply() -> None:
    print("\nT3  DatasetProfile.apply fills the config")
    cfg = get_config()
    p = _synthetic()
    p.apply(cfg, verbose=False)
    check(cfg.data.n_train == 261, f"n_train {cfg.data.n_train}")
    check(cfg.data.n_val == 160, f"n_val {cfg.data.n_val}")
    check(cfg.data.n_test == 165, f"n_test {cfg.data.n_test}")
    check(cfg.data.n_train_boxes == 948,
          f"n_train_boxes {cfg.data.n_train_boxes}")
    check(abs(cfg.model.prior_w - 0.046094) < 1e-9,
          f"prior_w {cfg.model.prior_w}")
    check(abs(cfg.model.prior_h - 0.179167) < 1e-9,
          f"prior_h {cfg.model.prior_h}")
    check(abs(cfg.eval.deploy_score_thresh - 0.42) < 1e-9,
          f"deploy_score_thresh {cfg.eval.deploy_score_thresh} "
          f"(measured by evaluate.py, consumed by live_nirdet.py)")

    # The box count must actually drive copy_paste_p.
    from config import copy_paste_p_for
    check(abs(copy_paste_p_for(cfg.data.n_train_boxes) - 0.50) < 1e-9,
          f"copy_paste_p_for(948) == 0.50 (the cap)")
    check(copy_paste_p_for(146_000) < copy_paste_p_for(948),
          "copy_paste_p falls as the box count rises")


def t4_check_canvas_raises() -> None:
    print("\nT4  check_canvas raises on a mismatch")
    cfg = get_config()
    ok_profile = _synthetic(cfg.data.img_h, cfg.data.img_w, cfg.model.strides)
    try:
        check_canvas(cfg, ok_profile)
        check(True, "matching fingerprint passes")
    except CanvasMismatchError:
        check(False, "matching fingerprint should not raise")

    for bad in (_synthetic(384, 640, cfg.model.strides),
                _synthetic(cfg.data.img_h, cfg.data.img_w, (8, 16)),
                DatasetProfile(canvas_fingerprint="")):
        raised = False
        try:
            check_canvas(cfg, bad)
        except CanvasMismatchError:
            raised = True
        check(raised,
              f"mismatch ({bad.img_h}x{bad.img_w}, strides "
              f"{tuple(bad.strides)}, fp "
              f"{(bad.canvas_fingerprint or '<empty>')[:8]}) raises")

    raised = False
    try:
        _synthetic(384, 640).apply(get_config(), verbose=False)
    except CanvasMismatchError:
        raised = True
    check(raised, "apply() runs check_canvas BEFORE writing any prior")


def t5_round_trip() -> None:
    print("\nT5  YAML round trip and deploy-threshold write-back")
    p = _synthetic()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prof.yaml")
        p.save(path)
        check(os.path.isfile(path), "save() wrote the file")
        q = DatasetProfile.load(path)
        check(q.canvas_fingerprint == p.canvas_fingerprint,
              "fingerprint survived the round trip")
        check(q.n_train_boxes == p.n_train_boxes and
              abs(q.prior_w - p.prior_w) < 1e-12,
              "counts and priors survived the round trip")

        q.update_deploy_thresh(path, 0.37)
        r = DatasetProfile.load(path)
        check(abs(r.deploy_score_thresh - 0.37) < 1e-12,
              "update_deploy_thresh persisted 0.37 (this is the evaluate.py "
              "-> live_nirdet.py hand-off)")
        check(r.prior_w == p.prior_w and r.n_train == p.n_train,
              "the write-back left everything else untouched")


def t6_label_fingerprint() -> None:
    print("\nT6  label fingerprint is content-sensitive (warning-level)")
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.txt")
        b = os.path.join(d, "b.txt")
        with open(a, "w", encoding="utf-8") as fh:
            fh.write("0 0.5 0.5 0.05 0.18\n")
        with open(b, "w", encoding="utf-8") as fh:
            fh.write("0 0.2 0.3 0.04 0.16\n")
        f1 = label_fingerprint([a, b])
        check(label_fingerprint([b, a]) == f1,
              "order-independent (paths are sorted)")
        with open(b, "w", encoding="utf-8") as fh:
            fh.write("0 0.2 0.3 0.04 0.17\n")
        check(label_fingerprint([a, b]) != f1,
              "editing a label changes the fingerprint")
        check(label_fingerprint([a]) != f1,
              "dropping a file changes the fingerprint")


def t7_real_profile(path: str) -> None:
    print(f"\nT7  real profile {path}")
    if not os.path.isfile(path):
        print(f"  SKIP  not found")
        return
    cfg = get_config()
    p = DatasetProfile.load(path)
    try:
        p.apply(cfg, verbose=False)
        check(True, f"applies cleanly at {cfg.data.img_h}x{cfg.data.img_w}")
    except CanvasMismatchError as exc:
        check(False, f"canvas mismatch: {str(exc).splitlines()[0]}")
        return
    check(p.n_train > 0 and p.n_train_boxes > 0,
          f"{p.n_train} train images, {p.n_train_boxes} boxes")
    check(0.0 < p.prior_w < 1.0 and 0.0 < p.prior_h < 1.0,
          f"priors w {p.prior_w:.6f} h {p.prior_h:.6f}")
    check(0.0 < p.deploy_score_thresh < 1.0,
          f"deploy_score_thresh {p.deploy_score_thresh:.3f}")
    tr = p.splits.get("train", {})
    if tr:
        print(f"        train: {tr.get('n_boxes')} boxes, "
              f"{tr.get('boxes_per_image_mean')} per image, "
              f"small/medium/large "
              f"{tr.get('n_small')}/{tr.get('n_medium')}/{tr.get('n_large')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="datasets/miniNIRPed_261.yaml")
    args = ap.parse_args()

    print("=" * 70)
    print("  dataset profile tests")
    print("=" * 70)
    t1_fingerprint_sensitivity()
    t2_priors_in_range()
    t3_apply()
    t4_check_canvas_raises()
    t5_round_trip()
    t6_label_fingerprint()
    t7_real_profile(args.profile)
    print("=" * 70)
    if _failures:
        print(f"  {len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  all dataset profile tests PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
