#!/usr/bin/env python
"""
test_dataset_profiles.py — Upgrade H verification
==================================================
Dataset profiles with content fingerprinting. Every check builds synthetic
label+image trees in a throwaway temp directory; the real miniNIRPed
labels are never modified. Run inside the project venv:

    python test_dataset_profiles.py

Exit code 0 = all checks PASS.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config, validate_config
from dataset_profiles import (DatasetProfile, StaleProfileError, scan,
                              verify_fresh, apply)

FAILURES = []


def check(n: int, name: str, fn) -> None:
    try:
        fn()
        print(f"PASS  {n:2d}  {name}")
    except Exception:
        print(f"FAIL  {n:2d}  {name}")
        traceback.print_exc()
        FAILURES.append(n)


# ---------------------------------------------------------------------------
# synthetic fixture
# ---------------------------------------------------------------------------

def yolo_line(cx, cy, w, h, cls=0):
    return f"{cls} {cx} {cy} {w} {h}\n"


# All images 1280x720, exactly like the real miniNIRPed. With the default
# 384x640 canvas: scale = min(640/1280, 384/720) = 0.5, so
#   w_canvas = w * 1280 * 0.5 / 640  = w        (unchanged)
#   h_canvas = h * 720  * 0.5 / 384  = h * 0.9375
SRC_W, SRC_H = 1280, 720


def build_fixture(root: Path):
    """train: 6 label+image pairs (2 box-less), val: 3 (1 empty), test: 2.

    train raw widths  = [0.10, 0.20, 0.20, 0.40, 0.20]  -> canvas median 0.20
    train raw heights = [0.50, 0.30, 0.30, 0.10, 0.30]  -> canvas median
                        0.30*0.9375 = 0.28125
    """
    from PIL import Image
    tr_l = root / "labels" / "train"
    tr_i = root / "images" / "train"
    va_l = root / "labels" / "val"
    va_i = root / "images" / "val"
    te_l = root / "labels" / "test"
    te_i = root / "images" / "test"
    for d in (tr_l, tr_i, va_l, va_i, te_l, te_i):
        d.mkdir(parents=True)

    (tr_l / "a.txt").write_text("")                                     # empty
    (tr_l / "b.txt").write_text(yolo_line(0.5, 0.5, 0.10, 0.50))
    (tr_l / "c.txt").write_text(yolo_line(0.1, 0.1, 0.20, 0.30)
                              + yolo_line(0.8, 0.8, 0.20, 0.30))
    (tr_l / "d.txt").write_text(yolo_line(0.5, 0.9, 0.40, 0.10))
    (tr_l / "e.txt").write_text(yolo_line(0.3, 0.3, 0.20, 0.30))
    (tr_l / "f.txt").write_text("# malformed line only\n0.5 0.5\n")     # 0 boxes
    (va_l / "g.txt").write_text(yolo_line(0.5, 0.5, 0.30, 0.60))
    (va_l / "h.txt").write_text("")                                      # empty
    (va_l / "i.txt").write_text(yolo_line(0.2, 0.2, 0.25, 0.35))
    (te_l / "j.txt").write_text(yolo_line(0.5, 0.5, 0.55, 0.85))
    (te_l / "k.txt").write_text(yolo_line(0.6, 0.4, 0.15, 0.25))

    img = Image.new("L", (SRC_W, SRC_H), 128)
    for split, stems in (("train", "abcdef"), ("val", "ghi"), ("test", "jk")):
        for stem in stems:
            img.save(root / "images" / split / f"{stem}.png")


TMP = Path(tempfile.mkdtemp(prefix="nirdet_profile_test_"))
ROOT = TMP / "ds"
build_fixture(ROOT)
PROFILE_PATH = TMP / "profiles" / "ds_6.yaml"


# ---------------------------------------------------------------------------
# 1-3: scan / build
# ---------------------------------------------------------------------------

def t01_scan_derives_counts():
    d = scan(ROOT)
    assert d["n_train"] == 6, d["n_train"]
    assert d["n_val"] == 3, d["n_val"]
    assert d["n_train_images"] == 6
    assert d["stats"]["n_label_files_by_split"] == {"train": 6, "val": 3, "test": 2}
    assert d["stats"]["n_images_by_split"] == {"train": 6, "val": 3, "test": 2}


def t02_scan_derives_canvas_priors():
    d = scan(ROOT)
    # letterbox 1280x720 -> 384x640: w unchanged, h *= 0.9375
    assert abs(d["prior_w"] - 0.20) < 1e-9, d["prior_w"]
    assert abs(d["prior_h"] - 0.30 * 0.9375) < 1e-9, d["prior_h"]
    s = d["stats"]["train"]
    assert abs(s["raw_median_w"] - 0.20) < 1e-9
    assert abs(s["raw_median_h"] - 0.30) < 1e-9


def t03_scan_derives_diagnostics():
    s = scan(ROOT)["stats"]["train"]
    assert s["n_boxes"] == 5
    assert abs(s["boxes_per_image"]["mean"] - 5 / 6) < 1e-4
    assert s["boxes_per_image"]["median"] == 1.0
    assert s["boxes_per_image"]["max"] == 2
    assert abs(s["empty_label_fraction"] - 2 / 6) < 1e-6
    for key in ("p1", "p5", "p25", "p50", "p75", "p95", "p99", "min", "max"):
        assert key in s["width_percentiles"], key
        assert key in s["height_percentiles"], key


# ---------------------------------------------------------------------------
# 4-5: fingerprints / round-trip
# ---------------------------------------------------------------------------

def t04_fingerprints_stable_and_sha256():
    d1, d2 = scan(ROOT), scan(ROOT)
    assert d1["label_fingerprint"] == d2["label_fingerprint"]
    assert d1["image_geometry_fingerprint"] == d2["image_geometry_fingerprint"]
    for key in ("label_fingerprint", "image_geometry_fingerprint"):
        f = d1[key]
        assert f.startswith("sha256:") and len(f) == 7 + 64, f


def t05_save_load_roundtrip():
    p = DatasetProfile.build(ROOT, name="ds_test")
    out = p.save(PROFILE_PATH)
    assert out == PROFILE_PATH and PROFILE_PATH.exists()
    q = DatasetProfile.load(PROFILE_PATH)
    assert q.derived == p.derived, "derived section mutated by YAML round-trip"


# ---------------------------------------------------------------------------
# 6-13: verify_fresh
# ---------------------------------------------------------------------------

def _mutated_copy():
    """Copy the fixture + a fresh profile, so mutations never leak."""
    root2 = TMP / "ds2"
    if root2.exists():
        shutil.rmtree(root2)
    shutil.copytree(ROOT, root2)
    prof = DatasetProfile.build(root2)
    prof.save(TMP / "profiles" / "ds2.yaml")
    return root2, DatasetProfile.load(TMP / "profiles" / "ds2.yaml")


def t06_fresh_passes():
    prof = DatasetProfile.load(PROFILE_PATH)
    prof.verify_fresh()          # must not raise
    verify_fresh(prof, ROOT)     # module-level form


def t07_content_edit_detected():
    root2, prof = _mutated_copy()
    # Median of 5 is robust to a single outlier, so this edit must move BOTH
    # medians: c.txt holds two of the three 0.20/0.30-width boxes.
    (root2 / "labels" / "train" / "c.txt").write_text(
        yolo_line(0.1, 0.1, 0.10, 0.10) + yolo_line(0.8, 0.8, 0.10, 0.10))
    try:
        prof.verify_fresh()
        raise AssertionError("stale profile accepted after label edit")
    except StaleProfileError as e:
        msg = str(e)
        assert "label_fingerprint" in msg, msg
        assert "prior_w" in msg and "prior_h" in msg, msg


def t08_added_label_file_detected():
    root2, prof = _mutated_copy()
    (root2 / "labels" / "train" / "zz_new.txt").write_text(
        yolo_line(0.5, 0.5, 0.2, 0.3))
    try:
        prof.verify_fresh()
        raise AssertionError("stale profile accepted after file added")
    except StaleProfileError as e:
        assert "n_train" in str(e), str(e)


def t09_removed_label_file_detected():
    root2, prof = _mutated_copy()
    (root2 / "labels" / "val" / "h.txt").unlink()
    try:
        prof.verify_fresh()
        raise AssertionError("stale profile accepted after file removed")
    except StaleProfileError as e:
        assert "n_val" in str(e), str(e)


def t10_renamed_label_file_detected():
    root2, prof = _mutated_copy()
    (root2 / "labels" / "train" / "b.txt").rename(
        root2 / "labels" / "train" / "b_renamed.txt")
    try:
        prof.verify_fresh()
        raise AssertionError("rename aliased the fingerprint (rel path must be hashed)")
    except StaleProfileError:
        pass


def t11_hand_edit_of_derived_detected():
    root2, prof = _mutated_copy()
    data = yaml.safe_load((TMP / "profiles" / "ds2.yaml").read_text())
    data["derived"]["prior_w"] = 0.999              # fingerprints still valid
    (TMP / "profiles" / "ds2.yaml").write_text(yaml.safe_dump(data))
    prof = DatasetProfile.load(TMP / "profiles" / "ds2.yaml")
    try:
        prof.verify_fresh()
        raise AssertionError("hand-edited derived section accepted")
    except StaleProfileError as e:
        msg = str(e)
        assert "derived.prior_w" in msg, msg
        assert "hand-edited" in msg, msg


def t12_image_reencode_not_fingerprinted():
    root2, prof = _mutated_copy()
    # Re-encode an image at the SAME size: new bytes, same geometry.
    from PIL import Image
    p = root2 / "images" / "train" / "b.png"
    Image.new("L", (SRC_W, SRC_H), 200).save(p, format="JPEG")
    # .png path now holds JPEG bytes; Pillow still reads the header fine
    prof.verify_fresh()          # must not raise


def t13_image_resize_detected():
    root2, prof = _mutated_copy()
    from PIL import Image
    # Resize the image carrying two of the three median-height boxes
    # (c.txt), from 1280x720 (scale 0.5, height-limited 0.9375x) to
    # 800x400 (scale 0.8, width-limited): h_canvas 0.28125 -> 0.25, so
    # the median and therefore prior_h must move.
    Image.new("L", (800, 400), 128).save(root2 / "images" / "train" / "c.png")
    try:
        prof.verify_fresh()
        raise AssertionError("image resize accepted with stale canvas priors")
    except StaleProfileError as e:
        msg = str(e)
        assert "image_geometry_fingerprint" in msg, msg
        assert "prior_h" in msg, msg


def t14_missing_root_raises_with_message():
    prof = DatasetProfile.load(PROFILE_PATH)
    bad = DatasetProfile({"name": "x", "root": "Z:/nope",
                          "derived": prof.derived})
    try:
        bad.verify_fresh()
        raise AssertionError("missing root accepted")
    except StaleProfileError as e:
        assert "Z:/nope" in str(e), str(e)


# ---------------------------------------------------------------------------
# 15-17: apply / canvas
# ---------------------------------------------------------------------------

def t15_apply_overrides_exactly_five_fields():
    prof = DatasetProfile.load(PROFILE_PATH)
    before = get_config()
    before.data.root = "SENTINEL_ROOT"
    before.data.n_train = -1
    before.data.n_val = -1
    before.model.prior_w = 0.123456
    before.model.prior_h = 0.654321
    before.train.epochs = 7
    before.model.nms_score_thresh = 0.5
    before.data.img_h = 384
    snap = {k: dict(v) for k, v in
            ((s, vars(getattr(before, s))) for s in
             ("data", "model", "loss", "aug", "train", "eval", "export"))}
    apply(prof, before)
    assert before.data.root == prof.root
    assert before.data.n_train == 6
    assert before.data.n_val == 3
    assert before.model.prior_w == 0.20
    assert before.model.prior_h == 0.28125
    # nothing else moved
    for sec, fields in snap.items():
        cur = vars(getattr(before, sec))
        for k, v in fields.items():
            if (sec, k) not in (("data", "root"), ("data", "n_train"),
                                ("data", "n_val"), ("model", "prior_w"),
                                ("model", "prior_h")):
                assert cur[k] == v, f"apply() touched {sec}.{k}"


def t16_apply_then_validate_config_passes():
    cfg = get_config()
    prof = DatasetProfile.load(PROFILE_PATH)
    prof.apply(cfg)
    assert validate_config(cfg, verbose=False)


def t17_check_canvas_guards_resolution():
    prof = DatasetProfile.load(PROFILE_PATH)
    prof.check_canvas(get_config())            # 384x640 == derived -> OK
    cfg = get_config()
    cfg.data.img_h = 480                       # profile is 384x640
    try:
        prof.check_canvas(cfg)
        raise AssertionError("canvas mismatch accepted")
    except RuntimeError as e:
        assert "384x640" in str(e) and "480x640" in str(e), str(e)


# ---------------------------------------------------------------------------
# 18-20: deploy_score_thresh write-back
# ---------------------------------------------------------------------------

def t18_record_then_reload():
    root2, prof = _mutated_copy()
    out = prof.record_deploy_threshold(0.4137, path=TMP / "profiles" / "ds2.yaml")
    reloaded = DatasetProfile.load(out)
    assert reloaded.data["deploy_score_thresh"] == 0.4137
    reloaded.verify_fresh()               # write-back must not disturb derived
    assert reloaded.derived == prof.derived


def t19_record_refuses_stale():
    root2, prof = _mutated_copy()
    (root2 / "labels" / "train" / "b.txt").write_text("")
    try:
        prof.record_deploy_threshold(0.5, path=TMP / "profiles" / "ds2.yaml")
        raise AssertionError("threshold written back onto a stale profile")
    except StaleProfileError:
        pass


def t20_record_rejects_out_of_range():
    prof = DatasetProfile.load(PROFILE_PATH)
    for bad in (0.0, 1.0, 1.5, -0.1, float("nan")):
        try:
            prof.record_deploy_threshold(bad, path=PROFILE_PATH)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# 21-24: regeneration / unlabeled-image drift
# ---------------------------------------------------------------------------

def t21_regen_preserves_measured_threshold():
    root2, prof = _mutated_copy()
    prof.record_deploy_threshold(0.4137, path=TMP / "profiles" / "ds2.yaml")
    import dataset_profiles as dp
    dp.main(["--root", str(root2),
             "--out", str(TMP / "profiles" / "ds2.yaml")])
    kept = DatasetProfile.load(TMP / "profiles" / "ds2.yaml")
    assert kept.data["deploy_score_thresh"] == 0.4137, \
        "regeneration with unchanged labels lost the measured threshold"
    kept.verify_fresh()


def t22_regen_after_geometry_change_resets_threshold():
    root2, prof = _mutated_copy()
    prof.record_deploy_threshold(0.4137, path=TMP / "profiles" / "ds2.yaml")
    from PIL import Image
    Image.new("L", (640, 640), 128).save(root2 / "images" / "train" / "b.png")
    import dataset_profiles as dp
    dp.main(["--root", str(root2),
             "--out", str(TMP / "profiles" / "ds2.yaml")])
    kept = DatasetProfile.load(TMP / "profiles" / "ds2.yaml")
    assert kept.data["deploy_score_thresh"] is None, \
        "threshold preserved across an image-geometry change (priors moved)"


def t23_unlabeled_image_added_detected():
    root2, prof = _mutated_copy()
    from PIL import Image
    Image.new("L", (SRC_W, SRC_H), 128).save(
        root2 / "images" / "train" / "unlabeled.png")   # no .txt pairing
    try:
        prof.verify_fresh()
        raise AssertionError("unlabeled image added after profiling accepted")
    except StaleProfileError as e:
        assert "image_geometry_fingerprint" in str(e), str(e)


def t24_label_without_image_is_invisible_to_priors():
    # dataset.py iterates IMAGES, so a label file with no image never reaches
    # training; scan must derive priors the same way (and must not crash).
    root2 = TMP / "ds3"
    if root2.exists():
        shutil.rmtree(root2)
    shutil.copytree(ROOT, root2)
    (root2 / "labels" / "train" / "orphan.txt").write_text(
        yolo_line(0.5, 0.5, 0.99, 0.99))          # would move medians if counted
    d = scan(root2)
    assert d["n_train"] == 7                      # label dir IS scanned for counts
    assert d["prior_w"] == 0.20 and d["prior_h"] == 0.28125  # priors unchanged


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 62)
    print("  test_dataset_profiles.py — Upgrade H")
    print("=" * 62)
    for n, name, fn in [
        (1, "scan derives n_train/n_val/split counts (labels+images)", t01_scan_derives_counts),
        (2, "priors are canvas-space (letterbox), raw medians kept in stats", t02_scan_derives_canvas_priors),
        (3, "scan derives diagnostics (boxes/img, empty frac, percentiles)", t03_scan_derives_diagnostics),
        (4, "label + geometry fingerprints deterministic, sha256", t04_fingerprints_stable_and_sha256),
        (5, "save/load round-trips derived unchanged", t05_save_load_roundtrip),
        (6, "verify_fresh passes on unchanged tree", t06_fresh_passes),
        (7, "label content edit -> StaleProfileError names fingerprint+priors", t07_content_edit_detected),
        (8, "label file added -> StaleProfileError names n_train", t08_added_label_file_detected),
        (9, "label file removed -> StaleProfileError names n_val", t09_removed_label_file_detected),
        (10, "label file renamed -> detected (rel path hashed)", t10_renamed_label_file_detected),
        (11, "hand-edited derived -> detected, names derived.prior_w", t11_hand_edit_of_derived_detected),
        (12, "image re-encoded at same size -> still fresh", t12_image_reencode_not_fingerprinted),
        (13, "image resized -> detected via geometry fingerprint + prior_h", t13_image_resize_detected),
        (14, "missing root -> StaleProfileError naming the root", t14_missing_root_raises_with_message),
        (15, "apply() overrides exactly the 5 permitted fields", t15_apply_overrides_exactly_five_fields),
        (16, "apply() then validate_config() passes", t16_apply_then_validate_config_passes),
        (17, "check_canvas() refuses a resolution mismatch", t17_check_canvas_guards_resolution),
        (18, "record_deploy_threshold writes + reloads + stays fresh", t18_record_then_reload),
        (19, "record_deploy_threshold refuses a stale profile", t19_record_refuses_stale),
        (20, "record_deploy_threshold rejects out-of-range values", t20_record_rejects_out_of_range),
        (21, "regeneration with unchanged labels keeps the threshold", t21_regen_preserves_measured_threshold),
        (22, "regeneration after geometry change resets the threshold", t22_regen_after_geometry_change_resets_threshold),
        (23, "unlabeled image added -> detected", t23_unlabeled_image_added_detected),
        (24, "orphan label file: counted in n_train, invisible to priors", t24_label_without_image_is_invisible_to_priors),
    ]:
        check(n, name, fn)
    print("=" * 62)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("  all 24 checks PASS")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    code = main()
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(code)
