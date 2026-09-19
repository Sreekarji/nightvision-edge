"""
dataset_profiles.py — measure a dataset once, derive everything from it
========================================================================
Nothing in this codebase should contain a number tuned around one particular
dataset. This module is the mechanism: it scans a YOLO-format root, computes
box statistics in CANVAS space, and writes a YAML that config.py consumes
through DatasetProfile.apply(cfg).

WHAT IS DERIVED
---------------
    cfg.data.root                          the dataset root itself
    cfg.data.n_train / n_val / n_test      split sizes
    cfg.data.n_train_boxes                 -> copy_paste_p_for()
    cfg.model.prior_w / prior_h            -> head size_pred bias init
    cfg.eval.deploy_score_thresh           -> live_nirdet.py --profile
    cfg.eval.baseline_map50 / source       -> evaluate.py (when measured)
    cfg.aug.flat_field_path / clahe_*      -> THE PREPROCESSING CONTRACT

THE PREPROCESSING CONTRACT TRAVELS WITH THE PROFILE
---------------------------------------------------
flat_field_path, clahe_enabled, clahe_clip and clahe_grid are the other half
of the train/deploy contract (config.deploy_contract hashes them). Without
them in the profile there is no mechanism that transports the preprocessing
from the training machine to the Pi: live_nirdet.py --flat-field defaults to
cfg.aug.flat_field_path, which is None in a fresh config regardless of what
training used.

VERIFICATION IS OPT-OUT
-----------------------
apply(verify=False) skips the on-disk freshness check. The Pi does not host
the dataset, and it needs only the derived scalars; an unconditional
verify_fresh() made the documented deployment command impossible to run. The
CANVAS fingerprint check is NEVER skipped — it costs no I/O and it is the one
that silently biases the head's size-prior init.

TWO FINGERPRINTS, TWO SEVERITIES
--------------------------------
image geometry      HARD ERROR: dimensions set the letterbox scale, so a
                    resize invalidates prior_w / prior_h and the deploy
                    threshold.
label_sha256        WARNING: labels get corrected without geometry moving.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("dataset_profiles.py needs PyYAML: pip install pyyaml") from exc

from config import (DEFAULT_DEPLOY_SCORE_THRESH, STRIDES, Config, get_config)
from dataset import label_path_for, read_yolo_labels
# F57: path/split discovery now lives in the torch-free preprocess module.
from preprocess import list_images, resolve_split_dirs

PERCENTILES: Tuple[int, ...] = (5, 25, 50, 75, 95)

# COCO area thresholds, applied to CANVAS pixels.
SMALL_MAX_AREA = 32.0 ** 2
MEDIUM_MAX_AREA = 96.0 ** 2

# Boxes below this CANVAS extent are dropped by NIRPedDataset.__getitem__, so
# they must not contribute to the priors or to the box count that drives
# copy_paste_p either (F26).
MIN_CANVAS_BOX_PX: float = 2.0


def _fmt_thresh(v) -> str:
    """Format an Optional deploy threshold for prints (F13: None is legal)."""
    return "unset (run evaluate.py)" if v is None else f"{float(v):.3f}"


# ===========================================================================
# fingerprints
# ===========================================================================

def canvas_fingerprint(img_h: int, img_w: int,
                       strides: Sequence[int]) -> str:
    blob = f"{int(img_h)}x{int(img_w)}:" + \
           ",".join(str(int(s)) for s in strides)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def geometry_fingerprint(root: str, splits) -> str:
    """
    SHA-256 over (relative image path, width, height) for every image.

    Image BYTES are not hashed — re-encoding must stay free — but DIMENSIONS
    cannot be ignored: they set the letterbox scale, so resizing a source
    image silently moves prior_w / prior_h while leaving label_sha256 alone.

    An unreadable image is a HARD ERROR (F25). Encoding the failure as
    b"<unreadable>" meant a file that was temporarily locked at profile time
    and readable later produced a StaleProfileError claiming the geometry had
    changed — wrong and unactionable.
    """
    h = hashlib.sha256()
    for split in sorted(splits):
        try:
            img_dir, _ = resolve_split_dirs(root, split)
        except FileNotFoundError:
            continue
        for p in sorted(list_images(img_dir)):
            try:
                ih, iw = _image_size(p)
            except Exception as exc:
                raise RuntimeError(
                    f"cannot read image dimensions for {p}: {exc}. Fix or "
                    f"remove the file — its dimensions set the letterbox "
                    f"scale and therefore prior_w/prior_h.") from exc
            rel = os.path.relpath(p, root).replace("\\", "/").encode("utf-8")
            h.update(len(rel).to_bytes(4, "big"))
            h.update(rel)
            h.update(int(iw).to_bytes(4, "big"))
            h.update(int(ih).to_bytes(4, "big"))
    return h.hexdigest()


def label_fingerprint(label_files: Sequence[str], root: str = "") -> str:
    """
    Content-sensitive fingerprint of a set of label files.

    Paths are hashed ROOT-RELATIVE so that moving a file between splits
    (train ↔ test) changes the fingerprint even when the file contents are
    unchanged — a move invalidates prior_w/prior_h and deploy_score_thresh.
    """
    h = hashlib.sha256()
    for p in sorted(label_files):
        rel = os.path.relpath(p, root).replace("\\", "/") if root else os.path.basename(p)
        rel_b = rel.encode("utf-8")
        h.update(len(rel_b).to_bytes(4, "big"))
        h.update(rel_b)
        try:
            with open(p, "rb") as fh:
                h.update(fh.read())
        except OSError:
            pass
    return h.hexdigest()


# ===========================================================================
# image size probing
# ===========================================================================

try:
    from PIL import Image as _PIL_Image
except ImportError:
    raise SystemExit(
        "dataset_profiles.py requires Pillow: pip install Pillow")

_SIZE_CACHE: dict = {}

def _image_size(path: str) -> Tuple[int, int]:
    hit = _SIZE_CACHE.get(path)
    if hit is not None:
        return hit
    with _PIL_Image.open(path) as im:
        result = (im.height, im.width)
    _SIZE_CACHE[path] = result
    return result


def _letterbox_params(raw_h: int, raw_w: int, out_h: int, out_w: int
                      ) -> Tuple[float, int, int]:
    scale = min(float(out_w) / float(raw_w), float(out_h) / float(raw_h))
    nw = max(1, int(round(raw_w * scale)))
    nh = max(1, int(round(raw_h * scale)))
    return scale, (out_w - nw) // 2, (out_h - nh) // 2


# ===========================================================================
# split statistics
# ===========================================================================

@dataclass
class SplitStats:
    n_images: int = 0
    n_boxes: int = 0
    n_boxes_dropped_subpixel: int = 0
    boxes_per_image_mean: float = 0.0
    boxes_per_image_std: float = 0.0
    boxes_per_image_max: int = 0
    n_images_empty: int = 0
    w_pct: Dict[str, float] = field(default_factory=dict)
    h_pct: Dict[str, float] = field(default_factory=dict)
    ar_pct: Dict[str, float] = field(default_factory=dict)
    n_small: int = 0
    n_medium: int = 0
    n_large: int = 0
    median_w: float = 0.0
    median_h: float = 0.0
    zero_pad_fraction: float = 0.0


def _pct_dict(v: np.ndarray) -> Dict[str, float]:
    if v.size == 0:
        return {f"p{p}": 0.0 for p in PERCENTILES}
    q = np.percentile(v, PERCENTILES)
    return {f"p{p}": float(round(float(x), 6))
            for p, x in zip(PERCENTILES, q)}


def profile_split(root: str, split: str, img_h: int, img_w: int
                  ) -> Tuple[SplitStats, List[str]]:
    """
    Scan one split. Returns (stats, label_file_paths).

    All width/height/aspect statistics are canvas-normalised; the size buckets
    are canvas pixels. Boxes smaller than MIN_CANVAS_BOX_PX in either axis are
    EXCLUDED, because NIRPedDataset.__getitem__ drops them too — counting them
    biases prior_w/prior_h downward and inflates the count that drives
    copy_paste_p (F26).
    """
    img_dir, lbl_dir = resolve_split_dirs(root, split)
    images = list_images(img_dir)
    if not images:
        raise RuntimeError(f"no images in {img_dir}")

    label_files: List[str] = []
    per_img_counts: List[int] = []
    ws: List[float] = []
    hs: List[float] = []
    ars: List[float] = []
    areas_px: List[float] = []
    zero_pad = 0
    dropped = 0

    for p in images:
        lp = label_path_for(p, lbl_dir)
        label_files.append(lp)
        boxes = read_yolo_labels(lp)

        raw_h, raw_w = _image_size(p)
        scale, pad_x, pad_y = _letterbox_params(raw_h, raw_w, img_h, img_w)
        if pad_x == 0 and pad_y == 0:
            zero_pad += 1
        if boxes.shape[0] == 0:
            per_img_counts.append(0)
            continue

        bw_px = boxes[:, 2] * raw_w * scale
        bh_px = boxes[:, 3] * raw_h * scale
        keep = (bw_px > MIN_CANVAS_BOX_PX) & (bh_px > MIN_CANVAS_BOX_PX)
        dropped += int((~keep).sum())
        bw_px, bh_px = bw_px[keep], bh_px[keep]
        per_img_counts.append(int(keep.sum()))
        if bw_px.size == 0:
            continue

        ws.extend((bw_px / float(img_w)).tolist())
        hs.extend((bh_px / float(img_h)).tolist())
        ars.extend((bh_px / np.clip(bw_px, 1e-6, None)).tolist())
        areas_px.extend((bw_px * bh_px).tolist())

    counts = np.asarray(per_img_counts, dtype=np.float64)
    w_arr = np.asarray(ws, dtype=np.float64)
    h_arr = np.asarray(hs, dtype=np.float64)
    ar_arr = np.asarray(ars, dtype=np.float64)
    area_arr = np.asarray(areas_px, dtype=np.float64)

    st = SplitStats(
        n_images=len(images),
        n_boxes=int(counts.sum()),
        n_boxes_dropped_subpixel=int(dropped),
        boxes_per_image_mean=float(round(float(counts.mean()) if counts.size else 0.0, 4)),
        boxes_per_image_std=float(round(float(counts.std()) if counts.size else 0.0, 4)),
        boxes_per_image_max=int(counts.max()) if counts.size else 0,
        n_images_empty=int((counts == 0).sum()),
        w_pct=_pct_dict(w_arr),
        h_pct=_pct_dict(h_arr),
        ar_pct=_pct_dict(ar_arr),
        n_small=int((area_arr < SMALL_MAX_AREA).sum()),
        n_medium=int(((area_arr >= SMALL_MAX_AREA) &
                      (area_arr <= MEDIUM_MAX_AREA)).sum()),
        n_large=int((area_arr > MEDIUM_MAX_AREA).sum()),
        median_w=float(round(float(np.median(w_arr)), 6)) if w_arr.size else 0.0,
        median_h=float(round(float(np.median(h_arr)), 6)) if h_arr.size else 0.0,
        zero_pad_fraction=float(round(zero_pad / max(1, len(images)), 4)),
    )
    return st, label_files


# ===========================================================================
# profile
# ===========================================================================

class StaleProfileError(RuntimeError):
    """Label bytes or image dimensions moved since the profile was generated."""


class CanvasMismatchError(RuntimeError):
    """The profile was generated at a different canvas than the live config."""


@dataclass
class DatasetProfile:
    name: str = "unnamed"
    root: str = ""
    img_h: int = 288
    img_w: int = 512
    strides: List[int] = field(default_factory=lambda: list(STRIDES))
    canvas_fingerprint: str = ""
    label_sha256: str = ""
    image_geometry_fingerprint: str = ""

    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_train_boxes: int = 0
    n_val_boxes: int = 0
    n_test_boxes: int = 0

    # Canvas-space medians of the TRAIN split.
    prior_w: float = 0.0
    prior_h: float = 0.0

    # Fallback until evaluate.py measures the best-F1 threshold on the report
    # split and writes it back here. None = not yet measured (F13).
    deploy_score_thresh: Optional[float] = DEFAULT_DEPLOY_SCORE_THRESH

    baseline_map50: Optional[float] = None
    baseline_source: str = ""

    # --- preprocessing contract, transported to every consumer (F23) ---
    flat_field_path: Optional[str] = None
    clahe_enabled: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    splits: Dict[str, dict] = field(default_factory=dict)

    # ------------------------------------------------------------------ #

    @classmethod
    def from_root(cls, root: str, img_h: int, img_w: int,
                  strides: Sequence[int] = STRIDES,
                  name: Optional[str] = None,
                  deploy_score_thresh: Optional[float] = DEFAULT_DEPLOY_SCORE_THRESH,
                  baseline_map50: Optional[float] = None,
                  baseline_source: str = "",
                  flat_field_path: Optional[str] = None,
                  clahe_enabled: bool = True,
                  clahe_clip: float = 2.0,
                  clahe_grid: int = 8) -> "DatasetProfile":
        splits: Dict[str, dict] = {}
        all_labels: List[str] = []
        stats: Dict[str, SplitStats] = {}

        for split in ("train", "val", "test"):
            try:
                st, labels = profile_split(root, split, img_h, img_w)
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"[profile] skipping '{split}': {exc}")
                continue
            stats[split] = st
            splits[split] = asdict(st)
            all_labels += labels

        if "train" not in stats:
            raise RuntimeError(f"no train split found under {root!r}")
        tr = stats["train"]
        if tr.n_boxes == 0:
            raise RuntimeError(
                f"train split under {root!r} has zero usable boxes; priors "
                f"cannot be derived and TAL would cold-start with no positives")

        prof = cls(
            name=name or os.path.basename(os.path.normpath(root)),
            root=os.path.abspath(root),
            img_h=int(img_h), img_w=int(img_w),
            strides=[int(s) for s in strides],
            canvas_fingerprint=canvas_fingerprint(img_h, img_w, strides),
            label_sha256=label_fingerprint(all_labels, root=root),
            image_geometry_fingerprint=geometry_fingerprint(root, stats.keys()),
            n_train=tr.n_images,
            n_val=stats["val"].n_images if "val" in stats else 0,
            n_test=stats["test"].n_images if "test" in stats else 0,
            n_train_boxes=tr.n_boxes,
            n_val_boxes=stats["val"].n_boxes if "val" in stats else 0,
            n_test_boxes=stats["test"].n_boxes if "test" in stats else 0,
            prior_w=tr.median_w,
            prior_h=tr.median_h,
            deploy_score_thresh=(float(deploy_score_thresh)
                                 if deploy_score_thresh is not None else None),
            baseline_map50=(float(baseline_map50)
                            if baseline_map50 is not None else None),
            baseline_source=str(baseline_source or ""),
            flat_field_path=(str(flat_field_path) if flat_field_path else None),
            clahe_enabled=bool(clahe_enabled),
            clahe_clip=float(clahe_clip),
            clahe_grid=int(clahe_grid),
            splits=splits,
        )
        return prof

    # ------------------------------------------------------------------ #

    def save(self, path: str) -> str:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        # Atomic: a crash mid-write must not leave a half-parsed profile.
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False,
                           default_flow_style=False)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str) -> "DatasetProfile":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"profile not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__}      # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            print(f"[profile] ignoring unknown keys in {path}: "
                  f"{sorted(unknown)}")
        prof = cls(**{k: v for k, v in d.items() if k in known})
        prof.strides = [int(s) for s in prof.strides]
        # Missing keys silently take dataclass defaults, and prior_w = 0.0
        # would otherwise surface much later as math.log(0.0) inside
        # head._init_predictions (F28).
        if not (0.0 < float(prof.prior_w) < 1.0 and
                0.0 < float(prof.prior_h) < 1.0):
            raise RuntimeError(
                f"profile {path} has prior_w={prof.prior_w} "
                f"prior_h={prof.prior_h}; both must be in (0, 1). The file is "
                f"missing its priors or was written by an older version. "
                f"Regenerate it.")
        return prof

    def verify_fresh(self, root: Optional[str] = None,
                     allow_unfingerprinted: bool = False) -> None:
        """
        Recompute both fingerprints and compare.

        Geometry drift is a HARD ERROR; label drift is a WARNING. An EMPTY
        image_geometry_fingerprint is also a hard error by default: it used to
        pass verification unconditionally, so a profile with no fingerprint
        silently applied unverified priors.
        """
        r = root or self.root
        if not os.path.isdir(r):
            raise StaleProfileError(
                f"profile '{self.name}' points at root '{r}', which does not "
                f"exist. If this is the deployment host (which does not carry "
                f"the dataset), call apply(..., verify=False).")
        splits = list(self.splits.keys()) or ["train", "val", "test"]

        have = str(self.image_geometry_fingerprint or "")
        want = geometry_fingerprint(r, splits)
        if not have:
            if not allow_unfingerprinted:
                raise StaleProfileError(
                    f"profile '{self.name}' has an empty "
                    f"image_geometry_fingerprint; its priors cannot be "
                    f"verified. Regenerate the profile.")
        elif have != want:
            raise StaleProfileError(
                f"image geometry changed since profile '{self.name}' was "
                f"generated ({have[:16]} -> {want[:16]}). The letterbox "
                f"scale — and therefore prior_w ({self.prior_w}), prior_h "
                f"({self.prior_h}), and deploy_score_thresh — moved. "
                f"Regenerate:\n"
                f"  python dataset_profiles.py --root {r!r} "
                f"--img-h {self.img_h} --img-w {self.img_w} --out <path>")

        labels: List[str] = []
        for split in splits:
            try:
                img_dir, lbl_dir = resolve_split_dirs(r, split)
            except FileNotFoundError:
                continue
            labels += [label_path_for(p, lbl_dir)
                       for p in list_images(img_dir)]
        have_l = str(self.label_sha256 or "")
        want_l = label_fingerprint(labels, root=r)
        if have_l and have_l != want_l:
            print(f"[profile] WARNING label contents changed since this "
                  f"profile was generated ({have_l[:16]} -> {want_l[:16]}). "
                  f"prior_w/prior_h may no longer describe this dataset; "
                  f"regenerate when convenient.")

    def update_deploy_thresh(self, path: str, thresh: float) -> None:
        """Rewrite only deploy_score_thresh in an existing YAML."""
        self.deploy_score_thresh = float(thresh)
        self.save(path)

    # ------------------------------------------------------------------ #

    def apply(self, cfg: Config, verbose: bool = True,
              verify: bool = True) -> Config:
        """
        Push every derived value into the live config.

        ``verify=False`` skips verify_fresh(). Use it on machines that do not
        host the dataset (the Pi reads only deploy_score_thresh and the
        preprocessing parameters) and in unit tests with a synthetic root.
        The CANVAS fingerprint check is NEVER skipped: it costs no I/O and it
        is the one that silently biases the head's size-prior init.
        """
        check_canvas(cfg, self)
        # F23: validate priors against the reg-log clamp BEFORE writing any
        # value to cfg — a profile that cannot seed head.size_pred.bias must
        # never partially mutate the config. Ordered after check_canvas only
        # because math.log(0.0) on an empty default profile would raise
        # ValueError and mask the CanvasMismatchError that check_canvas owns.
        import math as _math
        from config import REG_LOG_CLAMP_MIN, REG_LOG_CLAMP_MAX
        for _label, _prior in (("prior_w", self.prior_w), ("prior_h", self.prior_h)):
            _lg = _math.log(float(_prior)) if float(_prior) > 0.0 else float("-inf")
            if not (REG_LOG_CLAMP_MIN < _lg < REG_LOG_CLAMP_MAX):
                raise RuntimeError(
                    f"profile '{self.name}': log({_label}) = {_lg:.3f} is outside "
                    f"the regression log-clamp ({REG_LOG_CLAMP_MIN}, "
                    f"{REG_LOG_CLAMP_MAX}). head.size_pred.bias would be clamped and "
                    f"TAL would cold-start with zero positives. "
                    f"Regenerate the profile at this canvas: "
                    f"python dataset_profiles.py --root <path> "
                    f"--img-h {self.img_h} --img-w {self.img_w} --out <out.yaml>")
        if verify:
            self.verify_fresh()

        cfg.data.root = str(self.root)
        cfg.data.profile_applied = True
        cfg.data.profile_name = str(self.name)

        cfg.data.n_train = int(self.n_train)
        cfg.data.n_val = int(self.n_val)
        cfg.data.n_test = int(self.n_test)
        cfg.data.n_train_boxes = int(self.n_train_boxes)
        # F08 (audit fix): transport the MEASURED zero-padding fraction so
        # validate_config's F21 letterbox-padding warning can actually fire.
        cfg.data.train_zero_pad_fraction = float(
            self.splits.get("train", {}).get("zero_pad_fraction", 1.0))

        cfg.model.prior_w = float(self.prior_w)
        cfg.model.prior_h = float(self.prior_h)

        if self.deploy_score_thresh is not None:
            cfg.eval.deploy_score_thresh = float(self.deploy_score_thresh)
        if self.baseline_map50 is not None:
            cfg.eval.baseline_map50 = float(self.baseline_map50)
        if self.baseline_source:
            cfg.eval.baseline_source = str(self.baseline_source)

        # ---- preprocessing contract ----
        cfg.aug.flat_field_path = (str(self.flat_field_path)
                                   if self.flat_field_path else None)
        cfg.aug.clahe_enabled = bool(self.clahe_enabled)
        cfg.aug.clahe_clip = float(self.clahe_clip)
        cfg.aug.clahe_grid = int(self.clahe_grid)

        if verbose:
            print(f"[profile] applied '{self.name}': "
                  f"{self.n_train}/{self.n_val}/{self.n_test} images, "
                  f"{self.n_train_boxes} train boxes"
                  f"{'' if verify else '  (verify=False)'}")
            print(f"[profile]   prior_w {cfg.model.prior_w:.6f}  "
                  f"prior_h {cfg.model.prior_h:.6f}  "
                  f"deploy_score_thresh {_fmt_thresh(cfg.eval.deploy_score_thresh)}")
            print(f"[profile]   preprocessing: clahe={cfg.aug.clahe_enabled} "
                  f"clip={cfg.aug.clahe_clip} grid={cfg.aug.clahe_grid} "
                  f"flat_field={cfg.aug.flat_field_path!r}")
            if self.label_sha256:
                print(f"[profile]   label_sha256 {self.label_sha256[:16]}")
        return cfg

    # ------------------------------------------------------------------ #

    def table(self) -> str:
        lines = [
            "=" * 76,
            f"  dataset profile: {self.name}",
            "=" * 76,
            f"  root       : {self.root}",
            f"  canvas     : {self.img_h}x{self.img_w}  strides {tuple(self.strides)}",
            f"  fingerprint: {self.canvas_fingerprint[:16]}",
            f"  labels     : {self.label_sha256[:16]}",
            f"  geometry   : {self.image_geometry_fingerprint[:16]}",
            f"  priors     : prior_w {self.prior_w:.6f}  prior_h {self.prior_h:.6f}"
            f"   (canvas-space medians, train split)",
            f"  deploy thr : {_fmt_thresh(self.deploy_score_thresh)}",
            f"  preprocess : clahe={self.clahe_enabled} clip={self.clahe_clip} "
            f"grid={self.clahe_grid} flat_field={self.flat_field_path!r}",
            "-" * 76,
        ]
        for split in ("train", "val", "test"):
            s = self.splits.get(split)
            if not s:
                continue
            lines += [
                f"  [{split}]  images {s['n_images']}  boxes {s['n_boxes']}  "
                f"empty {s['n_images_empty']}  "
                f"boxes/img {s['boxes_per_image_mean']:.2f}"
                f" +/- {s['boxes_per_image_std']:.2f} (max {s['boxes_per_image_max']})",
                "        w  " + "  ".join(
                    f"{k} {v:.4f}" for k, v in s["w_pct"].items()),
                "        h  " + "  ".join(
                    f"{k} {v:.4f}" for k, v in s["h_pct"].items()),
                "        ar " + "  ".join(
                    f"{k} {v:.3f}" for k, v in s["ar_pct"].items()),
                f"        size  small {s['n_small']}  medium {s['n_medium']}  "
                f"large {s['n_large']}   zero-pad frac "
                f"{s['zero_pad_fraction']:.3f}   sub-2px dropped "
                f"{s.get('n_boxes_dropped_subpixel', 0)}",
            ]
        lines.append("=" * 76)
        return "\n".join(lines)


# ===========================================================================
# canvas check
# ===========================================================================

def check_canvas(cfg: Config, profile: DatasetProfile) -> None:
    """
    Hard error on a canvas fingerprint mismatch.

    A profile generated at 384x640 and applied to a 512x288 config supplies
    priors in the wrong units, which shifts head.size_pred.bias, which can
    push the epoch-0 IoU low enough that TAL assigns nothing.
    """
    want = canvas_fingerprint(cfg.data.img_h, cfg.data.img_w,
                              cfg.model.strides)
    have = str(profile.canvas_fingerprint or "")
    if have == want:
        return
    raise CanvasMismatchError(
        f"dataset profile canvas fingerprint mismatch.\n"
        f"  profile : {profile.img_h}x{profile.img_w} strides "
        f"{tuple(profile.strides)}  -> {have[:16] or '<empty>'}\n"
        f"  config  : {cfg.data.img_h}x{cfg.data.img_w} strides "
        f"{tuple(cfg.model.strides)}  -> {want[:16]}\n"
        f"Every derived value in the profile is expressed in canvas units: "
        f"prior_w/prior_h, the width/height/aspect percentiles, the "
        f"small/medium/large buckets and deploy_score_thresh. Applying them "
        f"across a canvas change biases the head's size-prior init and can "
        f"cold-start TAL with zero positives. Regenerate with:\n"
        f"  python dataset_profiles.py --root {profile.root!r} "
        f"--img-h {cfg.data.img_h} --img-w {cfg.data.img_w} --out <path>")


def load_and_apply(path: str, cfg: Config, verbose: bool = True,
                   verify: bool = True) -> DatasetProfile:
    prof = DatasetProfile.load(path)
    prof.apply(cfg, verbose=verbose, verify=verify)
    return prof


# ===========================================================================
# CLI
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Profile a YOLO-format NIR pedestrian dataset and write "
                    "the derived-values YAML.")
    ap.add_argument("--root", required=True, help="dataset root to profile")
    ap.add_argument("--img-h", type=int, default=None)
    ap.add_argument("--img-w", type=int, default=None)
    ap.add_argument("--strides", type=int, nargs="+", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--baseline-map50", type=float, default=None,
                    help="optional reference mAP50 for this dataset")
    ap.add_argument("--baseline-source", default="",
                    help="provenance of --baseline-map50, printed verbatim by "
                         "evaluate.py (e.g. 'YOLO11n fine-tune, same splits')")
    ap.add_argument("--flat-field", default=None,
                    help="flat-field gain map; recorded in the profile so the "
                         "Pi applies the SAME preprocessing as training")
    ap.add_argument("--no-clahe", action="store_true",
                    help="record clahe_enabled=False in the profile")
    ap.add_argument("--clahe-clip", type=float, default=None)
    ap.add_argument("--clahe-grid", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="output path (default: datasets/<profile-name>.yaml)")
    args = ap.parse_args()

    cfg = get_config()
    root = args.root
    img_h = int(args.img_h or cfg.data.img_h)
    img_w = int(args.img_w or cfg.data.img_w)
    strides = tuple(args.strides) if args.strides else tuple(cfg.model.strides)

    prof = DatasetProfile.from_root(
        root, img_h, img_w, strides, name=args.name,
        deploy_score_thresh=(float(cfg.eval.deploy_score_thresh)
                             if cfg.eval.deploy_score_thresh is not None else None),
        baseline_map50=args.baseline_map50,
        baseline_source=args.baseline_source,
        flat_field_path=(args.flat_field
                         if args.flat_field is not None
                         else cfg.aug.flat_field_path),
        clahe_enabled=(not args.no_clahe) and bool(cfg.aug.clahe_enabled),
        clahe_clip=float(args.clahe_clip if args.clahe_clip is not None
                         else cfg.aug.clahe_clip),
        clahe_grid=int(args.clahe_grid if args.clahe_grid is not None
                       else cfg.aug.clahe_grid),
    )
    print(prof.table())
    out = args.out or os.path.join("datasets", f"{prof.name}.yaml")
    path = prof.save(out)
    print(f"\nwrote {path}")
    print("deploy_score_thresh is a FALLBACK until evaluate.py measures the "
          "best-F1 threshold on the report split and writes it back here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
