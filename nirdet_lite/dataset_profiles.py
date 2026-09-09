#!/usr/bin/env python
"""
dataset_profiles.py — dataset profiles with content fingerprinting
====================================================================
Run this BEFORE switching to any new dataset:

    python dataset_profiles.py --root <dataset root> --out datasets/<name>.yaml
    python train.py    --profile datasets/<name>.yaml
    python evaluate.py --profile datasets/<name>.yaml   # writes deploy_score_thresh back

WHY THIS EXISTS
---------------
cfg.model.prior_w / prior_h are baked into the head's regression bias
(head._init_predictions) and cfg.data.root decides which labels are read.
Switch to a new dataset and forget to update any of those, and an entire
run trains against the wrong box size with no error message anywhere.
The profile makes that class of failure impossible:

  * build(root) scans the label directory: n_train, n_val, prior_w /
    prior_h (median normalised box w/h) and diagnostics (boxes per image,
    empty-label fraction, w/h percentiles).
  * A SHA-256 fingerprint is computed over the exact byte contents of
    every label file. Images are excluded from it on purpose: they can be
    re-encoded without changing box statistics.
  * verify_fresh(root) recomputes everything from disk and raises
    StaleProfileError naming the exact derived fields that moved.
  * apply(profile, cfg) overrides ONLY cfg.data.root, cfg.data.n_train,
    cfg.data.n_val, cfg.model.prior_w, cfg.model.prior_h — nothing else —
    and train.py still runs validate_config(cfg) afterwards.

CANVAS-SPACE PRIORS (important)
-------------------------------
prior_w / prior_h feed the head's regression bias, which predicts boxes on
the letterboxed (img_h x img_w) canvas — NOT in raw label-file
coordinates. The letterbox rescales every box (on this dataset 1280x720
-> 640x384 multiplies normalised heights by 720*scale/384 = 0.9375), so
the profile derives priors by applying the same arithmetic as
dataset.letterbox_boxes() to every train box:

    w_canvas = w_raw * src_w * scale / img_w
    h_canvas = h_raw * src_h * scale / img_h,   scale = min(img_w/src_w, img_h/src_h)

This reproduces the established config.py values (canvas medians
w=0.046094 / h=0.167969 ~= the hand-set 0.0461 / 0.1680); a raw-label
median would silently shift prior_h to 0.179167 on the first profiled
run. Source image dimensions are read from image headers (Pillow opens
lazily — no pixel decode). Raw-label medians are kept under stats for
reference.

Priors are derived IMAGE-DRIVEN, exactly like dataset.NIRDetDataset:
iterate images/<split>/, read labels/<split>/<stem>.txt per image (a
missing label file is a negative sample, an image-less label file is
invisible to training). check_canvas() refuses a canvas mismatch, since
canvas-space priors do not transfer between input resolutions.

Because image DIMENSIONS change canvas-space priors while leaving label
bytes untouched, derived also carries an image_geometry_fingerprint: a
SHA-256 over (image relative path, width, height) of every image in every
labelled split. Re-encoding an image at the same size still passes
verify_fresh; resizing one does not.

The `derived:` section of the profile YAML is machine-generated and must
not be hand-edited. That is not enforced by a comment: verify_fresh()
recomputes every value from the label files and image headers, so a hand
edit shows up as a stored != recomputed diff and training refuses to start.

deploy_score_thresh is deliberately NOT derived — it requires a training
run to know. evaluate.py writes the best-F1 threshold back into the
profile (DatasetProfile.record_deploy_threshold) after each full training
+ evaluation cycle, so it is preserved for the next export.

Requires: PyYAML, numpy, Pillow (all in the project venv). This module
deliberately does NOT import dataset.py: profiles must be usable without
torch / albumentations installed. The letterbox arithmetic is duplicated
from dataset.letterbox_boxes by design — if that ever changes, mirror the
change here or verify_fresh() will flag every profile as stale.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

__all__ = ["DatasetProfile", "StaleProfileError", "scan",
           "verify_fresh", "apply"]

# Must mirror dataset.IMG_EXTS globs (the .pickle sidecar files in this
# dataset are invisible to training and are ignored here too).
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Canvas default, matching config.DataCfg.img_h / img_w. Kept as constants
# so this module stays importable without config.py.
DEFAULT_IMG_H, DEFAULT_IMG_W = 384, 640

_REQUIRED_DERIVED = ("n_train", "n_val", "prior_w", "prior_h",
                     "label_fingerprint", "image_geometry_fingerprint")

# Stored values are rounded to 6 decimals and YAML round-trips floats via
# repr, so a legitimate reload is bit-identical far below this tolerance.
_FLOAT_TOL = 1e-9


class StaleProfileError(RuntimeError):
    """The profile's derived section no longer matches the labels on disk."""


# ===========================================================================
# label + image scanning
# ===========================================================================

def _parse_boxes(path: Path) -> np.ndarray:
    """YOLO lines "cls cx cy w h" -> (N, 4) float64 of (cx, cy, w, h).

    Mirrors dataset.NIRDetDataset._read_labels exactly (skip malformed
    lines, clamp coordinates the same way) so the statistics describe what
    training actually sees. If that method ever changes its tolerance,
    mirror the change here or the profile will fingerprint statistics
    training does not consume.
    """
    boxes: List[Tuple[float, float, float, float]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return np.zeros((0, 4), np.float64)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            int(float(parts[0]))                 # class must parse, as in dataset.py
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append((min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0),
                      min(max(w, 1e-6), 1.0), min(max(h, 1e-6), 1.0)))
    if not boxes:
        return np.zeros((0, 4), np.float64)
    return np.asarray(boxes, np.float64)


def _label_files(root: Path) -> List[Path]:
    """Every .txt under <root>/labels, sorted for a deterministic stream."""
    lbl = root / "labels"
    if not lbl.is_dir():
        raise FileNotFoundError(f"no labels/ directory under {root}")
    files = sorted(p for p in lbl.rglob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no .txt label files under {lbl}")
    for f in files:
        if len(f.relative_to(lbl).parts) < 2:
            raise ValueError(
                f"expected labels/<split>/*.txt layout, found {f} directly "
                f"under labels/")
    return files


def _image_files(root: Path, split: str) -> List[Path]:
    """Images of one split, sorted — exactly dataset.NIRDetDataset's listing
    (glob per extension in IMG_EXTS order; .pickle sidecars excluded)."""
    img_dir = root / "images" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"no images/{split} directory under {root}")
    out: List[Path] = []
    for ext in IMG_EXTS:
        out.extend(sorted(img_dir.glob("*" + ext)))
    if not out:
        raise FileNotFoundError(f"no images under {img_dir}")
    return sorted(out)


def _label_for(root: Path, image: Path) -> Path:
    """The label file dataset.py would read for this image (stem pairing;
    a missing file simply means 'no boxes' — a negative sample)."""
    split = image.relative_to(root / "images").parts[0]
    return root / "labels" / split / (image.stem + ".txt")


def _read_image_size(path: Path) -> Tuple[int, int]:
    """(width, height) from the image header. Pillow opens lazily, so no
    pixel decode happens — this is cheap even on hundreds of files."""
    from PIL import Image                    # deferred: only needed for scan
    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


def _fingerprint(root: Path, files: List[Path]) -> str:
    """SHA-256 over the exact byte contents of every label file.

    The path relative to <root> and the file size are length-prefixed into
    the hash, so renames cannot alias the old stream and content
    boundaries cannot shift. Even a line-ending rewrite counts as a change
    — regenerating the profile is cheap, a silent stale prior is not.
    """
    h = hashlib.sha256()
    for f in files:
        rel = f.relative_to(root).as_posix().encode("utf-8")
        data = f.read_bytes()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def _geometry_fingerprint(root: Path, images: List[Path],
                          sizes: Dict[Path, Tuple[int, int]]) -> str:
    """SHA-256 over (image rel path, width, height) for every image in
    every labelled split.

    Image BYTES are deliberately not hashed (re-encoding must stay free),
    but dimensions cannot be: they set the letterbox scale, so they move
    the canvas-space priors while the label bytes — and therefore the
    label fingerprint — stay untouched.
    """
    h = hashlib.sha256()
    for img in images:                        # sorted, deterministic
        w, hh = sizes[img]
        rel = img.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(w.to_bytes(4, "big"))
        h.update(hh.to_bytes(4, "big"))
    return h.hexdigest()


def _percentiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}
    qs = (1, 5, 25, 50, 75, 95, 99)
    out = {"min": round(float(values.min()), 6),
           "max": round(float(values.max()), 6)}
    for q, v in zip(qs, np.percentile(values, qs)):
        out[f"p{q}"] = round(float(v), 6)
    return out


def scan(root, img_h: int = DEFAULT_IMG_H,
         img_w: int = DEFAULT_IMG_W) -> Dict:
    """Scan the label directory (+ image headers) -> the derived payload.

    n_train / n_val count label files per split (spec: scan the label
    directory). The priors and diagnostics come from the TRAIN split and
    are derived IMAGE-DRIVEN, exactly like dataset.NIRDetDataset: iterate
    images/train/, read labels/train/<stem>.txt per image. Both
    fingerprints cover every split, so val/test edits are caught too.

    prior_w / prior_h are medians over CANVAS-space boxes (see the module
    docstring): each train box is mapped through the same letterbox
    arithmetic dataset.py applies, using the source image's native size.
    """
    root = Path(root)
    img_h, img_w = int(img_h), int(img_w)
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"canvas size must be positive, got {img_h}x{img_w}")

    files = _label_files(root)
    by_split: Dict[str, List[Path]] = {}
    for f in files:
        split = f.relative_to(root / "labels").parts[0]
        by_split.setdefault(split, []).append(f)
    splits = sorted(by_split)
    if "train" not in by_split:
        raise FileNotFoundError(f"no labels/train split under {root}/labels "
                                f"(found: {splits})")
    n_train = len(by_split["train"])
    n_val = len(by_split.get("val", []))

    # All images of every labelled split: drives priors (train) and the
    # geometry fingerprint (all splits).
    images_by_split = {s: _image_files(root, s) for s in splits}
    train_imgs = images_by_split["train"]
    all_images = [img for s in splits for img in images_by_split[s]]
    sizes = {img: _read_image_size(img) for img in all_images}

    counts = np.zeros(len(train_imgs), np.int64)
    raw_w: List[float] = []
    raw_h: List[float] = []
    canvas_w: List[float] = []
    canvas_h: List[float] = []
    for i, img in enumerate(train_imgs):
        b = _parse_boxes(_label_for(root, img))
        counts[i] = len(b)
        if b.shape[0] == 0:
            continue
        src_w, src_h = sizes[img]
        scale = min(img_w / float(src_w), img_h / float(src_h))
        for cx, cy, w, h in b:
            raw_w.append(w)
            raw_h.append(h)
            canvas_w.append(w * src_w * scale / img_w)   # dataset.letterbox_boxes
            canvas_h.append(h * src_h * scale / img_h)
    cw = np.asarray(canvas_w, np.float64)
    ch = np.asarray(canvas_h, np.float64)
    rw = np.asarray(raw_w, np.float64)
    rh = np.asarray(raw_h, np.float64)
    if cw.size == 0:
        raise ValueError(f"cannot derive priors: no boxes under "
                        f"{root}/labels/train pair with images under "
                        f"{root}/images/train")

    return {
        "img_h": img_h,
        "img_w": img_w,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_train_images": int(len(train_imgs)),
        "prior_w": round(float(np.median(cw)), 6),
        "prior_h": round(float(np.median(ch)), 6),
        "label_fingerprint": "sha256:" + _fingerprint(root, files),
        "image_geometry_fingerprint": "sha256:" + _geometry_fingerprint(
            root, all_images, sizes),
        "stats": {
            "n_label_files_by_split": {s: len(by_split[s]) for s in splits},
            "n_images_by_split": {s: len(images_by_split[s]) for s in splits},
            "coordinate_space": f"letterbox canvas ({img_h}x{img_w})",
            "train": {
                "n_boxes": int(cw.size),
                "boxes_per_image": {
                    "mean": round(float(counts.mean()), 4),
                    "median": round(float(np.median(counts)), 4),
                    "max": int(counts.max()),
                },
                "empty_label_fraction": round(
                    float((counts == 0).sum()) / len(train_imgs), 6),
                # raw-label medians (image-independent) — the label
                # fingerprint covers exactly these; canvas values
                # additionally depend on image geometry.
                "raw_median_w": round(float(np.median(rw)), 6),
                "raw_median_h": round(float(np.median(rh)), 6),
                "width_percentiles": _percentiles(cw),
                "height_percentiles": _percentiles(ch),
            },
        },
    }


# ===========================================================================
# derived-section diffing (verify_fresh names the exact fields that moved)
# ===========================================================================

def _short(v) -> str:
    s = str(v)
    return s if len(s) <= 24 else s[:14] + "..." + s[-6:]


def _diff_dict(path: str, stored, current, out: Optional[List[str]] = None
               ) -> List[str]:
    """Recursive stored-vs-recomputed diff; floats via _FLOAT_TOL."""
    out = [] if out is None else out
    if isinstance(stored, dict) and isinstance(current, dict):
        for k in sorted(set(stored) | set(current)):
            p = f"{path}.{k}"
            if k not in stored:
                out.append(f"{p}: <absent> -> {_short(current[k])} (added)")
            elif k not in current:
                out.append(f"{p}: {_short(stored[k])} -> <absent> (removed)")
            else:
                _diff_dict(p, stored[k], current[k], out)
        return out
    if isinstance(stored, bool) or isinstance(current, bool):
        if stored != current:
            out.append(f"{path}: {stored} -> {current}")
        return out
    if isinstance(stored, (int, float)) and isinstance(current, (int, float)):
        if not math.isclose(float(stored), float(current),
                            rel_tol=0.0, abs_tol=_FLOAT_TOL):
            out.append(f"{path}: {stored} -> {current}")
        return out
    if stored != current:
        out.append(f"{path}: {_short(stored)} -> {_short(current)}")
    return out


# ===========================================================================
# DatasetProfile
# ===========================================================================

class DatasetProfile:
    """
    Profile YAML layout (machine-managed except deploy_score_thresh):

        name: miniNIRPed_261
        root: C:/projects/nightvision/data/raw/miniNIRPed
        generated_utc: "2026-09-07T...+00:00"
        derived:                      # NEVER hand-edit; recomputed by scan()
          img_h / img_w               # canvas the priors were derived at
          n_train / n_val              # label-file counts per split
          prior_w / prior_h           # median canvas-space box w/h (train)
          label_fingerprint           # sha256 over label file bytes
          image_geometry_fingerprint  # sha256 over image paths + dimensions
          stats: {...}
        deploy_score_thresh: null     # written by evaluate.py after a full cycle
    """

    _HEADER = (
        "NIRDet-Lite dataset profile. The 'derived:' section is machine-",
        "generated from the label file bytes and the labelled images'",
        "dimensions — never hand-edit it; verify_fresh() detects edits and",
        "training refuses to run against a stale profile. Regenerate with:",
        "    python dataset_profiles.py --root <root> --out <this file>",
        "",
        "prior_w / prior_h are medians over LETTERBOX-CANVAS boxes (see the",
        "dataset_profiles.py docstring), which is what the head consumes.",
        "",
        "deploy_score_thresh is NOT derived (it requires a training run);",
        "evaluate.py writes the best-F1 threshold back here after each full",
        "training + evaluation cycle.",
    )

    def __init__(self, data: Dict, path=None) -> None:
        missing = [k for k in ("name", "root", "derived") if k not in data]
        if missing:
            raise ValueError(f"profile is missing top-level keys {missing}")
        miss = [k for k in _REQUIRED_DERIVED if k not in data["derived"]]
        if miss:
            raise ValueError(f"profile derived section is missing {miss}")
        self.data = data
        self.path = Path(path) if path is not None else None

    # -------------------------------------------------------------- build

    @classmethod
    def build(cls, root, name: Optional[str] = None,
              img_h: int = DEFAULT_IMG_H,
              img_w: int = DEFAULT_IMG_W) -> "DatasetProfile":
        """Scan <root>/labels (+ image headers) and derive everything."""
        root = Path(root).resolve()
        derived = scan(root, img_h=img_h, img_w=img_w)
        if name is None:
            name = f"{root.name}_{derived['n_train']}"
        return cls({
            "name": name,
            "root": root.as_posix(),
            "generated_utc": datetime.now(timezone.utc)
                                     .isoformat(timespec="seconds"),
            "derived": derived,
            "deploy_score_thresh": None,
        })

    @classmethod
    def load(cls, path) -> "DatasetProfile":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no dataset profile at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a profile YAML mapping")
        return cls(data, path=path)

    # ------------------------------------------------------------ accessors

    @property
    def root(self) -> str:
        return str(self.data["root"])

    @property
    def derived(self) -> Dict:
        return self.data["derived"]

    def _whence(self) -> str:
        return (str(self.path) if self.path is not None
                else str(self.data.get("name", "<unnamed>")))

    # ---------------------------------------------------------- verify_fresh

    def verify_fresh(self, root=None) -> None:
        """Recompute everything from disk; raise StaleProfileError on drift.

        With no argument the profile's own recorded root is checked, so
        ``profile.verify_fresh()`` in train.py validates the exact dataset
        the profile was generated from. The error message names every
        derived field that changed, fingerprint lines first, and
        distinguishes label-content drift from image-geometry drift from
        hand-edits of the stored values.
        """
        check_root = Path(root) if root is not None else Path(self.root)
        if not check_root.is_dir():
            raise StaleProfileError(
                f"dataset profile '{self._whence()}' points at root "
                f"'{self.root}' which does not exist")
        d = self.derived
        img_h = int(d.get("img_h", DEFAULT_IMG_H))
        img_w = int(d.get("img_w", DEFAULT_IMG_W))
        try:
            current = scan(check_root, img_h=img_h, img_w=img_w)
        except (FileNotFoundError, ValueError) as exc:
            raise StaleProfileError(
                f"dataset profile '{self._whence()}' can no longer be "
                f"verified against '{check_root.as_posix()}': {exc}"
            ) from exc

        stored = d
        fp_changed = (stored.get("label_fingerprint")
                      != current.get("label_fingerprint"))
        geo_changed = (stored.get("image_geometry_fingerprint")
                       != current.get("image_geometry_fingerprint"))
        # both fingerprints get their own annotated lines; diff every other
        # value so the exact moved fields are named
        skip = ("label_fingerprint", "image_geometry_fingerprint")
        stored_rest = {k: v for k, v in stored.items() if k not in skip}
        current_rest = {k: v for k, v in current.items() if k not in skip}
        diffs = _diff_dict("derived", stored_rest, current_rest)
        if not diffs and not fp_changed and not geo_changed:
            return

        lines = [f"dataset profile '{self._whence()}' is stale for root "
                 f"'{check_root.as_posix()}':"]
        if fp_changed:
            lines.append(
                f"  derived.label_fingerprint: "
                f"{_short(stored.get('label_fingerprint'))} -> "
                f"{_short(current.get('label_fingerprint'))}"
                f"  (label file contents changed)")
        if geo_changed:
            lines.append(
                f"  derived.image_geometry_fingerprint: "
                f"{_short(stored.get('image_geometry_fingerprint'))} -> "
                f"{_short(current.get('image_geometry_fingerprint'))}"
                f"  (image dimensions changed; label bytes unchanged — "
                f"canvas-space priors moved)")
        lines.extend(f"  {line}" for line in diffs)
        if not diffs and (fp_changed or geo_changed):
            lines.append("  (contents changed but no derived statistic "
                         "moved — regenerate anyway)")
        if diffs and not fp_changed and not geo_changed:
            lines.append("  (both fingerprints match, so the stored derived "
                         "values were hand-edited — regenerate)")
        out = (str(self.path) if self.path is not None
               else f"datasets/{self.data.get('name', '<name>')}.yaml")
        lines.append("Regenerate before training:")
        lines.append(f"  python dataset_profiles.py --root "
                     f"{check_root.as_posix()} --out {out}")
        lines.append("Do NOT hand-edit the derived: section; it is "
                     "recomputed from the label files.")
        raise StaleProfileError("\n".join(lines))

    # --------------------------------------------------------------- apply

    def apply(self, cfg) -> None:
        """Override ONLY these five config fields — nothing else:

            cfg.data.root, cfg.data.n_train, cfg.data.n_val,
            cfg.model.prior_w, cfg.model.prior_h

        validate_config(cfg) still runs afterwards in train.py, so range
        checks (e.g. priors normalised into (0, 1)) still apply.
        """
        d = self.derived
        cfg.data.root = self.root
        cfg.data.n_train = int(d["n_train"])
        cfg.data.n_val = int(d["n_val"])
        cfg.model.prior_w = float(d["prior_w"])
        cfg.model.prior_h = float(d["prior_h"])
        print(f"[profile] {self.data['name']} applied -> "
              f"root={self.root} n_train={cfg.data.n_train} "
              f"n_val={cfg.data.n_val} prior_w={cfg.model.prior_w} "
              f"prior_h={cfg.model.prior_h}")

    def check_canvas(self, cfg) -> None:
        """Priors are canvas-space, so training must happen at the canvas
        the profile was derived at. Call AFTER all img-size overrides are
        applied to cfg. Read-only: overrides nothing.
        """
        want = (int(self.derived["img_h"]), int(self.derived["img_w"]))
        got = (int(cfg.data.img_h), int(cfg.data.img_w))
        if got != want:
            raise RuntimeError(
                f"canvas mismatch with profile '{self._whence()}': priors "
                f"were derived at {want[0]}x{want[1]} but training would "
                f"run at {got[0]}x{got[1]}. Canvas-space priors do not "
                f"transfer. Regenerate the profile at {got[0]}x{got[1]}: "
                f"python dataset_profiles.py --root {self.root} "
                f"--img-h {got[0]} --img-w {got[1]} --out "
                f"{self.path if self.path is not None else '<profile>'}")

    # ------------------------------------------------- deploy_score_thresh

    def record_deploy_threshold(self, score_thresh: float,
                                path=None) -> Path:
        """evaluate.py writes the best-F1 threshold back after a full
        training + evaluation cycle. Refuses to touch a stale profile: a
        threshold measured against labels the model never trained on must
        not be preserved for the next export.
        """
        t = float(score_thresh)
        if not math.isfinite(t) or not (0.0 < t < 1.0):
            raise ValueError(f"deploy_score_thresh must be finite in (0, 1), "
                             f"got {score_thresh!r}")
        self.verify_fresh()
        self.data["deploy_score_thresh"] = round(t, 4)
        out = self.save(path)
        print(f"[profile] deploy_score_thresh={round(t, 4)} -> {out}")
        return out

    # --------------------------------------------------------------- save

    def save(self, path=None) -> Path:
        out = Path(path) if path is not None else self.path
        if out is None:
            raise ValueError("no path known: pass path= or load() a file")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump(self.data, sort_keys=False,
                              default_flow_style=False, width=100,
                              allow_unicode=True)
        header = "\n".join("# " + line for line in self._HEADER)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(header + "\n\n" + body, encoding="utf-8")
        tmp.replace(out)                       # atomic, like checkpoints
        return out


# ===========================================================================
# module-level forms (the class methods satisfy the same contracts)
# ===========================================================================

def verify_fresh(profile: DatasetProfile, root=None) -> None:
    """verify_fresh(profile, root): recompute and raise on drift."""
    return profile.verify_fresh(root)


def apply(profile: DatasetProfile, cfg) -> None:
    """apply(profile, cfg): same as profile.apply(cfg)."""
    return profile.apply(cfg)


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        "dataset_profiles.py — generate a NIRDet-Lite dataset profile")
    p.add_argument("--root", required=True,
                   help="dataset root containing labels/<split>/*.txt")
    p.add_argument("--out", required=True, help="profile YAML to write")
    p.add_argument("--name", default=None,
                   help="profile name (default: <root dir>_<n_train>)")
    p.add_argument("--img-h", type=int, default=DEFAULT_IMG_H,
                   help=f"canvas height priors are derived at "
                        f"(default {DEFAULT_IMG_H}, = config.DataCfg.img_h)")
    p.add_argument("--img-w", type=int, default=DEFAULT_IMG_W,
                   help=f"canvas width priors are derived at "
                        f"(default {DEFAULT_IMG_W}, = config.DataCfg.img_w)")
    args = p.parse_args(argv)

    prof = DatasetProfile.build(args.root, name=args.name,
                                img_h=args.img_h, img_w=args.img_w)
    out = Path(args.out)

    # Regeneration with unchanged labels AND unchanged image geometry
    # keeps the measured deploy threshold; either changing resets it
    # (stale by definition — it needs a new training run).
    if out.exists():
        try:
            old = DatasetProfile.load(out)
            same = (old.derived.get("label_fingerprint")
                    == prof.derived["label_fingerprint"]
                    and old.derived.get("image_geometry_fingerprint")
                    == prof.derived["image_geometry_fingerprint"])
            if same:
                keep = old.data.get("deploy_score_thresh")
                prof.data["deploy_score_thresh"] = keep
                print("labels + image geometry unchanged: deploy_score_thresh "
                      + (f"preserved ({keep})" if keep is not None
                         else "still null"))
            else:
                print("labels and/or image geometry changed: "
                      "deploy_score_thresh reset to null — re-derive with "
                      "evaluate.py after the next training run")
        except (ValueError, FileNotFoundError) as exc:
            print(f"! existing {out} unreadable ({exc}); writing fresh")

    path = prof.save(out)
    d = prof.derived
    s = d["stats"]["train"]
    print("=" * 62)
    print(f"  profile     : {path}")
    print(f"  root        : {prof.root}")
    print(f"  canvas      : {d['img_h']}x{d['img_w']} (priors are canvas-space)")
    print(f"  n_train     : {d['n_train']} labels / {d['n_train_images']} images"
          f"    n_val: {d['n_val']}")
    print(f"  prior_w     : {d['prior_w']}   (median canvas box width)")
    print(f"  prior_h     : {d['prior_h']}   (median canvas box height)")
    print(f"  raw medians : w={s['raw_median_w']} h={s['raw_median_h']} "
          f"(label-file space, reference only)")
    print(f"  train boxes : {s['n_boxes']}  "
          f"({s['boxes_per_image']['mean']:.2f}/img, "
          f"max {s['boxes_per_image']['max']}, "
          f"empty {s['empty_label_fraction']:.1%})")
    print(f"  w p5/p50/p95: {s['width_percentiles']['p5']} / "
          f"{s['width_percentiles']['p50']} / "
          f"{s['width_percentiles']['p95']}")
    print(f"  h p5/p50/p95: {s['height_percentiles']['p5']} / "
          f"{s['height_percentiles']['p50']} / "
          f"{s['height_percentiles']['p95']}")
    print(f"  labels sha  : {d['label_fingerprint']}")
    print(f"  images geom : {d['image_geometry_fingerprint']}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
