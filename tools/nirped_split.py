# tools/nirped_split.py
"""
Physically organise the flat NIRPed val dump into train / val / test splits
that resolve_split_dirs() can find.

SOURCE (flat):
  <root>/images/<filename>.png          (or wherever images sit)
  <root>/labels/<filename>.txt

TARGET layout (option A — simpler):
  <root>/images/train/<filename>.png
  <root>/labels/train/<filename>.txt
  <root>/images/val/<filename>.png
  <root>/labels/val/<filename>.txt
  <root>/images/test/<filename>.png
  <root>/labels/test/<filename>.txt

Usage:
  python tools/nirped_split.py \
      --json   val/labels/val.json \
      --images val/images \
      --labels val/labels \
      --root   val
"""

import argparse
import json
import os
import shutil
from pathlib import Path

TRAIN = {20181220193,20181220194,20181220202,20181220203,20181220204,
         20181220205,20181220210,20190113190,20190113191,20190113192,
         20190113193,20190113194,20190113195,20190113200,20190113201}
VAL   = {20181220192, 20190113185, 20181219200}
TEST  = {20180710193, 20190508204, 20200630194}


def rec_to_split(rid: int) -> str:
    if rid in TRAIN: return "train"
    if rid in VAL:   return "val"
    if rid in TEST:  return "test"
    return "unused"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",   required=True, help="path to val.json")
    ap.add_argument("--images", required=True, help="flat images dir")
    ap.add_argument("--labels", required=True, help="flat labels dir (contains val.json)")
    ap.add_argument("--root",   required=True, help="dataset root (images/ and labels/ live here)")
    ap.add_argument("--move",   action="store_true",
                    help="move files instead of copying (saves disk space)")
    args = ap.parse_args()

    op = shutil.move if args.move else shutil.copy2

    with open(args.json) as f:
        data = json.load(f)

    img_to_rec = {img["id"]: img["recordings_id"] for img in data["images"]}
    fname_to_rec = {img["file_name"]: img["recordings_id"] for img in data["images"]}

    root = Path(args.root)
    img_src = Path(args.images)
    lbl_src = Path(args.labels)

    # Create destination dirs
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0, "unused": 0, "missing_img": 0, "missing_lbl": 0}

    for fname, rid in fname_to_rec.items():
        split = rec_to_split(rid)
        if split == "unused":
            counts["unused"] += 1
            continue

        img_file = img_src / fname
        # label stem: strip extension
        stem = Path(fname).stem
        lbl_file = lbl_src / (stem + ".txt")

        if not img_file.exists():
            counts["missing_img"] += 1
            continue

        dst_img = root / "images" / split / fname
        dst_lbl = root / "labels" / split / (stem + ".txt")

        if not dst_img.exists():
            op(str(img_file), str(dst_img))
        if lbl_file.exists() and not dst_lbl.exists():
            op(str(lbl_file), str(dst_lbl))
        elif not lbl_file.exists():
            counts["missing_lbl"] += 1

        counts[split] += 1

    print(f"Done.")
    for k, v in counts.items():
        print(f"  {k:12s}: {v}")
    print(f"\nLayout written to {root}/images/{{train,val,test}}/ and {root}/labels/{{train,val,test}}/")
    print(f"\nNext:")
    print(f"  python dataset_profiles.py --root {args.root} --out datasets/NIRPed.yaml")


if __name__ == "__main__":
    main()