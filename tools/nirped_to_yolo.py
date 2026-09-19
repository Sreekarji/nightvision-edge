"""
tools/nirped_coco_to_yolo.py
Convert full NIRPed val.json to YOLO txt labels.

Filters:
  - category_id 1 (Ped) and 2 (Peo) -> class 0
  - Ign == 1 -> dropped
  - bbox w or h <= 0 -> dropped (degenerate)
"""

import json
import os
from collections import defaultdict
from pathlib import Path


def convert(json_path: str, out_label_dir: str) -> None:
    Path(out_label_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading {json_path} ...")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_map = {img["id"]: img for img in data["images"]}

    ann_map = defaultdict(list)
    n_total = 0
    n_dropped_ign = 0
    n_dropped_cat = 0
    n_dropped_degen = 0

    for ann in data["annotations"]:
        n_total += 1
        if ann.get("Ign", 0) == 1:
            n_dropped_ign += 1
            continue
        if ann["category_id"] not in (1, 2):
            n_dropped_cat += 1
            continue
        ann_map[ann["image_id"]].append(ann)

    print(f"Annotations: {n_total} total  "
          f"{n_dropped_ign} dropped Ign=1  "
          f"{n_dropped_cat} dropped wrong category  "
          f"{n_total - n_dropped_ign - n_dropped_cat} kept")

    n_files = 0
    n_empty = 0
    n_boxes_written = 0

    for img_id, img_info in img_map.items():
        W = img_info["width"]
        H = img_info["height"]
        fname = Path(img_info["file_name"]).stem
        out_path = os.path.join(out_label_dir, fname + ".txt")

        lines = []
        for ann in ann_map.get(img_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                n_dropped_degen += 1
                continue
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            bw = w / W
            bh = h / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        n_files += 1
        if not lines:
            n_empty += 1
        else:
            n_boxes_written += len(lines)

    print(f"Written: {n_files} label files  "
          f"{n_empty} empty (no annotations)  "
          f"{n_dropped_degen} degenerate boxes dropped  "
          f"{n_boxes_written} boxes total")

    # Spot-check: print first non-empty label
    for img_id, img_info in img_map.items():
        if ann_map.get(img_id):
            fname = Path(img_info["file_name"]).stem
            sample_path = os.path.join(out_label_dir, fname + ".txt")
            with open(sample_path, encoding="utf-8") as f:
                content = f.read()
            print(f"Sample ({fname}.txt):\n{content[:300]}")
            break


if __name__ == "__main__":
    JSON_PATH     = r"C:\projects\nightvision\val\labels\val.json"
    OUT_LABEL_DIR = r"C:\projects\nightvision\val\labels"

    convert(JSON_PATH, OUT_LABEL_DIR)