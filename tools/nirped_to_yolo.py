import json, os
from pathlib import Path
from collections import defaultdict

def convert(json_path, out_label_dir):
    Path(out_label_dir).mkdir(parents=True, exist_ok=True)
    with open(json_path) as f:
        data = json.load(f)

    img_map = {img["id"]: img for img in data["images"]}
    ann_map = defaultdict(list)
    for ann in data["annotations"]:
        ann_map[ann["image_id"]].append(ann)

    count = 0
    for img_id, img_info in img_map.items():
        W, H = img_info["width"], img_info["height"]
        fname = Path(img_info["file_name"]).stem
        lines = []
        for ann in ann_map[img_id]:
            x, y, w, h = ann["bbox"]
            cx = (x + w/2) / W
            cy = (y + h/2) / H
            bw = w / W
            bh = h / H
            if bw > 0 and bh > 0:
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(out_label_dir, fname + ".txt"), "w") as f:
            f.write("\n".join(lines))
        count += 1

    print(f"{json_path.split('/')[-1]}: {count} files converted")
    sample = os.listdir(out_label_dir)[0]
    with open(os.path.join(out_label_dir, sample)) as f:
        print(f"Sample ({sample}):", f.read()[:200])

BASE = r"C:\projects\nightvision\data\raw\miniNIRPed"
convert(f"{BASE}/labels/train_mini.json", f"{BASE}/labels/train")
convert(f"{BASE}/labels/val_mini.json",   f"{BASE}/labels/val")
convert(f"{BASE}/labels/test_mini.json",  f"{BASE}/labels/test")