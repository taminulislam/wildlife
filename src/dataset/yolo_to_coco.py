#!/usr/bin/env python3
"""
Convert the YOLO-format splits (data/dataset/yolo_v3) to COCO json annotations for
mmdetection training. One json per split; images stay where they are (abs paths not
needed - mmdet gets data_root + img prefix). Single class: deer (category_id 1).

Usage:
    python src/dataset/yolo_to_coco.py --root data/dataset/yolo_v3
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import cv2


def convert_split(root: str, split: str) -> dict:
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)
    images, anns = [], []
    ann_id = 1
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    W = H = None
    for img_id, ip in enumerate(paths, 1):
        # all frames share one FLIR geometry; read the first, reuse for the rest
        if W is None:
            H, W = cv2.imread(ip).shape[:2]
        name = os.path.basename(ip)
        images.append({"id": img_id, "file_name": name, "width": W, "height": H})
        lp = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        if not os.path.isfile(lp):
            continue
        with open(lp) as fh:
            for ln in fh:
                p = ln.split()
                if len(p) < 5:
                    continue
                _, xc, yc, bw, bh = (float(x) for x in p[:5])
                x = (xc - bw / 2) * W; y = (yc - bh / 2) * H
                ww = bw * W; hh = bh * H
                anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                             "bbox": [x, y, ww, hh], "area": ww * hh, "iscrowd": 0})
                ann_id += 1
    return {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "deer"}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset/yolo_v3")
    args = ap.parse_args()
    out_dir = os.path.join(args.root, "coco_annotations")
    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        d = convert_split(args.root, split)
        out = os.path.join(out_dir, f"{split}.json")
        with open(out, "w") as f:
            json.dump(d, f)
        print(f"{split}: {len(d['images'])} images, {len(d['annotations'])} boxes -> {out}")


if __name__ == "__main__":
    main()
