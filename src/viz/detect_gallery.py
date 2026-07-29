#!/usr/bin/env python3
"""
Detection output images for the paper / team review.

Renders the best detector's predictions on held-out TEST frames with the confidence
printed on every box. Frames are chosen by DEER SIZE so the gallery is readable:
mostly large and medium animals (where a reader can actually see what was detected),
plus a few small ones to show the hard regime honestly.

Size buckets use sqrt(box area) in pixels, on the GT boxes of the test split:
    large  >= 55 px      medium 30-55 px      small < 30 px
(The corpus median is ~27 px, so "large" here means large *for this dataset*.)

Each output frame shows:
  * green box + `deer 0.87` for every prediction above --conf
  * a thin blue outline for the human GT box, so over/under-detection is visible
  * a caption strip: video, frame index, #predictions vs #GT

Also writes a contact sheet per bucket for quick scanning.

Usage:
  python src/viz/detect_gallery.py --weights <best.pt> --imgsz 1280 \
      --out results/viz/detect_yolov9m1280
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import random
import sys

import cv2
import numpy as np
import yaml

GREEN = (80, 220, 80)
BLUE = (235, 180, 60)
YELLOW = (60, 230, 240)


def load_split(data_yaml: str, split: str):
    d = yaml.safe_load(open(data_yaml))
    root = d["path"]
    img_dir = os.path.join(root, d.get(split, f"images/{split}"))
    lbl_dir = img_dir.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    return sorted(glob.glob(os.path.join(img_dir, "*.png"))), lbl_dir


def gt_boxes(lbl, W, H):
    out = []
    if not os.path.isfile(lbl):
        return out
    for ln in open(lbl):
        p = ln.split()
        if len(p) < 5:
            continue
        _, xc, yc, bw, bh = (float(v) for v in p[:5])
        out.append([(xc - bw / 2) * W, (yc - bh / 2) * H,
                    (xc + bw / 2) * W, (yc + bh / 2) * H])
    return out


def bucket_of(boxes) -> str | None:
    """Bucket a frame by its LARGEST deer — that is what a reader will look at."""
    if not boxes:
        return None
    s = max(math.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1)) for b in boxes)
    return "large" if s >= 55 else ("medium" if s >= 30 else "small")


def draw(img, preds, confs, gts, caption):
    vis = img.copy()
    for g in gts:                                   # human GT, thin, for reference
        cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), BLUE, 1)
    for (x1, y1, x2, y2), c in zip(preds, confs):
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
        label = f"deer {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(int(y1) - 4, th + 4)
        cv2.rectangle(vis, (int(x1), ty - th - 4), (int(x1) + tw + 4, ty + 2),
                      (0, 0, 0), -1)
        cv2.putText(vis, label, (int(x1) + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, GREEN, 1, cv2.LINE_AA)
    bar = np.zeros((26, vis.shape[1], 3), np.uint8)
    cv2.putText(bar, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1,
                cv2.LINE_AA)
    return np.vstack([vis, bar])


def sheet(tiles, cols=3):
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * h:r * h + t.shape[0], c * w:c * w + t.shape[1]] = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arch", default="yolo", choices=["yolo", "rtdetr"])
    ap.add_argument("--data", default="data/dataset/yolo_v3/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--n-large", type=int, default=9)
    ap.add_argument("--n-medium", type=int, default=9)
    ap.add_argument("--n-small", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/viz/detect")
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from ultralytics import YOLO, RTDETR
    model = (RTDETR if args.arch == "rtdetr" else YOLO)(args.weights)

    imgs, lbl_dir = load_split(args.data, args.split)
    buckets: dict[str, list] = {"large": [], "medium": [], "small": []}
    for ip in imgs:
        im = cv2.imread(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        g = gt_boxes(os.path.join(lbl_dir,
                                  os.path.splitext(os.path.basename(ip))[0] + ".txt"), W, H)
        b = bucket_of(g)
        if b:
            buckets[b].append(ip)
    want = {"large": args.n_large, "medium": args.n_medium, "small": args.n_small}
    print({k: f"{len(v)} available, taking {min(want[k], len(v))}"
           for k, v in buckets.items()})

    index = []
    for bname, n in want.items():
        pool = buckets[bname]
        if not pool:
            continue
        random.shuffle(pool)
        chosen = pool[:n]
        tiles = []
        bdir = os.path.join(args.out, bname)
        os.makedirs(bdir, exist_ok=True)
        for ip in chosen:
            im = cv2.imread(ip)
            H, W = im.shape[:2]
            g = gt_boxes(os.path.join(
                lbl_dir, os.path.splitext(os.path.basename(ip))[0] + ".txt"), W, H)
            r = model.predict(source=im, imgsz=args.imgsz, conf=args.conf,
                              device=args.device, verbose=False)[0]
            pb = (r.boxes.xyxy.cpu().numpy().tolist()
                  if r.boxes is not None and len(r.boxes) else [])
            pc = (r.boxes.conf.cpu().numpy().tolist()
                  if r.boxes is not None and len(r.boxes) else [])
            stem = os.path.splitext(os.path.basename(ip))[0]
            cap = f"{stem[:44]}  pred {len(pb)} / GT {len(g)}"
            vis = draw(im, pb, pc, g, cap)
            cv2.imwrite(os.path.join(bdir, f"{stem}.jpg"), vis)
            tiles.append(vis)
            index.append((bname, stem, len(pb), len(g),
                          round(max(pc), 3) if pc else 0.0))
        sh = sheet(tiles)
        if sh is not None:
            cv2.imwrite(os.path.join(args.out, f"_sheet_{bname}.jpg"), sh)
            print(f"  {bname}: {len(tiles)} frames -> _sheet_{bname}.jpg")

    with open(os.path.join(args.out, "index.csv"), "w") as f:
        f.write("bucket,frame,n_pred,n_gt,max_conf\n")
        for row in index:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\n-> {args.out}/  (per-bucket folders + _sheet_*.jpg + index.csv)")
    print("   green box = prediction with confidence | thin blue = human GT")


if __name__ == "__main__":
    main()
