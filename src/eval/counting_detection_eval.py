#!/usr/bin/env python3
"""
Frame-level detection scored with COUNTING-relevant matching.

Standard mAP demands IoU>=0.5 — a box must sit tightly on the animal. For *counting*
that is the wrong question: if the detector puts a box overlapping the deer at all, the
tracker links it and the animal gets counted. Box tightness is irrelevant downstream.

So this reports precision/recall/F1 under four matching rules, strict -> permissive:
    iou50   IoU >= 0.50   (the detection-paper standard; reported for honesty)
    iou30   IoU >= 0.30
    touch   ANY overlap   <- the counting criterion
    center  detection centre inside the GT box (or vice versa)

NOTE ON REPORTING: publish `iou50` in the detection table. The permissive numbers belong
to the counting evaluation and MUST be named as such ("presence/counting recall"), never
relabelled as mAP — reporting mAP at IoU>0 reads as metric gaming and will sink a review.

Usage:
  python src/eval/counting_detection_eval.py --weights best.pt \
      --data data/dataset/yolo_v3/data.yaml --split test --conf 0.25
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from track_recall import CRITERIA, match_kind  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg")


def load_split(data_yaml: str, split: str):
    with open(data_yaml) as f:
        d = yaml.safe_load(f)
    root = d["path"]
    img_dir = os.path.join(root, d.get(split, f"images/{split}"))
    lbl_dir = img_dir.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    imgs = []
    for e in IMG_EXTS:
        imgs += glob.glob(os.path.join(img_dir, f"*{e}"))
    return sorted(imgs), lbl_dir


def gt_boxes(lbl_path: str, W: int, H: int):
    out = []
    if not os.path.isfile(lbl_path):
        return out
    with open(lbl_path) as f:
        for ln in f:
            p = ln.split()
            if len(p) < 5:
                continue
            _, xc, yc, bw, bh = (float(v) for v in p[:5])
            out.append([(xc - bw / 2) * W, (yc - bh / 2) * H,
                        (xc + bw / 2) * W, (yc + bh / 2) * H])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arch", default="yolo", choices=["yolo", "rtdetr", "mmdet"])
    ap.add_argument("--config", default="",
                    help="mmdetection config .py (required for --arch mmdet). mmdet "
                         "checkpoints carry no architecture, unlike Ultralytics .pt")
    ap.add_argument("--data", default="data/dataset/yolo_v3/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", default="results/counting_eval")
    args = ap.parse_args()

    # Both back-ends reduce to the same thing: a list of [x1,y1,x2,y2] per image, already
    # thresholded. Everything downstream is back-end agnostic.
    if args.arch == "mmdet":
        if not args.config:
            raise SystemExit("--arch mmdet requires --config <mmdet config .py>")
        from mmdet.apis import inference_detector, init_detector
        mm = init_detector(args.config, args.weights,
                           device=f"cuda:{args.device}" if args.device.isdigit()
                           else args.device)

        def predict(chunk):
            res = inference_detector(mm, chunk)
            if not isinstance(res, list):
                res = [res]
            out = []
            for r in res:
                p = r.pred_instances
                keep = p.scores >= args.conf
                out.append(p.bboxes[keep].cpu().numpy().tolist())
            return out
    else:
        from ultralytics import YOLO, RTDETR
        model = (RTDETR if args.arch == "rtdetr" else YOLO)(args.weights)

        def predict(chunk):
            res = model.predict(source=chunk, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)
            return [(r.boxes.xyxy.cpu().numpy().tolist()
                     if r.boxes is not None and len(r.boxes) else [])
                    for r in res]

    imgs, lbl_dir = load_split(args.data, args.split)
    tp = {c: 0 for c in CRITERIA}
    fn = {c: 0 for c in CRITERIA}
    fp = {c: 0 for c in CRITERIA}
    n_gt = 0

    for i in range(0, len(imgs), args.batch):
        chunk = imgs[i:i + args.batch]
        preds = predict(chunk)
        for ip, d in zip(chunk, preds):
            H, W = cv2.imread(ip).shape[:2]
            g = gt_boxes(os.path.join(
                lbl_dir, os.path.splitext(os.path.basename(ip))[0] + ".txt"), W, H)
            n_gt += len(g)
            for c in CRITERIA:
                # greedy one-to-one matching per criterion
                used = set()
                hit = 0
                for gi, gb in enumerate(g):
                    for di, db in enumerate(d):
                        if di in used:
                            continue
                        if c in match_kind(gb, db):
                            used.add(di); hit += 1
                            break
                tp[c] += hit
                fn[c] += len(g) - hit
                fp[c] += len(d) - len(used)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    print(f"\n=== [{args.tag}] {args.split} split, conf={args.conf}, "
          f"{len(imgs)} images, {n_gt} GT boxes ===")
    print(f"{'criterion':<10} {'precision':>10} {'recall':>8} {'F1':>7}   note")
    note = {"iou50": "standard detection metric",
            "iou30": "",
            "touch": "<-- COUNTING criterion (any overlap)",
            "center": "centre-in-box"}
    for c in CRITERIA:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"{c:<10} {p:10.3f} {r:8.3f} {f1:7.3f}   {note[c]}")
        rows.append({"tag": args.tag, "split": args.split, "conf": args.conf,
                     "criterion": c, "tp": tp[c], "fp": fp[c], "fn": fn[c],
                     "precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(f1, 4)})

    out_csv = os.path.join(args.out, f"{args.tag}_{args.split}_conf{args.conf}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out_csv}")


if __name__ == "__main__":
    main()
