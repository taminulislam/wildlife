#!/usr/bin/env python3
"""
Maximal detector evaluation on a dataset split — dump every metric we can, so we
can decide later which go in the paper.

Two complementary passes over the same split:

  1. Ultralytics native ``val`` — precision / recall / F1 / mAP50 / mAP75 / mAP50-95
     at the deployment operating point, per-class, plus speed and the confusion
     matrix (TP/FP/FN). Also writes PR / F1 / P / R curve plots.
  2. COCO (pycocotools) — size-stratified AP/AR: **AP_small / AP_medium / AP_large**
     and AR at 1/10/100 dets. Deer blobs are tiny, so AP_small is the metric that
     actually matters here; native mAP hides it.

Plus two domain-specific ("unique") metrics tied to our counting problem:
  * **FP per background frame** — mean false detections on deer-FREE test frames at
    the deployment conf. This is the count-inflation failure mode, measured directly.
  * **small-deer recall** — recall restricted to COCO-small GT boxes.

Reusable for any weights (new model, the old-labels baseline, RT-DETR) on the SAME
clean test split → honest head-to-head. Writes a JSON blob + appends a markdown row.

Usage:
  python src/detect/eval_full.py --weights .../best.pt --data data/dataset/yolo_v2/data.yaml \
      --split test --imgsz 640 --tag yolo11m_640_v2 --arch yolo
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import tempfile

import cv2
import yaml

IMG_EXTS = (".png", ".jpg", ".jpeg")
COCO_SMALL = 32 ** 2      # < 1024 px area
COCO_MEDIUM = 96 ** 2     # < 9216 px area


def load_model(weights: str, arch: str):
    from ultralytics import YOLO, RTDETR
    return RTDETR(weights) if arch.lower() == "rtdetr" else YOLO(weights)


def split_dirs(data_yaml: str, split: str) -> tuple[str, str]:
    with open(data_yaml) as f:
        d = yaml.safe_load(f)
    root = d["path"]
    img_dir = os.path.join(root, d.get(split, f"images/{split}"))
    lbl_dir = img_dir.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    return img_dir, lbl_dir


def list_images(img_dir: str) -> list[str]:
    out: list[str] = []
    for e in IMG_EXTS:
        out += glob.glob(os.path.join(img_dir, f"*{e}"))
    return sorted(out)


def build_coco_gt(img_dir: str, lbl_dir: str):
    """YOLO labels -> COCO GT dict + per-image (id,w,h). category_id=1 (deer)."""
    images, anns = [], []
    id_by_name: dict[str, int] = {}
    ann_id, img_id = 1, 0
    for ip in list_images(img_dir):
        img_id += 1
        im = cv2.imread(ip)
        h, w = im.shape[:2]
        name = os.path.basename(ip)
        id_by_name[name] = img_id
        images.append({"id": img_id, "file_name": name, "width": w, "height": h})
        lp = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.isfile(lp):
            with open(lp) as fh:
                for ln in fh:
                    p = ln.split()
                    if len(p) < 5:
                        continue
                    _, xc, yc, bw, bh = (float(x) for x in p[:5])
                    x = (xc - bw / 2) * w; y = (yc - bh / 2) * h
                    ww = bw * w; hh = bh * h
                    anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                 "bbox": [x, y, ww, hh], "area": ww * hh, "iscrowd": 0})
                    ann_id += 1
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "deer"}]}
    return coco, id_by_name


def run_coco_eval(model, img_dir, lbl_dir, imgsz, device, conf_deploy):
    """Low-conf inference -> pycocotools size-stratified AP/AR. Also computes the two
    domain metrics (FP/background-frame, small-deer recall) at the deployment conf."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt, id_by_name = build_coco_gt(img_dir, lbl_dir)
    imgs = list_images(img_dir)

    dets = []
    # deployment-conf bookkeeping for the unique metrics
    bg_frames = 0; bg_fp = 0            # FP on deer-free frames
    small_gt = 0; small_hit = 0         # (approx) small-deer recall proxy
    # GT small boxes per image for the recall proxy
    gt_small_by_img: dict[int, list] = {}
    for a in gt["annotations"]:
        if a["area"] < COCO_SMALL:
            gt_small_by_img.setdefault(a["image_id"], []).append(a["bbox"])
    gt_by_img: dict[int, int] = {}
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]] = gt_by_img.get(a["image_id"], 0) + 1

    # Chunk the source list OURSELVES: Ultralytics ignores predict(batch=) for list
    # sources and builds one giant batch of the whole split (observed: 36-43 GiB OOM).
    def _iter_results(chunk=16):
        for i in range(0, len(imgs), chunk):
            yield from model.predict(source=imgs[i:i + chunk], imgsz=imgsz,
                                     conf=0.001, iou=0.7, device=device,
                                     verbose=False)
    results = _iter_results()
    for ip, r in zip(imgs, results):
        img_id = id_by_name[os.path.basename(ip)]
        n_gt = gt_by_img.get(img_id, 0)
        deploy_boxes = 0
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), s in zip(xyxy, scores):
                dets.append({"image_id": img_id, "category_id": 1,
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             "score": float(s)})
                if s >= conf_deploy:
                    deploy_boxes += 1
        if n_gt == 0:                       # deer-free frame
            bg_frames += 1
            bg_fp += deploy_boxes
        # small-deer recall proxy: count small GT vs deploy dets that overlap them
        for gb in gt_small_by_img.get(img_id, []):
            small_gt += 1
            if r.boxes is not None and len(r.boxes):
                if _any_iou(gb, r.boxes.xyxy.cpu().numpy(),
                            r.boxes.conf.cpu().numpy(), conf_deploy, 0.5):
                    small_hit += 1

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(gt, tf); gt_path = tf.name
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(dets) if dets else coco_gt.loadRes([])
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    os.unlink(gt_path)
    s = ev.stats  # 12 COCO numbers
    coco = {
        "AP": s[0], "AP50": s[1], "AP75": s[2],
        "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5],
        "AR1": s[6], "AR10": s[7], "AR100": s[8],
        "AR_small": s[9], "AR_medium": s[10], "AR_large": s[11],
    }
    unique = {
        "background_frames": bg_frames,
        "fp_per_background_frame": (bg_fp / bg_frames) if bg_frames else 0.0,
        "small_gt": small_gt,
        "small_deer_recall_proxy": (small_hit / small_gt) if small_gt else 0.0,
        "deploy_conf": conf_deploy,
    }
    return coco, unique


def _any_iou(gt_box_xywh, dt_xyxy, dt_conf, conf_thr, iou_thr):
    gx, gy, gw, gh = gt_box_xywh
    gx2, gy2 = gx + gw, gy + gh
    ga = gw * gh
    for (x1, y1, x2, y2), s in zip(dt_xyxy, dt_conf):
        if s < conf_thr:
            continue
        ix1, iy1 = max(gx, x1), max(gy, y1)
        ix2, iy2 = min(gx2, x2), min(gy2, y2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            continue
        ua = ga + (x2 - x1) * (y2 - y1) - inter
        if ua > 0 and inter / ua >= iou_thr:
            return True
    return False


def native_val(model, data, split, imgsz, device, project, name):
    res = model.val(data=data, split=split, imgsz=imgsz, device=device,
                    project=project, name=name, exist_ok=True, plots=True, verbose=False)
    b = res.box
    cm = None
    try:
        m = res.confusion_matrix.matrix  # (nc+1, nc+1); single class -> 2x2
        tp = float(m[0, 0]); fp = float(m[0, 1]); fn = float(m[1, 0])
        cm = {"tp": tp, "fp": fp, "fn": fn,
              "precision_cm": tp / (tp + fp) if (tp + fp) else 0.0,
              "recall_cm": tp / (tp + fn) if (tp + fn) else 0.0}
    except Exception:
        pass
    return {
        "precision": float(b.mp), "recall": float(b.mr),
        "mAP50": float(b.map50), "mAP75": float(b.map75), "mAP50-95": float(b.map),
        "f1": float(2 * b.mp * b.mr / (b.mp + b.mr)) if (b.mp + b.mr) else 0.0,
        "fitness": float(res.fitness) if hasattr(res, "fitness") else None,
        "speed_ms": {k: float(v) for k, v in (res.speed or {}).items()},
        "confusion": cm,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/dataset/yolo_v2/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--arch", default="yolo", choices=["yolo", "rtdetr"])
    ap.add_argument("--tag", required=True, help="row label in the results table")
    ap.add_argument("--deploy-conf", type=float, default=0.5)
    ap.add_argument("--out-dir", default="results/detection_eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.weights, args.arch)
    img_dir, lbl_dir = split_dirs(args.data, args.split)

    print(f"=== [{args.tag}] native val on '{args.split}' ===", flush=True)
    native = native_val(model, args.data, args.split, args.imgsz, args.device,
                        project=args.out_dir, name=f"{args.tag}_native")
    print(f"=== [{args.tag}] COCO size-stratified + unique metrics ===", flush=True)
    coco, unique = run_coco_eval(model, img_dir, lbl_dir, args.imgsz, args.device,
                                 args.deploy_conf)

    blob = {"tag": args.tag, "weights": os.path.abspath(args.weights),
            "arch": args.arch, "split": args.split, "imgsz": args.imgsz,
            "native": native, "coco": coco, "unique": unique}
    jpath = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(jpath, "w") as f:
        json.dump(blob, f, indent=2)

    # human-readable console summary
    print("\n" + "=" * 66)
    print(f"  {args.tag}  ({args.arch}, {args.split}, imgsz {args.imgsz})")
    print("-" * 66)
    print(f"  P {native['precision']:.4f}  R {native['recall']:.4f}  "
          f"F1 {native['f1']:.4f}")
    print(f"  mAP50 {native['mAP50']:.4f}  mAP75 {native['mAP75']:.4f}  "
          f"mAP50-95 {native['mAP50-95']:.4f}")
    print(f"  COCO AP {coco['AP']:.4f} | small {coco['AP_small']:.4f}  "
          f"med {coco['AP_medium']:.4f}  large {coco['AP_large']:.4f}")
    print(f"  AR100 {coco['AR100']:.4f} | AR_small {coco['AR_small']:.4f}")
    print(f"  [unique] FP/background-frame {unique['fp_per_background_frame']:.3f}  "
          f"small-deer recall {unique['small_deer_recall_proxy']:.3f}")
    print("=" * 66)

    # append a compact markdown row
    md = os.path.join(args.out_dir, "eval_table.md")
    new = not os.path.exists(md)
    with open(md, "a") as f:
        if new:
            f.write("| tag | P | R | F1 | mAP50 | mAP50-95 | AP_small | "
                    "AR_small | FP/bg-frame |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
        f.write(f"| {args.tag} | {native['precision']:.3f} | {native['recall']:.3f} "
                f"| {native['f1']:.3f} | {native['mAP50']:.3f} | "
                f"{native['mAP50-95']:.3f} | {coco['AP_small']:.3f} | "
                f"{coco['AR_small']:.3f} | {unique['fp_per_background_frame']:.2f} |\n")
    print(f"wrote {jpath}\nappended {md}", flush=True)


if __name__ == "__main__":
    main()
