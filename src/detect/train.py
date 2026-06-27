#!/usr/bin/env python3
"""
Train a YOLO deer detector on the site-split dataset, then evaluate on the held-out
test site. Single class (deer); thermal frames are 640x512.

Reports val and test metrics (mAP50, mAP50-95, precision, recall) so the held-out-site
number reflects generalization to a new location — the way the client will use it.

Run (on a GPU node):
    python src/detect/train.py --model weights/yolo11s.pt \
        --data data/dataset/yolo/data.yaml \
        --project /work/hdd/bgte/tislam6/wildlife_outputs/runs --name yolo11s_SHWtest
"""
from __future__ import annotations
import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/yolo11s.pt")
    ap.add_argument("--data", default="data/dataset/yolo/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="/work/hdd/bgte/tislam6/wildlife_outputs/runs")
    ap.add_argument("--name", default="yolo_deer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        # tiny thermal blobs: keep mosaic but close it early so the model sees
        # native-scale deer before the final epochs; deer are pale-on-dark so
        # geometric aug helps more than colour aug.
        close_mosaic=15,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        fliplr=0.5, flipud=0.0,
        plots=True,
        exist_ok=True,
    )

    run_dir = os.path.join(args.project, args.name)
    best = os.path.join(run_dir, "weights", "best.pt")
    print(f"\n=== best weights: {best} ===")

    # Held-out-site test metrics — the headline generalization number.
    print("\n=== evaluating on held-out TEST split ===")
    test_metrics = YOLO(best).val(
        data=args.data, split="test", imgsz=args.imgsz, device=args.device,
        project=args.project, name=args.name + "_test", exist_ok=True,
    )
    b = test_metrics.box
    print("\n================  HELD-OUT TEST (new site)  ================")
    print(f"  precision : {b.mp:.4f}")
    print(f"  recall    : {b.mr:.4f}")
    print(f"  mAP@50    : {b.map50:.4f}")
    print(f"  mAP@50-95 : {b.map:.4f}")
    print("===========================================================")


if __name__ == "__main__":
    main()
