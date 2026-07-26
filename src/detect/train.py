#!/usr/bin/env python3
"""
Train a YOLO/RT-DETR deer detector on the site-split dataset, then run the full
evaluation (src/detect/eval_full.py) on the held-out test site.

Built for long unattended GPU runs:
  * **Live progress** — every epoch prints current val mAP50 / mAP50-95 / P / R plus
    epoch time, running average, and **ETA to finish** (min + h). Tail the .out log
    to watch it in real time.
  * **Resume-safe** — checkpoints every ``--save-period`` epochs; ``--resume`` picks up
    from ``<run>/weights/last.pt`` with the exact original args (Ultralytics restores
    them). Re-submitting the same job after an error/timeout just continues.
  * **Maximal metrics** — after training, eval_full dumps native + COCO size-stratified
    + domain metrics to results/detection_eval/.

Run (GPU node):
    python src/detect/train.py --model weights/yolo11m.pt \
        --data data/dataset/yolo_v2/data.yaml \
        --project /work/hdd/bgte/tislam6/wildlife_outputs/runs --name yolo11m_640_v2_SHWtest
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time


def _make_eta_logger():
    """Ultralytics 'on_fit_epoch_end' callback: live per-epoch metrics + ETA."""
    state = {"start": time.time(), "last": None, "times": []}

    def cb(trainer):
        now = time.time()
        if state["last"] is not None:
            state["times"].append(now - state["last"])
        state["last"] = now
        e = int(trainer.epoch) + 1
        E = int(trainer.epochs)
        times = state["times"]
        avg = sum(times) / len(times) if times else 0.0
        remaining = avg * (E - e)
        elapsed = now - state["start"]
        m = getattr(trainer, "metrics", {}) or {}

        def g(k):
            return float(m.get(k, float("nan")))

        print(
            f"[epoch {e:>3}/{E}] "
            f"mAP50={g('metrics/mAP50(B)'):.4f} "
            f"mAP50-95={g('metrics/mAP50-95(B)'):.4f} "
            f"P={g('metrics/precision(B)'):.4f} R={g('metrics/recall(B)'):.4f} "
            f"| epoch={times[-1] if times else 0:.0f}s avg={avg:.0f}s "
            f"elapsed={elapsed/60:.1f}min ETA={remaining/60:.1f}min ({remaining/3600:.2f}h)",
            flush=True,
        )

    return cb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/yolo11m.pt")
    ap.add_argument("--arch", default="yolo", choices=["yolo", "rtdetr"])
    ap.add_argument("--data", default="data/dataset/yolo_v2/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8,
                    help="dataloader workers PER training process. At 1280px each "
                         "worker buffers large decoded images; 2 concurrent trainings "
                         "x 8 workers exceeded 200GB host RAM (job 2728187 OOM-killed).")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--save-period", type=int, default=10,
                    help="checkpoint every N epochs (for mid-run resume)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="/work/hdd/bgte/tislam6/wildlife_outputs/runs")
    ap.add_argument("--name", default="yolo_deer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", default="auto",
                    help="SGD/AdamW/auto. Explicit beats Ultralytics 8.4 'auto', which "
                         "picks MuSGD@0.01 and collapses fine-tunes / diverges DETRs.")
    ap.add_argument("--lr0", type=float, default=0.0,
                    help="initial LR (0 = use Ultralytics default for the optimizer)")
    # --- augmentation / warmup knobs (strategy-B: grow positives, tame warmup) ---
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--close-mosaic", type=int, default=15)
    ap.add_argument("--copy-paste", type=float, default=0.0,
                    help="paste deer from other imgs to boost positive instances")
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--translate", type=float, default=0.1)
    ap.add_argument("--degrees", type=float, default=0.0)
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--warmup-bias-lr", type=float, default=0.1,
                    help="0.0 removes the warmup bias-LR spike that collapses mAP")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <project>/<name>/weights/last.pt if present")
    ap.add_argument("--eval-conf", type=float, default=0.5,
                    help="deployment conf for the domain (FP/bg-frame) metric")
    args = ap.parse_args()

    from ultralytics import YOLO, RTDETR
    Model = RTDETR if args.arch == "rtdetr" else YOLO

    run_dir = os.path.join(args.project, args.name)
    last = os.path.join(run_dir, "weights", "last.pt")
    resuming = args.resume and os.path.isfile(last)

    if resuming:
        print(f"=== RESUMING from {last} ===", flush=True)
        model = Model(last)
        train_kwargs = dict(resume=True)
    else:
        if args.resume:
            print(f"=== --resume set but no checkpoint at {last}; starting fresh ===",
                  flush=True)
        model = Model(args.model)
        train_kwargs = dict(
            data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
            patience=args.patience, device=args.device, project=args.project,
            name=args.name, seed=args.seed, save_period=args.save_period,
            workers=args.workers,
            # tiny thermal blobs: keep mosaic but close it early so the model sees
            # native-scale deer before the final epochs; deer are pale-on-dark so
            # geometric aug helps more than colour aug.
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, fliplr=0.5, flipud=0.0,
            plots=True, exist_ok=True, verbose=True, optimizer=args.optimizer,
            mosaic=args.mosaic, close_mosaic=args.close_mosaic,
            copy_paste=args.copy_paste, mixup=args.mixup, scale=args.scale,
            translate=args.translate, degrees=args.degrees,
            warmup_epochs=args.warmup_epochs, warmup_bias_lr=args.warmup_bias_lr,
        )
        if args.lr0 > 0:
            train_kwargs["lr0"] = args.lr0

    model.add_callback("on_fit_epoch_end", _make_eta_logger())
    model.train(**train_kwargs)

    best = os.path.join(run_dir, "weights", "best.pt")
    print(f"\n=== best weights: {best} ===", flush=True)

    # Full evaluation on the held-out test site. Kept in its own process so a metrics
    # hiccup can never lose the trained weights; re-runnable standalone afterwards.
    print("\n=== full evaluation on held-out TEST split ===", flush=True)
    try:
        subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "eval_full.py"),
             "--weights", best, "--data", args.data, "--split", "test",
             "--imgsz", str(args.imgsz), "--device", args.device,
             "--arch", args.arch, "--tag", args.name.replace("_SHWtest", ""),
             "--deploy-conf", str(args.eval_conf)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[warn] eval_full failed ({e}); weights are safe at {best}. "
              f"Re-run eval_full.py manually.", flush=True)


if __name__ == "__main__":
    main()
