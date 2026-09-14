#!/usr/bin/env python3
"""Throughput of the counting pipeline and of every ultralytics detector in the roster.

Two questions, answered separately because they are different quantities:

  1. Pipeline speed as deployed (src/app/engine.py, pass 1): decode -> CLAHE -> YOLO11m@640
     -> BoT-SORT, one frame at a time, batch 1, FP32. Timed stage by stage on N frames of a
     held-out video, then end to end on the whole video via engine.analyse(). The source is
     60 fps, so fps / 60 is the real-time factor.
  2. Detector-only speed for the benchmark roster: batch-1 latency (median ms, which is what
     a live system sees) and batch-32 throughput (frames/s, which is what an offline survey
     sees), on the same CLAHE frames. FP32 as evaluated in the paper; FP16 added for the
     deployed model only.

Timing rules: 50 warm-up frames discarded; torch.cuda.synchronize() around every measured
call; medians reported alongside means; GPU, driver and library versions recorded so the
numbers can be attributed to hardware.

  python src/eval/speed_benchmark.py --video <held-out mp4> --out <dir> [--frames 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "common"))
sys.path.insert(0, os.path.join(_ROOT, "src", "app"))
from thermal import enhance_contrast   # noqa: E402
import engine                          # noqa: E402

RUNS = "/work/hdd/bgte/tislam6/wildlife_outputs/runs"
ROSTER = [  # name, weights, imgsz  (the eleven-model table plus the 1280 ablation rows)
    ("YOLOv8m@640",  f"{RUNS}/yolov8m_640_v3pooled/weights/best.pt", 640),
    ("YOLOv9m@640",  f"{RUNS}/yolov9m_640_v3pooled/weights/best.pt", 640),
    ("YOLOv10m@640", f"{RUNS}/yolov10m_640_v3pooled/weights/best.pt", 640),
    ("YOLO11m@640",  f"{RUNS}/yolo11m_640_v3pooled/weights/best.pt", 640),
    ("YOLO12m@640",  f"{RUNS}/yolo12m_640_v3pooled/weights/best.pt", 640),
    ("RT-DETR-L@640", f"{RUNS}/rtdetr-l_640_v3pooled/weights/best.pt", 640),
    ("YOLOv9m@1280", f"{RUNS}/yolov9m_1280_v3pooled/weights/best.pt", 1280),
    ("YOLOv10m@1280", f"{RUNS}/yolov10m_1280_v3pooled/weights/best.pt", 1280),
]
DEPLOYED = "YOLO11m@640"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, items, warm=50):
    """Per-item wall time in ms after `warm` discarded calls. -> (list_ms, total_s)"""
    for x in items[:warm]:
        fn(x)
    sync()
    ms = []
    t_all = time.perf_counter()
    for x in items[warm:]:
        t = time.perf_counter()
        fn(x)
        sync()
        ms.append((time.perf_counter() - t) * 1e3)
    return ms, time.perf_counter() - t_all


def summarize(ms, n_frames_per_item=1):
    fps = 1e3 * n_frames_per_item / statistics.mean(ms)
    return dict(n=len(ms), mean_ms=round(statistics.mean(ms), 3),
                median_ms=round(statistics.median(ms), 3),
                p95_ms=round(float(np.percentile(ms, 95)), 3), fps=round(fps, 1))


def load_frames(video, n):
    cap = cv2.VideoCapture(video)
    raw, t_dec = [], 0.0
    while len(raw) < n:
        t = time.perf_counter()
        ok, f = cap.read()
        t_dec += time.perf_counter() - t
        if not ok:
            break
        raw.append(f)
    cap.release()
    return raw, t_dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--skip-full", action="store_true", help="skip the whole-video analyse()")
    ap.add_argument("--skip-cpu", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from ultralytics import YOLO, RTDETR

    res = dict(video=os.path.basename(args.video), frames_timed=args.frames,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
               torch=torch.__version__, cuda=torch.version.cuda, cpu=platform.processor(),
               n_cpu=os.cpu_count(), opencv=cv2.__version__)
    import ultralytics
    res["ultralytics"] = ultralytics.__version__
    fps_src, total, W, H = engine.video_meta(args.video)
    res.update(source_fps=fps_src, source_frames=total, width=W, height=H)
    print(json.dumps(res, indent=1), flush=True)

    # ------------------------------------------------------------------ stage timings
    raw, t_dec = load_frames(args.video, args.frames)
    n = len(raw)
    res["decode"] = dict(n=n, ms_per_frame=round(1e3 * t_dec / n, 3), fps=round(n / t_dec, 1))
    t = time.perf_counter()
    clahe = [enhance_contrast(f, method="clahe") for f in raw]
    t_cl = time.perf_counter() - t
    res["clahe"] = dict(n=n, ms_per_frame=round(1e3 * t_cl / n, 3), fps=round(n / t_cl, 1))
    print("decode", res["decode"], "clahe", res["clahe"], flush=True)

    cfg = engine.Config(device=args.device)

    def loader(path):
        return RTDETR(path) if "rtdetr" in path.lower() else YOLO(path)

    # detector only, batch 1, deployed settings
    m = loader(cfg.weights)
    ms, _ = timed(lambda f: m.predict(f, imgsz=640, conf=cfg.conf, iou=cfg.iou,
                                      device=args.device, verbose=False), clahe)
    res["detector_b1"] = summarize(ms)
    print("detector b1", res["detector_b1"], flush=True)

    # detector + BoT-SORT, batch 1, as engine.analyse does it (fresh model => fresh tracker)
    m = loader(cfg.weights)
    ms, _ = timed(lambda f: m.track(source=f, tracker=cfg.tracker, persist=True, conf=cfg.conf,
                                    iou=cfg.iou, imgsz=640, device=args.device, verbose=False),
                  clahe)
    res["detector_tracker_b1"] = summarize(ms)
    res["tracker_overhead_ms"] = round(res["detector_tracker_b1"]["mean_ms"]
                                       - res["detector_b1"]["mean_ms"], 3)
    print("detector+tracker b1", res["detector_tracker_b1"], flush=True)

    # per-frame budget of the deployed loop = decode + clahe + detector + tracker
    per = (res["decode"]["ms_per_frame"] + res["clahe"]["ms_per_frame"]
           + res["detector_tracker_b1"]["mean_ms"])
    res["pipeline_b1_estimate"] = dict(ms_per_frame=round(per, 3), fps=round(1e3 / per, 1),
                                       realtime_factor=round(1e3 / per / fps_src, 2))
    print("pipeline estimate", res["pipeline_b1_estimate"], flush=True)

    # ------------------------------------------------------------------ whole video, as deployed
    if not args.skip_full:
        t = time.perf_counter()
        tracks, fps_v, nfr, _, _ = engine.analyse(args.video, cfg)
        wall = time.perf_counter() - t
        t2 = time.perf_counter()
        n_conf = sum(1 for tr in tracks.values() if tr.confirmed(cfg, fps_v))
        t_rule = time.perf_counter() - t2
        res["pipeline_full_video"] = dict(frames=nfr, wall_s=round(wall, 1),
                                          fps=round(nfr / wall, 1),
                                          realtime_factor=round(nfr / wall / fps_v, 2),
                                          tracks=len(tracks), confirmed=n_conf,
                                          rule_ms_total=round(1e3 * t_rule, 3))
        print("full video", res["pipeline_full_video"], flush=True)

    # ------------------------------------------------------------------ roster
    roster = []
    for name, w, sz in ROSTER:
        if not os.path.isfile(w):
            roster.append(dict(model=name, error="weights missing")); continue
        m = loader(w)
        try:
            info = m.info(verbose=False)   # (layers, params, gradients, GFLOPs)
            params, gflops = info[1], info[3]
        except Exception:
            params, gflops = None, None
        row = dict(model=name, imgsz=sz, params_M=round(params / 1e6, 2) if params else None,
                   gflops=round(gflops, 1) if gflops else None)
        ms, _ = timed(lambda f: m.predict(f, imgsz=sz, conf=0.25, device=args.device,
                                          verbose=False), clahe[:550])
        row["b1_fp32"] = summarize(ms)
        chunks = [clahe[i:i + 32] for i in range(0, 32 * 40, 32)]
        ms, _ = timed(lambda c: m.predict(c, imgsz=sz, conf=0.25, device=args.device,
                                          verbose=False), chunks, warm=5)
        row["b32_fp32"] = summarize(ms, 32)
        if name == DEPLOYED:
            ms, _ = timed(lambda f: m.predict(f, imgsz=sz, conf=0.25, half=True,
                                              device=args.device, verbose=False), clahe[:550])
            row["b1_fp16"] = summarize(ms)
            ms, _ = timed(lambda c: m.predict(c, imgsz=sz, conf=0.25, half=True,
                                              device=args.device, verbose=False), chunks, warm=5)
            row["b32_fp16"] = summarize(ms, 32)
        print(json.dumps(row), flush=True)
        roster.append(row)
        del m
        torch.cuda.empty_cache()
    res["roster"] = roster

    # ------------------------------------------------------------------ CPU, deployed model
    if not args.skip_cpu:
        torch.set_num_threads(min(16, os.cpu_count() or 1))
        m = loader(cfg.weights)
        ms, _ = timed(lambda f: m.predict(f, imgsz=640, conf=cfg.conf, iou=cfg.iou,
                                          device="cpu", verbose=False), clahe[:250], warm=20)
        res["detector_b1_cpu"] = dict(threads=torch.get_num_threads(), **summarize(ms))
        print("cpu b1", res["detector_b1_cpu"], flush=True)

    with open(os.path.join(args.out, "speed_benchmark.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", os.path.join(args.out, "speed_benchmark.json"))


if __name__ == "__main__":
    main()
