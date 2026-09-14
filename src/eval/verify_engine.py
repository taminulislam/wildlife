#!/usr/bin/env python3
"""Does src/app/engine.py reproduce the paper's pool-C run on a held-out video?

Runs engine.analyse() end to end (also the wall-clock speed measurement), applies the
frozen rule, and prints the confirmed count next to the paper's per-track record for the
same video so the two can be compared track by track.

  python src/eval/verify_engine.py --video <mp4> --paper-tracks <phaseC merged/counts.csv>
"""
import argparse
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "app"))
import engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--paper-tracks", required=True, help="per-track counts.csv from count_deer.py")
ap.add_argument("--label", default="PAPER")
ap.add_argument("--min-frames", type=int, default=20)
ap.add_argument("--min-topk", type=float, default=0.65)
args = ap.parse_args()

cfg = engine.Config(min_frames=args.min_frames, min_topk_conf=args.min_topk)
print("tracker:", cfg.tracker, "| conf", cfg.conf, "iou", cfg.iou, "imgsz", cfg.imgsz, flush=True)
t = time.perf_counter()
tracks, fps, n, W, H = engine.analyse(args.video, cfg)
wall = time.perf_counter() - t
conf = [tr for tr in tracks.values() if tr.confirmed(cfg, fps)]
print(f"\nENGINE  frames {n}  wall {wall:.1f}s  {n / wall:.1f} fps  ({n / wall / fps:.2f}x real time)")
print(f"ENGINE  tracks {len(tracks)}  confirmed {len(conf)}")
rows = sorted(tracks.values(), key=lambda tr: -tr.topk_conf(cfg.topk))[:8]
for tr in rows:
    print(f"  id {tr.tid:>5} n {tr.n():>4} span {tr.span_s(fps):6.2f}s topk {tr.topk_conf(cfg.topk):.4f}"
          f" first {tr.obs[0][0]:>6} orphan {int(tr.orphan)} {'CONFIRMED' if tr.confirmed(cfg, fps) else ''}")

stem = os.path.splitext(os.path.basename(args.video))[0]
paper = [r for r in csv.DictReader(open(args.paper_tracks)) if r["video"] == stem]
pc = [r for r in paper if int(r["n_frames"]) >= args.min_frames and float(r["topk_conf"]) >= args.min_topk]
print(f"\n{args.label:<7} tracks {len(paper)}  confirmed under the frozen rule {len(pc)}")
for r in sorted(paper, key=lambda r: -float(r["topk_conf"]))[:8]:
    ok = int(r["n_frames"]) >= args.min_frames and float(r["topk_conf"]) >= args.min_topk
    print(f"  id {r['track_id']:>5} n {r['n_frames']:>4} span {float(r['span_s']):6.2f}s topk {float(r['topk_conf']):.4f}"
          f" first {r['first_frame']:>6} {'CONFIRMED' if ok else ''}")
