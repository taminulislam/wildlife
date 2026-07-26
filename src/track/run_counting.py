#!/usr/bin/env python3
"""
Run detect -> track -> count over the 32 GT-labelled videos and emit counts that are
1:1 comparable with data/annotate_v2/count_gt.csv.

Why a driver instead of `count_deer.py --source data/raw`:
  * The canonical video set is defined by the 32 CVAT exports, not by whatever mp4s
    happen to sit under data/raw.
  * MAS Visit1/Visit2 share IDENTICAL mp4 filenames. Deriving the video id from the
    filename collides the two visits; `find_video()` disambiguates using the CVAT
    task name's _V1/_V2 suffix, so we key every row by the CVAT stem instead.
  * Sharding: one GPU per shard so 32 long videos finish in ~1/N the wall-clock.

Outputs (per shard, merged later by count_eval.py):
  counts_<shard>.csv   one row per CANDIDATE track (confirmation applied post-hoc)
  tracks_<shard>.csv   per-frame boxes, for evidence rendering / temporal-head features

Detection runs at LOW conf on purpose: §3 of docs/RESULTS_LOG.md shows the regime is
recall-limited (FP/background-frame ~0.000-0.005), so we over-generate candidates and
let the confirmation rule (or, later, the learned temporal head) do the filtering.

Usage:
  python src/track/run_counting.py --weights <best.pt> --out results/counts/run \
      --shard 0 --nshards 4
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
import time
from argparse import Namespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "dataset"))
from count_deer import count_video  # noqa: E402
from cvat_to_yolo import find_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arch", default="yolo", choices=["yolo", "rtdetr"])
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--out", default="results/counts/run")
    ap.add_argument("--tracker", default=os.path.join(_HERE, "botsort_deer.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--contrast", default="clahe", choices=["clahe", "stretch", "none"])
    ap.add_argument("--conf", type=float, default=0.10,
                    help="LOW by design: over-generate candidates, confirm downstream")
    ap.add_argument("--iou", type=float, default=0.5)
    # confirmation defaults only fill the `confirmed` column; count_eval.py re-applies
    # arbitrary rules post-hoc from counts.csv, so these are not the final word.
    ap.add_argument("--min-hits", type=int, default=8)
    ap.add_argument("--min-span-s", type=float, default=0.3)
    ap.add_argument("--conf-track", type=float, default=0.30)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    from ultralytics import YOLO, RTDETR
    model = (RTDETR if args.arch == "rtdetr" else YOLO)(args.weights)

    xmls = sorted(f for f in os.listdir(args.cvat_dir) if f.endswith(".xml"))
    mine = [x for i, x in enumerate(xmls) if i % args.nshards == args.shard]
    os.makedirs(args.out, exist_ok=True)
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} of {len(xmls)} videos",
          flush=True)

    all_rows, all_tracks, summaries = [], [], []
    for i, xml in enumerate(mine, 1):
        stem = os.path.splitext(xml)[0].replace("_annotations", "")
        vpath = find_video(args.source, stem)
        if vpath is None:
            print(f"[{i}/{len(mine)}] {stem}: VIDEO NOT FOUND — skipped", flush=True)
            continue
        t0 = time.time()
        sub = Namespace(**vars(args))
        sub.save_video = False
        rows, summary, track_rows = count_video(model, vpath, sub)
        # key every row by the CVAT stem so MAS V1/V2 stay distinct and the join with
        # count_gt.csv is exact
        for r in rows:
            r["video"] = stem
        for r in track_rows:
            r["video"] = stem
        summary["video"] = stem
        summary["seconds"] = round(time.time() - t0, 1)
        all_rows += rows; all_tracks += track_rows; summaries.append(summary)
        print(f"[{i}/{len(mine)}] {stem}: {summary['candidate_tracks']} candidate tracks, "
              f"{summary['count']} confirmed (default rule), {summary['seconds']}s",
              flush=True)

    def dump(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"-> {path} ({len(rows)} rows)", flush=True)

    s = args.shard
    dump(os.path.join(args.out, f"counts_{s}.csv"), all_rows)
    dump(os.path.join(args.out, f"tracks_{s}.csv"), all_tracks)
    dump(os.path.join(args.out, f"summary_{s}.csv"), summaries)


if __name__ == "__main__":
    main()
