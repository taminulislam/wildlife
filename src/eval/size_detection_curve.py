#!/usr/bin/env python3
"""
P(detected) and P(counted) as a function of an animal's apparent size in pixels.

Box size is the pipeline's only proxy for range: a deer's pixel extent falls with its
distance from the camera. So P(counted | size) is the empirical detection function a
distance-sampling abundance estimate needs, and the question a survey ecologist asks when
deciding how to scale per-transect counts up to an area.

Per ground-truth animal:
  size      median sqrt(box area) over its visible frames, in pixels
  detected  some candidate track touches it on at least one frame
  counted   some candidate the frozen rule ACCEPTS touches it (identity-matched counting)

Usage:
  python src/eval/size_detection_curve.py --counts-dir <pool C run> --out results/counting_eval
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "temporal"))
from label_tracks import gt_tracks_of, overlaps                       # noqa: E402

M_MIN, S_MIN, C_MIN = 20, 0.0, 0.65          # the published rule, frozen


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--splits", default="data/temporal/video_splits.json")
    ap.add_argument("--scope", default="heldout", choices=["heldout", "all"])
    ap.add_argument("--out", default="results/counting_eval")
    args = ap.parse_args()

    split = json.load(open(args.splits))
    keep = (lambda v: split.get(v) in ("val", "test")) if args.scope == "heldout" else (lambda v: True)

    # candidate tracks: accepted or not, from the per-track summary
    accepted: dict[tuple, bool] = {}
    for f in glob.glob(os.path.join(args.counts_dir, "counts_shard*.csv")):
        for r in csv.DictReader(open(f)):
            accepted[(r["video"], int(r["track_id"]))] = (
                int(r["n_frames"]) >= M_MIN and float(r["span_s"]) >= S_MIN
                and float(r["topk_conf"]) >= C_MIN)
    # per-frame boxes of every candidate
    boxes: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(args.counts_dir, "shard*", "tracks.csv")):
        for r in csv.DictReader(open(f)):
            xc, yc, w, h = (float(r[k]) for k in ("xc", "yc", "w", "h"))
            boxes[(r["video"], int(r["track_id"]))][int(r["frame"])].append(
                (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    by_video = defaultdict(list)
    for (v, t) in boxes:
        by_video[v].append(t)

    animals = []
    for x in sorted(glob.glob(os.path.join(args.cvat_dir, "*.xml"))):
        video = os.path.splitext(os.path.basename(x))[0].replace("_annotations", "")
        if not keep(video):
            continue
        for g in gt_tracks_of(x):
            if not g:
                continue                           # never visible: cannot be detected
            sides = [math.sqrt(max(0.0, (b[2] - b[0]) * (b[3] - b[1]))) for b in g.values()]
            size = statistics.median(sides)
            det = cnt = False
            for t in by_video.get(video, []):
                pb = boxes[(video, t)]
                if any(fr in g and any(overlaps(b, g[fr]) for b in bs) for fr, bs in pb.items()):
                    det = True
                    if accepted.get((video, t)):
                        cnt = True
                        break
            animals.append({"video": video, "size_px": round(size, 1),
                            "frames": len(g), "detected": int(det), "counted": int(cnt)})

    os.makedirs(args.out, exist_ok=True)
    per = os.path.join(args.out, f"size_curve_animals_{args.scope}.csv")
    with open(per, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(animals[0]))
        w.writeheader(); w.writerows(animals)

    edges = [0, 20, 30, 40, 60, 100, 10_000]
    rows = []
    print(f"{args.scope}: {len(animals)} animals")
    print(f"  {'size (px)':>12s} {'n':>4s} {'P(detected)':>18s} {'P(counted)':>18s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [a for a in animals if lo <= a["size_px"] < hi]
        n = len(sel)
        if not n:
            continue
        d = sum(a["detected"] for a in sel); c = sum(a["counted"] for a in sel)
        dl, dh = wilson(d, n); cl, ch = wilson(c, n)
        lab = f"{lo}-{hi}" if hi < 10_000 else f">={lo}"
        print(f"  {lab:>12s} {n:4d}   {d/n:5.2f} [{dl:.2f},{dh:.2f}]   {c/n:5.2f} [{cl:.2f},{ch:.2f}]")
        rows.append({"size_bin": lab, "n": n, "p_detected": round(d/n, 3),
                     "det_lo": round(dl, 3), "det_hi": round(dh, 3),
                     "p_counted": round(c/n, 3), "cnt_lo": round(cl, 3), "cnt_hi": round(ch, 3)})
    summ = os.path.join(args.out, f"size_curve_{args.scope}.csv")
    with open(summ, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    tot_d = sum(a["detected"] for a in animals); tot_c = sum(a["counted"] for a in animals)
    print(f"  overall: detected {tot_d}/{len(animals)}  counted {tot_c}/{len(animals)}")
    print(f"-> {per}\n-> {summ}")


if __name__ == "__main__":
    main()
