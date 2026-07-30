#!/usr/bin/env python3
"""
How many GT deer does a candidate pool actually REACH?

label_tracks.py reports its `primary` count as the ceiling, but `primary` is a set of
CANDIDATES, not a set of deer:

    best_for_gt[gi] = <candidate that covers deer gi best>
    primaries       = set(best_for_gt.values())      # <-- collapses collisions

If one candidate track is the best match for TWO deer — a track that drifts from one
animal onto another, or two deer walking close together — the set collapses it to a
single entry and both deer are billed as one. The reported ceiling is then lower than
the pool's true reach, and the missing deer look undetectable when they are not.

This reports both numbers, plus the collisions that separate them:

  reached   deer with >=1 touching candidate   <- the true ceiling for a head that may
                                                  also SPLIT a track into two counts
  primaries distinct candidates awarded        <- the ceiling if every count must be a
                                                  whole candidate track (what label_tracks
                                                  reports, and what the current rule can do)

Usage:
  python src/eval/pool_coverage.py --counts-dir <run> [--cvat-dir data/cvat_export]
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "temporal"))
from label_tracks import gt_tracks_of, overlaps                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--min-overlap-frames", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # frame -> LIST of boxes, not frame -> box. A candidate can legitimately hold several
    # boxes in one frame (the orphan linker groups simultaneous detections), and keying by
    # frame silently kept only the last row read — 27 655 of 117 978 boxes discarded on the
    # Phase-E pool, with the survivor decided by CSV row order.
    pred: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(args.counts_dir, "shard*", "tracks.csv"))) or \
        sorted(glob.glob(os.path.join(args.counts_dir, "tracks.csv")))
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                xc, yc = float(r["xc"]), float(r["yc"])
                w, h = float(r["w"]), float(r["h"])
                pred[(r["video"], int(r["track_id"]))][int(r["frame"])].append(
                    (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    if not pred:
        raise SystemExit(f"no tracks.csv under {args.counts_dir}")

    by_video: dict[str, list] = defaultdict(list)
    for (v, t) in pred:
        by_video[v].append(t)

    tot_gt = tot_reached = tot_primary = tot_collide = 0
    rows = []
    for x in sorted(glob.glob(os.path.join(args.cvat_dir, "*.xml"))):
        video = os.path.splitext(os.path.basename(x))[0].replace("_annotations", "")
        gts = gt_tracks_of(x)
        cands = [(video, t) for t in by_video.get(video, [])]
        ov = defaultdict(dict)
        for c in cands:
            pboxes = pred[c]
            for gi, g in enumerate(gts):
                n = sum(1 for fr, bs in pboxes.items()
                        if fr in g and any(overlaps(b, g[fr]) for b in bs))
                if n >= args.min_overlap_frames:
                    ov[c][gi] = n
        best_for_gt = {}
        for gi in range(len(gts)):
            contenders = [(c, d[gi]) for c, d in ov.items() if gi in d]
            if contenders:
                best_for_gt[gi] = max(contenders, key=lambda t: t[1])[0]
        reached = len(best_for_gt)
        primaries = len(set(best_for_gt.values()))
        owns = defaultdict(int)
        for c in best_for_gt.values():
            owns[c] += 1
        collide = sum(n - 1 for n in owns.values() if n > 1)
        tot_gt += len(gts); tot_reached += reached
        tot_primary += primaries; tot_collide += collide
        rows.append([video, len(gts), reached, primaries, collide, len(cands)])

    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'video':<{w}} {'GT':>4} {'reached':>8} {'primary':>8} {'lost to':>8} "
          f"{'cands':>7}")
    print(f"{'':<{w}} {'':>4} {'':>8} {'':>8} {'collide':>8} {'':>7}")
    for r in rows:
        print(f"{r[0]:<{w}} {r[1]:>4} {r[2]:>8} {r[3]:>8} {r[4]:>8} {r[5]:>7}")
    print(f"\n{'TOTAL':<{w}} {tot_gt:>4} {tot_reached:>8} {tot_primary:>8} "
          f"{tot_collide:>8} {sum(r[5] for r in rows):>7}")
    print(f"\nreached   {tot_reached}/{tot_gt} = {100*tot_reached/max(tot_gt,1):.1f}%  "
          f"<- deer with at least one touching candidate")
    print(f"primary   {tot_primary}/{tot_gt} = {100*tot_primary/max(tot_gt,1):.1f}%  "
          f"<- distinct candidates (what label_tracks reports)")
    print(f"gap       {tot_reached - tot_primary} deer share a candidate with another "
          f"deer and are invisible to a one-count-per-track head")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["video", "gt", "reached", "primary", "lost_to_collision",
                         "n_candidates"])
            cw.writerows(rows)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
