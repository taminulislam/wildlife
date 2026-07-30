#!/usr/bin/env python3
"""
PHASE F step 1 — label each candidate track with HOW MANY deer it should count as.

label_tracks.py asks a binary question ("is this candidate a deer?") and that question
has a ceiling built into it. On the 13 unseen videos, 14 deer are missed and 10 of them
SHARE a candidate track with another deer: a track that drifts from one animal onto
another, or covers two deer walking together. Those deer are not undetected — they are
inside a candidate that is already in the pool and already counted once. No confirmation
head that emits keep/drop can ever reach them.

So the label here is a COUNT, not a class:

    count(candidate) = #{ GT deer for which this candidate is the best-covering candidate }

  0  covers no deer, or a deer that some other candidate covers better  (false/duplicate)
  1  the canonical track of exactly one deer                            (primary)
  2+ one track spanning several deer                                    <- the new signal

Summing this over a video's candidates reproduces the reachable count exactly, so
"count = sum of predicted counts" is correct by construction — the same property that
made `primary` a valid target, but without collapsing collisions.

This also explains why six learned confirmers lost to a 3-parameter rule. Duplicates sat
between primaries and false candidates in every feature (median conf 0.42 vs 0.72/0.20),
which reads as contradictory supervision. Part of that contradiction was mislabelling:
tracks carrying two deer were being taught to look like the single-deer class.

Matching is the project's any-overlap criterion, identical to label_tracks.py, so the two
label sets are directly comparable on the same pool.

Usage:
  python src/temporal/label_track_counts.py \
      --counts-dir <run> --cvat-dir data/cvat_export \
      --out data/temporal/track_counts.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from label_tracks import gt_tracks_of, overlaps                        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--out", default="data/temporal/track_counts.csv")
    ap.add_argument("--min-overlap-frames", type=int, default=1)
    args = ap.parse_args()

    # ---- candidate tracks: {(video, tid): {frame: [box, ...]}} ----
    pred: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(args.counts_dir, "shard*", "tracks.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                xc, yc = float(r["xc"]), float(r["yc"])
                w, h = float(r["w"]), float(r["h"])
                pred[(r["video"], int(r["track_id"]))][int(r["frame"])].append(
                    (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    if not pred:
        raise SystemExit(f"no shard*/tracks.csv under {args.counts_dir}")

    # ---- per-track summary stats (the rule's features) ----
    stats: dict[tuple, dict] = {}
    for f in sorted(glob.glob(os.path.join(args.counts_dir, "counts_shard*.csv"))) or \
             sorted(glob.glob(os.path.join(args.counts_dir, "shard*", "counts.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                stats[(r["video"], int(r["track_id"]))] = r

    by_video: dict[str, list] = defaultdict(list)
    for (v, t) in pred:
        by_video[v].append(t)

    rows = []
    hist: dict[int, int] = defaultdict(int)
    tot_gt = tot_reached = 0
    for x in sorted(glob.glob(os.path.join(args.cvat_dir, "*.xml"))):
        video = os.path.splitext(os.path.basename(x))[0].replace("_annotations", "")
        gts = gt_tracks_of(x)
        cands = [(video, t) for t in by_video.get(video, [])]

        ov = defaultdict(dict)                    # cand -> {gt_idx: n_overlap_frames}
        for c in cands:
            pboxes = pred[c]
            for gi, g in enumerate(gts):
                n = sum(1 for fr, bs in pboxes.items()
                        if fr in g and any(overlaps(b, g[fr]) for b in bs))
                if n >= args.min_overlap_frames:
                    ov[c][gi] = n

        # every reachable deer awards itself to its best-covering candidate; a candidate
        # may win several deer, and THAT is the count we want to predict
        owns: dict[tuple, int] = defaultdict(int)
        for gi in range(len(gts)):
            contenders = [(c, d[gi]) for c, d in ov.items() if gi in d]
            if contenders:
                owns[max(contenders, key=lambda t: t[1])[0]] += 1
        tot_gt += len(gts); tot_reached += sum(owns.values())

        for c in cands:
            k = owns.get(c, 0)
            hist[min(k, 3)] += 1
            s = stats.get(c, {})
            rows.append({
                "video": video, "site": s.get("site", ""), "track_id": c[1],
                "count": k,
                "touches_n_deer": len(ov.get(c, {})),
                "n_frames": s.get("n_frames", len(pred[c])),
                "span_s": s.get("span_s", ""),
                "mean_conf": s.get("mean_conf", ""),
                "topk_conf": s.get("topk_conf", ""),
                "mean_box_px": s.get("mean_box_px", "")})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"candidate tracks : {len(rows)}")
    for k in sorted(hist):
        lab = f"{k}+" if k == 3 else str(k)
        print(f"  count = {lab:<3}      : {hist[k]}")
    multi = sum(v for k, v in hist.items() if k >= 2)
    print(f"\ntracks carrying >1 deer: {multi}")
    print(f"deer reachable via these labels: {tot_reached}/{tot_gt} "
          f"({100*tot_reached/max(tot_gt,1):.1f}%)  <- vs the binary target's ceiling")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
