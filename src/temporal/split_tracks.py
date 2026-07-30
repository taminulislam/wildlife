#!/usr/bin/env python3
"""
PHASE F — recover collided deer by SPLITTING candidate tracks at teleports.

The problem (§6.4.3): 228 of 235 deer have a touching candidate but only 212 have their
OWN candidate. 16 deer share a track with another animal, and on the unseen videos that
is 10 of the 14 misses. They cannot be recovered by any keep/drop confirmer, because the
track holding them is already counted once.

The obvious fix is to learn a COUNT per track (0/1/2+) instead of a binary label. The
label distribution kills that idea on this corpus:

    count = 0 : 18 137        count = 1 : 196        count = 2 : 16

Sixteen positives, spread over 8 CV folds, is two examples per fold. A classifier cannot
learn "this track is two deer" from that, and claiming otherwise in a paper invites the
obvious question about sample size.

So this does it deterministically instead. A track that drifts from one deer onto another
has to physically jump between them, and a jump is measurable: deer move ~0.15 box-widths
per frame in this corpus (median 2.28 box-widths over a median 15-frame keyframe gap), so
a displacement of several box-widths in one or two frames is not an animal moving — it is
the tracker changing its mind about which animal it is following.

    split when   dist(centre_i, centre_i+1) / (mean_box_scale * frame_gap)  >  --thresh

Each resulting segment becomes its own candidate, so the existing binary confirmer and
the hand-tuned rule both work unchanged on the output — one parameter, no new training,
and it composes with everything already built.

The sweep prints, per threshold, how many of the 235 deer end up with their own candidate
(`primary`). That number rising toward `reached` (228) is the whole point; candidate
count rising is the cost.

Usage:
  python src/temporal/split_tracks.py --counts-dir <run> --sweep
  python src/temporal/split_tracks.py --counts-dir <run> --thresh 3.0 --out-dir <new>
"""
from __future__ import annotations
import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from label_tracks import gt_tracks_of, overlaps                        # noqa: E402


def read_tracks(counts_dir: str) -> dict[tuple, list]:
    """(video, tid) -> [(frame, conf, xc, yc, w, h), ...] sorted by frame."""
    seqs: dict[tuple, list] = defaultdict(list)
    files = sorted(glob.glob(os.path.join(counts_dir, "shard*", "tracks.csv"))) or \
        sorted(glob.glob(os.path.join(counts_dir, "tracks.csv")))
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                seqs[(r["video"], int(r["track_id"]))].append((
                    int(r["frame"]), float(r["conf"]), float(r["xc"]),
                    float(r["yc"]), float(r["w"]), float(r["h"]),
                    r.get("site", "")))
    for k in seqs:
        seqs[k].sort()
    return seqs


def split_one(seq: list, thresh: float) -> list[list]:
    """Cut a track wherever the per-frame, size-normalised displacement exceeds thresh."""
    if len(seq) < 2 or thresh <= 0:
        return [seq]
    out, cur = [], [seq[0]]
    for prev, curr in zip(seq, seq[1:]):
        gap = max(1, curr[0] - prev[0])
        scale = (math.sqrt(max(prev[4] * prev[5], 1.0)) +
                 math.sqrt(max(curr[4] * curr[5], 1.0))) / 2.0
        dist = math.hypot(curr[2] - prev[2], curr[3] - prev[3])
        if dist / (scale * gap) > thresh:
            out.append(cur); cur = [curr]
        else:
            cur.append(curr)
    out.append(cur)
    return out


def apply_split(seqs: dict, thresh: float) -> dict:
    """-> same shape as seqs, with segment index folded into a fresh track id."""
    out: dict[tuple, list] = {}
    nxt: dict[str, int] = defaultdict(int)
    for (video, tid), seq in seqs.items():
        segs = split_one(seq, thresh)
        for s in segs:
            nxt[video] += 1
            out[(video, nxt[video])] = s
    return out


def coverage(seqs: dict, gt_by_video: dict, min_overlap: int = 1):
    """-> (n_gt, reached, primary). Same matching as pool_coverage.py."""
    by_video: dict[str, list] = defaultdict(list)
    for (v, t) in seqs:
        by_video[v].append(t)
    tot_gt = tot_reached = tot_primary = 0
    for video, gts in gt_by_video.items():
        boxes: dict[int, dict] = {}
        for t in by_video.get(video, []):
            d = defaultdict(list)                 # frame -> [box, ...]; see pool_coverage
            for (fr, _cf, xc, yc, w, h, _s) in seqs[(video, t)]:
                d[fr].append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
            boxes[t] = d
        ov = defaultdict(dict)
        for t, pb in boxes.items():
            for gi, g in enumerate(gts):
                n = sum(1 for fr, bs in pb.items()
                        if fr in g and any(overlaps(b, g[fr]) for b in bs))
                if n >= min_overlap:
                    ov[t][gi] = n
        best = {}
        for gi in range(len(gts)):
            cont = [(t, d[gi]) for t, d in ov.items() if gi in d]
            if cont:
                best[gi] = max(cont, key=lambda x: x[1])[0]
        tot_gt += len(gts); tot_reached += len(best)
        tot_primary += len(set(best.values()))
    return tot_gt, tot_reached, tot_primary


def write_pool(seqs: dict, out_dir: str, fps: float = 60.0) -> None:
    """Emit shard0/{tracks,counts}.csv so every downstream tool works unchanged."""
    d = os.path.join(out_dir, "shard0")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "tracks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "site", "track_id", "frame", "t_s", "xc", "yc", "w", "h",
                    "conf", "confirmed"])
        for (video, tid), seq in sorted(seqs.items()):
            for (fr, cf, xc, yc, bw, bh, site) in seq:
                w.writerow([video, site, tid, fr, round(fr / fps, 3),
                            xc, yc, bw, bh, cf, 0])
    with open(os.path.join(d, "counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "site", "track_id", "first_frame", "last_frame", "first_s",
                    "last_s", "n_frames", "span_s", "mean_conf", "topk_conf",
                    "mean_box_px", "confirmed"])
        for (video, tid), seq in sorted(seqs.items()):
            confs = [s[1] for s in seq]
            top = sorted(confs, reverse=True)[:5]
            f0, f1 = seq[0][0], seq[-1][0]
            w.writerow([video, seq[0][6], tid, f0, f1, round(f0 / fps, 3),
                        round(f1 / fps, 3), len(seq), round((f1 - f0) / fps, 3),
                        round(sum(confs) / len(confs), 4),
                        round(sum(top) / len(top), 4),
                        round(sum(s[4] * s[5] for s in seq) / len(seq), 1), 0])
    print(f"-> {d}/  ({len(seqs)} candidate tracks)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--thresh", type=float, default=3.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    seqs = read_tracks(args.counts_dir)
    gt_by_video = {}
    for x in sorted(glob.glob(os.path.join(args.cvat_dir, "*.xml"))):
        name = os.path.splitext(os.path.basename(x))[0].replace("_annotations", "")
        gt_by_video[name] = gt_tracks_of(x)
    print(f"loaded {len(seqs)} candidate tracks, {len(gt_by_video)} videos")

    if args.sweep:
        print(f"\n{'thresh':>7} {'candidates':>11} {'reached':>8} {'primary':>8} "
              f"{'gain':>6}")
        base = None
        for t in (0.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0):
            s = apply_split(seqs, t) if t > 0 else seqs
            g, re_, pr = coverage(s, gt_by_video)
            if base is None:
                base = pr
            tag = "(no split)" if t == 0 else ""
            print(f"{t:>7.1f} {len(s):>11} {re_:>8} {pr:>8} {pr-base:>+6} {tag}")
        print(f"\nGT deer: {g}. 'primary' is what a keep/drop confirmer can reach; "
              f"'reached' ({re_}) is the pool's limit.")
        return

    s = apply_split(seqs, args.thresh)
    g, re_, pr = coverage(s, gt_by_video)
    print(f"thresh {args.thresh}: {len(s)} candidates, reached {re_}/{g}, primary {pr}/{g}")
    if args.out_dir:
        write_pool(s, args.out_dir)


if __name__ == "__main__":
    main()
