#!/usr/bin/env python3
"""
PHASE C step 1 — turn candidate tracks into SUPERVISED training data for the temporal
counting head.

The counting run emits ~984 candidate tracks (tracks.csv) for 236 real deer. The
hand-tuned rule decides which to keep and gets MAE 1.88, under-counting by 48 deer.
To learn that decision we need a label per candidate track. We derive it by matching
each candidate against the CVAT ground-truth tracks:

  primary   (label 1) - the candidate that covers a GT deer BEST. Exactly one per GT
                        deer, so "count = number of primaries" is correct by construction.
  duplicate (label 0) - also lands on that deer, but a *fragment* of an already-covered
                        animal. Counting it would over-count.
  false     (label 0) - overlaps no GT deer at all (warm rock, structure, noise).

Labelling duplicates as negative is what lets a single confirmation head fix BOTH
failure modes at once: it learns to keep one track per animal, so fragmentation is
handled without a separate re-ID stage.

Matching uses the project's counting criterion (ANY overlap between a predicted box and
a GT box on the same frame) — box tightness is irrelevant for counting.

Usage:
  python src/temporal/label_tracks.py \
      --counts-dir /work/hdd/.../counts/phaseB_yolo11m_conf0.10 \
      --cvat-dir data/cvat_export --source data/raw \
      --out data/temporal/tracks_labelled.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "dataset"))
sys.path.insert(0, os.path.join(_HERE, "..", "mining"))


def gt_tracks_of(xml_path: str) -> list[dict[int, tuple]]:
    """-> one dict {frame: (x1,y1,x2,y2)} per GT deer track."""
    root = ET.parse(xml_path).getroot()
    out = []
    for tr in root.findall("track"):
        if tr.get("label") != "deer":
            continue
        boxes = {}
        for b in tr.findall("box"):
            if b.get("outside") == "1":
                continue
            boxes[int(b.get("frame"))] = (
                float(b.get("xtl")), float(b.get("ytl")),
                float(b.get("xbr")), float(b.get("ybr")))
        if boxes:
            out.append(boxes)
    return out


def overlaps(a, b) -> bool:
    """ANY overlap — the counting criterion."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--out", default="data/temporal/tracks_labelled.csv")
    ap.add_argument("--min-overlap-frames", type=int, default=1,
                    help="frames of overlap needed to associate a candidate with a GT deer")
    args = ap.parse_args()

    # ---- predicted tracks: {(video, tid): {frame: [box, ...]}} ----
    # A LIST per frame: the orphan linker can put several simultaneous detections in one
    # pseudo-track, and keying frame -> box kept only whichever row was read last (27 655
    # boxes dropped on the Phase-E pool, survivor decided by CSV order).
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

    # ---- per-track summary stats (the hand-tuned rule's features) ----
    stats: dict[tuple, dict] = {}
    for f in sorted(glob.glob(os.path.join(args.counts_dir, "counts_shard*.csv"))) or \
             sorted(glob.glob(os.path.join(args.counts_dir, "shard*", "counts.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                stats[(r["video"], int(r["track_id"]))] = r

    # ---- GT per video ----
    gt_by_video: dict[str, list] = {}
    for x in sorted(glob.glob(os.path.join(args.cvat_dir, "*.xml"))):
        name = os.path.splitext(os.path.basename(x))[0].replace("_annotations", "")
        gt_by_video[name] = gt_tracks_of(x)

    rows = []
    n_primary = n_dup = n_false = 0
    for video, gts in sorted(gt_by_video.items()):
        cands = [(v, t) for (v, t) in pred if v == video]
        # overlap count between every candidate and every GT deer
        ov = defaultdict(dict)          # cand -> {gt_idx: n_frames}
        for c in cands:
            pboxes = pred[c]
            for gi, g in enumerate(gts):
                n = sum(1 for fr, bs in pboxes.items()
                        if fr in g and any(overlaps(b, g[fr]) for b in bs))
                if n >= args.min_overlap_frames:
                    ov[c][gi] = n
        # each GT deer awards ONE primary: the candidate covering it best
        best_for_gt = {}
        for gi in range(len(gts)):
            contenders = [(c, d[gi]) for c, d in ov.items() if gi in d]
            if contenders:
                best_for_gt[gi] = max(contenders, key=lambda t: t[1])[0]
        primaries = set(best_for_gt.values())

        for c in cands:
            matched = ov.get(c, {})
            if not matched:
                label, kind = 0, "false"; n_false += 1
            elif c in primaries:
                label, kind = 1, "primary"; n_primary += 1
            else:
                label, kind = 0, "duplicate"; n_dup += 1
            s = stats.get(c, {})
            rows.append({
                "video": video, "site": s.get("site", ""), "track_id": c[1],
                "label": label, "kind": kind,
                "n_frames": s.get("n_frames", len(pred[c])),
                "span_s": s.get("span_s", ""),
                "mean_conf": s.get("mean_conf", ""),
                "topk_conf": s.get("topk_conf", ""),
                "mean_box_px": s.get("mean_box_px", ""),
                "first_frame": s.get("first_frame", min(pred[c])),
                "last_frame": s.get("last_frame", max(pred[c])),
                "gt_overlap_frames": max(matched.values()) if matched else 0,
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    total_gt = sum(len(g) for g in gt_by_video.values())
    print(f"candidate tracks : {len(rows)}")
    print(f"  primary  (=1)  : {n_primary}   <- ideal count would be {total_gt}")
    print(f"  duplicate(=0)  : {n_dup}   (fragments of an already-covered deer)")
    print(f"  false    (=0)  : {n_false}")
    print(f"\nCEILING: a perfect confirmation head scores {n_primary}/{total_gt} deer "
          f"= {100*n_primary/total_gt:.1f}% (the rest are never detected at all)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
