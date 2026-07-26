#!/usr/bin/env python3
"""
Label every predicted candidate track against the CVAT ground truth — the training set
for the temporal counting head (Phase C).

The head has three outputs, and each needs supervision that only this matching provides:

  1. CONFIRMATION  is this candidate a real deer?      -> `is_real`
  2. RE-ID         which GT animal is it?              -> `gt_track` (+ gt_coverage.csv
                                                          tells us which GT animals were
                                                          split across several candidates)
  3. MULTIPLICITY  how many animals are in this blob?  -> `multiplicity`

Matching uses the COUNTING criterion (any overlap / centre-in-box), not IoU>=0.5,
consistent with docs/RESULTS_LOG.md §4.3: for counting, a box touching the animal is a
hit. A predicted track is matched to a GT track when they co-occur on at least
`--min-match-frames` frames.

Note on label noise: 94% of GT boxes are CVAT-interpolated and drift a median 2.28
box-widths between keyframes (§1.1). `--keyframes-only` restricts matching to
human-placed boxes — fewer frames to match on, but no drift. Both label sets are
written so the head can be trained on either and the choice ablated.

Outputs (to --out):
  track_labels.csv   one row per predicted track: features + is_real/gt_track/multiplicity
  gt_coverage.csv    one row per GT deer: how many candidates cover it (fragmentation)
  label_summary.md   class balance + fragmentation/merge statistics

Usage:
  python src/temporal/build_track_labels.py --counts results/counts/yolov9m_1280_conf0.10
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
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))
from track_recall import iou_xyxy, center_inside  # noqa: E402


def parse_cvat_tracks(xml_path: str, keyframes_only: bool = False):
    """-> list of {frame: (x1,y1,x2,y2)}, one dict per GT deer track.

    cvat_to_yolo.parse_cvat() drops track identity (it returns boxes pooled per frame),
    but re-ID and multiplicity supervision need to know WHICH animal each box belongs to.
    """
    root = ET.parse(xml_path).getroot()
    out = []
    for tr in root.findall("track"):
        if tr.get("label") != "deer":
            continue
        boxes = {}
        for b in tr.findall("box"):
            if b.get("outside") == "1":
                continue
            if keyframes_only and b.get("keyframe") != "1":
                continue
            boxes[int(b.get("frame"))] = (
                float(b.get("xtl")), float(b.get("ytl")),
                float(b.get("xbr")), float(b.get("ybr")))
        if boxes:
            out.append(boxes)
    return out


def load_pred_tracks(counts_dir: str):
    """tracks_*.csv -> {video: {track_id: {frame: (box, conf)}}}"""
    per = defaultdict(lambda: defaultdict(dict))
    files = sorted(glob.glob(os.path.join(counts_dir, "tracks_*.csv")))
    if not files:
        raise SystemExit(f"no tracks_*.csv under {counts_dir}")
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                xc, yc = float(r["xc"]), float(r["yc"])
                w, h = float(r["w"]), float(r["h"])
                per[r["video"]][int(r["track_id"])][int(r["frame"])] = (
                    (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2), float(r["conf"]))
    return per


def overlaps(a, b) -> bool:
    """Counting criterion: any overlap, or one centre inside the other."""
    return iou_xyxy(a, b) > 0 or center_inside(a, b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True, help="dir with tracks_*.csv")
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--out", default=None, help="default: <counts>/labels")
    ap.add_argument("--min-match-frames", type=int, default=3,
                    help="frames of co-occurrence before a candidate is called real")
    ap.add_argument("--keyframes-only", action="store_true",
                    help="match only against human-placed boxes (no interpolation drift)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.counts, "labels")
    os.makedirs(out, exist_ok=True)

    pred = load_pred_tracks(args.counts)
    rows, cov_rows = [], []
    tot_real = tot_false = tot_merged = 0

    for xml in sorted(f for f in os.listdir(args.cvat_dir) if f.endswith(".xml")):
        stem = os.path.splitext(xml)[0].replace("_annotations", "")
        gts = parse_cvat_tracks(os.path.join(args.cvat_dir, xml), args.keyframes_only)
        ptracks = pred.get(stem, {})
        # hits[(pred_id, gt_idx)] = frames of co-occurrence
        hits = defaultdict(int)
        for pid, frames in ptracks.items():
            for fi, (pbox, _c) in frames.items():
                for gi, gboxes in enumerate(gts):
                    gb = gboxes.get(fi)
                    if gb is not None and overlaps(gb, pbox):
                        hits[(pid, gi)] += 1

        covered = defaultdict(list)          # gt_idx -> [pred ids]
        for pid, frames in ptracks.items():
            matches = {gi: n for (p, gi), n in hits.items()
                       if p == pid and n >= args.min_match_frames}
            best_gi = max(matches, key=matches.get) if matches else -1
            confs = [c for (_b, c) in frames.values()]
            fr = sorted(frames)
            is_real = int(best_gi >= 0)
            mult = len(matches)              # >1 == merged blob (multiplicity target)
            tot_real += is_real; tot_false += 1 - is_real
            tot_merged += int(mult > 1)
            for gi in matches:
                covered[gi].append(pid)
            rows.append({
                "video": stem, "track_id": pid,
                "n_frames": len(frames),
                "first_frame": fr[0], "last_frame": fr[-1],
                "span_frames": fr[-1] - fr[0] + 1,
                "mean_conf": round(sum(confs) / len(confs), 4),
                "max_conf": round(max(confs), 4),
                "is_real": is_real,
                "gt_track": best_gi,
                "match_frames": matches.get(best_gi, 0),
                "multiplicity": max(1, mult) if is_real else 0,
            })
        for gi in range(len(gts)):
            ps = covered.get(gi, [])
            cov_rows.append({"video": stem, "gt_track": gi,
                             "n_pred_tracks": len(ps),
                             "detected": int(len(ps) > 0),
                             "fragmented": int(len(ps) > 1)})

    if not rows:
        raise SystemExit("no predicted tracks matched any video — check --counts dir")

    for name, data in (("track_labels.csv", rows), ("gt_coverage.csv", cov_rows)):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)

    n_gt = len(cov_rows)
    det = sum(r["detected"] for r in cov_rows)
    frag = sum(r["fragmented"] for r in cov_rows)
    frag_tracks = sum(r["n_pred_tracks"] for r in cov_rows if r["fragmented"])
    md = [
        "# Temporal-head training labels",
        "",
        f"Matching: counting criterion (any overlap), >= {args.min_match_frames} "
        f"co-occurring frames{' , HUMAN KEYFRAMES ONLY' if args.keyframes_only else ''}.",
        "",
        "## Confirmation head (is_real)",
        "",
        f"| candidates | real | false | % false |",
        "|---|---|---|---|",
        f"| {len(rows)} | {tot_real} | {tot_false} | "
        f"{100*tot_false/len(rows):.1f}% |",
        "",
        "The false candidates are exactly what the hand-tuned rule tries to remove and",
        "what the learned confirmation head must remove better.",
        "",
        "## Re-ID head (fragmentation)",
        "",
        f"| GT deer | detected | fragmented across >1 candidate | candidates covering them |",
        "|---|---|---|---|",
        f"| {n_gt} | {det} ({100*det/n_gt:.1f}%) | {frag} ({100*frag/n_gt:.1f}%) | "
        f"{frag_tracks} |",
        "",
        f"Fragmentation is the over-count mechanism: {frag} animals produce "
        f"{frag_tracks} candidate tracks, i.e. {frag_tracks - frag} spurious extra counts",
        "unless the re-ID head merges them.",
        "",
        "## Multiplicity head (merged blobs)",
        "",
        f"{tot_merged} candidates overlap more than one GT animal — the under-count",
        "mechanism the multiplicity head must recover.",
        "",
    ]
    with open(os.path.join(out, "label_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
