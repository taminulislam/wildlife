#!/usr/bin/env python3
"""
Convert count_deer.py track dumps -> CVAT-for-video 1.1 XML, one file per video,
so the model's tracks can be PRE-LOADED into CVAT as editable tracks. The annotator
then KEEPS good ones, DELETES false positives, ADDS missed deer, FIXES mislocalized
boxes -- far less work than drawing every deer from scratch.

Each model track becomes a CVAT <track label="deer">. Every observed frame is written
as a keyframe (outside=0); a terminating outside=1 box closes the track at last+1.
We mark every frame a keyframe on purpose: the raw detections are jumpy, so we do NOT
want CVAT to interpolate across gaps -- the annotator prunes/keeps explicit boxes.

Reads a count run dir (has tracks.csv from the updated count_deer.py).
Writes <out>/<stem>.xml -- upload one per CVAT task (one task = one video).

Usage:
  python src/track/tracks_to_cvat.py \
      --counts-dir /work/hdd/.../counts/full_m640_TON \
      --out /work/hdd/.../cvat_preload/full_m640_TON \
      [--videos GiantCityRd_TON_12.03.25_LS,...] \
      [--confirmed-only] [--min-conf 0.0] [--label deer]

Import in CVAT: open the task -> Menu -> Upload annotations -> "CVAT for video 1.1".
"""
from __future__ import annotations
import argparse
import csv
import os
from collections import defaultdict
from xml.sax.saxutils import escape


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def box_xtlbr(xc, yc, w, h):
    return xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0


def write_video_xml(path: str, label: str, tracks: dict[int, list[dict]]) -> int:
    """tracks: {orig_track_id: [row,...]} -> CVAT for video 1.1 XML. Returns n_tracks."""
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<annotations>",
        "  <version>1.1</version>",
        "  <meta><task><labels>",
        f"    <label><name>{escape(label)}</name><attributes></attributes></label>",
        "  </labels></task></meta>",
    ]
    out_id = 0
    for _orig, rows in sorted(tracks.items(), key=lambda kv: int(kv[0])):
        rows = sorted(rows, key=lambda r: int(r["frame"]))
        lines.append(f'  <track id="{out_id}" label="{escape(label)}" source="manual">')
        last_frame = 0
        for r in rows:
            fi = int(r["frame"])
            last_frame = max(last_frame, fi)
            x1, y1, x2, y2 = box_xtlbr(
                float(r["xc"]), float(r["yc"]), float(r["w"]), float(r["h"]))
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            lines.append(
                f'    <box frame="{fi}" outside="0" occluded="0" keyframe="1" '
                f'xtl="{x1:.2f}" ytl="{y1:.2f}" xbr="{x2:.2f}" ybr="{y2:.2f}">'
                f'</box>')
        # terminate the track so it doesn't persist to the end of the video
        lines.append(
            f'    <box frame="{last_frame + 1}" outside="1" occluded="0" '
            f'keyframe="1" xtl="{x1:.2f}" ytl="{y1:.2f}" xbr="{x2:.2f}" '
            f'ybr="{y2:.2f}"></box>')
        lines.append("  </track>")
        out_id += 1
    lines.append("</annotations>\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return out_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True, help="dir with tracks.csv")
    ap.add_argument("--out", required=True, help="output dir for per-video XML")
    ap.add_argument("--videos", default="",
                    help="comma list of video stems to export (default: all)")
    ap.add_argument("--confirmed-only", action="store_true",
                    help="only export tracks the model confirmed (fewer FPs to delete)")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="drop tracks whose max frame conf < this (prune weak FPs)")
    ap.add_argument("--label", default="deer")
    args = ap.parse_args()

    tracks_csv = os.path.join(args.counts_dir, "tracks.csv")
    if not os.path.exists(tracks_csv):
        raise SystemExit(f"missing {tracks_csv}")
    want = {v.strip() for v in args.videos.split(",") if v.strip()}

    # group rows by (video, track_id)
    by_vid: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in read_csv(tracks_csv):
        if want and r["video"] not in want:
            continue
        if args.confirmed_only and r.get("confirmed") != "1":
            continue
        by_vid[r["video"]][int(r["track_id"])].append(r)

    if args.min_conf > 0:
        for vid, tracks in list(by_vid.items()):
            for tid in list(tracks):
                if max(float(r["conf"]) for r in tracks[tid]) < args.min_conf:
                    del tracks[tid]

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for vid, tracks in sorted(by_vid.items()):
        if not tracks:
            continue
        p = os.path.join(args.out, f"{vid}.xml")
        n = write_video_xml(p, args.label, tracks)
        total += n
        print(f"  {vid}: {n} tracks -> {p}")
    print(f"\nwrote {total} pre-load tracks across {len(by_vid)} video(s) -> {args.out}")


if __name__ == "__main__":
    main()
