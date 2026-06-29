#!/usr/bin/env python3
"""
Convert count_deer.py track dumps -> a Label Studio import JSON, so the model's
tracks PRE-LOAD into Label Studio video tasks as editable annotations. The annotator
then KEEPS good ones, DELETES false positives, ADDS missed deer, FIXES bad boxes.

One Label Studio task = one video. Each model track becomes a VideoRectangle region
with a keyframe `sequence` (LS interpolates between keyframes). Boxes are stored as
PERCENTAGES of frame size, frames are 1-indexed (LS convention), so we read each
video's W/H/fps/frame-count with OpenCV to convert from the pixel xc,yc,w,h dump.

Videos are referenced via Label Studio LOCAL FILE SERVING (data stays on Delta):
  data.video = "/data/local-files/?d=<path relative to --doc-root>"
so start LS with:
  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
  LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<--doc-root>

Usage:
  python src/track/tracks_to_labelstudio.py \
      --counts-dir /work/hdd/.../counts/full_m640_TON \
      --source data/raw \
      --doc-root /work/nvme/bgte/tislam6/wildlife_project/data/raw \
      --out /work/hdd/.../ls_import/pilot_tasks.json \
      [--videos GiantCityRd_TON_12.03.25_LS,...] [--confirmed-only] [--min-conf 0.0]

Import: Label Studio project -> Import -> upload this JSON. Paste the labeling config
from docs/LABELSTUDIO_SETUP.md (the <Video>/<VideoRectangle> template).
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import cv2


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_video(source: str, stem: str) -> str | None:
    for ext in ("mp4", "MP4", "avi", "mov"):
        hits = glob.glob(os.path.join(source, "**", f"{stem}.{ext}"), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def video_meta(path: str) -> tuple[int, int, float, int]:
    c = cv2.VideoCapture(path)
    W = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = c.get(cv2.CAP_PROP_FPS) or 60.0
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.release()
    return W, H, fps, n


def short_id(stem: str, tid: int) -> str:
    """Short, stable region id (LS wants a compact alphanumeric id)."""
    base = "".join(ch for ch in stem if ch.isalnum())[-8:]
    return f"{base}{tid}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True, help="dir with tracks.csv")
    ap.add_argument("--source", default="data/raw", help="root to locate the videos")
    ap.add_argument("--doc-root", required=True,
                    help="LS LOCAL_FILES_DOCUMENT_ROOT (video paths are relative to it)")
    ap.add_argument("--out", required=True, help="output tasks JSON path")
    ap.add_argument("--videos", default="", help="comma list of stems (default: all)")
    ap.add_argument("--confirmed-only", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--label", default="deer")
    args = ap.parse_args()

    tracks_csv = os.path.join(args.counts_dir, "tracks.csv")
    if not os.path.exists(tracks_csv):
        raise SystemExit(f"missing {tracks_csv}")
    want = {v.strip() for v in args.videos.split(",") if v.strip()}
    doc_root = os.path.abspath(args.doc_root)

    by_vid: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in read_csv(tracks_csv):
        if want and r["video"] not in want:
            continue
        if args.confirmed_only and r.get("confirmed") != "1":
            continue
        by_vid[r["video"]][int(r["track_id"])].append(r)

    tasks = []
    for stem, tracks in sorted(by_vid.items()):
        vpath = find_video(args.source, stem)
        if vpath is None:
            print(f"  ! video not found for {stem}; skipping")
            continue
        vpath = os.path.abspath(vpath)
        rel = os.path.relpath(vpath, doc_root)
        if rel.startswith(".."):
            print(f"  ! {stem} is outside --doc-root; LS can't serve it. skipping")
            continue
        W, H, fps, nframes = video_meta(vpath)
        duration = (nframes / fps) if fps else 0.0

        regions = []
        for tid, rows in sorted(tracks.items()):
            if args.min_conf > 0 and max(float(x["conf"]) for x in rows) < args.min_conf:
                continue
            rows = sorted(rows, key=lambda x: int(x["frame"]))
            seq = []
            for x in rows:
                fi = int(x["frame"])
                xc, yc, w, h = (float(x["xc"]), float(x["yc"]),
                                float(x["w"]), float(x["h"]))
                seq.append({
                    "frame": fi + 1,                      # LS frames are 1-indexed
                    "enabled": True,
                    "rotation": 0,
                    "x": max(0.0, (xc - w / 2) / W * 100),
                    "y": max(0.0, (yc - h / 2) / H * 100),
                    "width": w / W * 100,
                    "height": h / H * 100,
                    "time": fi / fps if fps else 0.0,
                })
            if not seq:
                continue
            # close the track one frame after the last detection
            last = dict(seq[-1]); last["frame"] = int(rows[-1]["frame"]) + 2
            last["enabled"] = False
            seq.append(last)
            regions.append({
                "id": short_id(stem, tid),
                "type": "videorectangle",
                "from_name": "box", "to_name": "video",
                "origin": "manual",
                "value": {
                    "framesCount": nframes, "duration": duration,
                    "sequence": seq, "labels": [args.label],
                },
            })

        tasks.append({
            "data": {"video": f"/data/local-files/?d={rel}",
                     "video_stem": stem, "fps": round(fps, 3)},
            "annotations": [{"result": regions}],
        })
        print(f"  {stem}: {len(regions)} tracks  ({W}x{H} {fps:.1f}fps {nframes}f)  rel={rel}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(tasks, f, indent=1)
    print(f"\nwrote {len(tasks)} task(s) -> {args.out}")
    if tasks:
        print(f"set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={doc_root}")


if __name__ == "__main__":
    main()
