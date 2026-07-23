#!/usr/bin/env python3
"""
Convert a CVAT-for-video 1.1 export (one labeled video) -> a YOLO training set:
extract the RIGHT frames from the source video and write images + YOLO labels that
merge into data/annotate/{frames,labels}.

Why not CVAT's own YOLO export: it dumps EVERY frame of the video (tens of thousands)
as images. We instead curate:
  * POSITIVES - frames with deer boxes, subsampled (consecutive 60fps frames are near
    duplicates) but always keeping human keyframes. Multi-deer frames keep all boxes.
  * NEGATIVES - deer-free frames for hard-negative training, mined first from where the
    OLD detector false-fired (warm rocks/structures), then spread across the video.
Frames are named `<SITE>__<transect>_v<visit>_<side>_f<frame>.png` (the annotate key),
with a matching `.txt` (empty .txt = explicit negative, which build_yolo_dataset keeps).

Also emits count_gt.csv: video, site, unique_deer (= number of tracks).

Usage:
  python src/dataset/cvat_to_yolo.py \
      --xml .../GiantCityRd_..._annotations.xml \
      --source data/raw \
      --out /work/hdd/.../yolo_from_cvat/GiantCityRd \
      [--model-tracks .../counts/full_m640_TON/tracks.csv] \
      [--pos-stride 6] [--neg-per-pos 1.5] [--neg-gap 15] [--preview]
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mining"))
from filename_meta import parse_path  # noqa: E402

GREEN = (0, 255, 0)


def find_video(source: str, stem: str) -> str | None:
    # MAS Visit1/Visit2 share identical mp4 names; the CVAT task name carries a
    # _V1/_V2 suffix the raw file lacks. Strip it and constrain the search to the
    # matching visit folder so we extract from the correct video.
    visit, base = None, stem
    if base.endswith("_V1"):
        visit, base = "visit1", base[:-3]
    elif base.endswith("_V2"):
        visit, base = "visit2", base[:-3]
    for ext in ("mp4", "MP4", "avi", "mov"):
        hits = glob.glob(os.path.join(source, "**", f"{base}.{ext}"), recursive=True)
        if visit:
            hits = [h for h in hits if visit in h.lower()]
        if hits:
            return sorted(hits)[0]
    return None


def parse_cvat(xml_path: str):
    """-> (n_tracks, {frame: [(x1,y1,x2,y2), ...]}, set(keyframe_frames))."""
    root = ET.parse(xml_path).getroot()
    tracks = root.findall("track")
    by_frame: dict[int, list[tuple]] = defaultdict(list)
    keyframes: set[int] = set()
    for tr in tracks:
        if tr.get("label") != "deer":
            continue
        for b in tr.findall("box"):
            if b.get("outside") == "1":
                continue
            fi = int(b.get("frame"))
            x1, y1 = float(b.get("xtl")), float(b.get("ytl"))
            x2, y2 = float(b.get("xbr")), float(b.get("ybr"))
            by_frame[fi].append((x1, y1, x2, y2))
            if b.get("keyframe") == "1":
                keyframes.add(fi)
    deer_tracks = [t for t in tracks if t.get("label") == "deer"]
    return len(deer_tracks), by_frame, keyframes


def pick_positives(by_frame, keyframes, stride):
    """All keyframes + every `stride`-th deer frame (dedup)."""
    deer_frames = sorted(by_frame)
    sel = set(f for f in deer_frames if f in keyframes)
    sel |= set(deer_frames[::max(1, stride)])
    return sorted(sel)


def pick_negatives(n_target, all_deer_frames, nframes, model_tracks_csv, video, neg_gap):
    """Mine deer-free frames: hard negatives (old-model false fires) first, then spread."""
    forbidden = set()
    for f in all_deer_frames:
        for d in range(-neg_gap, neg_gap + 1):
            forbidden.add(f + d)

    hard = []
    if model_tracks_csv and os.path.exists(model_tracks_csv):
        with open(model_tracks_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["video"] != video:
                    continue
                fi = int(r["frame"])
                if fi not in forbidden:
                    hard.append(fi)
    # dedup, keep spread: sort unique hard-neg frames
    hard = sorted(set(hard))

    # evenly spread filler negatives across the whole video
    spread = [int(i * nframes / (n_target * 3 + 1)) for i in range(1, n_target * 3 + 1)]
    spread = [f for f in spread if f not in forbidden]

    out, seen = [], set()
    for f in hard + spread:
        if f in seen or f in forbidden:
            continue
        seen.add(f); out.append(f)
        if len(out) >= n_target:
            break
    return sorted(out), len(hard)  # second val: hard-neg pool size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-tracks", default="")
    ap.add_argument("--pos-stride", type=int, default=6)
    ap.add_argument("--neg-per-pos", type=float, default=1.5)
    ap.add_argument("--neg-gap", type=int, default=15)
    ap.add_argument("--preview", action="store_true",
                    help="also write a contact sheet of sampled pos/neg for approval")
    args = ap.parse_args()

    n_tracks, by_frame, keyframes = parse_cvat(args.xml)
    # video stem: CVAT export filename minus _annotations
    stem = os.path.splitext(os.path.basename(args.xml))[0].replace("_annotations", "")
    vpath = find_video(args.source, stem)
    if vpath is None:
        raise SystemExit(f"source video not found for {stem} under {args.source}")
    meta = parse_path(vpath)
    key = meta.key  # e.g. TON__GiantCityRd_v1_LS

    pos = pick_positives(by_frame, keyframes, args.pos_stride)
    n_neg = round(args.neg_per_pos * len(pos))
    cap = cv2.VideoCapture(vpath)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    neg, hard_pool = pick_negatives(n_neg, set(by_frame), nframes,
                                    args.model_tracks, stem, args.neg_gap)

    img_dir = os.path.join(args.out, "frames")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)

    prev_pos, prev_neg = [], []
    # Exact-frame extraction by SEQUENTIAL decode. cap.set(POS_FRAMES, n) lands on the
    # wrong frame on compressed mp4 (B-frames/GOP) -> boxes would miss fast/close deer.
    target = {f: True for f in pos}
    for f in neg:
        target.setdefault(f, False)
    W = H = 0
    n_written = n_boxes = 0
    fi = -1
    while target:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        is_pos = target.pop(fi, None)
        if is_pos is None:
            continue
        H, W = frame.shape[:2]
        name = f"{key}_f{fi}"
        cv2.imwrite(os.path.join(img_dir, f"{name}.png"), frame)
        lines = []
        if is_pos:
            for (x1, y1, x2, y2) in by_frame[fi]:
                xc = ((x1 + x2) / 2) / W; yc = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W; bh = (y2 - y1) / H
                xc = min(max(xc, 0), 1); yc = min(max(yc, 0), 1)
                lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                n_boxes += 1
        with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(lines))
        n_written += 1
        if args.preview and (is_pos or len(prev_neg) < 15):
            tile = frame.copy()
            if is_pos:
                for (x1, y1, x2, y2) in by_frame[fi]:
                    cv2.rectangle(tile, (int(x1), int(y1)), (int(x2), int(y2)),
                                  GREEN, 2)
            tag = f"f{fi} {'DEER' if is_pos else 'neg'}"
            cv2.putText(tile, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            (prev_pos if is_pos else prev_neg).append(tile)
    cap.release()
    # preview: evenly sampled positives (so all events show) first, then negatives
    if len(prev_pos) > 15:
        step = len(prev_pos) / 15
        prev_pos = [prev_pos[int(i * step)] for i in range(15)]
    prev_tiles = prev_pos + prev_neg

    # count GT row
    with open(os.path.join(args.out, "count_gt.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video", "site", "unique_deer"])
        w.writerow([stem, meta.site, n_tracks])

    if args.preview and prev_tiles:
        import math
        cols = 5; h = max(t.shape[0] for t in prev_tiles); w_ = max(t.shape[1] for t in prev_tiles)
        rows = math.ceil(len(prev_tiles) / cols)
        import numpy as np
        sheet = np.zeros((rows * h, cols * w_, 3), "uint8")
        for i, t in enumerate(prev_tiles):
            r, c = divmod(i, cols)
            sheet[r * h:r * h + t.shape[0], c * w_:c * w_ + t.shape[1]] = t
        cv2.imwrite(os.path.join(args.out, "_preview.jpg"), sheet)

    print(f"video {stem}  site {meta.site}  key {key}  ({W}x{H}, {nframes} frames)")
    print(f"  unique deer (tracks): {n_tracks}")
    print(f"  positives: {len(pos)} frames, {n_boxes} deer boxes")
    print(f"  negatives: {len(neg)} frames (hard-neg pool from model FPs: {hard_pool})")
    print(f"  wrote {n_written} images+labels -> {args.out}")
    if args.preview:
        print(f"  preview -> {os.path.join(args.out, '_preview.jpg')}")


if __name__ == "__main__":
    main()
