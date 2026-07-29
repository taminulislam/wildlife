#!/usr/bin/env python3
"""
Frame-level COUNTING visualisation: every tracked deer in the frame, boxed and labelled
with its TRACK ID.

Why this exists: export_evidence.py documents ONE deer per sheet, so a frame containing
three deer shows only the one being documented — which reads as "the model missed the
others". This renders the opposite view: pick the frames where the most deer are tracked
simultaneously and draw ALL of them, each in its own colour with `ID <track> <conf>` at
the top-left corner of its box.

That makes the two counting failure modes visible at a glance:
  * two boxes with DIFFERENT ids on the same animal  -> fragmentation (over-count)
  * a visible deer with no box                        -> miss (under-count)

Colour is derived from the track id, so the same animal keeps its colour across frames.

Usage:
  python src/viz/count_frames.py \
      --counts-dir <run>/merged --source data/raw \
      --out results/viz/count_frames [--per-video 3] [--only-confirmed]
"""
from __future__ import annotations
import argparse
import colorsys
import csv
import math
import os
from collections import defaultdict

import cv2
import numpy as np


def find_video(source: str, stem: str) -> str | None:
    import glob
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


def colour_for(tid: int):
    """Stable, well-separated colour per track id (golden-ratio hue hop)."""
    h = (int(tid) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_frame(img, dets, caption):
    """Two different confidences are drawn, because they answer different questions:
         det  = the DETECTOR's per-frame score: 'is there a deer in this box?'
         ID   = the CALIBRATED track score: 'is this ID a distinct deer worth counting?'
    Detector confidence cannot express ID correctness — a duplicate fragment sitting on
    an already-counted deer still scores ~0.42 because a deer really is there. Solid
    thick box = counted; thin box = tracked but rejected."""
    vis = img.copy()
    for d in dets:
        c = colour_for(d["track_id"])
        x1 = int(d["xc"] - d["w"] / 2); y1 = int(d["yc"] - d["h"] / 2)
        x2 = int(d["xc"] + d["w"] / 2); y2 = int(d["yc"] + d["h"] / 2)
        thick = 2 if d.get("counted", d["confirmed"]) else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, thick)
        # Only the ID score is shown. Detector confidence answers "is there a deer?"
        # and belongs to the DETECTION figure; here the question is "is this ID a
        # distinct deer worth counting?", which is the calibrated track score.
        idsc = d.get("id_score")
        label = (f"ID {d['track_id']}  {idsc:.2f}" if idsc is not None
                 else f"ID {d['track_id']}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 - 3 if y1 - th - 6 > 0 else y2 + th + 5      # flip below if near top
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + 3), (0, 0, 0), -1)
        cv2.putText(vis, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1,
                    cv2.LINE_AA)
    bar = np.zeros((26, vis.shape[1], 3), np.uint8)
    cv2.putText(bar, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 230, 240),
                1, cv2.LINE_AA)
    return np.vstack([vis, bar])


def sheet(tiles, cols=2):
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * h:r * h + t.shape[0], c * w:c * w + t.shape[1]] = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True, help="dir with tracks.csv (merged)")
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--out", default="results/viz/count_frames")
    ap.add_argument("--per-video", type=int, default=3,
                    help="how many multi-deer frames to render per video")
    ap.add_argument("--showcase-per-video", type=int, default=2,
                    help="additionally render the CLEAREST frames per video: largest "
                         "deer with the highest ID score. Single-deer frames allowed — "
                         "these are the figure-quality examples.")
    ap.add_argument("--showcase-min-score", type=float, default=0.80,
                    help="minimum calibrated ID score for a showcase frame")
    ap.add_argument("--only-confirmed", action="store_true",
                    help="draw only tracks the counter accepted (default: all tracks, "
                         "so misses and rejected candidates are visible too)")
    ap.add_argument("--track-scores", default="",
                    help="per_track_confidence.csv from calibrated_confirmer.py "
                         "(--target primary): the CALIBRATED probability that this track "
                         "is a distinct countable deer, i.e. an ID-correctness score")
    ap.add_argument("--contrast", default="clahe")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
    from thermal import enhance_contrast

    scores: dict[tuple, dict] = {}
    if args.track_scores and os.path.isfile(args.track_scores):
        with open(args.track_scores) as f:
            for r in csv.DictReader(f):
                scores[(r["video"], int(r["track_id"]))] = {
                    "id_score": float(r["confidence"]), "counted": r["counted"] == "1"}
        print(f"loaded {len(scores)} calibrated track scores")

    # frame -> detections, per video
    per_video: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    confirmed_per_video: dict[str, set] = defaultdict(set)
    with open(os.path.join(args.counts_dir, "tracks.csv")) as f:
        for r in csv.DictReader(f):
            conf_flag = r.get("confirmed", "1") == "1"
            if args.only_confirmed and not conf_flag:
                continue
            per_video[r["video"]][int(r["frame"])].append({
                "track_id": int(r["track_id"]), "xc": float(r["xc"]),
                "yc": float(r["yc"]), "w": float(r["w"]), "h": float(r["h"]),
                "conf": float(r["conf"]), "confirmed": conf_flag,
                **scores.get((r["video"], int(r["track_id"])), {})})
            if conf_flag:
                confirmed_per_video[r["video"]].add(int(r["track_id"]))

    def showcase_picks(frames):
        """Clearest frames: biggest box x highest ID score, among COUNTED tracks.
        Single-deer frames are fine here — the point is legibility, not density."""
        best_per_track: dict[int, tuple] = {}
        for fi, dets in frames.items():
            for d in dets:
                sc = d.get("id_score")
                if sc is None or sc < args.showcase_min_score:
                    continue
                if not d.get("counted", d["confirmed"]):
                    continue
                quality = math.sqrt(max(d["w"] * d["h"], 1.0)) * sc
                cur = best_per_track.get(d["track_id"])
                if cur is None or quality > cur[0]:
                    best_per_track[d["track_id"]] = (quality, fi, dets)
        ranked = sorted(best_per_track.values(), key=lambda t: -t[0])
        out, used = [], set()
        for _q, fi, dets in ranked:
            if fi in used:
                continue
            used.add(fi); out.append((fi, dets))
            if len(out) >= args.showcase_per_video:
                break
        return out

    index, all_tiles, show_tiles = [], [], []
    for video, frames in sorted(per_video.items()):
        # frames with the MOST simultaneous distinct tracks
        ranked = sorted(frames.items(),
                        key=lambda kv: (-len({d["track_id"] for d in kv[1]}), kv[0]))
        picks, seen_sets = [], []
        for fi, dets in ranked:
            ids = frozenset(d["track_id"] for d in dets)
            if len(ids) < 2:
                break
            if any(ids == s for s in seen_sets):     # skip near-duplicate frames
                continue
            seen_sets.append(ids); picks.append((fi, dets))
            if len(picks) >= args.per_video:
                break
        shows = showcase_picks(frames)
        show_frames = {fi for fi, _ in shows}
        for fi, dets in shows:                       # add, avoiding duplicates
            if fi not in {f for f, _ in picks}:
                picks.append((fi, dets))
        if not picks:
            continue
        vpath = find_video(args.source, video)
        if vpath is None:
            print(f"[warn] no source video for {video}"); continue
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        want = {fi: dets for fi, dets in picks}
        vdir = os.path.join(args.out, video)
        os.makedirs(vdir, exist_ok=True)
        fi = -1
        while want:
            ok, frame = cap.read()
            if not ok:
                break
            fi += 1
            dets = want.pop(fi, None)
            if dets is None:
                continue
            frame = enhance_contrast(frame, method=args.contrast)
            n_ids = len({d["track_id"] for d in dets})
            n_conf = len({d["track_id"] for d in dets if d["confirmed"]})
            cap_txt = (f"{video[:40]}  f{fi}  t={fi/fps:.1f}s  |  {n_ids} tracked "
                       f"({n_conf} counted)  |  video total {len(confirmed_per_video[video])}")
            vis = draw_frame(frame, dets, cap_txt)
            tag = "showcase_" if fi in show_frames else ""
            out_p = os.path.join(vdir, f"{tag}f{fi}_{n_ids}deer.jpg")
            cv2.imwrite(out_p, vis)
            (show_tiles if fi in show_frames else all_tiles).append(vis)
            index.append((video, fi, round(fi / fps, 1), n_ids, n_conf, out_p))
        cap.release()
        print(f"{video[:44]:<44} {len(picks)} frames, "
              f"{len(confirmed_per_video[video])} deer counted", flush=True)

    if all_tiles:
        sh = sheet(all_tiles[:8])
        if sh is not None:
            cv2.imwrite(os.path.join(args.out, "_sheet_multideer.jpg"), sh)
    if show_tiles:
        sh = sheet(show_tiles[:8])
        if sh is not None:
            cv2.imwrite(os.path.join(args.out, "_sheet_showcase.jpg"), sh)
        print(f"showcase frames: {len(show_tiles)}")
    with open(os.path.join(args.out, "index.csv"), "w") as f:
        f.write("video,frame,t_s,n_tracked,n_counted,path\n")
        for row in index:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\n-> {args.out}/   ({len(index)} frames)")
    print("   every tracked deer is boxed; label = ID <track> <detector conf>;")
    print("   colour is stable per track id, so the same animal keeps its colour.")


if __name__ == "__main__":
    main()
