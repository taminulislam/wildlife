#!/usr/bin/env python3
"""
Render team-facing EVIDENCE for every counted deer: the exact frames the model
counted, with the detection box drawn and labelled (video, track, time, confidence).

Reads the artifacts produced by count_deer.py (a count run's output dir):
  counts.csv   per-track summary incl. `confirmed`
  tracks.csv   per-frame boxes for each track (xc,yc,w,h,conf,frame,t_s)
and seeks the original videos (CPU only, no GPU) to draw the evidence.

For each confirmed deer (track) it writes:
  <out>/<SITE>/<video>/trk<ID>_sheet.jpg   contact sheet: top-K frames, zoomed on
                                           the deer, each captioned frame/time/conf
  <out>/<SITE>/<video>/trk<ID>_best.jpg    the single best frame, FULL frame with
                                           box drawn (shows the deer in context)
And site-level / global rollups:
  <out>/<SITE>/_gallery.jpg                one best crop per confirmed deer in the site
  <out>/evidence_index.csv                 every image -> video,track,frame,t_s,conf

Usage:
  python src/track/export_evidence.py \
      --counts-dir /work/hdd/.../counts/full_m640_MAS \
      --source data/raw \
      --out /work/hdd/.../evidence/full_m640_MAS [--topk 6]
"""
from __future__ import annotations
import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import cv2

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


def find_video(source: str, stem: str) -> str | None:
    """Locate the original video file for a given count-run video stem."""
    for ext in ("mp4", "MP4", "avi", "mov"):
        hits = glob.glob(os.path.join(source, "**", f"{stem}.{ext}"), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def box_xyxy(xc, yc, w, h, W, H):
    x1 = max(0, int(xc - w / 2)); y1 = max(0, int(yc - h / 2))
    x2 = min(W - 1, int(xc + w / 2)); y2 = min(H - 1, int(yc + h / 2))
    return x1, y1, x2, y2


def label(img, text, org, color=GREEN, scale=0.6):
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def crop_around(frame, x1, y1, x2, y2, min_side=220, pad_frac=0.6):
    """Zoomed crop centred on the box so a small thermal deer is clearly visible."""
    H, W = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = int(max(min_side, bw * (1 + 2 * pad_frac), bh * (1 + 2 * pad_frac)))
    half = side // 2
    cx1 = max(0, min(cx - half, W - side)) if side <= W else 0
    cy1 = max(0, min(cy - half, H - side)) if side <= H else 0
    cx2 = min(W, cx1 + side); cy2 = min(H, cy1 + side)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    # box coords relative to the crop
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), GREEN, 2)
    return crop


def grid(tiles, cols, cap_h=0):
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    canvas = cv2.copyMakeBorder  # noqa: F841 (kept for clarity)
    import numpy as np
    sheet = (0 * np.ones((rows * h, cols * w, 3), dtype="uint8"))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        th, tw = t.shape[:2]
        sheet[r * h:r * h + th, c * w:c * w + tw] = t
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True,
                    help="a count_deer.py output dir (has counts.csv + tracks.csv)")
    ap.add_argument("--source", default="data/raw", help="root of original videos")
    ap.add_argument("--out", required=True, help="output dir for evidence images")
    ap.add_argument("--topk", type=int, default=6, help="frames per deer on the sheet")
    ap.add_argument("--all-candidates", action="store_true",
                    help="also render unconfirmed candidate tracks (for review)")
    args = ap.parse_args()

    tracks_csv = os.path.join(args.counts_dir, "tracks.csv")
    if not os.path.exists(tracks_csv):
        raise SystemExit(f"missing {tracks_csv} (run the updated count_deer.py first)")

    obs = read_csv(tracks_csv)
    # group observations by (video, track_id)
    by_track: dict[tuple, list[dict]] = defaultdict(list)
    for o in obs:
        by_track[(o["video"], o["track_id"])].append(o)

    os.makedirs(args.out, exist_ok=True)
    index_rows = []
    site_gallery: dict[str, list] = defaultdict(list)
    vcache: dict[str, cv2.VideoCapture] = {}
    n_deer = 0

    # stable order: by site, video, then track
    keys = sorted(by_track.keys(), key=lambda k: (by_track[k][0]["site"], k[0], int(k[1])))
    for (video, tid) in keys:
        rows = by_track[(video, tid)]
        confirmed = rows[0]["confirmed"] == "1"
        if not confirmed and not args.all_candidates:
            continue
        site = rows[0]["site"] or "UNK"
        vpath = find_video(args.source, video)
        if vpath is None:
            print(f"  ! video not found for {video}; skipping trk{tid}")
            continue
        cap = vcache.get(vpath) or cv2.VideoCapture(vpath)
        vcache[vpath] = cap

        rows_sorted = sorted(rows, key=lambda r: float(r["conf"]), reverse=True)
        picks = rows_sorted[:args.topk]
        tiles, best_full = [], None
        for j, r in enumerate(picks):
            fi = int(r["frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = box_xyxy(float(r["xc"]), float(r["yc"]),
                                      float(r["w"]), float(r["h"]), W, H)
            cf, t_s = float(r["conf"]), float(r["t_s"])
            tile = crop_around(frame, x1, y1, x2, y2)
            label(tile, f"f{fi} t={t_s:.1f}s c={cf:.2f}", (4, tile.shape[0] - 6))
            tiles.append(tile)
            if j == 0:  # best detection -> full-frame context image
                full = frame.copy()
                cv2.rectangle(full, (x1, y1), (x2, y2), GREEN, 2)
                label(full, f"{video}  trk{tid}  t={t_s:.1f}s  conf={cf:.2f}",
                      (10, 28), color=YELLOW, scale=0.7)
                best_full = full

        if not tiles:
            continue
        n_deer += 1
        vdir = os.path.join(args.out, site, video)
        os.makedirs(vdir, exist_ok=True)
        sheet = grid(tiles, cols=min(3, len(tiles)))
        sheet_p = os.path.join(vdir, f"trk{tid}_sheet.jpg")
        cv2.imwrite(sheet_p, sheet)
        best_p = os.path.join(vdir, f"trk{tid}_best.jpg")
        if best_full is not None:
            cv2.imwrite(best_p, best_full)
        # caption tile for the site gallery
        g = tiles[0].copy()
        label(g, f"{video[:18]} trk{tid} c={float(picks[0]['conf']):.2f}",
              (4, 18), color=YELLOW, scale=0.45)
        site_gallery[site].append(g)
        for r in picks:
            index_rows.append({
                "site": site, "video": video, "track_id": tid,
                "frame": r["frame"], "t_s": r["t_s"], "conf": r["conf"],
                "confirmed": r["confirmed"], "sheet": sheet_p,
            })
        print(f"  [{site}] {video} trk{tid}: {len(tiles)} frames -> {sheet_p}")

    for cap in vcache.values():
        cap.release()

    # per-site gallery (slide-ready: one best crop per confirmed deer)
    for site, tiles in site_gallery.items():
        sheet = grid(tiles, cols=min(5, len(tiles)))
        if sheet is not None:
            p = os.path.join(args.out, site, "_gallery.jpg")
            cv2.imwrite(p, sheet)
            print(f"  gallery [{site}]: {len(tiles)} deer -> {p}")

    idx_p = os.path.join(args.out, "evidence_index.csv")
    with open(idx_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()) if index_rows
                           else ["site", "video", "track_id", "frame", "t_s",
                                 "conf", "confirmed", "sheet"])
        w.writeheader()
        w.writerows(index_rows)
    print(f"\n{n_deer} deer rendered. index -> {idx_p}")


if __name__ == "__main__":
    main()
