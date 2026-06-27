#!/usr/bin/env python3
"""
Grow the deer training set by harvesting MULTIPLE frames per confirmed deer event.

Context
-------
Verification produced 72 deer frames / 110 boxes, but each is the single PEAK frame of a
distinct warm-blob event. Every deer is actually visible for many frames (the event spans
hundreds-to-thousands of 60 fps frames). This script extracts a temporally-spread set of
frames around each labeled peak and PROPAGATES the verified box(es) across them with an
OpenCV tracker, writing them as *pre-labels* into a SEPARATE harvest dir.

These pre-labels are NOT ground truth. They must be verified/corrected in
`src/annotate/server.py` before being merged into the precious verified set. This script
never writes into data/annotate/labels/.

Outputs (default under data/annotate/harvest/):
  frames/<key>_f<frame>.png       extracted grayscale frames
  labels/<key>_f<frame>.txt       YOLO pre-labels (class 0), '' if a box was lost
  harvest_manifest.csv            provenance: source event, peak, propagation status

Typical use
-----------
  python src/dataset/harvest_event_frames.py \
      --raw data/raw --window-s 1.5 --stride 9 --max-per-event 24 --group-first
"""
from __future__ import annotations
import argparse, csv, glob, os, sys
from collections import OrderedDict

import numpy as np
import cv2


# ----------------------------- box / IO helpers -----------------------------
def yolo_to_xyxy(line, W, H):
    c, cx, cy, bw, bh = (float(x) for x in line.split())
    x0 = (cx - bw / 2) * W; y0 = (cy - bh / 2) * H
    x1 = (cx + bw / 2) * W; y1 = (cy + bh / 2) * H
    return [x0, y0, x1, y1]


def xyxy_to_yolo(b, W, H):
    x0, y0, x1, y1 = b
    cx = (x0 + x1) / 2 / W; cy = (y0 + y1) / 2 / H
    bw = (x1 - x0) / W;     bh = (y1 - y0) / H
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def match_box(prev_gray, cur_gray, box, search=40, min_corr=0.40):
    """Propagate one box to the next frame by NCC template matching in a local window.

    Appearance trackers (CSRT/KCF) fail on tiny high-contrast thermal blobs; template
    matching the bright patch within a search window is robust and follows the panning
    motion. Returns the new [x0,y0,x1,y1] or None if the match is too weak.
    """
    H, W = cur_gray.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    tmpl = prev_gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if tmpl.size == 0 or tmpl.shape[0] < 2 or tmpl.shape[1] < 2:
        return None
    sx0 = max(0, x0 - search); sy0 = max(0, y0 - search)
    sx1 = min(W, x1 + search); sy1 = min(H, y1 + search)
    region = cur_gray[sy0:sy1, sx0:sx1]
    if region.shape[0] < tmpl.shape[0] or region.shape[1] < tmpl.shape[1]:
        return None
    res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    if maxv < min_corr:
        return None
    nx0 = sx0 + maxloc[0]; ny0 = sy0 + maxloc[1]
    return [nx0, ny0, nx0 + bw, ny0 + bh]


# ----------------------------- data joins -----------------------------------
def load_deer_events(labels_dir, frames_csv, master_csv):
    """Return list of dicts: one per labeled deer frame (= one event peak)."""
    deer = {}
    for p in glob.glob(os.path.join(labels_dir, "*.txt")):
        rows = [ln.strip() for ln in open(p) if ln.strip()]
        if rows:
            deer[os.path.basename(p)[:-4]] = rows  # stem -> yolo lines
    frames = {}
    with open(frames_csv) as f:
        for r in csv.DictReader(f):
            frames[r["name"][:-4]] = r  # stem -> row
    events = {}
    with open(master_csv) as f:
        for r in csv.DictReader(f):
            events[(r["key"], r["event_id"])] = r

    out = []
    for stem, lines in deer.items():
        fr = frames.get(stem)
        if not fr:
            print(f"[warn] {stem} not in frames.csv; skipping", file=sys.stderr)
            continue
        ev = events.get((fr["key"], fr["event_id"]), {})
        out.append({
            "stem": stem, "key": fr["key"], "site": fr["site"],
            "event_id": fr["event_id"], "video": ev.get("video", ""),
            "peak": int(fr["src_frame"]),
            "start": int(ev.get("start_frame", fr["src_frame"])),
            "end": int(ev.get("end_frame", fr["src_frame"])),
            "boxes": lines, "nboxes": len(lines),
        })
    return out


def index_videos(raw_dir):
    """basename(.mp4) -> full path, for resolving the event 'video' field."""
    idx = {}
    for p in glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True):
        idx[os.path.basename(p)] = p
    return idx


# ----------------------------- core harvest ---------------------------------
def harvest_event(ev, video_path, out_frames, out_labels, window_s, stride,
                  max_per_event, fps=60.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[warn] cannot open {video_path}", file=sys.stderr)
        return []
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 512

    half = int(window_s * fps)
    lo = max(ev["start"], ev["peak"] - half)
    hi = min(ev["end"],   ev["peak"] + half)

    # Read the [lo, hi] window once, sequentially (seeking per-frame is slow/unreliable).
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    win = OrderedDict()  # frame_idx -> gray frame
    fidx = lo
    while fidx <= hi:
        ok, fr = cap.read()
        if not ok:
            break
        win[fidx] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        fidx += 1
    cap.release()
    if ev["peak"] not in win:
        print(f"[warn] peak {ev['peak']} not decoded for {ev['key']}", file=sys.stderr)
        return []

    peak_xyxy = [yolo_to_xyxy(l, W, H) for l in ev["boxes"]]

    # For each sampled frame, hold a propagated box per original deer box.
    sampled = sorted(i for i in win if (i - ev["peak"]) % stride == 0)
    # cap total samples, keeping symmetry around the peak
    if max_per_event and len(sampled) > max_per_event:
        sampled.sort(key=lambda i: abs(i - ev["peak"]))
        sampled = sorted(sampled[:max_per_event])
    boxes_per_frame = {i: [None] * len(peak_xyxy) for i in sampled}
    for j, b in enumerate(peak_xyxy):
        boxes_per_frame[ev["peak"]][j] = b

    # Propagate each box forward then backward from the peak by chaining template
    # matches frame-by-frame through the decoded window (smooth, follows the pan).
    for j, b in enumerate(peak_xyxy):
        for direction in (+1, -1):
            cur_box = b
            prev_i = ev["peak"]
            i = ev["peak"] + direction
            while (i in win) and (lo <= i <= hi):
                nb = match_box(win[prev_i], win[i], cur_box)
                if nb is None:
                    break  # lost; stop this direction
                if i in boxes_per_frame:
                    boxes_per_frame[i][j] = nb
                cur_box = nb
                prev_i = i
                i += direction

    # Write frames + pre-labels.
    rows = []
    for i in sampled:
        name = f"{ev['key']}_f{i}"
        boxes = [b for b in boxes_per_frame[i] if b is not None]
        # clamp to image
        clean = []
        for b in boxes:
            x0 = min(max(b[0], 0), W - 1); y0 = min(max(b[1], 0), H - 1)
            x1 = min(max(b[2], 1), W);     y1 = min(max(b[3], 1), H)
            if x1 - x0 >= 3 and y1 - y0 >= 3:
                clean.append([x0, y0, x1, y1])
        cv2.imwrite(os.path.join(out_frames, name + ".png"), win[i])
        with open(os.path.join(out_labels, name + ".txt"), "w") as f:
            f.write("\n".join(xyxy_to_yolo(b, W, H) for b in clean))
        rows.append({
            "name": name + ".png", "key": ev["key"], "site": ev["site"],
            "event_id": ev["event_id"], "src_frame": i, "peak_frame": ev["peak"],
            "is_peak": int(i == ev["peak"]), "n_boxes": len(clean),
            "orig_boxes": ev["nboxes"], "status": "verified" if i == ev["peak"]
            else ("propagated" if clean else "lost"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--labels", default="data/annotate/labels")
    ap.add_argument("--frames-csv", default="data/annotate/frames.csv")
    ap.add_argument("--master", default="data/events/master_events.csv")
    ap.add_argument("--out", default="data/annotate/harvest")
    ap.add_argument("--window-s", type=float, default=1.5,
                    help="seconds each side of the peak to harvest")
    ap.add_argument("--stride", type=int, default=9,
                    help="sample every Nth frame (60fps/9 ~= 6.7fps)")
    ap.add_argument("--max-per-event", type=int, default=24)
    ap.add_argument("--group-first", action="store_true",
                    help="process multi-deer (group) events first")
    ap.add_argument("--limit", type=int, default=0, help="cap #events (debug)")
    args = ap.parse_args()

    out_frames = os.path.join(args.out, "frames")
    out_labels = os.path.join(args.out, "labels")
    os.makedirs(out_frames, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    events = load_deer_events(args.labels, args.frames_csv, args.master)
    if args.group_first:
        events.sort(key=lambda e: -e["nboxes"])
    if args.limit:
        events = events[:args.limit]

    vids = index_videos(args.raw)
    manifest = []
    for n, ev in enumerate(events, 1):
        vp = vids.get(ev["video"]) or vids.get(os.path.basename(ev["video"]))
        if not vp:
            print(f"[warn] no video for {ev['key']} ({ev['video']})", file=sys.stderr)
            continue
        rows = harvest_event(ev, vp, out_frames, out_labels,
                             args.window_s, args.stride, args.max_per_event)
        manifest.extend(rows)
        kept = sum(1 for r in rows if r["n_boxes"])
        print(f"[{n}/{len(events)}] {ev['key']} ev{ev['event_id']} "
              f"({ev['nboxes']} box) -> {len(rows)} frames, {kept} with boxes")

    mpath = os.path.join(args.out, "harvest_manifest.csv")
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "key", "site", "event_id",
                          "src_frame", "peak_frame", "is_peak", "n_boxes",
                          "orig_boxes", "status"])
        w.writeheader(); w.writerows(manifest)

    tot_box = sum(r["n_boxes"] for r in manifest)
    print(f"\nHARVEST DONE: {len(manifest)} frames, {tot_box} boxes "
          f"(pre-labels) -> {args.out}")
    print(f"Manifest: {mpath}")
    print("NEXT: verify/correct in src/annotate/server.py before merging into "
          "data/annotate/labels/.")


if __name__ == "__main__":
    main()
