"""
Build a presentable results package from the confirmed deer annotations.

This is a PROTOTYPE deliverable to show the team the approach and preliminary findings —
not the final trained model. It demonstrates, end to end, what the pipeline produces:
FLIR video -> warm-body detection -> human-verified deer -> per-site counts + visuals.

Outputs under results/:
  * per_site_counts.csv     preliminary deer counts per site
  * deer_detections.mp4      montage of every confirmed deer frame with boxes + a
                             running count (what the detector's output looks like)
  * deer_grid.png            single contact-sheet image of all confirmed deer (for a slide)

Usage:
    python src/demo/build_results.py
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import cv2

ANNO = "data/annotate"
FRAMES_DIR = os.path.join(ANNO, "frames")
LABELS_DIR = os.path.join(ANNO, "labels")
OUT = "results"


def _read_boxes(stem: str):
    path = os.path.join(LABELS_DIR, stem + ".txt")
    boxes = []
    if os.path.isfile(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5:
                boxes.append([float(x) for x in p[1:]])  # xc,yc,w,h normalized
    return boxes


def _draw(frame, boxes, header):
    h, w = frame.shape[:2]
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * w); y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w); y2 = int((yc + bh / 2) * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 90), 2)
    # header banner
    cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(frame, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    index = list(csv.DictReader(open(os.path.join(ANNO, "frames.csv"))))

    # Collect deer frames (those with >=1 box), in a stable order grouped by site.
    deer = []
    for r in index:
        stem = os.path.splitext(r["name"])[0]
        boxes = _read_boxes(stem)
        if boxes:
            deer.append({**r, "boxes": boxes, "stem": stem})
    deer.sort(key=lambda r: (r["site"], r["key"], int(r["src_frame"])))

    # Per-site counts.
    frames_by_site = defaultdict(int)
    boxes_by_site = defaultdict(int)
    for d in deer:
        frames_by_site[d["site"]] += 1
        boxes_by_site[d["site"]] += len(d["boxes"])
    with open(os.path.join(OUT, "per_site_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "deer_sightings", "deer_individuals"])
        for s in sorted(frames_by_site):
            w.writerow([s, frames_by_site[s], boxes_by_site[s]])
        w.writerow(["TOTAL", sum(frames_by_site.values()), sum(boxes_by_site.values())])

    # Montage video: each deer frame held ~1.2s, with a running individual count.
    SCALE = 1.5
    fps = 25
    hold = int(1.2 * fps)
    W, H = int(640 * SCALE), int(512 * SCALE)
    vw = cv2.VideoWriter(os.path.join(OUT, "deer_detections.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    running = 0
    for i, d in enumerate(deer, 1):
        img = cv2.imread(os.path.join(FRAMES_DIR, d["name"]))
        if img is None:
            continue
        running += len(d["boxes"])
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        # rescale normalized boxes are resolution-independent, so draw directly
        header = (f"FLIR Deer Detection  |  Site {d['site']}  |  sighting {i}/{len(deer)}"
                  f"  |  deer counted: {running}")
        frame = _draw(img, d["boxes"], header)
        for _ in range(hold):
            vw.write(frame)
    vw.release()

    # Contact-sheet grid of all deer frames (downscaled), for a slide.
    cols = 8
    tw, th = 200, 160
    rows = (len(deer) + cols - 1) // cols
    grid = 255 * 0  # placeholder
    import numpy as np
    grid = np.full((rows * th, cols * tw, 3), 20, dtype=np.uint8)
    for i, d in enumerate(deer):
        img = cv2.imread(os.path.join(FRAMES_DIR, d["name"]))
        if img is None:
            continue
        img = _draw(img, d["boxes"], d["site"])
        thumb = cv2.resize(img, (tw, th))
        r, c = divmod(i, cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb
    cv2.imwrite(os.path.join(OUT, "deer_grid.png"), grid)

    print(f"Wrote results/ : per_site_counts.csv, deer_detections.mp4, deer_grid.png")
    print(f"Deer sightings: {len(deer)}, individuals: {sum(boxes_by_site.values())}")


if __name__ == "__main__":
    main()
