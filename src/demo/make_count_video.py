#!/usr/bin/env python3
"""
Render one transect with the pipeline's OWN output drawn on it: the boxes it produced, the
identities it assigned, and the count it arrived at.

This differs from make_smooth_demo.py, which draws human-verified annotations and therefore
shows what a perfect system would do. Here the counter increments when TRACT confirms a
track, so the number on screen is the number the paper reports -- including its mistakes.
Over-counts appear as two identities on one animal; misses appear as an animal nobody boxes.

Usage:
  python src/demo/make_count_video.py --video GolfDr_SHB_12.11.2025 \
      --out /work/hdd/bgte/tislam6/wildlife_outputs/demo/golfdr_count.mp4
"""
from __future__ import annotations
import argparse, colorsys, csv, os, sys
from collections import defaultdict
import cv2, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from thermal import enhance_contrast                                   # noqa: E402

COUNTS = "/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10/merged/tracks.csv"
SCORES = "results/temporal/calibrated_orphan/per_track_confidence.csv"


def colour(tid: int):
    r, g, b = colorsys.hsv_to_rgb((int(tid) * 0.61803398875) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def find_video(stem: str, root="data/raw"):
    import glob
    for ext in ("mp4", "MP4", "avi", "mov"):
        hit = glob.glob(os.path.join(root, "**", f"{stem}.{ext}"), recursive=True)
        if hit:
            return sorted(hit)[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--stride", type=int, default=1, help=">1 speeds playback and rendering")
    ap.add_argument("--hold", type=int, default=8, help="frames to keep a box after its last hit")
    args = ap.parse_args()

    per_frame: dict[int, list] = defaultdict(list)
    confirmed, first_seen = set(), {}
    with open(COUNTS) as f:
        for r in csv.DictReader(f):
            if r["video"] != args.video:
                continue
            fi, tid = int(r["frame"]), int(r["track_id"])
            ok = r.get("confirmed", "1") == "1"
            per_frame[fi].append((tid, float(r["xc"]), float(r["yc"]),
                                  float(r["w"]), float(r["h"]), ok))
            if ok:
                confirmed.add(tid)
                first_seen[tid] = min(first_seen.get(tid, fi), fi)
    if not per_frame:
        raise SystemExit(f"no tracks for {args.video}")

    score = {}
    if os.path.isfile(SCORES):
        for r in csv.DictReader(open(SCORES)):
            if r["video"] == args.video:
                score[int(r["track_id"])] = float(r["confidence"])

    src = find_video(args.video)
    if src is None:
        raise SystemExit(f"source video for {args.video} not found")
    cap = cv2.VideoCapture(src)
    W, H = int(cap.get(3)), int(cap.get(4))
    BAR = 46
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps / args.stride, (W, H + BAR))

    # a box lingers briefly after its last detection so it does not strobe on 60 fps footage
    live: dict[int, tuple] = {}
    fi, written, counted = -1, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        for tid, xc, yc, w, h, cf in per_frame.get(fi, []):
            live[tid] = (xc, yc, w, h, cf, fi)
        live = {t: v for t, v in live.items() if fi - v[5] <= args.hold}
        counted = sum(1 for t, f0 in first_seen.items() if f0 <= fi)
        if fi % args.stride:
            continue

        vis = enhance_contrast(frame, method="clahe")
        for tid, (xc, yc, w, h, cf, _) in live.items():
            c = colour(tid) if cf else (150, 150, 150)
            x1, y1 = int(xc - w / 2), int(yc - h / 2)
            x2, y2 = int(xc + w / 2), int(yc + h / 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2 if cf else 1)
            lab = f"ID {tid}" + (f"  {score[tid]:.2f}" if tid in score else "")
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 6
            cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + 3), (0, 0, 0), -1)
            cv2.putText(vis, lab, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)

        bar = np.zeros((BAR, W, 3), np.uint8)
        cv2.putText(bar, f"DEER COUNTED: {counted}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 240, 120), 2, cv2.LINE_AA)
        cv2.putText(bar, f"{args.video[:26]}   t={fi/args.fps:5.1f}s",
                    (W - 305, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        vw.write(np.vstack([vis, bar]))
        written += 1
        if written % 1500 == 0:
            print(f"  {written} frames written (count {counted})", flush=True)

    cap.release(); vw.release()
    print(f"-> {args.out}\n   {written} frames, final count {counted}, "
          f"{len(confirmed)} confirmed tracks")


if __name__ == "__main__":
    main()
