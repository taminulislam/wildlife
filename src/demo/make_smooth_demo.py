"""
Render a smooth, continuous detection demo from one real source video.

Unlike the montage (deer frames only), this plays the actual FLIR footage straight
through — mostly empty forest — and when a confirmed deer appears, its box pops up for a
short window around the moment it was verified, and a running deer counter increments.
This is the "video in -> boxes + live count out" demo, with natural empty stretches so it
looks like real playback.

Boxes come from the human-verified annotations (peak frame of each confirmed deer event),
shown for +/- a short hold around that moment (so they appear while the deer is most
visible and don't drift as the vehicle moves).

Usage:
    python src/demo/make_smooth_demo.py --key SHB__RedFoxLn_v1_LS
    python src/demo/make_smooth_demo.py --key SHB__RedFoxLn_v1_LS --fps 30 --max-seconds 120
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mining"))
from filename_meta import iter_videos  # noqa: E402

ANNO = "data/annotate"
EVENTS = "data/events"


def _source_for_key(raw_dir: str, key: str) -> str | None:
    for path, meta in iter_videos(raw_dir):
        if meta.key == key:
            return path
    return None


def _confirmed_deer_events(key: str):
    """Return list of (peak_frame, [boxes]) for events confirmed as deer."""
    ev_path = os.path.join(EVENTS, key, "events.csv")
    if not os.path.isfile(ev_path):
        return []
    events = {int(r["peak_frame"]): r for r in csv.DictReader(open(ev_path))}
    out = []
    for peak in events:
        stem = f"{key}_f{peak}"
        lp = os.path.join(ANNO, "labels", stem + ".txt")
        if not os.path.isfile(lp):
            continue
        boxes = []
        for line in open(lp):
            p = line.split()
            if len(p) == 5:
                boxes.append([float(x) for x in p[1:]])  # xc,yc,w,h
        if boxes:
            out.append((peak, boxes))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True, help="Video key, e.g. SHB__RedFoxLn_v1_LS")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=25.0, help="Output fps")
    ap.add_argument("--hold", type=float, default=1.2,
                    help="Seconds to show each deer box around its verified moment")
    ap.add_argument("--full", action="store_true",
                    help="Render the entire video (large). Default is a condensed reel.")
    ap.add_argument("--lead", type=float, default=4.0,
                    help="Condensed mode: empty seconds shown before each deer arrival")
    ap.add_argument("--trail", type=float, default=3.0,
                    help="Condensed mode: seconds shown after each deer moment")
    a = ap.parse_args()

    src = _source_for_key(a.raw, a.key)
    if not src:
        raise SystemExit(f"No source video for key {a.key} under {a.raw}")
    deer_events = _confirmed_deer_events(a.key)
    if not deer_events:
        raise SystemExit(f"No confirmed deer events for {a.key}")

    cap = cv2.VideoCapture(src)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build per-source-frame box list and a "count increments here" marker.
    hold_f = int(a.hold * src_fps)
    active: dict[int, list] = {}
    increment_at: dict[int, int] = {}
    for peak, boxes in sorted(deer_events):
        start = max(0, peak - hold_f)
        end = peak + hold_f
        increment_at[start] = increment_at.get(start, 0) + len(boxes)
        for fr in range(start, end + 1):
            active.setdefault(fr, []).extend(boxes)

    # Condensed mode (default): only render a window around each deer event, so the reel
    # stays short. Frames outside every window are skipped (a clock overlay makes the
    # time jumps explicit). --full renders everything.
    render_frame = None  # None => render all
    if not a.full:
        lead_f, trail_f = int(a.lead * src_fps), int(a.trail * src_fps)
        keep = set()
        for peak, _ in deer_events:
            for fr in range(max(0, peak - lead_f), peak + trail_f + 1):
                keep.add(fr)
        render_frame = keep

    out_path = a.out or os.path.join("results", f"smooth_demo_{a.key}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stride = max(1, round(src_fps / a.fps))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))

    count = 0
    idx = 0
    flash = 0  # frames remaining to show "DEER DETECTED" banner
    while idx < total:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in increment_at:
            count += increment_at[idx]
            flash = int(0.8 * src_fps)
        do_render = (render_frame is None or idx in render_frame) and idx % stride == 0
        if do_render:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            boxes = active.get(idx, [])
            for xc, yc, bw, bh in boxes:
                x1 = int((xc - bw / 2) * w); y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w); y2 = int((yc + bh / 2) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 90), 2)
                cv2.putText(frame, "deer", (x1, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 220, 90), 1, cv2.LINE_AA)
            # top banner: site/transect + source clock + running count + status
            t = idx / src_fps
            clock = f"{int(t // 60):02d}:{int(t % 60):02d}"
            cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
            cv2.putText(frame, f"{a.key}  t={clock}", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Deer counted: {count}", (w - 180, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 90), 1, cv2.LINE_AA)
            if flash > 0:
                cv2.putText(frame, "DEER DETECTED", (w // 2 - 80, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 90), 2, cv2.LINE_AA)
            vw.write(frame)
        flash = max(0, flash - 1)
        idx += 1

    vw.release()
    cap.release()
    dur = total / src_fps
    print(f"Wrote {out_path}  ({dur:.0f}s of footage, {count} deer counted)")


if __name__ == "__main__":
    main()
