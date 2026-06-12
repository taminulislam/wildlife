"""
Prepare frames for manual correction in the box editor.

For each mined event it extracts the peak frame and writes:
  * data/annotate/frames/<key>_f<frame>.png   — the image to correct
  * data/annotate/labels/<key>_f<frame>.txt   — SEED boxes (YOLO format, class 0=deer)
        from the (merged) blob detector, as a starting point you then fix
  * data/annotate/frames.csv                   — index (order = score desc) for the UI

The seed boxes are just the detector's guesses — many will be wrong (hot rocks, tree
trunks, fragments). You delete the wrong ones, fix partial boxes, and draw any missed
deer. A frame you clear to zero boxes becomes a valid negative (no deer) example.

Usage:
    python src/annotate/prepare_frames.py
    python src/annotate/prepare_frames.py --max-per-video 20 --min-score 22

Re-running skips frames already extracted (so it won't clobber your corrections).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mining"))
from filename_meta import iter_videos  # noqa: E402
from mine_events import find_blobs  # noqa: E402


def _read_events(events_root: str, key: str) -> list[dict]:
    path = os.path.join(events_root, key, "events.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _seed_label(gray, *, min_contrast: int, min_area: int, merge_gap: int,
                max_seeds: int) -> list[tuple[float, float, float, float]]:
    """Run the (merged) blob detector and return YOLO-normalized seed boxes.

    Seeding is deliberately conservative (stricter than mining): a few strong candidate
    boxes are easier to correct than dozens of tiny specks. Missed deer are drawn by hand.
    """
    h, w = gray.shape[:2]
    blobs = find_blobs(gray, min_contrast=min_contrast, min_area=min_area,
                       merge_gap=merge_gap)
    blobs.sort(key=lambda b: b.area, reverse=True)
    if max_seeds > 0:
        blobs = blobs[:max_seeds]
    out = []
    for b in blobs:
        out.append(((b.x + b.w / 2) / w, (b.y + b.h / 2) / h, b.w / w, b.h / h))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--events-root", default="data/events")
    p.add_argument("--out", default="data/annotate")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--max-per-video", type=int, default=0,
                   help="Cap events per video (0 = all)")
    p.add_argument("--seed-contrast", type=int, default=26,
                   help="Min contrast for SEED boxes (stricter than mining)")
    p.add_argument("--seed-min-area", type=int, default=15,
                   help="Min blob area for seed boxes")
    p.add_argument("--seed-merge-gap", type=int, default=12)
    p.add_argument("--max-seeds", type=int, default=6,
                   help="Max seed boxes per frame (largest kept); 0 = unlimited")
    a = p.parse_args()

    frames_dir = os.path.join(a.out, "frames")
    labels_dir = os.path.join(a.out, "labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    index: list[dict] = []
    n_new = 0
    for video_path, meta in iter_videos(a.raw):
        events = _read_events(a.events_root, meta.key)
        events = [e for e in events
                  if float(e.get("peak_score", 0) or 0) >= a.min_score]
        events.sort(key=lambda e: -float(e.get("peak_score", 0) or 0))
        if a.max_per_video > 0:
            events = events[: a.max_per_video]
        if not events:
            continue

        cap = None
        for e in events:
            pf = int(e["peak_frame"])
            score = float(e.get("peak_score", 0) or 0)
            name = f"{meta.key}_f{pf}"
            img_path = os.path.join(frames_dir, name + ".png")
            lbl_path = os.path.join(labels_dir, name + ".txt")
            index.append({"name": name + ".png", "key": meta.key, "site": meta.site,
                          "event_id": e["event_id"], "src_frame": pf,
                          "score": round(score, 1)})
            if os.path.isfile(img_path):
                continue  # already extracted; don't clobber corrections
            if cap is None:
                cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pf)
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(img_path, frame)
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not os.path.isfile(lbl_path):  # seed only if no label yet
                seeds = _seed_label(
                    gray, min_contrast=a.seed_contrast, min_area=a.seed_min_area,
                    merge_gap=a.seed_merge_gap, max_seeds=a.max_seeds)
                with open(lbl_path, "w") as lf:
                    for xc, yc, bw, bh in seeds:
                        lf.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            n_new += 1
        if cap is not None:
            cap.release()

    # Sort the whole index by score desc so the UI shows likely-real first.
    index.sort(key=lambda r: -float(r["score"]))
    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "key", "site", "event_id",
                                          "src_frame", "score"])
        w.writeheader()
        w.writerows(index)

    print(f"Prepared {len(index)} frames ({n_new} newly extracted) -> {frames_dir}")
    print(f"Index -> {os.path.join(a.out, 'frames.csv')}")
    print("Next: python src/annotate/server.py")


if __name__ == "__main__":
    main()
