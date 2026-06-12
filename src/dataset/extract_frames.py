"""
Extract still frames for image-based annotation and for building the YOLO dataset.

Two kinds of frames are pulled:
  * POSITIVE candidates — sampled within mined events (where warm bodies were detected),
    at a reduced fps so frames aren't near-duplicates.
  * HARD NEGATIVES — frames from *outside* any event, to teach the model what empty
    forest / hot rocks / vehicles look like. Without these the detector learns
    "bright blob = deer" and false-positives everywhere, corrupting the count.

Every extracted frame is recorded in ``frames_manifest.csv`` with its source video,
source frame index, and whether it came from an event (``kind`` = pos/neg), so dataset
splits can stay video-aware and provenance is never lost.

Usage:
    python src/dataset/extract_frames.py --video data/raw/Foo.mp4
    python src/dataset/extract_frames.py --all --pos-fps 3 --neg-per-video 30

Outputs:
    data/frames/<video_stem>/<stem>_f<frame>.png
    data/frames/frames_manifest.csv  (appended)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2


def _events_csv_for(events_root: str, stem: str) -> str:
    return os.path.join(events_root, stem, "events.csv")


def read_event_spans(events_csv: str) -> list[tuple[int, int]]:
    if not os.path.isfile(events_csv):
        return []
    spans = []
    with open(events_csv, newline="") as f:
        for r in csv.DictReader(f):
            spans.append((int(r["start_frame"]), int(r["end_frame"])))
    return spans


def _in_any_span(idx: int, spans: list[tuple[int, int]], pad: int) -> bool:
    for a, b in spans:
        if a - pad <= idx <= b + pad:
            return True
    return False


def extract_for_video(
    video_path: str,
    events_root: str,
    frames_root: str,
    *,
    pos_fps: float = 3.0,
    neg_per_video: int = 30,
    neg_pad_s: float = 2.0,
    manifest_writer: "csv.writer | None" = None,
) -> tuple[int, int]:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    spans = read_event_spans(_events_csv_for(events_root, stem))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ! could not open {video_path}")
        return (0, 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = os.path.join(frames_root, stem)
    os.makedirs(out_dir, exist_ok=True)
    pos_stride = max(1, round(src_fps / pos_fps))
    neg_pad = int(neg_pad_s * src_fps)

    # Decide positive frame indices (sampled inside events).
    pos_idx = set()
    for a, b in spans:
        for i in range(a, b + 1, pos_stride):
            pos_idx.add(i)

    # Decide negative candidate indices: evenly spaced frames not near any event.
    neg_idx: list[int] = []
    if neg_per_video > 0:
        step = max(1, total // (neg_per_video * 3))  # oversample then filter
        for i in range(0, total, step):
            if not _in_any_span(i, spans, neg_pad):
                neg_idx.append(i)
            if len(neg_idx) >= neg_per_video:
                break

    wanted = {i: "pos" for i in sorted(pos_idx)}
    for i in neg_idx:
        wanted.setdefault(i, "neg")

    n_pos = n_neg = 0
    for idx in sorted(wanted):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        kind = wanted[idx]
        fname = f"{stem}_f{idx}.png"
        cv2.imwrite(os.path.join(out_dir, fname), frame)
        if manifest_writer is not None:
            manifest_writer.writerow([stem, fname, idx, round(idx / src_fps, 3), kind])
        if kind == "pos":
            n_pos += 1
        else:
            n_neg += 1

    cap.release()
    print(f"  {stem}: {n_pos} positive + {n_neg} negative frames -> {out_dir}")
    return (n_pos, n_neg)


def discover_mined_videos(raw_dir: str, events_root: str) -> list[str]:
    out: list[str] = []
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".ts"):
        for v in glob.glob(os.path.join(raw_dir, "**", f"*{ext}"), recursive=True):
            stem = os.path.splitext(os.path.basename(v))[0]
            if os.path.isfile(_events_csv_for(events_root, stem)):
                out.append(v)
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video")
    g.add_argument("--all", action="store_true")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--events-root", default="data/events")
    p.add_argument("--frames-root", default="data/frames")
    p.add_argument("--pos-fps", type=float, default=3.0,
                   help="Frames per second to sample inside events (default 3)")
    p.add_argument("--neg-per-video", type=int, default=30,
                   help="Hard-negative frames to pull from outside events (default 30)")
    p.add_argument("--neg-pad-s", type=float, default=2.0,
                   help="Keep negatives at least this many seconds from any event")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    os.makedirs(a.frames_root, exist_ok=True)
    manifest_path = os.path.join(a.frames_root, "frames_manifest.csv")
    new_manifest = not os.path.isfile(manifest_path)
    with open(manifest_path, "a", newline="") as mf:
        mw = csv.writer(mf)
        if new_manifest:
            mw.writerow(["video_stem", "filename", "src_frame", "src_time_s", "kind"])
        videos = ([a.video] if a.video
                  else discover_mined_videos(a.raw, a.events_root))
        if not videos:
            print("No videos with events.csv found. Run the miner first.")
            return
        tp = tn = 0
        for v in videos:
            p_, n_ = extract_for_video(
                v, a.events_root, a.frames_root,
                pos_fps=a.pos_fps, neg_per_video=a.neg_per_video,
                neg_pad_s=a.neg_pad_s, manifest_writer=mw,
            )
            tp += p_
            tn += n_
    print(f"Total: {tp} positive + {tn} negative frames. Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
