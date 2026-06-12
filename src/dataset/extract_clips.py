"""
Cut mined events into short clips for annotation in CVAT (video + track interpolation).

For each event in a video's ``events.csv`` (produced by the miner), this writes a short
clip covering the event plus padding, downsampled to a lower fps so CVAT interpolation
keyframes are meaningful and the clips are small. A ``clips_manifest.csv`` maps every
clip frame back to its source video and source frame index, so annotations always trace
back to the original footage.

Why downsample: at 60 fps, consecutive frames are near-duplicates — wasteful to annotate
and to interpolate across. ~10 fps keeps motion visible while cutting frame count 6x.

Usage:
    # one video's events
    python src/dataset/extract_clips.py --video data/raw/Foo.mp4
    # every video that has an events.csv
    python src/dataset/extract_clips.py --all
    python src/dataset/extract_clips.py --all --pad 1.5 --out-fps 10 --min-duration 0.5

Outputs under data/clips/<video_stem>/:
    event_000.mp4, event_001.mp4, ...
    (and appends rows to data/clips/clips_manifest.csv)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mining"))
from filename_meta import iter_videos, parse_path  # noqa: E402


def _events_csv_for(events_root: str, key: str) -> str:
    return os.path.join(events_root, key, "events.csv")


def read_events(events_csv: str) -> list[dict]:
    if not os.path.isfile(events_csv):
        return []
    with open(events_csv, newline="") as f:
        return list(csv.DictReader(f))


def extract_for_video(
    video_path: str,
    events_root: str,
    clips_root: str,
    *,
    pad: float = 1.0,
    out_fps: float = 10.0,
    min_duration: float = 0.0,
    manifest_writer: "csv.writer | None" = None,
) -> int:
    key = parse_path(video_path).key
    events = read_events(_events_csv_for(events_root, key))
    if not events:
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ! could not open {video_path}")
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = os.path.join(clips_root, key)
    os.makedirs(out_dir, exist_ok=True)
    # Sampling stride to hit roughly out_fps from src_fps.
    stride = max(1, round(src_fps / out_fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    n_written = 0
    for e in events:
        start_f = int(e["start_frame"])
        end_f = int(e["end_frame"])
        pad_f = int(pad * src_fps)
        a = max(0, start_f - pad_f)
        b = min(total - 1, end_f + pad_f)
        if (b - a) / src_fps < min_duration:
            continue

        eid = int(e["event_id"])
        clip_path = os.path.join(out_dir, f"event_{eid:03d}.mp4")
        writer = cv2.VideoWriter(clip_path, fourcc, out_fps, (w, h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        src_idx = a
        clip_idx = 0
        while src_idx <= b:
            ok, frame = cap.read()
            if not ok:
                break
            if (src_idx - a) % stride == 0:
                writer.write(frame)
                if manifest_writer is not None:
                    manifest_writer.writerow([
                        key, eid, os.path.relpath(clip_path).replace("\\", "/"),
                        clip_idx, src_idx, round(src_idx / src_fps, 3),
                    ])
                clip_idx += 1
            src_idx += 1
        writer.release()
        n_written += 1

    cap.release()
    print(f"  {key}: wrote {n_written} clips -> {out_dir}")
    return n_written


def discover_mined_videos(raw_dir: str, events_root: str) -> list[str]:
    """Find raw videos that have a matching events.csv (by VideoMeta.key)."""
    out: list[str] = []
    for v, meta in iter_videos(raw_dir):
        if os.path.isfile(_events_csv_for(events_root, meta.key)):
            out.append(v)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="Single raw video path")
    g.add_argument("--all", action="store_true", help="All videos with an events.csv")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--events-root", default="data/events")
    p.add_argument("--clips-root", default="data/clips")
    p.add_argument("--pad", type=float, default=1.0, help="Seconds of padding each side")
    p.add_argument("--out-fps", type=float, default=10.0, help="Clip output fps")
    p.add_argument("--min-duration", type=float, default=0.0,
                   help="Skip events whose padded clip is shorter than this (s)")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    os.makedirs(a.clips_root, exist_ok=True)
    manifest_path = os.path.join(a.clips_root, "clips_manifest.csv")
    new_manifest = not os.path.isfile(manifest_path)
    with open(manifest_path, "a", newline="") as mf:
        mw = csv.writer(mf)
        if new_manifest:
            mw.writerow(["video_stem", "event_id", "clip_path", "clip_frame",
                         "src_frame", "src_time_s"])
        videos = ([a.video] if a.video
                  else discover_mined_videos(a.raw, a.events_root))
        if not videos:
            print("No videos with events.csv found. Run the miner first.")
            return
        total = 0
        for v in videos:
            total += extract_for_video(
                v, a.events_root, a.clips_root,
                pad=a.pad, out_fps=a.out_fps, min_duration=a.min_duration,
                manifest_writer=mw,
            )
    print(f"Total clips written: {total}. Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
