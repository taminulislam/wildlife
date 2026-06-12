"""
Mine candidate "warm-body" events from FLIR thermal videos.

Deer in this footage appear as locally bright (warm) blobs against a cooler, textured
forest background. This script scans a video at a reduced frame rate, finds bright blobs
that stand out from their local surroundings, groups temporally adjacent hits into
*events*, and writes an event list plus optional thumbnail frames.

The goal is NOT accurate detection — it is cheap recall-oriented triage so a human only
has to skim a few short clips per video instead of 6 minutes of mostly-empty forest, and
so we can sample annotation frames from where the action is.

Usage:
    python src/mining/mine_events.py --video data/raw/SomeVideo.mp4
    python src/mining/mine_events.py --video data/raw/SomeVideo.mp4 --save-thumbs --debug

Outputs (under data/events/<video_stem>/):
    events.csv     one row per event: start/end frame & time, peak score, n_hits, bbox
    hits.csv       one row per sampled frame that had >=1 blob (for debugging/tuning)
    thumbs/        annotated thumbnail of each event's peak frame (if --save-thumbs)
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Blob:
    x: int
    y: int
    w: int
    h: int
    area: int
    score: float  # local-contrast brightness score


@dataclass
class Event:
    start_frame: int
    end_frame: int
    fps: float
    peak_frame: int = 0
    peak_score: float = 0.0
    n_hits: int = 0
    # union bbox over the event, in pixels
    x0: int = 10**9
    y0: int = 10**9
    x1: int = 0
    y1: int = 0
    peak_blobs: list = field(default_factory=list)

    def add_hit(self, frame_idx: int, blobs: list[Blob]) -> None:
        self.end_frame = frame_idx
        self.n_hits += 1
        frame_score = max((b.score for b in blobs), default=0.0)
        if frame_score >= self.peak_score:
            self.peak_score = frame_score
            self.peak_frame = frame_idx
            self.peak_blobs = blobs
        for b in blobs:
            self.x0 = min(self.x0, b.x)
            self.y0 = min(self.y0, b.y)
            self.x1 = max(self.x1, b.x + b.w)
            self.y1 = max(self.y1, b.y + b.h)

    @property
    def start_time(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_time(self) -> float:
        return self.end_frame / self.fps


def find_blobs(
    gray: np.ndarray,
    *,
    tophat_kernel: int = 15,
    min_contrast: int = 18,
    min_area: int = 6,
    max_area_frac: float = 0.08,
    max_aspect: float = 6.0,
    min_extent: float = 0.20,
) -> list[Blob]:
    """Find locally bright blobs via white top-hat + adaptive thresholding.

    White top-hat keeps bright structures smaller than the kernel and removes the slowly
    varying background (sky glow, large warm ground patches). This is what isolates a
    compact warm animal from broad warm regions.
    """
    # Light denoise so sensor noise doesn't fragment blobs.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_kernel, tophat_kernel))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, k)

    # Threshold relative to the top-hat response. min_contrast is in 8-bit intensity units
    # above the local background.
    _, mask = cv2.threshold(tophat, min_contrast, 255, cv2.THRESH_BINARY)

    # Close small gaps so a deer split by a twig stays one blob.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_area = max_area_frac * gray.shape[0] * gray.shape[1]
    blobs: list[Blob] = []
    for i in range(1, n):  # skip background label 0
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        # Shape filters: drop motion-streak artifacts (very elongated) and wispy blobs
        # (low fill ratio). A deer's thermal signature is compact and reasonably solid.
        aspect = max(w, h) / max(min(w, h), 1)
        extent = area / max(w * h, 1)
        if aspect > max_aspect or extent < min_extent:
            continue
        # Score = mean top-hat response inside the blob (how much it pops locally).
        comp = tophat[y : y + h, x : x + w][labels[y : y + h, x : x + w] == i]
        score = float(comp.mean()) if comp.size else 0.0
        blobs.append(Blob(x, y, w, h, area, score))
    return blobs


def mine(
    video_path: str,
    out_dir: str,
    *,
    sample_every: int = 6,
    gap_seconds: float = 1.0,
    min_event_hits: int = 2,
    save_thumbs: bool = False,
    debug: bool = False,
    **blob_kwargs,
) -> list[Event]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(out_dir, exist_ok=True)
    thumbs_dir = os.path.join(out_dir, "thumbs")
    if save_thumbs:
        os.makedirs(thumbs_dir, exist_ok=True)

    gap_frames = gap_seconds * fps  # tolerated gap (in source frames) within one event
    events: list[Event] = []
    cur: Event | None = None
    hits_rows: list[dict] = []

    idx = 0
    n_sampled = 0
    while True:
        # Seek by sampling stride; reading sequentially and skipping is reliable on MP4.
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every != 0:
            idx += 1
            continue
        n_sampled += 1
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blobs = find_blobs(gray, **blob_kwargs)

        if blobs:
            peak = max(b.score for b in blobs)
            hits_rows.append(
                {"frame": idx, "time_s": round(idx / fps, 2), "n_blobs": len(blobs),
                 "peak_score": round(peak, 1)}
            )
            if cur is None:
                cur = Event(start_frame=idx, end_frame=idx, fps=fps)
            elif idx - cur.end_frame > gap_frames:
                events.append(cur)
                cur = Event(start_frame=idx, end_frame=idx, fps=fps)
            cur.add_hit(idx, blobs)

        if debug and n_sampled % 200 == 0:
            print(f"  ...{idx}/{total} frames, {len(events)} events so far", flush=True)
        idx += 1

    if cur is not None:
        events.append(cur)

    # Drop flicker: events with too few hits are likely noise, not a warm body.
    events = [e for e in events if e.n_hits >= min_event_hits]

    _write_events(events, out_dir, video_path)
    _write_hits(hits_rows, out_dir)
    if save_thumbs:
        _save_thumbs(events, video_path, thumbs_dir)

    print(
        f"{os.path.basename(video_path)}: sampled {n_sampled} frames, "
        f"found {len(events)} events -> {out_dir}"
    )
    cap.release()
    return events


def _write_events(events: list[Event], out_dir: str, video_path: str) -> None:
    path = os.path.join(out_dir, "events.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "event_id", "start_frame", "end_frame", "start_s", "end_s",
                    "duration_s", "n_hits", "peak_frame", "peak_score",
                    "x0", "y0", "x1", "y1"])
        for i, e in enumerate(events):
            w.writerow([os.path.basename(video_path), i, e.start_frame, e.end_frame,
                        round(e.start_time, 2), round(e.end_time, 2),
                        round(e.end_time - e.start_time, 2), e.n_hits, e.peak_frame,
                        round(e.peak_score, 1), e.x0, e.y0, e.x1, e.y1])


def _write_hits(hits_rows: list[dict], out_dir: str) -> None:
    path = os.path.join(out_dir, "hits.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_s", "n_blobs", "peak_score"])
        w.writeheader()
        w.writerows(hits_rows)


def _save_thumbs(events: list[Event], video_path: str, thumbs_dir: str) -> None:
    cap = cv2.VideoCapture(video_path)
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e.peak_frame)
        ok, frame = cap.read()
        if not ok:
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for b in e.peak_blobs:
            cv2.rectangle(frame, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 0, 255), 1)
        label = f"e{i} t={e.start_time:.1f}-{e.end_time:.1f}s score={e.peak_score:.0f}"
        cv2.putText(frame, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(thumbs_dir, f"event_{i:03d}.png"), frame)
    cap.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Path to a FLIR .mp4")
    p.add_argument("--out", default=None,
                   help="Output dir (default: data/events/<video_stem>)")
    p.add_argument("--sample-every", type=int, default=6,
                   help="Process every Nth frame (default 6 ~= 10 fps at 60 fps source)")
    p.add_argument("--gap-seconds", type=float, default=1.0,
                   help="Max gap within one event before it splits (default 1.0 s)")
    p.add_argument("--min-event-hits", type=int, default=2,
                   help="Drop events with fewer than this many blob frames (default 2)")
    p.add_argument("--tophat-kernel", type=int, default=15)
    p.add_argument("--min-contrast", type=int, default=18,
                   help="Min local brightness above background, 8-bit units (default 18)")
    p.add_argument("--min-area", type=int, default=6, help="Min blob area in px")
    p.add_argument("--max-aspect", type=float, default=6.0,
                   help="Drop blobs more elongated than this (motion streaks)")
    p.add_argument("--min-extent", type=float, default=0.20,
                   help="Drop blobs filling less than this fraction of their bbox")
    p.add_argument("--save-thumbs", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    from filename_meta import parse_path
    out_dir = a.out or os.path.join("data", "events", parse_path(a.video).key)
    mine(
        a.video, out_dir,
        sample_every=a.sample_every, gap_seconds=a.gap_seconds,
        min_event_hits=a.min_event_hits, save_thumbs=a.save_thumbs, debug=a.debug,
        tophat_kernel=a.tophat_kernel, min_contrast=a.min_contrast, min_area=a.min_area,
        max_aspect=a.max_aspect, min_extent=a.min_extent,
    )


if __name__ == "__main__":
    main()
