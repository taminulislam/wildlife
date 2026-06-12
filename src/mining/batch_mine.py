"""
Run the warm-blob event miner over every video in a folder and build a master index.

This is the unattended triage pass over the whole archive (~96 videos). For each video
it calls the same miner as ``mine_events.py``, then aggregates all events into one
``data/events/master_events.csv`` enriched with site/visit/side metadata, plus a
``data/events/summary_by_video.csv`` with per-video event counts and footage stats.

Usage:
    python src/mining/batch_mine.py --raw data/raw
    python src/mining/batch_mine.py --raw data/raw --save-thumbs --workers 4

Re-running skips videos already mined unless --force is given, so it is safe to run
incrementally as downloads complete.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from filename_meta import parse_filename
from mine_events import mine

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".ts")


def _mine_one(args: tuple) -> dict:
    video_path, events_root, save_thumbs, blob_kwargs = args
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(events_root, stem)
    events = mine(video_path, out_dir, save_thumbs=save_thumbs, **blob_kwargs)
    meta = parse_filename(video_path)
    return {
        "video_path": video_path,
        "out_dir": out_dir,
        "meta": meta.as_dict(),
        "n_events": len(events),
        "events": [
            {
                "event_id": i,
                "start_frame": e.start_frame,
                "end_frame": e.end_frame,
                "start_s": round(e.start_time, 2),
                "end_s": round(e.end_time, 2),
                "duration_s": round(e.end_time - e.start_time, 2),
                "n_hits": e.n_hits,
                "peak_frame": e.peak_frame,
                "peak_score": round(e.peak_score, 1),
                "x0": e.x0, "y0": e.y0, "x1": e.x1, "y1": e.y1,
            }
            for i, e in enumerate(events)
        ],
    }


def discover_videos(raw_dir: str) -> list[str]:
    vids: list[str] = []
    for ext in VIDEO_EXTS:
        vids.extend(glob.glob(os.path.join(raw_dir, "**", f"*{ext}"), recursive=True))
    return sorted(set(vids))


def already_mined(events_root: str, video_path: str) -> bool:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.isfile(os.path.join(events_root, stem, "events.csv"))


def batch(raw_dir: str, events_root: str, *, save_thumbs: bool, workers: int,
          force: bool, blob_kwargs: dict) -> None:
    os.makedirs(events_root, exist_ok=True)
    videos = discover_videos(raw_dir)
    if not videos:
        print(f"No videos found under {raw_dir} (looked for {', '.join(VIDEO_EXTS)}).")
        return

    todo = [v for v in videos if force or not already_mined(events_root, v)]
    skipped = len(videos) - len(todo)
    print(f"Found {len(videos)} videos; mining {len(todo)}"
          + (f", skipping {skipped} already done" if skipped else "") + ".")

    tasks = [(v, events_root, save_thumbs, blob_kwargs) for v in todo]
    results: list[dict] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_mine_one, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for t in tasks:
            results.append(_mine_one(t))

    # Always rebuild the master index from ALL mined videos (todo + previously done),
    # so the index is complete even on incremental runs.
    _rebuild_master_index(events_root, videos)
    print(f"Done. Master index -> {os.path.join(events_root, 'master_events.csv')}")


def _read_events_csv(out_dir: str) -> list[dict]:
    path = os.path.join(out_dir, "events.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _rebuild_master_index(events_root: str, videos: list[str]) -> None:
    master_rows: list[dict] = []
    summary_rows: list[dict] = []
    for video_path in videos:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(events_root, stem)
        rows = _read_events_csv(out_dir)
        meta = parse_filename(video_path)
        total_event_seconds = sum(float(r.get("duration_s", 0) or 0) for r in rows)
        summary_rows.append({
            "video": meta.filename, "site": meta.site, "visit_key": meta.visit_key,
            "date": meta.date, "side": meta.side, "n_events": len(rows),
            "event_seconds": round(total_event_seconds, 1),
        })
        for r in rows:
            master_rows.append({
                "video": meta.filename, "site": meta.site, "visit_key": meta.visit_key,
                "date": meta.date, "side": meta.side, **r,
            })

    if master_rows:
        cols = ["video", "site", "visit_key", "date", "side"] + [
            c for c in master_rows[0] if c not in
            ("video", "site", "visit_key", "date", "side")
        ]
        with open(os.path.join(events_root, "master_events.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(master_rows)

    with open(os.path.join(events_root, "summary_by_video.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "site", "visit_key", "date", "side",
                                          "n_events", "event_seconds"])
        w.writeheader()
        w.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw", help="Folder of raw videos")
    p.add_argument("--events-root", default="data/events")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel video workers (each video is single-threaded)")
    p.add_argument("--force", action="store_true", help="Re-mine even if already done")
    p.add_argument("--save-thumbs", action="store_true")
    # Blob/event params forwarded to the miner (keep defaults in sync with mine_events).
    p.add_argument("--sample-every", type=int, default=6)
    p.add_argument("--min-contrast", type=int, default=18)
    p.add_argument("--min-event-hits", type=int, default=2)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    blob_kwargs = dict(
        sample_every=a.sample_every,
        min_contrast=a.min_contrast,
        min_event_hits=a.min_event_hits,
    )
    batch(a.raw, a.events_root, save_thumbs=a.save_thumbs, workers=a.workers,
          force=a.force, blob_kwargs=blob_kwargs)


if __name__ == "__main__":
    main()
