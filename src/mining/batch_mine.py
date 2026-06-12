"""
Run the warm-blob event miner over every video in a folder and build a master index.

Unattended triage over the whole archive (~96 videos). For each video it calls the same
miner as ``mine_events.py`` (output keyed by ``VideoMeta.key`` so same-named transects
across visits don't collide), then aggregates all events into one
``data/events/master_events.csv`` enriched with site/transect/visit/side metadata, plus
``data/events/summary_by_video.csv`` with per-video event counts.

Per project decision, visits are combined into their site — grouping/splitting is by
SITE, never by visit.

Usage:
    python src/mining/batch_mine.py --raw data/raw
    python src/mining/batch_mine.py --raw data/raw --save-thumbs --workers 4

Re-running skips videos already mined unless --force is given, so it is safe to run
incrementally as downloads complete.
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from filename_meta import iter_videos, parse_path
from mine_events import mine


def _mine_one(args: tuple) -> dict:
    video_path, events_root, save_thumbs, blob_kwargs = args
    meta = parse_path(video_path)
    out_dir = os.path.join(events_root, meta.key)
    events = mine(video_path, out_dir, save_thumbs=save_thumbs, **blob_kwargs)
    return {"video_path": video_path, "n_events": len(events)}


def already_mined(events_root: str, meta) -> bool:
    return os.path.isfile(os.path.join(events_root, meta.key, "events.csv"))


def batch(raw_dir: str, events_root: str, *, save_thumbs: bool, workers: int,
          force: bool, blob_kwargs: dict) -> None:
    os.makedirs(events_root, exist_ok=True)
    catalog = list(iter_videos(raw_dir))
    if not catalog:
        print(f"No videos found under {raw_dir}.")
        return

    todo = [p for p, m in catalog if force or not already_mined(events_root, m)]
    skipped = len(catalog) - len(todo)
    print(f"Found {len(catalog)} videos; mining {len(todo)}"
          + (f", skipping {skipped} already done" if skipped else "") + ".")

    tasks = [(p, events_root, save_thumbs, blob_kwargs) for p in todo]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_mine_one, t) for t in tasks]
            for fut in as_completed(futs):
                fut.result()
    else:
        for t in tasks:
            _mine_one(t)

    # Rebuild the master index from ALL mined videos so it's complete on incremental runs.
    _rebuild_master_index(events_root, catalog)
    print(f"Done. Master index -> {os.path.join(events_root, 'master_events.csv')}")


def _read_events_csv(out_dir: str) -> list[dict]:
    path = os.path.join(out_dir, "events.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _rebuild_master_index(events_root: str, catalog: list) -> None:
    master_rows: list[dict] = []
    summary_rows: list[dict] = []
    meta_cols = ["video", "key", "site", "transect", "visit", "date", "side"]
    for video_path, meta in catalog:
        out_dir = os.path.join(events_root, meta.key)
        rows = _read_events_csv(out_dir)
        base = {"video": meta.filename, "key": meta.key, "site": meta.site,
                "transect": meta.transect, "visit": meta.visit, "date": meta.date,
                "side": meta.side}
        total_event_seconds = sum(float(r.get("duration_s", 0) or 0) for r in rows)
        summary_rows.append({**base, "n_events": len(rows),
                             "event_seconds": round(total_event_seconds, 1)})
        for r in rows:
            master_rows.append({**base, **r})

    if master_rows:
        cols = meta_cols + [c for c in master_rows[0] if c not in meta_cols]
        with open(os.path.join(events_root, "master_events.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(master_rows)

    with open(os.path.join(events_root, "summary_by_video.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=meta_cols + ["n_events", "event_seconds"])
        w.writeheader()
        w.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--events-root", default="data/events")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel video workers (each video is single-threaded)")
    p.add_argument("--force", action="store_true", help="Re-mine even if already done")
    p.add_argument("--save-thumbs", action="store_true")
    p.add_argument("--sample-every", type=int, default=6)
    p.add_argument("--min-contrast", type=int, default=18)
    p.add_argument("--min-event-hits", type=int, default=2)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    blob_kwargs = dict(sample_every=a.sample_every, min_contrast=a.min_contrast,
                       min_event_hits=a.min_event_hits)
    batch(a.raw, a.events_root, save_thumbs=a.save_thumbs, workers=a.workers,
          force=a.force, blob_kwargs=blob_kwargs)


if __name__ == "__main__":
    main()
