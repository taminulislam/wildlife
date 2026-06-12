"""
Select a balanced first annotation batch from the master event index.

Annotating every mined event across 96 videos would be wasteful and skewed toward
action-heavy videos. This picks a diverse, capped subset that spreads annotation effort
across all sites and visits, so the labeled set represents every location — which is what
makes a site-held-out test split meaningful.

Selection strategy (greedy, diversity-first):
  * Group events by site, then by video.
  * Round-robin across sites so no site dominates the batch.
  * Within a site, prefer a spread of event "scores" and durations (a mix of strong/weak
    and short/long events), not just the highest-scoring ones.
  * Cap events per video and a global total.

Output: ``data/dataset/annotation_batch.csv`` — the chosen events, ready to drive the
clip extractor (annotate just these clips first).

Usage:
    python src/dataset/select_for_annotation.py --total 400 --per-video 8
    python src/dataset/select_for_annotation.py --min-duration 0.5 --min-score 22
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def load_master(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise SystemExit(f"Master index not found: {path}. Run batch_mine.py first.")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _spread_order(events: list[dict]) -> list[dict]:
    """Order a video's events to maximize variety: alternate high/low score so a greedy
    per-video cap grabs a spread rather than the top-N near-identical strong events."""
    by_score = sorted(events, key=lambda e: float(e["peak_score"]), reverse=True)
    out, lo, hi = [], 0, len(by_score) - 1
    take_high = True
    while lo <= hi:
        if take_high:
            out.append(by_score[lo]); lo += 1
        else:
            out.append(by_score[hi]); hi -= 1
        take_high = not take_high
    return out


def select(rows: list[dict], *, total: int, per_video: int,
           min_duration: float, min_score: float) -> list[dict]:
    # Filter by quality gates.
    rows = [r for r in rows
            if float(r.get("duration_s", 0) or 0) >= min_duration
            and float(r.get("peak_score", 0) or 0) >= min_score]

    # site -> video(key) -> [events], each video's events ordered for variety.
    # Group by 'key' not 'video': same transect filename can recur across visits.
    by_site: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_site[r["site"]][r["key"]].append(r)
    for site in by_site:
        for vid in by_site[site]:
            by_site[site][vid] = _spread_order(by_site[site][vid])

    # Round-robin across sites, then across that site's videos, honoring per-video cap.
    taken: list[dict] = []
    taken_per_video: dict[str, int] = defaultdict(int)
    sites = list(by_site)
    # cursors[site][video] = next index to take
    cursors: dict[str, dict[str, int]] = {
        s: {v: 0 for v in by_site[s]} for s in by_site
    }

    progressed = True
    while len(taken) < total and progressed:
        progressed = False
        for site in sites:
            for vid in by_site[site]:
                if len(taken) >= total:
                    break
                if taken_per_video[vid] >= per_video:
                    continue
                cur = cursors[site][vid]
                evs = by_site[site][vid]
                if cur < len(evs):
                    taken.append(evs[cur])
                    cursors[site][vid] += 1
                    taken_per_video[vid] += 1
                    progressed = True
    return taken


def write_batch(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["video", "key", "site", "transect", "visit", "date", "side", "event_id",
            "start_frame", "end_frame", "start_s", "end_s", "duration_s",
            "n_hits", "peak_frame", "peak_score"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict]) -> None:
    by_site: dict[str, int] = defaultdict(int)
    for r in rows:
        by_site[r["site"]] += 1
    print(f"Selected {len(rows)} events across {len(by_site)} sites:")
    for site, n in sorted(by_site.items()):
        print(f"  {site}: {n}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default="data/events/master_events.csv")
    p.add_argument("--out", default="data/dataset/annotation_batch.csv")
    p.add_argument("--total", type=int, default=400, help="Max events in the batch")
    p.add_argument("--per-video", type=int, default=8, help="Max events per video")
    p.add_argument("--min-duration", type=float, default=0.4,
                   help="Drop events shorter than this (s)")
    p.add_argument("--min-score", type=float, default=20.0,
                   help="Drop events whose peak local-contrast score is below this")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    rows = load_master(a.master)
    chosen = select(rows, total=a.total, per_video=a.per_video,
                    min_duration=a.min_duration, min_score=a.min_score)
    write_batch(chosen, a.out)
    summarize(chosen)
    print(f"Batch -> {a.out}")


if __name__ == "__main__":
    main()
