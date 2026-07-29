#!/usr/bin/env python3
"""
Merge a sharded counting run into one `merged/` directory.

count_deer.py is sharded over 4 GPUs, so each shard writes its own counts.csv /
tracks.csv covering a disjoint set of videos. Everything downstream
(label_tracks.py, calibrated_confirmer.py, export_evidence.py, count_frames.py)
expects those two files in ONE directory, so this concatenates them.

Track ids are already unique per (video, track_id) pair and shards hold disjoint
videos, so no renumbering is needed.

Usage:
  python src/track/merge_shards.py --run <run_dir> [--out <run_dir>/merged]
"""
from __future__ import annotations
import argparse
import csv
import glob
import os


def concat(paths: list[str], out_path: str) -> int:
    header, rows = None, []
    for p in paths:
        with open(p) as f:
            r = csv.DictReader(f)
            if header is None:
                header = r.fieldnames
            rows.extend(r)
    if header is None:
        return 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing shard*/")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "merged")
    os.makedirs(out, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.run, "shard*")))
    if not shards:
        raise SystemExit(f"no shard*/ dirs under {args.run}")
    for name in ("counts.csv", "summary.csv", "tracks.csv"):
        paths = [p for p in (os.path.join(s, name) for s in shards)
                 if os.path.isfile(p)]
        if not paths:
            continue
        n = concat(paths, os.path.join(out, name))
        print(f"{name:<14} {n:>7} rows from {len(paths)} shards")
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
