#!/usr/bin/env python3
"""
Fold the human-audited harvest set into the canonical annotate set.

After auditing data/annotate/harvest/ in src/annotate/server.py, this merges the
verified frames + labels into data/annotate/{frames,labels}/ and updates
data/annotate/frames.csv (the index). Harvest is authoritative: when a frame
exists in both (a peak that was re-audited), the harvest label WINS.

A frame's label may be empty — that's a verified negative (no deer), and it is
kept (negatives matter for training).

Run:
    python src/dataset/merge_harvest.py --dry-run   # preview
    python src/dataset/merge_harvest.py             # apply
"""
from __future__ import annotations
import argparse
import csv
import filecmp
import os
import shutil


def read_index(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if os.path.isfile(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["name"]] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate", default="data/annotate")
    ap.add_argument("--harvest", default="data/annotate/harvest")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src_frames = os.path.join(a.harvest, "frames")
    src_labels = os.path.join(a.harvest, "labels")
    dst_frames = os.path.join(a.annotate, "frames")
    dst_labels = os.path.join(a.annotate, "labels")
    for d in (dst_frames, dst_labels):
        os.makedirs(d, exist_ok=True)

    h_index = read_index(os.path.join(a.harvest, "frames.csv"))
    a_index = read_index(os.path.join(a.annotate, "frames.csv"))

    harvest_pngs = sorted(f for f in os.listdir(src_frames) if f.endswith(".png"))

    new_img = overwritten_img = same_img = 0
    new_lbl = changed_lbl = same_lbl = 0
    with_box = empty = 0

    for png in harvest_pngs:
        stem = os.path.splitext(png)[0]
        txt = stem + ".txt"
        s_img, d_img = os.path.join(src_frames, png), os.path.join(dst_frames, png)
        s_lbl, d_lbl = os.path.join(src_labels, txt), os.path.join(dst_labels, txt)

        # --- image ---
        if not os.path.exists(d_img):
            new_img += 1
            if not a.dry_run:
                shutil.copy2(s_img, d_img)
        elif filecmp.cmp(s_img, d_img, shallow=False):
            same_img += 1
        else:
            overwritten_img += 1
            if not a.dry_run:
                shutil.copy2(s_img, d_img)

        # --- label (harvest wins) ---
        has_box = os.path.isfile(s_lbl) and os.path.getsize(s_lbl) > 0
        with_box += int(has_box)
        empty += int(not has_box)
        if not os.path.exists(d_lbl):
            new_lbl += 1
        elif filecmp.cmp(s_lbl, d_lbl, shallow=False):
            same_lbl += 1
        else:
            changed_lbl += 1
        if not a.dry_run:
            if os.path.isfile(s_lbl):
                shutil.copy2(s_lbl, d_lbl)
            else:
                open(d_lbl, "w").close()

        # --- index row (harvest score wins) ---
        a_index[png] = h_index.get(png, a_index.get(png, {
            "name": png, "key": stem.rsplit("_f", 1)[0], "site": "",
            "event_id": "", "src_frame": "", "score": "merged"}))

    if not a.dry_run:
        out_csv = os.path.join(a.annotate, "frames.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "key", "site", "event_id",
                                              "src_frame", "score"])
            w.writeheader()
            for name in sorted(a_index):
                r = a_index[name]
                w.writerow({k: r.get(k, "") for k in
                            ["name", "key", "site", "event_id", "src_frame", "score"]})

    tag = "[dry-run] " if a.dry_run else ""
    print(f"{tag}harvest frames merged: {len(harvest_pngs)}  "
          f"({with_box} with deer, {empty} empty negatives)")
    print(f"{tag}images : {new_img} new, {overwritten_img} updated, {same_img} identical")
    print(f"{tag}labels : {new_lbl} new, {changed_lbl} changed (harvest won), "
          f"{same_lbl} identical")
    print(f"{tag}annotate index now: {len(a_index)} frames")


if __name__ == "__main__":
    main()
