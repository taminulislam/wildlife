#!/usr/bin/env python3
"""
Prepare the harvest dir for a manual audit pass in src/annotate/server.py.

Does two things:
 1. Extracts the PEAK frame of every mined event that was NEVER reviewed (in
    master_events.csv but absent from data/annotate/frames.csv) into the harvest dir
    with an EMPTY label, so the audit covers *all* events, not just known deer events.
 2. Writes data/annotate/harvest/frames.csv — the index server.py reads — ordered for
    efficient review: deer-event frames first (group events first, frames of one event
    kept together), then the new unreviewed peaks. The `score` column carries a short
    status tag ("g6/propagated", "lost", "NEW", ...) shown in the sidebar.

Run:
    python src/dataset/prepare_harvest_review.py
    python src/annotate/server.py --root data/annotate/harvest
"""
from __future__ import annotations
import argparse, csv, glob, os
import cv2


def load_reviewed_event_keys(frames_csv):
    keys = set()
    with open(frames_csv) as f:
        for r in csv.DictReader(f):
            keys.add((r["key"], r["event_id"]))
    return keys


def index_videos(raw_dir):
    return {os.path.basename(p): p
            for p in glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)}


def extract_unreviewed_peaks(master_csv, frames_csv, raw_dir, out_frames, out_labels):
    reviewed = load_reviewed_event_keys(frames_csv)
    vids = index_videos(raw_dir)
    rows = []
    with open(master_csv) as f:
        events = list(csv.DictReader(f))
    todo = [e for e in events if (e["key"], e["event_id"]) not in reviewed]
    print(f"unreviewed events to extract: {len(todo)}")
    for e in todo:
        vp = vids.get(e["video"])
        if not vp:
            print(f"  [warn] no video for {e['key']} ({e['video']})")
            continue
        peak = int(e["peak_frame"])
        cap = cv2.VideoCapture(vp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, peak)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            print(f"  [warn] could not read peak {peak} of {e['key']}")
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        name = f"{e['key']}_f{peak}"
        cv2.imwrite(os.path.join(out_frames, name + ".png"), gray)
        open(os.path.join(out_labels, name + ".txt"), "w").close()  # empty label
        rows.append({"name": name + ".png", "key": e["key"], "site": e["site"],
                     "event_id": e["event_id"], "src_frame": peak,
                     "score": "NEW", "_order": (3, 0, e["key"], peak)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default="data/annotate/harvest")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--master", default="data/events/master_events.csv")
    ap.add_argument("--frames-csv", default="data/annotate/frames.csv")
    ap.add_argument("--skip-unreviewed", action="store_true",
                    help="only rebuild the index, don't extract the 79 new peaks")
    args = ap.parse_args()

    out_frames = os.path.join(args.harvest, "frames")
    out_labels = os.path.join(args.harvest, "labels")
    os.makedirs(out_frames, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    rows = []
    # 1. harvested deer-event frames (from the manifest)
    man = os.path.join(args.harvest, "harvest_manifest.csv")
    if os.path.isfile(man):
        with open(man) as f:
            for r in csv.DictReader(f):
                g = int(r["orig_boxes"])         # group size of the source event
                tag = ("verified" if r["is_peak"] == "1" else r["status"])
                score = f"g{g}/{tag}"
                # order: group events first (-g), keep an event's frames together, by frame
                rows.append({"name": r["name"], "key": r["key"], "site": r["site"],
                             "event_id": r["event_id"], "src_frame": r["src_frame"],
                             "score": score,
                             "_order": (1, -g, r["key"] + r["event_id"],
                                        int(r["src_frame"]))})
    else:
        print(f"[warn] {man} not found — run harvest_event_frames.py first")

    # 2. unreviewed event peaks
    if not args.skip_unreviewed:
        rows += extract_unreviewed_peaks(args.master, args.frames_csv, args.raw,
                                         out_frames, out_labels)

    rows.sort(key=lambda r: r["_order"])
    out_csv = os.path.join(args.harvest, "frames.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "key", "site", "event_id",
                                          "src_frame", "score"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in
                        ["name", "key", "site", "event_id", "src_frame", "score"]})
    print(f"\nwrote {out_csv}: {len(rows)} frames to audit")
    print("start the audit:  python src/annotate/server.py --root "
          f"{args.harvest}")


if __name__ == "__main__":
    main()
