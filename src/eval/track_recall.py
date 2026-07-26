#!/usr/bin/env python3
"""
Phase-B gate metric: **track-level recall** — of the 236 GT deer (one CVAT track =
one unique animal), how many does the detector find *at all*?

Why this and not mAP: counting only needs each animal detected in a FEW frames for the
tracker to form a candidate track and the temporal head to confirm it. Frame-level mAP50
of ~0.5 is compatible with track-level recall of 90%+ (misses scattered across frames)
OR with 60% (whole animals never seen) — only this measurement distinguishes them.
A deer detected in ZERO frames is unrecoverable by any downstream stage: that is the
real ceiling on counting accuracy.

Detection is run ONLY on frames where a GT box exists (subsampled by --stride), so this
is cheap compared to full-video inference.

Outputs (to --out):
  track_recall.csv    one row per GT track: video, split, track_id, gt_frames, checked,
                      hits, best_iou, best_conf, recovered@1, recovered@3
  track_recall_summary.csv  recall by split and by site, at several conf thresholds

Usage:
  python src/eval/track_recall.py --weights <best.pt> --conf 0.10 --out results/track_recall
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
from thermal import enhance_contrast  # noqa: E402
sys.path.insert(0, os.path.join(_HERE, "..", "dataset"))
from cvat_to_yolo import parse_cvat, find_video  # noqa: E402
sys.path.insert(0, os.path.join(_HERE, "..", "mining"))
from filename_meta import parse_path  # noqa: E402


def split_map(dataset_root: str) -> dict[str, str]:
    """video-key -> train/val/test, read from the materialized split dirs."""
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        d = os.path.join(dataset_root, "labels", split)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            key = fn.rsplit("_f", 1)[0]
            out[key] = split
    return out


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center_inside(gt, det) -> bool:
    """Detection centre falls inside the GT box (or vice versa) — the counting-relevant
    question: did the detector point AT this animal, regardless of box tightness."""
    for a, b in ((gt, det), (det, gt)):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if a[0] <= cx <= a[2] and a[1] <= cy <= a[3]:
            return True
    return False


# Matching criteria, strict -> permissive. For COUNTING, `touch` and `center` are the
# operationally meaningful ones (a box overlapping the deer is enough to count it);
# iou50 is kept because the detection table must report the standard definition.
CRITERIA = ("iou50", "iou30", "touch", "center")


def match_kind(gt, det) -> set:
    """Which criteria this (gt, det) pair satisfies."""
    v = iou_xyxy(gt, det)
    out = set()
    if v >= 0.5:
        out.add("iou50")
    if v >= 0.3:
        out.add("iou30")
    if v > 0:
        out.add("touch")          # ANY overlap at all
    if center_inside(gt, det):
        out.add("center")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arch", default="yolo", choices=["yolo", "rtdetr"])
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--dataset-root", default="data/dataset/yolo_v3")
    ap.add_argument("--out", default="results/track_recall")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="LOW conf on purpose: counting over-generates candidates and "
                         "lets the temporal head confirm them")
    ap.add_argument("--iou-match", type=float, default=0.3,
                    help="IoU for calling a detection a hit on a GT box (0.3: deer are "
                         "~27px, so strict 0.5 punishes tiny localization error)")
    ap.add_argument("--stride", type=int, default=3,
                    help="check every Nth GT frame of each track")
    ap.add_argument("--max-check", type=int, default=60,
                    help="cap frames checked per track (long tracks don't need all)")
    ap.add_argument("--contrast", default="clahe", choices=["clahe", "stretch", "none"])
    args = ap.parse_args()

    from ultralytics import YOLO, RTDETR
    model = (RTDETR if args.arch == "rtdetr" else YOLO)(args.weights)

    splits = split_map(args.dataset_root)
    os.makedirs(args.out, exist_ok=True)
    rows = []

    xmls = sorted(f for f in os.listdir(args.cvat_dir) if f.endswith(".xml"))
    for xi, xml in enumerate(xmls, 1):
        xpath = os.path.join(args.cvat_dir, xml)
        stem = os.path.splitext(xml)[0].replace("_annotations", "")
        vpath = find_video(args.source, stem)
        if vpath is None:
            print(f"[{xi}/{len(xmls)}] {stem}: VIDEO NOT FOUND — skipped", flush=True)
            continue
        key = parse_path(vpath).key
        site = key.split("__")[0]
        split = splits.get(key, "?")
        n_tracks, by_frame, keyframes, per_track = parse_cvat(xpath)
        if not per_track:
            print(f"[{xi}/{len(xmls)}] {stem}: 0 deer tracks", flush=True)
            continue

        # frames we must decode = union of the (subsampled) frames of every track
        want: dict[int, list[int]] = defaultdict(list)   # frame -> [track idx]
        checked_per_track: list[int] = []
        for ti, (tframes, tkeys) in enumerate(per_track):
            sub = tframes[::max(1, args.stride)]
            if args.max_check and len(sub) > args.max_check:
                step = len(sub) / args.max_check
                sub = [sub[int(i * step)] for i in range(args.max_check)]
            sub = sorted(set(sub) | set(list(tkeys)[:10]))
            checked_per_track.append(len(sub))
            for f in sub:
                want[f].append(ti)

        hits = {c: [0] * len(per_track) for c in CRITERIA}
        best_iou = [0.0] * len(per_track)
        best_conf = [0.0] * len(per_track)

        cap = cv2.VideoCapture(vpath)
        fi = -1
        remaining = dict(want)
        while remaining:
            ok, frame = cap.read()
            if not ok:
                break
            fi += 1
            tidx = remaining.pop(fi, None)
            if tidx is None:
                continue
            frame = enhance_contrast(frame, method=args.contrast)
            r = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                              device=args.device, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            dets = r.boxes.xyxy.cpu().numpy()
            dconf = r.boxes.conf.cpu().numpy()
            gtb = by_frame.get(fi, [])
            for ti in tidx:
                # GT boxes of THIS track on THIS frame (boxes are stored per-frame, so
                # the best-matching GT box on the frame stands in for this track).
                bi, bc = 0.0, 0.0
                got: set = set()
                for g in gtb:
                    for d, c in zip(dets, dconf):
                        got |= match_kind(g, d)
                        v = iou_xyxy(g, d)
                        if v > bi:
                            bi, bc = v, float(c)
                for crit in got:
                    hits[crit][ti] += 1
                best_iou[ti] = max(best_iou[ti], bi)
                best_conf[ti] = max(best_conf[ti], bc)
        cap.release()

        for ti in range(len(per_track)):
            row = {
                "video": stem, "key": key, "site": site, "split": split,
                "track_idx": ti,
                "gt_frames": len(per_track[ti][0]),
                "checked": checked_per_track[ti],
                "best_iou": round(best_iou[ti], 3),
                "best_conf": round(best_conf[ti], 3),
            }
            for c in CRITERIA:
                row[f"hits_{c}"] = hits[c][ti]
                row[f"found_{c}"] = int(hits[c][ti] >= 1)
            rows.append(row)
        r_strict = sum(h >= 1 for h in hits["iou50"])
        r_count = sum(h >= 1 for h in hits["touch"])
        nt = len(per_track)
        print(f"[{xi}/{len(xmls)}] {stem} ({site},{split}): deer found — "
              f"counting(any-overlap) {r_count}/{nt} ({100*r_count/nt:.0f}%) | "
              f"strict(IoU>=.5) {r_strict}/{nt} ({100*r_strict/nt:.0f}%)", flush=True)

    # ---- write per-track rows ----
    fcsv = os.path.join(args.out, "track_recall.csv")
    with open(fcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- summaries ----
    scsv = os.path.join(args.out, "track_recall_summary.csv")
    with open(scsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "value", "gt_deer"] +
                   [f"recall_{c}" for c in CRITERIA])
        for gname, keyf in (("ALL", lambda r: "all"),
                            ("split", lambda r: r["split"]),
                            ("site", lambda r: r["site"])):
            groups = defaultdict(list)
            for r in rows:
                groups[keyf(r)].append(r)
            for gv, rs in sorted(groups.items()):
                n = len(rs)
                w.writerow([gname, gv, n] +
                           [round(sum(r[f"found_{c}"] for r in rs) / n, 4)
                            for c in CRITERIA])

    n = len(rows)
    print("\n=== TRACK-LEVEL RECALL (the Phase-B gate) ===")
    print(f"  GT deer (tracks): {n}\n")
    label = {"iou50": "strict IoU>=0.50 (detection-paper standard)",
             "iou30": "IoU>=0.30",
             "touch": "ANY overlap  <- the counting criterion",
             "center": "centre inside box"}
    for c in CRITERIA:
        f1 = sum(r[f"found_{c}"] for r in rows)
        f3 = sum(int(r[f"hits_{c}"] >= 3) for r in rows)
        print(f"  {label[c]:<44} found>=1frame {f1:3d}/{n} ({100*f1/n:5.1f}%)"
              f"   >=3frames {f3:3d}/{n} ({100*f3/n:5.1f}%)")
    print(f"\n  per-track: {fcsv}\n  summary : {scsv}")


if __name__ == "__main__":
    main()
