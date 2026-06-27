#!/usr/bin/env python3
"""
Detect -> track -> COUNT unique deer per video (the project deliverable).

Counting unit = a confirmed *track* (one unique animal), not a per-frame detection.
A deer is visible for ~hundreds of frames at 60 fps; naive per-frame counting would
multiply it by ~300. We run the detector every frame, link detections into tracks with
BoT-SORT (global motion compensation ON — the camera moves), then count tracks that pass
a confirmation rule, each with an aggregated, per-animal confidence.

Outputs (per run):
  counts.csv   one row per candidate track: video, track_id, first/last frame + time,
               n_frames seen, span_s, mean & top-k confidence, mean box size (px,
               a distance proxy), confirmed flag.
  summary.csv  one row per video: confirmed count, counts in confidence bands, fps,
               frames, runtime.

Usage:
  python src/track/count_deer.py --weights <best.pt> --source <video.mp4 | folder> \
      --out results/counts/run1 [--save-video]
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mining"))
try:
    from filename_meta import site_from_key  # noqa: E402
except Exception:                              # pragma: no cover
    def site_from_key(k):                      # fallback: SITE__...
        return k.split("__")[0] if "__" in k else ""

KNOWN_SITES = {"MAS", "SHB", "SHW", "TON"}


def site_of(stem: str) -> str:
    """Site from either an annotate-key (SITE__transect_vN_side) or a raw video
    filename (<transect>_<SITE>_<date>_<side>, e.g. N2379Rd_MAS_12.22.25_LS)."""
    if "__" in stem:
        s = site_from_key(stem.rsplit("_f", 1)[0])
        if s:
            return s
    for tok in stem.split("_"):
        if tok in KNOWN_SITES:
            return tok
    return ""


def list_videos(source: str) -> list[str]:
    if os.path.isfile(source):
        return [source]
    vids: list[str] = []
    for ext in ("*.mp4", "*.MP4", "*.avi", "*.mov"):
        vids += glob.glob(os.path.join(source, "**", ext), recursive=True)
    return sorted(set(vids))


def video_fps(path: str) -> float:
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS) or 60.0
    c.release()
    return fps


def aggregate_conf(confs: list[float], k: int = 5) -> float:
    """Top-k mean: robust to a single lucky high frame, rewards sustained detections."""
    if not confs:
        return 0.0
    top = sorted(confs, reverse=True)[:k]
    return sum(top) / len(top)


def count_video(model, path: str, args) -> tuple[list[dict], dict]:
    fps = video_fps(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    # per track-id: list of (frame_idx, conf, area_px, w, h)
    tracks: dict[int, list[tuple]] = defaultdict(list)

    save_dir = os.path.join(args.out, "review")
    results = model.track(
        source=path, tracker=args.tracker, persist=True, stream=True,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
        verbose=False, save=args.save_video, project=save_dir, name=stem, exist_ok=True,
    )
    for fi, r in enumerate(results):
        b = r.boxes
        if b is None or b.id is None:
            continue
        ids = b.id.int().tolist()
        confs = b.conf.tolist()
        whs = b.xywh.tolist()  # [x_c, y_c, w, h]
        for tid, cf, (xc, yc, w, h) in zip(ids, confs, whs):
            tracks[tid].append((fi, float(cf), float(w * h), float(w), float(h)))

    rows = []
    for tid, obs in tracks.items():
        frames = [o[0] for o in obs]
        confs = [o[1] for o in obs]
        areas = [o[2] for o in obs]
        n = len(obs)
        span_s = (max(frames) - min(frames) + 1) / fps
        mean_conf = sum(confs) / n
        topk = aggregate_conf(confs)
        confirmed = (n >= args.min_hits and span_s >= args.min_span_s
                     and topk >= args.conf_track)
        rows.append({
            "video": stem, "site": site_of(stem),
            "track_id": tid, "first_frame": min(frames), "last_frame": max(frames),
            "first_s": round(min(frames) / fps, 2), "last_s": round(max(frames) / fps, 2),
            "n_frames": n, "span_s": round(span_s, 2),
            "mean_conf": round(mean_conf, 4), "topk_conf": round(topk, 4),
            "mean_box_px": round(sum(areas) / n, 1),
            "confirmed": int(confirmed),
        })
    rows.sort(key=lambda r: (-r["confirmed"], -r["topk_conf"]))

    conf_rows = [r for r in rows if r["confirmed"]]
    summary = {
        "video": stem, "site": site_of(stem),
        "count": len(conf_rows),
        "count_high": sum(1 for r in conf_rows if r["topk_conf"] >= 0.90),
        "count_mid": sum(1 for r in conf_rows if 0.50 <= r["topk_conf"] < 0.90),
        "candidate_tracks": len(rows), "fps": round(fps, 1),
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="trained detector best.pt")
    ap.add_argument("--source", required=True, help="video file or folder of videos")
    ap.add_argument("--out", default="results/counts/run", help="output dir for csvs")
    ap.add_argument("--tracker",
                    default=os.path.join(os.path.dirname(__file__), "botsort_deer.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    # detection: low conf to feed weak detections into the tracker (recall-first)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.5)
    # track confirmation (precision recovered at track level)
    ap.add_argument("--min-hits", type=int, default=8,
                    help="min frames a track is detected to count")
    ap.add_argument("--min-span-s", type=float, default=0.3,
                    help="min wall-time span of a track to count")
    ap.add_argument("--conf-track", type=float, default=0.30,
                    help="min top-k aggregated confidence to count a track")
    ap.add_argument("--save-video", action="store_true",
                    help="write annotated review video (slow; goes to <out>/review/)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    vids = list_videos(args.source)
    if not vids:
        raise SystemExit(f"no videos under {args.source}")
    os.makedirs(args.out, exist_ok=True)
    print(f"counting deer in {len(vids)} video(s) -> {args.out}")

    all_rows, summaries = [], []
    for i, v in enumerate(vids, 1):
        print(f"[{i}/{len(vids)}] {os.path.basename(v)}")
        rows, summary = count_video(model, v, args)
        all_rows += rows
        summaries.append(summary)
        print(f"    -> count={summary['count']} "
              f"(high={summary['count_high']}, mid={summary['count_mid']}) "
              f"from {summary['candidate_tracks']} candidate tracks")

    counts_csv = os.path.join(args.out, "counts.csv")
    with open(counts_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else
                           ["video", "track_id", "confirmed"])
        w.writeheader()
        w.writerows(all_rows)
    summ_csv = os.path.join(args.out, "summary.csv")
    with open(summ_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    total = sum(s["count"] for s in summaries)
    print(f"\nwrote {counts_csv} and {summ_csv}")
    print(f"TOTAL confirmed deer across {len(vids)} video(s): {total}")


if __name__ == "__main__":
    main()
