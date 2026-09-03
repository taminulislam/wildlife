#!/usr/bin/env python3
"""
TRACT command line: same engine as the web interface, for batch runs and Slurm jobs.

  python src/app/cli.py --video raw.mp4 --out results/app_demo
  python src/app/cli.py --video folder/ --out results/app_demo --device 0

Writes <stem>_counted.mp4 and <stem>_tracks.csv per input, plus summary.csv over all.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine                                                          # noqa: E402


def videos_in(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    out: list[str] = []
    for ext in ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV"):
        out += glob.glob(os.path.join(path, "**", ext), recursive=True)
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video file or a folder of videos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=engine.DEFAULT_WEIGHTS)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--min-span-s", type=float, default=0.0)
    ap.add_argument("--min-topk-conf", type=float, default=0.65)
    ap.add_argument("--no-candidates", action="store_true",
                    help="draw only confirmed animals, not candidate tracks")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    vids = videos_in(args.video)
    if not vids:
        raise SystemExit(f"no videos found under {args.video}")
    cfg = engine.Config(weights=args.weights, device=args.device, imgsz=args.imgsz,
                        conf=args.conf, min_frames=args.min_frames,
                        min_span_s=args.min_span_s, min_topk_conf=args.min_topk_conf,
                        draw_unconfirmed=not args.no_candidates)
    os.makedirs(args.out, exist_ok=True)

    last = [0.0]
    def progress(stage, frame, total):
        if args.quiet:
            return
        now = time.time()
        if now - last[0] < 0.5 and stage != "Done":
            return
        last[0] = now
        pct = f"{100*frame/total:5.1f}%" if total else "  ?  "
        sys.stdout.write(f"\r  {stage:<24s} {pct} ({frame}/{total or '?'})   ")
        sys.stdout.flush()

    rows = []
    for i, v in enumerate(vids, 1):
        print(f"[{i}/{len(vids)}] {os.path.basename(v)}")
        t0 = time.time()
        r = engine.run(v, args.out, cfg, progress)
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"\r  counted {r['count']} deer from {r['candidates']} candidate tracks "
              f"in {r['seconds']}s -> {os.path.basename(r['video'])}          ")

    s = os.path.join(args.out, "summary.csv")
    with open(s, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["source", "count", "candidates", "frames", "fps",
                                "width", "height", "seconds", "rule", "video", "csv"])
        w.writeheader()
        w.writerows(rows)
    print(f"\ntotal counted: {sum(r['count'] for r in rows)} across {len(rows)} video(s)")
    print(f"summary -> {s}")


if __name__ == "__main__":
    main()
