#!/usr/bin/env python3
"""
Render the deer the counter MISSED, so a human can check whether they are real.

§6.10 leaves 36 of 83 unseen deer uncounted: 9 the detector never fired on, 27 that got a
candidate the rule then rejected. Their statistics differ sharply from the counted ones —
median 20 px and 28 px against 38 px — which is consistent with "small and faint".

But three deer in GiantCityRd carry boxes of 104, 149 and 157 px against a corpus median
of 27, and one of those is UNDETECTED at 149 px. A detector that finds 27 px deer does not
miss a 149 px one. Either those annotations are not deer, or they are group boxes, or the
track is mislabelled. That is a ground-truth question and it can only be settled by looking.

So this crops each missed deer at three points in its GT track (first / middle / last),
draws the GT box, and writes one panel per animal. Filenames sort worst-first by size so
the suspicious ones are on top:

    m01_undetected_sz149_GiantCityRd_deer6.jpg

Counted deer are rendered too (--include-counted) as a visual control: if the missed ones
look like the counted ones, the misses are a model problem; if they look like empty grass,
they are a label problem.

Usage:
  python src/viz/missed_deer.py --missed /tmp/missed.json --source data/raw \
      --out /work/hdd/bgte/tislam6/wildlife_outputs/viz/missed_deer
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "temporal"))
sys.path.insert(0, os.path.join(_HERE, "..", "track"))
from label_tracks import gt_tracks_of                                  # noqa: E402
from count_deer import list_videos                                     # noqa: E402

PAD = 60          # context around the box, so the animal is judged in its surroundings


def find_video(source: str, name: str) -> str:
    for v in list_videos(source):
        if os.path.splitext(os.path.basename(v))[0] == name:
            return v
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missed", default="/tmp/missed.json")
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-counted", action="store_true",
                    help="also render counted deer, as a visual control")
    ap.add_argument("--contrast", default="clahe")
    args = ap.parse_args()

    rows = json.load(open(args.missed))
    want = [r for r in rows if args.include_counted or r["state"] != "counted"]
    by_video: dict[str, list] = {}
    for r in want:
        by_video.setdefault(r["video"], []).append(r)
    os.makedirs(args.out, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) \
        if args.contrast == "clahe" else None

    # rank worst-first by box size so the implausible ones land at the top of a listing
    rank = {(r["video"], r["gi"]): i
            for i, r in enumerate(sorted(want, key=lambda r: -r["sz"]), 1)}

    for video, items in sorted(by_video.items()):
        vp = find_video(args.source, video)
        xml = os.path.join(args.cvat_dir, f"{video}.xml")
        if not vp or not os.path.exists(xml):
            print(f"[skip] {video}: video={bool(vp)} xml={os.path.exists(xml)}", flush=True)
            continue
        gts = gt_tracks_of(xml)

        # three sample frames per deer: first, middle, last of its GT track
        need: dict[int, list] = {}
        for r in items:
            g = gts[r["gi"]]
            frs = sorted(g)
            for tag, fr in (("first", frs[0]), ("mid", frs[len(frs) // 2]),
                            ("last", frs[-1])):
                need.setdefault(fr, []).append((r, tag, g[fr]))

        got: dict[tuple, dict] = {}
        cap = cv2.VideoCapture(vp)
        fr, last = 0, max(need)
        while fr <= last:
            ok, img = cap.read()
            if not ok:
                break
            for (r, tag, box) in need.get(fr, []):
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if clahe is not None:
                    img = clahe.apply(img)
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                H, W = img.shape[:2]
                cx1, cy1 = max(x1 - PAD, 0), max(y1 - PAD, 0)
                cx2, cy2 = min(x2 + PAD, W), min(y2 + PAD, H)
                crop = cv2.cvtColor(img[cy1:cy2, cx1:cx2], cv2.COLOR_GRAY2BGR)
                cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1),
                              (0, 0, 255), 1)
                cv2.putText(crop, f"f{fr} {tag}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 255, 255), 1)
                got.setdefault((r["video"], r["gi"]), {})[tag] = crop
            fr += 1
        cap.release()

        for r in items:
            panels = got.get((r["video"], r["gi"]))
            if not panels:
                continue
            imgs = [panels[t] for t in ("first", "mid", "last") if t in panels]
            h = max(i.shape[0] for i in imgs)
            imgs = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 6,
                                       cv2.BORDER_CONSTANT, value=(40, 40, 40))
                    for i in imgs]
            sheet = cv2.hconcat(imgs)
            name = (f"m{rank[(r['video'], r['gi'])]:02d}_{r['state']}"
                    f"_sz{r['sz']:.0f}_{r['video'][:24]}_deer{r['gi']}.jpg")
            cv2.imwrite(os.path.join(args.out, name), sheet)
        print(f"[ok] {video}: {len(items)} deer rendered", flush=True)

    print(f"\n-> {args.out}")
    print("Filenames sort worst-first by box size. A 'deer' of 100+ px that the detector "
          "never fired on is a ground-truth question, not a model failure.")


if __name__ == "__main__":
    main()
