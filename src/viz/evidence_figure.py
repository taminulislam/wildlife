#!/usr/bin/env python3
"""
Per-animal evidence figure: two animals, each across several frames.

The single-animal version could be read as one lucky track. Two rows show that the same
mechanism holds for different individuals in different videos, and lets the reader compare
how the per-frame confidence behaves for each -- which is the point of the figure, since the
paper's claim is that a *calibrated per-track* score is meaningful where raw per-frame
detector confidence is not.

Labels are drawn in a solid bar under each crop rather than over the thermal pixels. In the
previous version they were overlaid on the image and the value was clipped at the panel
edge, so the confidence -- the one quantity the figure exists to show -- was unreadable.

Both animals come from videos the detector never trained on.

Usage:
  python src/viz/evidence_figure.py --out overleaf_WACV/figures/evidence_two.jpg
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import re
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "track"))
from count_deer import list_videos                                     # noqa: E402

# (video stem, track id, caption). Both held out; picked as the highest-confidence
# confirmed tracks in the two unseen videos with the most animals.
# Animal B was originally a NShelbyRd track. That video holds 27 animals, so neighbouring
# deer sat unboxed inside the crop and read as missed detections. N25thBlue holds 3 and
# counts all 3 correctly, so the tracked animal is alone in frame.
# Animal B was originally a NShelbyRd track. That video holds 27 animals, so neighbouring
# deer sat unboxed inside the crop and read as missed detections. N25thBlue avoided that but
# its animal is a ~30 px blob, unreadable at figure scale. Robinson trk4344 is the
# compromise: 9,244 px^2 mean box, 64 detections, top-k confidence 0.798, in a 7-animal
# video sparse enough that the tracked deer is alone in a 132 px crop.
ANIMALS = [
    ("GolfDr_SHB_12.11.2025", 198, "Animal A"),
    ("Robinson_SHW_01.18.2026_LS", 4344, "Animal B"),
]
N_PANEL = 6
CROP = 132
PAD = 26


def find_video(source: str, name: str) -> str:
    """CVAT stem -> video path.

    MAS transects were driven twice. CVAT marks the passes with a _V1/_V2 suffix, but on
    disk the distinction is the DIRECTORY (Visit1/Visit2) and the two files share a name,
    so a plain basename match finds nothing for MAS. Same fix as src/viz/dataset_figure.py.
    """
    vids = list_videos(source)
    for v in vids:
        if os.path.splitext(os.path.basename(v))[0] == name:
            return v
    m = re.match(r"^(.*)_V(\d)$", name)
    if m:
        stem, visit = m.group(1), m.group(2)
        for v in vids:
            if (os.path.splitext(os.path.basename(v))[0] == stem
                    and f"visit{visit}" in v.lower()):
                return v
    return ""


def track_rows(counts_dir: str, video: str, tid: int) -> list[tuple]:
    out = []
    for f in sorted(glob.glob(os.path.join(counts_dir, "shard*", "tracks.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["video"] == video and int(r["track_id"]) == tid:
                    out.append((int(r["frame"]), float(r["conf"]), float(r["xc"]),
                                float(r["yc"]), float(r["w"]), float(r["h"])))
    out.sort()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="/work/hdd/bgte/tislam6/wildlife_outputs/counts/"
                                        "phaseC_orphan_yolo11m_conf0.10")
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--individual", action="store_true",
                    help="also write each panel as its own file, unlabelled,\n"
                         "for composing the figure by hand")
    args = ap.parse_args()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    rows_img = []

    for video, tid, caption in ANIMALS:
        seq = track_rows(args.counts, video, tid)
        if len(seq) < N_PANEL:
            print(f"[FAIL] {video} trk{tid}: {len(seq)} detections", flush=True)
            return
        # Best frame from each of N_PANEL equal time bins. Uniform sampling pulled in the
        # track's first and last detections, where the animal is entering or leaving the
        # crop and a neighbour can sit unboxed beside it. Taking the top N by confidence
        # instead collapsed onto consecutive frames -- four of Animal B's six landed inside
        # a 4-frame window -- so the row stopped reading as time at all. Binning first and
        # maximising within each bin gives both: a clean detection in every panel, spread
        # across the track's duration.
        lo, hi = seq[0][0], seq[-1][0]
        edges = np.linspace(lo, hi + 1, N_PANEL + 1)
        picks = {}
        for i in range(N_PANEL):
            binned = [r for r in seq if edges[i] <= r[0] < edges[i + 1]]
            if binned:
                b = max(binned, key=lambda r: r[1])
                picks[b[0]] = b
        vp = find_video(args.source, video)
        if not vp:
            print(f"[FAIL] no video for {video}", flush=True); return

        cap = cv2.VideoCapture(vp)
        fr, last, panels = 0, max(picks), {}
        while fr <= last:
            ok, img = cap.read()
            if not ok:
                break
            if fr in picks:
                _f, cf, xc, yc, w, h = picks[fr]
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = clahe.apply(img)
                H, W = img.shape[:2]
                half = CROP // 2
                cx, cy = int(round(xc)), int(round(yc))
                x1, y1 = max(cx - half, 0), max(cy - half, 0)
                x2, y2 = min(x1 + CROP, W), min(y1 + CROP, H)
                x1, y1 = x2 - CROP, y2 - CROP
                crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
                bx1, by1 = int(round(xc - w / 2)) - x1, int(round(yc - h / 2)) - y1
                bx2, by2 = int(round(xc + w / 2)) - x1, int(round(yc + h / 2)) - y1
                cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 235, 0), 2)
                # label bar BELOW the image, never over the thermal pixels
                bar = np.full((22, CROP, 3), 18, np.uint8)
                cv2.putText(bar, f"t={fr/60.0:5.1f}s  conf {cf:.2f}", (4, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (245, 245, 245), 1,
                            cv2.LINE_AA)
                panels[fr] = cv2.vconcat([crop, bar])
                # also write the panel on its own, for hand-composition in a drawing tool
                if args.individual:
                    d = os.path.join(os.path.dirname(args.out) or ".", "frames")
                    os.makedirs(d, exist_ok=True)
                    tag = caption.split()[-1]
                    cv2.imwrite(os.path.join(d, f"{tag}_f{fr}_conf{cf:.2f}.jpg"),
                                crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            fr += 1
        cap.release()

        strip = [panels[f] for f in sorted(panels)]
        strip = [cv2.copyMakeBorder(p, 0, 0, 0, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                 for p in strip[:-1]] + [strip[-1]]
        row = cv2.hconcat(strip)
        # Row label rotated into the left margin. A horizontal caption line above each row
        # cost 26 px of height per animal for one short string; vertically it costs 24 px
        # of width once, and the figure is far wider than it is tall.
        # Draw horizontally on a canvas whose WIDTH is the row height, then rotate: the
        # result is 24 px wide and exactly as tall as the row, so hconcat lines up without
        # a resize. Building it at the final orientation and resizing, as a first version
        # did, squashed the glyphs into an unreadable smear.
        lab = np.full((24, row.shape[0], 3), 255, np.uint8)
        # Centre on the IMAGERY, not on the row. The row is CROP px of thermal plus a
        # 22 px label bar, and centring across the whole 154 px puts the text 11 px below
        # where the eye reads the middle of the picture.
        (tw, _th), _bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(lab, caption, (max((CROP - tw) // 2, 2), 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        lab = cv2.rotate(lab, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rows_img.append(cv2.hconcat([lab, row]))
        print(f"[ok] {caption}: {video} trk{tid}, {len(strip)} panels", flush=True)

    gap = np.full((10, rows_img[0].shape[1], 3), 255, np.uint8)
    sheet = cv2.vconcat([rows_img[0], gap, rows_img[1]])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"-> {args.out}  {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
