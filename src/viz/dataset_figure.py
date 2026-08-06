#!/usr/bin/env python3
"""
Dataset example figure: original frame on top, ground truth below.

Four columns spanning both text columns. Each column is one video; the top panel is the
raw thermal frame exactly as the sensor delivers it, the bottom is the same frame with
every ground-truth box drawn. Showing the pair rather than the annotated frame alone is
the point -- a reader who has not worked with thermal transects cannot otherwise judge how
much of the animal is actually visible, and "the boxes look right" is not the same claim
as "you could have found these yourself".

Frames are chosen from HUMAN KEYFRAMES only, never interpolated ones, so the boxes shown
are what a person actually placed. Sizes deliberately span the corpus: two large animals
close to the road and two mid-range, because the median 27 px animal is unreadable at
figure scale and would misrepresent what the annotator saw.

Both panels get the same CLAHE normalisation the detector is trained and evaluated with,
so the figure shows the pixels the model actually consumes rather than a prettier version.

Usage:
  python src/viz/dataset_figure.py --out overleaf_WACV/figures/dataset
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import re
import xml.etree.ElementTree as ET

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "track"))
from count_deer import list_videos                                     # noqa: E402

# (video stem, frame, caption fragment). Selected by scanning every human keyframe for the
# largest box that clears all four frame edges by 40 px. That edge constraint matters more
# than it sounds: the corpus's three largest annotations (182, 143 and 96 px) are all
# animals half outside the frame, which in a figure reads as a cropping mistake rather than
# a close approach. Site coverage was tried and abandoned -- forcing one panel per site
# pulled in two edge-clipped frames and made the figure worse than picking on legibility.
PANELS = [
    ("Robinson_SHW_01.18.2026_LS",       14820, "SHW -- large and small in one frame"),
    ("SHog_SHW_01.18.2026_LS",            8760, "SHW -- large"),
    ("GolfDr_SHB_12.11.2025",             3413, "SHB -- group at three ranges"),
    ("OikosRd_TON_12.03.25_LS",           4908, "TON -- medium"),
]


def gt_boxes(xml_path: str, frame: int) -> list[tuple]:
    """Every non-outside deer box on one frame, keyframe or not -- a panel should show the
    whole scene's annotation, not only the animal that got it selected."""
    out = []
    for tr in ET.parse(xml_path).getroot().findall("track"):
        if tr.get("label") != "deer":
            continue
        for b in tr.findall("box"):
            if int(b.get("frame")) != frame or b.get("outside") == "1":
                continue
            out.append(tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")))
    return out


def find_video(source: str, name: str) -> str:
    """CVAT stem -> video path.

    MAS transects were driven twice and CVAT distinguishes the passes with a _V1/_V2
    suffix, but on disk the distinction is the DIRECTORY (Visit1/Visit2) and both files
    share a name. A plain basename match therefore finds nothing for MAS, and silently
    returning "" would have left the previous run's images in place under a new caption.
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


def grab(path: str, frame: int):
    """Sequential decode -- frame seeking is unreliable on these files."""
    cap = cv2.VideoCapture(path)
    img, i = None, 0
    while i <= frame:
        ok, f = cap.read()
        if not ok:
            break
        if i == frame:
            img = f
        i += 1
    cap.release()
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--out", required=True)
    ap.add_argument("--contrast", default="clahe")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) \
        if args.contrast == "clahe" else None

    for i, (video, frame, tag) in enumerate(PANELS, 1):
        vp = find_video(args.source, video)
        xml = os.path.join(args.cvat_dir, f"{video}.xml")
        # Remove any previous run's panel FIRST. A skip that leaves stale files behind
        # puts the wrong frame under a new caption, which no validation would catch.
        for suf in ("raw", "gt"):
            f = os.path.join(args.out, f"p{i}_{suf}.jpg")
            if os.path.exists(f):
                os.remove(f)
        if not vp or not os.path.exists(xml):
            print(f"[FAIL] p{i} {video}: video={bool(vp)} xml={os.path.exists(xml)}",
                  flush=True)
            continue
        img = grab(vp, frame)
        if img is None:
            print(f"[skip] {video}: frame {frame} not decodable", flush=True)
            continue
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if clahe is not None:
            img = clahe.apply(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(os.path.join(args.out, f"p{i}_raw.jpg"), rgb,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        ann = rgb.copy()
        boxes = gt_boxes(xml, frame)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), (0, 235, 0), 2)
        cv2.imwrite(os.path.join(args.out, f"p{i}_gt.jpg"), ann,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[ok] p{i} {video} f{frame} ({tag}) -- {len(boxes)} GT box(es)", flush=True)

    print(f"\n-> {args.out}/  (p1..p4 x raw/gt)")


if __name__ == "__main__":
    main()
