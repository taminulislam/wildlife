#!/usr/bin/env python3
"""
Cache every ground-truth deer crop, keyed by individual, so ReID training never re-decodes.

One CVAT track = one animal, so `(video, track_index)` IS the identity label — this corpus
already carries ReID annotation, it just was never used that way.

Decoding is the expensive part (521 930 frames), and the trainer needs many passes over
the data, so it happens exactly once here. Frame seeking is unreliable on these files, so
every video is walked sequentially and crops are taken as their frames go past.

Crops are stored at 64x64 grey. The animals are ~29x24 px native, so this upsamples; the
point is a fixed tensor shape for the CNN, not invented detail.

Output: one .npz per video with
    crops  uint8  [N, 64, 64]
    ident  int32  [N]        index of the CVAT track = the individual
    frame  int32  [N]        source frame, so the trainer can split a track TEMPORALLY
    box    int32  [N, 2]     native w, h before resize (a real cue: relative size)

Usage:
  python src/reid/build_crops.py --source data/raw --cvat-dir data/cvat_export \
      --out /work/hdd/bgte/tislam6/wildlife_outputs/reid_crops --shards 4 --shard 0
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "temporal"))
sys.path.insert(0, os.path.join(_HERE, "..", "track"))
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
from label_tracks import gt_tracks_of                                  # noqa: E402
from count_deer import list_videos                                     # noqa: E402

SIZE = 64


def find_video(source: str, name: str) -> str:
    for v in list_videos(source):
        if os.path.splitext(os.path.basename(v))[0] == name:
            return v
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--contrast", default="clahe",
                    help="must match what the detector saw; 'none' to disable")
    ap.add_argument("--max-per-ident", type=int, default=400,
                    help="cap so a 1000-frame track cannot dominate training")
    args = ap.parse_args()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) \
        if args.contrast == "clahe" else None

    os.makedirs(args.out, exist_ok=True)
    xmls = sorted(glob.glob(os.path.join(args.cvat_dir, "*.xml")))
    xmls = [x for i, x in enumerate(xmls) if i % args.shards == args.shard]

    for x in xmls:
        name = os.path.splitext(os.path.basename(x))[0].replace("_annotations", "")
        dst = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(dst):
            print(f"[have] {name}", flush=True)
            continue
        vp = find_video(args.source, name)
        if not vp:
            print(f"[skip] no video for {name}", flush=True)
            continue
        tracks = gt_tracks_of(x)
        if not tracks:
            print(f"[skip] {name}: no deer tracks", flush=True)
            continue

        # frame -> [(identity, box), ...]; thin each identity down to the cap up front so
        # the sequential walk stays cheap on a 68k-frame video
        need: dict[int, list] = {}
        for ti, g in enumerate(tracks):
            frs = sorted(g)
            if len(frs) > args.max_per_ident:
                idx = np.linspace(0, len(frs) - 1, args.max_per_ident).astype(int)
                frs = [frs[i] for i in idx]
            for fr in frs:
                need.setdefault(fr, []).append((ti, g[fr]))

        crops, ident, frame, box = [], [], [], []
        cap = cv2.VideoCapture(vp)
        fr, last = 0, max(need)
        while fr <= last:
            ok, img = cap.read()
            if not ok:
                break
            if fr in need:
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if clahe is not None:
                    img = clahe.apply(img)
                H, W = img.shape[:2]
                for (ti, b) in need[fr]:
                    x1, y1, x2, y2 = (int(round(v)) for v in b)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, W), min(y2, H)
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        continue
                    crops.append(cv2.resize(img[y1:y2, x1:x2], (SIZE, SIZE)))
                    ident.append(ti); frame.append(fr); box.append((x2 - x1, y2 - y1))
            fr += 1
        cap.release()

        if not crops:
            print(f"[skip] {name}: no usable crops", flush=True)
            continue
        np.savez_compressed(
            dst,
            crops=np.stack(crops).astype(np.uint8),
            ident=np.asarray(ident, dtype=np.int32),
            frame=np.asarray(frame, dtype=np.int32),
            box=np.asarray(box, dtype=np.int32))
        print(f"[ok] {name}: {len(crops)} crops, {len(set(ident))} identities", flush=True)


if __name__ == "__main__":
    main()
