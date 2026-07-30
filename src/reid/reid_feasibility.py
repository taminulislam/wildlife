#!/usr/bin/env python3
"""
Can a thermal ReID model tell individual deer apart on THIS corpus? Measure it first.

The request is a ReID stage keyed on body shape, antler structure, thermal silhouette,
relative size and movement. Two of those (relative size, movement) are already features of
the temporal head; the question is whether APPEARANCE adds identity signal at 27 px.

Rather than argue from resolution, this measures the ceiling directly, using the CVAT
tracks as identity labels — one CVAT track is one animal, so it is exactly a ReID
identity annotation.

PROTOCOL (standard closed-set ReID, within video)
  * take every GT deer track with >= 2*--per-half usable boxes
  * split each track TEMPORALLY in half — the gallery is the first half, the query the
    second. A temporal split is the honest one: adjacent frames are near-duplicates, so a
    random split would score high without any identity information at all.
  * embed each crop, average per half, L2-normalise
  * rank-1 = fraction of queries whose nearest gallery entry is the same animal
  * chance = 1 / n_identities in that video

Within-video is the CEILING, not the deployment number: the two halves share lighting,
range, pose and the same thermal calibration. A model that cannot separate individuals
HERE cannot separate them across videos either, so a low score is decisive while a high
score is only permissive.

Three embeddings are compared so the result attributes cause:
  geom       5 numbers — w, h, aspect, sqrt(area), mean intensity. The cues the pipeline
             ALREADY uses. This is the baseline appearance must beat to be worth adding.
  pixels     the raw 32x32 thermal crop, flattened. Silhouette + intensity, no learning.
  cnn        ImageNet ResNet-50 penultimate features (2048-d), the usual ReID backbone.

If `cnn` does not beat `geom`, a learned thermal ReID has nothing to learn from at this
scale and the honest move is to say so with a number.

Usage (GPU node):
  python src/reid/reid_feasibility.py --source data/raw --cvat-dir data/cvat_export \
      --out results/reid/feasibility.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "temporal"))
sys.path.insert(0, os.path.join(_HERE, "..", "track"))
from label_tracks import gt_tracks_of                                  # noqa: E402
from count_deer import list_videos                                     # noqa: E402

CROP = 32


def find_video(source: str, name: str) -> str:
    for v in list_videos(source):
        if os.path.splitext(os.path.basename(v))[0] == name:
            return v
    return ""


def collect_crops(path: str, tracks: list[dict], per_half: int) -> dict[int, list]:
    """Sequential decode (frame seeking is unreliable on these files) -> {tid: [crop,...]}."""
    need: dict[int, list] = {}
    for ti, g in enumerate(tracks):
        for fr in g:
            need.setdefault(fr, []).append((ti, g[fr]))
    out: dict[int, list] = {ti: [] for ti in range(len(tracks))}
    cap = cv2.VideoCapture(path)
    fr = 0
    last = max(need) if need else -1
    while fr <= last:
        ok, img = cap.read()
        if not ok:
            break
        for (ti, box) in need.get(fr, []):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            c = img[y1:y2, x1:x2]
            if c.ndim == 3:
                c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            out[ti].append((fr, x2 - x1, y2 - y1, cv2.resize(c, (CROP, CROP))))
        fr += 1
    cap.release()
    return out


def embed_geom(items) -> np.ndarray:
    v = []
    for (_fr, w, h, c) in items:
        v.append([w, h, w / max(h, 1), np.sqrt(w * h), float(c.mean())])
    return np.asarray(v, dtype=np.float32)


def embed_pixels(items) -> np.ndarray:
    return np.stack([c.astype(np.float32).ravel() / 255.0 for (_f, _w, _h, c) in items])


def embed_cnn(items, model, device) -> np.ndarray:
    import torch
    x = np.stack([c for (_f, _w, _h, c) in items]).astype(np.float32) / 255.0
    t = torch.from_numpy(x).unsqueeze(1).repeat(1, 3, 1, 1)            # grey -> 3-channel
    t = (t - 0.449) / 0.226
    with torch.no_grad():
        f = model(t.to(device)).squeeze(-1).squeeze(-1)
    return f.cpu().numpy()


def rank1(gal: np.ndarray, qry: np.ndarray) -> float:
    """gal/qry: [n_ident, dim], row i = identity i. -> fraction matched to themselves."""
    g = gal / (np.linalg.norm(gal, axis=1, keepdims=True) + 1e-9)
    q = qry / (np.linalg.norm(qry, axis=1, keepdims=True) + 1e-9)
    sim = q @ g.T
    return float((sim.argmax(axis=1) == np.arange(len(q))).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--out", default="results/reid/feasibility.csv")
    ap.add_argument("--per-half", type=int, default=4,
                    help="min crops each half of a track must contribute")
    ap.add_argument("--min-ident", type=int, default=3,
                    help="skip videos with fewer identities than this (rank-1 meaningless)")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    import torch
    import torchvision
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    net.fc = torch.nn.Identity()
    net = net.eval().to(device)

    xmls = sorted(glob.glob(os.path.join(args.cvat_dir, "*.xml")))
    xmls = [x for i, x in enumerate(xmls) if i % args.shards == args.shard]

    rows = []
    for x in xmls:
        name = os.path.splitext(os.path.basename(x))[0].replace("_annotations", "")
        vp = find_video(args.source, name)
        if not vp:
            print(f"[skip] no video for {name}", flush=True)
            continue
        tracks = gt_tracks_of(x)
        crops = collect_crops(vp, tracks, args.per_half)
        keep = [ti for ti, c in crops.items() if len(c) >= 2 * args.per_half]
        if len(keep) < args.min_ident:
            print(f"[skip] {name}: {len(keep)} usable identities", flush=True)
            continue

        res = {}
        for tag in ("geom", "pixels", "cnn"):
            gal, qry = [], []
            for ti in keep:
                items = sorted(crops[ti])
                half = len(items) // 2
                a, b = items[:half], items[half:]
                if tag == "geom":
                    ea, eb = embed_geom(a), embed_geom(b)
                elif tag == "pixels":
                    ea, eb = embed_pixels(a), embed_pixels(b)
                else:
                    ea, eb = embed_cnn(a, net, device), embed_cnn(b, net, device)
                gal.append(ea.mean(axis=0)); qry.append(eb.mean(axis=0))
            res[tag] = rank1(np.stack(gal), np.stack(qry))

        chance = 1.0 / len(keep)
        rows.append({"video": name, "identities": len(keep), "chance": round(chance, 4),
                     **{k: round(v, 4) for k, v in res.items()}})
        print(f"{name:<40} n={len(keep):>3}  chance={chance:.3f}  "
              f"geom={res['geom']:.3f}  pixels={res['pixels']:.3f}  cnn={res['cnn']:.3f}",
              flush=True)

    if not rows:
        raise SystemExit("no video produced enough identities")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # identity-weighted: a 27-deer video should count for more than a 3-deer one
    tot = sum(r["identities"] for r in rows)
    print(f"\n{'':<22}{'rank-1':>8}   (identity-weighted over {tot} deer, "
          f"{len(rows)} videos)")
    for tag in ("chance", "geom", "pixels", "cnn"):
        wavg = sum(r[tag] * r["identities"] for r in rows) / tot
        print(f"  {tag:<20}{wavg:>8.3f}")
    print(f"\n-> {args.out}")
    print("READ: cnn must beat geom for a learned appearance ReID to add anything, and "
          "both must beat chance by a wide margin for ReID to be viable at all. This is "
          "the WITHIN-video ceiling; cross-video would be lower.")


if __name__ == "__main__":
    main()
