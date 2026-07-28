#!/usr/bin/env python3
"""
Dataset for the temporal counting head.

One sample = one CANDIDATE TRACK: the sequence of its per-frame detections plus the
label from label_tracks.py (1 = primary track of a real deer, 0 = duplicate fragment
or false positive).

Per-timestep features (10) — deliberately geometry/confidence only, no appearance, so
the head stays tiny and cannot memorise 32 videos:
    conf, xc, yc, w, h, sqrt(area), dx, dy, d(sqrt area), dt
positions/sizes normalised by frame size; dx/dy/dt are deltas from the previous
observation, which is what exposes flicker (large dt) and ego-motion drift.

Splitting is BY VIDEO, reusing the same train/val/test assignment as the detector split
(data/dataset/yolo_v3) so a video never appears in two splits and the counting numbers
are comparable with the detection numbers.
"""
from __future__ import annotations
import csv
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "dataset"))
sys.path.insert(0, os.path.join(_HERE, "..", "mining"))

FRAME_W, FRAME_H = 640.0, 512.0
N_FEAT = 10


def video_splits(dataset_root: str = "data/dataset/yolo_v3",
                 cvat_dir: str = "data/cvat_export",
                 source: str = "data/raw",
                 cache: str = "data/temporal/video_splits.json") -> dict[str, str]:
    """CVAT video name -> train/val/test, matching the detector split."""
    if os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)
    from cvat_to_yolo import find_video
    from filename_meta import parse_path
    key_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        d = os.path.join(dataset_root, "labels", split)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            key_split[fn.rsplit("_f", 1)[0]] = split
    out: dict[str, str] = {}
    for x in sorted(glob.glob(os.path.join(cvat_dir, "*.xml"))):
        name = os.path.splitext(os.path.basename(x))[0].replace("_annotations", "")
        vp = find_video(source, name)
        if vp is None:
            continue
        out[name] = key_split.get(parse_path(vp).key, "train")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump(out, f, indent=1)
    return out


def load_sequences(counts_dir: str) -> dict[tuple, list]:
    """(video, track_id) -> [(frame, conf, xc, yc, w, h), ...] sorted by frame."""
    seqs: dict[tuple, list] = {}
    for f in sorted(glob.glob(os.path.join(counts_dir, "shard*", "tracks.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                k = (r["video"], int(r["track_id"]))
                seqs.setdefault(k, []).append((
                    int(r["frame"]), float(r["conf"]), float(r["xc"]),
                    float(r["yc"]), float(r["w"]), float(r["h"])))
    for k in seqs:
        seqs[k].sort()
    return seqs


def featurise(seq: list, max_len: int = 64) -> np.ndarray:
    """-> (max_len, N_FEAT) float32, evenly subsampled, zero-padded at the END."""
    if len(seq) > max_len:                      # keep shape, preserve span
        idx = np.linspace(0, len(seq) - 1, max_len).astype(int)
        seq = [seq[i] for i in idx]
    out = np.zeros((max_len, N_FEAT), dtype=np.float32)
    prev = None
    for t, (fr, conf, xc, yc, w, h) in enumerate(seq):
        area = (w * h) ** 0.5
        f = [conf, xc / FRAME_W, yc / FRAME_H, w / FRAME_W, h / FRAME_H, area / 100.0,
             0.0, 0.0, 0.0, 0.0]
        if prev is not None:
            pfr, _, pxc, pyc, pw, ph = prev
            f[6] = (xc - pxc) / FRAME_W
            f[7] = (yc - pyc) / FRAME_H
            f[8] = ((w * h) ** 0.5 - (pw * ph) ** 0.5) / 100.0
            f[9] = min((fr - pfr) / 60.0, 2.0)      # seconds, clipped
        out[t] = f
        prev = (fr, conf, xc, yc, w, h)
    return out


def build(counts_dir: str, labels_csv: str, max_len: int = 64,
          splits: dict | None = None):
    """-> dict split -> (X[N,max_len,F], mask[N,max_len], y[N], meta[N])"""
    splits = splits or video_splits()
    seqs = load_sequences(counts_dir)
    data = {s: {"X": [], "M": [], "y": [], "meta": []} for s in ("train", "val", "test")}
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            k = (r["video"], int(r["track_id"]))
            seq = seqs.get(k)
            if not seq:
                continue
            sp = splits.get(r["video"], "train")
            n = min(len(seq), max_len)
            m = np.zeros(max_len, dtype=np.float32); m[:n] = 1.0
            data[sp]["X"].append(featurise(seq, max_len))
            data[sp]["M"].append(m)
            data[sp]["y"].append(float(r["label"]))
            data[sp]["meta"].append({"video": r["video"], "track_id": k[1],
                                     "kind": r["kind"],
                                     "topk_conf": float(r["topk_conf"] or 0),
                                     "n_frames": int(r["n_frames"] or len(seq)),
                                     "span_s": float(r["span_s"] or 0)})
    out = {}
    for s, d in data.items():
        if not d["X"]:
            out[s] = None; continue
        out[s] = (np.stack(d["X"]), np.stack(d["M"]),
                  np.array(d["y"], dtype=np.float32), d["meta"])
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--labels", default="data/temporal/tracks_labelled.csv")
    a = ap.parse_args()
    d = build(a.counts_dir, a.labels)
    for s, v in d.items():
        if v is None:
            print(f"{s}: EMPTY"); continue
        X, M, y, meta = v
        vids = len({m['video'] for m in meta})
        print(f"{s:<6} tracks={len(y):>4}  positives={int(y.sum()):>3} "
              f"({100*y.mean():.0f}%)  videos={vids}  X={X.shape}")
