#!/usr/bin/env python3
"""
TTC v4 — DETECT-THEN-CLUSTER. The formulation that the evidence forced.

Why the previous formulation failed. We labelled a candidate `primary` (1) if it was the
best track of a deer and `duplicate` (0) if it was another fragment of that SAME deer.
But measured on the pool, duplicates sit BETWEEN primaries and false positives:

    primary    median conf 0.72, 40 frames
    duplicate  median conf 0.42,  6 frames
    false      median conf 0.20,  2 frames

So the label said "0.42-on-a-real-deer belongs to the same class as 0.20-noise, and the
opposite class from 0.72-on-a-real-deer". That is contradictory supervision along a
continuum, and it is why three different model classes (logreg, GBM, transformer) all
lost to a 3-threshold hand-tuned rule despite a ceiling of MAE 0.91.

The fix splits the problem at its natural seam:

  STAGE 1 (learned)  is this track ON a deer at all?  primary+duplicate = 1, false = 0.
                     Unambiguous, and 632 positives instead of 207.
  STAGE 2 (geometry) cluster the accepted tracks into ANIMALS: tracks that overlap in
                     time and sit within a box-scale of each other are the same deer.
                     count = number of clusters.

Fragmentation is then handled by clustering — which is what fragmentation actually is —
instead of asking a classifier to pick one of several look-alike fragments.

Usage:
  python src/temporal/train_cluster.py --counts-dir <run> --labels <labelled.csv>
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from dataset import load_sequences                       # noqa: E402
from context import compute as ctx_compute               # noqa: E402
from train_cv import (load_gt, metrics, rule_counts, sweep_rule,  # noqa: E402
                      TAB_FEATS, fit_tabular)


def load_pool(counts_dir: str, labels_csv: str):
    seqs = load_sequences(counts_dir)
    meta, y_real = [], []
    for r in csv.DictReader(open(labels_csv)):
        k = (r["video"], int(r["track_id"]))
        if k not in seqs:
            continue
        meta.append({"video": r["video"], "track_id": k[1], "kind": r["kind"],
                     "topk_conf": float(r["topk_conf"] or 0),
                     "n_frames": int(r["n_frames"] or 0),
                     "span_s": float(r["span_s"] or 0),
                     "mean_conf": float(r["mean_conf"] or 0),
                     "mean_box_px": float(r["mean_box_px"] or 0)})
        y_real.append(1.0 if r["kind"] in ("primary", "duplicate") else 0.0)
    return seqs, meta, np.array(y_real, dtype=np.float32), ctx_compute(meta, seqs)


def track_span_box(seq):
    frames = [o[0] for o in seq]
    xs = [o[2] for o in seq]; ys = [o[3] for o in seq]
    ws = [o[4] for o in seq]; hs = [o[5] for o in seq]
    return (min(frames), max(frames), float(np.mean(xs)), float(np.mean(ys)),
            float(np.mean(ws)), float(np.mean(hs)))


def cluster_count(accepted, seqs, gap_frames: int, dist_scale: float) -> dict[str, int]:
    """Union-find over accepted tracks: same animal if they overlap/adjoin in time AND
    their mean boxes are within `dist_scale` box-widths. count = #clusters per video."""
    by_video: dict[str, list] = {}
    for m in accepted:
        by_video.setdefault(m["video"], []).append(m)
    out: dict[str, int] = {}
    for v, ms in by_video.items():
        info = [track_span_box(seqs[(v, m["track_id"])]) for m in ms]
        parent = list(range(len(ms)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(ms)):
            f0i, f1i, xi, yi, wi, hi = info[i]
            for j in range(i + 1, len(ms)):
                f0j, f1j, xj, yj, wj, hj = info[j]
                # temporal: overlapping, or separated by less than `gap_frames`
                if f0j > f1i + gap_frames or f0i > f1j + gap_frames:
                    continue
                scale = max(wi, hi, wj, hj, 1.0)
                if abs(xi - xj) < dist_scale * scale and abs(yi - yj) < dist_scale * scale:
                    union(i, j)
        out[v] = len({find(i) for i in range(len(ms))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--gt", default="data/annotate_v2/count_gt.csv")
    ap.add_argument("--out", default="results/temporal/cluster")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seqs, meta, y, C = load_pool(args.counts_dir, args.labels)
    gt = load_gt(args.gt)
    videos = sorted({m["video"] for m in meta})
    print(f"pool {len(meta)} tracks | on-a-deer positives {int(y.sum())} "
          f"({100*y.mean():.1f}%) | videos {len(videos)}")

    def tab(idx):
        return np.asarray([[float(meta[i][k]) for k in TAB_FEATS] + list(C[i])
                           for i in idx], dtype=np.float32)

    rng = np.random.RandomState(args.seed)
    order = list(videos); rng.shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]

    pred_cluster: dict[str, int] = {}
    pred_rule: dict[str, int] = {}
    for fi, te_v in enumerate(folds, 1):
        te_v = set(te_v)
        if not te_v:
            continue
        rest = [v for v in videos if v not in te_v]
        n_val = max(1, len(rest) // 5)
        va_v, tr_v = set(rest[:n_val]), set(rest[n_val:])
        i_tr = [i for i, m in enumerate(meta) if m["video"] in tr_v]
        i_va = [i for i, m in enumerate(meta) if m["video"] in va_v]
        i_te = [i for i, m in enumerate(meta) if m["video"] in te_v]
        if not i_tr or not i_va or not i_te:
            continue

        clf = fit_tabular("gbm", tab(i_tr), y[i_tr])
        pva = clf.predict_proba(tab(i_va))[:, 1]
        # choose acceptance threshold AND clustering params on VAL by counting MAE
        best = (float("inf"), 0.5, 60, 2.5)
        for thr in np.arange(0.10, 0.91, 0.05):
            acc = [meta[i] for i, p in zip(i_va, pva) if p >= thr]
            for gapf in (30, 60, 120):
                for ds in (1.5, 2.5, 4.0):
                    mm = metrics(cluster_count(acc, seqs, gapf, ds), gt, va_v)
                    if mm["MAE"] < best[0]:
                        best = (mm["MAE"], float(thr), gapf, ds)
        _, thr, gapf, ds = best
        pte = clf.predict_proba(tab(i_te))[:, 1]
        acc_te = [meta[i] for i, p in zip(i_te, pte) if p >= thr]
        pred_cluster.update(cluster_count(acc_te, seqs, gapf, ds))
        for v in te_v:
            pred_cluster.setdefault(v, 0)
        cfg = sweep_rule(meta, i_tr + i_va, gt, tr_v | va_v)
        pred_rule.update(rule_counts(meta, i_te, *cfg))
        print(f"[fold {fi}] thr={thr:.2f} gap={gapf} dist={ds}  "
              f"val_MAE={best[0]:.2f}", flush=True)

    allv = set(videos)
    Cm = metrics(pred_cluster, gt, allv)
    Rm = metrics(pred_rule, gt, allv)
    json.dump({"cluster": Cm, "rule": Rm},
              open(os.path.join(args.out, "results.json"), "w"), indent=2)
    with open(os.path.join(args.out, "per_video.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video", "gt", "cluster_pred", "rule_pred"])
        for v in videos:
            w.writerow([v, gt[v], pred_cluster.get(v, 0), pred_rule.get(v, 0)])

    print(f"\n=== CROSS-VALIDATED, all {len(allv)} videos ({Cm['gt_total']} deer) ===")
    print(f"{'method':<32} {'MAE':>7} {'RMSE':>7} {'bias':>7} {'pred':>6}")
    print(f"{'hand-tuned rule (baseline)':<32} {Rm['MAE']:>7.2f} {Rm['RMSE']:>7.2f} "
          f"{Rm['bias']:>+7.2f} {Rm['pred_total']:>6}")
    print(f"{'learned accept + cluster (ours)':<32} {Cm['MAE']:>7.2f} {Cm['RMSE']:>7.2f} "
          f"{Cm['bias']:>+7.2f} {Cm['pred_total']:>6}")
    d = Rm["MAE"] - Cm["MAE"]
    print(f"\n  MAE {d:+.2f} ({100*d/Rm['MAE']:+.1f}%) vs baseline")


if __name__ == "__main__":
    main()
