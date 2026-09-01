#!/usr/bin/env python3
"""
Reliability of the per-track posterior (reviewer question 4).

The confirmer emits a 0-1 confidence per candidate track, isotonic-calibrated on the
validation videos only. This script asks whether that number behaves like a probability
on videos the calibrator never saw: expected calibration error, maximum calibration
error, Brier score, and the reliability curve behind them.

Positive class = the track is the PRIMARY track of a real animal. Duplicates count as
negatives, which is the decision the counting stage actually makes: a duplicate is a
correct detection that must not produce a second count.

Usage:
  python src/eval/calibration_quality.py \
      --scores results/temporal/calibrated_orphan/per_track_confidence.csv \
      --splits data/temporal/video_splits.json --out results/temporal/calibration
"""
from __future__ import annotations
import argparse
import csv
import json
import os

import numpy as np


def bin_stats(p: np.ndarray, y: np.ndarray, n_bins: int, strategy: str):
    """-> list of (lo, hi, n, mean_conf, frac_pos). Empty bins are dropped."""
    if strategy == "quantile":
        # Equal-count bins. With 94% of tracks below 0.2, equal-width bins put almost
        # every track in one bin and report an ECE dominated by that single bin.
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi, last in zip(edges[:-1], edges[1:], [False] * (len(edges) - 2) + [True]):
        m = (p >= lo) & (p <= hi if last else p < hi)
        if not m.any():
            continue
        out.append((float(lo), float(hi), int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return out


def ece(bins, n_total: int) -> tuple[float, float]:
    """Expected and maximum calibration error over the bins."""
    gaps = [(n / n_total, abs(conf - acc)) for _, _, n, conf, acc in bins]
    return float(sum(w * g for w, g in gaps)), float(max(g for _, g in gaps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--splits", default="data/temporal/video_splits.json")
    ap.add_argument("--out", default="results/temporal/calibration")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    split = json.load(open(args.splits))
    rows = list(csv.DictReader(open(args.scores)))
    for r in rows:
        r["split"] = split.get(r["video"], "train")

    os.makedirs(args.out, exist_ok=True)
    report: dict[str, dict] = {}
    # 74% of candidate tracks sit at the isotonic floor, so an aggregate ECE is dominated
    # by the region where no decision is ever made. The "decision region" scope restricts
    # to tracks that could plausibly be accepted, which is where calibration has to hold.
    for scope, keep, floor in (("held-out (val+test)", {"val", "test"}, -1.0),
                               ("held-out, decision region", {"val", "test"}, 0.01),
                               ("fit (train videos)", {"train"}, -1.0),
                               ("all 32 videos", {"train", "val", "test"}, -1.0)):
        sel = [r for r in rows if r["split"] in keep and float(r["confidence"]) > floor]
        p = np.array([float(r["confidence"]) for r in sel])
        y = np.array([1.0 if r["kind"] == "primary" else 0.0 for r in sel])
        if not len(p):
            continue
        qb = bin_stats(p, y, args.bins, "quantile")
        wb = bin_stats(p, y, args.bins, "uniform")
        e_q, m_q = ece(qb, len(p))
        e_w, m_w = ece(wb, len(p))
        # Brier, plus the base rate a constant predictor would achieve, so the score is
        # readable without knowing the class balance.
        brier = float(np.mean((p - y) ** 2))
        base = float(np.mean(y))
        report[scope] = {
            "tracks": len(p), "positives": int(y.sum()), "base_rate": round(base, 4),
            "ECE_quantile": round(e_q, 4), "MCE_quantile": round(m_q, 4),
            "ECE_equalwidth": round(e_w, 4), "MCE_equalwidth": round(m_w, 4),
            "Brier": round(brier, 4),
            "Brier_baseline_constant": round(float(np.mean((base - y) ** 2)), 4),
            "distinct_isotonic_levels": int(len(np.unique(p))),
            # Isotonic saturates at 1.0; whether the top level is honest is the single
            # most consequential calibration question for a counting threshold.
            "top_level": {"conf": float(p.max()),
                          "n": int((p == p.max()).sum()),
                          "frac_positive": round(float(y[p == p.max()].mean()), 4)},
            "reliability_quantile": [
                {"lo": lo, "hi": hi, "n": n, "mean_conf": round(c, 4),
                 "frac_positive": round(a, 4)} for lo, hi, n, c, a in qb],
        }

    json.dump(report, open(os.path.join(args.out, "calibration.json"), "w"), indent=2)
    for k, v in report.items():
        print(f"\n{k}: {v['tracks']} tracks, {v['positives']} primary "
              f"(base rate {v['base_rate']:.3f})")
        print(f"  ECE {v['ECE_quantile']:.4f} (quantile) / {v['ECE_equalwidth']:.4f} "
              f"(equal-width)   MCE {v['MCE_quantile']:.4f}")
        print(f"  Brier {v['Brier']:.4f}  vs constant-predictor "
              f"{v['Brier_baseline_constant']:.4f}")
        t = v["top_level"]
        print(f"  {v['distinct_isotonic_levels']} distinct levels; top level "
              f"conf={t['conf']:.3f} n={t['n']} actual={t['frac_positive']:.3f}")
        for b in v["reliability_quantile"]:
            print(f"    [{b['lo']:.3f},{b['hi']:.3f}] n={b['n']:5d} "
                  f"conf={b['mean_conf']:.3f} actual={b['frac_positive']:.3f}")
    print(f"\n-> {args.out}/calibration.json")


if __name__ == "__main__":
    main()
