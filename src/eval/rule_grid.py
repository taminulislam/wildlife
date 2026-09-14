#!/usr/bin/env python3
"""
The confirmation rule's parameter space, laid out in full (co-author review, 2026-09).

The published rule accepts a candidate track when it has >= m frames, spans >= s seconds,
and the mean of its k highest per-frame confidences is >= c, with (m, s, c, k) =
(20, 0, 0.65, 5). This script answers three questions about that choice:

  GRID m x c   Is the operating point a sharp optimum or one cell of a broad valley, and
               would a permissive rule such as m = 2, c = 0.3 remove the "bottleneck"?
  GRID k x c   Would averaging over fewer, higher frames (k = 2, c = 0.85) or more, lower
               ones (k = 10 or 20, c = 0.45) tell a different story?
  REFIT k      For each k, fit (m, c) on the 19 training videos and freeze, as the paper
               does, so no cell is chosen by looking at held-out results.

Every cell reports the fit-set MAE (what a selection procedure sees) and the held-out MAE
(what it would have delivered). Held-out values are descriptive: the published rule was chosen
on the fit videos, and reading the best held-out cell off this grid would be selecting on test.

Usage:
  python src/eval/rule_grid.py --counts <pool C run> --out results/counting/rule_grid
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from count_eval import load_gt, score                              # noqa: E402
from count_eval_heldout import load_rows                           # noqa: E402

M_GRID = (1, 2, 3, 5, 8, 12, 20, 30, 40)
C_GRID = (0.0, 0.15, 0.25, 0.3, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
K_GRID = (1, 2, 3, 5, 10, 20)
PUB = dict(m=20, s=0.0, c=0.65, k=5)


def topk(confs: list[float], k: int) -> float:
    top = sorted(confs, reverse=True)[:k]
    return sum(top) / len(top) if top else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--gt", default="data/annotate_v2/count_gt.csv")
    ap.add_argument("--splits", default="data/temporal/video_splits.json")
    ap.add_argument("--out", default="results/counting/rule_grid")
    args = ap.parse_args()

    rows = load_rows(args.counts)
    gt = load_gt(args.gt)
    split = json.load(open(args.splits))
    site_of = {r["video"]: r.get("site", "?") for r in rows}
    fit = {v: g for v, g in gt.items() if split.get(v) == "train"}
    held = {v: g for v, g in gt.items() if split.get(v) in ("val", "test")}

    # per-frame confidences, so the top-k mean can be recomputed for any k
    confs: dict[tuple, list] = defaultdict(list)
    for f in glob.glob(os.path.join(args.counts, "shard*", "tracks.csv")):
        for r in csv.DictReader(open(f)):
            confs[(r["video"], int(r["track_id"]))].append(float(r["conf"]))
    tracks = []
    for r in rows:
        key = (r["video"], int(r["track_id"]))
        cs = confs.get(key)
        tracks.append({"video": r["video"], "n": int(r["n_frames"]), "span": float(r["span_s"]),
                       "tk": {k: (topk(cs, k) if cs else float(r["topk_conf"])) for k in K_GRID},
                       "tk5_stored": float(r["topk_conf"])})
    drift = max(abs(t["tk"][5] - t["tk5_stored"]) for t in tracks if (t["video"],) and True)

    def run(m, s, c, k, gtset):
        pred = defaultdict(int)
        for t in tracks:
            if t["n"] >= m and t["span"] >= s and t["tk"][k] >= c:
                pred[t["video"]] += 1
        o, _ = score(pred, gtset, site_of)
        tot = sum(gtset.values())
        cov = 100.0 * sum(min(pred.get(v, 0), g) for v, g in gtset.items()) / tot
        return {"MAE": o["MAE"], "bias": o["bias"], "pred": sum(pred.get(v, 0) for v in gtset),
                "counted": cov}

    p = run(PUB["m"], PUB["s"], PUB["c"], PUB["k"], held)
    print(f"sanity: published rule on held-out -> MAE {p['MAE']:.2f}  predicted {p['pred']}  "
          f"counted {p['counted']:.1f}%   (paper: 2.38, 58, 66.3%)")
    print(f"        top-5 recomputed from per-frame confidences vs stored: max |diff| {drift:.4f}")

    os.makedirs(args.out, exist_ok=True)
    out = []
    for m in M_GRID:
        for c in C_GRID:
            f_ = run(m, 0.0, c, 5, fit); h = run(m, 0.0, c, 5, held)
            out.append({"grid": "m_x_c", "m": m, "c": c, "k": 5, "fit_MAE": round(f_["MAE"], 3),
                        "heldout_MAE": round(h["MAE"], 3), "heldout_bias": round(h["bias"], 3),
                        "heldout_pred": h["pred"], "heldout_counted": round(h["counted"], 1)})
    for k in K_GRID:
        for c in C_GRID:
            f_ = run(20, 0.0, c, k, fit); h = run(20, 0.0, c, k, held)
            out.append({"grid": "k_x_c", "m": 20, "c": c, "k": k, "fit_MAE": round(f_["MAE"], 3),
                        "heldout_MAE": round(h["MAE"], 3), "heldout_bias": round(h["bias"], 3),
                        "heldout_pred": h["pred"], "heldout_counted": round(h["counted"], 1)})
    refit = []
    for k in K_GRID:
        best = min(((run(m, 0.0, c, k, fit)["MAE"], m, c) for m in M_GRID for c in C_GRID))
        h = run(best[1], 0.0, best[2], k, held)
        refit.append({"k": k, "fit_m": best[1], "fit_c": best[2], "fit_MAE": round(best[0], 3),
                      "heldout_MAE": round(h["MAE"], 3), "heldout_bias": round(h["bias"], 3),
                      "heldout_pred": h["pred"], "heldout_counted": round(h["counted"], 1)})

    with open(os.path.join(args.out, "grid.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
    with open(os.path.join(args.out, "refit_by_k.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(refit[0])); w.writeheader(); w.writerows(refit)
    print(f"\nheld-out: {len(held)} videos, {sum(held.values())} animals | "
          f"fit: {len(fit)} videos, {sum(fit.values())} animals")
    print(f"-> {args.out}/grid.csv, refit_by_k.csv")


if __name__ == "__main__":
    main()
