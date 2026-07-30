#!/usr/bin/env python3
"""
The HONEST counting number: fit the rule on the detector's training videos, report on the
13 videos the detector never saw.

§6.7 flagged the problem — every counting figure in the log sweeps the 3-parameter rule
over all 32 videos and then reports MAE on those same 32, while the detector was trained
on 19 of them. That is optimistic twice over: the detector has seen most of the test
material, and the rule was tuned on it.

This closes both leaks in one pass:

  fit    sweep (min_hits, min_span_s, topk_conf) on the DETECTOR-TRAIN videos only
  report apply that single frozen rule to the 13 unseen (val + test) videos

The gap between the two columns is the generalisation cost, and it is the number the
paper should carry. Everything else is unchanged from count_eval.py — same grid, same
rule, same MAE definition — so the numbers are directly comparable to §6.1-§6.6.

Usage:
  python src/eval/count_eval_heldout.py --counts <run> --gt data/annotate_v2/count_gt.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from count_eval import load_counts, load_gt, predict, score        # noqa: E402


def load_rows(d: str) -> list[dict]:
    """count_eval.py wants counts_*.csv in one dir; merged runs use shard*/counts.csv."""
    try:
        return load_counts(d)
    except SystemExit:
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "shard*", "counts.csv"))) or \
                sorted(glob.glob(os.path.join(d, "merged", "counts.csv"))):
            with open(f) as fh:
                rows += list(csv.DictReader(fh))
        if not rows:
            raise
        for r in rows:
            r["n_frames"] = int(r["n_frames"])
            r["span_s"] = float(r["span_s"])
            r["topk_conf"] = float(r["topk_conf"])
        return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--gt", default="data/annotate_v2/count_gt.csv")
    ap.add_argument("--splits", default="data/temporal/video_splits.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = load_rows(args.counts)
    gt = load_gt(args.gt)
    splits = json.load(open(args.splits))
    site_of = {r["video"]: r.get("site", "?") for r in rows}

    seen = {v: g for v, g in gt.items() if splits.get(v) == "train"}
    unseen = {v: g for v, g in gt.items() if splits.get(v) in ("val", "test")}
    if not unseen:
        raise SystemExit("no val/test videos found — check --splits")

    # ---- fit on seen videos only ----
    grid = [(mh, ms, ct)
            for mh in (1, 2, 3, 5, 8, 12, 20)
            for ms in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
            for ct in (0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)]
    best, best_o = None, None
    for g in grid:
        o, _ = score(predict(rows, *g), seen, site_of)
        if best_o is None or (o["MAE"], o["RMSE"]) < (best_o["MAE"], best_o["RMSE"]):
            best, best_o = g, o

    # ---- freeze it, report on unseen ----
    pred = predict(rows, *best)
    uo, usite = score(pred, unseen, site_of)
    ao, _ = score(pred, gt, site_of)

    n_un = sum(unseen.values())
    got_un = sum(min(pred.get(v, 0), g) for v, g in unseen.items())

    print(f"rule fitted on {len(seen)} DETECTOR-TRAIN videos: "
          f"min_hits>={best[0]}, span_s>={best[1]}, topk_conf>={best[2]}\n")
    print(f"{'scope':<26} {'vids':>5} {'GT':>5} {'MAE':>7} {'RMSE':>7} {'bias':>7}")
    print(f"{'fit (detector saw these)':<26} {best_o['videos']:>5} {sum(seen.values()):>5} "
          f"{best_o['MAE']:>7.2f} {best_o['RMSE']:>7.2f} {best_o['bias']:>+7.2f}")
    print(f"{'HELD OUT (never seen)':<26} {uo['videos']:>5} {n_un:>5} "
          f"{uo['MAE']:>7.2f} {uo['RMSE']:>7.2f} {uo['bias']:>+7.2f}   <- PUBLISH THIS")
    print(f"{'all 32 (optimistic)':<26} {ao['videos']:>5} {sum(gt.values()):>5} "
          f"{ao['MAE']:>7.2f} {ao['RMSE']:>7.2f} {ao['bias']:>+7.2f}")
    print(f"\nheld-out coverage: {got_un}/{n_un} = {100*got_un/max(n_un,1):.1f}% "
          f"of unseen deer counted (capped per video, so over-counts cannot inflate it)")
    print(f"\nper site, held out:")
    for s in sorted(usite):
        d = usite[s]
        print(f"  {s:<5} {d['videos']:>2} vids  MAE {d['MAE']:.2f}  bias {d['bias']:+.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "split", "site", "gt", "predicted", "error"])
            for v, g in sorted(gt.items()):
                w.writerow([v, splits.get(v, "?"), site_of.get(v, "?"), g,
                            pred.get(v, 0), pred.get(v, 0) - g])
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
