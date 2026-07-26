#!/usr/bin/env python3
"""
THE PAPER'S HEADLINE METRIC: counting accuracy against the 236-deer ground truth.

Detection metrics (mAP, P/R) are supporting material. What this project claims is
*counting unique individuals*, so the number that matters is MAE / RMSE between the
predicted count of confirmed tracks and the CVAT track count, per video and per site.

Also produces the BASELINE THE CVPR CONTRIBUTION MUST BEAT: the best hand-tuned
confirmation rule. `count_deer.py` dumps every candidate track with its statistics
(n_frames, span_s, top-k confidence), so an arbitrary rule

    confirmed = n_frames >= min_hits AND span_s >= min_span_s AND topk_conf >= conf_track

can be re-applied post-hoc with NO GPU re-run. We sweep that 3-D grid, report the best
achievable hand-tuned result, and that becomes the number the learned temporal head has
to beat. Sweeping on the same data we report is optimistic ON PURPOSE — it makes the
baseline as strong as possible, so beating it is meaningful.

Outputs:
  count_eval_best.csv    per-video predicted vs GT for the best rule
  count_eval_sweep.csv   every rule tried, sorted by MAE
  count_eval_summary.md  paper-ready: overall + per-site MAE/RMSE/bias

Usage:
  python src/eval/count_eval.py --counts results/counts/run --gt data/annotate_v2/count_gt.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import math
import os
from collections import defaultdict


def load_counts(d: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "counts_*.csv"))):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no counts_*.csv under {d}")
    for r in rows:
        r["n_frames"] = int(r["n_frames"])
        r["span_s"] = float(r["span_s"])
        r["topk_conf"] = float(r["topk_conf"])
    return rows


def load_gt(path: str) -> dict[str, int]:
    gt = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            gt[r["video"]] = int(r["unique_deer"])
    return gt


def predict(rows, min_hits, min_span, conf_track) -> dict[str, int]:
    out = defaultdict(int)
    for r in rows:
        if (r["n_frames"] >= min_hits and r["span_s"] >= min_span
                and r["topk_conf"] >= conf_track):
            out[r["video"]] += 1
    return out


def score(pred: dict[str, int], gt: dict[str, int], site_of: dict[str, str]):
    """-> overall dict + per-site dict. MAE/RMSE/bias over videos."""
    errs, per_site = [], defaultdict(list)
    for v, g in gt.items():
        e = pred.get(v, 0) - g          # +over-count, -under-count
        errs.append(e)
        per_site[site_of.get(v, "?")].append(e)

    def agg(es):
        n = len(es)
        return {
            "videos": n,
            "MAE": sum(abs(e) for e in es) / n,
            "RMSE": math.sqrt(sum(e * e for e in es) / n),
            "bias": sum(es) / n,                       # signed: >0 over, <0 under
            "over": sum(e for e in es if e > 0),
            "under": -sum(e for e in es if e < 0),
        }

    return agg(errs), {s: agg(es) for s, es in per_site.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="results/counts/run")
    ap.add_argument("--gt", default="data/annotate_v2/count_gt.csv")
    ap.add_argument("--out", default=None, help="default: <counts>/eval")
    args = ap.parse_args()
    out = args.out or os.path.join(args.counts, "eval")
    os.makedirs(out, exist_ok=True)

    rows = load_counts(args.counts)
    gt = load_gt(args.gt)
    site_of = {r["video"]: r["site"] for r in rows}
    for v in gt:                                  # videos with 0 predicted tracks
        site_of.setdefault(v, v.split("_")[1] if "_" in v else "?")

    missing = set(gt) - {r["video"] for r in rows}
    if missing:
        print(f"[warn] {len(missing)} GT videos have no candidate tracks at all: "
              f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    # ---- sweep the hand-tuned confirmation rule ----
    sweep = []
    for mh in (1, 2, 3, 5, 8, 12, 20, 30):
        for ms in (0.0, 0.1, 0.3, 0.5, 1.0, 2.0):
            for ct in (0.15, 0.25, 0.35, 0.50, 0.65, 0.80):
                o, _ = score(predict(rows, mh, ms, ct), gt, site_of)
                sweep.append({"min_hits": mh, "min_span_s": ms, "conf_track": ct, **o})
    sweep.sort(key=lambda r: (r["MAE"], r["RMSE"]))
    with open(os.path.join(out, "count_eval_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader(); w.writerows(sweep)

    best = sweep[0]
    pred = predict(rows, best["min_hits"], best["min_span_s"], best["conf_track"])
    overall, per_site = score(pred, gt, site_of)

    with open(os.path.join(out, "count_eval_best.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "site", "gt_deer", "predicted", "error"])
        for v in sorted(gt):
            w.writerow([v, site_of.get(v, "?"), gt[v], pred.get(v, 0),
                        pred.get(v, 0) - gt[v]])

    total_gt = sum(gt.values())
    total_pred = sum(pred.get(v, 0) for v in gt)
    lines = [
        "# Counting results — tracking-by-detection with a hand-tuned rule",
        "",
        "**This is the baseline the learned temporal head must beat.**",
        "",
        f"Best hand-tuned rule (swept on this data, i.e. optimistically favourable to "
        f"the baseline): `min_hits >= {best['min_hits']}`, "
        f"`span_s >= {best['min_span_s']}`, `topk_conf >= {best['conf_track']}`",
        "",
        "| Scope | Videos | MAE | RMSE | Bias (+over/-under) | Total over | Total under |",
        "|---|---|---|---|---|---|---|",
        f"| **ALL** | {overall['videos']} | **{overall['MAE']:.2f}** | "
        f"{overall['RMSE']:.2f} | {overall['bias']:+.2f} | {overall['over']:.0f} | "
        f"{overall['under']:.0f} |",
    ]
    for s in sorted(per_site):
        a = per_site[s]
        lines.append(f"| {s} | {a['videos']} | {a['MAE']:.2f} | {a['RMSE']:.2f} | "
                     f"{a['bias']:+.2f} | {a['over']:.0f} | {a['under']:.0f} |")
    lines += ["",
              f"Total GT deer: **{total_gt}** | total predicted: **{total_pred}** "
              f"({100*total_pred/total_gt:.1f}% of truth)", ""]
    md = "\n".join(lines)
    with open(os.path.join(out, "count_eval_summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n-> {out}/count_eval_{{best.csv,sweep.csv,summary.md}}")


if __name__ == "__main__":
    main()
