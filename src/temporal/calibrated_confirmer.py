#!/usr/bin/env python3
"""
Capacity-matched, CALIBRATED confirmer — the deliverable the proposal actually specifies,
plus the last principled hypothesis about why learned confirmers kept losing.

Scoreboard so far (8-fold CV over videos, identical protocol, orphan pool):
    hand-tuned rule (3 params)            MAE 1.88
    GBM depth-3, 200 iters                    2.41
    TTC transformer (~60k params)             2.81
    logistic regression                       3.41
    learned-accept + clustering               3.66
    soft-count MLP                            4.44

The consistent pattern is capacity, not architecture: the rule has THREE parameters fitted
directly on 32 videos, while every learned competitor had hundreds to tens of thousands.
With ~200 positives that is a variance problem no amount of regularisation inside a big
model fixes. So this matches capacity deliberately — depth-1 stumps, few iterations, a
handful of features — which is the fair comparison the paper should report.

Independently of who wins on MAE, this produces what docs/SERVER_HANDOFF.md requires and
the project had never delivered:
  * a 0-1 per-deer confidence CALIBRATED on verified data (isotonic, fitted on val only),
  * a tuned accept threshold, with a band below it FLAGGED FOR REVIEW rather than counted,
  * a per-video count uncertainty (Poisson-binomial sd over accepted probabilities).

Note on the ">=0.80 confidence" requirement: raw detector scores cannot satisfy it (no
track in the corpus exceeds 0.90 on 27 px deer). It is only meaningful as a calibrated
posterior, which is what `confidence` means here.
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
from dataset import load_sequences                                  # noqa: E402
from context import compute as ctx_compute                          # noqa: E402
from train_cv import load_gt, metrics, rule_counts, sweep_rule, TAB_FEATS  # noqa: E402


POS_KINDS = ("primary",)     # set by --target


def load_pool(counts_dir, labels_csv):
    seqs = load_sequences(counts_dir)
    meta, y = [], []
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
        y.append(1.0 if r["kind"] in POS_KINDS else 0.0)
    C = ctx_compute(meta, seqs)
    X = np.asarray([[float(m[k]) for k in TAB_FEATS] + list(c)
                    for m, c in zip(meta, C)], dtype=np.float32)
    return meta, X, np.asarray(y, dtype=np.float32)


def fit(kind, Xtr, ytr):
    from sklearn.ensemble import HistGradientBoostingClassifier
    if kind == "stump":            # capacity-matched to the 3-parameter rule
        return HistGradientBoostingClassifier(
            max_depth=1, max_iter=40, learning_rate=0.10, l2_regularization=5.0,
            min_samples_leaf=40, random_state=0).fit(Xtr, ytr)
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.06, l2_regularization=1.0,
        random_state=0).fit(Xtr, ytr)


def counts_from(p, meta, idx, thr):
    out: dict[str, int] = {}
    for pi, i in zip(p, idx):
        v = meta[i]["video"]
        out[v] = out.get(v, 0) + int(pi >= thr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--gt", default="data/annotate_v2/count_gt.csv")
    ap.add_argument("--out", default="results/temporal/calibrated")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--review-lo", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="primary", choices=["primary", "ondeer"],
                    help="'primary' = P(this is the canonical track of a deer), used for "
                         "COUNTING. 'ondeer' = P(this track is on a real deer) — the "
                         "proposal's 'is-it-an-animal' score, and the right quantity for "
                         "the per-deer CONFIDENCE deliverable.")
    args = ap.parse_args()
    global POS_KINDS
    POS_KINDS = ("primary",) if args.target == "primary" else ("primary", "duplicate")
    os.makedirs(args.out, exist_ok=True)
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    meta, X, y = load_pool(args.counts_dir, args.labels)
    gt = load_gt(args.gt)
    videos = sorted({m["video"] for m in meta})
    print(f"pool {len(meta)} tracks | primaries {int(y.sum())} | videos {len(videos)}")

    rng = np.random.RandomState(args.seed)
    order = list(videos); rng.shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]

    res = {k: {} for k in ("stump", "deep")}
    rule_pred: dict[str, int] = {}
    rows, aucs, aps = [], [], []
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
        for kind in ("stump", "deep"):
            clf = fit(kind, X[i_tr], y[i_tr])
            pv = clf.predict_proba(X[i_va])[:, 1]
            cal = None
            if len(set(y[i_va].tolist())) > 1:
                cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                cal.fit(pv, y[i_va])
                pv = np.clip(cal.predict(pv), 0, 1)
            thr, best = 0.5, float("inf")
            for t in np.arange(0.10, 0.91, 0.05):
                mm = metrics(counts_from(pv, meta, i_va, t), gt, va_v)
                if mm["MAE"] < best:
                    best, thr = mm["MAE"], float(t)
            pt = clf.predict_proba(X[i_te])[:, 1]
            if cal is not None:
                pt = np.clip(cal.predict(pt), 0, 1)
            res[kind].update(counts_from(pt, meta, i_te, thr))
            for v in te_v:
                res[kind].setdefault(v, 0)
            if kind == "stump":
                if len(set(y[i_te].tolist())) > 1:
                    aucs.append(roc_auc_score(y[i_te], pt))
                    aps.append(average_precision_score(y[i_te], pt))
                for p, i in zip(pt, i_te):
                    rows.append({"video": meta[i]["video"],
                                 "track_id": meta[i]["track_id"],
                                 "kind": meta[i]["kind"],
                                 "confidence": round(float(p), 4),
                                 "counted": int(p >= thr),
                                 "review": int(args.review_lo <= p < thr)})
        cfg = sweep_rule(meta, i_tr + i_va, gt, tr_v | va_v)
        rule_pred.update(rule_counts(meta, i_te, *cfg))

    allv = set(videos)
    R = metrics(rule_pred, gt, allv)
    S = metrics(res["stump"], gt, allv)
    D = metrics(res["deep"], gt, allv)
    acc = [r for r in rows if r["counted"]]
    pct80 = 100 * float(np.mean([r["confidence"] >= 0.80 for r in acc])) if acc else 0.0
    meanc = float(np.mean([r["confidence"] for r in acc])) if acc else 0.0

    # per-video count uncertainty from the accepted probabilities
    pv_rows = []
    for v in sorted(allv):
        ps = np.array([r["confidence"] for r in rows if r["video"] == v and r["counted"]])
        pv_rows.append([v, gt[v], res["stump"].get(v, 0),
                        round(float(ps.sum()), 2) if len(ps) else 0.0,
                        round(float(math.sqrt((ps * (1 - ps)).sum())), 2) if len(ps) else 0.0,
                        round(float(ps.mean()), 3) if len(ps) else 0.0,
                        round(100 * float((ps >= 0.80).mean()), 1) if len(ps) else 0.0])
    with open(os.path.join(args.out, "per_video_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "gt", "counted", "expected_count", "count_sd",
                    "mean_confidence", "pct_conf_ge_80"])
        w.writerows(pv_rows)
    with open(os.path.join(args.out, "per_track_confidence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump({"rule": R, "stump": S, "deep": D,
               "track_AUC": round(float(np.mean(aucs)), 3) if aucs else None,
               "track_AP": round(float(np.mean(aps)), 3) if aps else None,
               "pct_counted_conf_ge_80": round(pct80, 1),
               "mean_conf_counted": round(meanc, 3),
               "flagged_for_review": int(sum(r["review"] for r in rows))},
              open(os.path.join(args.out, "results.json"), "w"), indent=2)

    print(f"\n=== CROSS-VALIDATED, all {len(allv)} videos ({R['gt_total']} deer) ===")
    print(f"{'method':<36} {'MAE':>7} {'RMSE':>7} {'pred':>6}")
    for nm, m in (("hand-tuned rule (3 params)", R),
                  ("capacity-matched stumps (ours)", S),
                  ("depth-3 GBM (over-parameterised)", D)):
        print(f"{nm:<36} {m['MAE']:>7.2f} {m['RMSE']:>7.2f} {m['pred_total']:>6}")
    print(f"\n=== CONFIDENCE (proposal deliverable) ===")
    print(f"  track-level AUC {np.mean(aucs):.3f} | AP {np.mean(aps):.3f}"
          if aucs else "  (AUC unavailable)")
    print(f"  counted deer with calibrated confidence >= 0.80 : {pct80:.1f}%")
    print(f"  mean confidence of counted deer                 : {meanc:.3f}")
    print(f"  flagged for manual review                       : "
          f"{sum(r['review'] for r in rows)}")
    print(f"-> {args.out}/")


if __name__ == "__main__":
    main()
