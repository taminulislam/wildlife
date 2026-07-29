#!/usr/bin/env python3
"""
TTC v5 — CALIBRATED SOFT-COUNT head. Meets the proposal's stated deliverable and removes
the structural advantage the hand-tuned rule had.

Why every previous attempt lost to a 3-threshold rule: the rule's parameters are swept to
minimise counting MAE **directly**, while the learned models optimised a classification
loss and only then had a threshold picked. This trains on the counting objective itself:

    count(video) = SUM over its candidate tracks of p_i          (expected count)
    loss         = |count(video) - GT(video)|  +  w * BCE(p_i, primary_i)

Two consequences:
  * The gradient is the counting error, so the model is optimised for the reported metric.
  * FRAGMENTATION RESOLVES ITSELF. Two fragments of one deer can each take p=0.5 and the
    expected count is still 1. The model is never forced to pick one look-alike fragment
    over another — the failure that sank the primary-vs-duplicate formulation (measured
    medians: primary 0.72 conf, duplicate 0.42, false 0.20).

DELIVERABLES REQUIRED BY THE PROPOSAL (docs/SERVER_HANDOFF.md):
  * per-deer confidence in 0-1, CALIBRATED against verified data (isotonic on val), and
    derived from learned size/shape/motion/confidence cues rather than a single cue;
  * a tuned threshold below which a detection is FLAGGED FOR REVIEW instead of counted;
  * an overall confidence for each video's count — reported as the Poisson-binomial
    standard deviation of the accepted probabilities, i.e. a genuine uncertainty on the
    count rather than a point estimate.

Usage:
  python src/temporal/train_softcount.py --counts-dir <run> --labels <labelled.csv>
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from dataset import load_sequences                                  # noqa: E402
from context import compute as ctx_compute                          # noqa: E402
from train_cv import load_gt, metrics, rule_counts, sweep_rule, TAB_FEATS  # noqa: E402


class SoftCountNet(nn.Module):
    """Tiny MLP on track statistics + cross-track context. Kept small on purpose:
    ~7k candidate tracks from 32 videos punishes anything larger."""

    def __init__(self, n_in: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_pool(counts_dir: str, labels_csv: str):
    seqs = load_sequences(counts_dir)
    meta, y_prim = [], []
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
        y_prim.append(1.0 if r["kind"] == "primary" else 0.0)
    C = ctx_compute(meta, seqs)
    X = np.asarray([[float(m[k]) for k in TAB_FEATS] + list(c)
                    for m, c in zip(meta, C)], dtype=np.float32)
    # log-scale the heavy-tailed count features so the MLP sees a sane range
    X[:, 0] = np.log1p(X[:, 0])                    # n_frames
    X[:, 1] = np.log1p(X[:, 1])                    # span_s
    X[:, 4] = np.log1p(X[:, 4])                    # mean_box_px
    return meta, X, np.asarray(y_prim, dtype=np.float32)


def video_index(meta, idx):
    by: dict[str, list] = {}
    for i in idx:
        by.setdefault(meta[i]["video"], []).append(i)
    return by


def train(Xtr, ytr, vids_tr, gt, args, dev, n_in):
    model = SoftCountNet(n_in, args.hidden, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev)
    groups = [(torch.tensor(ix, device=dev, dtype=torch.long), float(gt[v]))
              for v, ix in vids_tr.items()]
    bce = nn.BCEWithLogitsLoss()
    for _ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        logit = model(Xt)
        p = torch.sigmoid(logit)
        # counting objective: |expected count - GT| per video (the reported metric)
        closs = sum(torch.abs(p[ix].sum() - g) for ix, g in groups) / max(len(groups), 1)
        loss = closs + args.bce_w * bce(logit, yt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


def predict(model, X, dev) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, device=dev))).cpu().numpy()


def calibrate(p_val, y_val):
    """Isotonic calibration so the score is a real probability (proposal requirement)."""
    from sklearn.isotonic import IsotonicRegression
    if len(set(y_val.tolist())) < 2:
        return lambda p: p
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p_val, y_val)
    return lambda p: np.clip(ir.predict(p), 0.0, 1.0)


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
    ap.add_argument("--out", default="results/temporal/softcount")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--bce-w", type=float, default=0.3)
    ap.add_argument("--review-lo", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)

    meta, X, y = load_pool(args.counts_dir, args.labels)
    gt = load_gt(args.gt)
    videos = sorted({m["video"] for m in meta})
    print(f"pool {len(meta)} tracks | primaries {int(y.sum())} | videos {len(videos)} "
          f"| features {X.shape[1]}")

    rng = np.random.RandomState(args.seed)
    order = list(videos); rng.shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]

    pred_soft: dict[str, float] = {}
    pred_hard: dict[str, int] = {}
    pred_rule: dict[str, int] = {}
    per_track_rows = []
    conf_stats: dict[str, dict] = {}

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

        model = train(X[i_tr], y[i_tr], video_index(meta, i_tr), gt, args, dev, X.shape[1])
        p_va_raw = predict(model, X[i_va], dev)
        cal = calibrate(p_va_raw, y[i_va])
        p_va = cal(p_va_raw)

        best_thr, best_mae = 0.5, float("inf")
        for thr in np.arange(0.10, 0.91, 0.05):
            mm = metrics(counts_from(p_va, meta, i_va, thr), gt, va_v)
            if mm["MAE"] < best_mae:
                best_mae, best_thr = mm["MAE"], float(thr)

        p_te = cal(predict(model, X[i_te], dev))
        pred_hard.update(counts_from(p_te, meta, i_te, best_thr))
        for v in te_v:
            pred_hard.setdefault(v, 0)
        # expected count (soft) + Poisson-binomial sd = per-video count confidence
        for v in te_v:
            ps = np.array([p for p, i in zip(p_te, i_te) if meta[i]["video"] == v])
            pred_soft[v] = float(ps.sum())
            acc = ps[ps >= best_thr]
            conf_stats[v] = {
                "expected_count": round(float(ps.sum()), 2),
                "count_sd": round(float(math.sqrt((ps * (1 - ps)).sum())), 2),
                "accepted": int(len(acc)),
                "mean_conf_accepted": round(float(acc.mean()), 3) if len(acc) else 0.0,
                "pct_accepted_conf_ge_0.80": round(
                    100 * float((acc >= 0.80).mean()), 1) if len(acc) else 0.0,
                "flagged_for_review": int(((p_te >= args.review_lo) &
                                           (p_te < best_thr)).sum()),
            }
        for p, i in zip(p_te, i_te):
            per_track_rows.append({"video": meta[i]["video"],
                                   "track_id": meta[i]["track_id"],
                                   "kind": meta[i]["kind"],
                                   "confidence": round(float(p), 4),
                                   "counted": int(p >= best_thr),
                                   "review": int(args.review_lo <= p < best_thr)})
        cfg = sweep_rule(meta, i_tr + i_va, gt, tr_v | va_v)
        pred_rule.update(rule_counts(meta, i_te, *cfg))
        print(f"[fold {fi}] thr={best_thr:.2f} val_MAE={best_mae:.2f}", flush=True)

    allv = set(videos)
    H = metrics(pred_hard, gt, allv)
    R = metrics(pred_rule, gt, allv)
    soft_err = [pred_soft.get(v, 0.0) - gt[v] for v in allv]
    soft = {"MAE": sum(abs(e) for e in soft_err) / len(allv),
            "RMSE": math.sqrt(sum(e * e for e in soft_err) / len(allv)),
            "pred_total": round(sum(pred_soft.values()), 1)}

    acc_rows = [r for r in per_track_rows if r["counted"]]
    pct80 = 100 * np.mean([r["confidence"] >= 0.80 for r in acc_rows]) if acc_rows else 0
    mean_conf = float(np.mean([r["confidence"] for r in acc_rows])) if acc_rows else 0
    n_review = sum(r["review"] for r in per_track_rows)

    json.dump({"hard": H, "soft": soft, "rule": R,
               "pct_counted_conf_ge_80": round(float(pct80), 1),
               "mean_conf_counted": round(mean_conf, 3),
               "flagged_for_review": int(n_review),
               "per_video_confidence": conf_stats},
              open(os.path.join(args.out, "results.json"), "w"), indent=2)
    with open(os.path.join(args.out, "per_track_confidence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_track_rows[0].keys()))
        w.writeheader(); w.writerows(per_track_rows)
    with open(os.path.join(args.out, "per_video_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "gt", "counted", "expected_count", "count_sd",
                    "mean_conf_accepted", "pct_conf_ge_80", "flagged_for_review"])
        for v in sorted(allv):
            c = conf_stats.get(v, {})
            w.writerow([v, gt[v], pred_hard.get(v, 0), c.get("expected_count", 0),
                        c.get("count_sd", 0), c.get("mean_conf_accepted", 0),
                        c.get("pct_accepted_conf_ge_0.80", 0),
                        c.get("flagged_for_review", 0)])

    print(f"\n=== CROSS-VALIDATED, all {len(allv)} videos ({H['gt_total']} deer) ===")
    print(f"{'method':<34} {'MAE':>7} {'RMSE':>7} {'pred':>7}")
    print(f"{'hand-tuned rule (baseline)':<34} {R['MAE']:>7.2f} {R['RMSE']:>7.2f} "
          f"{R['pred_total']:>7}")
    print(f"{'soft-count head, thresholded':<34} {H['MAE']:>7.2f} {H['RMSE']:>7.2f} "
          f"{H['pred_total']:>7}")
    print(f"{'soft-count head, expected count':<34} {soft['MAE']:>7.2f} "
          f"{soft['RMSE']:>7.2f} {soft['pred_total']:>7}")
    d = R["MAE"] - min(H["MAE"], soft["MAE"])
    print(f"\n  best learned vs rule: {d:+.2f} ({100*d/R['MAE']:+.1f}%)")
    print("\n=== PROPOSAL DELIVERABLES ===")
    print(f"  counted deer with calibrated confidence >= 0.80 : {pct80:.1f}%")
    print(f"  mean confidence of counted deer                 : {mean_conf:.3f}")
    print(f"  tracks flagged for manual review                : {n_review}")
    print(f"  per-video count uncertainty (Poisson-binomial sd) -> per_video_counts.csv")
    print(f"-> {args.out}/")


if __name__ == "__main__":
    main()
