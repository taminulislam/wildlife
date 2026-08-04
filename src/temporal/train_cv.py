#!/usr/bin/env python3
"""
TTC v2 — cross-validated training of the temporal confirmation head.

Two changes from v1, both aimed at getting a TRUSTWORTHY number:

1. **Cross-track context** (context.py) is fed to the model. v1 judged each track alone,
   but "duplicate" means *another track already covers this deer* — invisible from a
   single track, so v1 accepted both copies and over-counted (+1.11 bias).

2. **K-fold CV over videos instead of one split.** v1's test set was 9 videos / 38 deer,
   where MAE +-2 is pure noise. Here every video is held out exactly once, so the final
   MAE is measured on ALL 32 videos / 236 deer. Within each fold an inner split picks the
   decision threshold, so test folds are never used for fitting.

The hand-tuned rule is evaluated under the IDENTICAL protocol: swept on each fold's
training videos, applied to that fold's held-out videos. Same folds, same metric, so the
comparison is apples-to-apples.

Usage:
  python src/temporal/train_cv.py --counts-dir <counts run> --folds 8 --out results/temporal/ttc_cv
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
from dataset import build, featurise, load_sequences, N_FEAT  # noqa: E402
from context import compute as ctx_compute, N_CTX             # noqa: E402
from model import TemporalTrackNet, count_params              # noqa: E402


def load_all(counts_dir: str, labels_csv: str, max_len: int):
    """All tracks in one flat set (no split) + per-track context."""
    seqs = load_sequences(counts_dir)
    X, M, y, meta = [], [], [], []
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            k = (r["video"], int(r["track_id"]))
            seq = seqs.get(k)
            if not seq:
                continue
            n = min(len(seq), max_len)
            m = np.zeros(max_len, dtype=np.float32); m[:n] = 1.0
            X.append(featurise(seq, max_len)); M.append(m)
            y.append(float(r["label"]))
            meta.append({"video": r["video"], "track_id": k[1], "kind": r["kind"],
                         "topk_conf": float(r["topk_conf"] or 0),
                         "n_frames": int(r["n_frames"] or len(seq)),
                         "span_s": float(r["span_s"] or 0),
                         "mean_conf": float(r["mean_conf"] or 0),
                         "mean_box_px": float(r["mean_box_px"] or 0)})
    return (np.stack(X), np.stack(M), np.array(y, dtype=np.float32), meta,
            ctx_compute(meta, seqs))


def load_gt(path: str) -> dict[str, int]:
    return {r["video"]: int(r["unique_deer"]) for r in csv.DictReader(open(path))}


def metrics(pred: dict[str, int], gt: dict[str, int], videos) -> dict:
    errs = [pred.get(v, 0) - gt[v] for v in videos]
    n = len(errs) or 1
    return {"videos": len(errs),
            "MAE": sum(abs(e) for e in errs) / n,
            "RMSE": math.sqrt(sum(e * e for e in errs) / n),
            "bias": sum(errs) / n,
            "over": sum(e for e in errs if e > 0),
            "under": -sum(e for e in errs if e < 0),
            "pred_total": sum(pred.get(v, 0) for v in videos),
            "gt_total": sum(gt[v] for v in videos)}


def rule_counts(meta, idx, mh, ms, ct) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in idx:
        m = meta[i]
        out[m["video"]] = out.get(m["video"], 0) + int(
            m["n_frames"] >= mh and m["span_s"] >= ms and m["topk_conf"] >= ct)
    return out


def sweep_rule(meta, idx, gt, videos):
    best, cfg = None, None
    for mh in (1, 2, 3, 5, 8, 12, 20, 30):
        for ms in (0.0, 0.1, 0.3, 0.5, 1.0, 2.0):
            for ct in (0.15, 0.25, 0.35, 0.50, 0.65, 0.80):
                m = metrics(rule_counts(meta, idx, mh, ms, ct), gt, videos)
                if best is None or (m["MAE"], m["RMSE"]) < (best["MAE"], best["RMSE"]):
                    best, cfg = m, (mh, ms, ct)
    return cfg


TAB_FEATS = ["n_frames", "span_s", "topk_conf", "mean_conf", "mean_box_px"]


def tabular(meta, C, idx) -> np.ndarray:
    """Track summary stats + cross-track context, for the tabular learners.

    With only ~700 tracks a gradient-boosted tree is a far better-matched model class
    than a 60k-parameter transformer; including it keeps the paper's claim ("a LEARNED
    confirmer beats the hand-tuned rule") from depending on one architecture."""
    rows = []
    for i in idx:
        m = meta[i]
        rows.append([float(m[k]) for k in TAB_FEATS] + list(C[i]))
    return np.asarray(rows, dtype=np.float32)


def fit_tabular(kind, Xtr, ytr):
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                             learning_rate=0.06,
                                             l2_regularization=1.0, random_state=0)
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return clf


def head_counts(probs, meta, idx, thr) -> dict[str, int]:
    out: dict[str, int] = {}
    for p, i in zip(probs, idx):
        v = meta[i]["video"]
        out[v] = out.get(v, 0) + int(p >= thr)
    return out


def train_fold(Xtr, Mtr, Ctr, ytr, Xva, Mva, Cva, yva, args, dev):
    """Trains with VAL-BASED EARLY STOPPING. v2 ran all 150 epochs with no model
    selection and overfit (~500 tracks, 60k params) -> MAE got WORSE than the rule.
    We keep the epoch with the best val AP, mirroring v1's best-epoch behaviour."""
    model = TemporalTrackNet(n_feat=N_FEAT, d_model=args.d_model, layers=args.layers,
                             dropout=args.dropout, n_ctx=N_CTX).to(dev)
    t = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    Xtr_, Mtr_, Ctr_, ytr_ = t(Xtr), t(Mtr), t(Ctr), t(ytr)
    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    from sklearn.metrics import average_precision_score
    n = len(ytr_)
    best = {"ap": -1.0, "state": None}
    for _ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.batch):
            j = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(model(Xtr_[j], Mtr_[j], Ctr_[j]), ytr_[j])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(t(Xva), t(Mva), t(Cva))).cpu().numpy()
        if len(set(yva.tolist())) > 1:
            ap = float(average_precision_score(yva, pv))
            if ap > best["ap"]:
                best = {"ap": ap, "state": {k: v.detach().cpu().clone()
                                            for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(t(Xva), t(Mva), t(Cva))).cpu().numpy()
    return model, pva


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--labels", default="data/temporal/tracks_labelled.csv")
    ap.add_argument("--gt", default="data/annotate_v2/count_gt.csv")
    ap.add_argument("--out", default="results/temporal/ttc_cv")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", action="store_true",
                    help="single fold = the 13 videos the DETECTOR never saw, instead of "
                         "k-fold over all 32. Comparable to count_eval_heldout.py.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)

    X, M, y, meta, C = load_all(args.counts_dir, args.labels, args.max_len)
    gt = load_gt(args.gt)
    videos = sorted({m["video"] for m in meta})
    print(f"tracks={len(y)}  positives={int(y.sum())}  videos={len(videos)}  "
          f"feat={X.shape[-1]}+{C.shape[-1]}ctx")

    rng = np.random.RandomState(args.seed)
    order = list(videos); rng.shuffle(order)
    if args.heldout:
        # ONE fold, and it is the honest one: the 13 videos the DETECTOR never saw.
        # K-fold over all 32 mixes detector-train videos into every test fold, so its MAE
        # inherits the §6.7 optimism. This mode is directly comparable to
        # count_eval_heldout.py, whose frozen rule scores MAE 2.38 / 55 of 83 deer.
        from dataset import video_splits                              # noqa: E402
        sp = video_splits()
        unseen = [v for v in videos if sp.get(v) in ("val", "test")]
        if not unseen:
            raise SystemExit("--heldout: no val/test videos found")
        folds = [unseen]
        print(f"HELD-OUT protocol: train on {len(videos)-len(unseen)} detector-train "
              f"videos, test on {len(unseen)} never-seen videos")
    else:
        folds = [order[i::args.folds] for i in range(args.folds)]

    head_pred: dict[str, int] = {}
    rule_pred: dict[str, int] = {}
    tab_pred: dict[str, dict[str, int]] = {"gbm": {}, "logreg": {}}
    fold_rows = []
    for fi, test_vids in enumerate(folds, 1):
        test_vids = set(test_vids)
        if not test_vids:
            continue
        rest = [v for v in videos if v not in test_vids]
        # inner split for threshold selection
        n_val = max(1, len(rest) // 5)
        val_vids = set(rest[:n_val]); tr_vids = set(rest[n_val:])
        idx_tr = [i for i, m in enumerate(meta) if m["video"] in tr_vids]
        idx_va = [i for i, m in enumerate(meta) if m["video"] in val_vids]
        idx_te = [i for i, m in enumerate(meta) if m["video"] in test_vids]
        if not idx_tr or not idx_va or not idx_te:
            continue

        model, pva = train_fold(X[idx_tr], M[idx_tr], C[idx_tr], y[idx_tr],
                                X[idx_va], M[idx_va], C[idx_va], y[idx_va], args, dev)
        best_thr, best_mae = 0.5, float("inf")
        for thr in np.arange(0.05, 0.96, 0.05):
            mm = metrics(head_counts(pva, meta, idx_va, thr), gt, val_vids)
            if mm["MAE"] < best_mae:
                best_mae, best_thr = mm["MAE"], float(thr)

        t = lambda a: torch.tensor(a, device=dev)  # noqa: E731
        with torch.no_grad():
            pte = torch.sigmoid(model(t(X[idx_te]), t(M[idx_te]),
                                      t(C[idx_te]))).cpu().numpy()
        head_pred.update(head_counts(pte, meta, idx_te, best_thr))
        cfg = sweep_rule(meta, idx_tr + idx_va, gt, tr_vids | val_vids)
        rule_pred.update(rule_counts(meta, idx_te, *cfg))

        # ---- tabular learners, identical protocol ----
        for kind in ("gbm", "logreg"):
            clf = fit_tabular(kind, tabular(meta, C, idx_tr), y[idx_tr])
            pv = clf.predict_proba(tabular(meta, C, idx_va))[:, 1]
            bt, bm = 0.5, float("inf")
            for thr in np.arange(0.05, 0.96, 0.05):
                mm = metrics(head_counts(pv, meta, idx_va, thr), gt, val_vids)
                if mm["MAE"] < bm:
                    bm, bt = mm["MAE"], float(thr)
            pt = clf.predict_proba(tabular(meta, C, idx_te))[:, 1]
            tab_pred[kind].update(head_counts(pt, meta, idx_te, bt))

        h = metrics(head_counts(pte, meta, idx_te, best_thr), gt, test_vids)
        r = metrics(rule_counts(meta, idx_te, *cfg), gt, test_vids)
        fold_rows.append({"fold": fi, "test_videos": len(test_vids),
                          "head_MAE": h["MAE"], "rule_MAE": r["MAE"], "thr": best_thr})
        print(f"[fold {fi}/{len(folds)}] {len(test_vids)} vids  "
              f"head MAE {h['MAE']:.2f}  rule MAE {r['MAE']:.2f}  thr {best_thr:.2f}",
              flush=True)

    # In held-out mode only the 13 unseen videos were predicted; scoring against all 32
    # would silently count the 19 untested ones as zero-prediction misses.
    allv = set(folds[0]) if args.heldout else set(videos)
    H = metrics(head_pred, gt, allv)
    R = metrics(rule_pred, gt, allv)
    T = {k: metrics(v, gt, allv) for k, v in tab_pred.items()}
    res = {"folds": args.folds, "per_fold": fold_rows, "head": H, "rule": R,
           "tabular": T, "n_tracks": int(len(y)), "n_positives": int(y.sum())}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(args.out, "per_video.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "gt", "head_pred", "rule_pred"])
        for v in videos:
            w.writerow([v, gt[v], head_pred.get(v, 0), rule_pred.get(v, 0)])

    print(f"\n=== {'HELD OUT (detector never saw these)' if args.heldout else 'CROSS-VALIDATED'} over {len(allv)} videos "
          f"({H['gt_total']} GT deer) ===")
    print(f"{'method':<28} {'MAE':>7} {'RMSE':>7} {'bias':>7} {'over':>5} "
          f"{'under':>6} {'pred':>6}")
    print(f"{'hand-tuned rule (baseline)':<28} {R['MAE']:>7.2f} {R['RMSE']:>7.2f} "
          f"{R['bias']:>+7.2f} {R['over']:>5.0f} {R['under']:>6.0f} {R['pred_total']:>6}")
    for k, m in T.items():
        print(f"{'learned confirmer: ' + k:<28} {m['MAE']:>7.2f} {m['RMSE']:>7.2f} "
              f"{m['bias']:>+7.2f} {m['over']:>5.0f} {m['under']:>6.0f} "
              f"{m['pred_total']:>6}")
    print(f"{'TTC transformer (ours)':<28} {H['MAE']:>7.2f} {H['RMSE']:>7.2f} "
          f"{H['bias']:>+7.2f} {H['over']:>5.0f} {H['under']:>6.0f} {H['pred_total']:>6}")
    bestk = min(list(T.items()) + [("TTC", H)], key=lambda kv: kv[1]["MAE"])
    print(f"\n  BEST learned: {bestk[0]} MAE {bestk[1]['MAE']:.2f} vs rule "
          f"{R['MAE']:.2f}  ({R['MAE']-bestk[1]['MAE']:+.2f}, "
          f"{100*(R['MAE']-bestk[1]['MAE'])/R['MAE']:+.1f}%)")
    d = R["MAE"] - H["MAE"]
    print(f"\n  MAE {d:+.2f} ({100*d/R['MAE']:+.1f}%) vs baseline"
          if R["MAE"] else "")
    print(f"-> {args.out}/results.json")


if __name__ == "__main__":
    main()
