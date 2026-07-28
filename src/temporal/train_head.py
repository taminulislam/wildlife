#!/usr/bin/env python3
"""
Train the temporal confirmation head and compare it against the hand-tuned rule —
THE headline experiment of the paper.

Two things make this comparison honest:

1. **Both are evaluated the same way**: predicted count per video vs CVAT GT, MAE/RMSE
   on the held-out TEST videos. Classification accuracy is reported but is NOT the
   objective — a head with great AUC that mis-counts is useless.
2. **The baseline is given every advantage**: its 3 thresholds are swept exhaustively on
   the TRAIN+VAL videos and the best is applied to test. The head's decision threshold
   is likewise chosen on VAL only. Neither sees test during fitting.

Usage:
  python src/temporal/train_head.py --counts-dir <counts run> \
      --labels data/temporal/tracks_labelled.csv --out results/temporal/ttc_v1
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
from dataset import build, video_splits          # noqa: E402
from model import TemporalTrackNet, count_params  # noqa: E402


def load_gt(path: str) -> dict[str, int]:
    return {r["video"]: int(r["unique_deer"])
            for r in csv.DictReader(open(path))}


def count_metrics(pred_per_video: dict[str, int], gt: dict[str, int],
                  videos: set[str]) -> dict:
    errs = [pred_per_video.get(v, 0) - gt[v] for v in videos]
    n = len(errs) or 1
    return {
        "videos": len(errs),
        "MAE": sum(abs(e) for e in errs) / n,
        "RMSE": math.sqrt(sum(e * e for e in errs) / n),
        "bias": sum(errs) / n,
        "over": sum(e for e in errs if e > 0),
        "under": -sum(e for e in errs if e < 0),
        "pred_total": sum(pred_per_video.get(v, 0) for v in videos),
        "gt_total": sum(gt[v] for v in videos),
    }


def rule_counts(meta, min_hits, min_span, conf_track) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in meta:
        ok = (m["n_frames"] >= min_hits and m["span_s"] >= min_span
              and m["topk_conf"] >= conf_track)
        out[m["video"]] = out.get(m["video"], 0) + int(ok)
    return out


def sweep_rule(meta, gt, videos):
    best, best_cfg = None, None
    for mh in (1, 2, 3, 5, 8, 12, 20, 30):
        for ms in (0.0, 0.1, 0.3, 0.5, 1.0, 2.0):
            for ct in (0.15, 0.25, 0.35, 0.50, 0.65, 0.80):
                m = count_metrics(rule_counts(meta, mh, ms, ct), gt, videos)
                if best is None or (m["MAE"], m["RMSE"]) < (best["MAE"], best["RMSE"]):
                    best, best_cfg = m, (mh, ms, ct)
    return best, best_cfg


def head_counts(probs, meta, thr) -> dict[str, int]:
    out: dict[str, int] = {}
    for p, m in zip(probs, meta):
        out[m["video"]] = out.get(m["video"], 0) + int(p >= thr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--labels", default="data/temporal/tracks_labelled.csv")
    ap.add_argument("--gt", default="data/annotate_v2/count_gt.csv")
    ap.add_argument("--out", default="results/temporal/ttc_v1")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    gt = load_gt(args.gt)
    splits = video_splits()
    data = build(args.counts_dir, args.labels, args.max_len, splits)
    for s in ("train", "val", "test"):
        if data[s] is None:
            raise SystemExit(f"split '{s}' is empty — check the labels/counts dir")
        X, M, y, meta = data[s]
        print(f"{s:<6} tracks={len(y):>4} positives={int(y.sum()):>3} "
              f"videos={len({m['video'] for m in meta}):>2}")

    dev = torch.device(args.device)
    model = TemporalTrackNet(d_model=args.d_model, layers=args.layers,
                             dropout=args.dropout).to(dev)
    print(f"\nTemporalTrackNet params: {count_params(model):,}")

    Xtr, Mtr, ytr, mtr = data["train"]
    Xva, Mva, yva, mva = data["val"]
    Xte, Mte, yte, mte = data["test"]
    t = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    Xtr_, Mtr_, ytr_ = t(Xtr), t(Mtr), t(ytr)

    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    val_videos = {m["video"] for m in mva}
    test_videos = {m["video"] for m in mte}
    train_val_meta = mtr + mva
    train_val_videos = {m["video"] for m in train_val_meta}

    best = {"mae": float("inf"), "state": None, "thr": 0.5, "epoch": -1}
    n = len(ytr_)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            logit = model(Xtr_[idx], Mtr_[idx])
            loss = lossf(logit, ytr_[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(t(Xva), t(Mva))).cpu().numpy()
        # choose the decision threshold on VAL by counting MAE (not accuracy)
        best_thr, best_mae = 0.5, float("inf")
        for thr in np.arange(0.05, 0.96, 0.05):
            m = count_metrics(head_counts(pv, mva, thr), gt, val_videos)
            if m["MAE"] < best_mae:
                best_mae, best_thr = m["MAE"], float(thr)
        if best_mae < best["mae"]:
            best = {"mae": best_mae, "thr": best_thr, "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        if ep % 20 == 0 or ep == 1:
            print(f"[ep {ep:>3}] train_loss={tot/n:.4f}  val_count_MAE={best_mae:.3f} "
                  f"(thr {best_thr:.2f})  best={best['mae']:.3f}@{best['epoch']}")

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pte = torch.sigmoid(model(t(Xte), t(Mte))).cpu().numpy()
    head_test = count_metrics(head_counts(pte, mte, best["thr"]), gt, test_videos)

    # ---- baseline: sweep the rule on TRAIN+VAL, apply to TEST ----
    _, cfg = sweep_rule(train_val_meta, gt, train_val_videos)
    rule_test = count_metrics(rule_counts(mte, *cfg), gt, test_videos)

    acc = float(((pte >= best["thr"]).astype(float) == yte).mean())
    res = {
        "test_videos": sorted(test_videos),
        "head": head_test, "head_threshold": best["thr"],
        "head_best_epoch": best["epoch"], "head_track_accuracy": acc,
        "rule": rule_test,
        "rule_cfg": {"min_hits": cfg[0], "min_span_s": cfg[1], "conf_track": cfg[2]},
        "params": count_params(model),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    torch.save({"state_dict": best["state"], "threshold": best["thr"],
                "args": vars(args)}, os.path.join(args.out, "ttc.pt"))

    print("\n=== HELD-OUT TEST VIDEOS "
          f"({len(test_videos)} videos, {head_test['gt_total']} GT deer) ===")
    print(f"{'method':<28} {'MAE':>7} {'RMSE':>7} {'bias':>7} {'pred':>6} {'gt':>4}")
    print(f"{'hand-tuned rule (baseline)':<28} {rule_test['MAE']:>7.2f} "
          f"{rule_test['RMSE']:>7.2f} {rule_test['bias']:>+7.2f} "
          f"{rule_test['pred_total']:>6} {rule_test['gt_total']:>4}")
    print(f"{'TTC temporal head (ours)':<28} {head_test['MAE']:>7.2f} "
          f"{head_test['RMSE']:>7.2f} {head_test['bias']:>+7.2f} "
          f"{head_test['pred_total']:>6} {head_test['gt_total']:>4}")
    d = rule_test["MAE"] - head_test["MAE"]
    print(f"\n  MAE improvement: {d:+.2f} "
          f"({100*d/rule_test['MAE']:+.1f}% vs baseline)" if rule_test["MAE"] else "")
    print(f"  track-level accuracy {acc:.3f}, threshold {best['thr']:.2f}, "
          f"best epoch {best['epoch']}")
    print(f"-> {args.out}/results.json")


if __name__ == "__main__":
    main()
