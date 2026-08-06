#!/usr/bin/env python3
"""
EVIDENCE-ACCUMULATING CONFIRMATION — the one architectural change the diagnosis asks for.

What the measurements say is broken (§6.11). Of the 83 deer in the held-out videos, the
detector finds 74 and the confirmation rule then discards 27 of them. **26 of those 27
failed on one scalar: top-k mean confidence below 0.65.** They are faint, distant animals
whose per-frame confidence never rises, and a threshold on a summary statistic cannot tell
them apart from noise that also never rises.

Why more capacity did not help. Six learned confirmers -- transformer, GBM, logistic
regression, stumps, clustering, soft-count -- all lost to the 3-parameter rule under three
protocols (§6.2, §6.5, §6.10). They consumed the same summary statistics the rule does and
had far more freedom to overfit ~200 positives.

The change here is not more capacity. It is a different *quantity*.

    rule:      accept if  max/top-k conf  >=  c
    ours:      accept if  SUM over the track of  w(t) * logit(conf_t)  >=  tau

A faint animal observed consistently for 50 frames accumulates evidence; a faint noise blob
observed 50 times does not, because the weight w(t) is not 1 -- it is a kinematic
consistency term. Each detection is down-weighted by how far it sits from where the track's
own constant-velocity model predicted it, normalised by the animal's scale:

    w(t) = exp( -||p_t - p_hat_t|| / (kappa * s_t) )   *   g(s_t / s_median)

The second factor penalises frames whose box size jumps relative to the track's own median,
which is what a tracker latching onto background clutter does. Neither factor looks at
appearance, so nothing here depends on the 27-pixel resolution limit measured in §6.9.

It is a *sum*, not a max, which is the whole point: persistence becomes evidence, so a
30-frame animal at conf 0.45 can outscore a 3-frame spike at conf 0.80. The 3-parameter
rule cannot express that at any threshold setting. It also has only TWO continuous
parameters against the rule's three, so a win could not have been dismissed as extra
capacity.

RESULT (measured, held-out, 2026-08-04) -- IT LOSES:

    pool        rule MAE / counted     evidence head MAE / counted
    C  orphan   2.38 / 55 of 83        3.08 / 46 of 83   (sqrt-normalised)
    G  ReID     2.38 / 55 of 83        3.62 / 39 of 83

The diagnosis of WHY is the useful part, and it is visible in the fit/held-out split:
1.63 fit against 3.08 held-out, a far larger generalisation gap than the rule's
1.53 -> 2.38 despite having fewer parameters. Accumulated evidence is not comparable
across videos -- track-length and density distributions differ by site, so any threshold
on an accumulated quantity is implicitly a threshold on video composition. Normalising by
sqrt(n) recovers part of it (3.23 -> 3.08) and is not enough.

This is the SEVENTH confirmation variant to lose to three hand-tuned parameters
(transformer, GBM, logistic regression, stumps, clustering, soft-count, and now evidence
accumulation), across four evaluation protocols. Reported as a bounded negative result.

Fitted the same way as everything else: sweep on the 19 detector-training videos, freeze,
report on the 13 the detector never saw.

Usage:
  python src/temporal/evidence_confirm.py --counts-dir <run> --sweep
  python src/temporal/evidence_confirm.py --counts-dir <run> --kappa 2.0 --tau 6.0
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))


def load_tracks(counts_dir: str) -> dict[tuple, list]:
    """(video, tid) -> [(frame, conf, xc, yc, w, h), ...] sorted by frame."""
    seqs: dict[tuple, list] = defaultdict(list)
    files = sorted(glob.glob(os.path.join(counts_dir, "shard*", "tracks.csv"))) or \
        sorted(glob.glob(os.path.join(counts_dir, "tracks.csv")))
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                seqs[(r["video"], int(r["track_id"]))].append((
                    int(r["frame"]), float(r["conf"]), float(r["xc"]), float(r["yc"]),
                    float(r["w"]), float(r["h"])))
    for k in seqs:
        seqs[k].sort()
    return seqs


def logit(p: float, eps: float = 1e-3) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def evidence(seq: list, kappa: float) -> float:
    """Kinematically-weighted log-odds sum over one candidate track.

    The weight is the whole contribution: without it this is just a length-weighted
    confidence sum, which the 3-parameter rule already approximates through min_hits.
    """
    if not seq:
        return -1e9
    sizes = [math.sqrt(max(s[4] * s[5], 1.0)) for s in seq]
    s_med = st.median(sizes)
    total = 0.0
    for i, (fr, cf, xc, yc, bw, bh) in enumerate(seq):
        if i >= 2:
            # constant-velocity prediction from the two previous observations, with the
            # real (possibly uneven) frame gaps rather than an assumed stride
            f0, _c0, x0, y0, _w0, _h0 = seq[i - 2]
            f1, _c1, x1, y1, _w1, _h1 = seq[i - 1]
            dt0 = max(f1 - f0, 1)
            dt1 = max(fr - f1, 1)
            px = x1 + (x1 - x0) * dt1 / dt0
            py = y1 + (y1 - y0) * dt1 / dt0
            resid = math.hypot(xc - px, yc - py)
            scale = max(sizes[i], 1.0) * max(dt1, 1)
            w_kin = math.exp(-resid / (kappa * scale))
        else:
            w_kin = 1.0                      # no history yet; do not penalise the start
        # size-stability: a box that jumps relative to the track's own median is clutter
        ratio = sizes[i] / max(s_med, 1e-6)
        w_size = math.exp(-abs(math.log(max(ratio, 1e-6))))
        total += w_kin * w_size * logit(cf)
    return total


def counts_for(seqs: dict, kappa: float, tau: float, norm: str = "none") -> dict[str, int]:
    """norm: how the accumulated evidence is normalised before thresholding.

    'none' thresholds the raw sum, which is what the docstring describes -- but the sum
    grows with track length, and track-length distributions differ by site and by video,
    so an absolute tau fitted on one set of videos does not transfer. 'sqrt' divides by
    sqrt(n): evidence still accumulates (a 50-frame animal beats a 3-frame spike) but the
    scale is comparable across videos. 'mean' divides by n, which removes the accumulation
    entirely and should therefore collapse toward the confidence-threshold baseline -- it
    is included precisely as that control.
    """
    out: dict[str, int] = defaultdict(int)
    for (video, _tid), seq in seqs.items():
        e = evidence(seq, kappa)
        n = max(len(seq), 1)
        if norm == "sqrt":
            e /= math.sqrt(n)
        elif norm == "mean":
            e /= n
        out[video] += int(e >= tau)
    return dict(out)


def score(pred: dict, gt: dict, videos) -> dict:
    errs = [pred.get(v, 0) - gt[v] for v in videos]
    n = len(errs) or 1
    return {"MAE": sum(abs(e) for e in errs) / n,
            "RMSE": math.sqrt(sum(e * e for e in errs) / n),
            "bias": sum(errs) / n,
            "counted": sum(min(pred.get(v, 0), gt[v]) for v in videos),
            "gt": sum(gt[v] for v in videos)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--gt", default="data/annotate_v2/count_gt.csv")
    ap.add_argument("--splits", default="data/temporal/video_splits.json")
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=6.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    seqs = load_tracks(args.counts_dir)
    if not seqs:
        raise SystemExit(f"no tracks.csv under {args.counts_dir}")
    gt = {r["video"]: int(r["unique_deer"]) for r in csv.DictReader(open(args.gt))}
    sp = json.load(open(args.splits))
    fit = [v for v in gt if sp.get(v) == "train"]
    heldout = [v for v in gt if sp.get(v) in ("val", "test")]
    print(f"{len(seqs)} candidate tracks | fit on {len(fit)} videos, "
          f"report on {len(heldout)} never-seen videos")

    # ---- fit (kappa, tau) on the detector-training videos ONLY ----
    best, bcfg = None, None
    grids = {"none": [x * 2.0 for x in range(0, 26)],
             "sqrt": [x * 0.25 for x in range(0, 41)],
             "mean": [x * 0.1 for x in range(-30, 31)]}
    for norm in ("none", "sqrt", "mean"):
        for kappa in (0.5, 1.0, 2.0, 4.0, 8.0):
            for tau in grids[norm]:
                m = score(counts_for(seqs, kappa, tau, norm), gt, fit)
                if best is None or (m["MAE"], m["RMSE"]) < (best["MAE"], best["RMSE"]):
                    best, bcfg = m, (kappa, tau, norm)
        if args.sweep:
            k, t, nm = bcfg
            h = score(counts_for(seqs, k, t, nm), gt, heldout)
            print(f"  best so far with norm<={norm}: kappa {k} tau {t} ({nm})  "
                  f"fit {best['MAE']:.2f}  held-out {h['MAE']:.2f}  "
                  f"counted {h['counted']}/{h['gt']}")

    kappa, tau, norm = bcfg
    pred = counts_for(seqs, kappa, tau, norm)
    f_, h_ = score(pred, gt, fit), score(pred, gt, heldout)
    print(f"\nfitted on training videos: kappa={kappa}, tau={tau}, norm={norm}  "
          f"(2 continuous parameters + a 3-way normalisation choice)")
    print(f"{'scope':<28}{'MAE':>7}{'RMSE':>7}{'bias':>8}{'counted':>12}")
    print(f"{'fit (detector saw these)':<28}{f_['MAE']:>7.2f}{f_['RMSE']:>7.2f}"
          f"{f_['bias']:>+8.2f}{f_['counted']:>7}/{f_['gt']:<4}")
    print(f"{'HELD OUT (never seen)':<28}{h_['MAE']:>7.2f}{h_['RMSE']:>7.2f}"
          f"{h_['bias']:>+8.2f}{h_['counted']:>7}/{h_['gt']:<4}  <- PUBLISH")
    print(f"\nbaseline (3-param rule, same protocol): MAE 2.38, 55/83 counted")
    d = 2.38 - h_["MAE"]
    print(f"VERDICT: evidence head {'BEATS' if d > 0.005 else 'TIES' if d > -0.005 else 'LOSES to'} "
          f"the rule by {abs(d):.2f} MAE")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "split", "gt", "predicted", "error"])
            for v in sorted(gt):
                w.writerow([v, sp.get(v, "?"), gt[v], pred.get(v, 0),
                            pred.get(v, 0) - gt[v]])
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
