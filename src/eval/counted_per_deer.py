#!/usr/bin/env python3
"""
Identity-matched COUNTED: how many individual animals a confirmed track actually covers.

The decomposition reports `reached` and `primary` per ANIMAL, but the headline counted
figure is a per-video capped aggregate, sum_v min(pred_v, gt_v) / sum_v gt_v. Those are
different accounting schemes sitting in the same row, and the aggregate is the more
generous of the two: it credits a video for predicting the right NUMBER even when the
accepted tracks do not land on the animals. On pool C that is 55 of 83 against 47 when
matched per animal, so mixing them understates the confirmation loss -- 17% of primaries
rather than 29%.

This computes the per-animal figure with the same frozen rule the aggregate uses: sweep
(min_hits, min_span_s, topk_conf) on the 19 detector-training videos, freeze, then ask, for
each of the 83 unseen animals, whether ANY accepted candidate touches it.

Usage:
  python src/eval/counted_per_deer.py --counts <run> [--out results/...csv]
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
sys.path.insert(0, os.path.join(_HERE, "..", "temporal"))
from count_eval import load_gt, predict, score                        # noqa: E402
from count_eval_heldout import load_rows                              # noqa: E402
from label_tracks import gt_tracks_of, overlaps                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--gt", default="data/annotate_v2/count_gt.csv")
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--splits", default="data/temporal/video_splits.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = load_rows(args.counts)
    gt = load_gt(args.gt)
    sp = json.load(open(args.splits))
    site_of = {r["video"]: r.get("site", "?") for r in rows}
    seen = {v: g for v, g in gt.items() if sp.get(v) == "train"}

    # ---- same rule sweep as count_eval_heldout, so the two figures are comparable ----
    grid = [(mh, ms, ct)
            for mh in (1, 2, 3, 5, 8, 12, 20)
            for ms in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
            for ct in (0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)]
    best_cfg, best = None, None
    for g in grid:
        o, _ = score(predict(rows, *g), seen, site_of)
        if best is None or (o["MAE"], o["RMSE"]) < (best["MAE"], best["RMSE"]):
            best, best_cfg = o, g
    mh, ms, ct = best_cfg

    accepted = {(r["video"], int(r["track_id"])) for r in rows
                if r["n_frames"] >= mh and r["span_s"] >= ms and r["topk_conf"] >= ct}

    # ---- boxes of accepted candidates only ----
    pred: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(args.counts, "shard*", "tracks.csv"))) or \
            sorted(glob.glob(os.path.join(args.counts, "tracks.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                k = (r["video"], int(r["track_id"]))
                if k not in accepted:
                    continue
                xc, yc = float(r["xc"]), float(r["yc"])
                w, h = float(r["w"]), float(r["h"])
                pred[k][int(r["frame"])].append(
                    (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))

    out_rows, tot, cov = [], 0, 0
    for x in sorted(glob.glob(os.path.join(args.cvat_dir, "*.xml"))):
        video = os.path.splitext(os.path.basename(x))[0].replace("_annotations", "")
        if sp.get(video) not in ("val", "test"):
            continue
        gts = gt_tracks_of(x)
        cands = [k for k in pred if k[0] == video]
        n = sum(1 for g in gts
                if any(any(fr in g and any(overlaps(b, g[fr]) for b in bs)
                           for fr, bs in pred[c].items()) for c in cands))
        out_rows.append([video, len(gts), n])
        tot += len(gts); cov += n

    print(f"rule fitted on {len(seen)} training videos: "
          f"min_hits>={mh}, span_s>={ms}, topk_conf>={ct}")
    print(f"identity-matched COUNTED on unseen: {cov}/{tot} = {100*cov/max(tot,1):.1f}%")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["video", "gt", "counted_per_deer"])
            w.writerows(out_rows)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
