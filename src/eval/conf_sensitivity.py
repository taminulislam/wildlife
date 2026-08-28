#!/usr/bin/env python3
"""
How much of the count does the confidence condition carry?

The 3-parameter confirmation rule accepts a candidate track when it persists for >= m
frames, spans >= s seconds, and the mean of its five highest per-frame detector
confidences is >= c. The obvious objection to c is that it discards animals the detector
genuinely found, so removing it should recover them.

This measures that. For each value of c we re-fit the OTHER two parameters on the 19
detector-training videos --- so c is never handicapped by a stale m or s --- freeze the
rule, and report on the 13 held-out videos. That is the protocol of
src/eval/count_eval_heldout.py with c pinned instead of swept.

Two tables come out of it.

  REFIT   c pinned, m and s re-fitted on the 19 for that c. This is the fair question:
          "is the confidence condition earning its place in a 3-parameter rule?"
  FROZEN  m>=20, s>=0 held at the published operating point and only c moved. This is
          the literal question: "what happens if we just delete the threshold?" It is
          the more revealing of the two, because dropping c raises held-out coverage by
          23 points while flipping the system from under- to over-counting.

Usage:
  python src/eval/conf_sensitivity.py \
      --counts /work/hdd/.../counts/phaseC_orphan_yolo11m_conf0.10 \
      --out results/counting/conf_sensitivity.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from count_eval import load_gt, predict, score                     # noqa: E402
from count_eval_heldout import load_rows                           # noqa: E402

MIN_HITS = (1, 2, 3, 5, 8, 12, 20)
MIN_SPAN = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
CONF = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)

# The published operating point (sec. held-out protocol), held fixed for the FROZEN table.
PUBLISHED_M, PUBLISHED_S = 20, 0.0


def coverage(pred: dict[str, int], gt: dict[str, int]) -> float:
    """Capped per-video coverage: over-counting one video cannot mask misses in another."""
    tot = sum(gt.values())
    return 100.0 * sum(min(pred.get(v, 0), g) for v, g in gt.items()) / max(tot, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--gt", default="data/annotate_v2/count_gt.csv")
    ap.add_argument("--splits", default="data/temporal/video_splits.json")
    ap.add_argument("--out", default="results/counting/conf_sensitivity.csv")
    args = ap.parse_args()

    rows = load_rows(args.counts)
    gt = load_gt(args.gt)
    splits = json.load(open(args.splits))
    site_of = {r["video"]: r.get("site", "?") for r in rows}

    seen = {v: g for v, g in gt.items() if splits.get(v) == "train"}
    unseen = {v: g for v, g in gt.items() if splits.get(v) in ("val", "test")}
    if not unseen:
        raise SystemExit("no val/test videos found -- check --splits")

    # ---- FROZEN: the published (m, s), only c moves ----
    frozen = []
    for c in CONF:
        pred = predict(rows, PUBLISHED_M, PUBLISHED_S, c)
        uo, _ = score(pred, unseen, site_of)
        frozen.append({
            "conf_track": c,
            "heldout_MAE": round(uo["MAE"], 3), "heldout_bias": round(uo["bias"], 3),
            "heldout_over": int(uo["over"]), "heldout_under": int(uo["under"]),
            "heldout_counted_pct": round(coverage(pred, unseen), 1),
            "predicted": sum(pred.get(v, 0) for v in unseen),
        })

    # ---- REFIT: m and s re-fitted on the 19 for each c ----
    out = []
    for c in CONF:
        # re-fit m and s for THIS c, on the detector-training videos only
        best, best_o = None, None
        for m in MIN_HITS:
            for s in MIN_SPAN:
                o, _ = score(predict(rows, m, s, c), seen, site_of)
                if best_o is None or (o["MAE"], o["RMSE"]) < (best_o["MAE"], best_o["RMSE"]):
                    best, best_o = (m, s), o
        pred = predict(rows, best[0], best[1], c)
        uo, _ = score(pred, unseen, site_of)
        out.append({
            "conf_track": c, "fit_min_hits": best[0], "fit_min_span_s": best[1],
            "fit_MAE": round(best_o["MAE"], 3),
            "heldout_MAE": round(uo["MAE"], 3), "heldout_RMSE": round(uo["RMSE"], 3),
            "heldout_bias": round(uo["bias"], 3),
            "heldout_over": int(uo["over"]), "heldout_under": int(uo["under"]),
            "heldout_counted_pct": round(coverage(pred, unseen), 1),
        })

    # How much of the full grid's top end depends on the winning c? If the best cells all
    # share one value, c is load-bearing rather than one knob among three.
    allcells = []
    for m in MIN_HITS:
        for s in MIN_SPAN:
            for c in CONF:
                o, _ = score(predict(rows, m, s, c), seen, site_of)
                allcells.append((o["MAE"], o["RMSE"], m, s, c))
    allcells.sort()

    fhdr = list(frozen[0])
    print(f"FROZEN  m>={PUBLISHED_M}, s>={PUBLISHED_S}; only c moves "
          f"({len(unseen)} held-out videos, {sum(unseen.values())} animals)")
    print(" ".join(f"{h:>20}" for h in fhdr))
    for r in frozen:
        print(" ".join(f"{r[h]:>20}" for h in fhdr))

    hdr = list(out[0])
    print(f"\nREFIT   c pinned, m and s re-fitted on the {len(seen)} detector-train videos")
    print(" ".join(f"{h:>18}" for h in hdr))
    for r in out:
        print(" ".join(f"{r[h]:>18}" for h in hdr))

    top = allcells[:12]
    print(f"\ngrid cells: {len(allcells)}")
    print(f"conf_track among the 12 best (fit MAE): "
          f"{sorted({t[4] for t in top})}")
    for mae, rmse, m, s, c in top[:5]:
        print(f"  fit MAE {mae:.3f}  RMSE {rmse:.3f}   m>={m}  s>={s}  c>={c}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(out)
    fout = args.out.replace(".csv", "_frozen.csv")
    with open(fout, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fhdr)
        w.writeheader()
        w.writerows(frozen)
    print(f"\n-> {args.out}\n-> {fout}")


if __name__ == "__main__":
    main()
