#!/usr/bin/env python3
"""
Leave-one-site-out counting (reviewer question 7).

The paper's held-out protocol splits by VIDEO, so a held-out video can come from a site the
detector saw. This asks the stricter question: train on three sites, count on the fourth,
four times over, so every one of the 235 animals is scored by a detector that never saw its
site.

The confirmation rule is NOT refitted per fold -- the published (m, s, c) is applied
unchanged, which is the honest reading of "does this pipeline transfer", and it keeps the
fold results comparable with the main table.

Usage:
  python src/eval/loso_counting.py --counts-root /work/hdd/.../counts --out results/counting_eval
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

# Published operating point, frozen (sec:sensitivity).
M_MIN, S_MIN, C_MIN = 20, 0.0, 0.65
# Fraction of n>=M_MIN tracks that c>=C_MIN accepts on the published pool C (409 tracks).
Q_POOLC = 0.5134
SITES = ("SHB", "TON", "SHW", "MAS")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def gt_per_video(cvat_dir: str) -> dict[str, int]:
    out = {}
    for x in sorted(glob.glob(os.path.join(cvat_dir, "*.xml"))):
        v = os.path.splitext(os.path.basename(x))[0].replace("_annotations", "")
        out[v] = len(ET.parse(x).getroot().findall(".//track"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-root", default="/work/hdd/bgte/tislam6/wildlife_outputs/counts")
    ap.add_argument("--coverage-dir", default="results/counting_eval")
    ap.add_argument("--cvat-dir", default="data/cvat_export")
    ap.add_argument("--out", default="results/counting_eval")
    args = ap.parse_args()

    gt = gt_per_video(args.cvat_dir)
    report: dict[str, dict] = {}
    tot = defaultdict(int)
    per_video_rows = []

    for site in SITES:
        d = os.path.join(args.counts_root, f"loso_{site}")
        files = sorted(glob.glob(os.path.join(d, "counts_shard*.csv")))
        if not files:
            print(f"  !! no counts for {site} under {d}")
            continue
        pred = defaultdict(int)
        for f in files:
            for r in csv.DictReader(f_ := open(f)):
                if r["site"] != site:           # only the held-out site's videos
                    continue
                if (int(r["n_frames"]) >= M_MIN and float(r["span_s"]) >= S_MIN
                        and float(r["topk_conf"]) >= C_MIN):
                    pred[r["video"]] += 1
            f_.close()

        vids = sorted(v for v in gt if f"_{site}_" in v or v.endswith(f"_{site}"))
        vids = [v for v in gt if csv_site(v) == site]
        errs, g_tot, p_tot, capped = [], 0, 0, 0
        for v in vids:
            g, p = gt[v], pred.get(v, 0)
            errs.append(abs(p - g)); g_tot += g; p_tot += p; capped += min(p, g)
            per_video_rows.append({"site": site, "video": v, "gt": g, "pred": p})

        # A fold-relative variant of the same rule: instead of the absolute c>=0.65,
        # accept the same FRACTION of long tracks that 0.65 accepts on pool C (51.34%).
        # This tests whether the threshold fails to transfer because it is absolute.
        longs = []
        for f in files:
            with open(f) as fh:
                longs += [r for r in csv.DictReader(fh)
                          if r["site"] == site and int(r["n_frames"]) >= M_MIN]
        rank_pred = defaultdict(int)
        rank_thr = None
        if longs:
            cs = sorted(float(r["topk_conf"]) for r in longs)
            rank_thr = cs[max(0, min(len(cs) - 1, int(round((1 - Q_POOLC) * len(cs)))))]
            for r in longs:
                if float(r["topk_conf"]) >= rank_thr:
                    rank_pred[r["video"]] += 1

        # reached / primary come from the coverage pass already on disk
        cov = os.path.join(args.coverage_dir, f"loso_coverage_{site}.csv")
        rc = pr = 0
        if os.path.isfile(cov):
            for r in csv.DictReader(open(cov)):
                if csv_site(r["video"]) == site:
                    rc += int(r["reached"]); pr += int(r["primary"])

        report[site] = {
            "videos": len(vids), "gt": g_tot, "pred": p_tot,
            "reached": rc, "primary": pr, "counted_capped": capped,
            "MAE": round(sum(errs) / max(len(errs), 1), 3),
            "bias": round((p_tot - g_tot) / max(len(vids), 1), 3),
            "max_topk_conf": round(max((float(r["topk_conf"]) for r in longs), default=0.0), 3),
            "rank_threshold": None if rank_thr is None else round(rank_thr, 3),
            "rank_pred": sum(rank_pred.get(v, 0) for v in vids),
            "rank_counted_capped": sum(min(rank_pred.get(v, 0), gt[v]) for v in vids),
            "rank_MAE": round(sum(abs(rank_pred.get(v, 0) - gt[v]) for v in vids)
                              / max(len(vids), 1), 3),
        }
        for k in ("rank_pred", "rank_counted_capped"):
            tot[k] += report[site][k]
        tot["rank_abserr"] += sum(abs(rank_pred.get(v, 0) - gt[v]) for v in vids)
        for k in ("gt", "pred", "reached", "primary", "counted_capped"):
            tot[k] += report[site][k]
        tot["videos"] += len(vids)
        tot["abserr"] += sum(errs)

    G = tot["gt"]
    report["POOLED"] = {
        "videos": tot["videos"], "gt": G, "pred": tot["pred"],
        "reached": tot["reached"], "primary": tot["primary"],
        "counted_capped": tot["counted_capped"],
        "MAE": round(tot["abserr"] / max(tot["videos"], 1), 3),
        "bias": round((tot["pred"] - G) / max(tot["videos"], 1), 3),
        "pct": {k: round(100 * tot[k] / max(G, 1), 1)
                for k in ("reached", "primary", "counted_capped")},
        "wilson": {k: [round(x, 1) for x in wilson(tot[k], G)]
                   for k in ("reached", "primary", "counted_capped", "rank_counted_capped")},
        "rank_counted_capped": tot["rank_counted_capped"],
        "rank_pred": tot["rank_pred"],
        "rank_MAE": round(tot["rank_abserr"] / max(tot["videos"], 1), 3),
        "rank_bias": round((tot["rank_pred"] - G) / max(tot["videos"], 1), 3),
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump(report, open(os.path.join(args.out, "loso_counting.json"), "w"), indent=2)
    with open(os.path.join(args.out, "loso_per_video.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, ["site", "video", "gt", "pred"]); w.writeheader()
        w.writerows(per_video_rows)

    for k, v in report.items():
        if k == "POOLED":
            continue
        print(f"{k}: v={v['videos']:2d} gt={v['gt']:3d} reached={v['reached']:3d} "
              f"primary={v['primary']:3d} counted={v['counted_capped']:3d} "
              f"MAE={v['MAE']:.2f} bias={v['bias']:+.2f}")
    p = report["POOLED"]
    G = p["gt"]
    print(f"\nPOOLED over {p['videos']} videos, {p['gt']} animals:")
    for k in ("reached", "primary", "counted_capped"):
        lo, hi = p["wilson"][k]
        print(f"  {k:15s} {p[k]:3d}/{p['gt']} = {p['pct'][k]:.1f}%  CI [{lo},{hi}]")
    print(f"  MAE {p['MAE']:.2f}   bias {p['bias']:+.2f}")
    lo, hi = p["wilson"]["rank_counted_capped"]
    print(f"  fold-relative threshold: counted {p['rank_counted_capped']}/{G} = "
          f"{100*p['rank_counted_capped']/G:.1f}% CI [{lo},{hi}]  MAE {p['rank_MAE']:.2f}  "
          f"bias {p['rank_bias']:+.2f}")
    print(f"\n-> {args.out}/loso_counting.json")


def csv_site(video: str) -> str:
    for s in SITES:
        if f"_{s}_" in video or video.endswith("_" + s):
            return s
    return "?"


if __name__ == "__main__":
    main()
