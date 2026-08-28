#!/usr/bin/env python3
"""
Per-epoch validation curves for every detector in the benchmark table, as a 3x4 grid of
small multiples.

One panel per architecture, drawn from the training logs the runs actually wrote --
`results.csv` for the Ultralytics models, `vis_data/scalars.json` for the mmdetection
ones. Nothing here is re-run or re-simulated; if a curve is short it is because that run
was short.

The two frameworks log different things, so the four series differ by family and each
panel carries its own legend:

  Ultralytics   Precision, Recall, F1, mAP@50          (F1 computed from P and R)
  mmdetection   mAP@50, mAP@75, mAP@[.50:.95], mAP-small

Palettes:
  --palette mpl   matplotlib's default blue/orange/green/red. This is the house style of
                  the companion figure, but its green and orange are indistinguishable
                  under protanopia (CVD dE 0.7), so the marker shapes are the only thing
                  separating those two series for a red-blind reader.
  --palette cvd   Okabe-Ito. Same visual character, and every adjacent pair clears the
                  CVD floor (worst dE 11.0). Preferred unless matching the other figure
                  matters more.

Usage:
  python src/viz/training_curves.py --out overleaf_MDPI/figures/fig5_training.pdf
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/work/hdd/bgte/tislam6/wildlife_outputs/runs"
MMDET = "/work/hdd/bgte/tislam6/wildlife_outputs/mmdet_runs"

PALETTES = {
    "mpl": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    "cvd": ["#0072B2", "#E69F00", "#009E73", "#D55E00"],
}
MARKERS = ["o", "s", "^", "P"]

# The eleven models the benchmark table reports, plus the 1280 px variant the resolution
# ablation turns on. Deformable DETR is deliberately absent: it produced two validation
# epochs before the compute budget ran out, which is not a curve -- that run's absence is
# the same evidence the table's exclusion note carries.
ULTRA = [
    ("YOLOv8m", "yolov8m_640_v3pooled"),
    ("YOLO11m", "yolo11m_640_v3pooled"),
    ("YOLO12m", "yolo12m_640_v3pooled"),
    ("YOLOv9m", "yolov9m_640_v3pooled"),
    ("YOLOv10m", "yolov10m_640_v3pooled"),
    ("RT-DETR-L", "rtdetr-l_640_v3pooled"),
    ("YOLOv9m @1280", "yolov9m_1280_v3pooled"),
]
MM = [
    ("ATSS R50", "atss_r50_v3pooled"),
    ("TOOD R50", "tood_r50_v3pooled"),
    ("DINO R50", "dino_r50_v3pooled"),
    ("Faster R-CNN R50", "faster-rcnn_r50_v3pooled"),
    ("RTMDet-m", "rtmdet_m_v3pooled"),
]


def load_ultra(run: str):
    """-> (epochs, {series: values}) in percent, or None if the run is not on disk."""
    path = os.path.join(RUNS, run, "results.csv")
    if not os.path.isfile(path):
        return None
    rows = list(csv.DictReader(path and open(path)))
    ep = [int(float(r["epoch"])) for r in rows]
    p = [float(r["metrics/precision(B)"]) * 100 for r in rows]
    r_ = [float(r["metrics/recall(B)"]) * 100 for r in rows]
    f1 = [2 * a * b / (a + b) if (a + b) else 0.0 for a, b in zip(p, r_)]
    m50 = [float(r["metrics/mAP50(B)"]) * 100 for r in rows]
    return ep, {"Precision (%)": p, "Recall (%)": r_, "F1 Score (%)": f1, "mAP@50 (%)": m50}


def load_mmdet(run: str):
    """mmdet writes one JSON object per log event; validation events carry coco/bbox_*."""
    hits = [f for f in glob.glob(os.path.join(MMDET, run, "*", "vis_data", "scalars.json"))
            if "test_eval" not in f]
    if not hits:
        return None
    vals = []
    with open(sorted(hits)[0]) as fh:
        for line in fh:
            d = json.loads(line)
            if "coco/bbox_mAP_50" in d:
                vals.append(d)
    if not vals:
        return None
    # "step" on a validation event is the epoch it ran at, and every config validates
    # every second epoch -- so enumerating the events instead would halve the apparent
    # length of every mmdet panel against the Ultralytics ones.
    ep = [v["step"] for v in vals]
    g = lambda k: [v[k] * 100 for v in vals]
    return ep, {"mAP@50 (%)": g("coco/bbox_mAP_50"), "mAP@75 (%)": g("coco/bbox_mAP_75"),
                "mAP@[.5:.95] (%)": g("coco/bbox_mAP"), "mAP-small (%)": g("coco/bbox_mAP_s")}


def draw(ax, ep, series, colours):
    # Markers on every epoch would be a solid band at 110 epochs, so thin them to ~22 per
    # line. The shape still does the work of separating series without the clutter.
    every = max(1, len(ep) // 22)
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(ep, ys, color=colours[i], marker=MARKERS[i], markersize=2.4,
                markevery=every, linewidth=0.9, label=name,
                markeredgewidth=0.0 if MARKERS[i] != "P" else 0.3)
    ax.set_xlabel("Epoch", fontsize=6.5)
    ax.set_ylabel("Score", fontsize=6.5)
    ax.tick_params(labelsize=5.8, length=2, width=0.5, pad=1.5)
    ax.grid(True, color="#e3e3e8", linewidth=0.45)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color("#9a9aa2")
    leg = ax.legend(fontsize=5.0, loc="best", frameon=True, handlelength=1.6,
                    borderpad=0.3, labelspacing=0.22, handletextpad=0.4, borderaxespad=0.3)
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("#c9c9d0")
    leg.get_frame().set_alpha(0.92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", choices=list(PALETTES), default="mpl")
    ap.add_argument("--out", default="overleaf_MDPI/figures/fig5_training.pdf")
    ap.add_argument("--width", type=float, default=7.27, help="inches; MDPI \\fulllength")
    ap.add_argument("--height", type=float, default=5.6)
    args = ap.parse_args()
    colours = PALETTES[args.palette]

    panels = []
    for name, run in ULTRA:
        d = load_ultra(run)
        panels.append((name, d))
    for name, run in MM:
        d = load_mmdet(run)
        panels.append((name, d))

    missing = [n for n, d in panels if d is None]
    if missing:
        print(f"[warn] no log on disk for: {', '.join(missing)}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": "white",
                         "figure.facecolor": "white", "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 4, figsize=(args.width, args.height))
    for k, (ax, (name, d)) in enumerate(zip(axes.ravel(), panels)):
        letter = chr(ord("a") + k)
        if d is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"({letter}) {name}\nlog not on disk", ha="center",
                    va="center", fontsize=6)
            continue
        draw(ax, d[0], d[1], colours)
        ax.set_title(f"({letter}) {name}", fontsize=7, fontweight="bold", pad=3, y=-0.42)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.985, bottom=0.105,
                        wspace=0.30, hspace=0.78)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"-> {args.out}\n-> {png}   ({args.palette} palette, "
          f"{sum(d is not None for _, d in panels)}/12 panels)")


if __name__ == "__main__":
    main()
