#!/usr/bin/env python3
"""
Radar of the leave-one-site-out decomposition (Table 12).

Why a radar here. The point of the LOSO result is not that any single rate is high or low but
that the three stages degrade with different SHAPES across sites: reached and primary stay
large and roughly regular, while counted collapses unevenly and reaches zero at MAS. A radar
puts those three shapes on one set of axes, which is what makes the irregularity visible.

Radar has two well-known defects and both are handled: area grows as the square of the radius,
so every vertex is labelled with its value rather than read off the ring, and the axis order is
arbitrary, so sites run clockwise by corpus size (SHB 132 animals down to MAS 15) and the
caption says so.

Numbers come from results/counting_eval/loso_counting.json, written by
src/eval/loso_counting.py -- nothing is typed in here.

Usage:  python src/viz/loso_radar.py --out docs/figures
"""
from __future__ import annotations
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import matplotlib.patheffects as pe                                    # noqa: E402
import numpy as np                                                     # noqa: E402

# Same palette as fig1_funnel: two lightnesses of one hue for the ordered pipeline stages,
# a warm accent for the outcome.
PALE, STEEL, CLAY = "#A7BFD4", "#5C7FA0", "#C58164"
INK, INK2, GRID = "#2F3437", "#6B7378", "#D9DEE2"
SERIES = [("Detected (reached)", "reached", PALE),
          ("Own track (primary)", "primary", STEEL),
          ("Counted", "counted_capped", CLAY)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/counting_eval/loso_counting.json")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--name", default="fig_loso_radar")
    args = ap.parse_args()

    rep = json.load(open(args.json))
    sites = [s for s in rep if s != "POOLED"]
    sites.sort(key=lambda s: -rep[s]["gt"])          # largest corpus first, clockwise
    n = len(sites)
    ang = [i / n * 2 * np.pi for i in range(n)]
    closed = ang + ang[:1]

    plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "font.size": 10, "text.color": INK, "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(5.6, 4.4), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for label, key, colour in SERIES:
        vals = [100.0 * rep[s][key] / rep[s]["gt"] for s in sites]
        v = vals + vals[:1]
        ax.plot(closed, v, color=colour, lw=2.2, label=label, zorder=3)
        ax.fill(closed, v, color=colour, alpha=0.18, zorder=2)
        # Label every vertex: a radar's radius is not readable by eye. Counted labels sit
        # inside the polygon to clear the primary label above them, except where the value
        # is near zero and inside would mean on top of the origin.
        for a, x in zip(ang, vals):
            if key == "counted_capped":
                off = (0, -14) if x > 12 else (0, 13)
            else:
                off = (0, 7)
            t = ax.annotate(f"{x:.0f}", (a, x), textcoords="offset points", xytext=off,
                            ha="center", fontsize=9.5, color=INK, zorder=5)
            t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

    ax.set_xticks(ang)
    ax.set_xticklabels([f"{s} ({rep[s]['gt']})" for s in sites], fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100%"], fontsize=8.5, color=INK2)
    ax.set_rlabel_position(180 / n)
    # Ring labels cross the filled polygons, so they carry a white plate.
    for lbl in ax.get_yticklabels():
        lbl.set_bbox(dict(facecolor="white", edgecolor="none", pad=1.2))
        lbl.set_zorder(6)
    ax.grid(color=GRID, lw=0.8)
    ax.spines["polar"].set_color(GRID)
    ax.tick_params(axis="x", pad=6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.19), ncol=3,
              frameon=False, fontsize=9.5, handlelength=1.2, columnspacing=1.2,
              borderaxespad=0, handletextpad=0.5)

    os.makedirs(args.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"{args.name}.{ext}"))
    plt.close(fig)
    print(f"-> {args.out}/{args.name}.pdf / .png")
    for s in sites:
        r = rep[s]
        print(f"   {s}: reached {100*r['reached']/r['gt']:.0f}%  "
              f"primary {100*r['primary']/r['gt']:.0f}%  "
              f"counted {100*r['counted_capped']/r['gt']:.0f}%")


if __name__ == "__main__":
    main()
