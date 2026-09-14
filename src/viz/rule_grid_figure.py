#!/usr/bin/env python3
"""
Figure (Section 4.6): the confirmation rule's m x c grid, fit-set error beside held-out error.

Two panels share one colour scale so they can be compared cell for cell. Error spans two
orders of magnitude (1.5 to 244 animals per video), so the scale is logarithmic; on a linear
scale every cell near the optimum would be the same colour and the valley would be invisible.
Every cell is labelled, and the published operating point is outlined in both panels.

Reads results/counting/rule_grid/grid.csv written by src/eval/rule_grid.py.
"""
from __future__ import annotations
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.colors import LogNorm                                  # noqa: E402
from matplotlib.patches import Rectangle                               # noqa: E402
import numpy as np                                                     # noqa: E402

INK, INK2 = "#2F3437", "#6B7378"


def main() -> None:
    rows = [r for r in csv.DictReader(open(os.environ.get("GRID","results/counting/rule_grid/grid.csv")))
            if r["grid"] == "m_x_c"]
    ms = sorted({int(r["m"]) for r in rows})
    cs = sorted({float(r["c"]) for r in rows})
    def mat(key):
        a = np.zeros((len(ms), len(cs)))
        for r in rows:
            a[ms.index(int(r["m"])), cs.index(float(r["c"]))] = float(r[key])
        return a
    fit, held = mat("fit_MAE"), mat("heldout_MAE")
    norm = LogNorm(vmin=1.4, vmax=max(fit.max(), held.max()))

    plt.rcParams.update({"font.size": 6.5, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 2.35), constrained_layout=True)
    for ax, a, title in ((axes[0], fit, "Fit set (19 training videos)"),
                         (axes[1], held, "Held out (13 videos)")):
        im = ax.imshow(a, cmap="viridis_r", norm=norm, aspect="auto")
        for i in range(len(ms)):
            for j in range(len(cs)):
                v = a[i, j]
                ax.text(j, i, f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", va="center",
                        fontsize=5.0, color="white" if v > 12 else INK)
        pi, pj = ms.index(20), cs.index(0.65)
        ax.add_patch(Rectangle((pj - .5, pi - .5), 1, 1, fill=False, ec="#C0392B", lw=1.3))
        ax.set_xticks(range(len(cs))); ax.set_xticklabels([f"{c:g}" for c in cs])
        ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms)
        ax.set_xlabel("minimum track score $c$")
        ax.set_ylabel("minimum track length $m$" if ax is axes[0] else "")
        ax.set_title(title, loc="left", fontsize=7.2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01, aspect=30)
    cb.set_label("MAE (log scale)", fontsize=6.5)
    cb.outline.set_visible(False)
    out = os.environ.get("OUT", "docs/figures"); os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig_rule_grid.{ext}")
    print(f"-> {out}/fig_rule_grid.pdf / .png")


if __name__ == "__main__":
    main()
