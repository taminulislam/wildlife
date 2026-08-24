import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

OUT = "/work/nvme/bgte/tislam6/wildlife_project/docs/slides/assets"
INK, MUTE, GRID = "#241C1E", "#7A6E70", "#E4DDDE"
ROCKET = ["#4A1B33", "#8C2D3F", "#C4453C", "#E87A4F"]   # paper's maroon/rocket family

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": GRID, "figure.facecolor": "none", "axes.facecolor": "none",
})

# ---------- 1. the decomposition funnel -------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=260)
labels = ["Detected\nat all", "Given its own\ntrack", "Counted"]
vals   = [98.8, 88.0, 62.7]
cols   = [ROCKET[0], ROCKET[1], ROCKET[2]]
y = np.arange(3)[::-1]
ax.barh(y, vals, height=0.62, color=cols, zorder=3)
ax.barh(y, [100]*3, height=0.62, color=GRID, alpha=.45, zorder=1)
for yi, v in zip(y, vals):
    ax.text(v + 1.6, yi, f"{v:.1f}%", va="center", ha="left",
            fontsize=17, fontweight="bold", color=INK)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=12.5, color=INK)
ax.set_xlim(0, 112); ax.set_xticks([])
for s in ("top", "right", "bottom", "left"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.set_xlabel("of 83 animals in held-out video", fontsize=11, color=MUTE, labelpad=8)
fig.tight_layout(); fig.savefig(f"{OUT}/funnel.png", transparent=True, bbox_inches="tight")
plt.close(fig)

# ---------- 2. the inversion -------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=260)
cand   = np.array([7008, 18349, 27679, 90239]) / 1000.0
counted= np.array([55, 54, 52, 41])
names  = ["baseline", "max\nrecall", "+ re-ID", "loose\nNMS"]
ax.plot(cand, counted, "-", color=GRID, lw=3, zorder=1)
for x, yv, c in zip(cand, counted, [ROCKET[1]]*3 + [ROCKET[3]]):
    ax.scatter([x], [yv], s=190, color=c, zorder=3, edgecolor="white", linewidth=2)
for x, yv, n in zip(cand, counted, names):
    ax.annotate(n, (x, yv), textcoords="offset points", xytext=(0, 15),
                ha="center", fontsize=10.5, color=MUTE)
    ax.annotate(str(yv), (x, yv), textcoords="offset points", xytext=(0, -24),
                ha="center", fontsize=13, fontweight="bold", color=INK)
ax.set_xscale("log")
ax.set_xticks([7, 20, 90]); ax.set_xticklabels(["7k", "20k", "90k"], fontsize=11)
ax.set_xlabel("candidate tracks generated  (log scale)", fontsize=11.5, color=MUTE, labelpad=6)
ax.set_ylabel("animals counted", fontsize=11.5, color=MUTE)
ax.set_ylim(33, 63); ax.set_yticks([40, 50, 60])
ax.grid(axis="y", color=GRID, lw=1, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/inversion.png", transparent=True, bbox_inches="tight")
plt.close(fig)
print("charts written")
