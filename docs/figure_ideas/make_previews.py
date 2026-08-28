#!/usr/bin/env python3
"""Draft renders of the six candidate figures, from real data, so the choice is made by
looking rather than by reading a description. Deliberately unpolished -- no caption fitting,
no print-size tuning. That work happens once one is chosen."""
from __future__ import annotations
import csv, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "docs/figure_ideas"
STATE_C = {"counted": "#0072B2", "rejected": "#E69F00", "undetected": "#D55E00"}
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.dpi": 150})
A = json.load(open("data/temporal/missed_unseen.json"))


def save(fig, name, note):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig); print(f"  {name}.png  -- {note}")


def idea1():
    """One row per animal, sorted by size; bar spans its time on screen."""
    a = sorted(A, key=lambda x: -x["sz"])
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    for i, x in enumerate(a):
        dur = (x["f1"] - x["f0"]) / 60.0
        ax.barh(i, dur, left=0, height=0.72, color=STATE_C[x["state"]],
                edgecolor="white", linewidth=0.3)
        ax.text(-0.6, i, f"{x['sz']:.0f}px", ha="right", va="center", fontsize=5.0,
                color="#555")
    ax.set_ylim(-1, len(a)); ax.invert_yaxis(); ax.set_yticks([])
    ax.set_xlabel("seconds on screen"); ax.set_xlim(-6, None)
    ax.set_title("Idea 1 — every held-out animal, largest at top\n"
                 "colour = fate; the losses are ordered by size", loc="left", fontsize=10)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.legend(handles=[plt.Line2D([],[],lw=6,color=c,label=k) for k,c in STATE_C.items()],
              fontsize=7, loc="lower right", frameon=False)
    ax.grid(axis="x", color="#EEE", lw=0.5); ax.set_axisbelow(True)
    save(fig, "idea1_per_animal_fate", "83 rows; size ordering visible")


def idea2():
    """11 detectors x 83 animals: who misses whom."""
    MODELS = [("yolov8m_640","YOLOv8m"),("yolov9m_640","YOLOv9m"),("yolov10m_640","YOLOv10m"),
              ("yolo11m_640","YOLO11m"),("yolo12m_640","YOLO12m"),("rtdetr-l_640","RT-DETR-L"),
              ("atss_r50","ATSS R50"),("tood_r50","TOOD R50"),("dino_r50","DINO R50"),
              ("faster-rcnn_r50","Faster R-CNN"),("rtmdet_m","RTMDet-m")]
    order = [(x["video"], x["gi"]) for x in sorted(A, key=lambda x: -x["sz"])]
    idx = {k: i for i, k in enumerate(order)}
    M = np.full((len(MODELS), len(order)), np.nan)
    for r, (tag, _) in enumerate(MODELS):
        f = f"results/track_recall/roster_conf0.25/{tag}/track_recall.csv"
        if not os.path.exists(f): continue
        for row in csv.DictReader(open(f)):
            k = (row["video"], int(row["track_idx"]))
            if k in idx: M[r, idx[k]] = float(row["found_touch"])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.imshow(M, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#D55E00", "#DCEBF5"]),
              interpolation="nearest")
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([m[1] for m in MODELS], fontsize=7.5)
    ax.set_xlabel("held-out animals, largest to smallest  →")
    ax.set_xticks([])
    miss = np.nansum(1 - M, axis=0)
    ax.set_title("Idea 2 — detector × animal coverage. Orange = this model never fired on "
                 "that animal.\nVertical orange bands = every architecture fails on the same "
                 f"animals ({int((miss>=8).sum())} animals missed by ≥8 of 11)",
                 loc="left", fontsize=10)
    save(fig, "idea2_coverage_matrix", "the one that yields a new result")


def idea3():
    """Failure as a joint function of size and persistence."""
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for st, c in STATE_C.items():
        pts = [(x["sz"], x["gt_frames"]) for x in A if x["state"] == st]
        ax.scatter(*zip(*pts), s=34, c=c, label=f"{st} ({len(pts)})",
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(95.6, color="#444", ls="--", lw=1.0, zorder=2)
    ax.text(97, ax.get_ylim()[1]*0.92, "largest box in the\ntraining split (95.6 px)",
            fontsize=7, color="#444")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("median box size (px)"); ax.set_ylabel("frames the animal is annotated in")
    ax.set_title("Idea 3 — size × persistence. Small animals survive if they persist;\n"
                 "large ones fail regardless, because training never showed them",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(color="#EEE", lw=0.5); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    save(fig, "idea3_size_vs_persistence", "replaces the scale histogram")


def idea4():
    """MAE over the confirmation rule's grid."""
    f = "/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10/eval/count_eval_sweep.csv"
    rows = [r for r in csv.DictReader(open(f))]
    spans = sorted({float(r["min_span_s"]) for r in rows})[:3]
    hits = sorted({int(r["min_hits"]) for r in rows}); confs = sorted({float(r["conf_track"]) for r in rows})
    fig, axes = plt.subplots(1, len(spans), figsize=(11, 3.3), sharey=True)
    for ax, sp in zip(axes, spans):
        G = np.full((len(confs), len(hits)), np.nan)
        for r in rows:
            if float(r["min_span_s"]) == sp:
                G[confs.index(float(r["conf_track"])), hits.index(int(r["min_hits"]))] = float(r["MAE"])
        im = ax.imshow(G, aspect="auto", origin="lower", cmap="viridis_r", vmin=1.5, vmax=7)
        ax.set_xticks(range(len(hits))); ax.set_xticklabels(hits, fontsize=7)
        ax.set_yticks(range(len(confs))); ax.set_yticklabels([f"{c:g}" for c in confs], fontsize=7)
        ax.set_title(f"min_span_s = {sp:g}", fontsize=8.5)
        ax.set_xlabel("min_hits", fontsize=8)
        if sp == 0.0:
            ax.plot(hits.index(20), confs.index(0.65), marker="*", ms=17,
                    color="white", markeredgecolor="black", markeredgewidth=0.8)
    axes[0].set_ylabel("conf_track", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02, label="MAE (animals / video)")
    fig.suptitle("Idea 4 — the rule's decision surface. Star = the published operating point; "
                 "the good region is a ridge, not a plateau", x=0.02, ha="left", fontsize=10)
    save(fig, "idea4_rule_surface", "retires a table, so free on page count")


def idea5():
    """Per-video: when in the transect the animals appear, and which were counted."""
    byv = {}
    for x in A: byv.setdefault(x["video"], []).append(x)
    vids = sorted(byv, key=lambda v: -len(byv[v]))[:9]
    fig, axes = plt.subplots(3, 3, figsize=(11, 5.6))
    for ax, v in zip(axes.ravel(), vids):
        an = sorted(byv[v], key=lambda x: x["f0"])
        for i, x in enumerate(an):
            ax.barh(i, (x["f1"]-x["f0"])/60.0, left=x["f0"]/60.0, height=0.7,
                    color=STATE_C[x["state"]])
        ax.set_title(f"{v[:26]}  ({len(an)} deer)", fontsize=7.2, loc="left")
        ax.tick_params(labelsize=6); ax.set_yticks([])
        for s in ("top","right","left"): ax.spines[s].set_visible(False)
        ax.grid(axis="x", color="#EEE", lw=0.4); ax.set_axisbelow(True)
    for ax in axes.ravel()[len(vids):]: ax.axis("off")
    fig.suptitle("Idea 5 — per-video timelines. x = seconds into the transect; "
                 "each bar is one animal, coloured by fate", x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.94])
    save(fig, "idea5_per_video_timeline", "best for a wildlife audience; most work")


def idea6():
    """Is the per-track posterior calibrated?"""
    f = "results/temporal/calibrated_orphan/per_track_confidence.csv"
    rows = [r for r in csv.DictReader(open(f))]
    p = np.array([float(r["confidence"]) for r in rows])
    y = np.array([r["counted"] == "1" for r in rows], float)
    bins = np.linspace(0, 1, 11); mid, obs, n = [], [], []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i+1] if i < 9 else p <= 1)
        if m.sum() >= 5: mid.append(p[m].mean()); obs.append(y[m].mean()); n.append(m.sum())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    a1.plot([0,1],[0,1], ls="--", color="#999", lw=1)
    a1.scatter(mid, obs, s=[min(300, 12+x/6) for x in n], color="#0072B2",
               edgecolor="white", zorder=3)
    a1.set_xlabel("predicted confidence"); a1.set_ylabel("observed fraction counted")
    a1.set_title("reliability (dot area = tracks in bin)", fontsize=9, loc="left")
    a2.hist(p, bins=30, color="#0072B2", alpha=0.85)
    a2.set_yscale("log"); a2.set_xlabel("per-track posterior"); a2.set_ylabel("tracks")
    a2.set_title(f"score distribution, {len(rows):,} tracks", fontsize=9, loc="left")
    for ax in (a1, a2):
        for s in ("top","right"): ax.spines[s].set_visible(False)
        ax.grid(color="#EEE", lw=0.5); ax.set_axisbelow(True)
    fig.suptitle("Idea 6 — calibration, which Section 3.4 currently only asserts",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.93])
    save(fig, "idea6_calibration", "smallest job; supports an unevidenced claim")


print("rendering previews:")
for fn in (idea1, idea2, idea3, idea4, idea5, idea6):
    try: fn()
    except Exception as e: print(f"  [FAIL] {fn.__name__}: {e}")
