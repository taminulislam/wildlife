"""Does the ground truth itself under-represent distant animals?

Box size is the only range proxy (same definition as Sec. 4.10: median sqrt(box area)
over a track's visible frames). For every annotated deer track in the 32 CVAT exports:

  size_med    median sqrt(area)              -> range proxy used in Fig. 10
  size_first  sqrt(area) on the first visible frame
  size_max    largest sqrt(area) = closest approach inside the field of view
  dur         visible frames (60 fps)

Tests
  T1  duration vs range: quartiles of size_med and size_first; Spearman pooled and
      within-video (sizes and durations ranked inside each video, so vehicle speed and
      road geometry, which vary by video, cannot drive the result).
  T2  formation vs range: animals per size bin against geometric availability. With
      s ~ 1/range and uniform ground density in the camera wedge, n(s) ds ~ s^-3 ds;
      anchored on 30-60 px, a shortfall at small s means distant animals are missing.
      (A closest-approach distance histogram is not usable: annotated boxes keep one
      size for most of a track.)
  T3  where tracks start: size at first sighting against the track's own later sizes.
Cluster bootstrap over videos (tracks within a video are not independent).
"""
import glob, math, os, random, statistics, sys, xml.etree.ElementTree as ET
import json

import numpy as np
from scipy import stats

CVAT = sys.argv[1] if len(sys.argv) > 1 else "data/cvat_export"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
W_IMG, H_IMG = 640, 512

rows = []
for xp in sorted(glob.glob(os.path.join(CVAT, "*.xml"))):
    video = os.path.splitext(os.path.basename(xp))[0]
    site = video.split("_")[1]
    for tr in ET.parse(xp).getroot().findall("track"):
        if tr.get("label") != "deer":
            continue
        bx = []
        for b in tr.findall("box"):
            if b.get("outside") == "1":
                continue
            x1, y1, x2, y2 = (float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            bx.append((int(b.get("frame")), x1, y1, x2, y2))
        if not bx:
            continue  # the one track outside on every frame (FernRidgeRd_TON)
        bx.sort()
        s = [math.sqrt(max(0.0, (b[3] - b[1]) * (b[4] - b[2]))) for b in bx]
        f0 = bx[0]
        cx, cy = (f0[1] + f0[3]) / 2, (f0[2] + f0[4]) / 2
        edge = min(f0[1], f0[2], W_IMG - f0[3], H_IMG - f0[4]) <= 3
        rows.append(dict(video=video, site=site, track=tr.get("id"), dur=len(bx),
                         span=bx[-1][0] - bx[0][0] + 1, size_med=statistics.median(s),
                         size_first=s[0], size_max=max(s), size_min=min(s),
                         first_is_min=s[0] <= min(s) * 1.001, first_at_edge=edge,
                         first_cy=cy))

n = len(rows)
vids = sorted({r["video"] for r in rows})
print(f"tracks with a visible box: {n} across {len(vids)} videos")


def arr(k, R=rows):
    return np.array([r[k] for r in R], float)


def within_video_ranks(R, k):
    """Rank k inside each video, scaled to (0,1], so between-video effects vanish."""
    out = np.empty(len(R))
    for v in {r["video"] for r in R}:
        idx = [i for i, r in enumerate(R) if r["video"] == v]
        rk = stats.rankdata([R[i][k] for i in idx])
        for j, i in enumerate(idx):
            out[i] = rk[j] / len(idx)
    return out


def rho_pooled(R, a, b):
    return stats.spearmanr([r[a] for r in R], [r[b] for r in R])[0]


def rho_within(R, a, b):
    multi = [r for r in R if sum(x["video"] == r["video"] for x in R) >= 3]
    return stats.spearmanr(within_video_ranks(multi, a), within_video_ranks(multi, b))[0]


def cluster_boot(fn, B=2000, seed=0):
    rng = random.Random(seed)
    byv = {v: [r for r in rows if r["video"] == v] for v in vids}
    vals = []
    for _ in range(B):
        R = [r for v in (rng.choice(vids) for _ in vids) for r in byv[v]]
        try:
            vals.append(fn(R))
        except Exception:
            pass
    vals = np.array([x for x in vals if np.isfinite(x)])
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


res = {"n_tracks": n, "n_videos": len(vids)}

# ---------------------------------------------------------------- T1 duration vs range
print("\nT1  track duration against range proxy (small box = far)")
for key in ("size_med", "size_first"):
    x = arr(key)
    q = np.quantile(x, [0.25, 0.5, 0.75])
    bins = np.digitize(x, q)
    print(f"\n  quartiles of {key}: cut points {q.round(1).tolist()} px")
    print(f"  {'quartile':<22s}{'n':>4s}{'median frames':>15s}{'IQR':>16s}{'median s':>10s}")
    qrows = []
    for b in range(4):
        d = arr("dur")[bins == b]
        lab = ["Q1 smallest (farthest)", "Q2", "Q3", "Q4 largest (nearest)"][b]
        iqr = np.percentile(d, [25, 75])
        print(f"  {lab:<22s}{len(d):4d}{np.median(d):15.0f}"
              f"{f'{iqr[0]:.0f}-{iqr[1]:.0f}':>16s}{np.median(d)/60:10.1f}")
        qrows.append(dict(quartile=lab, n=int(len(d)), median_frames=float(np.median(d)),
                          q25=float(iqr[0]), q75=float(iqr[1])))
    kw = stats.kruskal(*[arr("dur")[bins == b] for b in range(4)])
    rp = rho_pooled(rows, key, "dur")
    rw = rho_within(rows, key, "dur")
    cp = cluster_boot(lambda R: rho_pooled(R, key, "dur"))
    cw = cluster_boot(lambda R: rho_within(R, key, "dur"))
    print(f"  Kruskal-Wallis H={kw.statistic:.2f} p={kw.pvalue:.3g}")
    print(f"  Spearman(size, duration) pooled {rp:+.3f} [{cp[0]:+.3f},{cp[1]:+.3f}]"
          f"   within-video {rw:+.3f} [{cw[0]:+.3f},{cw[1]:+.3f}]  (95% video bootstrap)")
    print("  (positive rho = larger/nearer animals have longer tracks = duration falls with range)")
    res[f"T1_{key}"] = dict(cuts=q.tolist(), quartiles=qrows, kw_H=kw.statistic,
                            kw_p=kw.pvalue, rho_pooled=rp, rho_pooled_ci=cp,
                            rho_within=rw, rho_within_ci=cw)

# short-track share by quartile: are distant animals disproportionately brief?
x = arr("size_med"); q = np.quantile(x, [0.25, 0.5, 0.75]); bins = np.digitize(x, q)
print("\n  share of tracks shorter than 1 s (60 frames) by size_med quartile:",
      [f"{np.mean(arr('dur')[bins == b] < 60):.2f}" for b in range(4)])
res["T1_short_share"] = [float(np.mean(arr("dur")[bins == b] < 60)) for b in range(4)]

# ---------------------------------------------------------------- T2 formation vs range
# Annotated boxes keep one size for most of a track (checked below), so size_max is not a
# closest-approach distance and a perpendicular-distance histogram is not interpretable.
# Test in size space instead: with s ~ 1/range and uniform ground density inside the
# camera's wedge, animals at range r..r+dr scale as r dr, i.e. n(s) ds ~ s^-3 ds. Anchor
# that law on 30-60 px (far above any detection floor) and compare the smaller bins.
print("\nT2  animals per size bin against geometric availability (n(s) ~ s^-3)")
s_ = arr("size_med")
edges = [0, 15, 20, 25, 30, 40, 60, 100, 1e9]
cnt, _ = np.histogram(s_, edges)
eb = lambda a, b: (a ** -2 - b ** -2) / 2
k = (cnt[4] + cnt[5]) / (eb(30, 40) + eb(40, 60))
t2 = []
for i in range(len(cnt)):
    lo, hi = edges[i], edges[i + 1]
    e = k * eb(lo, hi) if lo >= 15 else float("nan")
    lab = f"{lo:g}-{hi:g}" if hi < 1e8 else f">={lo:g}"
    print(f"  {lab:>8s} px: observed {cnt[i]:3d}   expected {e:6.0f}")
    t2.append(dict(bin=lab, observed=int(cnt[i]), expected=e))
print(f"  anchor check 30-40 / 40-60: observed {cnt[4]/cnt[5]:.2f}, "
      f"predicted {eb(30,40)/eb(40,60):.2f}")
res["T2"] = dict(bins=t2, anchor_obs=cnt[4] / cnt[5], anchor_pred=eb(30, 40) / eb(40, 60))

# post hoc: the farthest quartile against the rest
q1 = np.quantile(s_, 0.25); dd = arr("dur")
mw = stats.mannwhitneyu(dd[s_ < q1], dd[s_ >= q1], alternative="less")
mr = np.median(dd[s_ < q1]) / np.median(dd[s_ >= q1])
ci = cluster_boot(lambda R: np.median([r["dur"] for r in R if r["size_med"] < q1]) /
                  np.median([r["dur"] for r in R if r["size_med"] >= q1]), B=4000, seed=1)
print(f"\n  post hoc Q1 vs Q2-Q4 duration: median ratio {mr:.2f} [{ci[0]:.2f},{ci[1]:.2f}], "
      f"Mann-Whitney one-sided p={mw.pvalue:.3f}")
res["T1_q1_vs_rest"] = dict(ratio=mr, ci=ci, mwu_p=mw.pvalue)

# within-track size constancy (why size_max cannot stand in for closest approach)
rr = arr("size_max") / arr("size_min")
print(f"  within-track max/min size: median {np.median(rr):.2f}; "
      f"constant (<1.1) in {np.mean(rr < 1.1):.2f} of tracks")
res["size_constancy"] = dict(median_ratio=float(np.median(rr)), share_const=float(np.mean(rr < 1.1)))

# ---------------------------------------------------------------- T3 where tracks start
print("\nT3  size at first sighting")
fs, mn, mx = arr("size_first"), arr("size_min"), arr("size_max")
print(f"  first-sighting sqrt(area): median {np.median(fs):.1f} px, "
      f"IQR {np.percentile(fs,25):.1f}-{np.percentile(fs,75):.1f}, "
      f"min {fs.min():.1f}, 5th pct {np.percentile(fs,5):.1f}")
print(f"  smallest box anywhere in any track: {mn.min():.1f} px; "
      f"5th pct of per-track minima {np.percentile(mn,5):.1f}")
print(f"  tracks whose first box is also their smallest: {np.mean(arr('first_is_min')):.2f}")
print(f"  tracks that start touching the frame border: {np.mean(arr('first_at_edge')):.2f}")
print(f"  first size / max size, median: {np.median(fs / mx):.2f}")
res["T3"] = dict(first_median=float(np.median(fs)), first_p5=float(np.percentile(fs, 5)),
                 min_box=float(mn.min()), first_is_min=float(np.mean(arr("first_is_min"))),
                 first_at_edge=float(np.mean(arr("first_at_edge"))),
                 first_over_max=float(np.median(fs / mx)))

import csv
with open(os.path.join(OUT, "range_censoring_tracks.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(res, open(os.path.join(OUT, "range_censoring.json"), "w"), indent=1, default=float)
print("\nwrote", OUT)
