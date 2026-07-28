#!/usr/bin/env python3
"""
CROSS-TRACK CONTEXT features — the fix for TTC v1's over-counting.

v1 classified each track in isolation and tied on MAE with the hand-tuned rule while
over-counting (+1.11 bias). The reason is structural: the `duplicate` label means
"another track already covers this deer", which is NOT a property of the track itself.
A primary fragment and a duplicate fragment are *identical* when viewed alone, so no
per-track model can separate them — it accepts both and over-counts.

These 8 features give the model the missing information, by describing each track
relative to the other candidate tracks it competes with in the same video:

    n_overlap        how many other tracks share frames with it
    max_iou_other    strongest spatial overlap with a temporally-overlapping track
    is_longest       1 if it has the most frames in its competing group
    is_most_conf     1 if it has the highest top-k confidence in its group
    len_rank         its length rank within the group (0 = longest)
    conf_rank        its confidence rank within the group (0 = most confident)
    group_size       size of the competing group
    frac_of_longest  n_frames / longest-in-group

"Competing group" = tracks that overlap it in TIME and are spatially close, i.e. the
tracks that could plausibly be the same animal. A true primary is typically the longest
/ most confident of its group; a fragment is not.
"""
from __future__ import annotations
import numpy as np

N_CTX = 8


def _time_overlap(a, b) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _box_overlap(a, b) -> float:
    """IoU of the two tracks' mean boxes (cheap proxy for 'same animal')."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compute(meta: list[dict], seqs: dict) -> np.ndarray:
    """-> (N, N_CTX) float32, aligned with `meta`.

    meta entries need: video, track_id, n_frames, topk_conf, and seqs[(video,tid)]
    supplies the per-frame boxes.
    """
    by_video: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        by_video.setdefault(m["video"], []).append(i)

    # per track: (first_frame, last_frame) and mean box
    span, mbox = {}, {}
    for i, m in enumerate(meta):
        s = seqs.get((m["video"], m["track_id"]))
        if not s:
            span[i] = (0, 0); mbox[i] = (0.0, 0.0, 0.0, 0.0); continue
        frames = [o[0] for o in s]
        xs = [o[2] for o in s]; ys = [o[3] for o in s]
        ws = [o[4] for o in s]; hs = [o[5] for o in s]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        w, h = float(np.mean(ws)), float(np.mean(hs))
        span[i] = (min(frames), max(frames))
        mbox[i] = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    out = np.zeros((len(meta), N_CTX), dtype=np.float32)
    for _v, idxs in by_video.items():
        for i in idxs:
            group = [j for j in idxs
                     if j != i and _time_overlap(span[i], span[j])
                     and _box_overlap(mbox[i], mbox[j]) > 0.0]
            n_ov = len(group)
            max_iou = max((_box_overlap(mbox[i], mbox[j]) for j in group), default=0.0)
            members = group + [i]
            lens = {j: meta[j]["n_frames"] for j in members}
            confs = {j: meta[j]["topk_conf"] for j in members}
            len_sorted = sorted(members, key=lambda j: -lens[j])
            conf_sorted = sorted(members, key=lambda j: -confs[j])
            longest = lens[len_sorted[0]] or 1
            out[i] = [
                min(n_ov / 5.0, 2.0),
                max_iou,
                float(len_sorted[0] == i),
                float(conf_sorted[0] == i),
                min(len_sorted.index(i) / 5.0, 2.0),
                min(conf_sorted.index(i) / 5.0, 2.0),
                min(len(members) / 5.0, 2.0),
                lens[i] / longest,
            ]
    return out
