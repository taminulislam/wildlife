#!/usr/bin/env python3
"""
TRACT engine: one raw video in, one annotated video plus a count out.

This is the published pipeline, not a demo approximation: CLAHE contrast normalisation,
YOLO11m at 640 px, BoT-SORT with global motion compensation, orphan recovery, and the
frozen three-parameter confirmation rule (n >= 20 frames, span >= 0 s, top-5 mean
confidence >= 0.65). Changing those defaults changes the numbers away from the paper's.

Two passes over the video. The first tracks and builds per-track records; the rule is then
applied to whole tracks, which is what the paper does and what a per-frame pass cannot do.
The second pass draws, so the on-screen counter can rise at the frame where each track
first satisfies the rule rather than jumping at the end.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from thermal import enhance_contrast                                   # noqa: E402

DEFAULT_WEIGHTS = "/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt"
DEFAULT_TRACKER = os.path.join(_ROOT, "track", "botsort_deer.yaml")

# Distinct, high-contrast track colours (BGR). Deliberately not a gradient: adjacent IDs
# must be told apart at a glance on a grey thermal frame.
PALETTE = [(0, 200, 255), (255, 128, 0), (0, 220, 120), (200, 60, 255), (60, 200, 255),
           (255, 200, 0), (120, 120, 255), (0, 255, 200), (255, 100, 160), (140, 255, 60)]


@dataclass
class Config:
    weights: str = DEFAULT_WEIGHTS
    tracker: str = DEFAULT_TRACKER
    imgsz: int = 640
    device: str = "0"
    conf: float = 0.10          # detector threshold the counting pipeline runs at
    iou: float = 0.50
    contrast: str = "clahe"
    keep_orphans: bool = True
    orphan_gap: int = 30
    # frozen confirmation rule
    min_frames: int = 20
    min_span_s: float = 0.0
    min_topk_conf: float = 0.65
    topk: int = 5
    draw_unconfirmed: bool = True


@dataclass
class Track:
    tid: int
    obs: list = field(default_factory=list)     # (frame, conf, xc, yc, w, h)
    orphan: bool = False

    def n(self) -> int:
        return len(self.obs)

    def span_s(self, fps: float) -> float:
        return (self.obs[-1][0] - self.obs[0][0]) / max(fps, 1e-6)

    def topk_conf(self, k: int) -> float:
        c = sorted((o[1] for o in self.obs), reverse=True)[:k]
        return sum(c) / len(c) if c else 0.0

    def confirm_frame(self, cfg: Config, fps: float) -> int | None:
        """Earliest frame at which the whole-track rule is already satisfied.

        Reported so the on-screen counter rises where the evidence arrives. A track can
        satisfy the rule and never lose it, because n and top-k mean are monotone in
        added observations only for n; top-k can fall, so we take the first frame from
        which the rule holds through the end of the track.
        """
        confs, best = [], None
        for i, (fr, cf, *_ ) in enumerate(self.obs):
            confs.append(cf)
            if i + 1 < cfg.min_frames:
                continue
            top = sorted(confs, reverse=True)[:cfg.topk]
            if (sum(top) / len(top)) >= cfg.min_topk_conf and \
               (fr - self.obs[0][0]) / max(fps, 1e-6) >= cfg.min_span_s:
                if best is None:
                    best = fr
            else:
                best = None                     # rule lapsed; require it to hold to the end
        return best

    def confirmed(self, cfg: Config, fps: float) -> bool:
        return (self.n() >= cfg.min_frames
                and self.span_s(fps) >= cfg.min_span_s
                and self.topk_conf(cfg.topk) >= cfg.min_topk_conf)


def video_meta(path: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def _link_orphans(orphans: list, cfg: Config, start_id: int) -> dict[int, Track]:
    """Detections the tracker never gave an ID, grouped into pseudo-tracks.

    BoT-SORT activates a track only after association across two frames, so an animal
    seen in isolated frames is dropped entirely. Recovering these is worth 17 animals in
    the paper's ablation.
    """
    out: dict[int, Track] = {}
    used = [False] * len(orphans)
    nid = start_id
    for i, o in enumerate(orphans):
        if used[i]:
            continue
        used[i] = True
        group, claimed = [o], {o[0]}
        for j in range(i + 1, len(orphans)):
            if used[j]:
                continue
            p = orphans[j]
            if p[0] in claimed or p[0] - group[-1][0] > cfg.orphan_gap:
                continue
            scale = max(group[-1][4], group[-1][5], 1.0)
            if abs(p[2] - group[-1][2]) < 3 * scale and abs(p[3] - group[-1][3]) < 3 * scale:
                used[j] = True
                group.append(p)
                claimed.add(p[0])
        t = Track(nid, group, orphan=True)
        out[nid] = t
        nid += 1
    return out


def analyse(video: str, cfg: Config, progress: Callable[[str, int, int], None] | None = None):
    """Pass 1: detect and track. -> (tracks, fps, n_frames, w, h)"""
    from ultralytics import YOLO
    fps, total, W, H = video_meta(video)
    model = YOLO(cfg.weights)

    tracks: dict[int, list] = defaultdict(list)
    orphans: list = []
    cap = cv2.VideoCapture(video)
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if progress and fi % 25 == 0:
            progress("Detecting and tracking", fi, total)
        frame = enhance_contrast(frame, method=cfg.contrast)
        r = model.track(source=frame, tracker=cfg.tracker, persist=True,
                        conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz,
                        device=cfg.device, verbose=False)[0]
        b = r.boxes
        if b is None or not len(b):
            continue
        ids = b.id.int().tolist() if b.id is not None else [None] * len(b)
        for tid, cf, (xc, yc, w, h) in zip(ids, b.conf.tolist(), b.xywh.tolist()):
            rec = (fi, float(cf), float(xc), float(yc), float(w), float(h))
            if tid is not None:
                tracks[int(tid)].append(rec)
            elif cfg.keep_orphans:
                orphans.append(rec)
    cap.release()

    out = {tid: Track(tid, obs) for tid, obs in tracks.items()}
    if orphans:
        out.update(_link_orphans(orphans, cfg, (max(out) if out else 0) + 1000))
    return out, fps, (total or fi + 1), W, H


# Ordered by browser playability, not by quality. This OpenCV build has no H.264
# encoder, and an mp4v file will not play in Chrome or Safari -- it downloads as a
# broken video, which looks like a pipeline failure to anyone using the interface.
# WebM/VP8 plays natively everywhere; mp4v is kept only as a last resort.
CODECS = [("VP80", ".webm"), ("vp09", ".webm"), ("avc1", ".mp4"), ("mp4v", ".mp4")]


def open_writer(out_base: str, fps: float, size: tuple[int, int]):
    """-> (writer, path). Tries codecs in order and returns the first that opens."""
    for fourcc, ext in CODECS:
        path = out_base + ext
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w, path
        w.release()
        if os.path.exists(path):
            os.remove(path)
    raise RuntimeError("no usable video encoder in this OpenCV build")


def render(video: str, out_base: str, tracks: dict[int, Track], cfg: Config, fps: float,
           total: int, progress: Callable[[str, int, int], None] | None = None) -> str:
    """Pass 2: draw boxes, identities and the running count. -> written path"""
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, out_path = open_writer(out_base, fps, (W, H))

    per_frame: dict[int, list] = defaultdict(list)
    confirm_at: dict[int, int] = {}
    for t in tracks.values():
        ok = t.confirmed(cfg, fps)
        cf = t.confirm_frame(cfg, fps) if ok else None
        if ok:
            confirm_at[t.tid] = cf if cf is not None else t.obs[-1][0]
        for (fr, cn, xc, yc, w, h) in t.obs:
            per_frame[fr].append((t.tid, cn, xc, yc, w, h, ok))

    events = sorted(confirm_at.values())
    fi, running = -1, 0
    ei = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if progress and fi % 25 == 0:
            progress("Rendering", fi, total)
        while ei < len(events) and events[ei] <= fi:
            running += 1
            ei += 1
        for (tid, cn, xc, yc, w, h, conf_ok) in per_frame.get(fi, []):
            if not conf_ok and not cfg.draw_unconfirmed:
                continue
            x1, y1 = int(xc - w / 2), int(yc - h / 2)
            x2, y2 = int(xc + w / 2), int(yc + h / 2)
            col = PALETTE[tid % len(PALETTE)]
            thick = 2 if conf_ok else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
            tag = f"#{tid} {cn:.2f}" + ("" if conf_ok else "  candidate")
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(frame, tag, (x1 + 3, max(9, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        _banner(frame, running, fi, fps, W)
        writer.write(frame)
    cap.release()
    writer.release()
    return out_path


def _banner(frame, running: int, fi: int, fps: float, W: int) -> None:
    bar = 34
    cv2.rectangle(frame, (0, 0), (W, bar), (0, 0, 0), -1)
    cv2.putText(frame, f"DEER COUNTED: {running}", (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 220, 120), 2, cv2.LINE_AA)
    t = fi / max(fps, 1e-6)
    stamp = f"t = {int(t // 60):d}:{t % 60:05.2f}   frame {fi}"
    (tw, _), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, stamp, (W - tw - 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)


def write_tracks_csv(path: str, tracks: dict[int, Track], cfg: Config, fps: float) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "source", "first_frame", "last_frame", "first_s", "last_s",
                    "n_frames", "span_s", "mean_conf", "topk_conf", "mean_box_px", "counted"])
        for t in sorted(tracks.values(), key=lambda x: x.obs[0][0]):
            confs = [o[1] for o in t.obs]
            areas = [o[4] * o[5] for o in t.obs]
            w.writerow([t.tid, "orphan" if t.orphan else "tracker",
                        t.obs[0][0], t.obs[-1][0],
                        round(t.obs[0][0] / fps, 2), round(t.obs[-1][0] / fps, 2),
                        t.n(), round(t.span_s(fps), 2),
                        round(sum(confs) / len(confs), 4), round(t.topk_conf(cfg.topk), 4),
                        round(sum(areas) / len(areas), 1),
                        int(t.confirmed(cfg, fps))])


def run(video: str, out_dir: str, cfg: Config | None = None,
        progress: Callable[[str, int, int], None] | None = None) -> dict:
    """Full pipeline. Returns a summary dict."""
    cfg = cfg or Config()
    if not os.path.isfile(cfg.weights):
        raise RuntimeError(f"detector weights not found: {cfg.weights}")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]

    tracks, fps, total, W, H = analyse(video, cfg, progress)
    counted = [t for t in tracks.values() if t.confirmed(cfg, fps)]

    out_video = render(video, os.path.join(out_dir, f"{stem}_counted"),
                       tracks, cfg, fps, total, progress)
    out_csv = os.path.join(out_dir, f"{stem}_tracks.csv")
    write_tracks_csv(out_csv, tracks, cfg, fps)

    if progress:
        progress("Done", total, total)
    return {"video": out_video, "csv": out_csv, "count": len(counted),
            "candidates": len(tracks), "frames": total, "fps": round(fps, 2),
            "width": W, "height": H, "source": os.path.basename(video),
            "rule": f"n>={cfg.min_frames}, span>={cfg.min_span_s}s, top{cfg.topk}>={cfg.min_topk_conf}"}
