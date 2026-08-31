#!/usr/bin/env python3
"""Three figures, one per stage of the decomposition: detection, tracking, counting.

Each is drawn from the same run so the stages are the same animals at successive stages,
not three unrelated illustrations.

  detection  per-frame boxes with the detector's own score, no identity -- what stage 2 emits
  tracking   the same scene at four instants, boxes coloured by track id -- identity surviving
             ego-motion, which is what stage 3 adds
  counting   every track the rule accepted, with its calibrated score and whether it is the
             animal's own track or a duplicate of one already counted -- stage 5's decision

Accept/reject comes from tracks.csv's `confirmed` flag, which is the three-parameter rule the
paper publishes. per_track_confidence.csv carries a *different* confirmer's decision in its
`counted` column and is used here only for the calibrated score.
"""
import csv, glob, os, colorsys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "common"))
from thermal import enhance_contrast

TR = "/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10/merged/tracks.csv"
SC = "results/temporal/calibrated_orphan/per_track_confidence.csv"
EV = "/work/hdd/bgte/tislam6/wildlife_outputs/viz/counting_evidence"
OUT = "overleaf_MDPI/figures"
FONT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSansCondensed-Bold.ttf"
fnt = lambda s: (ImageFont.truetype(FONT, s) if os.path.exists(FONT) else ImageFont.load_default())

TRACK_VID, TRACK_FRAMES = "NShelbyRd(blue)_SHB_12.11.2025_LS", [8494, 8512, 8538, 8544]
COUNT_VID = "GolfDr_SHB_12.11.2025"

def col(t):
    r, g, b = colorsys.hsv_to_rgb((int(t) * 0.61803398875) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)

def find_video(stem, root="data/raw"):
    for e in ("mp4", "MP4", "avi", "mov"):
        h = glob.glob(os.path.join(root, "**", f"{stem}.{e}"), recursive=True)
        if h: return sorted(h)[0]

rows = [r for r in csv.DictReader(open(TR))]
byfr = {}
for r in rows:
    if r["video"] == TRACK_VID and int(r["frame"]) in TRACK_FRAMES:
        byfr.setdefault(int(r["frame"]), []).append(r)

# decode the four frames once
cap = cv2.VideoCapture(find_video(TRACK_VID)); frames = {}; i = -1
while len(frames) < len(TRACK_FRAMES):
    ok, fr = cap.read()
    if not ok: break
    i += 1
    if i in TRACK_FRAMES: frames[i] = enhance_contrast(fr, method="clahe")
cap.release()
print(f"  decoded {len(frames)} frames from {TRACK_VID}")

def crop_box(dets, pad_l=26, pad_r=104, pad_t=36, pad_b=18, W=640, H=512):
    xs = [(float(d["xc"]) - float(d["w"]) / 2, float(d["xc"]) + float(d["w"]) / 2) for d in dets]
    ys = [(float(d["yc"]) - float(d["h"]) / 2, float(d["yc"]) + float(d["h"]) / 2) for d in dets]
    return (max(0, int(min(a for a, _ in xs) - pad_l)), max(0, int(min(a for a, _ in ys) - pad_t)),
            min(W, int(max(b for _, b in xs) + pad_r)), min(H, int(max(b for _, b in ys) + pad_b)))

def draw(frame, dets, mode):
    im = frame.copy()
    for d in dets:
        x1 = int(float(d["xc"]) - float(d["w"]) / 2); y1 = int(float(d["yc"]) - float(d["h"]) / 2)
        x2 = int(float(d["xc"]) + float(d["w"]) / 2); y2 = int(float(d["yc"]) + float(d["h"]) / 2)
        if mode == "detect":
            c, lab = (60, 220, 90), f"{float(d['conf']):.2f}"      # one colour: no identity yet
        else:
            r, g, b = col(d["track_id"]); c, lab = (b, g, r), f"ID {d['track_id']}"
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 6
        cv2.rectangle(im, (x1, ty - th - 4), (x1 + tw + 6, ty + 3), (0, 0, 0), -1)
        cv2.putText(im, lab, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, c, 1, cv2.LINE_AA)
    return im

def to_pil(a): return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))

def tile(panels, labels, gap=12, lab=30, cols=None):
    cols = cols or len(panels)
    rows = [(panels[i:i+cols], labels[i:i+cols]) for i in range(0, len(panels), cols)]
    H = max(p.height for p in panels)
    roww = lambda ps: sum(p.width for p in ps) + gap * (len(ps) - 1)
    W = max(roww(ps) for ps, _ in rows)
    sh = Image.new("RGB", (W, len(rows) * (H + lab) + gap * (len(rows) - 1)), "white")
    d = ImageDraw.Draw(sh)
    for ri, (ps, ls) in enumerate(rows):
        x, y = (W - roww(ps)) // 2, ri * (H + lab + gap)
        for p, l in zip(ps, ls):
            sh.paste(p, (x, y)); d.rectangle([x, y, x + p.width - 1, y + H - 1], outline=(70, 70, 70))
            d.text((x + 2, y + H + 6), l, font=fnt(15), fill="black"); x += p.width + gap
    return sh

# ---- 1. DETECTION -------------------------------------------------------------
f = TRACK_FRAMES[2]
dets = byfr[f]
x0, y0, x1, y1 = crop_box(dets)
p = to_pil(draw(frames[f], dets, "detect")).crop((x0, y0, x1, y1))
tile([p], [f"{len(dets)} detections in one frame, labelled with the detector's score — no identity yet"]
     ).save(f"{OUT}/stage1_detection.png")
print(f"  -> stage1_detection.png  {p.size}")

# ---- 2. TRACKING --------------------------------------------------------------
ps, ls = [], []
for fr in TRACK_FRAMES:
    dts = byfr[fr]
    a, b, c2, d2 = crop_box(dts)
    ps.append(to_pil(draw(frames[fr], dts, "track")).crop((a, b, c2, d2)))
    ls.append(f"frame {fr}   t = {fr/60:.1f} s   —   {len({x['track_id'] for x in dts})} identities")
H = min(p.height for p in ps)
ps = [p.resize((int(p.width * H / p.height), H), Image.LANCZOS) for p in ps]
tile(ps, ls, cols=2).save(f"{OUT}/stage2_tracking.png")
print(f"  -> stage2_tracking.png   4 frames spanning {(TRACK_FRAMES[-1]-TRACK_FRAMES[0])/60:.1f} s")

# ---- 3. COUNTING --------------------------------------------------------------
conf_ids, kinds, score = set(), {}, {}
for r in rows:
    if r["video"] == COUNT_VID and r.get("confirmed", "1") == "1":
        conf_ids.add(int(r["track_id"]))
for r in csv.DictReader(open(SC)):
    if r["video"] == COUNT_VID:
        kinds[int(r["track_id"])] = r["kind"]; score[int(r["track_id"])] = float(r["confidence"])
sel = sorted(conf_ids, key=lambda t: (kinds.get(t) != "primary", -score.get(t, 0)))
tiles, labs = [], []
for t in sel:
    hit = glob.glob(f"{EV}/*/{COUNT_VID}/trk{t}_best.jpg")
    if not hit: continue
    im = Image.open(hit[0]).convert("RGB")
    dup = kinds.get(t) != "primary"
    bd = (214, 94, 0) if dup else (0, 140, 80)
    fr = Image.new("RGB", (im.width + 8, im.height + 8), bd); fr.paste(im, (4, 4))
    tiles.append(fr)
    labs.append(f"ID {t}   {score.get(t,0):.2f}   {'DUPLICATE' if dup else 'own track'}")
H = min(t.height for t in tiles)
tiles = [t.resize((int(t.width * H / t.height), H), Image.LANCZOS) for t in tiles]
sh = tile(tiles, labs, cols=5)
d = ImageDraw.Draw(sh)
nd = sum(1 for t in sel if kinds.get(t) != "primary")
print(f"  -> stage3_counting.png   {len(sel)} confirmed = {len(sel)-nd} own-track + {nd} duplicates "
      f"(GT 12, counted {len(sel)})")
sh.save(f"{OUT}/stage3_counting.png")
