#!/usr/bin/env python3
"""Crop chosen detection frames to the region holding animals, and tile them.

The rendered frames are mostly empty forest; at page width the animals become a few
millimetres of grey. Cropping to the union of the track boxes puts the animals at a size a
reader can actually assess, which is the whole point of a qualitative figure.

Right-hand padding is larger than left because each box's label is drawn from the box's left
edge and runs rightwards, so a symmetric crop guillotines the label of the rightmost animal.
"""
import csv, os
from PIL import Image, ImageDraw, ImageFont

TR = "/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10/merged/tracks.csv"
IDX = "docs/figure_ideas/candidates/index.csv"
FONT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSansCondensed-Bold.ttf"
f = lambda s: (ImageFont.truetype(FONT, s) if os.path.exists(FONT) else ImageFont.load_default())

PICKS = [(6,  "NShelbyRd(blue)_SHB_12.11.2025_LS",   8538, "(a)"),
         (7,  "GolfDr_SHB_12.11.2025",               3650, "(b)"),
         (9,  "NShelbyRd(blue)_SHB_12.11.2025_LS",   8506, "(c)"),
         (11, "NWolfCreek(orange)_SHB_12.11.2025_LS", 7271, "(d)")]
FRAME_H = 512                       # rendered frame; a 26 px status bar sits below it
PAD_L, PAD_R, PAD_TOP, PAD_BOT = 26, 104, 36, 18

want = {(v, fr) for _, v, fr, _ in PICKS}
boxes = {}
for r in csv.DictReader(open(TR)):
    k = (r["video"], int(r["frame"]))
    if k in want:
        xc, yc, w, h = (float(r[c]) for c in ("xc", "yc", "w", "h"))
        boxes.setdefault(k, []).append((xc - w/2, yc - h/2, xc + w/2, yc + h/2))
idx = {r["n"]: r for r in csv.DictReader(open(IDX))}

crops = []
for n, v, fr, tag in PICKS:
    im = Image.open(idx[str(n)]["path"]).convert("RGB").crop((0, 0, 640, FRAME_H))
    bb = boxes[(v, fr)]
    x0 = max(0, min(b[0] for b in bb) - PAD_L)
    x1 = min(640, max(b[2] for b in bb) + PAD_R)
    y0 = max(0, min(b[1] for b in bb) - PAD_TOP)
    y1 = min(FRAME_H, max(b[3] for b in bb) + PAD_BOT)
    crops.append((tag, v, fr, len(bb), im.crop((int(x0), int(y0), int(x1), int(y1)))))
    print(f"  {tag} {v[:26]:<26} f{fr}  {crops[-1][4].size}  {len(bb)} tracks")

H = max(c[4].height for c in crops)      # common height so animals compare at one scale
sc = [(t, v, fr, k, im.resize((int(im.width * H / im.height), H), Image.LANCZOS))
      for t, v, fr, k, im in crops]
GAP, LAB = 12, 30
rows = (sc[:2], sc[2:])
roww = lambda r: sum(im.width for *_, im in r) + GAP * (len(r) - 1)
Wt = max(roww(r) for r in rows)
sheet = Image.new("RGB", (Wt, 2 * (H + LAB) + GAP), "white")
d = ImageDraw.Draw(sheet)
for ri, row in enumerate(rows):
    x, y = (Wt - roww(row)) // 2, ri * (H + LAB + GAP)
    for tag, v, fr, k, im in row:
        sheet.paste(im, (x, y))
        d.rectangle([x, y, x + im.width - 1, y + H - 1], outline=(70, 70, 70), width=1)
        d.text((x + 2, y + H + 6),
               f"{tag} {v.split('_')[0][:22]}  f{fr}  —  {k} animals tracked",
               font=f(15), fill="black")
        x += im.width + GAP
out = "overleaf_MDPI/figures/qualitative_crops.png"
sheet.save(out)
print(f"\n-> {out}  {sheet.size}")
