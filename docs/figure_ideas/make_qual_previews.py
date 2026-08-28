#!/usr/bin/env python3
"""Draft composites for the qualitative figure ideas, built from frames already rendered in
wildlife_outputs/viz -- no video decoding, so these run on a login node in seconds."""
from __future__ import annotations
import glob, os, re
from PIL import Image, ImageDraw, ImageFont

V = "/work/hdd/bgte/tislam6/wildlife_outputs/viz/missed_deer"
OUT = "docs/figure_ideas/qualitative"
os.makedirs(OUT, exist_ok=True)
FONT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSansCondensed-Bold.ttf"
f = lambda s: (ImageFont.truetype(FONT, s) if os.path.exists(FONT) else ImageFont.load_default())


def parse(p):
    """m02_undetected_sz149_Video_deer6.jpg -> (2, 'undetected', 149)"""
    m = re.match(r"m(\d+)_(\w+?)_sz(\d+)_", os.path.basename(p))
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None


def middle(p):
    """each render is three panels (first / mid / last); the middle one is the cleanest."""
    im = Image.open(p).convert("RGB")
    w, h = im.size
    return im.crop((w // 3, 0, 2 * w // 3, h))


ITEMS = sorted((parse(p) + (p,) for p in glob.glob(f"{V}/*.jpg") if parse(p)),
               key=lambda t: -t[2])


def sample_a():
    """The three oversized misses beside a median animal, at TRUE relative scale."""
    picks = [i for i in ITEMS if i[2] >= 100][:3]
    small = [i for i in ITEMS if 24 <= i[2] <= 30][:1]
    sel = picks + small
    SCALE = 2.0                      # px of canvas per px of animal
    tiles = []
    for _n, state, sz, p in sel:
        im = middle(p)
        side = int(sz * SCALE)
        tiles.append((im.resize((side, side), Image.LANCZOS), state, sz))
    pad, lab = 18, 34
    W = sum(t[0].width for t in tiles) + pad * (len(tiles) + 1)
    H = max(t[0].height for t in tiles) + pad * 2 + lab
    c = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(c)
    x = pad
    for im, state, sz in tiles:
        y = pad + (H - pad * 2 - lab - im.height)
        c.paste(im, (x, y))
        d.rectangle([x, y, x + im.width - 1, y + im.height - 1],
                    outline=(213, 94, 0) if state != "counted" else (0, 158, 115), width=2)
        d.text((x, H - lab + 4), f"{sz} px  {state}", font=f(15), fill="black")
        x += im.width + pad
    d.text((pad, 4), "Idea C — the oversized misses, at true relative scale",
           font=f(16), fill="black")
    c.save(f"{OUT}/ideaC_oversized_misses.png"); print("  ideaC_oversized_misses.png")


def sample_b():
    """Every missed animal, size-ordered, as one contact sheet."""
    COLS, TH = 8, 96
    tiles = [(middle(p).resize((TH, TH), Image.LANCZOS), st, sz) for _n, st, sz, p in ITEMS]
    rows = (len(tiles) + COLS - 1) // COLS
    pad, lab = 6, 16
    c = Image.new("RGB", (COLS * (TH + pad) + pad, rows * (TH + pad + lab) + pad + 26), "white")
    d = ImageDraw.Draw(c)
    d.text((pad, 6), f"Idea B — all {len(tiles)} animals the pipeline failed to count, "
                     "largest to smallest", font=f(15), fill="black")
    for i, (im, st, sz) in enumerate(tiles):
        r, col = divmod(i, COLS)
        x = pad + col * (TH + pad); y = 26 + pad + r * (TH + pad + lab)
        c.paste(im, (x, y))
        col_ = (213, 94, 0) if st == "undetected" else (230, 159, 0)
        d.rectangle([x, y, x + TH - 1, y + TH - 1], outline=col_, width=2)
        d.text((x, y + TH + 2), f"{sz}px {st[:4]}", font=f(11), fill=col_)
    c.save(f"{OUT}/ideaB_failure_gallery.png"); print("  ideaB_failure_gallery.png")


def sample_a2():
    """Scale ladder: one animal per size decade, true relative scale, on one baseline."""
    want, seen = [], set()
    for n, st, sz, p in ITEMS:
        b = sz // 12
        if b not in seen:
            seen.add(b); want.append((st, sz, p))
    want = want[:8]
    SCALE = 2.2; pad = 16; lab = 30
    tiles = [(middle(p).resize((int(sz*SCALE), int(sz*SCALE)), Image.LANCZOS), sz) for _s, sz, p in want]
    W = sum(t[0].width for t in tiles) + pad * (len(tiles) + 1)
    H = max(t[0].height for t in tiles) + pad * 2 + lab + 22
    c = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(c)
    d.text((pad, 4), "Idea A — the scale ladder: what 'median 27 px' actually looks like",
           font=f(16), fill="black")
    x = pad
    for im, sz in tiles:
        y = 22 + pad + (H - 22 - pad*2 - lab - im.height)
        c.paste(im, (x, y)); d.rectangle([x, y, x+im.width-1, y+im.height-1],
                                         outline=(120,120,120), width=1)
        d.text((x, H - lab + 6), f"{sz} px", font=f(14), fill="black")
        x += im.width + pad
    c.save(f"{OUT}/ideaA_scale_ladder.png"); print("  ideaA_scale_ladder.png")


print("qualitative drafts:")
for fn in (sample_a2, sample_b, sample_a):
    try: fn()
    except Exception as e: print(f"  [FAIL] {fn.__name__}: {e}")
