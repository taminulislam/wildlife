#!/usr/bin/env python3
"""
Figure 7: tracking on held-out video, 2 rows x 3 columns.

Six panels from four held-out transects. The first two are one scene 0.3 s apart, so identity
persistence through ego-motion is visible rather than asserted; the other four are separate
scenes covering overlap, low contrast, dense groups and range spread.

The source renders bake a caption strip whose text overruns 640 px and is clipped at the
tile edge. We crop that strip off and redraw it, so nothing is truncated.

Usage:  python docs/figures/make_qualitative_grid.py --out docs/figures
"""
from __future__ import annotations
import argparse
import csv
import os

from PIL import Image, ImageDraw, ImageFont

SRC = "/work/hdd/bgte/tislam6/wildlife_outputs/viz/qual_candidates"
STRIP = 26          # height of the baked-in caption strip on each source render
BAR = 30            # height of the strip we draw instead
GUT = 8             # gutter between tiles

# (video, frame, short label). The first two are the SAME scene 0.3 s apart, which is the
# persistence pair the caption relies on; the rest are separate scenes.
#
# NShelbyRd f8538 was tried as a third frame of that scene and rejected: checked against the
# track table, 3035 ends at frame 8535 and 3041 at 8512, so a three-frame row would show
# identity turnover while appearing to show persistence. Only 3040 spans all three.
PANELS = [
    ("NShelbyRd(blue)_SHB_12.11.2025_LS", 8494, "NShelbyRd, SHB"),
    ("NShelbyRd(blue)_SHB_12.11.2025_LS", 8512, "NShelbyRd, SHB"),
    ("GolfDr_SHB_12.11.2025",             3652, "GolfDr, SHB"),
    ("Robinson_SHW_01.18.2026_LS",       14892, "Robinson, SHW"),
    ("NWolfCreek(orange)_SHB_12.11.2025_LS", 7257, "NWolfCreek, SHB"),
    ("GolfDr_SHB_12.11.2025",             3396, "GolfDr, SHB"),
]
NCOL = 3

FONT_DIRS = ("/usr/share/fonts/dejavu-sans-fonts", "/usr/share/fonts/truetype/dejavu")


def font(size: int):
    for d in FONT_DIRS:
        for name in ("DejaVuSansCondensed-Bold.ttf", "DejaVuSans-Bold.ttf"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def index() -> dict[tuple[str, int], dict]:
    with open(os.path.join(SRC, "index.csv")) as fh:
        return {(r["video"], int(r["frame"])): r for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--name", default="qualitative")
    args = ap.parse_args()

    idx = index()
    tiles = []
    for video, frame, short in PANELS:
        r = idx.get((video, frame))
        if r is None:
            raise SystemExit(f"no index row for {video} f{frame}")
        im = Image.open(r["path"]).convert("RGB")
        im = im.crop((0, 0, im.width, im.height - STRIP))      # drop the clipped strip
        tiles.append((im, short, float(r["t_s"]), int(r["n_tracked"]), int(r["n_counted"])))

    tw, th = tiles[0][0].size
    nrow = (len(tiles) + NCOL - 1) // NCOL
    W = NCOL * tw + (NCOL + 1) * GUT
    H = nrow * (th + BAR) + (nrow + 1) * GUT
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    f = font(15)

    for i, (im, short, t_s, ntr, nct) in enumerate(tiles):
        cx, cy = i % NCOL, i // NCOL
        x = GUT + cx * (tw + GUT)
        y = GUT + cy * (th + BAR + GUT)
        canvas.paste(im, (x, y))
        draw.rectangle([x, y + th, x + tw - 1, y + th + BAR - 1], fill=(24, 24, 24))
        label = f"{short}   t = {t_s:.1f} s   |   {ntr} tracked, {nct} counted"
        # Shrink only if this particular label would overrun its tile.
        ff = f
        for size in (15, 14, 13, 12):
            ff = font(size)
            if draw.textlength(label, font=ff) <= tw - 16:
                break
        draw.text((x + 8, y + th + (BAR - ff.size) // 2 - 1), label, font=ff, fill="white")

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f"{args.name}.png")
    canvas.save(p, quality=95)
    print(f"-> {p}  {canvas.size[0]}x{canvas.size[1]}  ({len(tiles)} panels, {nrow}x{NCOL})")


if __name__ == "__main__":
    main()
