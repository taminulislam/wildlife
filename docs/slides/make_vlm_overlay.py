"""Illustrative detection/segmentation overlay for the VLM proposal slide.

These boxes are hand-placed to illustrate a PROPOSED capability. No model produced
them -- our detector is trained on 640x512 thermal, not daylight RGB trail-camera
imagery, so running it here would be meaningless. The slide labels this a mockup.
"""
from PIL import Image, ImageDraw, ImageFont
import os

SRC = "/work/nvme/bgte/tislam6/wildlife_project/docs/slides/deer_03.JPG"
OUT = "/work/nvme/bgte/tislam6/wildlife_project/docs/slides/assets/vlm_scene.png"

im = Image.open(SRC).convert("RGB")
W, H = im.size
ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(ov)

def font(sz):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

F = font(30)

# (x1, y1, x2, y2, label, colour)
DEER = [
    (430,  545, 1015, 988, "deer 1  ~4 m",  (232, 122,  79)),
    ( 78,  478,  318,  700, "deer 2  ~14 m", (196,  69,  60)),
    (262,  442,  402,  628, "deer 3  ~16 m", (140,  45,  63)),
    (1588, 428, 1748,  640, "deer 4  ~19 m", ( 74,  27,  51)),
]

for x1, y1, x2, y2, lab, col in DEER:
    d.rectangle([x1, y1, x2, y2], outline=col + (255,), width=6)
    d.rectangle([x1 + 3, y1 + 3, x2 - 3, y2 - 3], fill=col + (46,))   # "mask" tint
    tw = d.textbbox((0, 0), lab, font=F)[2]
    ty = y1 - 42 if y1 > 46 else y2 + 4
    d.rectangle([x1, ty, x1 + tw + 20, ty + 40], fill=col + (235,))
    d.text((x1 + 10, ty + 4), lab, font=F, fill=(255, 255, 255, 255))

im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
im.save(OUT, quality=92)
print("wrote", OUT, im.size)
