"""Step-by-step pipeline images for the lab talk, drawn from one real held-out scene.

Scene: GolfDr_SHB_12.11.2025, frame 3652 (a video the detector never trained on).
Boxes, confidences and track identities are read from the pipeline's own output
(counts/phaseC_orphan_yolo11m_conf0.10/merged/tracks.csv); nothing is drawn by hand.
The acceptance shown in step 5 is the published rule: >= 20 detections and mean of the
five highest confidences >= 0.65.
"""
import sys, os, csv, collections
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, 'src/dataset'); from cvat_to_yolo import find_video

VIDEO, F0 = 'GolfDr_SHB_12.11.2025', 3652
TRACK_FRAMES = [3652, 3661, 3670]
COUNTS = '/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10/merged/tracks.csv'
OUT, S = sys.argv[1], 2
CROP = (50, 70, 640, 365)   # x1, y1, x2, y2 in native pixels: the band the animals occupy (2:1)
PAL = {0: (242, 145, 27), 1: (79, 155, 224), 2: (111, 207, 151), 3: (235, 87, 87), 4: (187, 134, 252)}

def font(sz):
    for p in ('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

rows = collections.defaultdict(list)
for r in csv.DictReader(open(COUNTS)):
    if r['video'] == VIDEO:
        rows[int(r['track_id'])].append((int(r['frame']), float(r['conf']), float(r['xc']), float(r['yc']), float(r['w']), float(r['h'])))
for k in rows: rows[k].sort()
on = sorted(t for t, v in rows.items() if any(f == F0 for f, *_ in v))
colour = {t: PAL[i % 5] for i, t in enumerate(on)}
stat = {t: dict(n=len(rows[t]), top5=float(np.mean(sorted(c for _, c, *_ in rows[t])[-5:]))) for t in on}
for t in on: stat[t]['ok'] = stat[t]['n'] >= 20 and stat[t]['top5'] >= 0.65

need = sorted(set([F0] + TRACK_FRAMES))
cap = cv2.VideoCapture(find_video('data/raw', VIDEO)); frames = {}; i = 0
while i <= max(need):
    ok, img = cap.read()
    if not ok: break
    if i in need: frames[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    i += 1
cap.release()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def base(gray):
    g = cv2.resize(gray, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(g).convert('RGB')

PLACED = []
def label(d, x, y, text, fill, fg=(18, 22, 28), sz=22, below=None):
    """Draw a label above the box; if it would cover a label already drawn, put it below the box."""
    f = font(sz); w = d.textlength(text, font=f); h = sz + 8
    y0 = max(y - h - 2, 2)
    r = [x, y0, x + w + 12, y0 + h]
    if below is not None and any(not (r[2] < q[0] or q[2] < r[0] or r[3] < q[1] or q[3] < r[1]) for q in PLACED):
        r = [x, below + 2, x + w + 12, below + 2 + h]
    PLACED.append(r)
    d.rectangle(r, fill=fill); d.text((r[0] + 6, r[1] + 3), text, font=f, fill=fg)

def save(im, name):
    PLACED.clear()
    im.crop([c * S for c in CROP]).save(f'{OUT}/{name}')

def boxes_at(f):
    out = []
    for t in on:
        for fr, c, xc, yc, w, h in rows[t]:
            if fr == f: out.append((t, c, (xc - w / 2) * S, (yc - h / 2) * S, (xc + w / 2) * S, (yc + h / 2) * S))
    return out

save(base(frames[F0]), 'step1_raw.png')
save(base(clahe.apply(frames[F0])), 'step2_clahe.png')

im = base(clahe.apply(frames[F0])); d = ImageDraw.Draw(im)
for t, c, x1, y1, x2, y2 in boxes_at(F0):
    d.rectangle([x1, y1, x2, y2], outline=(0, 235, 0), width=4); label(d, x1, y1, f'deer {c:.2f}', (0, 235, 0), below=y2)
save(im, 'step3_detect.png')

for k, f in enumerate(TRACK_FRAMES):
    im = base(clahe.apply(frames[f])); d = ImageDraw.Draw(im)
    for t in on:
        pts = [(xc * S, yc * S) for fr, c, xc, yc, w, h in rows[t] if fr <= f]
        if len(pts) > 1: d.line(pts[-40:], fill=colour[t], width=3)
    for t, c, x1, y1, x2, y2 in boxes_at(f):
        d.rectangle([x1, y1, x2, y2], outline=colour[t], width=4); label(d, x1, y1, f'ID {t}', colour[t], below=y2)
    label(d, CROP[0] * S + 14, CROP[1] * S + 52, f't = {f/60:.2f} s', (18, 22, 28), fg=(255, 255, 255), sz=26)
    save(im, f'step4_track_{k+1}.png')

im = base(clahe.apply(frames[F0])); d = ImageDraw.Draw(im)
for t, c, x1, y1, x2, y2 in boxes_at(F0):
    if stat[t]['ok']:
        d.rectangle([x1, y1, x2, y2], outline=(0, 235, 0), width=4); label(d, x1, y1, 'counted', (0, 235, 0), below=y2)
    else:
        d.rectangle([x1, y1, x2, y2], outline=(235, 87, 87), width=4); label(d, x1, y1, 'rejected', (235, 87, 87), fg=(255, 255, 255), below=y2)
save(im, 'step5_confirm.png')

with open(f'{OUT}/scene_tracks.csv', 'w') as fh:
    fh.write('track_id,n_detections,top5_mean_conf,accepted\n')
    for t in on: fh.write(f"{t},{stat[t]['n']},{stat[t]['top5']:.3f},{int(stat[t]['ok'])}\n")
print(open(f'{OUT}/scene_tracks.csv').read()); print('frames present on each track frame:', {f: len(boxes_at(f)) for f in TRACK_FRAMES})
