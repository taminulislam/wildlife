"""Results deck for the WACV thermal-deer paper.

Every number here is copied from overleaf_WACV/, so the talk and the manuscript
cannot drift apart. Nothing is computed fresh.
"""
import os
from pptx import Presentation
from pptx.util import Inches as I, Pt, Emu
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

FIG = "/work/nvme/bgte/tislam6/wildlife_project/overleaf_WACV/figures"
AST = "/work/nvme/bgte/tislam6/wildlife_project/docs/slides/assets"
OUT = "/work/nvme/bgte/tislam6/wildlife_project/docs/slides/wildlife_results.pptx"

PAPER  = C(0xFA, 0xF8, 0xF7)
INK    = C(0x24, 0x1C, 0x1E)
MUTE   = C(0x7A, 0x6E, 0x70)
RULE   = C(0xE4, 0xDD, 0xDE)
MAROON = C(0x8C, 0x2D, 0x3F)
DEEP   = C(0x4A, 0x1B, 0x33)
EMBER  = C(0xC4, 0x45, 0x3C)

prs = Presentation()
prs.slide_width, prs.slide_height = I(13.333), I(7.5)
BLANK = prs.slide_layouts[6]
W, H = 13.333, 7.5
M = 0.72                       # page margin


def slide(bg=PAPER):
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    r.fill.solid(); r.fill.fore_color.rgb = bg; r.line.fill.background()
    r.shadow.inherit = False
    return s


def text(s, x, y, w, h, body, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
         spacing=1.0, font="Calibri", anchor=MSO_ANCHOR.TOP, gap=0):
    tb = s.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, line in enumerate(body if isinstance(body, list) else [body]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = spacing
        if i and gap:
            p.space_before = Pt(gap)
        txt, sz, cl, bd = (line if isinstance(line, tuple) else (line, size, color, bold))
        run = p.add_run(); run.text = txt
        run.font.size = Pt(sz); run.font.color.rgb = cl
        run.font.bold = bd; run.font.name = font
    return tb


def bar(s, x, y, w, h, color):
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, I(x), I(y), I(w), I(h))
    r.fill.solid(); r.fill.fore_color.rgb = color
    r.line.fill.background(); r.shadow.inherit = False
    return r


def header(s, kicker, title):
    bar(s, M, 0.62, 0.055, 0.52, MAROON)
    text(s, M + 0.22, 0.60, 9.0, 0.3, kicker.upper(), size=11.5, color=MAROON, bold=True)
    text(s, M + 0.22, 0.86, 11.6, 0.5, title, size=25, bold=True)


def picture_fit(s, path, x, y, w, h):
    """Contain-fit, so nothing is ever stretched or cropped."""
    from PIL import Image
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return s.shapes.add_picture(path, I(x + (w - pw) / 2), I(y + (h - ph) / 2),
                                I(pw), I(ph))


def kpi(s, x, y, w, value, label, color=MAROON):
    bar(s, x, y, 0.045, 1.28, color)
    text(s, x + 0.19, y - 0.06, w, 0.62, value, size=38, bold=True, color=INK)
    text(s, x + 0.19, y + 0.62, w, 0.66, label, size=12.5, color=MUTE, spacing=1.15)


# ══════════════════════════════════════════════════ 1 — title
s = slide()
bar(s, 0, 0, W, 0.11, MAROON)
text(s, M, 2.05, 11.4, 0.34, "WACV 2027 SUBMISSION  ·  RESULTS TO DATE",
     size=13, color=MAROON, bold=True)
text(s, M, 2.55, 11.6, 1.9,
     "Counting Is Not Detecting", size=52, bold=True)
text(s, M, 3.62, 11.6, 0.9,
     "Localizing the bottleneck in thermal wildlife video", size=25, color=MUTE)
bar(s, M, 4.72, 3.0, 0.02, RULE)
text(s, M, 5.00, 11.6, 1.2,
     ["White-tailed deer detection and counting from vehicle-mounted FLIR thermal transects",
      ("32 videos  ·  521,930 frames  ·  236 individually tracked animals  ·  4 sites, southern Illinois",
       15, MUTE, False)],
     size=15, color=INK, spacing=1.35, gap=9)

# ══════════════════════════════════════════════════ 2 — pipeline
s = slide()
header(s, "Approach", "One detector, one tracker, a three-parameter confirmation rule")
picture_fit(s, f"{FIG}/wildlife_pipeline.png", M, 1.95, W - 2 * M, 2.5)
bar(s, M, 4.78, W - 2 * M, 0.02, RULE)
cw = (W - 2 * M - 1.0) / 3
for i, (t, b) in enumerate([
    ("Detection", "YOLO11m @ 640 px, chosen from a 12-architecture benchmark on the "
                  "criterion that matters downstream, not on mAP."),
    ("Association", "BoT-SORT with global motion compensation — mandatory, the vehicle "
                    "travels 25–90 km/h and the whole frame translates."),
    ("Confirmation", "Accept a track on length, span and top-5 mean confidence. Three "
                     "parameters, swept on training video and frozen."),
]):
    x = M + i * (cw + 0.5)
    text(s, x, 5.05, cw, 0.32, t, size=15.5, bold=True, color=MAROON)
    text(s, x, 5.44, cw, 1.5, b, size=12.5, color=INK, spacing=1.3)

# ══════════════════════════════════════════════════ 3 — headline numbers
s = slide()
header(s, "Results", "Detection is not the bottleneck")
for i, (v, l, c) in enumerate([
    ("98.8%", "of held-out animals\nare detected at all", DEEP),
    ("88.0%", "acquire their own\ndistinct track", MAROON),
    ("62.7%", "survive confirmation\nand are counted", EMBER),
    ("2.38", "mean absolute error,\nanimals per video", MUTE),
]):
    kpi(s, M + i * 3.06, 2.05, 2.7, v, l, c)
bar(s, M, 3.66, W - 2 * M, 0.02, RULE)
picture_fit(s, f"{AST}/funnel.png", M - 0.1, 3.95, 6.4, 3.05)
text(s, 7.15, 3.95, 5.45, 0.34, "What this says", size=15.5, bold=True, color=MAROON)
text(s, 7.15, 4.38, 5.45, 2.2,
     ["Nearly every animal is found. The loss is concentrated after detection — "
      "in association, and above all in the decision of which candidate tracks are real.",
      "A better backbone cannot recover the 36 points between detected and counted, "
      "because those animals were never missed by the detector in the first place."],
     size=13.5, spacing=1.28, gap=11)
text(s, 7.15, 6.72, 5.45, 0.3,
     "Held out: rule swept on 19 training videos, frozen, reported on 13 unseen.",
     size=11, color=MUTE)

# ══════════════════════════════════════════════════ 4 — qualitative
s = slide()
header(s, "Qualitative results", "Tracking holds through overlap and low contrast")
picture_fit(s, f"{FIG}/qualitative.png", M, 1.85, W - 2 * M, 3.55)
bar(s, M, 5.62, W - 2 * M, 0.02, RULE)
cw = (W - 2 * M - 1.0) / 3
for i, (t, b) in enumerate([
    ("Identities persist", "Top row is the same scene 0.3 s apart. Tracks keep their "
                           "identities while the camera translates beneath them."),
    ("Overlap is resolved", "Animals whose boxes intersect are still carried as separate "
                            "tracks — the failure that silently merges two deer into one."),
    ("Confidence is honest", "The per-track score falls with range, from a near animal at "
                             "87 px to a distant one at 22 px in a single frame."),
]):
    x = M + i * (cw + 0.5)
    text(s, x, 5.86, cw, 0.3, t, size=14.5, bold=True, color=MAROON)
    text(s, x, 6.22, cw, 1.0, b, size=12, color=INK, spacing=1.28)

# ══════════════════════════════════════════════════ 5 — the inversion
s = slide()
header(s, "Central finding", "Improving candidate generation makes counting worse")
picture_fit(s, f"{AST}/inversion.png", M - 0.1, 2.0, 6.5, 3.2)
text(s, M + 0.15, 5.42, 6.2, 0.9,
     "Four independent interventions each enlarged the candidate pool as designed. "
     "Every one of them counted fewer animals.",
     size=13, color=MUTE, spacing=1.3)
text(s, 7.35, 2.05, 5.25, 0.34, "Why", size=15.5, bold=True, color=MAROON)
text(s, 7.35, 2.48, 5.25, 2.2,
     "Each intervention adds far more false candidates than real ones — the richest pool "
     "holds 219 true tracks against 27,460 false. A rule tuned to survive that ratio has "
     "to be aggressive enough to reject real animals along with the noise.",
     size=14, spacing=1.32)
bar(s, 7.35, 4.72, 5.25, 0.02, RULE)
text(s, 7.35, 4.98, 5.25, 0.34, "And it resists learning", size=15.5, bold=True, color=MAROON)
text(s, 7.35, 5.41, 5.25, 1.8,
     "A transformer, gradient boosting and logistic regression all lose to the "
     "three-parameter rule under three separate protocols. Error grows with model "
     "capacity — the constraint is supervision, not architecture.",
     size=14, spacing=1.32)

# ══════════════════════════════════════════════════ 6 — takeaways
s = slide()
header(s, "Summary", "What we can claim today")
rows = [
    ("Corpus", "32 fully annotated thermal transects, 236 individually tracked deer, "
               "21,646 boxes. Released with the paper."),
    ("Detector", "12 architectures benchmarked under four matching criteria. YOLO11m @ 640 px, "
                 "0.897 precision, selected on the criterion the counter actually needs."),
    ("Preprocessing", "CLAHE normalization is worth more than any architectural choice we "
                      "tested — test mAP50 0.519 with it, 0.299 without."),
    ("Counting", "98.8% detected, 88.0% distinctly tracked, 62.7% counted on video the "
                 "detector never saw. MAE 2.38 animals per video, and the system under-counts "
                 "rather than over-counts."),
    ("Auditability", "Every counted animal carries a calibrated score and a contact sheet of "
                     "its own frames; 92.1% score above 0.80."),
]
y = 1.95
for t, b in rows:
    bar(s, M, y + 0.06, 0.035, 0.46, MAROON)
    text(s, M + 0.2, y, 2.15, 0.4, t, size=14.5, bold=True, color=MAROON)
    text(s, M + 2.5, y - 0.02, W - 2 * M - 2.5, 0.9, b, size=13.5, spacing=1.28)
    y += 0.94
bar(s, M, 6.72, W - 2 * M, 0.02, RULE)
text(s, M, 6.92, W - 2 * M, 0.34,
     "Manuscript drafted end to end; remaining work is length and final ablations.",
     size=12.5, color=MUTE)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
prs.save(OUT)
print("saved", OUT, os.path.getsize(OUT), "bytes,", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
