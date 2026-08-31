#!/usr/bin/env python3
"""Emit an editable draw.io file for the TRACT pipeline figure.

Generated rather than hand-drawn so the parameter values stay tied to their sources: the
tracker settings come from src/track/botsort_deer_recall.yaml, the rule from the frozen
sweep, the detector settings from the counting runs. Edit here and regenerate, or open the
.drawio and edit freely -- it is uncompressed XML.
"""
import html, os

W = lambda s: html.escape(s, quote=True)
CELLS, _id = [], [1]
# IDs are prefixed: draw.io reserves "0" for the model root and "1" for the default
# layer, and a generated cell colliding with either makes the editor drop the graph and
# render a blank page.
NEW = lambda: "c" + str(_id[0])

# Colour carries the paper's own claim: what is gradient-trained, what carries no weights,
# what is fitted by grid search, and what was ablated out. Nothing else is colour-coded.
LEARNED = "fillColor=#DCE9F7;strokeColor=#2E6DA4;"
FIXED   = "fillColor=#F2F2F2;strokeColor=#8C8C8C;"
FITTED  = "fillColor=#FBEBD2;strokeColor=#C8871B;"
ABLATED = "fillColor=#FAFAFA;strokeColor=#B0B0B0;dashed=1;"
EVAL    = "fillColor=#E6F4EA;strokeColor=#3C8C55;"
IO      = "fillColor=#EDE7F6;strokeColor=#6A4FA3;"

def node(x, y, w, h, label, style="", shape="rounded=1;arcSize=8;", font=9):
    i = NEW(); _id[0] += 1
    s = (f"{shape}whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
         f"fontSize={font};fontFamily=Helvetica;{style}")
    CELLS.append(f'<mxCell id="{i}" value="{W(label)}" style="{s}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i

def group(x, y, w, h, label, stroke="#8C8C8C", fill="none"):
    i = NEW(); _id[0] += 1
    s = (f"rounded=1;arcSize=4;whiteSpace=wrap;html=1;verticalAlign=top;align=left;"
         f"spacingLeft=8;spacingTop=4;fontSize=10;fontStyle=1;fontFamily=Helvetica;"
         f"dashed=1;dashPattern=6 4;fillColor={fill};strokeColor={stroke};")
    CELLS.append(f'<mxCell id="{i}" value="{W(label)}" style="{s}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i

def edge(a, b, label="", style="", dashed=False):
    i = NEW(); _id[0] += 1
    s = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=blockThin;endFill=1;"
         f"fontSize=8;fontFamily=Helvetica;strokeColor=#5A5A5A;{'dashed=1;' if dashed else ''}{style}")
    CELLS.append(f'<mxCell id="{i}" value="{W(label)}" style="{s}" edge="1" parent="1" '
                 f'source="{a}" target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>')

def note(x, y, w, h, text, font=8):
    node(x, y, w, h, text, "fillColor=none;strokeColor=none;fontColor=#5A5A5A;align=left;",
         shape="text;", font=font)

# ---------------------------------------------------------------- stage 0/1: input
group(30, 40, 300, 150, "0 — INPUT")
v  = node(50, 72, 260, 46, "FLIR thermal transect\n640 × 512 × 1,  60 fps,  ~10.6 k frames", IO)
mo = node(50, 130, 260, 44, "vehicle 25–90 km/h ⇒ whole-frame ego-motion\n(handled in stage 2, not here)",
          "fillColor=none;strokeColor=none;fontColor=#6A4FA3;", shape="text;", font=8)

group(360, 40, 250, 150, "1 — PREPROCESS   (no weights)")
cl = node(380, 78, 210, 52, "CLAHE\nper-frame local histogram equalisation\n640×512×1 → 640×512×3", FIXED)
note(380, 138, 215, 34, "worth more than any architecture\ntested: 0.519 vs 0.299 test mAP50")

# ---------------------------------------------------------------- stage 2: detector
group(640, 40, 430, 150, "2 — DETECTION   (the only gradient-trained block)")
det = node(660, 74, 390, 50, "YOLO11m  @ 640 px  ·  20.1 M parameters\nC3k2 backbone → SPPF → C2PSA → PAN neck → decoupled head", LEARNED)
nms = node(660, 136, 180, 38, "NMS,  IoU 0.50", FIXED)
dts = node(866, 136, 184, 38, "Dt = {(x, y, w, h, s)}\nscore ≥ 0.10", IO)
edge(v, cl); edge(cl, det); edge(det, nms); edge(nms, dts)

# ---------------------------------------------------------------- stage 3: association
group(30, 230, 1040, 300, "3 — ASSOCIATION   BoT-SORT   (zero learned weights)")
kal = node(55, 274, 200, 56, "Kalman predict\n8-state (x, y, a, h, ẋ, ẏ, ȧ, ḣ)\nconstant-velocity motion model", FIXED)
gmc = node(285, 274, 210, 56, "Global motion compensation\nsparseOptFlow → affine warp\napplied to every track state", FIXED)
cst = node(525, 274, 200, 56, "cost matrix\nIoU fused with detection score\nmatch_thresh 0.80", FIXED)
hun = node(755, 274, 150, 56, "Hungarian\nassignment", FIXED)
byt = node(285, 356, 210, 54, "two-pass association\nhigh ≥ 0.15  then  low ≥ 0.05\n(recovers faint detections)", FIXED)
lif = node(525, 356, 200, 54, "track lifecycle\nnew_track_thresh 0.15\nbuffer 60 frames", FIXED)
emb = node(55, 356, 200, 54, "appearance embedding\nfrozen detector backbone\nwith_reid = False", ABLATED)
note(55, 418, 205, 40, "ablated in §4.4: best pool built\n(82/83 reached) yet 3 fewer counted")
orp = node(755, 356, 285, 54, "ORPHAN RECOVERY\nunmatched detections linked when Δt ≤ gap and\nscale-normalised distance ≤ τ  →  pseudo-tracks", FIXED)
note(755, 418, 285, 26, "+17 animals; without it single-frame animals are discarded")
cand = node(755, 462, 285, 40, "M candidate tracks\n(7 008 on the held-out set)", IO)

edge(dts, kal, "per frame"); edge(kal, gmc); edge(gmc, cst); edge(cst, hun)
edge(byt, cst, "", "exitX=0.5;exitY=0;entryX=0;entryY=1;"); edge(emb, cst, "", "", dashed=True)
edge(hun, lif, "matched", "exitX=0.5;exitY=1;entryX=1;entryY=0.5;")
edge(hun, orp, "unmatched", "exitX=1;exitY=1;entryX=0.5;entryY=0;")
edge(lif, cand, "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

# ---------------------------------------------------------------- stage 4/5: confirm
group(30, 570, 640, 200, "4–5 — TRACK FEATURES AND CONFIRMATION   (fitted by grid search, not trained)")
fea = node(55, 610, 260, 62, "per-track features\nn  = frames the track survives\ns  = seconds it spans\nc  = mean of its 5 highest detection scores", IO)
rul = node(345, 610, 300, 62, "CONFIRMATION RULE\naccept ⟺  n ≥ 20  ∧  s ≥ 0  ∧  c ≥ 0.65\n3 parameters, swept on the 19 training videos, frozen", FITTED)
alt = node(55, 692, 590, 46, "ablated alternatives — all lose to the rule (§4.7):  temporal transformer 6×10⁴ params  ·  gradient boosting 10³  ·  logistic regression 13", ABLATED)
edge(cand, fea, "", "exitX=0;exitY=1;entryX=0.5;entryY=0;")
edge(fea, rul); edge(alt, rul, "", "exitX=0.5;exitY=0;entryX=0.5;entryY=1;", dashed=True)

# ---------------------------------------------------------------- stage 6: output
group(700, 570, 370, 200, "6 — CALIBRATION AND OUTPUT")
cal = node(720, 610, 330, 46, "per-track posterior  P(track is a distinct animal)\nAUC 0.893 · 92.1 % of counted tracks ≥ 0.80", FITTED)
pb  = node(720, 668, 330, 40, "Poisson-binomial over accepted tracks\n⇒ count ± σ, not a point estimate", FITTED)
out = node(720, 718, 330, 40, "COUNT   +   per-animal evidence sheets   +\n793 low-confidence tracks flagged for review", IO)
edge(rul, cal, "accepted", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
edge(cal, pb); edge(pb, out)

# ---------------------------------------------------------------- evaluation instrument
group(1100, 40, 300, 730, "EVALUATION INSTRUMENT   (offline; not part of the runtime)", "#3C8C55")
gt  = node(1120, 80, 260, 52, "CVAT ground truth\none annotated track = one animal\n236 animals · 21 646 boxes", EVAL)
mt  = node(1120, 152, 260, 46, "matching criterion\nany-overlap (counting) vs IoU ≥ 0.50", EVAL)
r1  = node(1120, 224, 260, 44, "REACHED\ntouched by ≥ 1 candidate → detection", EVAL)
r2  = node(1120, 288, 260, 52, "PRIMARY\nits best cover is not also another animal's\n→ association", EVAL)
r3  = node(1120, 360, 260, 44, "COUNTED\nconfirmation accepted it → end to end", EVAL)
res = node(1120, 428, 260, 58, "held-out result\n98.8 % → 88.0 % → 62.7 %\nMAE 2.38 animals / video", EVAL)
note(1120, 500, 265, 60, "The gaps attribute error to a stage.\nReporting only the last makes a detection\nfailure and a confirmation failure look\nidentical.")
edge(gt, mt); edge(mt, r1); edge(r1, r2); edge(r2, r3); edge(r3, res)
edge(cand, r1, "candidates", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;", dashed=True)
edge(out, r3, "", "exitX=1;exitY=0.5;entryX=0;entryY=1;", dashed=True)

# ---------------------------------------------------------------- legend
group(1100, 590, 300, 180, "LEGEND")
node(1118, 622, 120, 26, "gradient-trained", LEARNED, font=8)
node(1250, 622, 130, 26, "no weights at all", FIXED, font=8)
node(1118, 656, 120, 26, "fitted, not trained", FITTED, font=8)
node(1250, 656, 130, 26, "ablated / unused", ABLATED, font=8)
node(1118, 690, 262, 26, "evaluation only — never touches the runtime", EVAL, font=8)
note(1118, 722, 265, 40, "One learned component in the whole system.\nThe contribution is the arrangement and the\nmeasurement, not the architecture.")

xml = ('<mxfile host="app.diagrams.net" type="device">\n'
       '  <diagram id="tract-pipeline" name="TRACT pipeline">\n'
       '    <mxGraphModel dx="1600" dy="1100" grid="1" gridSize="10" guides="1" tooltips="1" '
       'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1654" pageHeight="1169" '
       'math="0" shadow="0">\n      <root>\n'
       '        <mxCell id="0"/>\n        <mxCell id="1" parent="0"/>\n        '
       + "\n        ".join(CELLS) +
       '\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n')

out_path = "docs/figures/tract_pipeline.drawio"
open(out_path, "w").write(xml)
print(f"-> {out_path}  ({len(CELLS)} cells, {os.path.getsize(out_path)} bytes)")
