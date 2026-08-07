# Prompt for generating the pipeline / architecture figure

Copy everything below the line into Gemini.

---

Create a **neural-network architecture diagram** for a computer-vision paper, in the style
used in object-detection papers: labelled blocks, arrows for tensor flow, feature-map sizes
on the connections. Flat vector, thin lines, no 3D, no shadows. Wide layout, spanning two
text columns.

Three modules, left to right, each in its own light bounding box.

---

## MODULE 1 — Detector (YOLO11m)

Input `640 x 512`. Draw the standard three-part layout:

- **Backbone**: a descending column of `Conv` and `C3k2` blocks, ending in `SPPF` then
  `C2PSA`. Tap three feature maps labelled `P3 80x64`, `P4 40x32`, `P5 20x16`.
- **Neck**: PAN-FPN beside it — a top-down pass of `Upsample`+`Concat`, then a bottom-up
  pass of `Conv`+`Concat`. Thin diagonal skip arrows from the backbone taps.
- **Head**: three small parallel heads, one per scale, merging into one `NMS` block.

Output arrow: **boxes**.

---

## MODULE 2 — Tracker (BoT-SORT)

Four plain blocks in a row. Square corners and grey fill, clearly different from the
detector's blocks, because this module has no learned weights.

`Kalman` → `Motion compensation` → `Association` → `Tracks`

One small block below `Association` labelled `Appearance`, drawn with a **dashed** border,
with an arrow up into it.

Output arrow: **tracks**.

---

## MODULE 3 — Confirmation

One block labelled `Rule`, fed by three tiny input nodes labelled `length`, `span`, `conf`.
Two arrows out: one forward, one short arrow to a small grey cross.

Output: a large numeral, e.g. `5`, labelled **count**.

---

## Evaluation overlay

A thin light-grey band beneath the three modules with three descending bars labelled
`detected`, `own track`, `counted`, each joined by a faint dotted line to the module above
it. No numbers.

---

## Style

Muted blue for the detector blocks, grey for the tracker, one warm accent for the rule and
the numeral. Feature sizes in small monospace on the arrows. All text horizontal, short
labels only — no sentences, no parameter values.

## Do not include

No training loop, loss, optimiser or dataset branch; inference only. Do not draw the tracker
as neural layers. Do not invent extra blocks, layer names or channel counts. No title, no
legend, no caption.
