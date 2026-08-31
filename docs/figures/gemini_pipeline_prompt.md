# Gemini prompt — expand Figure 3 with the detector's internals

Attach `overleaf_MDPI/figures/wildlife_pipeline.png` to the prompt so it has the layout to
extend. Paste everything between the rules.

---

Here is a scientific pipeline figure from a computer-vision paper. **Redraw it in exactly the
same visual style, keeping the three-panel layout and every existing element, but expand the
middle panel so the neural network architecture is shown in full detail.**

**Keep unchanged:**
- The three rounded grey panels titled "Input", "Detection and tracking", "Counting", left to
  right, with the same soft grey fills and thin borders.
- The flat, clean, modern scientific-figure style: white background, rounded rectangles, thin
  grey arrows, sans-serif labels, no 3-D, no drop shadows, no gradients, no glow.
- The thermal imagery: a greyscale infrared frame of a deer in forest at left, the same frame
  brighter and higher-contrast after CLAHE, then a frame with green detection boxes.
- The BoT-SORT block with "Kalman" and "motion compensation" sub-boxes, the dashed
  "appearance" box connecting up into it, and the row of three small candidate-track thumbnails
  with coloured borders.
- The "Counting" panel: three inputs labelled length, span, conf feeding a pink rounded box
  labelled "rule", producing a large number "5" with a deer icon and a green "count" tag.
- The legend, and the emoji-style markers: flame = trainable, snowflake = frozen, gears = no
  learned weights.

**The one change — replace the small blue "YOLO11m" box that currently just says
backbone / neck / head with a full CNN architecture diagram, drawn in the classic
deep-learning-paper style:**

- Show the **backbone** as a horizontal sequence of 3-D rectangular feature-map slabs that get
  progressively **smaller in width and height and thicker in depth**, left to right: 640×640,
  then 320×320, 160×160, 80×80, 40×40, 20×20. Label the strides.
- Label the convolutional blocks between slabs as **C3k2**, and place **SPPF** and **C2PSA**
  blocks at the deepest end.
- Show the **PAN neck** as a second row beneath: upward diagonal arrows carrying deep features
  back to shallower resolutions, then downward arrows returning, forming the characteristic
  top-down / bottom-up lattice.
- Show **three decoupled detection heads** branching from the neck at the P3 (80×80), P4
  (40×40) and P5 (20×20) scales, each splitting into two short parallel branches for
  classification and box regression.
- Keep this whole expanded network inside a blue-tinted rounded container labelled
  **"YOLO11m · 20.1 M parameters"** with the flame marker, so it still reads as the one
  trainable component.
- Draw an arrow from the P3 head emphasised or annotated, since the targets are small objects
  detected mainly at the finest scale.

**Composition:** wide landscape, roughly 3.5:1, suitable for full page width in a journal
paper. The expanded network should occupy most of the middle panel; keep the input and
counting panels compact. Everything must fit without crowding — leave clear white space
between blocks and keep all arrows orthogonal or gently diagonal, never crossing.

**Colour:** blue for the trainable detector, neutral grey for the weightless tracker, pink for
the decision rule, green accents only for detection boxes and the count tag. Muted,
print-friendly, no saturated colours.

**Text:** every label must be spelled exactly as given and be crisply legible. Do not invent
extra labels, captions, titles or watermarks.

---

## Expect to fix the text afterwards

Image models garble small text in diagrams — this is the failure mode to plan for, not a
possibility. Treat the output as a **layout draft**, not a finished figure: generate it, decide
whether the arrangement works, then rebuild the final version in draw.io or Illustrator with
real text. A journal reviewer will notice a misspelled "C3k2" or a smudged "20.1 M" instantly,
and MDPI requires figures legible at print resolution.

`docs/figures/tract_pipeline.drawio` already holds the same content as editable vector shapes
if you would rather extend that than redraw.
