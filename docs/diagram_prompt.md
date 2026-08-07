# Prompt for generating the pipeline / architecture figure

Copy everything below the line into Gemini. Attach the three thumbnails listed at the end.

---

Create a **pipeline architecture diagram** in the style of a modern computer-vision paper:
three titled panels side by side, each a rounded light-grey box with its name in bold at the
top, real image thumbnails embedded at the input and output of each stage, and simple
rounded-rectangle blocks connected by thin arrows. Wide layout, about 4:1, to span two text
columns. Flat vector, no 3D, no shadows.

Colour code, shown in a small legend at the bottom right:
- **blue blocks** = trainable network, marked with a flame icon
- **grey blocks** = no learned weights, marked with a snowflake icon
- **pink block** = decision head

---

## Panel 1 — "Input"

- A thermal video frame thumbnail, low-contrast grey. Caption beneath in small grey text:
  `640 x 512, 60 fps`.
- Arrow to a grey block labelled `CLAHE`.
- Arrow to a second thumbnail of the same frame, visibly higher contrast.

## Panel 2 — "Detection and tracking"

- Arrow in from Panel 1 to a blue block labelled `YOLO11m`. Inside or beneath it, three tiny
  stacked sub-labels only: `backbone`, `neck`, `head`.
- Arrow out to a thumbnail of a frame with two green boxes on animals. Small grey caption:
  `per-frame boxes`.
- Arrow to a grey block labelled `BoT-SORT`, with two tiny sub-labels: `Kalman`,
  `motion compensation`.
- A small grey block labelled `appearance` sits below `BoT-SORT` with a **dashed** border
  and a dashed arrow up into it.
- Arrow out to a thumbnail showing three small frames in a row, the same animal boxed in the
  same colour in each, linked by a thin dotted line. Small grey caption: `candidate tracks`.

## Panel 3 — "Counting"

- Arrow in to a pink block labelled `rule`, fed from the left by three tiny nodes labelled
  `length`, `span`, `conf`.
- Two arrows out: one down to a small grey X, and one forward.
- The forward arrow ends at a large numeral, e.g. `5`, with a small deer glyph. Green caption
  beneath: `count`.

---

## Style

Short labels only — no sentences anywhere. Small grey annotations under thumbnails, as in
the reference. All text horizontal. No title, no caption text, no axis labels.

## Do not include

No training loop, loss, optimiser or dataset branch. No convolution-layer stacks or tensor
cuboids. Do not draw BoT-SORT in blue or with a flame; it has no learned weights. Do not add
blocks beyond those listed.

---

## Thumbnails to attach

1. `figures/dataset/p1_raw.jpg` — raw thermal frame, for Panel 1 input
2. `figures/dataset/p1_gt.jpg` — same frame with boxes, for Panel 2 detection output
3. `figures/qualitative/GolfDr_SHB_12.11.2025__f3652_5deer.jpg` — multiple tracked animals
   with coloured IDs, for Panel 2 tracking output

If the thumbnails cannot be used directly, draw simple grey placeholder rectangles with a
pale animal silhouette in the same positions.

---

# Correction prompt (round 2)

Paste this as a follow-up if the first output has the problems noted below.

---

Keep the layout, panels, blocks, arrows, colour code and legend exactly as they are. Change
only the image thumbnails and the track colours.

1. **The animals are far too large.** In the real data a deer occupies about 4% of the frame
   width — roughly 29 x 24 pixels inside a 640 x 512 frame. Redraw every thumbnail with the
   animal at that scale: a small pale shape, no more than one twentieth of the frame's width,
   positioned off-centre. It should look distant, not portrait-framed.

2. **The frames are not thermal.** Replace them with low-contrast greyscale imagery: a nearly
   uniform mid-grey field with faint horizontal banding, visible fine grain, no sky, no tree
   texture, no foliage detail. The animal is a slightly brighter blob with a barely
   discernible outline — not a rendered deer. The "before CLAHE" frame should be almost flat
   grey with the animal hard to see; the "after CLAHE" frame should be the same scene with
   the animal clearly brighter than the background.

3. **The three tracking frames use three different box colours.** In a tracker, colour
   denotes identity, so three colours reads as three different animals. Use **one colour for
   all three boxes**, since they show the same animal followed through time. Keep the dotted
   line joining them.

4. Replace the brown deer glyph beside the numeral `5` with a small flat grey silhouette, to
   match the thermal palette.

Change nothing else.
