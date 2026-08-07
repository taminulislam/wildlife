# Prompt for generating the pipeline / architecture figure

Copy everything below the line into Gemini.

---

Create a **minimal, modern architecture diagram** for a computer-vision paper. Wide and
short, roughly 4:1, to span two text columns. Almost no text — the figure should be readable
as a picture, with short labels only. Flat vector style, thin lines, generous white space,
no 3D, no shadows, no gradients.

## The visual story, left to right

Five blocks connected by thin arrows. Each block shows a small **picture of what that stage
produces**, with a one- or two-word label beneath it. The pictures matter more than the
labels.

**Block 1 — a stack of three overlapping video frames**, dark grey, slightly rotated like a
deck of cards, suggesting a video sequence. Inside the top frame, one small pale animal
shape. Label: *Thermal video*.

**Block 2 — the same single frame, now higher contrast**, the animal clearly brighter
against the background. An arrow from block 1. Label: *CLAHE*.

**Block 3 — the same frame with two thin green rectangles** drawn around animal shapes.
Label: *Detection*.

**Block 4 — three small frames side by side in a row**, each with a coloured box on the same
animal, the boxes linked left to right by a thin dotted line to show identity persisting
through time. Use two different colours for two different animals. Label: *Tracking*.

**Block 5 — a short vertical list of four small track icons**, two marked with a green
check and two with a grey cross, showing that some candidate tracks are accepted and some
rejected. Label: *Confirmation*.

**Final output — a single large numeral** to the right of the last arrow, e.g. a big
"5", with a very small deer glyph beside it. Label: *Count*.

## One extra element, kept subtle

Under blocks 3, 4 and 5, draw a thin horizontal band containing a simple **three-step
descending funnel or three descending bars**, shrinking left to right, with only three short
labels: *detected*, *own track*, *counted*. No numbers. Connect it to the blocks above with
two or three faint dotted lines. This band should be visually quiet — light grey, thin — so
it reads as a secondary layer beneath the pipeline rather than part of it.

## Colour and type

- One muted accent colour for the pipeline blocks. Green only for the detection boxes and
  the accept checks. Grey for everything secondary.
- The thermal frames should look like real thermal imagery: grey, low contrast, slightly
  grainy, animals as pale warm shapes. Not colourful, not rainbow-mapped.
- Sans-serif, small, all horizontal. Labels only — no sentences, no parameter values, no
  numbers anywhere except the single output numeral.

## Do not include

No neural-network layer stacks, no convolution blocks, no matrix or tensor illustrations.
No training loop, no loss, no dataset branch. No legend, no title, no caption text, no
arrows looping backwards. No photographic or realistic deer — simple pale silhouettes only.
