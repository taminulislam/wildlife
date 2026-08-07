# Prompt for generating the pipeline / architecture figure

Copy everything below the line into Gemini.

---

Create a **neural-network architecture diagram** for a computer-vision paper, in the style
used in object-detection papers: labelled blocks for network components, arrows for tensor
flow, feature-map sizes annotated on the connections. Flat vector, thin lines, no 3D, no
shadows. Wide layout to span two text columns of a double-column paper.

The system counts individual animals in thermal video. It has **three sequential modules**.
Draw them as three horizontally arranged groups, each inside its own light bounding box with
the module name on top, connected left to right by thick arrows.

---

## MODULE 1 — Detector (YOLO11m, anchor-free, single class)

Input: a thermal frame, `640 x 512 x 1`, replicated to 3 channels.

Draw the standard three-part detector layout:

**Backbone** — a descending column of blocks, each smaller than the last, showing
progressive downsampling:
- `Conv` stem, stride 2
- alternating `Conv` / `C3k2` blocks at strides 4, 8, 16, 32
- `SPPF` (spatial pyramid pooling, fast) near the bottom
- `C2PSA` (cross-stage partial with attention) as the final backbone block

Tap three feature maps out of the backbone, labelled **P3 (stride 8)**, **P4 (stride 16)**,
**P5 (stride 32)**, with sizes `80x64`, `40x32`, `20x16` written on the tap arrows.

**Neck** — a PAN-FPN drawn as the usual two-pass lattice beside the backbone:
- top-down pass: `Upsample` + `Concat` + `C3k2`, from P5 down to P3
- bottom-up pass: `Conv` stride 2 + `Concat` + `C3k2`, from P3 back up to P5
Draw the skip connections between backbone taps and neck nodes as thin diagonal arrows.

**Head** — three parallel **decoupled detection heads**, one per scale, each a small pair of
branches labelled `cls` and `reg (DFL)`. Anchor-free. Their outputs merge into a single
`NMS` block (IoU 0.50, confidence 0.10).

Module output, labelled on the outgoing arrow: **per-frame boxes**.

---

## MODULE 2 — Tracker (BoT-SORT)

This module is not a neural network; draw it as a **small dataflow graph of labelled
operational blocks**, visually distinct from the conv blocks — use rectangles with square
corners, or a different fill, so a reader does not mistake them for layers.

Blocks and connections:
- **Kalman filter** — constant-velocity state prediction per track
- **Global motion compensation** — sparse optical flow between consecutive frames, producing
  an affine camera-motion estimate that is applied to every predicted track state before
  matching. Draw this as a side block feeding into the Kalman prediction, and label it
  clearly; the camera is on a moving vehicle, so this is essential rather than optional.
- **Two-stage association** — draw as two sequential matching blocks: first matching
  high-confidence detections, then a second pass over the remaining low-confidence ones. Each
  uses an **IoU cost matrix** into **Hungarian matching**.
- **Appearance embedding** — a small optional block feeding an extra term into the cost
  matrix; draw it with a **dashed border** to mark it as ablated rather than always on.
- **Orphan recovery** — a side block taking detections that received no track identity and
  linking them into pseudo-tracks by temporal gap and scale-normalised distance. Its output
  merges back into the track set.

Module output, labelled on the outgoing arrow: **candidate tracks**.

---

## MODULE 3 — Confirmation

Draw as a single decision block fed by three small feature nodes computed per track:
- `track length` (number of detections)
- `span` (duration in seconds)
- `top-5 mean confidence`

These three feed a block labelled **threshold rule**, whose output is a binary
accept / reject per track. Show two outgoing arrows: accepted tracks going forward, rejected
tracks going to a small grey dead-end marker.

Final output: a **count**, drawn as a single large numeral.

---

## Evaluation overlay — keep subtle

Beneath the three modules, draw a thin light-grey band with three descending bars labelled
**detected**, **own track**, **counted**. Connect each with a faint dotted line up to the
module it measures: *detected* to Module 1, *own track* to Module 2, *counted* to Module 3.
No numbers. This band must read as a quiet secondary layer, not as part of the dataflow.

---

## Style

- Muted blue fills for the detector's learned blocks, grey for the tracker's operational
  blocks, a single warm accent only for the confirmation decision and the output numeral.
- Feature-map sizes in small monospace type on the arrows; block names in small sans-serif
  inside the blocks.
- All text horizontal. No title, no legend, no caption.

## Do not include

No training loop, no loss functions, no optimiser, no dataset or augmentation branch — this
is inference only. Do not invent layer names, channel counts, or extra stages beyond those
listed. Do not draw the tracker's blocks as neural layers; it contains no learned weights
apart from the optional dashed appearance block.
