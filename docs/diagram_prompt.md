# Prompt for generating the pipeline / architecture figure

Copy everything below the line into Gemini. It is written to be self-contained: every
number, threshold and model name is stated, so the generator does not have to guess.

---

Create a clean, publication-quality **system architecture diagram** for a computer-vision
paper submitted to WACV. The figure will span two text columns of a double-column paper, so
design it **wide and short** — roughly a 3.5:1 aspect ratio. It must be legible when printed
at about 7 inches wide and 2 inches tall, in grayscale as well as colour.

## What the system does

It counts individual white-tailed deer in nocturnal thermal video shot from a moving
vehicle. The goal is not "find deer in frames" but "report how many distinct animals exist
in this video."

## Main flow — left to right, five stages

**1. INPUT**
FLIR thermal video, 640 x 512 grayscale, 60 fps. Vehicle-mounted, moving 25–90 km/h, so the
whole frame translates continuously. Show a small dark-grey video-frame icon with a faint
deer silhouette.

**2. PREPROCESSING — CLAHE**
Contrast-limited adaptive histogram equalization, clip limit 2.0, 8 x 8 tiles. Applied
identically at training and inference. Annotate: "test mAP50 0.519 with, 0.299 without".

**3. DETECTION — YOLO11m @ 640 px**
Single class, confidence threshold 0.10, NMS IoU 0.50. Emits per-frame bounding boxes.
Annotate: "median animal 29 x 24 px, 71% COCO-small".

**4. TRACKING & ASSOCIATION — BoT-SORT**
Kalman motion model plus **sparse optical-flow global motion compensation** (label this
clearly; it is required because the camera moves). Emits **candidate tracks**.
This stage has a **side branch** feeding back into it, drawn as a small box beneath or
beside it:
  - **Orphan recovery**: detections the tracker returns with no track identity are linked
    into pseudo-tracks by temporal gap and scale-normalised distance. Annotate "+17 animals".
Optionally note that an appearance-embedding term can be enabled here.
Output annotation: "7,008 candidate tracks".

**5. CONFIRMATION — 3-parameter rule**
Accept a candidate track if **all three** hold:
  - persists for at least 20 frames
  - spans at least 0 seconds
  - mean of its five highest per-frame detector confidences is at least 0.65
Show these as three small stacked conditions inside the stage box.

**6. OUTPUT**
Three items, drawn as a small stack:
  - the count for the video
  - a calibrated confidence per counted animal (annotate "92.1% at >= 0.80")
  - an evidence contact sheet per animal for human audit

## The second, essential element — the evaluation decomposition

Below the main flow, draw a **separate horizontal band** labelled *Evaluation decomposition*.
This is the paper's main contribution and must be visually distinct from the pipeline itself
— use a different background tint or a dashed enclosing border, and connect it upward to the
stages it measures with thin dotted lines.

It has three nested stages, shown as a **funnel or descending bar chart** over 83 ground-truth
animals in held-out video:

| Quantity | Value | Measures | Connects up to |
|---|---|---|---|
| REACHED | 74 / 83 | animal touched by at least one candidate | Detection + Tracking |
| PRIMARY | 66 / 83 | animal's best-covering candidate is not also best for another animal | Association |
| COUNTED | 55 / 83 | confirmation accepted a candidate covering it | Confirmation |

Label the two gaps explicitly, since they are the paper's finding:
  - 74 → 66: "8 animals share a track with a neighbour"
  - 66 → 55: "11 animals rejected by the confirmation rule"

Add a short caption strip under the band: **"Detection is not the bottleneck."**

## Style requirements

- Flat, modern, academic. No 3D, no drop shadows, no gradients, no skeuomorphic icons.
- Rounded rectangles for stages, thin arrows, generous white space.
- One restrained accent colour for the pipeline stages (a muted blue or teal) and a second,
  warmer accent used **only** for the decomposition band, so the two layers never blur.
- Sans-serif type throughout. Stage titles bold, parameter annotations in a smaller regular
  weight, all annotations horizontal — never rotated or vertical.
- Every number listed above must appear as a label. Do not invent numbers, model names,
  thresholds or extra stages.
- No deep-learning layer diagrams, no convolution stacks, no neural-network cartoons — this
  is a systems diagram, not an architecture-of-a-network diagram.
- Do not add a training loop, a loss function, or a dataset-construction branch. The figure
  shows inference and evaluation only.
- Leave the figure untitled; the caption is set in LaTeX.

## What the reader should take away in five seconds

Thermal video flows left to right through five stages into a count. Underneath, a funnel
shows that of 83 animals, 74 are found, 66 get their own track, and only 55 are counted —
so the loss is concentrated at the end of the pipeline, not at the detector.
