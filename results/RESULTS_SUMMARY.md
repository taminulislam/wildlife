# FLIR Deer Detection — Prototype Results (Interim)

**Date:** 2026-06-12  **Scope:** first-half of the archive, Left-Side (LS) cameras only.
**Status:** working prototype of the data → detection → count pipeline. This is an early
proof of concept to validate the approach — **not** the final trained model.

---

## What we set out to do

Automatically detect and count white-tailed deer in FLIR thermal road-transect videos,
with a confidence score per animal, to support abundance/density estimation.

## What we processed

| | |
|---|---|
| Videos analyzed | **42** (Left-Side cameras) |
| Sites covered | **4** — MAS, SHB, SHW, TON |
| Footage | ~2.8 hours of thermal video (640×512, 60 fps) |
| Candidate warm-body events auto-detected | **1,460** |
| Frames human-verified | **1,381** |

## Preliminary deer findings

Every auto-detected warm-body event was reviewed by a human to separate real deer from
false alarms (hot rocks, tree trunks, vehicles). Confirmed results:

| Site | Deer sightings | Deer individuals (boxes) |
|------|---------------:|-------------------------:|
| SHB  | 34 | 58 |
| SHW  | 21 | 32 |
| MAS  | 9  | 11 |
| TON  | 8  | 9  |
| **Total** | **72** | **110** |

> "Sightings" = distinct moments a deer appeared; "individuals" = total deer boxes across
> those frames. SHB is clearly the most deer-active site in this half; TON the least.

## What the prototype demonstrates (the deliverables)

- **`deer_detections.mp4`** — a montage of every confirmed deer with its detection box and
  a running count. This is exactly the *form* of the final output: video in → boxed deer
  out → a count.
- **`deer_grid.png`** — all 72 verified deer on one sheet (overview of the detections).
- **`per_site_counts.csv`** — the per-site counts above, machine-readable.

## How it works (pipeline)

1. **Auto-triage** — scan every video for warm blobs, group them into events. Cheap, runs
   unattended; reduces 10+ hours of footage to ~1,500 moments worth a look.
2. **Human verification** — a fast review tool to confirm deer vs. not-deer and correct
   the boxes. (This is the step just completed.)
3. **Train a detector** — use the verified boxes to train a model that recognizes deer
   automatically (next phase).
4. **Track & count** — follow each deer across frames so one animal is counted once
   (not once per frame), with a confidence score. Produces the final per-video counts.

## Honest limitations of this interim result

- **LS cameras, first-half sites only.** Right-side cameras and the other 4 sites are not
  yet included.
- **Counts are from human-verified detections, not yet an automated model.** They show
  the data is real and the targets are detectable; the automated counter is the next step.
- **110 verified deer is a small training set.** The immediate next move is to harvest
  many more frames from the confirmed deer events (each deer is visible for seconds =
  many frames), growing the training data several-fold before training the detector.

## Next steps

1. Grow the deer training set from the confirmed events (in progress).
2. Train the first deer detector and run it on full videos.
3. Add tracking + per-deer confidence → automated per-video counts.
4. Extend to right-side cameras and the remaining 4 sites as footage arrives.
5. (Optional) Join counts to GNSS transect logs → deer per km for density estimation.
