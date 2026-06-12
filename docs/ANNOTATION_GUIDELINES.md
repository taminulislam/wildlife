# Annotation Guidelines — FLIR Deer Counting

These rules define how to label deer in the FLIR thermal clips/frames. Consistency here
matters more than speed: inconsistent labels cap the model's accuracy no matter how good
the training is. Read this fully before your first session.

## What we're labeling and why

We are building a model that **counts unique deer** per video. Annotation produces two
things the model needs:
1. **Boxes** around each animal's thermal signature (for the detector).
2. **Track IDs** — the same deer keeps the same ID across frames of a clip (for evaluating
   counting, and for training the tracker). CVAT gives track IDs automatically when you
   annotate a clip with interpolation.

## Classes

| Class | Use for |
|---|---|
| `deer` | Any white-tailed deer, even partially occluded or distant, as long as you can confirm it's a deer (using neighboring frames if needed). |
| `other_animal` | Any other warm animal: coyote, raccoon, livestock, dog, bird, etc. We label these so the model learns to tell them apart from deer — do **not** skip them. |

If you genuinely cannot tell whether a blob is a deer or another animal even after
scrubbing nearby frames, see "Uncertain blobs" below — do not guess `deer`.

## Box attributes (set on each box)

- `occluded` — yes if vegetation/terrain hides a substantial part of the animal.
- `truncated` — yes if the animal is cut off by the frame edge.

## Boxing rules

1. **Box the visible thermal signature**, not the imagined full body. If only the head
   and neck glow above a bush, box the head and neck.
2. **One box per animal.** Never draw a single box around a group. The count is per
   individual — a group of 3 deer = 3 boxes with 3 track IDs.
3. **Tight boxes.** The box edges should touch the outermost warm pixels of the animal,
   with no extra margin.
4. **Occluded but identifiable:** if a deer is mostly hidden but you can confirm its
   identity from adjacent frames (it was clearly there a moment ago), keep labeling it
   and mark `occluded`. Tracking needs these in-between frames.
5. **Minimum size:** label down to ~**6×6 pixels**. Below that, the signature is too
   ambiguous — skip it, but note the clip if you think a real deer was below this floor
   (this defines our detectability limit, which we report to the client).
6. **Re-appearance:** if a deer leaves behind a tree and re-emerges, give it the **same
   track ID** (in CVAT, resume the same track) — it's the same animal.

## Track rules (CVAT interpolation workflow)

- Annotate a clip by placing a box on a **keyframe**, then move to a later frame where
  the animal has moved and place another keyframe; CVAT interpolates between them.
- Add keyframes whenever the box would otherwise drift off the animal (fast motion,
  occlusion start/end). A keyframe every ~0.5–1 s is usually enough at 10 fps.
- **End the track** (mark "outside") on the frame where the animal fully leaves view or
  becomes unidentifiable.

## Uncertain blobs

- A warm blob you cannot confirm as an animal (could be a hot rock, culvert, stump,
  reflection) → **do not label it as `deer`**. Leave it unlabeled.
- If it's plausibly an animal but you're unsure of species or even animal-vs-object,
  flag the clip (CVAT issue / a note in the tracking sheet) for expert review. We will
  ask the client's biologists on a batch of these.
- When in doubt between `deer` and "nothing", prefer **not** labeling — a false deer
  label is worse for the count than a missed hard case, which active learning will
  recover later.

## Hard negatives (frames with no deer)

Some clips/frames are deliberately included that contain **no deer** (hot rocks,
vehicles, empty forest, other animals). These are not mistakes — label any
`other_animal` present, and otherwise leave the frame empty. An empty label file is a
valid, valuable training example.

## Quality assurance

- A reviewer re-checks 10–15% of each annotator's clips.
- A small shared set is labeled by everyone to measure inter-annotator agreement; we
  refine these rules where people disagree.
- Disagreements and new edge cases are logged and resolved here — **this document is
  living**; propose changes rather than silently inventing a rule.

## Quick checklist (per clip)

- [ ] Every visible deer boxed, one box each, tight.
- [ ] Same animal keeps one track ID across the clip (including after occlusion).
- [ ] `occluded` / `truncated` attributes set where they apply.
- [ ] Other animals labeled as `other_animal`.
- [ ] Uncertain blobs left unlabeled and flagged if plausibly animal.
- [ ] Tracks ended ("outside") when the animal leaves view.
