# Deer video annotation protocol (the rulebook)

Goal: a consistent, gold-standard label set that gives us **counts** (unique animals)
and **detector training boxes** at once. Read this before labeling so every video is
done the same way.

## Core rules

1. **One track per individual animal.** A deer that is visible, leaves, and comes back
   in the SAME video: if you are confident it's the same animal, reuse its track; if
   unsure, make a new track (over-splitting is safer than merging two real deer into
   one — note it).
2. **Never reuse a track ID across different animals.** Two deer on screen = two tracks,
   always, even if they overlap.
3. **Label each deer once, let CVAT interpolate.** Set a box on a keyframe, move ahead
   ~10–20 frames, adjust the box (new keyframe), repeat. CVAT fills the between-frames.
   Re-key whenever the deer changes direction/speed or the box drifts off it.
4. **Box tightness (thermal):** the box hugs the warm deer blob — include legs/head,
   exclude the cold background. Consistent tightness matters more than pixel perfection.
5. **End the track** when the deer leaves frame or is fully occluded. In CVAT: go to
   the last frame it's visible, step forward one frame, select the object and press
   **`O`** (Outside) — the box stops rendering/interpolating from there. Re-entry after a
   brief gap: press `O` again on the return frame to resume the SAME track. Don't let a
   box sit on empty background.

## What IS a deer

- A warm (bright) blob with deer shape/gait — body + legs/head, moving like an animal.
- Count it even if small/distant, as long as you can tell it's a deer.
- Partially occluded (behind brush, half out of frame) → still label the visible part.

## What is NOT a deer (delete these pre-loaded tracks)

- Warm rocks, road, rooftops, vehicles, equipment, vegetation hotspots.
- Humans, dogs, other clearly non-deer animals (we are deer-only, single class).
- Sensor artifacts / glare.
- If you genuinely cannot tell → leave it OUT (precision over recall for GT).

## Multiple deer (the under-count failure we're fixing)

- If a frame has 3 deer, there must be **3 tracks** with boxes in that frame.
- Don't let one box cover a group. One box = one animal.

## Mislocalized pre-loaded boxes (the localization failure)

- If the model's box is near a deer but off → drag it onto the deer (don't delete the
  track, just fix the box, add a keyframe).
- If the box is on nothing and there's no deer → delete the track.

## Occlusion / re-entry

- Brief occlusion (passes behind a tree, < ~1 s): keep the same track, mark the gap
  `outside`, resume when it reappears.
- Long gap or you lose certainty it's the same animal: end the track, start a new one.

## Edge calls — write them down

Keep a running notes line per video (in the count-review sheet) for anything ambiguous:
herds that split/merge, a blob you counted-but-weren't-sure, suspected same-animal
across visits, etc. These notes drive protocol tweaks after the pilot.

## Per-video checklist

- [ ] Every real deer has exactly one track.
- [ ] No track sits on a non-deer.
- [ ] Boxes are tight and on the animal across the whole track.
- [ ] Multiple-deer frames have one track each.
- [ ] Tracks end when the deer leaves.
- [ ] Saved (`Ctrl+S`) and exported (YOLO + CVAT-video).
- [ ] Logged the final unique-deer count for this video.
