Subject: FLIR Deer Detection — early prototype & approach (quick check)

Hi team,

Quick update on the deer-counting project. I built a working prototype on a small subset
of the footage to validate the approach before scaling up.

**What I did**
- Processed 42 left-side videos across 4 sites (MAS, SHB, SHW, TON).
- Auto-flagged ~1,460 warm-body moments, then manually verified deer vs. non-deer.
- Confirmed 72 deer sightings so far (SHB most active, TON least).

**Attached**
- `deer_detections.mp4` — detections with a running count (the output format).
- `deer_grid.png` — all verified deer on one sheet.

**Problem I'm solving**
Counting from a single frame undercounts groups: as the vehicle drives past, deer appear
and disappear behind vegetation, so no single frame shows the whole group (e.g., a frame
shows 3 but 5–6 are actually present).

**Planned solution**
Detect deer in *every* frame and track each animal with a unique ID across frames, so the
count = unique individuals over time — a deer seen repeatedly is counted once, and group
members visible at different moments are all recovered.

**Plan (short)**
1) Auto-detect warm bodies → 2) Human-verify/correct boxes → 3) Train deer detector →
4) Track + count unique deer with a confidence score → 5) Extend to right-side cameras and
the remaining sites (optional: deer/km using the GNSS logs).

So far this is a small subset; next I'll work on the full dataset you shared and train the
detector.

**Question:** Does this approach look right to you for the project before I scale up?

Thanks,
[Your name]
