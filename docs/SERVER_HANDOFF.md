# Server Handoff Prompt — FLIR Deer Detection & Counting

> Paste the section below into a fresh Claude Code session after moving this folder to the
> GPU server. It captures the full project state, decisions, and next steps.

---

## PROMPT TO PASTE

I'm continuing an existing project. The whole repo folder was moved here from a CPU laptop
to this GPU server. Read `PLAN.md`, `README.md`, `docs/WORKFLOW.md`, and
`docs/ANNOTATION_GUIDELINES.md` first, then confirm you understand the state below before
changing anything.

### Project goal
Automatically **detect and count white-tailed deer** in FLIR thermal road-transect videos
(Illinois), with a **per-animal confidence score**, to support abundance/density estimation.

### The core challenge (why this is non-trivial)
As the survey vehicle drives past, deer appear and disappear behind vegetation, so no single
frame shows the whole group (a frame may show 3 when 5–6 are present). Single-frame counting
undercuts; naive per-frame counting double-counts the same animal across ~300 frames.
**Solution = detect deer in every frame, then track each animal with a unique ID; the count
is the number of unique individuals over time.**

### Data on disk (came with the folder)
- `data/raw/` — **42 Left-Side (LS) videos**, 4 sites: MAS (12), SHB (10), SHW (12), TON (8).
  FLIR thermal, 640×512 grayscale, ~60 fps, 3–6 min each. Right-Side (RS) cameras and the
  remaining ~4 sites are NOT here yet (will be downloaded later; ~96 videos total expected).
- `data/events/<key>/` — mined warm-blob events per video (events.csv, hits.csv, thumbs).
  42 event dirs, `master_events.csv` + `summary_by_video.csv` aggregate them.
- `data/annotate/frames/` — extracted peak frames (PNG, git-ignored, regenerable).
- `data/annotate/labels/` — **HUMAN-VERIFIED YOLO labels — PRECIOUS, the ground truth.**
  1,381 frames reviewed → **72 deer frames, 110 deer boxes**, 1,309 confirmed no-deer
  (empty label files). All boxes are a single class `0 = deer`.
- `data/annotate/frames.csv` — index (name, key, site, event_id, src_frame, score).

### Verified findings (interim)
| Site | Deer frames | Deer boxes |
|------|---:|---:|
| SHB | 34 | 58 |
| SHW | 21 | 32 |
| MAS | 9 | 11 |
| TON | 8 | 9 |
| **Total** | **72** | **110** |
Group sizes: 50 frames w/1 deer, 13 w/2, 6 w/3, 1 w/4, 2 w/6.

### Pipeline & code (all under `src/`)
1. **Mining** (`src/mining/`): `filename_meta.py` (parses paths → VideoMeta with
   collision-proof `key` = `SITE__transect_vN_side`, e.g. `SHB__RedFoxLn_v1_LS`;
   site codes MAS/SHB/SHW/TON), `mine_events.py` (warm-blob detector: white top-hat +
   adaptive threshold + shape filters + union-find blob merging), `batch_mine.py` (runs all
   videos). DONE — events already mined.
2. **Dataset** (`src/dataset/`): `prepare_frames.py`+`server.py` were the annotation flow;
   `build_yolo_dataset.py` assembles a YOLO dataset **split by SITE** (`--test-sites`,
   `--val-frac`) to measure cross-site generalization. `select_for_annotation.py`,
   `extract_frames.py`, `extract_clips.py`, `build_contact_sheet.py`,
   `filter_events_by_verdict.py` support the labeling loop.
3. **Annotation** (`src/annotate/`): `server.py` is a zero-dependency local web tool for
   verifying/correcting boxes. Verification is DONE for the 42 LS videos.
4. **Demo** (`src/demo/`): `build_results.py` and `make_smooth_demo.py` produced the
   team-facing results in `results/`.
5. **`src/detect/`** and **`src/track/`** — empty placeholders; THIS IS THE NEXT WORK.

### Decisions already locked (do NOT relitigate)
- **Species-level classification is OUT OF SCOPE.** Confirmed with the team: they have no
  species-confirmed data, and everything labeled is deer. We deliver **deer vs. not-deer
  detection + an "is-it-an-animal" confidence score only.** Do not build a species classifier.
- **Confidence score:** every detection gets a 0–1 score; detections below a tuned threshold
  (~0.5, to be calibrated against the verified data) are flagged for manual review rather
  than auto-counted. Score derives from learned appearance (heat + size + shape), not a
  single cue.
- **Overlapping deer:** tracker should split merged heat blobs into separate IDs when they
  separate over time; persistently-merged cases get flagged for manual ID assignment.
- **Scope so far = LS cameras, 4 sites.** RS + remaining sites are a later extension.
- **Split by site, not by frame,** when training, to measure generalization to new sites.

### WORKFLOW CONSTRAINT (must follow every time)
After every edit, **commit and push to GitHub** (`git@github.com:taminulislam/wildlife.git`).
This is a standing rule from the user. Note: `data/raw/`, frames, `*.png/*.jpg`, weights
(`*.pt/*.onnx`), and secrets are git-ignored; the verified `data/annotate/labels/` and
`results/` ARE tracked. (Security: a GitHub PAT was previously shared in plaintext and
should be rotated.)

### Immediate next steps (the reason for moving to GPU)
1. **Grow the training set:** harvest MULTIPLE frames per confirmed deer event (each deer is
   visible for seconds = many frames), prioritizing the 9 group events, to expand well beyond
   110 boxes. Reuse `data/events/*/events.csv` peak frames + neighbors.
2. **Build `src/detect/`:** train the first deer detector (YOLO recommended; GPU now
   available). Use `build_yolo_dataset.py` to assemble the split. Report per-site metrics.
3. **Build `src/track/`:** detect-every-frame + multi-object tracking (unique IDs) →
   per-video unique-deer counts with the confidence score and overlap-split logic above.
4. **Calibrate the confidence threshold** against verified data; wire low-score → review.
5. **Extend to RS cameras + remaining sites** as footage is downloaded.

First, please: verify GPU is available (`nvidia-smi`), set up the Python env from
`requirements.txt` (add torch/ultralytics for training), confirm the data moved intact
(42 videos, 1,381 label files, 110 boxes), then propose a concrete plan for step 1–2 before
running anything heavy.

---

## Quick integrity checks to run on the server
```bash
# videos present (expect 42)
find data/raw -type f -iname '*.mp4' | wc -l
# label files present (expect 1381) and deer boxes (expect 110)
ls data/annotate/labels/*.txt | wc -l
cat data/annotate/labels/*.txt | grep -c .
# GPU
nvidia-smi
```
