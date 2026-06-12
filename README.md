# Wildlife FLIR Deer Detection & Counting

Automatically detect and count white-tailed deer in FLIR thermal videos recorded along
road transects in Illinois, producing a per-video count of unique animals each with a
confidence score. Counts feed downstream abundance and density estimation.

See [`PLAN.md`](PLAN.md) for the full project plan.

## Data

- ~96 videos: **8 sites × 2 visits × 6 videos**, 640×512 grayscale thermal, ~60 fps,
  ~6 min each. FLIR ADK 2.0, 24° HFOV, side-facing from a moving vehicle.
- Filenames: `<Site>_<Observer>_<Date>_<LS|RS>.mp4` (LS/RS = left/right side window).
- **Raw videos and extracted frames are git-ignored** — they are large and regenerable.
  Keep them under `data/raw/` locally.

## Approach (high level)

`detect → track → count`. The counted unit is a **track** (one unique deer), not a
per-frame detection, to avoid counting the same deer across its ~300 frames of visibility.

## Repository layout

```
data/
  raw/            # source videos (git-ignored)
  events/         # mined warm-blob event lists per video (CSV/JSON)
  frames/         # extracted candidate frames (git-ignored)
  clips/          # short clips for CVAT annotation (git-ignored)
  dataset/        # curated, labeled dataset (YOLO format) — splits by video/site
src/              # pipeline code
  mining/         # event detection / frame extraction
  dataset/        # dataset assembly, splits, conversion
  detect/         # detector training & inference (run on GPU machine)
  track/          # tracking & counting
docs/
PLAN.md
```

## Environment

CPU-only Windows dev machine for all dataset work; detector training happens on a
separate GPU machine. Python + OpenCV + ultralytics (YOLO) + CVAT for annotation.

## Status

Dataset-creation pipeline is built and tested on the example video (CPU-only). See
[`docs/WORKFLOW.md`](docs/WORKFLOW.md) for the end-to-end runbook and
[`docs/ANNOTATION_GUIDELINES.md`](docs/ANNOTATION_GUIDELINES.md) for the label rules.

Ready to run on the full archive as soon as videos land in `data/raw/`:

1. `src/mining/batch_mine.py` — mine warm-body events across all videos → master index
2. `src/dataset/select_for_annotation.py` — balanced annotation batch across sites
3. `src/dataset/extract_clips.py` / `extract_frames.py` — clips/frames for CVAT
4. Annotate in CVAT → `src/dataset/build_yolo_dataset.py` — site-split YOLO dataset
5. Train on GPU machine; tracking + counting pipeline (`src/track/`) comes next.
