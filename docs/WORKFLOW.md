# Workflow Runbook

End-to-end commands to go from raw videos to a trainable dataset and, later, counts.
Run from the repo root. Everything through Step 5 is CPU-only and runs on this machine;
Step 6+ (training) happens on a GPU machine.

## 0. Setup (once)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. Drop videos in place

Put all raw videos under `data/raw/`. The archive layout is
`<SITE>_Videos_LS/visit<N>/[<SITE>_<date>/]<Transect>_<SITE>_<Date>_<Side>.mp4`.
The **site code is the 2nd filename token / parent folder** (MAS, SHB, SHW, TON, ...);
the first token is the transect/road name. Visits are combined into their site (we do
not split by visit), but each file gets a collision-proof key so same-named transects
across visits don't clash.

```powershell
# sanity-check parsing + per-site counts over the whole archive:
python src/mining/filename_meta.py
```

## 2. Mine warm-body events across the whole archive

Unattended triage. Produces per-video event lists + a master index. Safe to re-run as
more videos finish downloading (skips done ones).

```powershell
python src/mining/batch_mine.py --raw data/raw --save-thumbs --workers 4
```

Outputs:
- `data/events/<stem>/events.csv`, `hits.csv`, `thumbs/`
- `data/events/master_events.csv`, `data/events/summary_by_video.csv`

**Tune if needed:** if too much empty footage is flagged, raise `--min-contrast` (e.g.
22–26) and/or `--min-event-hits`. Inspect `thumbs/` to judge.

## 3. (Optional but recommended) Human skim

Open each video's `thumbs/` and skim flagged events. Mark deer / not-deer / unsure in a
simple sheet. This gives a **rough per-video deer count = ground truth** for validating
the final pipeline, and catches any events the miner missed. A few minutes per video.

## 4. Select a balanced first annotation batch

Spread annotation across all sites instead of over-labeling action-heavy videos.

```powershell
python src/dataset/select_for_annotation.py --total 400 --per-video 8
# -> data/dataset/annotation_batch.csv
```

## 5. Cut clips (and/or frames) for annotation

**Clips for CVAT video annotation (primary — gives track IDs):**
```powershell
python src/dataset/extract_clips.py --all --pad 1.0 --out-fps 10
# -> data/clips/<stem>/event_***.mp4  + data/clips/clips_manifest.csv
```

**Still frames (for image-only annotation, and to pull hard negatives):**
```powershell
python src/dataset/extract_frames.py --all --pos-fps 3 --neg-per-video 30
# -> data/frames/<stem>/*.png + data/frames/frames_manifest.csv
```

### Annotate in CVAT

1. Run CVAT locally with Docker (`docker compose up -d` from a CVAT checkout) — see
   https://docs.cvat.ai. Open http://localhost:8080.
2. Create a project with classes `deer`, `other_animal` and attributes `occluded`,
   `truncated` (matches `docs/ANNOTATION_GUIDELINES.md`).
3. Upload the event clips from `data/clips/` as tasks. Annotate with **track
   interpolation** per the guidelines.
4. Export each task as **YOLO 1.1 / Ultralytics** format into `data/annotations_yolo/`,
   mirroring the image tree.

## 6. Build the YOLO dataset (split by site)

Hold out 1–2 entire sites for test so accuracy reflects new-location generalization.

```powershell
# Site codes are MAS, SHB, SHW, TON (first half). Hold out 1-2 entire sites for test:
python src/dataset/build_yolo_dataset.py `
  --images data/frames --labels data/annotations_yolo `
  --test-sites TON --val-frac 0.15
# -> data/dataset/yolo/{images,labels}/{train,val,test} + data.yaml
```

## 7. Train (GPU machine)

Ship `data/dataset/yolo/` to the GPU box and train YOLO:

```bash
yolo detect train model=yolo11s.pt data=data.yaml imgsz=640 epochs=100 \
     mosaic=0.5 hsv_h=0 hsv_s=0 fliplr=0.5 close_mosaic=20
```

Bring `best.pt` back here for inference/counting (Step 8, built later).

## 8. Track + count (pipeline TBD)

Detector + ByteTrack/BoT-SORT → confirmed tracks → calibrated per-deer confidence →
`counts.csv` / `summary.csv` + annotated review video. To be implemented in `src/track/`.

---

### Active learning loop (repeat 2–3×)
Run the current model on un-annotated footage → harvest failures (low-confidence
detections, early-dying tracks, recall misses where the blob miner fired but the model
didn't) → correct pre-labels in CVAT → rebuild dataset → retrain. Most accuracy gains
come from here.
