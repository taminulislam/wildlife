# Label Studio video labeling on Delta — setup & workflow

Delta has no Docker, so we use **Label Studio** (single pip process) instead of CVAT.
It does the same job: track-based video labeling (keyframes + interpolation + track
IDs). The model's tracks pre-load so you **correct** instead of draw from scratch.

Why track-based: one track = one unique deer (**count GT**) AND every frame's box =
**detector GT**, in one pass — you label each deer once, never double-count it.
Rules for *how* to label: see `ANNOTATION_PROTOCOL.md`.

---

## 1. Install (one time, into the wildlife env)

```bash
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
pip install label-studio
```

## 2. Run it on a COMPUTE node (never the login node — 30-min CPU cap)

> **Account note:** we only have a **GPU** allocation (`bgte-delta-gpu`); Delta rejects
> CPU-only jobs on it. So the labeling job must hold 1 GPU even though the work is pure
> CPU. It costs ~1 GPU-hour per wall-hour, so **`scancel` the job whenever you stop
> labeling** — your annotations persist on disk (`LABEL_STUDIO_BASE_DATA_DIR`) and
> reload next time. (The `gpuA100x4-interactive` queue caps at 1 h; use `gpuA100x4`,
> max 2 days.) For the full 42-video run, consider asking the PI to request a Delta
> **CPU allocation** so this is free instead of burning GPU-hours.

```bash
srun --account=bgte-delta-gpu --partition=gpuA100x4 --qos=bgte-delta-gpu \
     --gpus-per-node=1 --cpus-per-task=8 --mem=32G --time=08:00:00 --pty bash
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
hostname                       # <-- note this, e.g. gpua017  (used in the tunnel)

# serve the videos straight off disk (data never leaves Delta):
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/work/nvme/bgte/tislam6/wildlife_project/data/raw
export LABEL_STUDIO_BASE_DATA_DIR=/work/hdd/bgte/tislam6/wildlife_outputs/labelstudio

label-studio start --host 0.0.0.0 --port 8080
```

## 3. Reach it at localhost from your laptop (SSH tunnel)

```bash
# forwards login -> compute node port; replace gpua017 with the hostname above
ssh -L 8080:gpua017:8080 tislam6@dt-login.delta.ncsa.illinois.edu
```
Then open **http://localhost:8080** in your laptop browser.

> Using an OpenOnDemand **desktop** session on Delta? Skip the tunnel — just open
> `http://localhost:8080` inside that session's browser.

## 4. Create the project + paste the labeling config

New project → Settings → **Labeling Interface** → Code, paste:

```xml
<View>
  <Labels name="box" toName="video" allowEmpty="false">
    <Label value="deer" background="#1FE625"/>
  </Labels>
  <Video name="video" value="$video" framerate="60.0"/>
  <VideoRectangle name="box" toName="video"/>
</View>
```
(`framerate` matches our FLIR videos = 60 fps. Keep label name exactly **`deer`**.)

## 5. Import the pre-loaded tasks

Project → **Import** → upload:
```
/work/hdd/bgte/tislam6/wildlife_outputs/ls_import/pilot_tasks.json
```
This loads 3 tasks (GiantCityRd / NIron / RedFoxLn), each with the model's tracks as
an editable annotation. Open a task → the boxes are already there.

## 6. Correct (per ANNOTATION_PROTOCOL.md)

- Real deer, box loose → drag/resize, add a keyframe; CVAT-style interpolation fills
  between keyframes.
- Not a deer → delete the whole track (region panel → trash).
- Missed deer → add a new region, set keyframes; give each animal its own track.
- Multiple deer in a frame → one track each.
- **Submit** each task when done.

## 7. Export when the pilot tasks are done

Project → **Export** → **JSON** (the LS JSON with the video rectangle sequences).
Save it back under the cluster, e.g.
`/work/hdd/bgte/tislam6/wildlife_outputs/ls_export/pilot.json`, and tell me — I run the
converter (`ls_to_yolo.py`, built once we see your real export) to produce:
- YOLO training frames merged into `data/annotate/`, and
- a per-video **count GT** (unique tracks per video).

---

## Generate pre-load tasks for any video set

```bash
python src/track/tracks_to_labelstudio.py \
    --counts-dir /work/hdd/bgte/tislam6/wildlife_outputs/counts/full_m640_<SITE> \
    --source data/raw \
    --doc-root /work/nvme/bgte/tislam6/wildlife_project/data/raw \
    --out /work/hdd/bgte/tislam6/wildlife_outputs/ls_import/<SITE>_tasks.json \
    [--videos stem1,stem2] --confirmed-only [--min-conf 0.3]
```

`--confirmed-only` loads only the model's counted tracks (fewer false boxes to delete);
drop it to also pre-load weak candidates if you'd rather prune than redraw.
