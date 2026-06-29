# CVAT video labeling — setup & workflow

We are rebuilding ground truth by **labeling deer in the videos** (not loose frames).
Track-based annotation gives us BOTH deliverables from one pass:

- **Count GT** — one track ID = one unique animal (you label each deer once; it never
  gets double-counted).
- **Detector GT** — every frame's box becomes YOLO training data, including the hard
  frames the current model fails on.

The model's existing tracks are pre-loaded so you **correct** instead of draw from
scratch: KEEP good boxes, DELETE false positives, ADD missed deer, FIX bad boxes.

---

## 1. Install CVAT (self-hosted, data stays local)

On your own machine (needs Docker + Docker Compose):

```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
# create your login (one time):
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Open **http://localhost:8080** and log in. (Use Chrome/Chromium — CVAT recommends it.)

## 2. Get the videos + pre-load files onto your machine

From the cluster (run on your laptop, not on Delta):

```bash
# 3 pilot videos
scp tislam6@login.delta.ncsa.illinois.edu:/work/nvme/bgte/tislam6/wildlife_project/data/raw/TON_Videos_LS/visit1/GiantCityRd_TON_12.03.25_LS.mp4 .
scp tislam6@login.delta.ncsa.illinois.edu:/work/nvme/bgte/tislam6/wildlife_project/data/raw/SHW_Videos_LS/visit1/SHW_01.18.2026/NIron_SHW_01.18.2026_LS.mp4 .
scp tislam6@login.delta.ncsa.illinois.edu:/work/nvme/bgte/tislam6/wildlife_project/data/raw/SHB_Videos_LS/visit1/RedFoxLn_SHB_12.11.2025_LS.mp4 .

# matching pre-load track files (one XML per video)
scp -r tislam6@login.delta.ncsa.illinois.edu:/work/hdd/bgte/tislam6/wildlife_outputs/cvat_preload/pilot .
```

(For the full set later, use Globus instead of scp — it's tens of GB.)

## 3. Create one task per video

CVAT → **Tasks → +** → Create new task:

- **Name**: the video stem, e.g. `GiantCityRd_TON_12.03.25_LS`
- **Labels**: add one label named exactly **`deer`** (rectangle)
- **Select files**: upload the `.mp4`
- Create. Wait for CVAT to finish chunking the video.

> Keep the task name = the video stem. The export→YOLO step keys on it.

## 4. Pre-load the model's tracks

Open the task → **Menu (☰) → Upload annotations** → format **`CVAT for video 1.1`** →
pick that video's `.xml` from `pilot/`. The model's tracks appear as editable tracks.

## 5. Correct (see ANNOTATION_PROTOCOL.md for the rules)

- Scrub through. For each pre-loaded track: real deer → adjust the box if loose; not a
  deer → **delete the whole track** (right-click track → Remove).
- Missed deer → draw a **new track** (Track mode `N`), set boxes on keyframes; CVAT
  interpolates between them. Give every distinct animal its own track.
- Multiple deer in one frame → one track **each**.
- Save often (`Ctrl+S`).

## 6. Export when a task is done

Task → **Menu → Export annotations**:

- **`Ultralytics YOLO Detection 1.0`** (with images) → detector training frames.
- **`CVAT for video 1.1`** → track IDs for the count GT.

Send both back to the cluster (or drop into a shared folder); the
`cvat_to_yolo.py` / count-GT scripts ingest them and merge into `data/annotate/`.

---

## Generate pre-load files for any video set

```bash
# on the cluster, env: /work/nvme/bgte/tislam6/envs/wildlife
python src/track/tracks_to_cvat.py \
    --counts-dir /work/hdd/bgte/tislam6/wildlife_outputs/counts/full_m640_<SITE> \
    --out /work/hdd/bgte/tislam6/wildlife_outputs/cvat_preload/full_m640_<SITE> \
    [--videos stem1,stem2] --confirmed-only [--min-conf 0.3]
```

`--confirmed-only` loads just the model's counted tracks (fewer false boxes to delete);
drop it to also pre-load weak candidates if you'd rather prune than redraw.
