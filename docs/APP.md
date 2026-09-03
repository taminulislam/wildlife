# TRACT interface — video in, counted video out

Upload a raw FLIR thermal transect; get back the same video with detections, track
identities and a running count, plus a per-track CSV.

It runs the published pipeline, not a demo approximation: CLAHE contrast normalisation →
YOLO11m @ 640 px → BoT-SORT with global motion compensation → orphan recovery → the frozen
confirmation rule (n ≥ 20 frames, span ≥ 0 s, top-5 mean confidence ≥ 0.65). Numbers it
reports are the paper's numbers.

## Web interface

```bash
bash scripts/run_app.sh              # gets a GPU node, starts the server, prints the URL
```

The server prints an `ssh -L …` line. Run that on your laptop, then open
<http://localhost:8080>. Drag a video in, press **Run pipeline**.

Running it by hand instead:

```bash
srun --account=bgte-delta-gpu --partition=gpuA100x4-interactive \
     --gpus-per-node=1 --cpus-per-task=8 --mem=48g --time=01:00:00 --pty \
  /work/nvme/bgte/tislam6/envs/wildlife/bin/python src/app/server.py --port 8080 --device 0
```

`--device cpu` works but is roughly 20× slower; it exists for a laptop, not for the login
node, which has a 30-minute CPU ceiling.

## Command line

```bash
python src/app/cli.py --video raw.mp4        --out results/app_run
python src/app/cli.py --video folder/videos/ --out results/app_run --device 0
```

Writes `<stem>_counted.webm`, `<stem>_tracks.csv` and a `summary.csv` across all inputs.

## What the interface exposes

| Control | Default | Effect |
|---|---|---|
| Detector confidence | 0.10 | the threshold the counting pipeline runs at, not the 0.25 of the detection table |
| Min frames (n) | 20 | confirmation rule: a track must be seen this often |
| Min track score | 0.65 | confirmation rule: mean of the track's five best per-frame scores |
| Draw candidates | yes | show rejected tracks thin and labelled `candidate`, so the rule's decisions are visible |

Lowering **min track score** recovers real animals and admits duplicates at roughly two per
animal recovered; §4.6 of the paper measures that trade. It is the one control worth moving,
and moving it changes the count away from the published operating point.

## Output

- **Video** — box colour is the track identity; solid boxes are counted animals, thin ones
  are candidates the rule rejected. The banner carries the running count, which rises at the
  frame where each track first satisfies the rule.
- **CSV** — one row per candidate track: first/last frame and time, frames seen, span,
  mean and top-5 confidence, mean box area in pixels, and whether it was counted.

## Notes

- **No third-party web framework.** Python's `http.server` only. Gradio, Streamlit and Flask
  are not installed in the inference environment and the project sits near its inode quota.
- **WebM/VP8 output.** This OpenCV build has no H.264 encoder; an `mp4v` file will not play
  in Chrome or Safari. `src/app/engine.py:CODECS` negotiates the first container the build
  can actually write, preferring one a browser can play.
- Two passes over the video: one to track, one to draw. The confirmation rule is a property
  of a whole track, so the drawing pass is what lets the counter rise where the evidence
  arrives rather than jumping at the end.
- Uploads are capped at 4 GB and land in `--work` (default `/tmp/tract_app`), which is
  node-local and cleared when the job ends.
