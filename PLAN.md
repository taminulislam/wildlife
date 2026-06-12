# FLIR Deer Detection & Counting — Project Plan

## 1. Project understanding

**Client goal:** Automatically detect and count white-tailed deer in FLIR thermal videos
recorded along road transects in Illinois, with a confidence/uncertainty score for each
detection. Counts feed into abundance and density estimation.

**Data characteristics (from the overview doc + example video):**

| Property | Value |
|---|---|
| Camera | Teledyne FLIR ADK 2.0 (P/N 40640U024-6PAAX) |
| Resolution | 640 × 512, grayscale thermal (stored as 8-bit MP4) |
| Frame rate | ~60 fps |
| Field of view | 24° HFOV, side-facing from a vehicle |
| Capture conditions | ½ hr before sunset → 2 hrs after sunset, 15–55 mph |
| Example video | `EagleCreekDrEtc_SHB_12.11.2025_LS.mp4` — 6 min, ~21,600 frames |
| Naming convention | `<Site>_<Observer?>_<Date>_<LS|RS>` (LS/RS = left/right side window) |
| Extra data available | GNSS track (Bad Elf receiver): speed, direction, coordinates per transect |

**Key technical challenges:**

1. **Small, low-contrast targets.** Deer at distance are a handful of pixels; thermal
   contrast varies with weather and distance.
2. **Vegetation occlusion.** The same deer appears/disappears between frames as the
   vehicle moves — detection per frame is unreliable; counting requires **tracking**.
3. **Double counting.** A deer visible for 5 seconds appears in ~300 frames. The unit of
   counting must be a *track* (one unique animal), not a detection.
4. **Confusers.** Other warm bodies (coyotes, raccoons, livestock, birds, humans,
   vehicles, hot culverts/rocks retaining heat) will trigger false positives.
5. **Motion blur / parallax.** At 55 mph with a 24° HFOV, near-field background streaks
   heavily while distant deer move slowly across the frame.

---

## 2. Phase 0 — Data audit & environment setup (Week 1)

- Inventory all videos: sites, dates, LS/RS, durations, total hours of footage.
- Request the GNSS logs now (speed per transect matters later for tracking parameters
  and for density estimation).
- Confirm whether any videos have **existing manual counts** — these become free
  ground truth for end-to-end count validation.
- Set up repo: Python, PyTorch, `ultralytics` (YOLO), OpenCV, ffmpeg, CVAT (or
  Roboflow/Label Studio) for annotation.
- Quick manual skim of the example video to log every deer encounter with timestamps —
  this becomes our first ground-truth count and tells us encounter frequency
  (deer events per minute of footage), which drives the sampling strategy below.

## 3. Phase 1 — Dataset creation from raw video (Weeks 1–2)

The raw footage is extremely redundant (60 fps, mostly empty forest). We must *not*
label frames uniformly — most would be empty.

**Frame extraction strategy:**

1. **Event-first mining.** Run cheap detectors over all available footage to find
   candidate "warm blob" events:
   - Intensity anomaly: deer are locally bright → top-hat morphology / adaptive
     threshold on each frame.
   - A pretrained generic detector (e.g., MegaDetector or COCO-pretrained YOLO run on
     contrast-stretched frames) as a second candidate source — imperfect on thermal but
     useful for recall.
   - Merge candidates into time windows ("events") with ±2 s padding.
2. **Sample within events** at 5–10 fps (every 6–12th frame) — consecutive 60 fps frames
   are near-duplicates and add label cost without information.
3. **Hard negatives:** sample frames from non-event footage too (~20–30% of the dataset):
   empty forest, hot rocks, vehicles, buildings, other animals. These teach the model
   what *not* to fire on.
4. Keep a frame-naming scheme that preserves provenance:
   `<video_stem>_f<frame_idx>.png` — needed later for video-level splits and for
   tracing label errors back to source.

**Target initial dataset:** ~3,000–5,000 labeled frames, of which ~70% contain ≥1 deer.
Expect to grow it via active learning (Phase 5).

## 4. Phase 2 — Annotation protocol (Weeks 2–4, overlaps Phase 1)

**Tool: CVAT** (self-hosted, free) — chosen because it supports *video annotation with
track interpolation*: you box a deer on keyframes and CVAT interpolates between them.
This is 5–10× faster than per-frame boxing and gives us **track IDs for free**, which we
need to evaluate counting, not just detection.

**Label schema:**

| Class | Notes |
|---|---|
| `deer` | Any visible white-tailed deer, even partially occluded |
| `other_animal` | Coyote, raccoon, livestock, etc. — kept separate so the model learns the distinction |
| (attributes) | `occluded` (yes/no), `truncated` (at frame edge) |

**Annotation rules (write these down before labeling starts):**
- Box the *visible* thermal signature, not the inferred full body.
- If a deer is occluded ≥ ~80% (only a leg/ear glow visible) but identity is clear from
  adjacent frames, still label it and mark `occluded` — tracking models need these.
- Groups: always individual boxes, never one box around a group (the count is per animal).
- Minimum size: label down to ~6×6 px; below that, skip (and note it — defines the
  detectability floor we report to the client).
- Ambiguous blobs that can't be confirmed as deer even with video context → don't label
  as deer; flag the clip for expert review (ask the client's biologists).

**QA:** second-pass review of 10–15% of each annotator's frames; disagreements resolved
and rules updated. Track inter-annotator agreement on a small shared subset.

**Dataset splits — by *video*, never by frame:** frames from the same video are highly
correlated; splitting them across train/val/test inflates metrics. Ideally also hold out
one entire *site* for the test set to measure generalization to new locations.

## 5. Phase 3 — Detection model (Weeks 4–6)

- **Baseline:** fine-tune YOLO (start with `yolo11s/m`) on the labeled frames at native
  640×512 (no downscaling — targets are small). Train with mosaic off or reduced late in
  training; small-object-friendly augmentation (scale jitter, brightness/contrast jitter
  to mimic thermal gain variation; no hue augmentation — grayscale).
- Replicate the single thermal channel to 3 channels to reuse RGB-pretrained weights;
  optionally compare against training from a thermal-pretrained checkpoint if available.
- If recall on tiny/distant deer is poor, evaluate **SAHI-style tiled inference** or a
  higher input resolution (1280) — measure the speed/recall trade-off.
- Report per-size-bucket recall (small/medium/large boxes) — the client should know the
  effective detection range, not just one mAP number.

**Detection metrics:** precision/recall, PR curves, mAP@0.5 — computed on the held-out
videos/site.

## 6. Phase 4 — Tracking & counting (Weeks 6–8)

Counting = number of confirmed **tracks**, not detections.

- Tracker: **ByteTrack or BoT-SORT** (built into `ultralytics`) on top of the detector.
  Tune for this scenario: high frame rate helps association; vehicle motion means
  *everything* moves — consider simple global motion compensation (frame-to-frame
  homography on the background) so the Kalman motion model sees deer motion, not
  vehicle motion.
- **Track confirmation rules** to suppress flicker false positives: a track counts as a
  deer only if it persists ≥ N frames (e.g., ≥ 0.5 s of cumulative detections) — tune N
  on validation videos.
- **Per-deer confidence score** (the client's explicit ask): aggregate detector
  confidences over the track (e.g., mean of top-k frame confidences), then **calibrate**
  on validation data so 0.9 means ~90% likely a real deer. Report it per counted animal.
- Handle re-appearance after occlusion: allow track re-association within a time/space
  gate (deer mostly stand still while the camera pans past — GNSS speed can inform the
  expected pixel velocity of static objects).

**Counting metrics:** per-video count error (MAE, bias) vs. manual ground truth;
tracking quality (IDF1, ID switches) on the CVAT track annotations. Count accuracy is
the headline metric for the client.

## 7. Phase 5 — Active learning loop (Weeks 6–10, iterative)

1. Run the current model on unlabeled footage.
2. Mine failures: low-confidence detections, tracks that died early, frames where the
   blob-detector fired but the model didn't (recall misses), confident detections on
   non-deer (precision misses).
3. Send those frames/clips to annotation; retrain.
4. Repeat 2–3 rounds. This is where most of the accuracy gain will come from, at a
   fraction of the labeling cost of Phase 2.

## 8. Phase 6 — Production pipeline & deliverables (Weeks 9–12)

A batch CLI: `count_deer <folder_of_videos>` producing per video:

- `counts.csv` — one row per counted deer: video, track ID, first/last timestamp,
  frame span, calibrated confidence, mean box size (distance proxy).
- `summary.csv` — one row per video: total count, count in confidence bands
  (e.g., ≥0.9 / 0.5–0.9), processing stats.
- **Review artifacts:** an annotated output video (or per-deer thumbnail clips) so
  biologists can spot-check every counted animal quickly — this builds trust and
  doubles as a continuing QA channel.
- Throughput target: faster than real time on a single GPU so the full archive is
  processable.

**Stretch (supports the client's density-estimation goal):** join counts to GNSS logs →
deer per transect-km; estimate distance from box size/elevation to support distance
sampling. Flag this to the client early — it shapes what we log per detection.

## 9. Risks & open questions for the client

1. How many total videos/hours exist, and across how many sites/seasons? (Drives
   labeling budget and generalization risk.)
2. Do manual counts exist for any videos? (Free evaluation ground truth.)
3. Same deer seen on LS and RS cameras, or on repeated transect drives — should the
   pipeline deduplicate across videos, or is that handled statistically downstream?
   (Assume downstream for now; confirm.)
4. What count accuracy is "good enough" for their density models? (Sets the stopping
   criterion for active-learning rounds.)
5. Other species in the area that resemble deer thermally (livestock, large dogs)?
   Local expertise will improve our annotation rules.
6. Raw 16-bit thermal available, or only 8-bit MP4? (16-bit radiometric data, if it
   exists, preserves contrast that the MP4 conversion throws away.)

## 10. Timeline summary

| Weeks | Phase |
|---|---|
| 1 | Data audit, environment, manual baseline count of example video |
| 1–2 | Frame mining & extraction pipeline |
| 2–4 | Annotation (CVAT, ~3–5k frames) |
| 4–6 | Detector training & evaluation |
| 6–8 | Tracking + counting + confidence calibration |
| 6–10 | Active learning rounds |
| 9–12 | Production pipeline, review tooling, final report |
