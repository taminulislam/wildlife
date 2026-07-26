# Results log — thermal deer detection & counting

**Living document.** Every new result gets appended here with its date, job ID, and the
command/config that produced it, so the paper can be written from this file alone.
Numbers here are *measured*, never estimated; anything unmeasured is marked TODO.

Last updated: **2026-07-25** (see [Changelog](#changelog))

---

## 1. Dataset

32 CVAT-labelled FLIR thermal road-transect videos, 8 per site, 4 sites (Illinois).
**This is the entire dataset** — earlier drafts said 42 videos, which was wrong.

| Site | Videos | Unique deer (CVAT tracks) |
|---|---|---|
| SHB | 8 | 132 |
| TON | 8 | 51 |
| SHW | 8 | 38 |
| MAS | 8 | 15 |
| **Total** | **32** | **236** |

- Frames: 640×512 grayscale, 60 fps. Single class `deer`.
- One CVAT track = one unique animal ⇒ tracks are simultaneously **count GT** and
  (per frame) **detection GT**.
- Ground-truth table: `data/annotate_v2/count_gt.csv`.
- Class imbalance to disclose: SHB alone holds 56% of all deer; MAS has 15. Per-site
  results for MAS will be noisy.

### 1.1 Annotation quality (important — affects every metric below)

Measured 2026-07-25 over all 32 CVAT exports:

| Quantity | Value |
|---|---|
| Total GT boxes | 21,646 |
| **Human keyframes** | **1,295 (6.0%)** |
| **CVAT-interpolated** | **20,351 (94.0%)** |
| Median gap between keyframes | 15 frames |
| Median deer-centre travel between keyframes | **2.28 box-widths** |
| Segments where deer moves >1 full box-width | **79%** |

**Consequence:** on a moving camera at 60 fps, linear interpolation cannot follow the
real path. Mid-gap GT boxes can sit entirely off the animal. This injects label noise
at training time *and* penalises correct detections at evaluation time. Quantified in
§4.2 — it accounts for roughly a third of the apparent detection error.

### 1.2 Splits

**Pooled site-stratified** (`data/dataset/yolo_v3`) — chosen 2026-07-25 so every site
appears in train/val/test (the deployment target is these same 4 sites):

| Split | Images | Deer frames | Boxes | Sites |
|---|---|---|---|---|
| train | 12,852 | 5,013 | 7,547 | all 4 |
| val | 2,337 | 935 | 2,564 | all 4 |
| test | 3,725 | 1,490 | 2,792 | all 4 |

Split by **video** (never by frame), allocated by deer count ≈ 65/19/16%.
- val videos: `SHB__NShelbyRdblue_v1`, `TON__GiantCityRd_v1`, `SHW__Robinson_v1`, `MAS__N25thBlue_v2`
- test videos: `SHB__GolfDr_v1`, `SHB__NWolfCreekorange_v1`, `TON__AquaCultureRd_v1`,
  `TON__OikosRd_v1`, `TON__TouchofNatureRd_v1`, `TON__ChipsRd_v1`, `SHW__Melvin_v1`,
  `SHW__SIron_v1`, `MAS__NMarseilles_v2`

A leave-one-site-out split (SHW held out) is retained for the cross-site
generalization experiment (Phase D). COCO-format copies for mmdetection live in
`data/dataset/yolo_v3/coco_annotations/`.

### 1.3 Object scale

71% of deer are COCO-**small** (area < 32²), median box ≈ 29×26 px (√area ≈ 27 px),
**0% large**. Resolution is therefore the strongest single accuracy lever.

---

## 2. Pipeline fixes that changed results

Documented because each one invalidates numbers measured before it.

| # | Issue | Effect | Fixed |
|---|---|---|---|
| 1 | **Crushed thermal contrast** — raw frames had pixel std 9.6, p1–p99 spanning 44/255 grey levels | ~0.3 mAP ceiling across *every* recipe; deer nearly invisible | `src/common/thermal.py` CLAHE, applied at extraction. Epoch-1 mAP50 went 0.073 → 0.323 |
| 2 | **Tiny val set** (3 videos, 15 deer) | val mAP oscillated 0.01–0.33 while train loss fell smoothly; early-stopping froze `best.pt` at epoch 1 | pooled split, 4 videos / 45 deer / 935 deer-frames |
| 3 | **GPU pinning** — `CUDA_VISIBLE_DEVICES=$g … --device 0`; Ultralytics `select_device()` rewrites CVD from `--device` | all 4 concurrent trainings landed on one A100; batches silently auto-shrunk (32→16→8) ⇒ invalid comparison | pass `--device $gpu`, never CVD |
| 4 | **COCO eval OOM** — Ultralytics ignores `predict(batch=)` for list sources | 36–43 GiB alloc; size-stratified metrics failed on all 6 models | chunk the path list manually in `eval_full.py` |
| 5 | **CLAHE missing at inference** — `count_deer.py` passed the video *path* to Ultralytics, so raw frames reached a CLAHE-trained model | train/inference distribution mismatch; **every count produced before this is invalid** | decode frames in-process + `enhance_contrast`, `--contrast` must match training |

---

## 3. Detector benchmark — Ultralytics roster @ 640 px

Job `20477045`, 2026-07-25. Pooled split, CLAHE frames, SGD lr0=0.01, batch 32,
150 epochs (patience 50), evaluated on the **held-out test split** (3,725 images / 2,792 boxes).

| Model | P | R | mAP50 | mAP50-95 | AP-small | AP-med | AR-small | FP/bg-frame | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| **YOLOv9m** | 0.704 | 0.448 | **0.498** | 0.184 | **0.128** | 0.265 | 0.301 | 0.000 | 22 |
| **YOLOv10m** | 0.674 | **0.455** | **0.498** | **0.197** | 0.117 | **0.299** | 0.321 | 0.001 | 21 |
| YOLO12m | 0.646 | 0.431 | 0.473 | 0.184 | 0.104 | 0.292 | **0.358** | 0.000 | 19 |
| YOLO11m | **0.732** | 0.434 | 0.466 | 0.177 | 0.081 | 0.291 | 0.196 | 0.005 | 60 |
| YOLOv8m ⚠ | 0.650 | 0.408 | 0.448 | 0.174 | 0.127 | 0.274 | 0.399 | 0.000 | 1 |
| RT-DETR-L | 0.713 | 0.451 | 0.433 | 0.155 | 0.072 | 0.259 | 0.168 | 0.044 | 19 |

⚠ YOLOv8m's `best.pt` is an epoch-1 checkpoint (warmup fitness spike fooled selection) —
treat its row as a floor, or re-score a later checkpoint before publishing.

AP-large is undefined (`-1`) for every model: **no large deer exist in the test set.**

### Findings

1. **Spread is modest (0.433–0.498).** No architecture is dramatically better; this is a
   data-limited regime, not an architecture-limited one.
2. **Small objects cost 2–3.5×.** AP-small vs AP-med: YOLO11m 0.081 vs 0.291. Direct
   justification for the 1280 px ablation.
3. **False positives are near zero** (0.000–0.005/bg-frame for YOLO; RT-DETR 0.044 is the
   outlier). The regime is **recall-limited, not precision-limited** ⇒ run the detector at
   *low* confidence and let the temporal head filter. Also means earlier count inflation
   came from track fragmentation, not detector FPs.
4. **Localization, not detection, is the weak axis.** AP@0.50 = 0.459 but AP@0.75 = 0.097
   (YOLOv8m). Fingerprint of imprecise GT — confirmed in §4.2.

---

## 4. Evaluation-methodology results

### 4.0 Two INDEPENDENT axes — read this before §4.2/§4.3

A detection is scored wrong for two unrelated reasons, and they need different fixes:

| Axis | Question | Fix | Needs retraining? |
|---|---|---|---|
| **A. Matching strictness** | GT box is on the deer, but is the prediction "close enough"? IoU≥0.5 says a loose box fails. | permissive matching (`touch` = any overlap) — §4.3 | **No.** Post-hoc metric only. |
| **B. GT correctness** | Is the GT box on the deer *at all*? 94% are interpolated; median drift 2.28 box-widths. | score against human keyframes — §4.2 | **No** for evaluation. Optional for training (see below). |

**Crucially, axis A cannot rescue axis B.** If an interpolated GT box has drifted a full
box-width or more off the animal (79% of segments), then a prediction sitting correctly
*on the deer* has **zero overlap with that GT box** — so it is scored as both a miss and a
false positive even under the most permissive "any overlap" rule. Loosening the matching
rule only helps where the GT box is still on the animal.

That is why §4.2 (keyframe GT) is reported at standard IoU≥0.5: it isolates axis B alone.
§4.3 isolates axis A. The full 2×2 grid (both axes, both GT sets) is job `20490065`.

**Does any of this require retraining? No.** All of it is measurement over already-saved
weights. Retraining is only worth considering for a *different* reason: the 94%
interpolated boxes are also noisy **training** labels, so training on cleaner/denser
labels could improve the model itself. That is an optional future experiment, not a
prerequisite for any number in this document.

### 4.1 Why mAP50-95 is the wrong headline for this paper

Deer are ~27 px. AP@0.75 requires box agreement within a few pixels, which is neither
achievable nor *needed* — counting requires knowing an animal is present, not its exact
outline. **Report mAP50 + counting MAE/RMSE; report mAP50-95 for completeness only.**
75% mAP50 is a plausible target; 75% mAP50-95 is not realistic for objects this small.

### 4.2 Ground-truth quality: keyframe-only evaluation ★ key result

Job `20490007`, 2026-07-25. Identical weights/images/code — only the GT changed. Scored on
**human-placed keyframe boxes only** (`data/dataset/yolo_v3_kf`: 182 images, 361 boxes).

| Model | mAP50 (full, 94% interpolated GT) | mAP50 (human keyframes) | Δ | P (kf) | R (kf) | mAP50-95 (kf) |
|---|---|---|---|---|---|---|
| YOLOv10m | 0.498 | **0.670** | **+35%** | 0.765 | 0.562 | 0.285 |
| YOLOv9m | 0.498 | **0.640** | **+29%** | 0.741 | 0.551 | 0.261 |
| YOLO11m | 0.466 | **0.636** | **+37%** | 0.807 | 0.573 | 0.234 |

**Interpretation:** roughly a third of the apparent detection error is an artefact of
interpolated ground truth drifting off the animals, not detector failure. True detector
quality is ≈ **0.64–0.67 mAP50**, and that is still the *strict* IoU≥0.5 figure.

**For the paper:** report both. "0.50 on full interpolated GT / 0.67 on human-verified
keyframes" demonstrates that the annotation pipeline was understood and its noise
quantified — a strength, not a weakness.

**Actionable:** denser keyframes on fast-moving deer would improve results more than any
architecture change. Labeling decision, not a GPU decision.

### 4.3 Counting-relevant matching (any-overlap criterion)

Rationale (project decision, 2026-07-25): for **counting**, a detection box that overlaps
the deer *at all* is a success — once the tracker links it, the animal is counted. Box
tightness is irrelevant downstream.

Implemented in `src/eval/track_recall.py` + `src/eval/counting_detection_eval.py` with four
criteria: `iou50` (standard), `iou30`, `touch` (any overlap), `center` (centre-in-box).

> **Reporting rule (do not violate):** publish `iou50` in the detection table. The
> permissive numbers belong to the *counting* evaluation and must be named
> "presence/counting recall" — **never** relabelled as mAP. mAP quoted at IoU>0 reads as
> metric gaming and will sink a review.

Results: **TODO** — job `20490065` pending. It runs the full 2×2 grid of §4.0:
{`iou50`, `iou30`, `touch`, `center`} × {full interpolated GT, human-keyframe GT} ×
{conf 0.25, 0.10}, for YOLOv9m / YOLOv10m / YOLO11m. The single number closest to "how
well can we actually find deer for counting" is **`touch` recall on keyframe GT**.

---

## 5. Pending / queued

| Job | Purpose | Status |
|---|---|---|
| `20490065` | counting-criterion detection eval (any-overlap), conf 0.25 & 0.10 | pending |
| `20487340` | **track-level recall gate** — of 236 GT deer, how many found in ≥1 / ≥3 frames, all 4 criteria, by split & site | pending |
| `20487321` | **1280 px ablation** — YOLOv10m + YOLOv9m, 70 epochs | pending (2×A100, 22 h) |

### Phase-B gate criterion (decided in advance)

- track-level recall **>90%** → detector is fine, proceed to the temporal counting head
- **70–90%** → proceed, state the ceiling honestly in the paper
- **<70%** → detector is the bottleneck; invest there first

### Deferred (built, tested, not run)

6 mmdetection models — Faster R-CNN R50, ATSS R50, TOOD R50, RTMDet-M, DINO-4scale R50,
Deformable-DETR R50 — configs in `configs/mmdet/`, env `envs/mmdet`, COCO checkpoints
staged in `weights_mmdet/`. Resubmit `scripts/train_mmdet_all.sbatch` when the broader
baseline table is needed. Deferred 2026-07-25 to prioritise the counting contribution.

---

## 6. Open risks for the paper

1. **Dataset scale.** 236 deer / 32 videos is thin for a benchmark contribution, and there
   is no unlabelled backlog — growth requires the RS footage (31 videos, SHW missing) or
   new collection.
2. **Annotation density.** 94% interpolated GT (§1.1) caps both training and evaluation
   quality.
3. **Site imbalance.** SHB 132 deer vs MAS 15 — per-site numbers for MAS are unstable.
4. **The detection table is not the contribution.** Temporal counting (MAE/RMSE, learned
   confirmation vs hand-tuned rules) is the headline; detection is supporting material.
5. **All pre-2026-07-25 counts are invalid** — measured before the CLAHE inference fix
   (§2, item 5).

---

## 7. Reproduction

```bash
# dataset (CLAHE frames from CVAT exports)
sbatch scripts/convert_cvat.sbatch
python src/dataset/build_yolo_dataset.py --images data/annotate_v2/frames \
  --labels data/annotate_v2/labels --out data/dataset/yolo_v3 \
  --val-keys <4 keys> --test-keys <9 keys>          # §1.2
python src/dataset/yolo_to_coco.py --root data/dataset/yolo_v3   # mmdet

# detector roster @640 (6 models, 4 GPUs)      -> §3
sbatch scripts/train_all.sbatch
# full metrics incl. COCO size-stratified      -> §3
sbatch scripts/eval_all.sbatch
# keyframe-only (human GT) evaluation          -> §4.2
sbatch scripts/eval_keyframe.sbatch
# counting-criterion evaluation                -> §4.3
sbatch scripts/counting_eval.sbatch
# track-level recall gate                      -> §5
sbatch scripts/track_recall.sbatch
# 1280 px ablation                             -> §5
sbatch scripts/train_res1280.sbatch
```

Outputs: `/work/hdd/bgte/tislam6/wildlife_outputs/{runs,logs}`, metrics under
`runs/detect/results/detection_eval/` and `results/`.

---

## Changelog

- **2026-07-25** — Added §4.0 clarifying that matching strictness and GT correctness are independent axes (permissive matching cannot rescue drifted GT); counting eval extended to the full 2x2 grid.
- **2026-07-25** — Created. Roster @640 results (§3); size-stratified + domain metrics;
  annotation-quality analysis (§1.1) and keyframe-only evaluation (§4.2, key result);
  counting-criterion framework (§4.3); pipeline fixes (§2); dataset corrected to 32 videos.
