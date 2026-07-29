# Results log — thermal deer detection & counting

**Living document.** Every new result gets appended here with its date, job ID, and the
command/config that produced it, so the paper can be written from this file alone.
Numbers here are *measured*, never estimated; anything unmeasured is marked TODO.

Last updated: **2026-07-26** (see [Changelog](#changelog))

## Headline numbers (as of 2026-07-26)

| What | Value | Where |
|---|---|---|
| Best detector (mAP50, test) | **YOLOv9m @1280 — 0.523** | §4.5 |
| Same model, human-verified GT | 0.640 @640 / see §4.5 for 1280 | §4.2 |
| Best counting-criterion P/R (keyframe GT) | **0.939 / 0.643** (YOLOv9m@1280, conf .25) | §4.5 |
| **Track-level recall (deer found at all)** | **97.0%** — 228/235 | §4.4 ★ |
| Deer never detected (counting floor) | 7/235 = 3.0% | §4.4 |
| **Counting MAE / RMSE (baseline)** | **2.28 / 3.74** — 215 of 236 deer (91.1%) | §4.7 ★ |

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

### 1.2b Video corpus scale (measured 2026-07-26)

| Quantity | Value |
|---|---|
| Videos | 32 |
| **Total frames** | **521,930** |
| Mean frames/video | ~16,300 |
| Largest videos | NWolfCreek(blue) 68k, NIron_SHW 36k, FernRidgeRd 35k, N25thBlue 25k |
| Frame rate / size | 60 fps, 640x512 grayscale |

Full-corpus inference costs ~2-4 GPU-hours at 1280 px (12-30 ms/frame incl. decode +
tracking) — cheap enough to re-run counting whenever the detector changes.

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

⚠ YOLOv8m: job `2728312` re-selected its checkpoint on the **val** split and found
epoch-0 genuinely IS its best (val mAP50 0.426 vs 0.412 @ep20, 0.388 @ep50). So the row is
**correct, not an artefact** — YOLOv8m peaks immediately and then degrades, plausibly
overfitting the 94%-interpolated labels (§1.1). Report as-is; the earlier "epoch-1 artefact"
caveat is withdrawn.

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

**RESULTS — jobs `20490065` (Delta) + `2728218` (DeltaAI GH200), 2026-07-25.** Full grid:
6 models × 4 criteria × {full GT, keyframe GT} × {conf 0.25, 0.10}. Paper-ready table:
`results/counting_eval/counting_eval_TABLE.md`; tidy CSV `counting_eval_ALL.csv`.

#### Headline cell — human-keyframe GT, any-overlap matching (the counting-relevant number)

| Model | conf 0.25 P / R / F1 | conf 0.10 P / R / F1 |
|---|---|---|
| **YOLO11m** | **0.942 / 0.634 / 0.758** | **0.861 / 0.720 / 0.784** |
| YOLOv12m | 0.937 / 0.535 / 0.681 | 0.773 / 0.737 / 0.755 |
| YOLOv9m | 0.936 / 0.565 / 0.705 | 0.787 / 0.684 / 0.732 |
| YOLOv10m | 0.901 / 0.557 / 0.688 | 0.684 / 0.737 / 0.709 |
| YOLOv8m ⚠ | 0.836 / 0.648 / 0.730 | 0.428 / 0.875 / 0.575 |
| RT-DETR-L | 0.818 / 0.720 / 0.766 | 0.401 / 0.792 / 0.533 |

#### Strict IoU≥0.50 on the same keyframe GT (for the standard detection table)

| Model | conf 0.25 P / R / F1 | conf 0.10 P / R / F1 |
|---|---|---|
| YOLO11m | 0.811 / 0.546 / 0.652 | 0.715 / 0.598 / 0.652 |
| YOLOv10m | 0.839 / 0.518 / 0.640 | 0.625 / 0.673 / 0.648 |
| YOLOv9m | 0.830 / 0.501 / 0.625 | 0.694 / 0.604 / 0.646 |
| YOLOv12m | 0.840 / 0.479 / 0.610 | 0.651 / 0.621 / 0.635 |
| RT-DETR-L | 0.704 / 0.621 / 0.660 | 0.330 / 0.651 / 0.438 |
| YOLOv8m ⚠ | 0.718 / 0.557 / 0.627 | 0.333 / 0.681 / 0.447 |

#### Findings

1. **Precision under the counting criterion is very high: 0.90–0.94** for the four good
   YOLO models at conf 0.25 on human GT. At operating confidence, ~19 of every 20 boxes
   the detector emits is genuinely on a deer.
2. **Loosening IoU 0.5 → any-overlap gains ~+0.10 precision at nearly constant recall**
   (YOLO11m 0.811 → 0.942, recall 0.546 → 0.634). So a large share of nominal "false
   positives" are boxes sitting on real deer that merely failed the tightness test —
   direct evidence that strict IoU is the wrong yardstick for this task.
3. **Both axes of §4.0 stack.** YOLO11m F1 rises 0.535 (full GT, IoU50, conf .25) →
   0.652 (keyframe GT, IoU50) → **0.758** (keyframe GT, any-overlap). GT quality and
   matching strictness contribute roughly equally.
4. **Confidence trades P for R as expected**; conf 0.10 is the better counting operating
   point (YOLO11m R 0.634 → 0.720 for −0.08 precision), consistent with the
   recall-limited regime of §3.
5. **YOLO11m is the best counting detector** despite ranking 4th on mAP50 in §3 — a
   reminder that the mAP ranking does not determine the counting ranking. **Use YOLO11m
   (or the 1280 winner) for the counting pipeline.**
6. ⚠ **RT-DETR-L and YOLOv8m degrade badly at conf 0.10** (precision 0.40 / 0.43 vs 0.79+
   for the rest). RT-DETR emits a fixed 300 queries with no NMS, so low-confidence
   detections proliferate; YOLOv8m's row is contaminated by the epoch-1 checkpoint
   (§3) — fix job `2728312` re-selects its checkpoint on the val split.




### 4.4 Track-level recall — THE PHASE-B GATE ★ key result

Job `2728289` (DeltaAI), 2026-07-25. YOLOv9m @640, conf 0.10, all 32 videos, 235 GT
deer tracks. Question: of the deer that exist, how many does the detector find *at all*?
A deer found in zero frames is unrecoverable by any downstream counting stage.

| Matching rule | found in >=1 frame | found in >=3 frames |
|---|---|---|
| strict IoU>=0.50 | 204/235 (86.8%) | 181/235 (77.0%) |
| IoU>=0.30 | 212/235 (90.2%) | 189/235 (80.4%) |
| **any overlap (counting)** | **220/235 (93.6%)** | 192/235 (81.7%) |
| centre-in-box | 219/235 (93.2%) | 192/235 (81.7%) |

**GATE PASSED (>90% criterion, pre-registered in §5).** The detector finds 93.6% of all
individual deer. Proceed to the counting harness and temporal head; further detector work
is optional, not blocking.

Per split (held-out **test** 94.7%, val 97.8%, train 92.1%) — the small train-split deficit
is the opposite of overfitting and reflects harder videos, not memorisation.

Per site: **SHW 100%** (38/38), MAS 100%, SHB 93.9%, **TON 86.0%** (43/50) — TON is the
weakest site and the only one below 90%; worth a sentence in the paper and a candidate for
targeted annotation.

**Ceiling implication:** ~6% of deer are invisible to the detector at conf 0.10, so counting
MAE has a hard floor of roughly 6% under-count unless recall improves (higher resolution,
more data, or lower confidence with the temporal head filtering).


### 4.5 Resolution ablation: 640 vs 1280 ★ key result

Jobs `2728654` (train, DeltaAI GH200) + `2729221` (eval), 2026-07-26. 70 epochs,
batch 32, same pooled split. **Every metric below is computed at the model's own
training resolution** — scoring a 1280 model at 640 would understate it.

| Model | res | P | R | mAP50 | mAP50-95 | AP-small |
|---|---|---|---|---|---|---|
| YOLOv9m | 640 | 0.704 | 0.448 | 0.498 | 0.184 | 0.128 |
| **YOLOv9m** | **1280** | **0.757** | **0.491** | **0.523** | **0.199** | 0.119 |
| YOLOv10m | 640 | 0.674 | 0.455 | 0.498 | 0.197 | 0.117 |
| YOLOv10m | 1280 | 0.670 | 0.432 | 0.458 | 0.163 | 0.093 |

**The ablation is architecture-dependent — that is the finding.** 1280 helps YOLOv9m
(+0.025 mAP50, +0.043 recall, +0.053 precision) and *hurts* YOLOv10m (−0.040 mAP50).
At 640 the two were tied at 0.498; at 1280 they diverge by 0.065. Do not report
"higher resolution helps" as a general claim.

Note AP-small does **not** improve for either model even where overall mAP50 does
(YOLOv9m 0.128 → 0.119), so the 640 AP-small/AP-medium gap of §3 was **not** primarily
a resolution deficit. The gain at 1280 comes from better recall and precision overall,
not from rescuing the smallest deer.

#### Track-level recall at 1280 — the counting-relevant gain

| Matching rule | 640 | **1280** | Δ |
|---|---|---|---|
| strict IoU>=0.50 | 86.8% | **95.7%** | +8.9 |
| **any overlap (counting)** | 93.6% | **97.0%** | **+3.4** |
| any overlap, >=3 frames | 81.7% | **91.1%** | +9.4 |

**Deer never detected at all: 15/235 (6.4%) → 7/235 (3.0%) — halved.** The hard floor on
counting under-count drops from ~6% to ~3%. The >=3-frame figure (81.7% → 91.1%) matters
even more for the temporal head, which needs several observations per animal to confirm
a track.

#### Counting-criterion detection, YOLOv9m @1280 (human-keyframe GT)

| conf | IoU>=0.50 P/R/F1 | any-overlap P/R/F1 |
|---|---|---|
| 0.25 | 0.866 / 0.593 / 0.704 | **0.939 / 0.643 / 0.763** |
| 0.10 | 0.792 / 0.673 / 0.728 | — |

vs the best 640 model (YOLO11m): 0.942 / 0.634 / 0.758. Essentially tied on precision,
slightly better recall — but YOLOv9m@1280 wins decisively on the metric that governs
counting, track-level recall (97.0% vs 93.6%).

**Decision: use YOLOv9m @1280 as the counting detector.** 640 remains the cheaper option
if inference cost matters (1280 is ~4x the pixels, ~1.6x the epoch time measured here).

### 4.6 Counting evaluation protocol (Phase B)

The paper's headline metric, implemented in `src/eval/count_eval.py`:
**MAE / RMSE / signed bias** between predicted confirmed-track counts and the CVAT
track count, per video and per site, with an explicit over-count / under-count split.

The comparison baseline is the **best hand-tuned confirmation rule**: `count_deer.py`
dumps every candidate track's statistics, so any rule
`n_frames >= min_hits AND span_s >= min_span_s AND topk_conf >= conf_track` can be
re-applied post-hoc with no GPU re-run. We sweep 288 combinations and report the best.
**The sweep tunes on the evaluation data deliberately** — it makes the hand-tuned
baseline as strong as possible, so the learned temporal head beating it is meaningful
rather than a straw-man comparison.

Detector for counting: **YOLOv9m @1280** (§4.5), chosen on track-level recall (97.0%),
not mAP. Detection runs at conf 0.10 — the regime is recall-limited (§3), so candidates
are over-generated and confirmation happens downstream.

Results: see §4.7.

### 4.7 Counting results — the baseline to beat ★ HEADLINE

Job `2730226` (DeltaAI, 1 h 17 m), 2026-07-26. YOLOv9m@1280, conf 0.10, BoT-SORT with
camera-motion compensation, all 32 videos / 521,930 frames. **First trustworthy counts
in the project** — everything before the CLAHE-at-inference fix (§2 #5) was measured on
the wrong image distribution.

Best hand-tuned rule (swept over 288 configs, tuned on this data to make the baseline
as strong as possible): `span_s >= 0.3`, `topk_conf >= 0.65`, `min_hits` irrelevant.

| Scope | Videos | MAE | RMSE | Bias | Over | Under |
|---|---|---|---|---|---|---|
| **ALL** | 32 | **2.28** | **3.74** | −0.66 | 26 | 47 |
| MAS | 8 | 0.25 | 0.50 | −0.25 | 0 | 2 |
| TON | 8 | 2.50 | 4.50 | −2.50 | 0 | 20 |
| SHB | 8 | 2.88 | 3.59 | −0.38 | 10 | 13 |
| SHW | 8 | 3.50 | 4.74 | +0.50 | 16 | 12 |

**Total: 215 predicted vs 236 true (91.1%).**

#### Findings

1. **MAE 2.28 deer/video is the number the temporal head must beat.** RMSE 3.74 >> MAE
   means the error is concentrated in a few bad videos, not spread evenly — see (4).
2. **`min_hits` has NO effect** (MAE identical for 1..30). Confidence and duration do all
   the filtering work. This is direct evidence that the hand-tuned rule is a blunt
   instrument: one of its three knobs is inert, and the other two are shared thresholds
   applied identically to a 27-px distant deer and a large near one. Good motivation for
   the learned head.
3. **Errors are bidirectional and site-dependent** — TON under-counts badly (−2.5/video,
   20 missed, 0 over), SHW over-counts (+0.5, 16 spurious). A single global rule cannot
   fix both directions at once; a learned per-track decision can.
4. **Worst videos:** NatureTrail_TON 15 GT → 4 predicted (−11), GiantCityRd_TON 8 → 2
   (−6); over-count side SBassett_SHW 11 → 22 (+11), EagleCreek_SHB 44 → 50 (+6).
   Note TON was also the weakest site in the detection gate (§4.4, 86% track recall),
   so its under-count is partly a detector deficit; SHW's over-count is fragmentation
   (SHW had 100% track recall), i.e. exactly the re-ID head's job.
5. Under-counting (47) exceeds over-counting (26): the pipeline is conservative overall.

**These four mechanisms — inert knobs, bidirectional site-dependent error, fragmentation
over-count, merged/missed under-count — are the specific failures the temporal head is
designed to fix, and they are now measured rather than assumed.**

---

## 5. Status of runs

### Completed

| Job | Cluster | Purpose | Result |
|---|---|---|---|
| `20477045` | Delta | 6-model roster @640, 150 ep | §3 |
| `20486592` | Delta | full metrics incl. COCO size-stratified | §3 |
| `20490007` | Delta | keyframe-only evaluation | §4.2 ★ |
| `20490065`+`2728218` | both | counting-criterion table, 6 models | §4.3 ★ |
| `2728289` | DeltaAI | track-level recall gate | §4.4 ★ |
| `2728312` | DeltaAI | yolov8m checkpoint re-selection | §3 note |
| `2728654`+`2729221` | DeltaAI | 1280 ablation + full eval battery | §4.5 ★ |
| `2728181` | DeltaAI | GH200 environment smoke test | 0.6696 vs 0.670 on A100 |

### Running / queued

| Job | Cluster | Purpose |
|---|---|---|
| `2730226` | DeltaAI | **Phase B counting run** — ✅ done in 1 h 17 m, §4.7 |
| `20493649` | Delta | hedge twin — both started before either could be cancelled; the
Delta copy was killed once the DeltaAI results landed (~1.5 GPU-h duplicated) |

Both are 4-GPU / 3-hour requests. Queue note (2026-07-26): DeltaAI had **zero idle
nodes** and the job's priority was 1,228 vs 11,265 at the queue head — fairshare
(1,115 of the 1,228) dominates and falls with recent account usage. Job size was NOT
the constraint: 1-, 2- and 4-GPU requests all returned identical start estimates.

### Phase C — built, awaiting the counting run

`src/temporal/build_track_labels.py` labels every candidate track against the CVAT GT
and emits supervision for all three heads: `is_real` (confirmation), `gt_track` +
`gt_coverage.csv` (re-ID / fragmentation = the over-count mechanism), `multiplicity`
(merged blobs = the under-count mechanism). Matching uses the counting criterion over
>=3 co-occurring frames; `--keyframes-only` allows ablating the 94%-interpolation drift
(§1.1). Verified on synthetic GT (clean track -> real/mult 1; distant noise -> false;
blob spanning two deer -> real/mult 2).

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

## 6. Counting (Phase B/C) — the paper's headline metric

Pipeline: detector -> BoT-SORT (+camera-motion compensation) -> confirmation rule/head.
Scored as MAE/RMSE between predicted and CVAT unique-deer counts, per video.

### 6.1 Baseline: tracking-by-detection + best hand-tuned rule

Job `2739440` (DeltaAI), YOLO11m@640, detector conf 0.10, rule swept on the same data
(deliberately optimistic for the baseline).

| Scope | Videos | MAE | RMSE | Bias | Over | Under |
|---|---|---|---|---|---|---|
| **ALL** | 32 | **1.88** | 3.04 | -1.12 | 12 | 48 |
| MAS | 8 | 0.50 | 1.00 | -0.50 | 0 | 4 |
| SHB | 8 | 2.00 | 3.00 | -1.50 | 2 | 14 |
| SHW | 8 | 2.38 | 3.52 | +0.12 | 10 | 9 |
| TON | 8 | 2.62 | 3.82 | -2.62 | 0 | 21 |

200/236 deer (84.7%). Mean 7.4 deer/video, so MAE 1.88 ~ 25% relative error; 41% of
videos exact, 59% within +-1, 75% within +-2. Error is **systematic under-counting**
(48 under vs 12 over) — the learnable failure mode the temporal head targets.

**Confidence note:** conf 0.10 is the *candidate-generation* threshold, not the
counting threshold. 984 candidates were generated, 784 (80%) discarded; the 200 counted
tracks had min 0.65 / median 0.74 top-k confidence. No track exceeded 0.90 — expected
for 27 px objects.

### 6.2 Learned confirmation does NOT beat the rule (cross-validated) ★

Job `20551694`. 8-fold CV over videos, all 30 videos with tracks / 236 deer. Rule swept
per fold on that fold's training videos; head threshold picked on an inner val split.
Identical protocol for every method.

| Method | MAE | RMSE | Bias | Under |
|---|---|---|---|---|
| Hand-tuned rule | 2.23 | 3.23 | -0.97 | 48 |
| **Gradient boosting** | **2.17** | 3.45 | -1.03 | 48 |
| Logistic regression | 2.83 | 4.10 | -0.97 | 57 |
| TTC transformer | 2.90 | 4.05 | -1.63 | 68 |

GBM ties (+3%, within noise). **Diagnosis:** rule and GBM under-count by *exactly* 48
because ~45 deer had no candidate track at all — you cannot select your way to a deer
that is not in the candidate set. Confirmation was never the binding constraint.

Two fixes were needed to reach even this: val-based **early stopping** (an earlier
version trained all 150 epochs and overfit ~500 tracks, scoring *worse* than the rule)
and **cross-track context** features (a per-track model cannot tell a `primary` from a
`duplicate` fragment — they are identical in isolation — so v1 accepted both and
over-counted, +1.11 bias).

### 6.3 The binding constraint is CANDIDATE GENERATION ★★ key result

Ceiling = MAE achievable by a *perfect* confirmer on a given candidate pool.

| Candidate pool | Tracks | Deer with a track | Ceiling MAE | Rule MAE |
|---|---|---|---|---|
| YOLO11m@640, conservative tracker | 711 | 190/236 | 1.44 | 1.88 |
| + recall-first tracker, keep n<3 | 1194 | 194/236 | 1.31 | 1.97 |
| YOLOv9m@1280 (best *detector*) | 1496 | 188/236 | 1.35 | 2.38 |
| **+ orphan-detection recovery** | **7008** | **207/236 (88%)** | **0.91** | 1.88 |

1. **The best detector produced the worst counts.** YOLOv9m@1280 wins on detection
   (keyframe mAP50 0.735 vs 0.640) yet counts worst. Cause is fragmentation, not
   detection: duplicates per deer rose 0.78 -> 1.08 -> **1.28** because association
   thresholds tuned at 640 mis-associate at 2x pixel scale. **Frame-level detection
   quality does not translate into counting accuracy** — a reportable finding.
2. **Loosening the tracker barely helped**: +68% tracks bought only +4 deer.
3. **The real cause was a pipeline bug.** `count_deer.py` had
   `if b.id is None: continue`, discarding *every* detection in frames where the tracker
   assigned no ID. BoT-SORT only activates a track after association across >=2 frames,
   so deer seen in a single isolated frame were deleted before counting. That is exactly
   the gap between 93.6% of deer being detected and ~80% forming a track.
   `--keep-orphans` retains them and links them into pseudo-tracks: **+17 deer**, more
   than tracker tuning and the 1280 detector achieved combined.

**Consequence for the contribution:** headroom for a learned confirmer went from 0.44
(1.88 -> 1.44) to **0.97** (1.88 -> 0.91), and the pool is now 7008 candidates at **91%
false** — a regime where threshold rules are hopeless and a learned confirmer is the
only workable option. Retraining on this pool: job `20565413`.

### 6.4 How close to 236 is reachable?

| Stage | Deer |
|---|---|
| Ground truth | 236 |
| Detected in >=1 frame (conf 0.10) | 220 (93.6%) |
| Recovered as a track (orphan pool) | 207 (88%) |

~13 deer are detected but still lost in track formation (recoverable by pipeline work);
~16 are **never detected** at conf 0.10 and are unreachable without a better detector.
Job `2764134` sweeps detectability at conf 0.05/0.02 for both detectors to set the hard
limit before committing GPU to ensembling or SAHI tiling.

---

## 7. Open risks for the paper

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

## 8. Reproduction

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
# track-level recall gate                      -> §4.4
sbatch scripts/dtai_track_recall.sbatch      # DeltaAI variant
# 1280 px ablation + its full eval battery     -> §4.5
sbatch scripts/dtai_train_1280.sbatch
sbatch --dependency=afterany:<trainjob> scripts/dtai_eval_1280.sbatch

# --- Phase B: counting (the headline metric) -> §4.6 ---
sbatch scripts/dtai_counting.sbatch          # or scripts/delta_counting.sbatch
python src/eval/count_eval.py --counts results/counts/yolov9m_1280_conf0.10

# --- Phase C: temporal-head training labels ---
python src/temporal/build_track_labels.py --counts results/counts/yolov9m_1280_conf0.10
python src/temporal/build_track_labels.py --counts <same> --keyframes-only \
    --out <dir>/labels_kf                    # ablation vs interpolation drift
```

Outputs: `/work/hdd/bgte/tislam6/wildlife_outputs/{runs,logs}`, metrics under
`runs/detect/results/detection_eval/` and `results/`.

---

## Changelog

- **2026-07-29** — Counting results (§6). Baseline MAE 1.88; learned confirmation ties
  the rule (§6.2); the binding constraint is candidate generation, and an orphan-detection
  bug was deleting single-frame deer — fixing it recovered 17 deer and cut the ceiling
  from 1.44 to 0.91 (§6.3). 1280 is the best detector but the worst counter (§6.3).

- **2026-07-26** — Phase-B gate PASSED: 93.6% track-level recall under the counting
  criterion (§4.4) -> detector is not the counting bottleneck, proceed to the temporal
  head. YOLOv8m epoch-1 checkpoint verified as genuine (§4.5). 1280 ablation OOM-killed
  (host RAM) and resubmitted with 440 GB / 4 workers.

- **2026-07-26** — ★ Phase B complete (§4.7): counting baseline MAE 2.28 / RMSE 3.74,
  215/236 deer. `min_hits` proven inert; errors bidirectional and site-dependent.

- **2026-07-26** — Phase B counting harness built (§4.6) and Phase C track-labelling
  verified; corpus scale measured (§1.2b, 521,930 frames); run-status section rewritten.

- **2026-07-26** — Resolution ablation (§4.5): YOLOv9m@1280 is the new best (test mAP50 0.523, track-level recall 97.0%, deer-never-seen halved to 3.0%); 1280 HURTS YOLOv10m (0.458). Phase B counting harness built and running.

- **2026-07-25** — Full counting-criterion detection table for all 6 models (§4.3): YOLO11m best for counting at 0.942 P / 0.634 R (keyframe GT, any-overlap, conf 0.25); moved compute to DeltaAI GH200.

- **2026-07-25** — Added §4.0 clarifying that matching strictness and GT correctness are independent axes (permissive matching cannot rescue drifted GT); counting eval extended to the full 2x2 grid.
- **2026-07-25** — Created. Roster @640 results (§3); size-stratified + domain metrics;
  annotation-quality analysis (§1.1) and keyframe-only evaluation (§4.2, key result);
  counting-criterion framework (§4.3); pipeline fixes (§2); dataset corrected to 32 videos.
