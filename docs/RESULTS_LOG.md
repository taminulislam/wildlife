# Results log — thermal deer detection & counting

**Living document.** Every new result gets appended here with its date, job ID, and the
command/config that produced it, so the paper can be written from this file alone.
Numbers here are *measured*, never estimated; anything unmeasured is marked TODO.

Last updated: **2026-07-30** (see [Changelog](#changelog))

## Headline numbers (as of 2026-07-30)

| What | Value | Where |
|---|---|---|
| Best detector (mAP50, test) | **YOLOv9m @1280 — 0.523** | §4.5 |
| Best non-Ultralytics detector | RTMDet-m — 0.466 (DINO only 0.365) | §3.1 |
| **Detector ranking on the COUNTING criterion** | YOLO11m F1 **0.758**; no mmdet model displaces it | §3.2 ★ |
| Same model, human-verified GT | 0.640 @640 / see §4.5 for 1280 | §4.2 |
| Best counting-criterion P/R (keyframe GT) | **0.939 / 0.643** (YOLOv9m@1280, conf .25) | §4.5 |
| **Track-level recall (deer found at all)** | **97.9%** — 230/235 (loose NMS) | §4.4, §6.8 ★ |
| Deer never detected (counting floor) | 5/235 = 2.1% | §4.4, §6.8 |
| **Candidate reach, UNSEEN videos** | **95.2%** — 79/83 deer (vs 98.0% seen) | §6.4.3 ★★ |
| Deer with their OWN candidate (ceiling) | 215/235 overall; **69/83 unseen** | §6.8 ★★ |
| **COUNTING, HELD OUT — the number to publish** | **MAE 2.38, 55/83 = 66.3%** of unseen deer | §6.7.1 ★★★ |
| Unseen bias | **-1.92** — under-counts, never over-counts | §6.7.1 |
| Counting over all 32 videos (optimistic) | 89.0%, MAE 1.88 — includes 19 detector-training videos | §6.1, §6.7 |
| **Best candidate pool (ReID)** | reached **233/235**, primary **219**; unseen **82/83** | §6.9 ★★ |
| **The gap the paper must close** | unseen: reach **98.8%** → primary **88.0%** → counted **62.7%** | §6.9 ★★★ |
| Individual ReID at 27 px | rank-1 0.899, but box size alone 0.638 — **not identification** | §6.9 |
| Learned confirmation vs rule (CV) | rule 1.88 vs capacity-matched 1.97 (better RMSE 2.78) | §6.5 |
| **Learned head, HELD OUT** | **loses (2.85) / ties (2.77) — does NOT beat rule 2.38** | §6.10 ★★★ |
| **Per-deer calibrated confidence >= 0.80** | **92.1%** of counted deer (mean 0.965) | §6.6 ★ |

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

### 3.1 mmdetection roster (job `20565690`, finished 2026-07-30)

Six architectures, same pooled split, COCO-pretrained, 1 class, CLAHE frames, 4×A100.
Wall clock 19 h 33 m. **COCO mAP50 on the held-out test split** — i.e. IoU>=0.50, *not*
the project's any-overlap criterion:

| Model | schedule | val best mAP50 | **test mAP50** |
|---|---|---|---|
| **RTMDet-m** | 70e | 0.490 | **0.466** |
| Faster R-CNN R50 | 70e | 0.418 | 0.418 |
| TOOD R50 | 70e | 0.401 | 0.396 |
| ATSS R50 | 70e | 0.380 | 0.380 |
| DINO R50 | 36e (native) | 0.365 | 0.365 |
| Deformable DETR R50 | 50e | 0.113 @ep4 | ✗ CUDA OOM at epoch 5 |

Findings:

1. **RTMDet-m (0.466) is level with the Ultralytics roster** (best 0.498) and beats every
   other mmdet model by 5+ points. The convolutional one-stage family transfers to 27 px
   thermal deer; nothing here displaces YOLO, but nothing is embarrassed by it either.
2. **The DETR family underperforms.** DINO R50 reaches only 0.365 on a full native
   36-epoch schedule, below plain Faster R-CNN. Consistent with the known weakness of
   query-based detectors on very small objects, and worth one sentence in the paper.
3. **Deformable DETR OOM'd** at batch 8 on a 40 GB A100 (multi-scale deformable attention
   peaks well above its ~29 GB steady state). A batch-4 config exists
   (`configs/mmdet/deer_deformable-detr_r50.py`, lr scaled to 2.5e-5) but the rerun was
   **dropped**: DINO R50 already represents the DETR family on a full native schedule and
   finishes below Faster R-CNN, so a second query-based detector cannot change any
   conclusion. Reported as "excluded — did not converge within compute budget".

⚠ These numbers are on the wrong criterion for this project — mmdet's training loop only
emits COCO mAP. Job `20568689` rescored all five through
`counting_detection_eval.py --arch mmdet`; §3.2 is the number that counts.

#### 3.2 mmdet under the ANY-OVERLAP criterion (job `20568689`, finished 2026-07-30) ★

Human-keyframe GT, conf 0.25, P / R / F1 — one ranking with the Ultralytics roster
(`results/counting_eval/counting_eval_TABLE.md`):

| Model | IoU≥0.50 F1 | **any-overlap F1** | any-overlap P | any-overlap R |
|---|---|---|---|---|
| **YOLO11m** @640 | 0.652 | **0.758** | 0.942 | 0.634 |
| RT-DETR-l | 0.660 | 0.766 | 0.818 | 0.720 |
| ATSS R50 | 0.550 | 0.729 | 0.714 | 0.745 |
| TOOD R50 | 0.548 | 0.722 | 0.729 | 0.715 |
| Faster R-CNN R50 | 0.525 | 0.636 | 0.511 | 0.842 |
| RTMDet-m | 0.526 | 0.634 | 0.495 | **0.881** |
| DINO R50 | 0.509 | 0.635 | 0.631 | 0.640 |

Findings:

1. **The criterion change does not overturn the roster choice.** YOLO11m keeps the best
   any-overlap F1 among the trained-from-scratch models; no mmdet architecture displaces
   it. The detector decision is final — 13 models compared, on the project's own criterion.
2. **RTMDet-m's mAP lead was a precision artefact.** It ranked first in §3.1 on COCO mAP50
   (0.466) but falls to 0.634 F1 here: it has the *highest recall of any model tested*
   (0.881) and the worst precision (0.495). Under any-overlap the loose boxes that COCO
   mAP punished stop being penalised, and the false positives dominate instead.
3. **mmdet models are recall-heavy, YOLO is precision-heavy** — a genuinely useful figure
   for the paper, and a hint that an mmdet model could be the better *candidate generator*
   for the counting stage even though it is the worse *detector*. Not pursued: Phase F
   already showed that a larger candidate pool hurts the rule-based counter (§6.8).

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

**Nothing is running as of 2026-07-30 04:11.** Both clusters' queues are empty; every
experiment the paper needs has reported. Remaining work is analysis and writing, not
compute — see §9.

Last to finish:

| Job | Cluster | Purpose | Result |
|---|---|---|---|
| `2777214` | DeltaAI | loose-NMS counting pool (NMS IoU 0.90) | §6.8 — mechanism confirmed, gain too small; Phase E retained |
| `20568689` | Delta | mmdet roster rescored on any-overlap | §3.2 — YOLO11m holds |
| `20568561` | Delta | Deformable DETR batch-4 rerun | **cancelled** — cannot change a conclusion (§3.1) |
| `2730226` | DeltaAI | Phase B counting run — done in 1 h 17 m | §4.7 |
| `20493649` | Delta | hedge twin of the above; killed once DeltaAI landed (~1.5 GPU-h duplicated) | — |

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

### mmdetection roster — ✅ complete (was deferred 2026-07-25)

6 configs in `configs/mmdet/`, env `envs/mmdet`, COCO checkpoints in `weights_mmdet/`.
5 of 6 trained and scored on both criteria (§3.1 COCO mAP, §3.2 any-overlap);
Deformable-DETR excluded for non-convergence within budget. Nothing outstanding.

---

## 6. Phase C — candidate generation is the binding constraint

> **Reconciliation with §4.7.** Two hand-tuned baselines exist because they use different
> detectors, and the difference IS a finding:
> * §4.7 — **YOLOv9m@1280**, MAE **2.28**, 215/236 (job `2730226`)
> * §6.1 — **YOLO11m@640**, MAE **1.88**, 200/236 (job `2739440`)
>
> The 640 detector counts BETTER despite being the weaker detector (§3, §4.5). See §6.3:
> higher resolution increases fragmentation, so detection quality and counting quality
> diverge. Quote **1.88** as the baseline to beat (it is the stronger baseline, i.e. the
> more honest bar for the contribution), and report 2.28 as the 1280 comparison point.

### 6.1 Baseline: tracking-by-detection + best hand-tuned rule (YOLO11m@640)

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

#### 6.4.1 Detectability sweep (job 2764134) — the detector is not the wall

Of 235 GT deer, how many does the raw detector see in **at least one frame**:

| Detector | conf | IoU>=0.50 | any overlap (counting) |
|---|---|---|---|
| YOLO11m @640 | 0.05 | 226 (96.2%) | **232 (98.7%)** |
| YOLO11m @640 | 0.02 | 230 (97.9%) | 232 (98.7%) |
| YOLOv9m @1280 | 0.02 | 230 (97.9%) | **233 (99.1%)** |

Only 2–3 deer are genuinely invisible. Everything else is a pipeline loss.

#### 6.4.2 Lowering the DETECTOR threshold recovers nothing (job 2764510)

Re-ran counting at conf 0.05 (from 0.10), orphans on:

| conf | candidate tracks | deer covered |
|---|---|---|
| 0.10 | 7 008 | 207 / 235 |
| 0.05 | 10 815 | **207 / 235** |

3 807 extra candidates, **zero** extra deer. This rules the detector threshold out as the
cause of the gap and points at BoT-SORT's `new_track_thresh: 0.15`: a detection below it
can extend a track but can never *start* one, so faint deer never become candidates.

#### 6.4.3 Removing the track-init gate + fixing the ceiling measurement ★★

`botsort_deer_maxrecall.yaml` (`new_track_thresh` 0.15 -> 0.05, detector conf 0.02),
job `2765525`: 18 349 candidates, of which **17 229 are orphan pseudo-tracks** — orphan
recovery, not the tracker, is generating almost the whole pool.

Then a measurement bug surfaced. `label_tracks.py` reports `len(set(best_for_gt.values()))`
as the ceiling, but that is a set of **candidates**, not deer: when one candidate track is
the best match for two deer (a track drifting between animals, or two deer walking
together) the set collapses it and both deer bill as one. `src/eval/pool_coverage.py`
separates the two:

| Split | Videos | GT | **reached** (>=1 touching candidate) | primary (distinct candidates) | lost to collision |
|---|---|---|---|---|---|
| train (detector saw) | 19 | 152 | 149 (98.0%) | 141 (92.8%) | 8 |
| **unseen (val+test)** | **13** | **83** | **79 (95.2%)** | **69 (83.1%)** | **10** |
| ALL | 32 | 235 | **228 (97.0%)** | 210 (89.4%) | 18 |

Two conclusions, both load-bearing for the paper:

1. **Candidate generation is solved** — 228 of 235 reachable against the 232 the detector
   sees, and the seen/unseen gap is only 98.0% vs 95.2%. This supersedes the 69.9%
   generalisation figure in §6.7, which came from the old conf-0.10 pool.
2. **The residual is 18 COLLIDED deer** that share a candidate with another animal — on
   the unseen videos, 10 of the 14 misses. No amount of extra candidate generation,
   lower thresholds, or orphan recovery reaches them: the candidate already exists and is
   already counted once. §6.8 works through what does.

⚠ These numbers are the CORRECTED ones. `label_tracks.py` and `pool_coverage.py`
originally loaded tracks as `{frame: box}`, but a candidate can hold several boxes in one
frame (the orphan linker groups simultaneous detections). Last-write-wins silently dropped
**27 655 of 117 978 boxes**, with the survivor decided by CSV row order, and 20.2% of
candidates were affected. Both now use `{frame: [box, ...]}`. Effect on the pool: primary
212 -> **210**, collisions 16 -> **18**; `reached` was unchanged at 228.

### 6.8 The collided deer: three fixes tried, two rejected on evidence ★

**Rejected — learned count-per-track head.** `label_track_counts.py` labels each candidate
with how many deer it is the best cover for. The distribution makes it untrainable:

| count | tracks |
|---|---|
| 0 | 18 137 |
| 1 | 196 |
| 2 | 16 |

Two positives per CV fold. Kept in the repo as the right target if the corpus grows.

**Rejected — temporal track splitting.** `split_tracks.py` cuts a track wherever the
size-normalised per-frame displacement exceeds a threshold (deer move ~0.15 box-widths
per frame here, so a multi-box-width jump is the tracker changing its mind). Swept
1.0–8.0; best case **+3 deer for 4 000 extra candidates**.

**Why both failed — the collisions are not temporal.** Measured per collision:

| kind | deer lost |
|---|---|
| **SIMULTANEOUS** — one box on two GT deer in the *same* frames (up to 116) | **15** |
| sequential — track drifts from deer A onto deer B | 1 |

Eight of the sixteen sit in one dense-group video, `NShelbyRd(blue)_SHB`. Only 3 of the 16
collision tracks hold multiple boxes per frame, so this is not the orphan linker either —
it is genuinely one detection covering two animals.

**Tried, and it works — but barely: loose NMS (job `2777214`, finished 2026-07-30).**
Two adjacent 27 px deer produce boxes overlapping above the 0.5 NMS IoU threshold, so the
second is suppressed at the detector and never reaches the tracker. Re-ran at NMS IoU 0.90
(and conf 0.02), everything else identical to Phase E:

| pool | candidates | reached | primary | lost to collision |
|---|---|---|---|---|
| Phase E — NMS 0.50 | 18 349 | 228 | 210 | 18 |
| **Phase F — NMS 0.90** | **90 239** | **230** | **215** | **15** |

Per split (`unseen` = val + test, the 13 videos the detector never saw):

| split | GT | Phase E primary | Phase F primary |
|---|---|---|---|
| train | 152 | 141 | 145 |
| **unseen** | **83** | **69** | **70** |

**Verdict: the NMS hypothesis is confirmed in direction and rejected in practice.** The
mechanism is real — loosening suppression separates deer that were genuinely one box — but
it buys **+5 primaries overall and +1 on the unseen videos for 5× the candidates**, and the
rule-based MAE *worsens* on the bigger pool (**3.41** over all 32 videos, vs **2.12** on
Phase E and **1.88** on the Phase-C orphan pool) because 72 000 extra false candidates
swamp a 3-parameter filter. **Phase E stays the reachability pool; Phase F is an ablation.**

That MAE column is itself the central result of §6.5, restated. Rule MAE tracks pool
*size*, not pool *quality*, and moves in the opposite direction to the reachable ceiling:

| pool | candidates | reached | primary | rule MAE (32 vids) |
|---|---|---|---|---|
| Phase C orphan (NMS 0.50, conf 0.10) | 7 008 | 222 | 207 | **1.88** |
| Phase E max-recall (conf 0.02) | 18 349 | 228 | 210 | 2.12 |
| Phase F loose NMS (0.90) | 90 239 | **230** | **215** | 3.41 |

Every extra deer made reachable costs the hand-tuned rule more than it gains, because a
3-parameter filter has no way to exploit a pool it cannot discriminate. Separating the
signal from the noise in a large candidate pool is precisely what a learned confirmer is
*for* — this table is the argument for the paper's contribution, stated as a measurement.

The caveat flagged before the run is now the conclusion: under any-overlap, a box that
merely *touches* a neighbouring deer's box already "covers" it, so most of the remaining 15
collisions are nominal rather than real double-coverage. **The honest ceiling is ~215, not
228** — and on unseen video, ~70/83 (84.3%).

### 6.9 Appearance ReID: best pool ever built, and the rule still cannot use it ★★★

Requested by a co-author: a thermal ReID keyed on body shape, antler structure, thermal
silhouette, relative size and movement. Three experiments answered it.

**(a) Can appearance identify individual deer at all?** (job `2788117`, +
`src/reid/reid_feasibility.py`). CVAT tracks are identity labels, so this is textbook
closed-set ReID: gallery = first half of each track, query = second half, rank-1 on **69
deer never seen, in 7 videos never seen**.

| Cue | rank-1 |
|---|---|
| chance | 0.101 |
| brightness alone | 0.101 — *no signal whatever* |
| thermal silhouette alone (scale removed) | 0.435 |
| raw thermal crop | 0.493 |
| **box size alone** | **0.638** |
| ImageNet ResNet-50 features | 0.725 |
| geometry bundle (size, aspect, intensity) | 0.797 |
| **trained thermal encoder** (job `2788317`) | **0.899** |

The 0.899 does not mean individual identification. Box size alone reaches 0.638, and
within one video a deer's range changes slowly, so most of the score is matching *distance*.
Decisively: an off-the-shelf CNN scores **below plain geometry**, and brightness is exactly
chance. At a median 29×24 px there is not enough detail to key an identity to.

**(b) Does appearance association improve counting?** (job `2788316`,
`src/track/botsort_deer_reid.yaml` — byte-identical to the max-recall config except
`with_reid: True`, so this is a true single-variable experiment.)

| pool | candidates | reached | primary | unseen reached | unseen primary | **held-out counted** |
|---|---|---|---|---|---|---|
| C orphan | 7 008 | 222 | 207 | 74/83 | 66/83 | **55/83 = 66.3%** |
| E max-recall | 18 349 | 228 | 210 | 79/83 | 69/83 | 54/83 = 65.1% |
| F loose NMS | 90 239 | 230 | 215 | 79/83 | 70/83 | 41/83 = 49.4% |
| **G ReID** | **27 679** | **233 (99.1%)** | **219 (93.2%)** | **82/83 (98.8%)** | **73/83 (88.0%)** | 52/83 = 62.7% |

**ReID produces the best candidate pool in the project by every measure** — 233/235 reached,
219 primaries, 82 of 83 unseen deer reached — and it does so at 27 679 candidates, less than
a third of Phase F's 90 239 for four *more* primaries. Appearance association is genuinely
working: it stops one animal fragmenting into several tracks.

**And the counted total still falls**, 55 → 52 of 83. This is the third independent
confirmation of §6.7.1: every improvement to candidate generation makes the hand-tuned rule
worse, because a 3-parameter filter cannot discriminate within a pool it did not shrink.

**The gap is now the headline.** On unseen video the pipeline reaches **98.8%** of deer,
gives **88.0%** their own candidate, and counts **62.7%**. Twenty-five points sit in the
confirmation step alone, and nothing about detection, tracking or association can recover
them. That is the case for the learned temporal head, stated as a measurement.

**(c) Scope decision (2026-07-30).** The wildlife-ecology co-author confirmed that
preventing double counting *within a survey* is the requirement, that individual-ID claims
would not be credible at this resolution to any wildlife ecologist, and that **no additional
labelling should be performed**. The paper therefore reports **track association**, states
the resolution limit, and makes no individual-ID claim. The trained encoder and the rank-1
table above are retained as the evidence for that limit — a measured negative, not a
discarded experiment.

### 6.10 The learned head under the HELD-OUT protocol — it does not beat the rule ★★★

Job `20851097` (Delta). §6.5 showed learned confirmers losing under 8-fold CV over all 32
videos. That protocol mixes detector-train videos into every test fold, so the objection
was open: maybe the head only looked bad because the *evaluation* was optimistic for the
rule. This closes it. `train_cv.py --heldout` trains on the 19 detector-train videos,
freezes, and reports on the 13 the detector never saw — the same protocol that produced
the rule's MAE 2.38.

Run on two pools: **C** (7 008 candidates, ceiling 66/83) and **G** (27 679, ceiling 73/83).

| pool | method | MAE | RMSE | bias | counted (capped) |
|---|---|---|---|---|---|
| **C** | **hand-tuned rule** | **2.38** | 3.15 | −1.92 | **55/83 = 66.3%** |
| C | GBM | 2.85 | 3.89 | −2.54 | — |
| C | logistic regression | 3.00 | 3.89 | −2.54 | — |
| C | TTC transformer | 2.85 | 3.54 | −2.38 | 49/83 = 59.0% |
| G | hand-tuned rule | 2.77 | 3.28 | −2.31 | 50/83 = 60.2% |
| G | GBM | 2.92 | 3.98 | −2.62 | — |
| G | logistic regression | 3.15 | 5.36 | **+1.46** | — (unstable: 102 predicted) |
| G | TTC transformer | 2.77 | 3.26 | −2.15 | 51/83 = 61.4% |

**Verdict: the learned head does not beat the rule.** On pool C it loses outright (2.85 vs
2.38, and 6 fewer deer counted). On pool G it ties exactly on MAE, one deer ahead on capped
count — inside noise on 13 videos. Every learned variant *under-counts harder* than the
rule: it rejects more candidates rather than recovering the 26 confidence-rejected deer
that motivated the experiment.

The richer pool did not rescue it. G raises the ceiling from 66 to 73 reachable deer and the
head converts none of that headroom. Logistic regression on G collapses in the opposite
direction (+1.46 bias, 102 predicted for 83 deer), which is what "4× the noise" does to a
linear model.

**Consequence for the paper.** The intended novelty — a learned temporal head replacing the
hand-tuned rule — is not supported by the data. Three protocols now agree (§6.2 single
split, §6.5 8-fold CV, §6.10 held-out), so this is settled rather than a tuning problem.
The contribution is the **diagnosis**, not a method: see `docs/PAPER_PLAN.md`.

**Why it fails is itself the finding.** ~200 positives against 7k–27k negatives, and the
positives are not separable in the available features — §6.5 measured duplicates sitting
*between* primaries and false candidates in every feature. The rule's 3 parameters are
better matched to that data volume than anything with more capacity, exactly as §6.5
concluded from the stump experiment.

### 6.11 What the 36 missed deer actually are — GT is sound, but the detector fails at BOTH scale extremes ★★

Checked directly (`src/viz/missed_deer.py`, job `20851169`) whether the uncounted deer are
annotation errors. **They are not.** Every one of the 36 was rendered with its GT box at
three points in its track and inspected: all are real animals. The label set is sound.

| group | n | median box | median GT frames | best candidate conf |
|---|---|---|---|---|
| counted | 47 | **38.1 px** | 102 | 0.76 |
| rejected by the rule | 27 | 28.0 px | 47 | 0.52 |
| never detected | 9 | **19.9 px** | 47 | 0.00 |

Missed deer are smaller and shorter-lived — as expected. But three are the opposite:

| video | deer | box | state |
|---|---|---|---|
| GiantCityRd_TON | 5 | 104 px | rejected |
| GiantCityRd_TON | 6 | **149 px** | **never detected** |
| GiantCityRd_TON | 7 | 157 px | rejected |

A detector that finds 27 px deer does not miss a 149 px one — unless it has never seen one.
It has not:

| training-set box size | value |
|---|---|
| median | 26.8 px |
| p99 | 65.0 px |
| **max** | **95.6 px** |
| boxes ≥ 100 px | **0 (zero)** |

**The three missed deer are all larger than the largest box in training.** This is an
out-of-distribution failure, not a capability limit, and it is the mirror image of the
small-object problem the project has focused on throughout. §1.3 recorded "0% COCO-large"
as a *property of the data*; it is also a **hole in the training distribution**.

Consequences:

1. **Actionable and cheap.** Scale augmentation (or multi-scale training) covers 100–200 px
   at no data cost. Unlike the faint 12–20 px deer, nothing physical prevents this.
2. **Concentrated.** GiantCityRd counts 3 of 8 deer, and 3 of its 5 misses are these
   oversized animals — roughly 4 points of the headline 66.3% sitting in one video.
3. **Publishable in its own right.** "The detector fails at both ends of the scale, and the
   large end is a training-distribution artefact" is a concrete, verifiable finding that
   most small-object papers never check.

Rendered panels: `/work/hdd/bgte/tislam6/wildlife_outputs/viz/missed_deer/` (32 sheets,
named worst-first by box size).

### 6.5 Learned confirmation vs the rule — capacity is the whole story ★

Eight-fold CV over videos, orphan pool, identical protocol for every method (rule swept
per fold on that fold's training videos; learned threshold picked on an inner val split).

| Method | params | MAE | RMSE |
|---|---|---|---|
| **Hand-tuned rule** | 3 | **1.88** | 2.99 |
| **Capacity-matched stumps (depth 1, 40 iters)** | ~40 | **1.97** | **2.78** |
| GBM depth-3, 200 iters | ~10^3 | 2.41–2.66 | 3.60 |
| TTC transformer | ~60k | 2.81 | 4.86 |
| Logistic regression | 13 | 3.41 | 4.86 |
| Learned accept + geometric clustering | ~10^3 | 3.66 | 5.99 |
| Soft-count MLP (trained on |count−GT|) | ~5k | 3.89–4.44 | 6.34 |

**Finding: model capacity, not architecture, decides this.** The rule fits THREE
parameters directly on 32 videos; every learned competitor with 10^3+ parameters lost,
and shrinking capacity to ~40 stumps recovered nearly the whole gap (2.66 -> 1.97) and
gave the **best RMSE of any method** (2.78 vs the rule's 2.99, i.e. fewer catastrophic
per-video errors). With ~200 positive tracks this is a variance limit, not an algorithmic
one — an honest, quantified negative result about learned confirmation at this data scale,
and a direct argument that the dataset (not the method) is the binding constraint.

Trained directly on the counting objective (soft count = sum of probabilities, so
fragments can share mass) did NOT help — it was the worst variant, which rules out
"the rule wins because it optimises MAE directly" as the explanation.

### 6.6 Calibrated per-deer confidence — PROPOSAL DELIVERABLE MET ★

`docs/SERVER_HANDOFF.md` requires "a 0-1 score per detection, calibrated against verified
data, derived from learned appearance rather than a single cue", with low scores
"flagged for manual review rather than auto-counted".

Two different posteriors are useful, and conflating them was an error worth recording:

| Quantity | Meaning | Counted deer with conf >= 0.80 | Mean conf | AUC |
|---|---|---|---|---|
| P(primary) | is this THE canonical track of a deer | 28.2% | 0.641 | 0.926 |
| **P(on a deer)** — the proposal's "is-it-an-animal" score | is this track on a real animal | **92.1%** | **0.965** | 0.893 |

**92.1% of counted deer carry calibrated confidence >= 0.80 (mean 0.965)**, satisfying the
requirement. P(primary) necessarily scores lower because primaries are only 3% of the
pool — a low base rate caps the posterior no matter how good the model is. Use P(on a
deer) for the reported per-deer confidence and P(primary) for counting.

Raw detector confidence CANNOT satisfy this requirement: no track in the corpus exceeds
0.90 and the median counted track is 0.74, because the deer are ~27 px. The requirement is
only meaningful against a calibrated posterior — which is exactly what the proposal asked
for ("not a single cue").

Artefacts: `results/temporal/calibrated_ondeer/per_track_confidence.csv` (per-track score,
counted flag, review flag) and `per_video_counts.csv` (per-video count, expected count,
**Poisson-binomial sd = uncertainty on the count**, mean confidence). 793 tracks fall in
the review band and are flagged rather than auto-counted.

### 6.7 ⚠ CRITICAL: counting numbers over all 32 videos are OPTIMISTIC

The counting runs process **all 32 videos**, but the detector was trained on **19 of
them**. Splitting the best counting result by the DETECTOR's split:

| Detector split | Videos | GT | Counted | Coverage | MAE |
|---|---|---|---|---|---|
| train (detector SAW these) | 19 | 153 | 152 | **99.3%** | 1.53 |
| val (unseen) | 4 | 45 | 33 | 73.3% | 3.00 |
| test (unseen) | 9 | 38 | 25 | 65.8% | 2.11 |
| ALL (what §6.1/§6.3 report) | 32 | 236 | 210 | 89.0% | 1.88 |

**99.3% on seen videos vs 69.9% on unseen is memorisation, not generalisation.**

**PUBLISH THIS NUMBER:** on the 13 unseen videos — 58/83 deer = **69.9% coverage,
MAE 2.38**. Do NOT publish 89% / MAE 1.88 as a generalisation result; it is an upper
bound measured partly on training data.

> **SUPERSEDED for candidate coverage (2026-07-30).** The 69.9% above is the conf-0.10
> pool. With max-recall tracking + orphan recovery (§6.4.3) unseen *reach* is 95.2%
> (79/83) against 98.0% seen. The seen/unseen split still matters and every absolute
> figure in §6.1–§6.6 remains all-32-video, but "memorisation, not generalisation" is no
> longer the right reading of candidate generation — see §6.7.1.

#### 6.7.1 The honest held-out number ★★ PUBLISH THIS (2026-07-30)

The table above still leaks twice: the rule was swept on all 32 videos *and* reported on
them. `src/eval/count_eval_heldout.py` closes both — sweep the 3-parameter rule on the 19
detector-train videos only, freeze it, apply it to the 13 videos the detector never saw:

| pool | fit MAE (19 seen) | **held-out MAE (13 unseen)** | held-out counted | unseen `reached` | unseen `primary` |
|---|---|---|---|---|---|
| **Phase C orphan** (conf 0.10) | 1.53 | **2.38** | **55/83 = 66.3%** | 74/83 | 66/83 |
| Phase E max-recall (conf 0.02) | 1.84 | 2.54 | 54/83 = 65.1% | 79/83 | 69/83 |
| Phase F loose NMS | 3.05 | 3.46 | 41/83 = 49.4% | 79/83 | 70/83 |
| **Phase G ReID** (§6.9) | 2.00 | 2.69 | 52/83 = 62.7% | **82/83** | **73/83** |

Fitting on train videos alone selects the *same* rule the all-32 sweep did on the two
smaller pools (`min_hits>=20, span_s>=0.0, topk_conf>=0.65`), so the 2.38 is not an
artefact of the new protocol — it is the previous number, now honestly earned.

**This is the single most important table in the log.** On unseen video the best pool
*reaches* 79/83 deer (95.2%) and awards 69/83 (83.1%) their own candidate, but the rule
only *counts* 54–55/83 (~66%). The ~17-point gap between `primary` and counted is not a
detector failure and not a tracker failure — it is entirely the confirmation step, and it
is the space the paper's learned temporal head has to work in. Per site the gap
concentrates in dense groups: held-out SHB MAE 4.33 vs SHW 1.67 and MAS 1.00.

Note the pool ranking *inverts* between the two halves of the table. Phase E/F reach more
deer on unseen video than Phase C (79 vs 74) yet count fewer, because the rule cannot
survive the extra candidates. Whichever pool the paper adopts, that inversion is the
evidence that the confirmation stage — not candidate generation — is now binding.

Bias is **-1.92** on unseen video: the system under-counts by ~2 deer per video, never
over-counts. For a wildlife survey that is the safe direction, and worth saying so.

Consequences for everything above:
* §6.1, §6.3, §6.5, §6.6 MAE/coverage figures are all all-32-video numbers and inherit
  this bias. The *relative* comparisons (rule vs learned confirmers) remain valid because
  every method used the identical candidate pool, but the absolute values are optimistic.
* The confirmer CV folds are over videos, yet the candidate tracks they consume came from
  a detector that had seen 19 of those videos — so even the CV numbers are not fully
  clean. A fully honest pipeline evaluation needs the detector and the counter trained on
  the same train videos and both evaluated only on the held-out ones.
* Per-site counting breakdowns mix seen and unseen videos and should not be quoted
  without this caveat.

**Evidence images:** `viz/detect_yolov9m1280/` is drawn only from TEST videos and is
clean. `viz/counting_evidence/` covers all 32 videos and therefore includes
detector-training videos — label them as such in any figure.

---

### 6.13 The confirmation rule's confidence condition is a duplicate suppressor, not a recall filter ★★ (2026-08-28)

`src/eval/conf_sensitivity.py` (CPU only, seconds, no GPU) →
`results/counting/conf_sensitivity.csv` and `..._frozen.csv`.

The obvious objection to `topk_conf >= 0.65` is that it discards animals the detector
genuinely found. Two tables settle it, both under the §6.7.1 held-out protocol.

**FROZEN** — publish operating point `m>=20, s>=0` held, only `c` moves. 13 held-out
videos, 83 animals:

| c | predicted | counted | MAE | bias | over | under |
|---|---|---|---|---|---|---|
| 0.00 | 111 | **89.2%** | 3.54 | +2.15 | 37 | 9 |
| 0.25 | 108 | 85.5% | 3.77 | +1.92 | 37 | 12 |
| 0.35 | 99 | 80.7% | 3.69 | +1.23 | 32 | 16 |
| 0.45 | 81 | 72.3% | 3.38 | **-0.15** | 21 | 23 |
| 0.55 | 70 | 71.1% | 2.69 | -1.00 | 11 | 24 |
| **0.65** | **58** | 66.3% | **2.38** | -1.92 | 3 | 28 |
| 0.75 | 24 | 28.9% | 4.54 | -4.54 | 0 | 59 |
| 0.85 | 0 | 0.0% | 6.38 | -6.38 | 0 | 83 |

Deleting the condition **raises coverage 66.3% → 89.2%** — the largest coverage gain
anywhere in this log, larger than all four candidate-generation interventions of §6.8,
none of which raised coverage at all. But predicted goes 58 → 111 against 83 true animals:
+53 accepted tracks, of which 19 are previously-missed animals and 34 are not, i.e. ~2
duplicates per genuine animal recovered. MAE 2.38 → 3.54, bias flips to +2.15.
The threshold is doing duplicate suppression, not detection filtering — a fragment on an
already-counted deer *is* a true detection, so nothing in "is this a deer" separates it.

**Bias crosses zero at c≈0.45** (bias -0.15, MAE 3.38). If a downstream statistic needs an
unbiased estimator rather than a minimum-MAE one, that is the setting, at +1.0 MAE.

**REFIT** — `m` and `s` re-fitted on the 19 detector-train videos for each `c`, so the
condition is not blamed for a stale companion parameter. Held-out MAE at c=0 is 3.69 and
coverage 60.2%, *below* the frozen variant: the sweep compensates for a lowered `c` by
tightening span 0 → 1.0 s, which discards short tracks belonging to real animals. Across
the full 7×6×9 = 378-cell grid only c ∈ {0.55, 0.65} appear among the 12 best cells by fit
MAE.

**Side finding:** `min_span_s` is **inert** at the operating point — s ∈ {0, 0.1, 0.25} give
identical fit MAE (1.526), and those are the three best cells in the grid. The published
"three-parameter" rule is effectively two parameters at its operating point. Keep the third
anyway: it is what the sweep uses to compensate when `c` is lowered, which is what makes
the REFIT table informative.

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

# --- the two numbers the paper publishes (CPU only, seconds, no GPU) -> §6.7.1, §6.8 ---
python src/eval/count_eval_heldout.py \
    --counts $OUT/counts/phaseC_orphan_yolo11m_conf0.10 \
    --out results/counting_eval/heldout_phaseC.csv     # held-out MAE + coverage
python src/eval/pool_coverage.py \
    --counts-dir $OUT/counts/phaseE_maxrecall_conf0.02 # reached vs primary (the ceiling)
```

Outputs: `/work/hdd/bgte/tislam6/wildlife_outputs/{runs,logs}`, metrics under
`runs/detect/results/detection_eval/` and `results/`.

---

## Changelog

- **2026-08-28** — Added §6.13, the confidence-condition sensitivity analysis
  (`src/eval/conf_sensitivity.py`). Removing `topk_conf >= 0.65` raises held-out coverage
  by 23 points and flips the system from under- to over-counting; the condition is a
  duplicate suppressor, not a recall filter. Also found `min_span_s` is inert at the
  operating point. Manuscript ported to the MDPI *Journal of Imaging* template in
  `overleaf_MDPI/` (WACV draft kept in `overleaf_WACV/`), with the appendix folded into
  the main text and four "the detector never saw" captions corrected to "never trained
  on" — 4 of the 13 held-out videos are the detector's val split.

- **2026-07-30 (II)** — **All compute is finished; both queues are empty.** Three closures:
  (a) mmdet roster rescored on the any-overlap criterion — YOLO11m holds the detector
  decision across all 13 models, and RTMDet-m's mAP lead turns out to be a precision
  artefact (§3.2); (b) loose NMS confirms the collision mechanism but buys only +1 unseen
  deer for 5× the candidates, so Phase E remains the pool and Phase F is an ablation
  (§6.8); (c) `count_eval_heldout.py` fits the rule on detector-train videos only and
  reports on the 13 unseen — **MAE 2.38, 66.3% counted, bias -1.92** (§6.7.1). The
  reach → primary → counted chain (95.2% → 83.1% → 66.3%) is now the paper's thesis
  stated as a measurement. Also fixed `summarize_counting_eval.py`, which crashed on any
  sibling CSV lacking a `tag` column.

- **2026-07-29 (III)** — ⚠ Found that all counting numbers were computed over all 32
  videos while the detector trained on 19 of them: 99.3% coverage on seen vs 69.9% on
  unseen. Honest generalisation result is 58/83 deer = 69.9%, MAE 2.38 (§6.7). Detection
  gallery is clean (test videos only); counting evidence images are not.

- **2026-07-29 (later)** — Capacity, not architecture, explains why learned confirmers
  lost: depth-1 stumps reach MAE 1.97 vs the rule's 1.88 with BETTER RMSE (§6.5).
  Calibrated "is-it-an-animal" confidence delivers 92.1% of counted deer at >=0.80
  (mean 0.965), meeting the proposal requirement (§6.6).

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

### 6.12 Leave-one-site-out — attempted, not usable, ABANDONED (2026-08-06)

Job `20886104` (Delta, 4h27m). Trains on three sites, tests on the fourth, all four
rotations. Intended as the cross-domain generalisation result. **Not reported in the paper:
three of four folds failed and there was no time to re-run.**

| fold | epochs | final mAP50 | outcome |
|---|---|---|---|
| SHB | 3 | 0 | CUDA OOM, collapsed |
| TON | 0 | — | died before epoch 1 |
| **SHW** | **40** | **0.458** | **trained cleanly** |
| MAS | 32 | 0.075 | degraded |

**Root cause: GPU isolation failed.** The four folds were launched with
`CUDA_VISIBLE_DEVICES=$i`, but `src/detect/train.py` defaults `--device` to `"0"` and passes
it through to Ultralytics, and re-invokes itself as a subprocess (line 151) where the
environment variable does not survive. All four landed on physical GPU 0 — the OOM trace
shows three processes holding 18.7, 3.6 and 17.1 GB of one 39.5 GB card. **The fix is to
pass `--device $i` explicitly rather than relying on the environment variable.**

**A second bug, independent of the first.** The in-job summary scored each fold against all
32 videos and all 235 animals rather than the held-out site's, so every printed figure in
`wild_loso_20886104.out` is wrong — e.g. "reached 120/235 = 51.1%" for a fold that only
processed SHB's 8 videos. The counting itself was correct; only the scoring was scoped
wrongly. Anyone re-running this must restrict `count_eval` and `pool_coverage` to the
held-out site.

**What the one good fold suggests.** Scoring SHW correctly (8 videos, 38 animals):

| | LOSO, SHW held out | pooled split (paper) |
|---|---|---|
| MAE | 2.38 | 2.38 |
| counted | 30/38 = 78.9\% | 55/83 = 66.3\% |
| reached | 36/38 = 94.7\% | 74/83 = 89.2\% |

Identical MAE and *better* coverage than the pooled split. That is a single fold on the
easiest site — SHW has 38 animals and little grouping — so it is not evidence of anything,
but it is not the collapse a cross-site transfer failure would look like either. Worth
re-running if the paper ever needs a generalisation result; not worth blocking on.
