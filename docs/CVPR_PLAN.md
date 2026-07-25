# CVPR plan — Temporal Counting of Unique Individuals in Moving-Camera Thermal Video

**Target:** CVPR main track, with PBVS / CV4Animals / EarthVision (CVPR workshops) as the
calibrated backup. **Chosen contribution:** temporal counting.

## 1. Problem & why it's novel

Count *unique* white-tailed deer from moving-camera FLIR thermal road-transect video.
Naive `detect → track → count` fails three ways (all observed in our data):

- **False-positive inflation** — warm rocks/structures flicker in as deer.
- **Under-counting** — multiple deer merge into one blob / one track.
- **Over-counting** — one deer fragments into several track IDs under ego-motion.

Current systems (ours included) paper over this with **hand-tuned confirmation thresholds**
(`min-hits`, `min-span-s`, `conf-track`) — brittle, site-specific, and the reason our first
counts were only ~30% reliable. **We replace the hand-tuned rule with a learned temporal
model.** Thermal (no color/texture) + moving camera + small blobs makes this a genuinely
hard, underserved regime; per-frame detection alone is not a contribution — temporal
counting is.

## 2. Method (working name: TTC — Temporal Track Counting)

Stage 1 — **Detector** (per frame): tiny-object thermal detector (YOLO11m / RT-DETR),
optionally SAHI tiling for distant blobs. Over-generate candidates (favor recall).

Stage 2 — **Tracker**: BoT-SORT w/ camera-motion compensation → *candidate* tracks
(deliberately over-segmented; fragmentation is fixed downstream).

Stage 3 — **Temporal counting head (the contribution).** For each candidate track, a small
temporal transformer over its sequence of `(box, conf, thermal-appearance embedding, motion
feature)` predicts:
1. **Confirmation** `P(real deer)` — learned, replacing hand thresholds → kills FP inflation.
2. **Re-ID embedding** for cross-fragment linking under ego-motion → repairs fragmentation.
3. **Multiplicity** `k = #animals in this blob` from temporal shape/intensity dynamics →
   fixes merged-deer under-counting.

**Count** = Σ over confirmed, de-fragmented tracks of their multiplicity `k`.
Supervision comes directly from our track GT (unique IDs) + box GT.

Ablation-friendly: each of the three heads can be toggled to isolate its gain.

## 3. Baselines (must beat / compare)

| family | methods |
|---|---|
| Detection | YOLO11m (ours), RT-DETR, Deformable-DETR, Faster R-CNN |
| Counting — tracking | ByteTrack / BoT-SORT unique-ID count; **tracking-by-detection + hand-tuned rules (the thing to beat)** |
| Counting — density | CSRNet, MCNN (per-frame density, temporally aggregated) |
| Counting — points | P2PNet |

## 4. Metrics & protocol

- **Detection:** mAP50, mAP50-95, AR_small.
- **Counting:** **MAE, RMSE** vs the 236-deer GT (per video + per site); **GAME(L)** for
  spatial correctness; over/under-count decomposition.
- **Generalization:** leave-one-site-out (SHW held out today) — report the cross-site gap.
- Seeds/variance reported. Per-site breakdown (MAS/SHB/SHW/TON).

## 5. Ablations

- learned confirmation vs hand-tuned rule (headline)
- temporal window length
- re-ID head on/off (fragmentation)
- multiplicity head on/off (merged blobs)
- detector choice; SAHI tiling on/off; GMC (camera-motion comp) on/off

## 6. Dataset contribution (release)

Moving-camera thermal wildlife counting benchmark: videos, unique-individual **track GT**,
site/season splits, annotation protocol, code, weights. **Scale is the main risk** — 236 deer
/ **32 videos** (8 per site x MAS/SHB/SHW/TON; per-site deer: SHB 132, TON 51, SHW 38, MAS 15)
is thin. These 32 CVAT-labelled videos are the *entire* dataset — earlier drafts said 42,
which was wrong. Mitigate: fold in the RS footage, label additional videos, add season/site
diversity. Target a defensible size before submission.

## 7. Execution phases (layered on the current pipeline)

- **A. Foundation** *(in progress)* — clean detector on new CVAT GT; the detection baseline.
- **B. Counting harness** — implement MAE/RMSE/GAME + tracking/density/point counting baselines.
- **C. Temporal head** — build & train TTC; beat the hand-tuned rule.
- **D. Rigor** — ablations + leave-one-site-out generalization + variance.
- **E. Package** — dataset release, code, weights, writing, figures.

Nothing in Phases 1-4 of the base pipeline is wasted — clean GT + counts are the foundation.
