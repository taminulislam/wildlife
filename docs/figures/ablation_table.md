# Pipeline ablation (held out — 13 videos the detector never saw, 83 deer)

Every row changes ONE stage; all other stages are held fixed. `reached` and
`primary` measure the candidate pool, `counted` and MAE measure the final count.

## Candidate generation (detector + tracker), confirmation rule fixed

| Pool | change | candidates | reached | primary | counted | MAE |
|---|---|---|---|---|---|---|
| C | orphan recovery, conf 0.10 (operating point) | 7,008 | 74/83 | 66/83 | **55/83 = 66.3%** | 2.38 |
| E | + track-init gate removed, conf 0.02 | 18,349 | 79/83 | 69/83 | **54/83 = 65.1%** | 2.54 |
| F | + NMS IoU 0.50 -> 0.90 | 90,239 | 79/83 | 70/83 | **41/83 = 49.4%** | 3.46 |
| G | + appearance ReID | 27,679 | 82/83 | 73/83 | **52/83 = 62.7%** | 2.69 |

Better candidate generation, worse counting: pool G reaches 82 of 83 deer and
counts fewer than pool C, which reaches 74. The confirmation stage cannot
exploit a pool it did not shrink.

## Confirmation stage, candidate pool fixed

| Pool | confirmer | MAE | counted |
|---|---|---|---|
| C | hand-tuned rule (3 params) | **2.38** | **55/83** |
| C | gradient boosting | 2.85 | — |
| C | logistic regression | 3.00 | — |
| C | temporal transformer | 2.85 | 49/83 |
| G | hand-tuned rule | 2.77 | 50/83 |
| G | gradient boosting | 2.92 | — |
| G | logistic regression | 3.15 | — (unstable, 102 predicted) |
| G | temporal transformer | 2.77 | 51/83 |

## Other ablations, reported in full in `RESULTS_LOG.md`

| Ablation | Section | Result |
|---|---|---|
| Input resolution 640 vs 1280 | §4.5 | best detector is the worst counter |
| CLAHE contrast normalisation | §2 | 0.519 vs 0.299 test mAP50 |
| 13 detector architectures | §3, §3.2 | YOLO11m holds under any-overlap |
| GT: human keyframes vs interpolated | §4.2 | ~1/3 of error is label noise |
| Matching criterion (4) x GT set (2) x conf (2) | §4.3 | ranking is criterion-dependent |
| Detector confidence 0.10 / 0.05 / 0.02 | §6.4.2 | recovers zero extra deer |
| Track-initialisation threshold | §6.4.3 | the real gate: +5 primaries |
| Orphan recovery on/off | §6.3 | +17 deer |
| ReID cue decomposition (7 variants) | §6.9 | box size dominates appearance |
| Track splitting threshold sweep | §6.8 | +3 deer for 4,000 candidates |
