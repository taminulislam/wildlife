# Counting results — tracking-by-detection with a hand-tuned rule

**This is the baseline the learned temporal head must beat.**

Best hand-tuned rule (swept on this data, i.e. optimistically favourable to the baseline): `min_hits >= 1`, `span_s >= 0.3`, `topk_conf >= 0.65`

| Scope | Videos | MAE | RMSE | Bias (+over/-under) | Total over | Total under |
|---|---|---|---|---|---|---|
| **ALL** | 32 | **2.28** | 3.74 | -0.66 | 26 | 47 |
| MAS | 8 | 0.25 | 0.50 | -0.25 | 0 | 2 |
| SHB | 8 | 2.88 | 3.59 | -0.38 | 10 | 13 |
| SHW | 8 | 3.50 | 4.74 | +0.50 | 16 | 12 |
| TON | 8 | 2.50 | 4.50 | -2.50 | 0 | 20 |

Total GT deer: **236** | total predicted: **215** (91.1% of truth)

