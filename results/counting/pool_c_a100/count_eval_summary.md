# Counting results — tracking-by-detection with a hand-tuned rule

**This is the baseline the learned temporal head must beat.**

Best hand-tuned rule (swept on this data, i.e. optimistically favourable to the baseline): `min_hits >= 20`, `span_s >= 0.0`, `topk_conf >= 0.65`

| Scope | Videos | MAE | RMSE | Bias (+over/-under) | Total over | Total under |
|---|---|---|---|---|---|---|
| **ALL** | 32 | **2.00** | 2.97 | -0.75 | 20 | 44 |
| MAS | 8 | 0.62 | 1.06 | -0.38 | 1 | 4 |
| SHB | 8 | 2.12 | 2.98 | -0.38 | 7 | 10 |
| SHW | 8 | 2.75 | 3.71 | +0.00 | 11 | 11 |
| TON | 8 | 2.50 | 3.39 | -2.25 | 1 | 19 |

Total GT deer: **236** | total predicted: **212** (89.8% of truth)

