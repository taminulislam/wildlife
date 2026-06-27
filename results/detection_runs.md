# Detection runs (held-out site = SHW)

Dataset: `data/dataset/yolo/` — site-split, single class `deer`.
train 1352 img / 663 box · val 156 img / 82 box (MAS+SHB+TON) · test 985 img / 277 box (SHW).
Runs live on `/work/hdd/bgte/tislam6/wildlife_outputs/runs/`.

| run | model | imgsz | **test mAP50** | test mAP50-95 | test P | test R |
|---|---|---|---|---|---|---|
| yolo11s_SHWtest      | yolo11s | 640  | 0.375 | 0.116 | 0.428 | 0.389 |
| yolo11s_1280_SHWtest | yolo11s | 1280 | 0.496 | 0.171 | 0.554 | 0.451 |
| **yolo11m_640_SHWtest** ⭐ | **yolo11m** | **640** | **0.611** | **0.220** | **0.699** | 0.560 |
| yolo11m_1280_SHWtest | yolo11m | 1280 | 0.566 | 0.197 | 0.590 | 0.581 |

**Winner: yolo11m @ 640** (best mAP50 / mAP50-95, best precision, ~4x faster than 1280).
Weights: `/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_SHWtest/weights/best.pt`.

Findings (model x resolution sweep)
- **Capacity > resolution**: m@640 (0.611) beats s@1280 (0.496). The s model was underfitting.
- **1280 hurt the m model** (0.611->0.566): recall up (0.560->0.581) but precision down
  (0.699->0.590), more false positives. Higher res only helped the under-capacity s model.
- Implication: don't chase resolution; next detector gains should come from more/diverse
  labels (active learning) and copy-paste aug, not bigger inputs. SAHI tiling still worth a
  test for the smallest/most-distant deer (recall 0.56 leaves room).
- Inference ~1-1.5 ms/frame on A100 (full 931k-frame archive is cheap to process).
