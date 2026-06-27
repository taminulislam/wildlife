# Detection runs (held-out site = SHW)

Dataset: `data/dataset/yolo/` — site-split, single class `deer`.
train 1352 img / 663 box · val 156 img / 82 box (MAS+SHB+TON) · test 985 img / 277 box (SHW).
Runs live on `/work/hdd/bgte/tislam6/wildlife_outputs/runs/`.

| run | model | imgsz | epochs | val mAP50 | val mAP50-95 | **test mAP50** | test mAP50-95 | test P | test R |
|---|---|---|---|---|---|---|---|---|---|
| yolo11s_SHWtest | yolo11s | 640 | 69 (conv. ~30) | 0.449 | 0.163 | **0.375** | 0.116 | 0.428 | 0.389 |

Notes
- v1 baseline. Converged by ~epoch 30 (val mAP50 plateaus ~0.44) → more epochs won't help;
  limiters are model capacity and input resolution (deer are tiny thermal blobs, frames 640x512).
- Likely biggest lever next: train/infer at higher imgsz (e.g. 1280) and/or yolo11m.
- Inference ~1.2 ms/frame on A100 (full-video processing is cheap).
