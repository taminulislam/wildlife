#!/bin/bash
#SBATCH --job-name=asym_scale
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/asym_scale_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/asym_scale_%j.err

# Close the training-distribution hole at the LARGE end, with a controlled experiment.
#
# The training split's largest box is 95.6 px and it holds zero boxes at or above 100,
# yet three held-out deer appear at 104, 149 and 157 px -- one never detected at all.
# Ultralytics' symmetric `scale=m` samples [1-m, 1+m], so reaching 2.2x also means
# sampling 0.0x, which pushes a 27 px median animal to sub-pixel. Version 8.4 accepts an
# explicit (lo, hi) pair, so we can enlarge without ever shrinking: scale=(0.9, 2.2)
# presents the 95.6 px box at up to 210 px and never takes a 27 px one below 24 px.
#
# WHY THIS RUN IS DIFFERENT FROM THE TWO THAT FAILED.
# Job 20882753 used scale=0.9 and yolo11m_640_scaleaug used scale=0.6 -- but the latter
# ALSO moved batch 32->48 and mosaic 1.0->0.4, and it diverged (mAP 0.406 -> 0.138 ->
# 0.001) at exactly the epoch the 3-epoch LR warmup completes. Three variables moved at
# once and the collapse looks like an optimizer divergence, not an augmentation effect,
# so that run does not establish what the paper currently claims it does.
#
# Here EVERY hyperparameter matches yolo11m_640_v3pooled -- epochs 150, batch 32, SGD
# lr0 0.01, patience 50, mosaic 1.0, close_mosaic 15, erasing 0.4, seed 0 -- and only
# `scale` changes. Baseline to beat: AP50 0.460, and pool C MAE 2.38 / 55 of 83 counted.
set -eo pipefail
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project
OUT=/work/hdd/bgte/tislam6/wildlife_outputs
NAME=yolo11m_640_asymscale
echo "=== job $SLURM_JOB_ID  asymmetric scale (0.9, 2.2)  $(date) ==="

python -u src/detect/train.py \
    --model weights/yolo11m.pt --arch yolo \
    --data data/dataset/yolo_v3/data.yaml \
    --imgsz 640 --epochs 150 --batch 32 --patience 50 --seed 0 \
    --optimizer SGD --lr0 0.01 --warmup-epochs 3.0 \
    --scale 0.9,2.2 --mosaic 1.0 --close-mosaic 15 \
    --project "$OUT/runs" --name "$NAME"

W="$OUT/runs/$NAME/weights/best.pt"
echo "=== detection eval (same harness as Table 1) ==="
python -u src/detect/eval_full.py --weights "$W" --imgsz 640 \
    --tag "${NAME}" --out results/detection_eval || echo "[warn] eval_full failed"

echo "=== per-animal recall: did the 104/149/157 px deer come back? ==="
python -u src/eval/track_recall.py --weights "$W" --conf 0.25 \
    --out "results/track_recall/roster_conf0.25/${NAME}" || echo "[warn] track_recall failed"

echo "=== end-to-end counting, frozen held-out protocol ==="
CO="$OUT/counts/asymscale_conf0.10"
python -u src/track/count_deer.py \
    --weights "$W" --source data/raw --cvat-dir data/cvat_export \
    --device 0 --imgsz 640 --conf 0.10 --iou 0.50 --contrast clahe \
    --dump-min-frames 1 --keep-orphans \
    --tracker src/track/botsort_deer_recall.yaml --out "$CO/shard0"
cp "$CO/shard0/counts.csv" "$CO/counts_shard0.csv" 2>/dev/null || true
python -u src/eval/pool_coverage.py --counts-dir "$CO" \
    --out results/counting_eval/pool_coverage_asymscale.csv || true
python -u src/eval/count_eval_heldout.py --counts "$CO" \
    --out results/counting_eval/heldout_asymscale.csv || true

echo "=== done $(date) ==="
