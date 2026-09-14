#!/bin/bash
#SBATCH --job-name=a100_cnt
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=02:30:00
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/a100_cnt_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/a100_cnt_%j.err
# One shard (8 of 32 videos) of the pool-C rerun on A100; same script, weights and
# arguments as scripts/dtai_count_orphan.sbatch (the paper's GH200 run). SHARD via --export.
set -e
cd /work/nvme/bgte/tislam6/wildlife_project
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python
W=/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt
OUT=/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10_A100
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> "$OUT/gpu_shard$SHARD.txt"
$PY -u src/track/count_deer.py \
    --weights "$W" --source data/raw --cvat-dir data/cvat_export \
    --shards 4 --shard "$SHARD" --device 0 \
    --imgsz 640 --conf 0.10 --contrast clahe --dump-min-frames 1 --keep-orphans \
    --tracker src/track/botsort_deer_recall.yaml --out "$OUT/shard$SHARD"
cp "$OUT/shard$SHARD/counts.csv" "$OUT/counts_shard$SHARD.csv"
