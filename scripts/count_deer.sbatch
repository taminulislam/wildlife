#!/bin/bash
#SBATCH --job-name=wild_count
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_count_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_count_%j.err

set -e
mkdir -p /work/hdd/bgte/tislam6/wildlife_outputs/logs

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project

echo "=== count job $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

# Override on submit:
#   sbatch --export=ALL,WEIGHTS=...,SOURCE=...,OUT=...,IMGSZ=1280 scripts/count_deer.sbatch
WEIGHTS=${WEIGHTS:-/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_SHWtest/weights/best.pt}
SOURCE=${SOURCE:-data/raw}
OUT=${OUT:-/work/hdd/bgte/tislam6/wildlife_outputs/counts/run}
IMGSZ=${IMGSZ:-640}
EXTRA=${EXTRA:-}

python -u src/track/count_deer.py \
    --weights "$WEIGHTS" --source "$SOURCE" --out "$OUT" \
    --imgsz "$IMGSZ" $EXTRA

echo "=== done $SLURM_JOB_ID ==="
