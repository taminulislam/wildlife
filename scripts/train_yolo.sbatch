#!/bin/bash
#SBATCH --job-name=wild_yolo
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_yolo_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_yolo_%j.err

set -e
mkdir -p /work/hdd/bgte/tislam6/wildlife_outputs/logs

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife

cd /work/nvme/bgte/tislam6/wildlife_project

echo "=== job $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

# Override on submit, e.g.:
#   sbatch --export=ALL,MODEL=yolo11m,IMGSZ=1280,BATCH=16 scripts/train_yolo.sbatch
MODEL=${MODEL:-yolo11s}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-32}
NAME=${NAME:-${MODEL}_${IMGSZ}_SHWtest}

python -u src/detect/train.py \
    --model weights/${MODEL}.pt \
    --data data/dataset/yolo/data.yaml \
    --imgsz "$IMGSZ" --epochs 150 --batch "$BATCH" --patience 40 \
    --project /work/hdd/bgte/tislam6/wildlife_outputs/runs \
    --name "$NAME"

echo "=== done $SLURM_JOB_ID ==="
