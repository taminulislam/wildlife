#!/bin/bash
#SBATCH --job-name=wild_rtdetr
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_rtdetr_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_rtdetr_%j.err

# RT-DETR-l on the SAME yolo_v2 split — Axis-1 detector comparison vs YOLO11m.
# Transformers are heavier: smaller batch. Same live-ETA + resume + full-eval harness.
set -e
mkdir -p /work/hdd/bgte/tislam6/wildlife_outputs/logs

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife

cd /work/nvme/bgte/tislam6/wildlife_project

echo "=== job $SLURM_JOB_ID on $SLURM_NODELIST  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

MODEL=${MODEL:-rtdetr-l}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-150}
DATA=${DATA:-data/dataset/yolo_v2/data.yaml}
NAME=${NAME:-${MODEL}_${IMGSZ}_v2_SHWtest}
RESUME=${RESUME:-1}

RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"

echo "=== train $MODEL @ $IMGSZ  batch=$BATCH  epochs=$EPOCHS  data=$DATA  resume=$RESUME ==="
python -u src/detect/train.py \
    --model weights/${MODEL}.pt --arch rtdetr \
    --data "$DATA" \
    --imgsz "$IMGSZ" --epochs "$EPOCHS" --batch "$BATCH" --patience 40 \
    --save-period 10 $RESUME_FLAG \
    --project /work/hdd/bgte/tislam6/wildlife_outputs/runs \
    --name "$NAME"

echo "=== done $SLURM_JOB_ID  $(date) ==="
