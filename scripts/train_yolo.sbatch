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
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_yolo_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_yolo_%j.err

set -e
mkdir -p /work/hdd/bgte/tislam6/wildlife_outputs/logs

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife

cd /work/nvme/bgte/tislam6/wildlife_project

echo "=== job $SLURM_JOB_ID on $SLURM_NODELIST  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

# Override on submit, e.g.:
#   sbatch --export=ALL,MODEL=yolo11m,IMGSZ=640,BATCH=32 scripts/train_yolo.sbatch
MODEL=${MODEL:-yolo11m}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-32}
EPOCHS=${EPOCHS:-150}
DATA=${DATA:-data/dataset/yolo_v2/data.yaml}
NAME=${NAME:-${MODEL}_${IMGSZ}_v2_SHWtest}
RESUME=${RESUME:-1}   # 1 = auto-continue from last.pt on re-submit; 0 = force fresh
PATIENCE=${PATIENCE:-50}

RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"

echo "=== train $MODEL @ $IMGSZ  batch=$BATCH  epochs=$EPOCHS  data=$DATA  resume=$RESUME ==="
# Explicit SGD (the recipe behind the old 0.611). Ultralytics 8.4 optimizer=auto
# picks MuSGD@0.01, which overshoots the pretrained basin and collapses mAP to 0.
OPT=${OPT:-SGD}
LR0=${LR0:-0.01}
# strategy-B augmentation/warmup knobs (defaults = strategy-A baseline)
MOSAIC=${MOSAIC:-1.0}
COPYPASTE=${COPYPASTE:-0.0}
MIXUP=${MIXUP:-0.0}
SCALE=${SCALE:-0.5}
WARMUPBIAS=${WARMUPBIAS:-0.1}
WARMUPEP=${WARMUPEP:-3.0}
python -u src/detect/train.py \
    --model weights/${MODEL}.pt --arch yolo \
    --data "$DATA" \
    --imgsz "$IMGSZ" --epochs "$EPOCHS" --batch "$BATCH" --patience "$PATIENCE" \
    --optimizer "$OPT" --lr0 "$LR0" \
    --mosaic "$MOSAIC" --copy-paste "$COPYPASTE" --mixup "$MIXUP" --scale "$SCALE" \
    --warmup-bias-lr "$WARMUPBIAS" --warmup-epochs "$WARMUPEP" \
    --save-period 10 $RESUME_FLAG \
    --project /work/hdd/bgte/tislam6/wildlife_outputs/runs \
    --name "$NAME"

# --- Fair baseline: score the OLD (loose-label) detector on the SAME clean test ---
# Isolates "better labels" from "different test set". Skips gracefully if absent.
OLD_BEST=/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_SHWtest/weights/best.pt
if [ -f "$OLD_BEST" ]; then
    echo "=== fair-baseline eval: OLD best.pt on new clean test ==="
    python -u src/detect/eval_full.py \
        --weights "$OLD_BEST" --data "$DATA" --split test \
        --imgsz "$IMGSZ" --arch yolo --tag yolo11m_640_v1_oldlabels || \
        echo "[warn] baseline eval failed; continuing"
fi

echo "=== done $SLURM_JOB_ID  $(date) ==="
