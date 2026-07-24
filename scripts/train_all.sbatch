#!/bin/bash
#SBATCH --job-name=wild_all
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_all_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_all_%j.err

# ONE job, ONE queue wait: trains the whole CVPR detector roster in parallel across
# every GPU on the node (dynamic scheduler below keeps all GPUs busy). Beats 6 separate
# jobs that each queue independently. Each model still writes its own run dir + full
# eval, and is resume-safe (--resume), so re-submitting continues where it left off.
set -e
mkdir -p /work/hdd/bgte/tislam6/wildlife_outputs/logs

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife

cd /work/nvme/bgte/tislam6/wildlife_project

echo "=== job $SLURM_JOB_ID on $SLURM_NODELIST  $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

DATA=${DATA:-data/dataset/yolo_v3/data.yaml}
IMGSZ=${IMGSZ:-640}
EPOCHS=${EPOCHS:-150}
TAG=${TAG:-v3pooled}
RUNS=/work/hdd/bgte/tislam6/wildlife_outputs/runs
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs

# model | arch | batch | optimizer | lr0   (YOLO=SGD@0.01, RT-DETR=AdamW@1e-4)
SPECS=(
  "yolov8m|yolo|32|SGD|0.01"
  "yolov9m|yolo|32|SGD|0.01"
  "yolov10m|yolo|32|SGD|0.01"
  "yolo11m|yolo|32|SGD|0.01"
  "yolo12m|yolo|32|SGD|0.01"
  "rtdetr-l|rtdetr|16|AdamW|0.0001"
)

NGPU=${SLURM_GPUS_ON_NODE:-4}
echo "=== ${#SPECS[@]} models across $NGPU GPUs  data=$DATA  imgsz=$IMGSZ  epochs=$EPOCHS ==="

run_one() {
  local gpu=$1 spec=$2
  IFS='|' read -r m arch batch opt lr <<< "$spec"
  local name=${m}_${IMGSZ}_${TAG}
  local log=$LOGS/wild_all_${m}_${SLURM_JOB_ID}.out
  echo "[gpu $gpu] START $m  batch=$batch $opt@$lr  -> $name  (log: $log)"
  CUDA_VISIBLE_DEVICES=$gpu python -u src/detect/train.py \
      --model weights/${m}.pt --arch "$arch" --data "$DATA" \
      --imgsz "$IMGSZ" --epochs "$EPOCHS" --batch "$batch" --patience 50 \
      --optimizer "$opt" --lr0 "$lr" --device 0 \
      --save-period 10 --resume \
      --project "$RUNS" --name "$name" > "$log" 2>&1
  echo "[gpu $gpu] DONE  $m  (exit $?)"
}

# --- dynamic scheduler: keep every GPU busy; hand a freed GPU to the next model ---
free=($(seq 0 $((NGPU-1))))
declare -A pid2gpu
idx=0; n=${#SPECS[@]}
while (( idx < n )) || (( ${#pid2gpu[@]} > 0 )); do
  while (( ${#free[@]} > 0 )) && (( idx < n )); do
    g=${free[0]}; free=("${free[@]:1}")
    run_one "$g" "${SPECS[$idx]}" &
    pid2gpu[$!]=$g
    idx=$((idx+1))
  done
  wait -n
  for p in "${!pid2gpu[@]}"; do
    if ! kill -0 "$p" 2>/dev/null; then
      free+=("${pid2gpu[$p]}"); unset "pid2gpu[$p]"
    fi
  done
done

echo "=== all models done  $SLURM_JOB_ID  $(date) ==="
echo "--- results dirs ---"
for spec in "${SPECS[@]}"; do
  m=${spec%%|*}; ls -d "$RUNS/${m}_${IMGSZ}_${TAG}" 2>/dev/null
done
