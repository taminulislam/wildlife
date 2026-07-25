#!/bin/bash
#SBATCH --job-name=wild_eval
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_eval_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_eval_%j.err

# Re-run the FULL evaluation (native + COCO size-stratified + domain metrics) for
# every finished roster model. Cheap (~5 min/model); safe to re-run any time.
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife

cd /work/nvme/bgte/tislam6/wildlife_project

DATA=${DATA:-data/dataset/yolo_v3/data.yaml}
RUNS=/work/hdd/bgte/tislam6/wildlife_outputs/runs
TAG=${TAG:-v3pooled}

for spec in "yolov8m|yolo" "yolov9m|yolo" "yolov10m|yolo" "yolo11m|yolo" \
            "yolo12m|yolo" "rtdetr-l|rtdetr"; do
  IFS='|' read -r m arch <<< "$spec"
  best=$RUNS/${m}_640_${TAG}/weights/best.pt
  [ -f "$best" ] || { echo "[skip] $m: no $best"; continue; }
  echo "=== eval $m ==="
  python -u src/detect/eval_full.py \
      --weights "$best" --data "$DATA" --split test \
      --imgsz 640 --arch "$arch" --tag ${m}_640_${TAG} || echo "[warn] $m eval failed"
done
echo "=== all evals done $(date) ==="
