#!/bin/bash
#SBATCH --job-name=wild_e1280
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_e1280_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_e1280_%j.err
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project
R=/work/hdd/bgte/tislam6/wildlife_outputs/runs
for m in yolov10m yolov9m; do
  b=$R/${m}_1280_v3pooled/weights/best.pt
  [ -f "$b" ] || continue
  echo "=== $m @1280 on TEST (full GT) ==="
  python -u src/detect/eval_full.py --weights "$b" --data data/dataset/yolo_v3/data.yaml \
      --split test --imgsz 1280 --arch yolo --tag ${m}_1280_v3pooled || true
  echo "=== $m @1280 on TEST (human-keyframe GT) ==="
  python -u src/detect/eval_full.py --weights "$b" --data data/dataset/yolo_v3_kf/data.yaml \
      --split test --imgsz 1280 --arch yolo --tag ${m}_1280_KEYFRAME || true
done
