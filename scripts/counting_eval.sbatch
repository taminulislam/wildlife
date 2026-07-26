#!/bin/bash
#SBATCH --job-name=wild_cnte
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cnte_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cnte_%j.err

# Detection scored the COUNTING way: any overlap with the deer = found.
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project
R=/work/hdd/bgte/tislam6/wildlife_outputs/runs

for m in yolov9m yolov10m yolo11m; do
  b=$R/${m}_640_v3pooled/weights/best.pt
  [ -f "$b" ] || continue
  for cf in 0.25 0.10; do
    python -u src/eval/counting_detection_eval.py --weights "$b" \
        --data data/dataset/yolo_v3/data.yaml --split test \
        --conf $cf --tag ${m}_640 || echo "[warn] $m conf$cf failed"
  done
done

LOG=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cnte_${SLURM_JOB_ID}.out
BODY=$(grep -E "^=== \[|^iou50|^iou30|^touch|^center|^criterion" "$LOG" 2>/dev/null)
printf '%b\n' "Counting-criterion detection eval finished $(date).\n\nScored so that ANY overlap with the deer counts as found (box tightness irrelevant for counting), alongside the strict IoU>=0.5 standard.\n\nReporting note: publish iou50 in the detection table; the permissive rows belong to the counting evaluation and should be called presence/counting recall, never mAP.\n\n$BODY" \
  | mail -s "[Delta] Counting-criterion detection results (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
