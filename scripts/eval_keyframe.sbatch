#!/bin/bash
#SBATCH --job-name=wild_kfev
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=48G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_kfev_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_kfev_%j.err

# Is the low mAP the DETECTOR or the GT? 94% of GT boxes are CVAT-interpolated and the
# deer moves ~2.3 box-widths between keyframes, so mid-gap boxes drift off the animal.
# Score the same models on HUMAN KEYFRAMES ONLY (361 boxes a person actually placed).
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project

R=/work/hdd/bgte/tislam6/wildlife_outputs/runs
D=data/dataset/yolo_v3_kf/data.yaml
OUT=""
for m in yolov9m yolov10m yolo11m; do
  b=$R/${m}_640_v3pooled/weights/best.pt
  [ -f "$b" ] || continue
  echo "=== keyframe-only eval: $m ==="
  python -u src/detect/eval_full.py --weights "$b" --data "$D" --split test \
      --imgsz 640 --arch yolo --tag ${m}_640_KEYFRAME || echo "[warn] $m failed"
done

LOG=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_kfev_${SLURM_JOB_ID}.out
BODY=$(grep -E "^=== keyframe-only eval|^\s+all " "$LOG" 2>/dev/null)
printf '%b\n' "Keyframe-only evaluation finished $(date).\n\nWhy: 94% of the GT boxes are CVAT-interpolated; between human keyframes the deer travels a median 2.3 box-widths, so interpolated boxes drift off the animal. This run scores the models ONLY on the 361 boxes a human actually placed (182 images).\n\nCompare to full-test mAP50: yolov9m 0.498, yolov10m 0.498, yolo11m 0.466.\nIf these keyframe numbers are much higher, the detector was being punished for bad GT, not for missing deer.\n\n$BODY" \
  | mail -s "[Delta] Keyframe-only eval: is the low mAP the detector or the GT? (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
