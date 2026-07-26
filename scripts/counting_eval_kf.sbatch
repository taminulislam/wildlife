#!/bin/bash
#SBATCH --job-name=wild_cntkf
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cntkf_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cntkf_%j.err
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project
R=/work/hdd/bgte/tislam6/wildlife_outputs/runs
# missing cell of the 2x2 grid: permissive matching x HUMAN-KEYFRAME GT
for m in yolov9m yolov10m yolo11m; do
  b=$R/${m}_640_v3pooled/weights/best.pt
  [ -f "$b" ] || continue
  for cf in 0.25 0.10; do
    python -u src/eval/counting_detection_eval.py --weights "$b" \
      --data data/dataset/yolo_v3_kf/data.yaml --split test \
      --conf $cf --tag ${m}_640_KEYFRAME || echo "[warn] $m $cf failed"
  done
done
LOG=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cntkf_${SLURM_JOB_ID}.out
printf '%b\n' "Counting-criterion eval on HUMAN-KEYFRAME GT finished $(date).\n\nThis is the last cell of the 2x2 grid: permissive matching (any overlap) scored against boxes a human actually placed - the truest measure of 'can we find the deer for counting'.\n\n$(grep -E '^=== \[|^iou50|^touch|^center' "$LOG" 2>/dev/null)" \
 | mail -s "[DeltaAI] Counting criterion x keyframe GT (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
