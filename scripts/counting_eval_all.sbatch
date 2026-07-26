#!/bin/bash
#SBATCH --job-name=wild_cntall
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cntall_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cntall_%j.err

# THE detection table for the paper: all 6 roster models x 4 matching criteria
# x {full interpolated GT, human-keyframe GT} x {conf 0.25, 0.10}.
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project
R=/work/hdd/bgte/tislam6/wildlife_outputs/runs

# model | arch
SPECS=("yolov8m|yolo" "yolov9m|yolo" "yolov10m|yolo" "yolo11m|yolo" \
       "yolo12m|yolo" "rtdetr-l|rtdetr")

for spec in "${SPECS[@]}"; do
  IFS='|' read -r m arch <<< "$spec"
  b=$R/${m}_640_v3pooled/weights/best.pt
  [ -f "$b" ] || { echo "[skip] $m (no weights)"; continue; }
  for cf in 0.25 0.10; do
    python -u src/eval/counting_detection_eval.py --weights "$b" --arch "$arch" \
        --data data/dataset/yolo_v3/data.yaml --split test --conf $cf \
        --tag ${m}_640_FULLGT   || echo "[warn] $m full $cf"
    python -u src/eval/counting_detection_eval.py --weights "$b" --arch "$arch" \
        --data data/dataset/yolo_v3_kf/data.yaml --split test --conf $cf \
        --tag ${m}_640_KEYFRAME || echo "[warn] $m kf $cf"
  done
done

python -u src/eval/summarize_counting_eval.py

TBL=results/counting_eval/counting_eval_TABLE.md
printf '%b\n' "Full detection table (all 6 models, 4 matching criteria, both GT sets, 2 confidences) finished $(date).\n\nPaper-ready table: $TBL\nTidy CSV: results/counting_eval/counting_eval_ALL.csv\n\n$(cat $TBL 2>/dev/null | head -60)" \
  | mail -s "[DeltaAI] FULL detection table — all models (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
