#!/bin/bash
#SBATCH --job-name=gh_ev1280
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_ev1280_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_ev1280_%j.err

# Same evaluation battery the 640 roster got, so 1280 vs 640 is apples-to-apples:
#   1. eval_full      -> native P/R/mAP50/mAP50-95 + COCO size-stratified + domain
#   2. counting eval  -> 4 matching criteria x {full GT, keyframe GT} x {conf .25,.10}
#   3. track recall   -> the 235-deer gate, to see if 1280 recovers deer 640 misses
# CRITICAL: --imgsz 1280 everywhere. Scoring a 1280-trained model at 640 understates it.
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project
R=/work/hdd/bgte/tislam6/wildlife_outputs/runs
SZ=1280

for m in yolov10m yolov9m; do
  b=$R/${m}_1280_v3pooled/weights/best.pt
  [ -f "$b" ] || { echo "[skip] $m — no best.pt"; continue; }
  echo "=== eval_full: $m @${SZ} ==="
  python -u src/detect/eval_full.py --weights "$b" --data data/dataset/yolo_v3/data.yaml \
      --split test --imgsz $SZ --arch yolo --tag ${m}_${SZ}_v3pooled || echo "[warn] eval_full $m"
  for cf in 0.25 0.10; do
    python -u src/eval/counting_detection_eval.py --weights "$b" --imgsz $SZ \
        --data data/dataset/yolo_v3/data.yaml --split test --conf $cf \
        --tag ${m}_${SZ}_FULLGT   || echo "[warn] counting full $m $cf"
    python -u src/eval/counting_detection_eval.py --weights "$b" --imgsz $SZ \
        --data data/dataset/yolo_v3_kf/data.yaml --split test --conf $cf \
        --tag ${m}_${SZ}_KEYFRAME || echo "[warn] counting kf $m $cf"
  done
done

# track-level recall gate at 1280 (best model only — this is the expensive one)
BEST=$R/yolov9m_1280_v3pooled/weights/best.pt
[ -f "$BEST" ] && python -u src/eval/track_recall.py --weights "$BEST" --imgsz $SZ \
    --conf 0.10 --out results/track_recall/yolov9m_1280_conf0.10 || echo "[warn] track recall"

python -u src/eval/summarize_counting_eval.py

LOG=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_ev1280_${SLURM_JOB_ID}.out
printf '%b\n' "1280px evaluation finished $(date) — same battery as the 640 roster.\n\n640 reference (test): yolov9m/yolov10m mAP50 0.498; keyframe-GT any-overlap best was YOLO11m 0.942P/0.634R; track-level recall 93.6%.\n\n=== native test rows ===\n$(grep -E '^\s+all ' $LOG | head -6)\n\n=== track-level recall @1280 ===\n$(grep -A8 'TRACK-LEVEL RECALL' $LOG)\n\nFull table: results/counting_eval/counting_eval_TABLE.md" \
  | mail -s "[DeltaAI] 1280px evaluation results (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
