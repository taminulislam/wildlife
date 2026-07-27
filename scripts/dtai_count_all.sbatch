#!/bin/bash
#SBATCH --job-name=gh_count
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=64 --mem=400G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_count_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_count_%j.err

# PHASE B: detect -> track -> count all 32 videos, then score against the 236-deer GT.
# Produces the paper's headline baseline: tracking-by-detection + best hand-tuned rule,
# i.e. the number the learned temporal head has to beat.
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project

W=${W:-/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt}
CONF=${CONF:-0.10}     # low on purpose: over-generate candidates, rule/head filters
OUT=/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseB_yolo11m_conf${CONF}
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
mkdir -p "$OUT"

echo "=== job $SLURM_JOB_ID  weights=$W  conf=$CONF  $(date) ==="
printf '%b\n' "Counting run started $(date) on DeltaAI (job $SLURM_JOB_ID).\n\nDetector: YOLO11m @640 (best COUNTING detector: P 0.942 / R 0.634 on human-keyframe GT, any-overlap).\nTracker: BoT-SORT + camera-motion compensation, detector conf $CONF.\nAll 32 videos, sharded over 4 GH200s.\n\nThis produces the paper's counting baseline (MAE/RMSE vs the 236-deer GT) with the best hand-tuned confirmation rule - the number the learned temporal head must beat.\n\nA results email follows." \
  | mail -s "[DeltaAI] Counting run STARTED (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true

pids=()
for i in 0 1 2 3; do
  python -u src/track/count_deer.py \
      --weights "$W" --source data/raw --cvat-dir data/cvat_export \
      --shards 4 --shard $i --device $i \
      --imgsz 640 --conf "$CONF" --contrast clahe \
      --out "$OUT/shard$i" > "$LOGS/gh_count_${SLURM_JOB_ID}_shard$i.out" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "shards done (fail=$fail)"

# count_eval.py globs counts_*.csv in ONE dir
for i in 0 1 2 3; do
  [ -f "$OUT/shard$i/counts.csv" ] && cp "$OUT/shard$i/counts.csv" "$OUT/counts_shard$i.csv"
done
python -u src/eval/count_eval.py --counts "$OUT" --gt data/annotate_v2/count_gt.csv

MD=$OUT/eval/count_eval_summary.md
printf '%b\n' "Counting run FINISHED $(date) (job $SLURM_JOB_ID, fail=$fail).\n\n$(cat $MD 2>/dev/null)\n\nFull outputs: $OUT" \
  | mail -s "[DeltaAI] COUNTING RESULTS — paper headline metric (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
