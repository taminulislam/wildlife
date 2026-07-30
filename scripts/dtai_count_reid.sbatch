#!/bin/bash
#SBATCH --job-name=gh_countreid
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=64 --mem=400G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_countreid_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_countreid_%j.err

# JOB A of two: does appearance-based association change the count?
#
# Best model on the project's own criterion is YOLO11m @640 (any-overlap F1 0.758, §3.2),
# and it is the same detector Phase E used, so this run differs from Phase E in EXACTLY
# one flag: with_reid True instead of False. Everything else -- weights, conf, NMS IoU,
# every tracker threshold -- is held fixed, which is what makes the comparison clean.
#
# Baseline to beat (Phase E, /work/hdd/.../counts/phaseE_maxrecall_conf0.02):
#   candidates 18 349 | reached 228/235 | primary 210/235 | held-out MAE 2.54, 54/83
#
# Read `primary` and the held-out counted total, not `reached`. ReID merges fragments of
# one animal; it cannot split a box that covers two. If it works, duplicates fall, the
# rule can afford to be less aggressive, and the counted total rises.
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project

W=${W:-/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt}
CONF=${CONF:-0.02}
OUT=/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseG_reid_conf${CONF}
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
mkdir -p "$OUT"
echo "=== job $SLURM_JOB_ID  weights=$W  conf=$CONF  with_reid=True  $(date) ==="

pids=()
for i in 0 1 2 3; do
  python -u src/track/count_deer.py \
      --weights "$W" --source data/raw --cvat-dir data/cvat_export \
      --shards 4 --shard $i --device $i \
      --imgsz 640 --conf "$CONF" --iou 0.50 --contrast clahe \
      --dump-min-frames 1 --keep-orphans \
      --tracker src/track/botsort_deer_reid.yaml \
      --out "$OUT/shard$i" > "$LOGS/gh_countreid_${SLURM_JOB_ID}_shard$i.out" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "shards done (fail=$fail)"
if [ "$fail" != "0" ]; then
  printf '%b\n' "ReID counting run FAILED (job $SLURM_JOB_ID). Shard logs:\n$(tail -n 30 $LOGS/gh_countreid_${SLURM_JOB_ID}_shard0.out)" \
    | mail -s "[DeltaAI] ReID counting FAILED (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
fi

for i in 0 1 2 3; do
  [ -f "$OUT/shard$i/counts.csv" ] && cp "$OUT/shard$i/counts.csv" "$OUT/counts_shard$i.csv"
done
python -u src/track/merge_shards.py --run "$OUT" || true
python -u src/eval/count_eval.py --counts "$OUT" --gt data/annotate_v2/count_gt.csv || true

# the three numbers that decide this experiment
python -u src/eval/pool_coverage.py --counts-dir "$OUT" \
    --out results/counting_eval/pool_coverage_reid.csv > "$LOGS/gh_countreid_${SLURM_JOB_ID}_verdict.out" 2>&1 || true
python -u src/eval/count_eval_heldout.py --counts "$OUT" \
    --out results/counting_eval/heldout_phaseG_reid.csv >> "$LOGS/gh_countreid_${SLURM_JOB_ID}_verdict.out" 2>&1 || true

printf '%b\n' "ReID counting FINISHED $(date) (job $SLURM_JOB_ID, fail=$fail).\n\nOnly change vs Phase E: with_reid True.\n\n=== PHASE E BASELINE ===\ncandidates 18349 | reached 228/235 | primary 210/235 | held-out MAE 2.54, 54/83 counted\n\n=== THIS RUN ===\n$(tail -n 25 $LOGS/gh_countreid_${SLURM_JOB_ID}_verdict.out)\n\nFull outputs: $OUT" \
  | mail -s "[DeltaAI] ReID counting: did appearance association help? (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
