#!/bin/bash
#SBATCH --job-name=gh_count
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=64 --mem=400G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_count_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_count_%j.err

# PHASE B: detect -> track -> count over all 32 GT videos, then score MAE/RMSE against
# the 236-deer ground truth. This produces the paper's HEADLINE metric and the
# hand-tuned baseline the learned temporal head must beat.
#
# Detector: YOLO11m — best COUNTING detector per docs/RESULTS_LOG.md §4.3
# (keyframe GT, any-overlap, conf .25: P 0.942 / R 0.634), even though it ranks 4th
# on mAP50. Detection conf 0.10: recall-first, confirmation applied post-hoc.
# CLAHE is applied at inference (pipeline fix #5) — earlier counts were invalid.
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project

W=${W:-/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt}
OUT=${OUT:-results/counts/yolo11m_640_conf0.10}
N=4

echo "=== counting run $(date) — $W -> $OUT ==="
pids=()
for s in 0 1 2 3; do
  python -u src/track/run_counting.py --weights "$W" --out "$OUT" \
      --conf 0.10 --imgsz 640 --device $s --shard $s --nshards $N \
      > /work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_count_shard${s}_${SLURM_JOB_ID}.out 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "shards done (fail=$fail) $(date)"

python -u src/eval/count_eval.py --counts "$OUT" --gt data/annotate_v2/count_gt.csv

SUM=$OUT/eval/count_eval_summary.md
printf '%b\n' "PHASE B — first valid counting results, $(date).\n\nThis is the paper's headline metric: MAE/RMSE of unique-deer counts vs the 236-deer CVAT ground truth, and the best hand-tuned confirmation rule (the baseline the learned temporal head must beat).\n\nDetector: YOLO11m @640, conf 0.10, CLAHE applied at inference.\nNOTE: all counts produced before the CLAHE-at-inference fix were invalid; these are the first trustworthy ones.\n\n$(cat $SUM 2>/dev/null)" \
  | mail -s "[DeltaAI] Phase B: counting results MAE/RMSE (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
