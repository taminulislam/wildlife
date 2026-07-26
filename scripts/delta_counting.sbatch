#!/bin/bash
#SBATCH --job-name=wild_count
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=32 --mem=200G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_count_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_count_%j.err

# PHASE B: detect -> track -> count over all 32 GT videos, then score MAE/RMSE against
# the 236-deer ground truth. This produces the paper's HEADLINE metric and the
# hand-tuned baseline the learned temporal head must beat.
#
# Detector: YOLOv9m @1280 — best per RESULTS_LOG 4.5 (track-level recall 97.0% vs
# 93.6% at 640; deer never detected halved to 3.0%). Counting cares about track
# recall, not mAP. Detection conf 0.10: recall-first, confirm post-hoc.
# CLAHE is applied at inference (pipeline fix #5) — earlier counts were invalid.
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project

W=${W:-/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolov9m_1280_v3pooled/weights/best.pt}
OUT=${OUT:-results/counts/yolov9m_1280_conf0.10_delta}
N=4

echo "=== counting run $(date) — $W -> $OUT ==="
pids=()
for s in 0 1 2 3; do
  python -u src/track/run_counting.py --weights "$W" --out "$OUT" \
      --conf 0.10 --imgsz 1280 --device $s --shard $s --nshards $N \
      > /work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_count_shard${s}_${SLURM_JOB_ID}.out 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "shards done (fail=$fail) $(date)"

python -u src/eval/count_eval.py --counts "$OUT" --gt data/annotate_v2/count_gt.csv

SUM=$OUT/eval/count_eval_summary.md
printf '%b\n' "PHASE B — first valid counting results, $(date).\n\nThis is the paper's headline metric: MAE/RMSE of unique-deer counts vs the 236-deer CVAT ground truth, and the best hand-tuned confirmation rule (the baseline the learned temporal head must beat).\n\nDetector: YOLO11m @640, conf 0.10, CLAHE applied at inference.\nNOTE: all counts produced before the CLAHE-at-inference fix were invalid; these are the first trustworthy ones.\n\n$(cat $SUM 2>/dev/null)" \
  | mail -s "[Delta] Phase B: counting results MAE/RMSE (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
