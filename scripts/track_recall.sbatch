#!/bin/bash
#SBATCH --job-name=wild_trec
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_trec_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_trec_%j.err

# Phase-B gate: track-level recall of the 236 GT deer. Small job -> A40 is fine.
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project

W=${W:-/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolov9m_640_v3pooled/weights/best.pt}
CONF=${CONF:-0.10}
OUT=${OUT:-results/track_recall/yolov9m_640_conf${CONF}}

echo "=== track recall: $W  conf=$CONF ==="
python -u src/eval/track_recall.py --weights "$W" --conf "$CONF" --out "$OUT"

SUM=$(cat "$OUT/track_recall_summary.csv" 2>/dev/null | head -20)
printf '%b\n' "Track-level recall finished $(date).\n\nWeights: $W\nDetector conf: $CONF\n\nThis is the Phase-B gate: of the 236 GT deer, how many the detector finds at all.\n>90% = detector is fine, proceed to the counting head.\n70-90% = proceed, note the ceiling.\n<70% = detector is the bottleneck.\n\n$SUM\n\nFull CSVs: $OUT" \
  | mail -s "[Delta] Track-level recall results (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
