#!/bin/bash
#SBATCH --job-name=gh_trec
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_trec_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_trec_%j.err

# Phase-B gate: track-level recall of the 236 GT deer. Small job -> A40 is fine.
set -e
module load python/anaconda3/2.12.0

source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
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
