#!/bin/bash
#SBATCH --job-name=wild_ttc
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_ttc_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_ttc_%j.err

# PHASE C: train the temporal confirmation head (TTC) and compare it head-to-head with
# the best hand-tuned rule on the SAME held-out test videos. Short job (<2 h) so it
# lands in Delta's fast backfill lane; the login node must never run model forwards.
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project

COUNTS=${COUNTS:-/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseB_yolo11m_conf0.10}
LABELS=${LABELS:-data/temporal/tracks_labelled.csv}
TAG=${TAG:-ttc_v1}
EPOCHS=${EPOCHS:-200}

echo "=== TTC training  counts=$COUNTS  labels=$LABELS  $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

python -u src/temporal/train_head.py \
    --counts-dir "$COUNTS" --labels "$LABELS" \
    --epochs "$EPOCHS" --out "results/temporal/$TAG"

LOG=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_ttc_${SLURM_JOB_ID}.out
printf '%b\n' "Temporal head (TTC) training finished $(date).\n\nThis is the paper's headline experiment: LEARNED track confirmation vs the best hand-tuned rule, both scored by counting MAE on the same held-out test videos (the rule's thresholds swept on train+val, the head's threshold chosen on val - neither sees test).\n\n$(sed -n '/HELD-OUT TEST/,$p' "$LOG" 2>/dev/null)" \
  | mail -s "[Delta] TTC temporal head results (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
