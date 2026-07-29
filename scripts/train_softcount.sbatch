#!/bin/bash
#SBATCH --job-name=wild_soft
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_soft_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_soft_%j.err
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project
POOL=${POOL:-/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10}
LAB=${LAB:-data/temporal/tracks_labelled_orphan.csv}
python -u src/temporal/train_softcount.py --counts-dir "$POOL" --labels "$LAB" \
    --out results/temporal/softcount_orphan
LOG=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_soft_${SLURM_JOB_ID}.out
printf '%b\n' "Calibrated soft-count head finished $(date).\n\nTrains on the COUNTING objective directly (count = sum of calibrated probabilities), which removes the hand-tuned rule's structural advantage and lets fragments share probability mass. Also produces the proposal's required deliverables: calibrated per-deer confidence, a review threshold, and per-video count uncertainty.\n\n$(sed -n '/CROSS-VALIDATED/,\$p' "\$LOG" 2>/dev/null)" \
  | mail -s "[Delta] Soft-count head + calibrated confidence (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
