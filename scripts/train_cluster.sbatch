#!/bin/bash
#SBATCH --job-name=wild_clus
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_clus_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_clus_%j.err
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project
python -u src/temporal/train_cluster.py \
  --counts-dir /work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10 \
  --labels data/temporal/tracks_labelled_orphan.csv \
  --out results/temporal/cluster_orphan
LOG=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_clus_${SLURM_JOB_ID}.out
printf '%b\n' "Detect-then-cluster counting finished $(date).\n\nReformulated: stage 1 learns 'is this track ON a deer' (unambiguous, 632 positives); stage 2 clusters accepted tracks into animals. Replaces the primary-vs-duplicate labelling that gave contradictory supervision.\n\n$(sed -n '/CROSS-VALIDATED/,\$p' "\$LOG" 2>/dev/null)" \
  | mail -s "[Delta] Detect-then-cluster counting results (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
