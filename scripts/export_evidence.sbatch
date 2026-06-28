#!/bin/bash
#SBATCH --job-name=wild_evid
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_evid_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_evid_%j.err

# CPU-only work (video seeking + drawing); uses the gpu queue for a reliable env.
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project

CBASE=${CBASE:-/work/hdd/bgte/tislam6/wildlife_outputs/counts}
EBASE=${EBASE:-/work/hdd/bgte/tislam6/wildlife_outputs/evidence}
SITES=${SITES:-MAS SHB SHW TON}
TOPK=${TOPK:-6}

for S in $SITES; do
  echo "=== evidence for $S ==="
  python -u src/track/export_evidence.py \
      --counts-dir "$CBASE/full_m640_$S" \
      --source data/raw \
      --out "$EBASE/full_m640_$S" \
      --topk "$TOPK"
done
echo "=== done $SLURM_JOB_ID ==="
