#!/bin/bash
#SBATCH --job-name=speed
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/speed_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/speed_%j.err

# Throughput of the deployed counting pipeline and of the detector roster, one A100,
# one process, nothing else on the GPU. Not preemptable on purpose: a requeue would
# restart the timers on a different node.
set -e
cd /work/nvme/bgte/tislam6/wildlife_project
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python
OUT=/u/tislam6/wildlife_speed_$SLURM_JOB_ID          # home: immune to the project quota
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm --format=csv | tee "$OUT/gpu.txt"
$PY -u src/eval/speed_benchmark.py \
  --video "data/raw/TON_Videos_LS/visit1/OikosRd_TON_12.03.25_LS.mp4" \
  --out "$OUT" --frames 2000 --device 0
