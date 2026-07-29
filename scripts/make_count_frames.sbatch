#!/bin/bash
#SBATCH --job-name=wild_cfrm
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cfrm_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cfrm_%j.err
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project
VIZ=/work/hdd/bgte/tislam6/wildlife_outputs/viz
python -u src/viz/count_frames.py \
    --counts-dir /work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10/merged \
    --source data/raw --out "$VIZ/count_frames_final" --per-video 2 --showcase-per-video 3 --showcase-min-score 0.80 \
    --track-scores results/temporal/calibrated_orphan/per_track_confidence.csv
echo "=== images ==="; find "$VIZ/count_frames" -name "*.jpg" | wc -l
printf '%b\n' "Frame-level counting images ready $(date).\n\n$VIZ/count_frames/ — for each video, the frames where the MOST deer are tracked at once, with EVERY tracked deer boxed and labelled 'ID <track> <conf>' at the box corner. Colour is stable per track id, so one animal keeps one colour across frames.\n\nThis makes both counting failure modes visible: two different IDs on one animal = fragmentation (over-count); a visible deer with no box = miss (under-count).\n\n_sheet_multideer.jpg = the 8 busiest frames." \
  | mail -s "[Delta] Frame-level counting images with track IDs (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
