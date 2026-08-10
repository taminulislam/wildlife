#!/bin/bash
#SBATCH --job-name=qual_frames
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/qual_frames_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/qual_frames_%j.err

# Candidate frames for the qualitative figure. Renders every held-out video that holds
# at least two animals, so the selection is made from real options rather than from the
# first frame that happened to look acceptable.
#
# The work is CPU only -- it reads tracks.csv and decodes video, no model runs -- but the
# only account available here is gpu-only, so it takes a GPU it will not use. Decoding is
# sequential from frame 0 (seeking is unreliable on this footage), which is why this is a
# job and not a login-node command: the login node has a 30-minute CPU ceiling.
set -e
cd /work/nvme/bgte/tislam6/wildlife_project
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python

RUN=/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10
OUT=/work/hdd/bgte/tislam6/wildlife_outputs/viz/qual_candidates

# The 11 held-out videos with >=2 annotated animals. Held-out only, because the figure
# caption claims these frames come from video the detector never saw.
VIDEOS='AquaCultureRd_TON_12.03.25_LS,GiantCityRd_TON_12.03.25_LS,GolfDr_SHB_12.11.2025,Melvin_SHW_01.18.2026_LS,N25thBlue_MAS_12.22.25_LS_V2,NMarseilles_MAS_12.22.25_LS_V2,NShelbyRd(blue)_SHB_12.11.2025_LS,NWolfCreek(orange)_SHB_12.11.2025_LS,OikosRd_TON_12.03.25_LS,Robinson_SHW_01.18.2026_LS,TouchofNatureRd_TON_12.03.25_LS'

echo "=== job $SLURM_JOB_ID  $(date) ==="
mkdir -p "$OUT"

"$PY" -u src/viz/count_frames.py \
    --counts-dir "$RUN/merged" \
    --source data/raw \
    --videos "$VIDEOS" \
    --per-video 4 --min-deer 2 \
    --showcase-per-video 1 --showcase-min-score 0.80 \
    --only-confirmed \
    --track-scores results/temporal/calibrated_orphan/per_track_confidence.csv \
    --out "$OUT"

echo "=== rendered ==="
find "$OUT" -name '*.jpg' | wc -l
echo "=== done $(date) ==="
