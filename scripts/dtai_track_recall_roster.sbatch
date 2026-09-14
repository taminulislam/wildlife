#!/bin/bash
#SBATCH --job-name=gh_trec_roster
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_trec_roster_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_trec_roster_%j.err

# DeltaAI half of the per-animal recall sweep: the six Ultralytics-family detectors only.
#
# The five mmdetection models cannot run here -- there is no ARM build of the mmdet env,
# only envs/mmdet, which is x86 -- so they stay on the Delta job
# (scripts/delta_track_recall_roster.sbatch). That job skips any model whose summary
# already exists, so the two can run concurrently and write into the same output tree
# without colliding.
set -u
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project

R=/work/hdd/bgte/tislam6/wildlife_outputs/runs
OUTROOT=results/track_recall/roster_conf0.25
CONF=0.25

echo "=== Ultralytics roster track recall on GH200, conf $CONF, $(date) ==="
for spec in "yolov8m_640:yolo" "yolov9m_640:yolo" "yolov10m_640:yolo" \
            "yolo11m_640:yolo" "yolo12m_640:yolo" "rtdetr-l_640:rtdetr"; do
    name=${spec%%:*}; arch=${spec##*:}
    w=$R/${name}_v3pooled/weights/best.pt
    out=$OUTROOT/$name
    if [ ! -f "$w" ]; then echo "[skip] no weights for $name"; continue; fi
    if [ -f "$out/track_recall_summary.csv" ]; then echo "[have] $name"; continue; fi
    echo "--- $name ($arch)"
    python -u src/eval/track_recall.py --weights "$w" --arch "$arch" \
        --conf "$CONF" --out "$out" || echo "[FAIL] $name"
done

echo "=== summary ==="
for d in $OUTROOT/*/; do
    s=$d/track_recall_summary.csv
    [ -f "$s" ] && echo "$(basename $d): $(awk -F, 'NR==2{print "all="$3" touch="$6}' $s)"
done
echo "=== done $(date) ==="
