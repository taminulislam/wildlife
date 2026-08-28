#!/bin/bash
#SBATCH --job-name=trec_roster
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/trec_roster_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/trec_roster_%j.err

# Per-ANIMAL recall for the whole benchmark roster, so the detection table can carry the
# quantity a counting paper actually needs: of the 235 annotated deer, how many does this
# detector fire on in at least one frame?
#
# Frame-level recall of ~0.45 is compatible with per-animal recall of 0.99 (misses
# scattered over frames of animals that are still found) or of 0.60 (whole animals never
# seen). Only this run distinguishes them, and the distinction is the paper's thesis.
#
# Detection runs only on GT frames, subsampled, so this is far cheaper than full-video
# inference. The two families need different conda envs, hence the split loop.
set -u
cd /work/nvme/bgte/tislam6/wildlife_project
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh

R=/work/hdd/bgte/tislam6/wildlife_outputs/runs
M=/work/hdd/bgte/tislam6/wildlife_outputs/mmdet_runs
OUTROOT=results/track_recall/roster_conf0.25
CONF=0.25          # the threshold the detection table reports at

echo "=== roster track recall, conf $CONF, started $(date) ==="

conda activate /work/nvme/bgte/tislam6/envs/wildlife
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

conda activate /work/nvme/bgte/tislam6/envs/mmdet
for name in atss_r50 tood_r50 dino_r50 faster-rcnn_r50 rtmdet_m; do
    cfg=$(ls $M/${name}_v3pooled/*.py 2>/dev/null | head -1)
    w=$(ls $M/${name}_v3pooled/best_coco*.pth 2>/dev/null | head -1)
    out=$OUTROOT/$name
    if [ -z "$cfg" ] || [ -z "$w" ]; then echo "[skip] no config/weights for $name"; continue; fi
    if [ -f "$out/track_recall_summary.csv" ]; then echo "[have] $name"; continue; fi
    echo "--- $name (mmdet)"
    python -u src/eval/track_recall.py --arch mmdet --config "$cfg" --weights "$w" \
        --conf "$CONF" --out "$out" || echo "[FAIL] $name"
done

echo "=== summary ==="
for d in $OUTROOT/*/; do
    s=$d/track_recall_summary.csv
    [ -f "$s" ] && echo "$(basename $d): $(awk -F, 'NR==2{print "all="$3" touch="$6}' $s)"
done
echo "=== done $(date) ==="
