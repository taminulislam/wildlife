#!/bin/bash
#SBATCH --job-name=trec_mmdet
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA40x4-preempt,gpuA100x4,gpuA100x4-preempt
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/trec_mmdet_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/trec_mmdet_%j.err

# The mmdetection half of the per-animal recall column. The six Ultralytics detectors run
# on DeltaAI (dtai_track_recall_roster.sbatch); these five cannot, because envs/mmdet is
# x86 and there is no ARM build. Kept as a separate script rather than reusing the roster
# one so it cannot touch a model DeltaAI is still working on.
set -u
cd /work/nvme/bgte/tislam6/wildlife_project
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/mmdet

M=/work/hdd/bgte/tislam6/wildlife_outputs/mmdet_runs
OUTROOT=results/track_recall/roster_conf0.25
CONF=0.25

echo "=== mmdet per-animal recall, conf $CONF, $(date) ==="
for name in atss_r50 tood_r50 dino_r50 faster-rcnn_r50 rtmdet_m; do
    cfg=$(ls $M/${name}_v3pooled/*.py 2>/dev/null | head -1)
    w=$(ls $M/${name}_v3pooled/best_coco*.pth 2>/dev/null | head -1)
    out=$OUTROOT/$name
    if [ -z "$cfg" ] || [ -z "$w" ]; then echo "[skip] no config/weights for $name"; continue; fi
    if [ -f "$out/track_recall_summary.csv" ]; then echo "[have] $name"; continue; fi
    echo "--- $name"
    python -u src/eval/track_recall.py --arch mmdet --config "$cfg" --weights "$w" \
        --conf "$CONF" --out "$out" || echo "[FAIL] $name"
done

echo "=== summary ==="
for d in $OUTROOT/*/; do
    s=$d/track_recall_summary.csv
    [ -f "$s" ] && awk -F, -v m="$(basename $d)" 'NR==2{printf "%-18s all=%s touch=%.3f\n", m, $3, $6}' "$s"
done
echo "=== done $(date) ==="
