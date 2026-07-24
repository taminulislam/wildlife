#!/bin/bash
#SBATCH --job-name=wild_cvat
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cvat_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cvat_%j.err

set -e
mkdir -p /work/hdd/bgte/tislam6/wildlife_outputs/logs

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife

cd /work/nvme/bgte/tislam6/wildlife_project

# Fresh staging set built purely from the 32 CVAT exports (no old loose positives).
OUT=${OUT:-data/annotate_v2}
STRIDE=${STRIDE:-2}
CAP=${CAP:-150}
NEGPP=${NEGPP:-1.5}

echo "=== job $SLURM_JOB_ID: CVAT -> YOLO, stride=$STRIDE cap=$CAP neg/pos=$NEGPP -> $OUT ==="
rm -rf "$OUT"; mkdir -p "$OUT"

n=0
for xml in data/cvat_export/*.xml; do
    n=$((n+1))
    echo "--- [$n] $(basename "$xml")"
    python -u src/dataset/cvat_to_yolo.py \
        --xml "$xml" --source data/raw --out "$OUT" \
        --pos-stride "$STRIDE" --pos-cap "$CAP" --neg-per-pos "$NEGPP" \
        --contrast "${CONTRAST:-clahe}" --preview
done

echo "=== done: $(ls "$OUT/frames" | wc -l) images, $(ls "$OUT/labels" | wc -l) labels ==="
echo "count GT:"; cat "$OUT/count_gt.csv"
