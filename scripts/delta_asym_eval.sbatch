#!/bin/bash
#SBATCH --job-name=asym_eval
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA100x4-preempt,gpuA40x4,gpuA40x4-preempt
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/asym_eval_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/asym_eval_%j.err

# Second half of the asymmetric-scale experiment: score the model delta_asym_scale.sbatch
# trained. Split out so a diverged training run does not also spend a counting pass, and
# so neither job needs a long walltime.
#
# Run this only once training has finished and its curve looks sane. Baselines to beat,
# fixed in advance: AP50 0.460, MAE 2.38, 55 of 83 held-out animals counted.
set -eo pipefail
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project
OUT=/work/hdd/bgte/tislam6/wildlife_outputs
NAME=yolo11m_640_asymscale
W="$OUT/runs/$NAME/weights/best.pt"
[ -f "$W" ] || { echo "no weights at $W -- has training finished?"; exit 1; }
echo "=== job $SLURM_JOB_ID  evaluating $NAME  $(date) ==="

echo "--- detection eval, same harness as the benchmark table"
python -u src/detect/eval_full.py --weights "$W" --imgsz 640 \
    --data data/dataset/yolo_v3/data.yaml \
    --tag "${NAME}" --out results/detection_eval || echo "[warn] eval_full failed"

echo "--- per-animal recall: did the 104/149/157 px deer come back?"
python -u src/eval/track_recall.py --weights "$W" --conf 0.25 \
    --out "results/track_recall/${NAME}" || echo "[warn] track_recall failed"

echo "--- end-to-end counting, frozen held-out protocol"
CO="$OUT/counts/asymscale_conf0.10"
python -u src/track/count_deer.py \
    --weights "$W" --source data/raw --cvat-dir data/cvat_export \
    --device 0 --imgsz 640 --conf 0.10 --iou 0.50 --contrast clahe \
    --dump-min-frames 1 --keep-orphans \
    --tracker src/track/botsort_deer_recall.yaml --out "$CO/shard0"
cp "$CO/shard0/counts.csv" "$CO/counts_shard0.csv" 2>/dev/null || true
python -u src/eval/pool_coverage.py --counts-dir "$CO" \
    --out results/counting_eval/pool_coverage_asymscale.csv || true
python -u src/eval/count_eval_heldout.py --counts "$CO" \
    --out results/counting_eval/heldout_asymscale.csv || true
echo "=== done $(date) ==="
