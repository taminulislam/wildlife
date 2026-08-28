#!/bin/bash
#SBATCH --job-name=gh_symctl
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_symctl_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_symctl_%j.err

# The control arm for the asymmetric-scale experiment: the standard symmetric scale=0.5,
# everything else identical, run in the same environment on the same hardware.
#
# It exists because the comparison baseline, yolo11m_640_v3pooled, was trained on Delta
# under Ultralytics 8.4.80, while the asymmetric run is on DeltaAI under 8.4.106. Comparing
# across those two would move the library as well as the augmentation, and 8.4.106 exposes
# defaults the older version did not. This arm removes that confound: asymmetric vs
# symmetric, one variable, same software.
#
# Filesystems are shared with Delta, so nothing needs copying and the outputs land in the
# same place either cluster runs it.
set -eo pipefail
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project
OUT=/work/hdd/bgte/tislam6/wildlife_outputs
NAME=yolo11m_640_symcontrol
echo "=== job $SLURM_JOB_ID  symmetric control (scale 0.5) on GH200  $(date) ==="

python -u src/detect/train.py \
    --model weights/yolo11m.pt --arch yolo \
    --data data/dataset/yolo_v3/data.yaml \
    --imgsz 640 --epochs 150 --batch 32 --patience 50 --seed 0 \
    --optimizer SGD --lr0 0.01 --warmup-epochs 3.0 \
    --scale 0.5 --mosaic 1.0 --close-mosaic 15 \
    --project "$OUT/runs" --name "$NAME"

W="$OUT/runs/$NAME/weights/best.pt"
python -u src/detect/eval_full.py --weights "$W" --imgsz 640 \
    --tag "${NAME}" --out results/detection_eval || echo "[warn] eval_full failed"
python -u src/eval/track_recall.py --weights "$W" --conf 0.25 \
    --out "results/track_recall/roster_conf0.25/${NAME}" || echo "[warn] track_recall failed"

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
