#!/bin/bash
#SBATCH --job-name=wild_scaleaug
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_scaleaug_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_scaleaug_%j.err

# Close the training-distribution hole at LARGE scale (§6.11).
#
# The training split's largest box is 95.6 px and it contains ZERO boxes at or above
# 100 px, yet three held-out deer appear at 104, 149 and 157 px -- one of them never
# detected at all. That is not a capability limit, it is a gap in what the detector was
# shown, and it is the one remaining lever that can move the headline number.
#
# FIRST ATTEMPT FAILED (job 20882753): scale=0.9 collapsed training to mAP 0 by epoch 5.
# Ultralytics' `scale` is SYMMETRIC -- 0.9 means resize in [0.1, 1.9] -- so it shrinks a
# 27 px deer to 2.7 px as often as it enlarges it, and mosaic=1.0 halves sizes on top of
# that. On a corpus that is 71% small objects those examples are unlearnable and poison
# the loss.
#
# This version enlarges without destroying the small end: scale 0.6 gives [0.4, 1.6] and
# mosaic is cut to 0.4 so most batches keep objects at native size rather than tiling four
# images into one. Net effect is more large presentations and no sub-pixel ones.
set -eo pipefail
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project
OUT=/work/hdd/bgte/tislam6/wildlife_outputs
echo "=== job $SLURM_JOB_ID  scale-aug  $(date) ==="

python -u src/detect/train.py \
    --model weights/yolo11m.pt --arch yolo \
    --data data/dataset/yolo_v3/data.yaml \
    --imgsz 640 --epochs 80 --batch 48 --patience 20 \
    --optimizer SGD --lr0 0.01 --scale 0.6 --mosaic 0.4 \
    --project "$OUT/runs" --name yolo11m_640_scaleaug

W="$OUT/runs/yolo11m_640_scaleaug/weights/best.pt"
CO="$OUT/counts/scaleaug_conf0.10"
python -u src/track/count_deer.py \
    --weights "$W" --source data/raw --cvat-dir data/cvat_export \
    --device 0 --imgsz 640 --conf 0.10 --iou 0.50 --contrast clahe \
    --dump-min-frames 1 --keep-orphans \
    --tracker src/track/botsort_deer_recall.yaml --out "$CO/shard0"
cp "$CO/shard0/counts.csv" "$CO/counts_shard0.csv" 2>/dev/null || true
python -u src/eval/pool_coverage.py --counts-dir "$CO" --out results/counting_eval/pool_coverage_scaleaug.csv
python -u src/eval/count_eval_heldout.py --counts "$CO" --out results/counting_eval/heldout_scaleaug.csv

printf '%b\n' "Scale-augmented detector finished $(date) (job $SLURM_JOB_ID).\n\nTargets the three held-out deer at 104/149/157 px that sit outside the training distribution (max training box 95.6 px, zero above 100).\n\nBaseline to beat: pool C, MAE 2.38, 55/83 = 66.3 percent counted.\n\n$(tail -n 14 $OUT/logs/wild_scaleaug_${SLURM_JOB_ID}.out 2>/dev/null)" \
  | mail -s "[Delta] Scale augmentation: did the large deer come back? (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
