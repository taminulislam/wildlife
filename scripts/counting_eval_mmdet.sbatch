#!/bin/bash
#SBATCH --job-name=wild_cntmm
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cntmm_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_cntmm_%j.err

# Score the mmdetection roster under the ANY-OVERLAP criterion, the same protocol the
# Ultralytics roster already went through (counting_eval_all.sbatch).
#
# Why: mmdet's own training logs only report COCO mAP, i.e. IoU>=0.50. For counting, box
# tightness is irrelevant — if the box touches the deer at all, the tracker links it and
# the animal gets counted. So the mmdet-vs-YOLO comparison built from mmAP is the wrong
# comparison for this project; this produces the right one.
#
# 4 criteria x {full interpolated GT, human-keyframe GT} x {conf 0.25, 0.10}.
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/mmdet
cd /work/nvme/bgte/tislam6/wildlife_project

WORK=/work/hdd/bgte/tislam6/wildlife_outputs/mmdet_runs

# name | config | work-dir suffix
SPECS=(
  "dino_r50|configs/mmdet/deer_dino_r50.py|dino_r50_v3pooled"
  "faster-rcnn_r50|configs/mmdet/deer_faster-rcnn_r50.py|faster-rcnn_r50_v3pooled"
  "atss_r50|configs/mmdet/deer_atss_r50.py|atss_r50_v3pooled"
  "tood_r50|configs/mmdet/deer_tood_r50.py|tood_r50_v3pooled"
  "rtmdet_m|configs/mmdet/deer_rtmdet_m.py|rtmdet_m_v3pooled"
  "deformable-detr_r50|configs/mmdet/deer_deformable-detr_r50.py|deformable-detr_r50_v3pooled_bs4"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r name cfg wdname <<< "$spec"
  wd=$WORK/$wdname
  ckpt=$(ls -t "$wd"/best_coco_bbox_mAP_50_epoch_*.pth 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=$(ls -t "$wd"/epoch_*.pth 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then echo "[skip] $name (no checkpoint in $wd)"; continue; fi
  echo "=== $name  <- $ckpt"
  for cf in 0.25 0.10; do
    python -u src/eval/counting_detection_eval.py \
        --arch mmdet --config "$cfg" --weights "$ckpt" --device 0 \
        --data data/dataset/yolo_v3/data.yaml --split test --conf $cf \
        --tag ${name}_FULLGT   || echo "[warn] $name full $cf"
    python -u src/eval/counting_detection_eval.py \
        --arch mmdet --config "$cfg" --weights "$ckpt" --device 0 \
        --data data/dataset/yolo_v3_kf/data.yaml --split test --conf $cf \
        --tag ${name}_KEYFRAME || echo "[warn] $name kf $cf"
  done
done

# regenerate the combined table so mmdet and Ultralytics sit in ONE ranking
python -u src/eval/summarize_counting_eval.py || true

TBL=results/counting_eval/counting_eval_TABLE.md
printf '%b\n' "mmdetection models scored under the ANY-OVERLAP counting criterion, $(date) (job $SLURM_JOB_ID).\n\nThey are now in the same table as the YOLO roster, so the comparison is like-for-like instead of mmdet-mAP vs YOLO-mAP.\n\n$(head -40 $TBL 2>/dev/null)" \
  | mail -s "[Delta] mmdet scored on the counting criterion (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
