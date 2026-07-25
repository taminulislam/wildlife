#!/bin/bash
#SBATCH --job-name=wild_wave2
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_wave2_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_wave2_%j.err

# ONE job, ONE queue wait, A100 only: all wave-2 trainings (yolov10m/yolov9m @1280 +
# 6 mmdetection models) chained back-to-back. A GPU that finishes one model starts
# the next instantly - nothing ever re-enters the queue. Heavy tasks first so total
# wall-clock ~= the longest single task (~18h). Every task is resume-safe, and the
# job auto-requeues on preemption.
set -e
mkdir -p /work/hdd/bgte/tislam6/wildlife_outputs/logs

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh

cd /work/nvme/bgte/tislam6/wildlife_project

EMAIL=islamtaminul@gmail.com
notify() { printf '%b\n' "$2" | mail -s "$1" "$EMAIL" 2>/dev/null || true; }

ENV_YOLO=/work/nvme/bgte/tislam6/envs/wildlife
ENV_MMDET=/work/nvme/bgte/tislam6/envs/mmdet
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
RUNS=/work/hdd/bgte/tislam6/wildlife_outputs/runs
WORK=/work/hdd/bgte/tislam6/wildlife_outputs/mmdet_runs
DATA=data/dataset/yolo_v3/data.yaml
mkdir -p "$WORK"

# task name | kind (yolo1280/mmdet) | payload | note   (heavy-first packing)
SPECS=(
  "dino_r50|mmdet|configs/mmdet/deer_dino_r50.py|DINO-4scale R50, 36e native"
  "yolov10m_1280|yolo1280|yolov10m|winner @1280, 70e"
  "yolov9m_1280|yolo1280|yolov9m|co-winner @1280, 70e"
  "deformable-detr_r50|mmdet|configs/mmdet/deer_deformable-detr_r50.py|Deformable DETR R50, 50e native"
  "faster-rcnn_r50|mmdet|configs/mmdet/deer_faster-rcnn_r50.py|Faster R-CNN R50, 70e"
  "atss_r50|mmdet|configs/mmdet/deer_atss_r50.py|ATSS R50, 70e"
  "tood_r50|mmdet|configs/mmdet/deer_tood_r50.py|TOOD R50, 70e"
  "rtmdet_m|mmdet|configs/mmdet/deer_rtmdet_m.py|RTMDet-M, 70e"
)

echo "=== job $SLURM_JOB_ID on $SLURM_NODELIST  $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv 2>/dev/null || true

body="Job $SLURM_JOB_ID started on $SLURM_NODELIST at $(date).\n\nWave-2 training: 8 models chained on 4 A100s (one-after-one per GPU, no re-queue):\n"
for s in "${SPECS[@]}"; do IFS='|' read -r n k p note <<< "$s"; body+="  - $n  ($note)\n"; done
body+="\nDataset: pooled all-site split (yolo_v3), CLAHE thermal frames.\n640px reference to beat: YOLOv9m/YOLOv10m mAP50 0.498.\nYou will get a results-summary email when everything finishes."
notify "[Delta] Training STARTED: wave-2 (2x YOLO@1280 + 6 mmdetection) job $SLURM_JOB_ID" "$body"

run_one() {
  local gpu=$1 spec=$2
  IFS='|' read -r name kind payload note <<< "$spec"
  local log=$LOGS/wild_wave2_${name}_${SLURM_JOB_ID}.out
  echo "[gpu $gpu] START $name ($note) -> log $log"
  if [ "$kind" = "yolo1280" ]; then
    bash -c "source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh && \
      conda activate $ENV_YOLO && \
      python -u src/detect/train.py \
        --model weights/${payload}.pt --arch yolo --data $DATA \
        --imgsz 1280 --epochs 70 --batch 16 --patience 50 \
        --optimizer SGD --lr0 0.01 --device $gpu \
        --save-period 10 --resume \
        --project $RUNS --name ${payload}_1280_v3pooled" > "$log" 2>&1
  else
    local wd=$WORK/${name}_v3pooled
    bash -c "source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh && \
      conda activate $ENV_MMDET && \
      TRAIN_PY=\$(python -c 'import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),\".mim\",\"tools\",\"train.py\"))') && \
      TEST_PY=\$(python -c 'import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),\".mim\",\"tools\",\"test.py\"))') && \
      CUDA_VISIBLE_DEVICES=$gpu python -u \$TRAIN_PY $payload \
        --work-dir $wd --resume --cfg-options randomness.seed=0 && \
      best=\$(ls -t $wd/best_coco_bbox_mAP_50_epoch_*.pth 2>/dev/null | head -1) ; \
      [ -z \"\$best\" ] && best=\$(ls -t $wd/epoch_*.pth 2>/dev/null | head -1) ; \
      [ -n \"\$best\" ] && CUDA_VISIBLE_DEVICES=$gpu python -u \$TEST_PY $payload \"\$best\" \
        --work-dir $wd/test_eval" > "$log" 2>&1
  fi
  echo "[gpu $gpu] DONE  $name"
}

NGPU=${SLURM_GPUS_ON_NODE:-4}
free=($(seq 0 $((NGPU-1))))
declare -A pid2gpu
idx=0; n=${#SPECS[@]}
while (( idx < n )) || (( ${#pid2gpu[@]} > 0 )); do
  while (( ${#free[@]} > 0 )) && (( idx < n )); do
    g=${free[0]}; free=("${free[@]:1}")
    run_one "$g" "${SPECS[$idx]}" &
    pid2gpu[$!]=$g
    idx=$((idx+1))
  done
  wait -n || true
  for p in "${!pid2gpu[@]}"; do
    if ! kill -0 "$p" 2>/dev/null; then
      free+=("${pid2gpu[$p]}"); unset "pid2gpu[$p]"
    fi
  done
done

# ---- results summary email ----
summary="Job $SLURM_JOB_ID (wave-2: 2x YOLO@1280 + 6 mmdetection) finished at $(date).\n\nTEST-split results:\n"
for s in "${SPECS[@]}"; do
  IFS='|' read -r name kind payload note <<< "$s"
  log=$LOGS/wild_wave2_${name}_${SLURM_JOB_ID}.out
  summary+="\n== $name ($note) ==\n"
  if [ "$kind" = "yolo1280" ]; then
    testln=$(awk '/native val on .test/{f=1} f' "$log" 2>/dev/null | grep -E "^\s+all " | head -1)
    ep=$(grep -c "^\[epoch" "$log" 2>/dev/null)
    summary+="  epochs run: ${ep}\n  TEST (images boxes P R mAP50 mAP50-95):${testln:-  not found}\n"
  else
    line=$(grep -oE "bbox_mAP_50: [0-9.]+.*" "$log" 2>/dev/null | tail -1)
    summary+="  ${line:-no test metrics found - check $log}\n"
  fi
done
summary+="\n640px roster reference: YOLOv9m/YOLOv10m mAP50 0.498, YOLO11m P 0.732.\nYOLO run dirs: $RUNS/<model>_1280_v3pooled | mmdet: $WORK/<model>_v3pooled"
notify "[Delta] Training ENDED: wave-2 results (job $SLURM_JOB_ID)" "$summary"

echo "=== all done $SLURM_JOB_ID  $(date) ==="
