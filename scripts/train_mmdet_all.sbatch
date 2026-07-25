#!/bin/bash
#SBATCH --job-name=wild_mmdet
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=40:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_mmdet_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_mmdet_%j.err

# Second CVPR detector wave: 6 mmdetection architectures on the same pooled split.
# Heavy models launch first so total wall-clock ~= the longest single model.
# Custom BEGIN/END emails: model list at start, results summary at end.
set -e
mkdir -p /work/hdd/bgte/tislam6/wildlife_outputs/logs

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/mmdet

cd /work/nvme/bgte/tislam6/wildlife_project

EMAIL=islamtaminul@gmail.com
notify() { printf '%b\n' "$2" | mail -s "$1" "$EMAIL" 2>/dev/null || true; }

LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
WORK=/work/hdd/bgte/tislam6/wildlife_outputs/mmdet_runs
mkdir -p "$WORK"
TRAIN_PY=$(python -c "import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),'.mim','tools','train.py'))")
TEST_PY=$(python -c "import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),'.mim','tools','test.py'))")

# name | config | epochs-note  (heavy transformers FIRST for best GPU packing)
SPECS=(
  "dino_r50|configs/mmdet/deer_dino_r50.py|36e (native DINO schedule)"
  "deformable-detr_r50|configs/mmdet/deer_deformable-detr_r50.py|50e (native schedule)"
  "faster-rcnn_r50|configs/mmdet/deer_faster-rcnn_r50.py|70e"
  "atss_r50|configs/mmdet/deer_atss_r50.py|70e"
  "tood_r50|configs/mmdet/deer_tood_r50.py|70e"
  "rtmdet_m|configs/mmdet/deer_rtmdet_m.py|70e"
)

echo "=== job $SLURM_JOB_ID on $SLURM_NODELIST  $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv 2>/dev/null || true

body="Job $SLURM_JOB_ID started on $SLURM_NODELIST at $(date).\n\nNow training 6 mmdetection architectures (4 GPUs, dynamic scheduling) on the pooled all-site deer split:\n"
for s in "${SPECS[@]}"; do IFS='|' read -r n c e <<< "$s"; body+="  - $n  ($e)\n"; done
body+="\nAll fine-tune from COCO-pretrained weights, 1 class (deer), CLAHE thermal frames.\nCompare against the Ultralytics roster (best 640px: YOLOv9m/YOLOv10m mAP50 0.498).\nYou will get a results-summary email when all models finish."
notify "[Delta] Training STARTED: 6 mmdetection models (job $SLURM_JOB_ID)" "$body"

run_one() {
  local gpu=$1 spec=$2
  IFS='|' read -r name cfg _ <<< "$spec"
  local wd=$WORK/${name}_v3pooled
  local log=$LOGS/wild_mmdet_${name}_${SLURM_JOB_ID}.out
  echo "[gpu $gpu] START $name  ($cfg) -> $wd"
  CUDA_VISIBLE_DEVICES=$gpu python -u "$TRAIN_PY" "$cfg" \
      --work-dir "$wd" --resume --cfg-options randomness.seed=0 \
      > "$log" 2>&1
  # score BEST checkpoint on the held-out TEST split
  local best=$(ls -t "$wd"/best_coco_bbox_mAP_50_epoch_*.pth 2>/dev/null | head -1)
  [ -z "$best" ] && best=$(ls -t "$wd"/epoch_*.pth 2>/dev/null | head -1)
  if [ -n "$best" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -u "$TEST_PY" "$cfg" "$best" \
        --work-dir "$wd/test_eval" >> "$log" 2>&1 || echo "[gpu $gpu] $name test-eval failed"
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

# ---- results summary email (TEST-split COCO metrics per model) ----
summary="Job $SLURM_JOB_ID (6 mmdetection models) finished at $(date).\n\nTEST-split results (held-out videos, all 4 sites):\n"
for s in "${SPECS[@]}"; do
  IFS='|' read -r name cfg _ <<< "$s"
  log=$LOGS/wild_mmdet_${name}_${SLURM_JOB_ID}.out
  line=$(grep -oE "bbox_mAP_50: [0-9.]+.*" "$log" 2>/dev/null | tail -1)
  summary+="\n== $name ==\n  ${line:-no test metrics found - check $log}\n"
done
summary+="\nUltralytics roster reference (640px test): YOLOv9m/YOLOv10m mAP50 0.498, YOLO11m P 0.732.\nWork dirs: $WORK/<model>_v3pooled"
notify "[Delta] Training ENDED: 6 mmdetection models results (job $SLURM_JOB_ID)" "$summary"

echo "=== all done $SLURM_JOB_ID  $(date) ==="
