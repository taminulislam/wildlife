#!/bin/bash
#SBATCH --job-name=wild_gh1280
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_gh1280_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_gh1280_%j.err

# 1280px resolution ablation on DeltaAI GH200 (aarch64, 120GB/GPU).
# Delta's A100 queue is saturated; DeltaAI has idle nodes and 2x the banked hours.
# The GH200's 120GB lets 1280px run at batch 32 (vs 16 on a 40GB A100).
# Storage is SHARED with Delta, so weights/results land in the usual places.
set -e
mkdir -p /work/hdd/bgte/tislam6/wildlife_outputs/logs

# NCSA's miniforge3-pytorch module already ships an aarch64/GH200-optimized torch
# (and cv2); we layer a --system-site-packages venv on it for ultralytics/pycocotools
# rather than pip-installing torch, whose generic wheels may not target GH200.
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate

cd /work/nvme/bgte/tislam6/wildlife_project

EMAIL=islamtaminul@gmail.com
notify() { printf '%b\n' "$2" | mail -s "$1" "$EMAIL" 2>/dev/null || true; }

LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
RUNS=/work/hdd/bgte/tislam6/wildlife_outputs/runs
DATA=data/dataset/yolo_v3/data.yaml
EPOCHS=${EPOCHS:-70}
BATCH=${BATCH:-32}
MODELS=(yolov10m yolov9m)

echo "=== job $SLURM_JOB_ID on $SLURM_NODELIST  $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

notify "[DeltaAI] Training STARTED: 1280px ablation on GH200 (job $SLURM_JOB_ID)" \
"Job $SLURM_JOB_ID started on $SLURM_NODELIST (NCSA DeltaAI, GH200) at $(date).\n
Now training, one per GPU:\n
  1. YOLOv10m @ 1280px  (640 baseline: mAP50 0.498 / 0.670 on human-keyframe GT)\n
  2. YOLOv9m  @ 1280px  (640 baseline: mAP50 0.498 / 0.640 on human-keyframe GT)\n
Config: ${EPOCHS} epochs, batch ${BATCH} (GH200 120GB allows 2x the A100 batch), SGD lr0=0.01,
patience 50, pooled all-site split, CLAHE thermal frames.\n
Why 1280: 71% of deer are COCO-small and AP-small is 2-3.5x worse than AP-medium at 640,
so input resolution is the largest remaining accuracy lever.\n
A results summary email follows when training ends."

pids=()
for i in 0 1; do
  m=${MODELS[$i]}
  name=${m}_1280_v3pooled
  log=$LOGS/wild_gh1280_${m}_${SLURM_JOB_ID}.out
  echo "[gpu $i] START $m @1280 batch=$BATCH -> $name"
  python -u src/detect/train.py \
      --model weights/${m}.pt --arch yolo --data "$DATA" \
      --imgsz 1280 --epochs "$EPOCHS" --batch "$BATCH" --patience 50 \
      --optimizer SGD --lr0 0.01 --device "$i" \
      --save-period 10 --workers 4 --resume \
      --project "$RUNS" --name "$name" > "$log" 2>&1 &
  pids+=($!)
  # Stagger: Ultralytics' AMP check downloads+unzips coco8 into ./datasets on first
  # run. Two processes starting together race on the unzip and one dies with
  # FileExistsError (job 2739410). 90s lets the first finish any such setup. STAGGER
  [ $i -eq 0 ] && sleep 90
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

summary="Job $SLURM_JOB_ID (1280px ablation, DeltaAI GH200) finished $(date). Status: $([ $fail = 0 ] && echo OK || echo 'SOME FAILURES').\n"
for m in "${MODELS[@]}"; do
  log=$LOGS/wild_gh1280_${m}_${SLURM_JOB_ID}.out
  n=$(grep -c "^\[epoch" "$log" 2>/dev/null)
  best=$(grep "^\[epoch" "$log" 2>/dev/null | awk -F'mAP50=' '{split($2,a," ");print a[1], $0}' | sort -g | tail -1 | cut -d' ' -f2-)
  testln=$(awk '/native val on .test/{f=1} f' "$log" 2>/dev/null | grep -E "^\s+all " | head -1)
  summary+="\n== ${m} @1280 ==\n  epochs run: ${n}\n  best val epoch: ${best:-n/a}\n  TEST (imgs boxes P R mAP50 mAP50-95):${testln:-  eval pending}\n"
done
summary+="\n640px reference (same test split): yolov10m 0.498, yolov9m 0.498 mAP50;\non human-keyframe GT: 0.670 / 0.640.\nRun dirs: $RUNS/<model>_1280_v3pooled"
notify "[DeltaAI] Training ENDED: 1280px ablation results (job $SLURM_JOB_ID)" "$summary"

echo "=== done $SLURM_JOB_ID  $(date) ==="
