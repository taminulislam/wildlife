#!/bin/bash
#SBATCH --job-name=wild_1280
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_1280_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_1280_%j.err

# Resolution ablation: the two 640-roster winners (yolov10m, yolov9m) retrained at
# 1280px. 71% of deer are COCO-small at native res -> resolution is the biggest
# accuracy lever. Custom BEGIN/END emails with model list + results summary.
set -e
mkdir -p /work/hdd/bgte/tislam6/wildlife_outputs/logs

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife

cd /work/nvme/bgte/tislam6/wildlife_project

EMAIL=islamtaminul@gmail.com
notify() { printf '%b\n' "$2" | mail -s "$1" "$EMAIL" 2>/dev/null || true; }

DATA=data/dataset/yolo_v3/data.yaml
EPOCHS=${EPOCHS:-70}
RUNS=/work/hdd/bgte/tislam6/wildlife_outputs/runs
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
MODELS=(yolov10m yolov9m)

notify "[Delta] Training STARTED: 1280px resolution ablation (job $SLURM_JOB_ID)" \
"Job $SLURM_JOB_ID started on $SLURM_NODELIST at $(date).\n
Now training (one per GPU, in parallel):\n
  1. YOLOv10m @ 1280px  (640-roster winner: mAP50 0.498, mAP50-95 0.197)\n
  2. YOLOv9m  @ 1280px  (640-roster co-winner: mAP50 0.498)\n
Config: ${EPOCHS} epochs, batch 16, SGD lr0=0.01, patience 50, pooled all-site split (yolo_v3).\n
Purpose: 71% of deer are COCO-small at 640 -> higher input resolution is the biggest accuracy lever.\n
You will get another email with the results summary when training ends."

pids=()
for i in 0 1; do
  m=${MODELS[$i]}
  name=${m}_1280_v3pooled
  log=$LOGS/wild_1280_${m}_${SLURM_JOB_ID}.out
  echo "[gpu $i] START $m @1280 -> $name (log: $log)"
  python -u src/detect/train.py \
      --model weights/${m}.pt --arch yolo --data "$DATA" \
      --imgsz 1280 --epochs "$EPOCHS" --batch 16 --patience 50 \
      --optimizer SGD --lr0 0.01 --device "$i" \
      --save-period 10 --resume \
      --project "$RUNS" --name "$name" > "$log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

# ---- results summary email ----
summary="Job $SLURM_JOB_ID (1280px ablation) finished at $(date). Status: $([ $fail = 0 ] && echo OK || echo 'SOME FAILURES').\n"
for m in "${MODELS[@]}"; do
  log=$LOGS/wild_1280_${m}_${SLURM_JOB_ID}.out
  bestln=$(grep "^\[epoch" "$log" 2>/dev/null | awk -F'mAP50=' '{split($2,a," "); print a[1], $0}' | sort -g | tail -1 | cut -d' ' -f2-)
  testln=$(awk '/native val on .test/{f=1} f' "$log" 2>/dev/null | grep -E "^\s+all " | head -1)
  n=$(grep -c "^\[epoch" "$log" 2>/dev/null)
  summary+="\n== ${m} @1280 ==\n  epochs run: ${n}\n  best val epoch line: ${bestln:-n/a}\n  TEST (images boxes P R mAP50 mAP50-95):${testln:-  eval pending}\n"
done
summary+="\n640px references: yolov10m mAP50 0.498 / yolov9m 0.498.\nRun dirs: $RUNS/<model>_1280_v3pooled"
notify "[Delta] Training ENDED: 1280px ablation results (job $SLURM_JOB_ID)" "$summary"

echo "=== done $SLURM_JOB_ID $(date) ==="
