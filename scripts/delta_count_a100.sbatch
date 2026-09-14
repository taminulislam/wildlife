#!/bin/bash
#SBATCH --job-name=a100_count4
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=64 --mem=200G
#SBATCH --time=03:00:00
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/a100_count4_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/a100_count4_%j.err

# Pool C, all 32 videos, on A100 with the CUDA-12.1 x86 environment. Same script, same
# weights, same arguments as scripts/dtai_count_orphan.sbatch, which produced the paper's
# numbers on GH200. The published statement is that counts were compared only on identical
# hardware; the determinism replicate on four videos differed from the GH200 run by one
# animal on three of them, so this measures the platform effect on the headline itself.
set -e
cd /work/nvme/bgte/tislam6/wildlife_project
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python
W=/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt
CONF=0.10
OUT=/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf${CONF}_A100
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv | tee "$OUT/gpu.txt"
$PY -c "import torch,ultralytics,cv2;print('torch',torch.__version__,'ultralytics',ultralytics.__version__,'opencv',cv2.__version__)" | tee "$OUT/env.txt"

pids=()
for i in 0 1 2 3; do
  $PY -u src/track/count_deer.py \
      --weights "$W" --source data/raw --cvat-dir data/cvat_export \
      --shards 4 --shard $i --device $i \
      --imgsz 640 --conf "$CONF" --contrast clahe --dump-min-frames 1 --keep-orphans --tracker src/track/botsort_deer_recall.yaml \
      --out "$OUT/shard$i" > "$LOGS/a100_count4_${SLURM_JOB_ID}_shard$i.out" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "shards done (fail=$fail)"
for i in 0 1 2 3; do
  [ -f "$OUT/shard$i/counts.csv" ] && cp "$OUT/shard$i/counts.csv" "$OUT/counts_shard$i.csv"
done
$PY -u src/eval/count_eval.py --counts "$OUT" --gt data/annotate_v2/count_gt.csv
echo "=== done $(date) ==="
