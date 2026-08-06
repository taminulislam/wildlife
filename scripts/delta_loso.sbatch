#!/bin/bash
#SBATCH --job-name=wild_loso
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=4 --mem=200G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_loso_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_loso_%j.err

# LEAVE-ONE-SITE-OUT: the generalisation evidence WACV asks for.
#
# Every number in the paper so far comes from one pooled split in which all four sites
# appear in training. That measures "does it work on unseen VIDEO", not "does it work at
# an unseen PLACE" -- and a survey system is deployed at new places. LOSO answers the
# second question with labels we already have, which matters because the right-side (RS)
# corpus is unannotated and no further labelling is being done.
#
# Four folds. Each trains YOLO11m from COCO on three sites and tests on the fourth, then
# runs the full counting pipeline on the held-out site's videos only. The four sites are
# genuinely different domains: SHB holds 132 deer and MAS holds 15, terrain and thermal
# conditions differ, so this is a real shift rather than a token split.
#
# Compare against the pooled result (MAE 2.38, 66.3% counted on unseen video). A large
# drop is itself publishable -- it bounds how far the system transfers.
#
# Tuned for turnaround, not for the last point of mAP: 80 epochs, patience 20, batch 48.
# §3 shows most models peak by epoch 22 (YOLO11m at 60), so 80 with early stopping is
# ample, and the four folds run concurrently on four GPUs -- wall clock is one training,
# not four. Expect ~2-3 h rather than the ~12 h a full-schedule sweep would take.
set -eo pipefail
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project

OUT=/work/hdd/bgte/tislam6/wildlife_outputs
LOGS=$OUT/logs
SITES="SHB TON SHW MAS"
echo "=== job $SLURM_JOB_ID  LOSO over $SITES  $(date) ==="

# ---------- phase 1: build one dataset per fold (CPU, fast) ----------
for S in $SITES; do
  D=data/dataset/loso_$S
  if [ ! -s "$D/data.yaml" ]; then
    echo "--- building $D (test site = $S) ---"
    python -u src/dataset/build_yolo_dataset.py \
        --images data/annotate_v2/frames --labels data/annotate_v2/labels \
        --out "$D" --test-sites "$S" --val-frac 0.15
  fi
done

# ---------- phase 2: train four detectors, one per GPU ----------
pids=()
i=0
for S in $SITES; do
  CUDA_VISIBLE_DEVICES=$i python -u src/detect/train.py \
      --model weights/yolo11m.pt --arch yolo \
      --data data/dataset/loso_$S/data.yaml \
      --imgsz 640 --epochs 80 --batch 48 --patience 20 \
      --optimizer SGD --lr0 0.01 \
      --project "$OUT/runs" --name "loso_${S}_yolo11m" \
      > "$LOGS/wild_loso_${SLURM_JOB_ID}_train_$S.out" 2>&1 &
  pids+=($!); i=$((i+1))
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "training done (fail=$fail)"

# ---------- phase 3: count + evaluate on the held-out site only ----------
pids=(); i=0
for S in $SITES; do
  (
    W=$(ls -1 "$OUT/runs/loso_${S}_yolo11m/weights/best.pt" 2>/dev/null | head -1)
    [ -z "$W" ] && { echo "[$S] no weights"; exit 0; }
    SRC=$(ls -d data/raw/${S}_[Vv]ideos_LS 2>/dev/null | head -1)
    [ -z "$SRC" ] && { echo "[$S] no video dir"; exit 0; }
    CO="$OUT/counts/loso_$S"
    python -u src/track/count_deer.py \
        --weights "$W" --source "$SRC" --cvat-dir data/cvat_export \
        --device $i --imgsz 640 --conf 0.10 --iou 0.50 --contrast clahe \
        --dump-min-frames 1 --keep-orphans \
        --tracker src/track/botsort_deer_recall.yaml --out "$CO/shard0"
    cp "$CO/shard0/counts.csv" "$CO/counts_shard0.csv" 2>/dev/null || true
    python -u src/eval/count_eval.py --counts "$CO" --gt data/annotate_v2/count_gt.csv
    python -u src/eval/pool_coverage.py --counts-dir "$CO" \
        --out results/counting_eval/loso_coverage_$S.csv
  ) > "$LOGS/wild_loso_${SLURM_JOB_ID}_count_$S.out" 2>&1 &
  pids+=($!); i=$((i+1))
done
for p in "${pids[@]}"; do wait "$p" || true; done

echo ""; echo "================ LOSO SUMMARY ================"
for S in $SITES; do
  echo "--- held-out site: $S"
  grep -E "^\| \*\*ALL\*\*|reached |primary " "$LOGS/wild_loso_${SLURM_JOB_ID}_count_$S.out" 2>/dev/null | head -4
done | tee "$OUT/counts/loso_SUMMARY.txt"

printf '%b\n' "Leave-one-site-out finished $(date) (job $SLURM_JOB_ID, train fail=$fail).\n\nTrains on three sites, tests on the fourth, all four rotations. This is the cross-domain generalisation result -- the pooled split in the paper has every site in training, so it measures unseen VIDEO, not an unseen PLACE.\n\nCompare against pooled: MAE 2.38, 66.3% counted.\n\n$(cat $OUT/counts/loso_SUMMARY.txt 2>/dev/null)" \
  | mail -s "[Delta] Leave-one-site-out: does it transfer to a new site? (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
