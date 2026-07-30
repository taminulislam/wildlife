#!/bin/bash
#SBATCH --job-name=gh_thermreid
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=64 --mem=400G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_thermreid_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_thermreid_%j.err

# JOB B of two: train a thermal-specific ReID embedding on individual deer.
#
# Step 1 caches every GT deer crop keyed by individual (CVAT track = animal), sharded over
# 4 GPUs because decoding 521 930 frames is the expensive part and the trainer needs many
# passes. Step 2 trains a small metric-learning encoder and scores it.
#
# Cues, as requested:
#   body shape + thermal silhouette -> the 64x64 crop the CNN sees
#   relative size                   -> native box w/h as two extra input channels, so the
#                                      fixed-size resize cannot erase scale
#   movement pattern                -> already dx/dy/dt in the temporal head; an appearance
#                                      encoder is the wrong place for it
#   antler structure                -> not resolvable at ~19 px per metre of animal, and
#                                      the corpus spans Dec-Jan shedding. A limitation.
#
# Evaluation is open-set: deer appear in exactly one video each, so the video split is
# also an identity split -- train on the ~152 deer in detector-train videos, score on deer
# never seen, in videos never seen. Rank-1 uses the same temporal gallery/query protocol as
# job 2788117, so the trained model, ImageNet ResNet-50 and geometry-only are one table.
#
# Exports ONNX so the encoder drops into BoT-SORT: set `model:` in the tracker yaml to the
# exported path instead of `auto`.
# pipefail matters below: step 2 pipes python into tee, and without it a training crash
# would be masked by tee's exit 0 and mailed out as if it were a result.
set -eo pipefail
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project

CROPS=/work/hdd/bgte/tislam6/wildlife_outputs/reid_crops
OUT=results/reid/thermal
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
mkdir -p "$CROPS" "$OUT"
echo "=== job $SLURM_JOB_ID  $(date) ==="

echo "--- step 1: cache GT deer crops (identity-labelled) ---"
pids=()
for i in 0 1 2 3; do
  python -u src/reid/build_crops.py \
      --source data/raw --cvat-dir data/cvat_export --contrast clahe \
      --out "$CROPS" --shards 4 --shard $i \
      > "$LOGS/gh_thermreid_${SLURM_JOB_ID}_crops$i.out" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
NPZ=$(ls -1 "$CROPS"/*.npz 2>/dev/null | wc -l)
echo "crop cache: $NPZ videos (fail=$fail)"
if [ "$NPZ" -lt 10 ]; then
  printf '%b\n' "Thermal ReID job $SLURM_JOB_ID ABORTED: only $NPZ videos cached.\n\n$(tail -n 20 $LOGS/gh_thermreid_${SLURM_JOB_ID}_crops0.out)" \
    | mail -s "[DeltaAI] Thermal ReID FAILED at crop stage (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
  exit 1
fi

echo "--- step 2: train the embedding ---"
python -u src/reid/train_thermal_reid.py --crops "$CROPS" --out "$OUT" \
    --epochs 60 --p-ident 16 --k-inst 4 --iters 100 \
    2>&1 | tee "$LOGS/gh_thermreid_${SLURM_JOB_ID}_train.out"

printf '%b\n' "Thermal ReID training FINISHED $(date) (job $SLURM_JOB_ID).\n\nOpen-set: trained on deer from detector-train videos, scored on deer never seen in videos never seen. Same rank-1 protocol as the feasibility job, so these sit in one table with the ImageNet-ResNet50 and geometry-only baselines.\n\n$(tail -n 22 $LOGS/gh_thermreid_${SLURM_JOB_ID}_train.out)\n\nONNX encoder: $OUT/thermal_reid.onnx -- drops into BoT-SORT by setting model: to that path.\n\nCaveat: within-video ceiling. No deer in this corpus appears in two videos, so cross-session identification cannot be measured." \
  | mail -s "[DeltaAI] Thermal ReID trained: rank-1 on unseen deer (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
