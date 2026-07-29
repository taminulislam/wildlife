#!/bin/bash
#SBATCH --job-name=gh_countmax
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=64 --mem=400G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_countmax_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_countmax_%j.err

# PHASE E: lift the CANDIDATE CEILING by removing the track-initialisation gate.
#
# Established by the conf-0.05 run: the detector already sees 232/235 deer, but the
# candidate pool contains only 207 — and halving the detector threshold (0.10 -> 0.05)
# added 3800 candidate tracks while recovering ZERO extra deer. The loss is therefore
# inside BoT-SORT, not the detector: `new_track_thresh: 0.15` refuses to START a track
# from any detection below 0.15, so faint deer can never become candidates.
#
# botsort_deer_maxrecall.yaml drops that gate to 0.05 (detector conf 0.02). Expect a
# much larger, noisier pool; the point is the ceiling (primaries / 235), not the rule's
# MAE on this pool.
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project

W=${W:-/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt}
CONF=${CONF:-0.02}     # detector floor; the tracker no longer gates initialisation
OUT=/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseE_maxrecall_conf${CONF}
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
mkdir -p "$OUT"

echo "=== job $SLURM_JOB_ID  weights=$W  conf=$CONF  $(date) ==="
printf '%b\n' "Max-recall counting run started $(date) on DeltaAI (job $SLURM_JOB_ID).\n\nDetector: YOLO11m @640, conf $CONF.\nTracker: BoT-SORT max-recall variant - new_track_thresh dropped 0.15 -> 0.05.\nAll 32 videos, sharded over 4 GH200s.\n\nWHY: the detector already sees 232 of 235 deer, but only 207 reach the candidate pool, and lowering the detector threshold recovered none of the missing ones. The gate is track INITIALISATION inside BoT-SORT. This run removes it.\n\nThe number to watch in the results email is the CEILING (how many of the 235 deer have a candidate track at all), not MAE.\n\nA results email follows." \
  | mail -s "[DeltaAI] Max-recall counting STARTED (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true

pids=()
for i in 0 1 2 3; do
  python -u src/track/count_deer.py \
      --weights "$W" --source data/raw --cvat-dir data/cvat_export \
      --shards 4 --shard $i --device $i \
      --imgsz 640 --conf "$CONF" --contrast clahe --dump-min-frames 1 --keep-orphans --tracker src/track/botsort_deer_maxrecall.yaml \
      --out "$OUT/shard$i" > "$LOGS/gh_countmax_${SLURM_JOB_ID}_shard$i.out" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "shards done (fail=$fail)"

# count_eval.py globs counts_*.csv in ONE dir
for i in 0 1 2 3; do
  [ -f "$OUT/shard$i/counts.csv" ] && cp "$OUT/shard$i/counts.csv" "$OUT/counts_shard$i.csv"
done
python -u src/eval/count_eval.py --counts "$OUT" --gt data/annotate_v2/count_gt.csv

# The point of this run is the CEILING, so measure it here instead of in a follow-up:
# merge the shards and label every candidate against the CVAT ground truth. The
# "primary" count is how many of the 235 deer are reachable at all.
python -u src/track/merge_shards.py --run "$OUT"
python -u src/temporal/label_tracks.py \
    --counts-dir "$OUT" --cvat-dir data/cvat_export --source data/raw \
    --out data/temporal/tracks_labelled_maxrecall.csv > "$LOGS/gh_countmax_${SLURM_JOB_ID}_label.out" 2>&1 || true
CEIL=$(grep -E "CEILING|primary|candidate tracks" "$LOGS/gh_countmax_${SLURM_JOB_ID}_label.out" 2>/dev/null)

MD=$OUT/eval/count_eval_summary.md
printf '%b\n' "Max-recall counting FINISHED $(date) (job $SLURM_JOB_ID, fail=$fail).\n\n=== CANDIDATE CEILING (the number this run was for) ===\n$CEIL\n\nPrevious pools both capped at 207/235 primaries.\n\n=== rule-based MAE on this pool ===\n$(cat $MD 2>/dev/null)\n\nFull outputs: $OUT" \
  | mail -s "[DeltaAI] Max-recall counting: candidate ceiling (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
