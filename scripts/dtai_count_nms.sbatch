#!/bin/bash
#SBATCH --job-name=gh_countnms
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=64 --mem=400G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_countnms_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_countnms_%j.err

# PHASE F: recover the COLLIDED deer by loosening NMS.
#
# State of the pool after Phase E: 228 of 235 deer have a touching candidate, but only
# ~211 have their OWN candidate. The other 16 share a candidate with a second deer, and
# on the unseen videos that is 10 of the 14 misses.
#
# The obvious remedies were tested and rejected:
#   * a learned count-per-track head — the corpus has 16 examples of "count = 2" against
#     18 137 of "count = 0". Two positives per CV fold; not learnable.
#   * temporal track splitting at teleports (src/temporal/split_tracks.py) — recovered
#     +3 deer for 4 000 extra candidates.
#
# Because the collisions are not temporal. Measured directly: 15 of the 16 are
# SIMULTANEOUS — one predicted box overlapping two GT deer in the SAME frames, for up to
# 116 frames — and only 1 is a track drifting from one animal to another. Half of them
# sit in a single dense-group video, NShelbyRd(blue)_SHB.
#
# One box on two deer at once is an NMS artefact: adjacent 27 px deer produce boxes that
# overlap above the 0.5 IoU threshold, so the second is suppressed and never becomes a
# detection at all. This run raises NMS IoU to 0.90 — everything else identical to Phase
# E — so both animals survive into the tracker.
#
# Watch `primary` in the results email, not `reached`: reached is already 228; the
# question is whether the collided deer now get candidates of their own.
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project

W=${W:-/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt}
CONF=${CONF:-0.02}
OUT=/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseF_nms_conf${CONF}
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
mkdir -p "$OUT"

echo "=== job $SLURM_JOB_ID  weights=$W  conf=$CONF  $(date) ==="
printf '%b\n' "Loose-NMS counting run started $(date) on DeltaAI (job $SLURM_JOB_ID).\n\nDetector: YOLO11m @640, conf $CONF, NMS IoU 0.90 (was 0.50).\nTracker: BoT-SORT max-recall variant.\nAll 32 videos, sharded over 4 GH200s.\n\nWHY: 16 deer share a candidate track with another deer, and 15 of those 16 are one box sitting on TWO deer in the same frames - not a track drifting between animals. That is NMS suppressing the second deer of an adjacent pair. A learned count-per-track head cannot fix it (only 16 training examples exist) and temporal splitting recovered just 3 deer, so this attacks the detection stage instead.\n\nThe number to watch is PRIMARY (deer with their own candidate), currently ~211 of 235.\n\nA results email follows." \
  | mail -s "[DeltaAI] Loose-NMS counting STARTED (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true

pids=()
for i in 0 1 2 3; do
  python -u src/track/count_deer.py \
      --weights "$W" --source data/raw --cvat-dir data/cvat_export \
      --shards 4 --shard $i --device $i \
      --imgsz 640 --conf "$CONF" --iou 0.90 --contrast clahe --dump-min-frames 1 --keep-orphans --tracker src/track/botsort_deer_maxrecall.yaml \
      --out "$OUT/shard$i" > "$LOGS/gh_countnms_${SLURM_JOB_ID}_shard$i.out" 2>&1 &
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
    --out data/temporal/tracks_labelled_nms.csv > "$LOGS/gh_countnms_${SLURM_JOB_ID}_label.out" 2>&1 || true
# reached-vs-primary is the actual verdict for this run
python -u src/eval/pool_coverage.py --counts-dir "$OUT" \
    --out results/counting_eval/pool_coverage_nms.csv \
    >> "$LOGS/gh_countnms_${SLURM_JOB_ID}_label.out" 2>&1 || true
CEIL=$(grep -E "CEILING|primary|candidate tracks" "$LOGS/gh_countnms_${SLURM_JOB_ID}_label.out" 2>/dev/null)

MD=$OUT/eval/count_eval_summary.md
printf '%b\n' "Loose-NMS counting FINISHED $(date) (job $SLURM_JOB_ID, fail=$fail).\n\n=== CANDIDATE CEILING (the number this run was for) ===\n$CEIL\n\nPhase E (NMS 0.50) reached 228/235 but only ~211 primaries. If loosening NMS worked, primary rises toward reached.\n\n=== rule-based MAE on this pool ===\n$(cat $MD 2>/dev/null)\n\nFull outputs: $OUT" \
  | mail -s "[DeltaAI] Loose-NMS counting: did the collided deer separate? (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
