#!/bin/bash
#SBATCH --job-name=gh_reidfeas
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=64 --mem=400G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_reidfeas_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_reidfeas_%j.err

# Does a thermal ReID stage have anything to learn from on this corpus?
#
# Asked for: a CNN/transformer ReID keyed on body shape, antler structure, thermal
# silhouette, relative size and movement. Relative size and movement are already features
# of the temporal head, so the open question is whether APPEARANCE carries identity at a
# median 29x24 px.
#
# This measures it instead of arguing it. CVAT tracks ARE identity labels, so we can run
# textbook closed-set ReID: split each deer's track temporally, gallery = first half,
# query = second half, report rank-1 against chance. Three embeddings, so the result says
# WHY: geometry-only (what we already use), raw pixels (silhouette, no learning), and
# ImageNet ResNet-50 (the standard ReID backbone).
#
# Decision rule fixed in advance:
#   cnn <= geom            -> appearance adds nothing; report as a measured negative
#   cnn > geom, near chance-> not viable at this resolution
#   cnn >> geom and >>chance -> worth building; train a thermal-specific embedding next
#
# Within-video is the CEILING (same range, lighting, calibration in both halves). No deer
# in this corpus appears in two videos, so cross-video ReID cannot be scored at all here.
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project

OUT=results/reid
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
mkdir -p "$OUT"
echo "=== job $SLURM_JOB_ID  $(date) ==="

pids=()
for i in 0 1 2 3; do
  python -u src/reid/reid_feasibility.py \
      --source data/raw --cvat-dir data/cvat_export \
      --shards 4 --shard $i --device $i \
      --out "$OUT/feasibility_shard$i.csv" \
      > "$LOGS/gh_reidfeas_${SLURM_JOB_ID}_shard$i.out" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "shards done (fail=$fail)"

python -u - <<'PY' | tee "$OUT/feasibility_SUMMARY.txt"
import csv, glob
rows = []
for f in sorted(glob.glob("results/reid/feasibility_shard*.csv")):
    rows += list(csv.DictReader(f=open(f)))
if not rows:
    raise SystemExit("no shard output")
tot = sum(int(r["identities"]) for r in rows)
print(f"videos scored: {len(rows)}   identities: {tot}\n")
print(f"{'embedding':<22}{'rank-1':>8}")
for tag in ("chance", "geom", "pixels", "cnn"):
    w = sum(float(r[tag]) * int(r["identities"]) for r in rows) / tot
    print(f"  {tag:<20}{w:>8.3f}")
g = sum(float(r["geom"]) * int(r["identities"]) for r in rows) / tot
c = sum(float(r["cnn"]) * int(r["identities"]) for r in rows) / tot
print(f"\nappearance gain over geometry: {c-g:+.3f}")
print("VERDICT:", "worth building" if c > g + 0.10 else
      "appearance adds little or nothing at this resolution")
PY

printf '%b\n' "Thermal ReID feasibility finished $(date) (job $SLURM_JOB_ID, fail=$fail).\n\n$(cat $OUT/feasibility_SUMMARY.txt)\n\nPer-video: $OUT/feasibility_shard*.csv" \
  | mail -s "[DeltaAI] Thermal ReID feasibility: does appearance carry deer identity? (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
