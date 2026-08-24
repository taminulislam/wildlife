#!/bin/bash
#SBATCH --job-name=determinism
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/determ_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/determ_%j.err

# Is the counting pipeline deterministic? Run the production configuration THREE times
# on the same GPU, same weights, same videos, same everything, into three output dirs.
#
# Controlled on purpose: same device, same process settings, back to back. If the counts
# differ across these three, the pipeline is nondeterministic on identical hardware. If
# they agree, then any disagreement we have seen between clusters is a hardware effect,
# not run-to-run noise. Those are different problems with different fixes, and the
# existing evidence cannot tell them apart.
#
# Four held-out videos carrying 55 of the 83 held-out animals.
set -e
cd /work/nvme/bgte/tislam6/wildlife_project
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python

W=/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt
BASE=/work/hdd/bgte/tislam6/wildlife_outputs/counts/determinism
SRC=$BASE/videos
rm -rf "$SRC"; mkdir -p "$SRC"
# Explicit link names. "ln -sf src dir/" fails when dir already exists, which is exactly
# the case on a requeue or a second attempt.
for f in "data/raw/SHB_Videos_LS/visit1/NShelbyRd(blue)_SHB_12.11.2025_LS.mp4" \
         "data/raw/SHB_Videos_LS/visit1/GolfDr_SHB_12.11.2025.mp4" \
         "data/raw/SHB_Videos_LS/visit1/NWolfCreek(orange)_SHB_12.11.2025_LS.mp4" \
         "data/raw/SHW_Videos_LS/visit1/SHW_01.18.2026/Robinson_SHW_01.18.2026_LS.mp4"; do
  ln -sfn "$PWD/$f" "$SRC/$(basename "$f")"
done
ls -l "$SRC" | tail -n +2

echo "=== job $SLURM_JOB_ID on $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader)  $(date) ==="

for R in 1 2 3; do
  echo "--- replicate $R  $(date) ---"
  "$PY" -u src/track/count_deer.py \
      --weights "$W" --source "$SRC" --cvat-dir data/cvat_export \
      --imgsz 640 --conf 0.10 --contrast clahe \
      --dump-min-frames 1 --keep-orphans \
      --tracker src/track/botsort_deer_recall.yaml \
      --device 0 --out "$BASE/rep$R"
done

echo "=== comparison ==="
"$PY" - "$BASE" <<'PYEOF'
import csv, glob, os, sys, itertools
base = sys.argv[1]
def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "summary*.csv")):
        for r in csv.DictReader(open(f)):
            out[r["video"]] = (int(r["count"]), int(r["candidate_tracks"]))
    return out
reps = {r: load(f"{base}/rep{r}") for r in (1, 2, 3)}
vids = sorted(set().union(*[set(v) for v in reps.values()]))
print(f"{'video':<44}" + "".join(f"{'rep'+str(r):>9}" for r in (1,2,3)) + "   same?")
nd = 0
for v in vids:
    cs = [reps[r].get(v, ('-',))[0] for r in (1,2,3)]
    same = len(set(cs)) == 1
    nd += 0 if same else 1
    print(f"{v[:44]:<44}" + "".join(f"{c:>9}" for c in cs) + ("   yes" if same else "   NO"))
tot = [sum(x[0] for x in reps[r].values()) for r in (1,2,3)]
print(f"\ntotals: {tot}")
print("VERDICT:", "deterministic on identical hardware" if nd == 0
      else f"NONDETERMINISTIC -- {nd} of {len(vids)} videos disagree across replicates")
PYEOF
echo "=== done $(date) ==="
