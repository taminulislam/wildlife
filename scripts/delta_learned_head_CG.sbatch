#!/bin/bash
#SBATCH --job-name=wild_headCG
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_headCG_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_headCG_%j.err

# THE experiment the whole project now points at.
#
# Established (§6.7.1, §6.9): six detector/tracker configurations spanning 7k-90k
# candidates and 74-82 of 83 unseen deer reached all count within 3.6 points of each
# other, 62.7-66.3%. The detector does not matter and the tracker does not matter. The
# binding constraint is the 3-parameter confirmation rule.
#
# Decomposing the 83 unseen deer under that rule:
#     9  never detected at all
#    27  HAD a candidate and the rule rejected it   <- the target
#    47  counted
# 26 of those 27 failed on topk_conf >= 0.65. They are faint, far, small deer whose
# detector confidence never reaches 0.65, and a confidence threshold cannot separate them
# from noise. A learned head can, if trajectory shape, motion consistency and size
# persistence carry the signal that confidence does not.
#
# Two pools, because they answer different questions:
#   C (orphan, 7008 cands)  the pool every §6.5 learned-confirmer ran on, so the result is
#                           directly comparable to the prior negative. Ceiling 66/83.
#   G (ReID, 27679 cands)   the best pool in the project. Ceiling 73/83 -- seven more deer
#                           -- but 4x the noise, which is exactly what beat the rule.
#
# HELD-OUT protocol, not k-fold: train on the 19 detector-train videos, freeze, test on
# the 13 the detector never saw. That makes the head's MAE directly comparable to the
# frozen rule's 2.38 / 55-of-83 (§6.7.1). K-fold over all 32 mixes training videos into
# every test fold and inherits the §6.7 optimism.
#
# Baselines to beat, on the 13 unseen videos:
#     rule on pool C   MAE 2.38   55/83 = 66.3%
#     rule on pool G   MAE 2.69   52/83 = 62.7%
set -eo pipefail
# envs/wildlife is a CONDA env with no bin/activate; sourcing that path fails instantly
# under set -e. Use the same incantation as the mmdet jobs, which work on this cluster.
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project

OUT=/work/hdd/bgte/tislam6/wildlife_outputs/counts
LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
echo "=== job $SLURM_JOB_ID  $(date) ==="
python -c "import torch,sklearn;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# ---- pool G has no label file yet (C's was built in §6.5) ----
if [ ! -s data/temporal/tracks_labelled_reid.csv ]; then
  echo "--- labelling pool G ---"
  python -u src/temporal/label_tracks.py \
      --counts-dir "$OUT/phaseG_reid_conf0.02" --cvat-dir data/cvat_export \
      --source data/raw --out data/temporal/tracks_labelled_reid.csv
fi

run () {   # $1 = tag, $2 = counts dir, $3 = labels csv
  echo ""; echo "################ POOL $1 ################"
  python -u src/temporal/train_cv.py \
      --counts-dir "$2" --labels "$3" --heldout \
      --out "results/temporal/head_heldout_$1" \
      2>&1 | tee "$LOGS/wild_headCG_${SLURM_JOB_ID}_$1.out"
}

run C "$OUT/phaseC_orphan_yolo11m_conf0.10" data/temporal/tracks_labelled_orphan.csv
run G "$OUT/phaseG_reid_conf0.02"           data/temporal/tracks_labelled_reid.csv

printf '%b\n' "Learned confirmation head, HELD-OUT on the 13 unseen videos, $(date) (job $SLURM_JOB_ID).\n\nBaselines: rule on pool C = MAE 2.38, 55/83 counted. Rule on pool G = MAE 2.69, 52/83.\n\n=== POOL C (7008 cands, ceiling 66/83) ===\n$(tail -n 12 $LOGS/wild_headCG_${SLURM_JOB_ID}_C.out)\n\n=== POOL G (27679 cands, ceiling 73/83) ===\n$(tail -n 12 $LOGS/wild_headCG_${SLURM_JOB_ID}_G.out)\n\nIf the head beats MAE 2.38 the paper has a method contribution. If not, the diagnostic result stands and the question is closed." \
  | mail -s "[Delta] Learned head vs the rule, held out (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
