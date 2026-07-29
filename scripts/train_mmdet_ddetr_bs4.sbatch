#!/bin/bash
#SBATCH --job-name=wild_ddetr
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_ddetr_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_ddetr_%j.err

# Deformable DETR rerun at batch 4. The roster run (job 20565690) died at epoch 5 with
# CUDA OOM inside multi_scale_deform_attn on a 40 GB A100 — steady state was ~29 GB but
# the attention peak overran it. Fresh work-dir (not --resume) because the batch size and
# lr both changed; the dead run was only 5 epochs in at mAP50 0.075, nothing worth
# keeping.
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/mmdet
cd /work/nvme/bgte/tislam6/wildlife_project

LOGS=/work/hdd/bgte/tislam6/wildlife_outputs/logs
WD=/work/hdd/bgte/tislam6/wildlife_outputs/mmdet_runs/deformable-detr_r50_v3pooled_bs4
CFG=configs/mmdet/deer_deformable-detr_r50.py
TRAIN_PY=$(python -c "import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),'.mim','tools','train.py'))")
TEST_PY=$(python -c "import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),'.mim','tools','test.py'))")

echo "=== job $SLURM_JOB_ID on $SLURM_NODELIST  $(date) ==="
python -u "$TRAIN_PY" "$CFG" --work-dir "$WD" --cfg-options randomness.seed=0

BEST=$(ls -t "$WD"/best_coco_bbox_mAP_50_epoch_*.pth 2>/dev/null | head -1)
[ -z "$BEST" ] && BEST=$(ls -t "$WD"/epoch_*.pth 2>/dev/null | head -1)
[ -n "$BEST" ] && python -u "$TEST_PY" "$CFG" "$BEST" --work-dir "$WD/test_eval" || true

printf '%b\n' "Deformable DETR (batch 4) finished $(date), job $SLURM_JOB_ID.\n\nBest checkpoint: $BEST\n\nTEST split:\n$(grep 'bbox_mAP_copypaste' $LOGS/wild_ddetr_${SLURM_JOB_ID}.out 2>/dev/null | tail -1)\n\nWork dir: $WD" \
  | mail -s "[Delta] Deformable DETR rerun done (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
echo "=== done $(date) ==="
