#!/bin/bash
#SBATCH --job-name=asym_xcheck
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA40x4-preempt,gpuA100x4,gpuA100x4-preempt
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=48G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/asym_xcheck_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/asym_xcheck_%j.err

# Is the asymmetric model's AP50 of 0.665 real, or an artefact of the evaluation software?
#
# It was scored on DeltaAI under Ultralytics 8.4.106; every number in the benchmark table
# was scored on Delta under 8.4.80. A jump from 0.460 to 0.665 is larger than the entire
# 0.365-0.510 spread across eleven architectures, which makes a measurement explanation at
# least as likely as a real one. This re-scores the identical checkpoint under 8.4.80.
#
#   same result  -> the gain survives, and the training difference is what to investigate
#   ~0.46-0.50   -> the gain is the evaluator, not the model
#
# It also re-scores the baseline in the same pass, so both sit under one software version.
set -u
cd /work/nvme/bgte/tislam6/wildlife_project
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
R=/work/hdd/bgte/tislam6/wildlife_outputs/runs
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

python -u src/detect/eval_full.py --weights "$R/yolo11m_640_asymscale/weights/best.pt" \
    --imgsz 640 --data data/dataset/yolo_v3/data.yaml \
    --tag yolo11m_640_asymscale_ON_DELTA --out results/detection_eval
python -u src/detect/eval_full.py --weights "$R/yolo11m_640_v3pooled/weights/best.pt" \
    --imgsz 640 --data data/dataset/yolo_v3/data.yaml \
    --tag yolo11m_640_baseline_RECHECK --out results/detection_eval
echo "=== done $(date) ==="
