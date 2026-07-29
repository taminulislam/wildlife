#!/bin/bash
#SBATCH --job-name=wild_viz
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_viz_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_viz_%j.err
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project
R=/work/hdd/bgte/tislam6/wildlife_outputs/runs
VIZ=/work/hdd/bgte/tislam6/wildlife_outputs/viz
mkdir -p "$VIZ"

echo "=== 1/2 DETECTION gallery: best detector (YOLOv9m@1280) ==="
python -u src/viz/detect_gallery.py \
    --weights $R/yolov9m_1280_v3pooled/weights/best.pt --imgsz 1280 --conf 0.25 \
    --n-large 9 --n-medium 9 --n-small 6 --out "$VIZ/detect_yolov9m1280"

echo "=== 2/2 COUNTING evidence: one sheet per counted deer ==="
python -u src/track/export_evidence.py \
    --counts-dir /work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10 \
    --source data/raw --out "$VIZ/counting_evidence" --topk 6 || \
    echo "[warn] evidence export failed"

echo "=== outputs ==="; find "$VIZ" -name "*.jpg" | head -20; find "$VIZ" -name "*.jpg" | wc -l
printf '%b\n' "Visualisations ready $(date).\n\nDETECTION: $VIZ/detect_yolov9m1280/ — frames bucketed by deer size (large/medium/small), green box + confidence on every prediction, thin blue = human GT. Contact sheets: _sheet_large.jpg / _sheet_medium.jpg / _sheet_small.jpg\n\nCOUNTING: $VIZ/counting_evidence/ — per counted deer a contact sheet of the frames it was counted from (trk<ID>_sheet.jpg), the single best full frame (trk<ID>_best.jpg), and a per-site gallery (_gallery.jpg) with one crop per deer." \
  | mail -s "[Delta] Detection + counting output images ready (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
