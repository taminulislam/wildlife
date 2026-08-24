#!/bin/bash
#SBATCH --job-name=det_gallery
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/detgal_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/detgal_%j.err

# Detection-output gallery for the PRODUCTION detector. The only gallery on disk was
# built from yolov9m@1280, which is not the detector the paper adopts -- showing it on a
# slide that says "YOLO11m @ 640" would be a mismatch a careful viewer would catch.
set -e
cd /work/nvme/bgte/tislam6/wildlife_project
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python

"$PY" -u src/viz/detect_gallery.py \
    --weights /work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt \
    --arch yolo --imgsz 640 --conf 0.25 --split test --device 0 \
    --n-large 6 --n-medium 6 --n-small 4 \
    --out /work/hdd/bgte/tislam6/wildlife_outputs/viz/detect_yolo11m_640
echo "=== done $(date) ==="
