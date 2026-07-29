#!/bin/bash
#SBATCH --job-name=wild_evid
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_evid_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/wild_evid_%j.err
set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bgte/tislam6/envs/wildlife
cd /work/nvme/bgte/tislam6/wildlife_project
VIZ=/work/hdd/bgte/tislam6/wildlife_outputs/viz
python -u src/track/export_evidence.py \
    --counts-dir /work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10/merged \
    --source data/raw --out "$VIZ/counting_evidence" --topk 6
echo "=== images ==="; find "$VIZ/counting_evidence" -name "*.jpg" | wc -l
printf '%b\n' "Counting evidence images ready $(date).\n\n$VIZ/counting_evidence/\n  <SITE>/<video>/trk<ID>_sheet.jpg  contact sheet of the frames each counted deer was seen in\n  <SITE>/<video>/trk<ID>_best.jpg   its clearest full frame with the box drawn\n  <SITE>/_gallery.jpg               one crop per counted deer in the site\n  evidence_index.csv                every image -> video, track, frame, time, confidence\n\nThese are the 210 deer counted by the best configuration (YOLO11m@640 + orphan recovery + tuned rule), i.e. the 210/236 = 89% result." \
  | mail -s "[Delta] Counting evidence images ready (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
