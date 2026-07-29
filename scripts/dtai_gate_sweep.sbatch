#!/bin/bash
#SBATCH --job-name=gh_gate
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=4 --mem=200G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_gate_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_gate_%j.err

# HOW CLOSE TO 236 CAN WE POSSIBLY GET? The counting pool now recovers 207/236 deer.
# Before spending GPU on pipeline changes, measure the hard limit: of the 236 GT deer,
# how many does each detector SEE at all, at progressively lower confidence?
# Anything above this number is unreachable without a better/ensembled detector.
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project
R=/work/hdd/bgte/tislam6/wildlife_outputs/runs

run() {  # gpu weights imgsz conf tag
  CUDA_VISIBLE_DEVICES=$1 python -u src/eval/track_recall.py \
     --weights "$2" --imgsz "$3" --conf "$4" --device 0 \
     --out "results/track_recall/$5" > \
     /work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_gate_${SLURM_JOB_ID}_$5.out 2>&1
}

run 0 $R/yolo11m_640_v3pooled/weights/best.pt   640  0.05 y11m640_c05 &
run 1 $R/yolo11m_640_v3pooled/weights/best.pt   640  0.02 y11m640_c02 &
run 2 $R/yolov9m_1280_v3pooled/weights/best.pt 1280  0.05 y9m1280_c05 &
run 3 $R/yolov9m_1280_v3pooled/weights/best.pt 1280  0.02 y9m1280_c02 &
wait

echo "=== DETECTABLE CEILING (of 235 GT deer) ==="
for t in y11m640_c05 y11m640_c02 y9m1280_c05 y9m1280_c02; do
  echo "-- $t"
  grep -A6 "TRACK-LEVEL RECALL" /work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_gate_${SLURM_JOB_ID}_$t.out 2>/dev/null | grep -E "ANY overlap|found"
done
LOG=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_gate_${SLURM_JOB_ID}.out
printf '%b\n' "Detectability sweep finished $(date).\n\nCounting currently recovers 207/236 deer (ceiling MAE 0.91). This measures the HARD limit: how many deer each detector sees at all, at lower confidence.\n\n$(sed -n '/DETECTABLE CEILING/,$p' "$LOG" 2>/dev/null)" \
  | mail -s "[DeltaAI] How close to 236 can we get? (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
