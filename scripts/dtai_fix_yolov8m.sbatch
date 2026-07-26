#!/bin/bash
#SBATCH --job-name=gh_v8fix
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_v8fix_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_v8fix_%j.err
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project
W=/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolov8m_640_v3pooled/weights

# yolov8m's best.pt is an epoch-1 artefact (warmup fitness spike fooled selection),
# which makes its row in the paper table meaningless. Pick the real best checkpoint
# on the VAL split (never test - that would be test-set selection), then re-score.
python - <<'PY'
import glob, os
from ultralytics import YOLO
W="/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolov8m_640_v3pooled/weights"
cands=sorted(glob.glob(f"{W}/epoch*.pt"))+[f"{W}/last.pt"]
best=(None,-1)
for c in cands:
    if not os.path.isfile(c): continue
    r=YOLO(c).val(data="data/dataset/yolo_v3/data.yaml", split="val", imgsz=640,
                  device=0, verbose=False)
    m=float(r.box.map50); print(f"VAL {os.path.basename(c)}: mAP50={m:.4f}", flush=True)
    if m>best[1]: best=(c,m)
print(f"SELECTED {best[0]} (val mAP50={best[1]:.4f})")
open("/tmp/v8_best.txt","w").write(best[0])
PY
B=$(cat /tmp/v8_best.txt)
echo "=== re-scoring yolov8m with $B ==="
python -u src/detect/eval_full.py --weights "$B" --data data/dataset/yolo_v3/data.yaml \
    --split test --imgsz 640 --arch yolo --tag yolov8m_640_v3pooled_FIXED || true
for cf in 0.25 0.10; do
  python -u src/eval/counting_detection_eval.py --weights "$B" \
      --data data/dataset/yolo_v3/data.yaml --split test --conf $cf --tag yolov8mfix_640_FULLGT
  python -u src/eval/counting_detection_eval.py --weights "$B" \
      --data data/dataset/yolo_v3_kf/data.yaml --split test --conf $cf --tag yolov8mfix_640_KEYFRAME
done
python -u src/eval/summarize_counting_eval.py
printf '%b\n' "yolov8m checkpoint fix done $(date).\nIts best.pt was an epoch-1 artefact; the real best checkpoint was selected on VAL and re-scored.\n\n$(grep -E 'VAL |SELECTED|^  all ' /work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_v8fix_${SLURM_JOB_ID}.out 2>/dev/null | head -20)" \
  | mail -s "[DeltaAI] yolov8m checkpoint fix (job $SLURM_JOB_ID)" islamtaminul@gmail.com 2>/dev/null || true
