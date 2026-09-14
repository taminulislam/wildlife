#!/bin/bash
#SBATCH --job-name=verify_engine
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=01:30:00
#SBATCH --mail-user=islamtaminul@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/verify_engine_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/verify_engine_%j.err
# Same GPU, same env, same video: does the app engine reproduce src/track/count_deer.py
# (the script behind the paper's pool-C numbers)? Then both against the paper's GH200 run.
set -e
cd /work/nvme/bgte/tislam6/wildlife_project
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python
W=/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolo11m_640_v3pooled/weights/best.pt
V="data/raw/TON_Videos_LS/visit1/OikosRd_TON_12.03.25_LS.mp4"
OUT=/u/tislam6/wildlife_verify_$SLURM_JOB_ID; mkdir -p "$OUT"
echo "################ count_deer.py on A100 (paper's script, paper's arguments)"
$PY -u src/track/count_deer.py --weights "$W" --source "$V" --imgsz 640 --conf 0.10 \
    --contrast clahe --dump-min-frames 1 --keep-orphans --tracker src/track/botsort_deer_recall.yaml \
    --out "$OUT/count_deer" --device 0
ls "$OUT/count_deer"
echo "################ engine vs count_deer (same GPU)"
$PY -u src/eval/verify_engine.py --video "$V" --paper-tracks "$OUT/count_deer/counts.csv" --label A100CD
echo "################ paper GH200 record for the same video"
$PY - "$V" <<'PYEOF'
import csv,sys,os
stem=os.path.splitext(os.path.basename(sys.argv[1]))[0]
rows=[r for r in csv.DictReader(open('/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10/merged/counts.csv')) if r['video']==stem]
ok=lambda r:int(r['n_frames'])>=20 and float(r['topk_conf'])>=0.65
print(f"GH200   tracks {len(rows)}  confirmed {sum(map(ok,rows))}")
for r in sorted(rows,key=lambda r:-float(r['topk_conf']))[:8]:
    print(f"  id {r['track_id']:>5} n {r['n_frames']:>4} span {float(r['span_s']):6.2f}s topk {float(r['topk_conf']):.4f} first {r['first_frame']:>6} {'CONFIRMED' if ok(r) else ''}")
PYEOF
