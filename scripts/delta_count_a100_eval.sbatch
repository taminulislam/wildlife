#!/bin/bash
#SBATCH --job-name=a100_cnt_eval
#SBATCH --account=bgte-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/a100_cnt_eval_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/a100_cnt_eval_%j.err
set -e
cd /work/nvme/bgte/tislam6/wildlife_project
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python
OUT=/work/hdd/bgte/tislam6/wildlife_outputs/counts/phaseC_orphan_yolo11m_conf0.10_A100
ls "$OUT"/counts_shard*.csv
$PY -c "import torch,ultralytics,cv2;print('torch',torch.__version__,'ultralytics',ultralytics.__version__,'opencv',cv2.__version__)" > "$OUT/env.txt"
$PY -u src/eval/count_eval.py --counts "$OUT" --gt data/annotate_v2/count_gt.csv
