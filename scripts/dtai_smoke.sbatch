#!/bin/bash
#SBATCH --job-name=gh_smoke
#SBATCH --account=bgte-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_smoke_%j.out
#SBATCH --error=/work/hdd/bgte/tislam6/wildlife_outputs/logs/gh_smoke_%j.err
set -e
module load python/anaconda3/2.12.0
source /work/nvme/bgte/tislam6/envs/wildlife_gh_venv/bin/activate
cd /work/nvme/bgte/tislam6/wildlife_project
echo "=== GH200 smoke test $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import platform, torch, numpy, cv2
print("machine :", platform.machine())
print("torch   :", torch.__version__, "| cuda avail:", torch.cuda.is_available())
print("gpu     :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("numpy   :", numpy.__version__, "| cv2:", cv2.__version__)
import ultralytics; print("ultralytics:", ultralytics.__version__)
from ultralytics import YOLO
m = YOLO("/work/hdd/bgte/tislam6/wildlife_outputs/runs/yolov10m_640_v3pooled/weights/best.pt")
r = m.val(data="data/dataset/yolo_v3_kf/data.yaml", split="test", imgsz=640,
          device=0, verbose=False)
print("SMOKE OK — keyframe-set mAP50 =", round(float(r.box.map50), 4),
      "(Delta A100 gave 0.670)")
PY
echo "=== smoke done $(date) ==="
