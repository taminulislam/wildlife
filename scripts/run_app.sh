#!/usr/bin/env bash
# Launch the TRACT interface on a Delta GPU node and print the tunnel command.
#
#   bash scripts/run_app.sh            # interactive, 1 h
#   bash scripts/run_app.sh 8081 04:00:00
set -euo pipefail
PORT="${1:-8080}"
TIME="${2:-01:00:00}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python

echo "requesting a GPU for ${TIME}; the URL appears once the node is allocated"
srun --account=bgte-delta-gpu --partition=gpuA100x4-interactive \
     --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=48g \
     --time="${TIME}" --pty \
     "$PY" "$ROOT/src/app/server.py" --port "$PORT" --device 0
