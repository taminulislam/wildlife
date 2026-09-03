#!/usr/bin/env bash
# One command to open the TRACT interface, run entirely on Delta.
#
#   bash scripts/run_app.sh                  # port 8080, 2 h
#   bash scripts/run_app.sh 8095 04:00:00
#
# It submits a GPU job, waits for the node, then holds a local forward so that
# http://localhost:<port> ON THE LOGIN NODE reaches the server on the compute node.
# VS Code / Cursor Remote-SSH forwards that to the browser on your own machine
# automatically; with a plain terminal, add -L <port>:localhost:<port> to your ssh
# to the login node.
#
# Ctrl+C stops the forward and cancels the job.
set -uo pipefail
PORT="${1:-8080}"
TIME="${2:-02:00:00}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python
LOG="$ROOT/.tract_app.$PORT.log"
ACCT=bgte-delta-gpu

command -v sbatch >/dev/null || { echo "sbatch not found; are you on a Delta login node?"; exit 1; }
[ -x "$PY" ] || { echo "python not found at $PY"; exit 1; }

JOB=$(sbatch --parsable --account="$ACCT" --partition=gpuA100x4-interactive \
      --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=48g \
      --time="$TIME" --job-name=tract-app --output="$LOG" \
      --wrap "$PY -u $ROOT/src/app/server.py --port $PORT --device 0") || exit 1

cleanup() { echo; echo "stopping (job $JOB)"; scancel "$JOB" 2>/dev/null; exit 0; }
trap cleanup INT TERM

echo "submitted job $JOB, waiting for a GPU node..."
NODE=""
for _ in $(seq 1 600); do
  NODE=$(squeue -j "$JOB" -h -o %N 2>/dev/null)
  ST=$(squeue -j "$JOB" -h -o %T 2>/dev/null)
  [ -n "$NODE" ] && [ "$ST" = "RUNNING" ] && break
  [ -z "$ST" ] && { echo "job ended before starting; see $LOG"; exit 1; }
  sleep 2
done
[ -n "$NODE" ] || { echo "timed out waiting for a node"; scancel "$JOB"; exit 1; }
echo "allocated $NODE, waiting for the server to come up..."

for _ in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://$NODE:$PORT/" 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 2
done
[ "$code" = "200" ] || { echo "server did not answer; see $LOG"; scancel "$JOB"; exit 1; }

cat <<MSG

  ────────────────────────────────────────────────────────────
   TRACT is running on $NODE (job $JOB)

   Open   http://localhost:$PORT

   In VS Code / Cursor the port is forwarded for you. From a
   plain terminal, reconnect with:
     ssh -L $PORT:localhost:$PORT $USER@login.delta.ncsa.illinois.edu

   Leave this terminal open. Ctrl+C stops the app and frees the GPU.
   Server log: $LOG
  ────────────────────────────────────────────────────────────

MSG
ssh -N -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
    -L "$PORT:localhost:$PORT" "$NODE"
cleanup
