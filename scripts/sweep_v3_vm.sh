#!/usr/bin/env bash
# Ultimate sweep V3: full grid on VM (8 workers). Logs to data/sweep_v3_run.log
set -euo pipefail
cd ~/scalping-v10-final
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p data
LOG="data/sweep_v3_run.log"
echo "=== Sweep V3 (all rounds, full grid) — logging to ${LOG} ==="
nohup python3 profit_lab_v3.py \
  --data-dir data/raw \
  --out-json data/sweep_v3_results.json \
  --workers 8 \
  > "${LOG}" 2>&1 &
echo "Started PID $! — tail -f ${LOG}"
