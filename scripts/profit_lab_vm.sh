#!/usr/bin/env bash
# Profit-first lab V2: full grid on VM (8 workers). Logs to data/profit_lab_run.log
# Sync repo to ~/scalping-v10-final first (see scripts/ml_v2_vm_setup.sh).
set -euo pipefail
cd ~/scalping-v10-final
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
mkdir -p data
LOG="data/profit_lab_run.log"
echo "=== Profit lab V2 (all rounds, full grid) — logging to ${LOG} ==="
nohup python3 profit_lab.py \
  --data-dir data/raw \
  --ml-dir data/ml \
  --out-json data/profit_lab_results.json \
  --workers 8 \
  > >(tee -a "${LOG}") 2>&1 &
echo "Started PID $! — tail -f ${LOG}"
