#!/usr/bin/env bash
# Run on ml-training VM after repo sync to ~/scalping-v10-final (see scripts/ml_v2_vm_setup.sh).
set -euo pipefail
cd ~/scalping-v10-final
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
echo "=== Strategy lab (rounds 1–3, OOS from strategy_arena.config) ==="
python3 strategy_lab.py --data-dir data/raw --out-json data/strategy_lab_results.json
echo "Done. Inspect data/strategy_lab_results.json"
