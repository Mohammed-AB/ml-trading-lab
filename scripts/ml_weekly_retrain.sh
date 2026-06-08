#!/usr/bin/env bash
# Weekly ML refresh (cron). Run from repo root on a machine with FX CSVs + deps.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 ml_features.py
python3 ml_train.py
# Optional: python3 ml_backtest.py >> data/ml/weekly_backtest.log
