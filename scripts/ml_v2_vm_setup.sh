#!/usr/bin/env bash
# Run on the training VM after code is in ~/scalping-v10-final
set -euo pipefail
cd ~/scalping-v10-final
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
echo "VM deps OK. Next:"
echo "  source .venv/bin/activate"
echo "  python scripts/fetch_historical.py --v2 --months 12 --output data/raw --env .env"
echo "  python3 ml_features.py --data-dir data/raw --fetch-months 12 --pairs EUR_USD GBP_USD USD_JPY USD_CAD AUD_USD NZD_USD"
echo "  nohup python3 ml_train.py --pairs EUR_USD GBP_USD USD_JPY USD_CAD AUD_USD NZD_USD \\"
echo "    --train-end '2025-11-30 23:59:59' --val-end '2026-02-28 23:59:59' > train.log 2>&1 &"
