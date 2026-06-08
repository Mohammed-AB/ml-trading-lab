# ML pipeline V2 (walk-forward + ATR labels + rule features)

## 1. Regenerate feature parquet (includes `atr14`, ATR labels, optional `rule_*`)

```bash
# Base + ATR labels (default)
python3 ml_features.py --data-dir data/raw --out-dir data/ml

# Also append ~100 rule signal columns (trims rows to multiple of 5 M1 bars)
python3 ml_features.py --data-dir data/raw --out-dir data/ml --with-rules
```

## 2. Train single-split models (unchanged)

```bash
python3 ml_train.py --ml-dir data/ml
```

## 3. Walk-forward training

```bash
# Targets: fixed pip labels (default) or ATR labels
python3 ml_train_wf.py --ml-dir data/ml --label-long label_long_atr --label-short label_short_atr

# With rule columns in the parquet:
python3 ml_train_wf.py --ml-dir data/ml --features-v2 --label-long label_long_atr --label-short label_short_atr
```

Writes `data/ml/wf/model_{long,short}_fold{k}.txt` and `data/ml/wf_manifest.json`.

## 4. Arena sweeps

```bash
# Legacy: fixed SL/TP grid, single `model_long.txt` / `model_short.txt`
python3 -m strategy_arena --ml

# V2: WF folds + ATR SL/TP + threshold grid (needs new parquet with `atr14`)
python3 -m strategy_arena --ml-v2
```

## 5. Simulator changes (book + arena research)

`simulate_trades_vec` applies exit half-spread when `half_spread > 0`. Same-bar SL+TP → **SL (conservative)**. Spreads are unified with `scalp_mode.ml.bar_features.SPREAD_PIPS_DEFAULT`.
