#!/usr/bin/env python3
"""Quick sweep: ML V2 with TP=2.0x ATR."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from strategy_arena.ml_sweep import run_ml_v2_sweep

rows = run_ml_v2_sweep(ROOT / "data" / "ml", tp_mult=2.0)
if not rows:
    print("No results")
    sys.exit(1)
print("ML V2 sweep: TP 2.0x ATR\n")
hdr = f"{'Thr':>5} {'TPxATR':>7} {'n':>8} {'WR%':>7} {'PF':>6} {'Pips':>10} {'tpd':>6}"
print(hdr)
print("-" * 60)
for r in rows:
    print(
        f"{r['threshold']:5.2f} {r['tp_atr_mult']:7.2f} "
        f"{r['n']:8d} {r['wr']*100:6.1f} {r['pf']:6.2f} {r['total_pips']:+10.0f} {r['tpd']:6.2f}"
    )
