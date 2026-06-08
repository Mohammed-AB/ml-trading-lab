"""python -m strategy_arena [options]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from strategy_arena.config import DEFAULT_DATA_RAW  # noqa: E402
from strategy_arena.runner import run_arena  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run book + research strategies on OANDA M1 (V2 pairs, realistic spreads).",
    )
    ap.add_argument(
        "--ml",
        action="store_true",
        help="Run ML LightGBM SL/TP grid (needs data/ml/features_*.parquet + model_*.txt)",
    )
    ap.add_argument(
        "--ml-v2",
        action="store_true",
        help="Walk-forward ML V2: wf_manifest.json + wf/model_*_fold*.txt, ATR SL/TP, threshold sweep",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / DEFAULT_DATA_RAW,
        help="Directory with {PAIR}_M1_12m.csv",
    )
    ap.add_argument(
        "--in-sample",
        action="store_true",
        help="Filter to in-sample (before OOS) instead of out-of-sample",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Use entire loaded range (no IS/OOS split); ignores --in-sample",
    )
    ap.add_argument(
        "--only-research",
        action="store_true",
        help="R1–R7 only (skip book STRATEGIES)",
    )
    ap.add_argument(
        "--only-book",
        action="store_true",
        help="S1–S16 book strategies only (skip R*)",
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write full result JSON to path",
    )
    ap.add_argument(
        "--lab",
        action="store_true",
        help="Run high-WR strategy lab (multi-family search, OOS scoring; see strategy_lab.py)",
    )
    ap.add_argument(
        "--profit-lab",
        action="store_true",
        help="Run profit-first lab V2 (PF focus, wide R:R, BE/trail; see profit_lab.py)",
    )
    ap.add_argument(
        "--sweep-v3",
        action="store_true",
        help="Ultimate sweep V3: 80 strats x TFs x exits x filters (see profit_lab_v3.py)",
    )
    args = ap.parse_args()

    if args.lab and (args.ml or args.ml_v2 or args.profit_lab):
        print("Use --lab alone, or --ml / --ml-v2 / --profit-lab, not combined.", file=sys.stderr)
        sys.exit(2)

    if args.profit_lab and (args.ml or args.ml_v2):
        print("Use --profit-lab alone, or --ml / --ml-v2, not combined.", file=sys.stderr)
        sys.exit(2)

    if args.ml and args.ml_v2:
        print("Use either --ml or --ml-v2, not both.", file=sys.stderr)
        sys.exit(2)

    if args.lab:
        from strategy_lab import main as lab_main  # noqa: WPS433

        raise SystemExit(lab_main(["--data-dir", str(args.data_dir)]))

    if args.profit_lab:
        from profit_lab import main as profit_main  # noqa: WPS433

        raise SystemExit(profit_main(["--data-dir", str(args.data_dir)]))

    if args.sweep_v3:
        from profit_lab_v3 import main as sweep_main  # noqa: WPS433

        raise SystemExit(sweep_main(["--data-dir", str(args.data_dir)]))

    if args.ml_v2:
        from strategy_arena.ml_sweep import run_ml_v2_sweep  # noqa: WPS433

        rows = run_ml_v2_sweep(ROOT / "data" / "ml")
        if not rows:
            print(
                "No ML V2 sweep — need data/ml/wf_manifest.json, wf/model_*_fold*.parquet, "
                "and features_*.parquet with atr14 + feature columns (run ml_train_wf.py).",
                file=sys.stderr,
            )
            sys.exit(1)
        print("ML V2 walk-forward sweep (ATR SL/TP, thresholds 0.50–0.65)\n")
        print(f"{'Thr':>5} {'TP×ATR':>7} {'n':>8} {'WR%':>7} {'PF':>6} {'Pips':>10} {'tpd':>6}")
        print("-" * 60)
        for r in rows:
            print(
                f"{r['threshold']:5.2f} {r['tp_atr_mult']:7.2f} "
                f"{r['n']:8d} {r['wr']*100:6.1f} {r['pf']:6.2f} {r['total_pips']:+10.0f} {r['tpd']:6.2f}"
            )
        sys.exit(0)

    if args.ml:
        from strategy_arena.ml_sweep import run_ml_sltp_sweep  # noqa: WPS433

        rows = run_ml_sltp_sweep(ROOT / "data" / "ml")
        if not rows:
            print("No ML data or empty sweep — need data/ml/features_*.parquet", file=sys.stderr)
            sys.exit(1)
        print("ML SL/TP sweep (hold-out from 2026-03-01), fixed threshold 0.45\n")
        print(f"{'SL':>4} {'TP':>4} {'BE%':>6} {'n':>8} {'WR%':>7} {'PF':>6} {'Pips':>10} {'tpd':>6}")
        print("-" * 60)
        for r in rows:
            print(
                f"{r['sl_pips']:4.0f} {r['tp_pips']:4.0f} {r['breakeven_wr_pct']:6.1f} "
                f"{r['n']:8d} {r['wr']*100:6.1f} {r['pf']:6.2f} {r['total_pips']:+10.0f} {r['tpd']:6.2f}"
            )
        sys.exit(0)

    res = run_arena(
        args.data_dir,
        oos=not args.in_sample,
        only_research=args.only_research,
        only_book=args.only_book,
        full=args.full,
    )
    if "error" in res:
        print(res["error"], file=sys.stderr)
        sys.exit(1)

    print(f"Window: {res['summary']['window']}  OOS from {res['summary']['oos_start_utc']}")
    print(f"Pairs: {res['summary']['pairs']}")
    print(f"Total filtered trades: {len(res['trades']):,}\n")
    print(f"{'Strategy':16s} {'N':>8} {'WR%':>7} {'PF':>6} {'Pips':>10} {'Avg':>8}")
    print("-" * 64)
    for s, n, wr, pf, tp, apv in res["ranked"][:50]:
        print(f"{s:16s} {n:8d} {wr:6.1f} {pf:6.2f} {tp:+10.0f} {apv:+8.2f}")
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {k: v for k, v in res.items() if k != "trades" or len(res.get("trades", [])) < 200_000},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
