# An Honest Out-of-Sample Evaluation of an ML Signal Gate for Intraday Forex

**A 62-feature gradient-boosted classifier achieves a real but modest predictive signal (test AUC ≈ 0.66) on 2.2M bars — yet neither the model nor 48 rule strategies produce a tradeable edge once realistic spread and a true out-of-time hold-out are enforced. This is a negative result, reported in full.**

*Mohammed Abumtary · [github.com/Mohammed-AB/ml-trading-lab](https://github.com/Mohammed-AB/ml-trading-lab) · 2026 · Educational / research only — not financial advice*

---

## TL;DR

- Trained long/short **LightGBM** classifiers on **2.21M M1 bars** across six FX majors with a **strictly chronological** train/validation/test split (no look-ahead).
- **Test AUC: 0.663 (long) / 0.649 (short)** — better than chance, but modest. AUC decays from train (≈0.75) → val (≈0.70) → test (≈0.66): the edge shrinks out of time.
- The strongest features are **session / time-of-day structure**, not classic price action — a caution about *what* the model is actually learning.
- On a true out-of-time hold-out, a probability-thresholded ML strategy **never reaches profit factor (PF) ≥ 1.0** (best PF = 0.78; every threshold nets negative pips).
- Across a **48-strategy rule arena** (in-sample vs out-of-sample), **zero strategies** show a reliable edge — the single OOS PF > 1.0 occurs on **61 trades** (statistical noise).
- **Conclusion:** a 0.66-AUC signal is real but does not survive costs. Reporting that honestly is the point.

---

## 1. Why this exists

Most public "trading bot" repos advertise win rates and equity curves but quietly skip out-of-sample testing and transaction costs. This project does the opposite: it builds a defensible ML + backtesting pipeline, then **stress-tests its own claims to destruction** — reporting what survives (little) and what doesn't (almost everything).

## 2. Data & setup

- **Instruments:** EUR/USD, GBP/USD, USD/JPY, USD/CAD, AUD/USD, NZD/USD (OANDA; M1, with M5 context).
- **Samples:** train = 1,327,895 · validation = 539,902 · test = 342,754 bars (≈ **2.21M** total).
- **Split:** strictly **chronological** — train ends before validation, validation before test. The model is always scored on its *future*.
- **Labels:** forward **triple-barrier** over a 120-bar horizon (take-profit / stop barriers); a same-bar TP+SL resolves to **SL (conservative)** so wins are never over-counted.
- **Cost:** a half-spread offset is charged **at label time**, so even the targets reflect realistic friction.
- **Features:** 62 engineered features — price-action, moving-average geometry, volatility, momentum, market structure, session/time, and VWAP — built by a *single function shared by training and live scoring* (no train/serve skew).
- **Model:** LightGBM gradient-boosted trees; separate long and short binary classifiers.

## 3. The signal: real, but modest

| Model | Train AUC | Val AUC | **Test AUC** | Test accuracy |
|-------|----------:|--------:|-------------:|--------------:|
| Long  | 0.755 | 0.703 | **0.663** | 0.803 |
| Short | 0.733 | 0.712 | **0.649** | 0.801 |

Two honest observations:

1. **Better than chance.** Test AUC ≈ 0.66 is a genuine, repeatable signal on 343k unseen, strictly-future bars.
2. **It decays out of time.** AUC falls monotonically train → val → test — the signature of a model that overfits somewhat and whose edge erodes as the market drifts from the training regime. (Test *accuracy* ≈ 0.80 looks high but mostly reflects class imbalance: most bars are "no trade.")

**What is the model actually using?** Top features by gain are dominated by **session/time structure** — minutes since the London open, hour-of-day (sin/cos), day-of-week — alongside gap-from-prior-bar and 60-bar high/low touch zones:

| Rank | Long model | Short model |
|------|-----------|-------------|
| 1 | `sess_min_since_london` | `pa_gap_atr` |
| 2 | `pa_gap_atr` | `struct_touch_low_zone_60` |
| 3 | `struct_touch_high_zone_60` | `struct_touch_high_zone_60` |
| 4 | `sess_hour_cos` | `sess_min_since_london` |
| 5 | `struct_touch_low_zone_60` | `sess_hour_cos` |

That's an important caveat: a large share of the "signal" is the model learning **when** (session timing) more than a robust **what** (directional price pattern) — exactly the kind of regime-bound effect that tends not to survive a changing market.

## 4. Does the signal become a strategy? (No.)

A predictive AUC is not a P&L. I swept the model's probability threshold on a **true out-of-time hold-out** (from 2026-03-01; SL = 10, TP = 15, 120-bar horizon, per-bar spread):

| Threshold | Trades | Win rate | **Profit factor** | Net pips |
|-----------|-------:|---------:|------------------:|---------:|
| 0.45 | 257 | 36.2% | **0.78** | −340 |
| 0.50 | 70  | 35.7% | 0.77 | −100 |
| 0.55 | 35  | 25.7% | 0.48 | −131 |
| 0.60 | 21  | 23.8% | 0.38 | −99  |

At **every** threshold, PF < 1.0 and total pips are negative. Raising the threshold (trading only the highest-confidence signals) *reduces* trades but **worsens** PF — the opposite of an edge that strengthens with conviction.

## 5. The rule-strategy arena: 48 strategies, 0 edges

Separately, I evaluated **48 hand-built rule strategies** (breakouts, mean-reversion, MACD/RSI/Bollinger variants, VWAP, inside-bar, etc.) in an arena that scores each in-sample (IS) and on a held-out out-of-sample (OOS) window, with realistic spread:

- **In-sample PF ranged 0.00–0.83 — every single strategy lost money even on the data it was built on.**
- **Out-of-sample, exactly one of 48 strategies posted PF > 1.0 — on 61 trades** (PF 1.21; its IS PF was 0.49), i.e., statistical noise.
- The highest-sample, most statistically reliable strategies were unambiguous losers — e.g. one linear-regression-channel rule at 28,339 IS / 4,792 OOS trades posted PF 0.50 / 0.53.

There is no surviving edge here, and the arena says so plainly. (Full 48-row leaderboard: [`docs/ARENA_LEADERBOARD.md`](docs/ARENA_LEADERBOARD.md).)

## 6. Why a 0.66-AUC model still loses

- **The cost barrier.** Intraday FX spread is a large fraction of a scalp's target. A classifier must be right *and* right by enough to clear spread — AUC 0.66 isn't.
- **Direction ≠ magnitude.** AUC measures the ranking of "up vs down," not whether moves are large enough to pay for the trade.
- **Regime-bound signal.** With session/timing features dominating, much of the edge is "what usually happens at this hour" — fragile as conditions shift, and visible in the train → test AUC decay.

## 7. Limitations (what would change the conclusion)

- Single broker, six majors, ~one-year window — a different cost model, instrument set, or horizon could shift results.
- Fixed SL/TP with triple-barrier labeling is one modeling choice among many.
- Portfolio-level effects, explicit regime conditioning, and folding the rule signals into the ML gate as features were not tested end-to-end.

A tradeable FX edge isn't disproven *in general* — but **this** pipeline, tested honestly, does not have one.

## 8. Reproducibility

Code, methodology, and the full 48-row arena leaderboard are public in **[ml-trading-lab](https://github.com/Mohammed-AB/ml-trading-lab)** (`docs/ARENA_LEADERBOARD.md`, `docs/ML_PIPELINE_V2.md`, `docs/ml_v2_backtest_holdout.txt`). Defaults ship in OANDA practice mode.

## 9. Takeaway

The interesting result isn't a profitable bot — it's a **clean negative**: a real but modest ML signal (AUC ≈ 0.66) does not translate into a tradeable edge once spread and out-of-sample reality are enforced, and a 48-strategy search confirms it. The contribution is the **methodology and the willingness to report it in full** — which is, ultimately, what separates research from marketing.
