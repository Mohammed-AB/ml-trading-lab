# Scalp Bot V10 — Complete Change Log

**Original:** `scalping-v10.2-final` (Practice account, $10k+, Limit orders)
**Current:** `scalping-v10-final` (Live account, ~$557, Market orders)
**Date:** April 12, 2026
**Files changed:** 30 | **Lines changed:** ~600

---

## Repeat-signal bleed fix (April 2026)

- **Risk:** Pending `pair+direction` dedup via `PendingOrderManager.snapshot_for_risk()`; SL-loss cooldown (30m) from `outcomes.jsonl`; dynamic pair tier confidence from `pair_stats.json` (limited / default / preferred); rejections logged to `alerts.jsonl`.
- **Orchestrator:** Wired pending provider; removed Model C solo pre-filter; `ManagedTrade` stores short `model` id + `cluster_id` (15m window).
- **Brain:** `pair_stats` tracks `avg_win`, `avg_loss`, `bad_rr`; `format_pair_stats_for_strategy_prompt()` for Strategy Agent.
- **Learning:** Daily review prepends 7-day `model × pair × direction` markdown table.
- **Deploy:** Restart container after pull; confirm `main.py` wires `set_pending_orders_provider` after `PendingOrderManager` init.

---

## 1. Configuration (config/settings.yaml)

### 1.1 Instruments — 3 pairs added

| Original (5 pairs) | Current (8 pairs) |
|---------------------|-------------------|
| EUR_USD | EUR_USD |
| USD_JPY | USD_JPY |
| GBP_USD | GBP_USD |
| AUD_USD | AUD_USD |
| EUR_GBP | EUR_GBP |
| — | **USD_CHF** (new) |
| — | **USD_CAD** (new) |
| — | **NZD_USD** (new) |

### 1.2 Sessions — opened up trading hours

| Setting | Original | Current |
|---------|----------|---------|
| `mode` | `overlap_only` (London–NY overlap, ~4 hrs/day) | **`weekday_extended`** (Mon–Fri all hours) |
| Blocked windows | witching_hour, rollover | Same (no change) |

### 1.3 Spread limits — widened for all pairs

| Pair | Original | Current |
|------|----------|---------|
| EUR_USD | 0.8 | **2.0** |
| USD_JPY | 0.8 | **2.0** |
| GBP_USD | 1.0 | **2.5** |
| AUD_USD | 1.2 | **1.8** |
| EUR_GBP | 1.2 | **1.8** |
| USD_CHF | — | **2.0** (new) |
| USD_CAD | — | **2.0** (new) |
| NZD_USD | — | **2.0** (new) |

### 1.4 Model A — wider stop loss

| Setting | Original | Current |
|---------|----------|---------|
| `sl_atr` | 0.8 | **2.0** (stop = 2× ATR instead of 0.8×) |

### 1.5 Model B — enabled

| Setting | Original | Current |
|---------|----------|---------|
| `enabled` | `false` | **`true`** (Range regime reversal trades active) |

### 1.6 Risk management — reworked for small accounts

| Setting | Original | Current | Purpose |
|---------|----------|---------|---------|
| `account_currency` | not present (hardcoded GBP) | **`USD`** | Correct pip→USD sizing |
| `risk_pct` | 0.0025 (0.25%) | **0.07** (7%) | More aggressive for small account |
| `daily_loss` | 0.01 (1%) | **0.28** (28%) | Scaled 4× risk_pct |
| `leverage` | not present | **33** | Worst-case OANDA margin rate |
| `margin_cap_safety` | not present | **0.35** | Max 35% of margin per trade |

### 1.7 Orders — switched from Limit to Market

| Setting | Original | Current | Purpose |
|---------|----------|---------|---------|
| `primary_order_type` | not present (always Limit) | **`MARKET`** | Immediate fills |
| `limit_ttl_seconds` | 180 | **300** | Longer window for limit fallbacks |
| `fallback_max_atr_distance` | 0.3 | **0.5** | Wider fallback acceptance |
| `price_bound_slippage` | 0.2 | **2.0** | Less restrictive slippage |

### 1.8 AI modules

| Module | Original | Current |
|--------|----------|---------|
| Regime Classifier | `enabled: false`, timeout: 800ms | **`enabled: true`**, timeout: **10000ms** |
| Borderline Reviewer | `enabled: false`, timeout: 500ms | `enabled: false`, timeout: **10000ms** |
| Post-Trade Analyst | `enabled: false` | **`enabled: true`** |

### 1.9 OANDA endpoint — Practice (default)

This research build defaults to the OANDA **practice (demo)** endpoint.
Switching to a live real-money endpoint is intentionally left to the user and
is not supported or encouraged by this repository.

| Setting | Value |
|---------|-------|
| `environment` | `practice` |
| `base_url` | `api-fxpractice.oanda.com` |
| `stream_url` | `stream-fxpractice.oanda.com` |

---

## 2. Main Loop (main.py)

### 2.1 Live-mode confirmation prompt
- Live mode always requires an interactive `input()` confirmation. The
  auto-confirm bypass was removed in the public/research build so the human
  confirmation can never be silently skipped.

### 2.2 Account currency from OANDA (not hardcoded)
- **Original:** `risk_config["account_currency"] = "GBP"` — hardcoded, wrong for USD accounts.
- **Current:** Fetches from OANDA `/summary` → `account.currency`. Falls back to `OANDA_ACCOUNT_CURRENCY` env, then `"USD"`.

### 2.3 Safe NAV parsing (the ~55k unit fix)
- **Original:** `nav = float(account.get("NAV", 10000))` — if NAV is missing, defaults to **$10,000**. Combined with real margin (~$556), this produced ~55k unit orders that OANDA rejected.
- **Current:** `parse_account_nav()` returns `None` if no valid NAV → **skips the pair** instead of trading on fake equity. `parse_oanda_decimal()` handles OANDA's string number format for `marginAvailable`.

### 2.4 Minute-key dedup (bug fix)
- **Original:** `last_cycle_second` — stayed at `0` after first run and could skip future minutes.
- **Current:** Uses `minute_key = (year, month, day, hour, minute)` tuple — correct deduplication.

### 2.5 P/L logging on trade close
- **Original:** No P/L recorded anywhere in logs.
- **Current:** Logs `order_type: "CLOSE"` to `trade_log.jsonl` with:
  - `pnl_pips`, `exit_price`, `exit_reason`, `hold_time_seconds`
  - For both bot-initiated closes (time_stop, etc.) and broker-reconciled closes (TP/SL hit)

### 2.6 Telegram alerts on trade events
- **Original:** No alerts sent for trade lifecycle events.
- **Current:** `alert_mgr.alert_info()` fires on:
  - `SL_MOVED` — stop loss trailing
  - `TRADE_CLOSED` — bot closes trade
  - `TRADE_CLOSED_BROKER` — broker TP/SL hit
  - `TRADE_SIGNAL` — new signal with entry/sl/tp/units

### 2.7 dotenv support
- Added `from dotenv import load_dotenv` + `load_dotenv()` so `.env` file works locally.

---

## 3. Executor (executor.py)

### 3.1 `parse_oanda_decimal()` — new function
Safely parses OANDA numeric fields that may be strings (`"556.12"`), ints, floats, or `None`. Returns `None` for invalid values — never invents a default.

### 3.2 `parse_account_nav()` — new function
Reads `NAV` first, then falls back to `balance`. Returns `None` if neither field is valid — prevents the catastrophic 10k default that caused ~55k unit orders.

---

## 4. Risk Manager (risk_manager.py)

### 4.1 `notional_usd_per_base_unit()` — new function
Calculates the USD notional value per 1 unit of base currency:
- EUR_USD → ~$1.08 (price of 1 EUR in USD)
- USD_JPY → $1.00 (base is already USD)
- EUR_GBP → mid × GBP_USD rate

Used for realistic margin estimation matching OANDA's actual calculation.

### 4.2 Margin cap — completely rewritten (core fix)

**Original formula (broken):**
```
estimated_margin = units × pip_value × 100
```
This does NOT match how OANDA calculates margin. For EUR/USD it massively underestimated, allowing ~55k units on a $556 account.

**New formula (correct):**
```
est_margin = units × notional_per_unit / leverage
cap = margin_available × margin_cap_safety (35%)
if est_margin > cap: units = cap × leverage / notional_per_unit
```
Matches OANDA's actual margin model. With leverage=33 and cap=35%, a $557 account gets ~6k–10k units per trade.

### 4.3 New configurable parameters
- `leverage` (default 50, set to 33 for worst-case OANDA pairs)
- `margin_cap_safety` (default 0.90, set to 0.35 per trade)
- `mid_price` parameter added to `evaluate()` for accurate notional calculation

---

## 5. Order Builder (order_builder.py)

### 5.1 `build_market()` — new method
Builds a Market order as the **primary** entry type (not just a fallback). Uses current mid price + slippage for price_bound calculation.

### 5.2 `use_market_primary` — new property
Returns `True` when `primary_order_type == "MARKET"` in config. Used by the decision pipeline to choose between `build_market()` and `build_limit()`.

### 5.3 Market order payload — removed FOK and priceBound

**Original OANDA payload:**
```json
{
  "type": "MARKET",
  "timeInForce": "FOK",
  "priceBound": "1.08557"
}
```
OANDA cancelled these when price couldn't fill exactly within the bound.

**Current payload:**
```json
{
  "type": "MARKET"
}
```
OANDA defaults to IOC (Immediate Or Cancel), allowing best-effort fills at market price.

---

## 6. Decision Pipeline (decision_pipeline.py)

### 6.1 Session gate — now configurable
- **Original:** `is_session_allowed(utc_now)` — hardcoded overlap-only.
- **Current:** Passes `mode` and `block` from config settings, enabling `weekday_extended` mode.

### 6.2 Market vs Limit — conditional order building
- **Original:** Always called `build_limit()`.
- **Current:** Checks `use_market_primary` → calls `build_market()` for immediate fills, or `build_limit()` for limit entries.

### 6.3 Mid price passed to risk manager
Now passes `mid_price=(bid + ask) / 2` into `risk.evaluate()` for accurate notional-based margin calculation.

---

## 7. Session Gate (session_gate.py)

### 7.1 Two modes supported
- **Original:** Only `overlap_only` (London–NY overlap, ~4 hours/day).
- **Current:** Added `weekday_extended` mode — Mon–Fri all hours with optional witching/rollover blocks. Configurable via `sessions.mode` and `sessions.block` in YAML.

### 7.2 Refactored blocking logic
Extracted `_blocked_by_named_windows()` helper so both modes share the same sub-window check logic. Cleaner, no duplication.

---

## 8. AI Prompts (regime_classifier.py, borderline_reviewer.py)

### 8.1 Regime Classifier prompt — made strict

**Original (verbose):**
```
You are a forex market regime classifier for scalping.

Current M5 indicators:
  EMA_slope: 0.0012
  BB_width: 0.00450
  ...

Classify as exactly one of: Trend_Up, Trend_Down, Range, NoTrade
Reply with: REGIME <name> <confidence 0-1>
```
Claude returned verbose explanations that broke the parser.

**Current (strict):**
```
Classify this forex M5 data into a regime. Reply ONLY with: REGIME <name> <confidence>
Valid names: Trend_Up, Trend_Down, Range, NoTrade
Example: REGIME Trend_Up 0.85

EMA_slope=0.0012 BB_width=0.00450 RSI=58.3 ...
Rule-based=Trend_Up Recent=[Trend_Up, Trend_Up, Range]
```

### 8.2 Borderline Reviewer prompt — same pattern
Condensed from verbose multi-line explanation to strict `APPROVE/REJECT <confidence> <reason>` format with example. Prevents unparseable responses.

---

## 9. Pending Manager (pending_manager.py)

### 9.1 Smarter cancel logic
- **Original:** Always called `cancel_order()` before fallback — caused noisy OANDA "ORDER_CANCEL_REJECTED" errors on already-expired orders.
- **Current:** Checks `check_order_status()` first:
  - If already `filled` → register trade, no fallback
  - If already `expired/cancelled` → skip cancel, go straight to fallback
  - If still `pending` → cancel, then fallback

### 9.2 Fill logging in `_register_trade()`
Now logs an `order_type: "FILL"` event to `trade_log.jsonl` with:
- `fill_price`, `expected_entry_price`, `actual_slippage_pips`
- `spread_at_signal`, `order_sent_ts`, `fill_received_ts`
- `sl_price`, `tp_price`, `units`, `signal_id`

### 9.3 Reconciler returns closed trades
`BrokerReconciler.reconcile()` now stores the `ManagedTrade` object in `changes["closed_trades"]` so `main.py` can calculate and log P/L for broker-closed positions.

---

## 10. Pip Utils (pip_utils.py)

### 10.1 EUR_GBP pip conversion for USD accounts
- **Original:** JPY conversion existed but GBP-quoted pairs (like EUR_GBP) returned raw pip value — wrong for USD accounts.
- **Current:** Detects `_GBP` suffix → converts pip value from GBP to USD via `pv × GBP_USD` rate. Falls back to `pv × 1.25`.

---

## 11. Monitoring (monitoring.py)

### 11.1 Telegram HTML formatting
- **Original:** Used Markdown (`*bold*`) — broke on event names with underscores (e.g. `MANUAL_TEST` parsed as italic).
- **Current:** Uses HTML (`<b>bold</b>`) with `html.escape()` on all fields. No more broken formatting.

---

## 12. Infrastructure (new files)

| File | Purpose |
|------|---------|
| `Dockerfile` | Python 3.12-slim container for the bot |
| `docker-compose.yml` | Single-service compose with .env, log/data volumes |
| `.gitignore` | Standard Python + logs + data ignores |
| `requirements.txt` | Added `anthropic>=0.39.0` for AI modules |

---

## 13. Test Files (8 files updated)

| Test File | Changes |
|-----------|---------|
| `test_config.py` | Assertions match new YAML values (risk_pct=0.07, daily_loss=0.28, leverage=33, sl_atr=2.0, etc.) |
| `test_executor.py` | Added `TestOandaAccountParsing` class (parse_oanda_decimal, parse_account_nav) |
| `test_risk_manager.py` | Added `TestNotionalMarginCap` class (small account EUR, USD_JPY margin tests) |
| `test_order_builder.py` | Market order body asserts **no** `timeInForce` or `priceBound` |
| `test_decision_pipeline.py` | `MockConfig` includes `sessions` property |
| `test_integration.py` | `MockConfig` includes `sessions` property |
| `test_coverage_boost.py` | Cancel assertion updated for skip-cancel-when-expired logic |
| `test_session_gate.py` | Updated for `weekday_extended` mode parameter |

---

## Impact Summary

| Before (Original) | After (Current) |
|--------------------|-----------------|
| NAV defaults to 10k if missing | **Skips trade if NAV unavailable** |
| ~55k units → OANDA cancels | **~6k–10k units → orders fill** |
| Margin: `pip × 100` (wrong) | **Margin: `notional / leverage` (matches OANDA)** |
| Limit orders only (rarely filled) | **Market orders primary (immediate fill)** |
| FOK + priceBound (orders rejected) | **Removed — best-effort fill** |
| 4 hrs/day trading window | **Mon–Fri all hours** |
| 5 pairs | **8 pairs** |
| No P/L logging | **Full fill/close/P/L in trade_log.jsonl** |
| No Telegram alerts | **Alerts on signals, closes, SL moves** |
| AI modules disabled | **Regime classifier + post-trade enabled** |
| Hardcoded GBP account currency | **Auto-detected from OANDA (USD)** |
| No Docker support | **Dockerized (practice-mode research deployment)** |
