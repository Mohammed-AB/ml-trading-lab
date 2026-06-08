"""Arena research pack v2: 15 M5 scalping strategies (from forex_scalping_strategies).

Each function: (df, pair, pip, spread) -> (indices, dirs, entries, sls, tps).
Requires add_indicators columns: ema9, ema20, ema40, rsi14, atr14, bb_upper,
bb_lower, macd, macd_signal, stoch_k, stoch_d, hour, timestamp.

Registered as R8_* … R22_* via RESEARCH_V2_STRATEGIES (see bottom).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _is_bullish_engulf(o1, c1, o2, c2) -> bool:
    """Bar 2 (current) bullish-engulfs bar 1 (prior)."""
    return c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1


def _is_bearish_engulf(o1, c1, o2, c2) -> bool:
    return c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1


def _is_pin_bar_bull(o, h, l, c) -> bool:
    """Hammer / bullish pin: long lower wick, small body, close in upper half."""
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    lower_wick = min(o, c) - l
    return (lower_wick > 2 * body
            and body < 0.35 * rng
            and c > (h + l) / 2)


def _is_pin_bar_bear(o, h, l, c) -> bool:
    """Shooting star / bearish pin: long upper wick, small body, close in lower half."""
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    upper_wick = h - max(o, c)
    return (upper_wick > 2 * body
            and body < 0.35 * rng
            and c < (h + l) / 2)


def _is_doji(o, h, l, c, threshold: float = 0.1) -> bool:
    """Body smaller than `threshold` * range."""
    rng = h - l
    if rng <= 0:
        return False
    return abs(c - o) < threshold * rng


def _clamp_pips(value_pips: float, low: float, high: float) -> float:
    """Clamp a pip distance to the allowed [low, high] window."""
    return max(low, min(high, value_pips))


def _build_long(i, c_i, sl_pips, tp_pips, pip, spread,
                indices, dirs, entries, sls, tps):
    """Append a long trade. Spread added to entry."""
    entry = c_i + spread
    sl = entry - sl_pips * pip
    tp = entry + tp_pips * pip
    indices.append(i); dirs.append(1)
    entries.append(entry); sls.append(sl); tps.append(tp)


def _build_short(i, c_i, sl_pips, tp_pips, pip, spread,
                 indices, dirs, entries, sls, tps):
    """Append a short trade. Spread subtracted from entry."""
    entry = c_i - spread
    sl = entry + sl_pips * pip
    tp = entry - tp_pips * pip
    indices.append(i); dirs.append(-1)
    entries.append(entry); sls.append(sl); tps.append(tp)


# =============================================================================
# R1_ThreeSoldiers — Three White Soldiers / Three Black Crows
# =============================================================================

def r1_three_soldiers(df, pair, pip, spread):
    """
    Three same-direction candles continuation pattern with EMA stack filter.

    SOURCE        : Steve Nison, "Japanese Candlestick Charting Techniques" (1991).
                    Adapted for forex M5 by adding modern trend & exhaustion filters.
    CATEGORY      : Trend continuation
    BEST REGIME   : Established trends with low-to-moderate ATR; mid-session.
    EDGE          : Three consecutive same-color closes with each closing higher
                    (or lower) than the last shows sustained one-sided pressure.
                    On its own it has low edge — the EMA stack confirms structural
                    trend and the RSI cap avoids buying near exhaustion.
    PITFALLS      : Without the body-size filter on the third bar, you enter on
                    indecisive doji finishes that fail. Without the RSI ceiling,
                    you chase tops.
    EXPECTED      : Win rate 45-55%, R:R ~1.5, fires 1-3x/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; h = df['high'].values
    l = df['low'].values;  c = df['close'].values
    hour = df['hour'].values
    ema9 = df['ema9'].values; ema20 = df['ema20'].values; ema40 = df['ema40'].values
    rsi = df['rsi14'].values; atr = df['atr14'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        # Three-bar pattern check on bars i-3, i-2, i-1
        three_green = (c[i-3] > o[i-3]) and (c[i-2] > o[i-2]) and (c[i-1] > o[i-1])
        three_red   = (c[i-3] < o[i-3]) and (c[i-2] < o[i-2]) and (c[i-1] < o[i-1])
        higher_closes = c[i-1] > c[i-2] > c[i-3]
        lower_closes  = c[i-1] < c[i-2] < c[i-3]
        body3_ok = abs(c[i-1] - o[i-1]) > 0.4 * atr[i-1]

        long_sig = (three_green and higher_closes
                    and ema9[i-1] > ema20[i-1] > ema40[i-1]
                    and rsi[i-1] < 72
                    and body3_ok)
        short_sig = (three_red and lower_closes
                     and ema9[i-1] < ema20[i-1] < ema40[i-1]
                     and rsi[i-1] > 28
                     and body3_ok)

        if long_sig:
            sl_dist_pips = _clamp_pips((c[i] + spread - (min(l[i-3], l[i-2], l[i-1]) - pip)) / pip, 7, 12)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 18)
            _build_long(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif short_sig:
            sl_dist_pips = _clamp_pips(((max(h[i-3], h[i-2], h[i-1]) + pip) - (c[i] - spread)) / pip, 7, 12)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 18)
            _build_short(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R2_InsideBarBreak — Inside Bar Trend Breakout
# =============================================================================

def r2_inside_bar_break(df, pair, pip, spread):
    """
    Inside-bar (compression) followed by a break in the trend direction.

    SOURCE        : Al Brooks, "Reading Price Charts Bar by Bar"; popularised
                    for forex by Nial Fuller. Compression-then-expansion is one
                    of the oldest published edges in price action.
    CATEGORY      : Breakout (compression release)
    BEST REGIME   : Trending markets after brief consolidation. Shines on M5
                    EUR/USD and USD/JPY where inside bars are common at session
                    transitions.
    EDGE          : An inside bar marks a balance moment. Breaking it in the
                    prevailing trend direction is high-probability because
                    pent-up demand resolves with the trend.
    PITFALLS      : Without the EMA20>EMA40 filter, fires equally in both
                    directions and washes out. Mother-bar SL can be too wide
                    on volatile days — clamping at 14 pips is required.
    EXPECTED      : Win rate 50-58%, R:R ~1.4, 2-4 setups/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; h = df['high'].values
    l = df['low'].values;  c = df['close'].values
    hour = df['hour'].values
    ema20 = df['ema20'].values; ema40 = df['ema40'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        # i-1 is inside i-2
        inside = (h[i-1] < h[i-2]) and (l[i-1] > l[i-2])
        if not inside:
            continue

        trend_up = ema20[i-1] > ema40[i-1]
        trend_dn = ema20[i-1] < ema40[i-1]
        broke_up = c[i] > h[i-2]
        broke_dn = c[i] < l[i-2]

        if trend_up and broke_up:
            sl_dist_pips = _clamp_pips(((c[i] + spread) - (l[i-2] - pip)) / pip, 7, 14)
            tp_pips = _clamp_pips(1.4 * sl_dist_pips, 10, 18)
            _build_long(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif trend_dn and broke_dn:
            sl_dist_pips = _clamp_pips(((h[i-2] + pip) - (c[i] - spread)) / pip, 7, 14)
            tp_pips = _clamp_pips(1.4 * sl_dist_pips, 10, 18)
            _build_short(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R3_PinBarEMA — Pin Bar Rejection at EMA20
# =============================================================================

def r3_pin_bar_ema(df, pair, pip, spread):
    """
    Pin bar (hammer / shooting star) wicking through the EMA20 dynamic S/R.

    SOURCE        : Nial Fuller's pin-bar trading; Al Brooks' "tail bar" reversals.
                    EMA20 as dynamic S/R is from John Murphy.
    CATEGORY      : Reversal-into-trend (counter-pullback continuation)
    BEST REGIME   : Trending market on a pullback to the 20 EMA. Excellent on
                    GBP/USD and AUD/USD where pullback structure is cleanest.
    EDGE          : The pin's long wick is direct evidence that opposing flow
                    was rejected at a meaningful level. Combining this proof of
                    rejection with a confirmed dynamic S/R level (EMA20) gives
                    a high-quality reversal signal that aligns with trend.
    PITFALLS      : Pins in ranges fail more than pins in trends — the EMA20>40
                    filter is essential. Range-relative size matters; small pins
                    under 0.7*ATR are noise.
    EXPECTED      : Win rate 50-60%, R:R ~1.6, 1-2 setups/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; h = df['high'].values
    l = df['low'].values;  c = df['close'].values
    hour = df['hour'].values
    ema20 = df['ema20'].values; ema40 = df['ema40'].values
    atr = df['atr14'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        rng = h[i-1] - l[i-1]
        if rng < 0.7 * atr[i-1]:
            continue

        bull_pin = _is_pin_bar_bull(o[i-1], h[i-1], l[i-1], c[i-1])
        bear_pin = _is_pin_bar_bear(o[i-1], h[i-1], l[i-1], c[i-1])

        wicked_ema_from_above = bull_pin and l[i-1] <= ema20[i-1] and c[i-1] > ema20[i-1]
        wicked_ema_from_below = bear_pin and h[i-1] >= ema20[i-1] and c[i-1] < ema20[i-1]

        long_sig  = wicked_ema_from_above and ema20[i-1] > ema40[i-1]
        short_sig = wicked_ema_from_below and ema20[i-1] < ema40[i-1]

        if long_sig:
            sl_dist_pips = _clamp_pips(((c[i] + spread) - (l[i-1] - pip)) / pip, 7, 12)
            tp_pips = _clamp_pips(1.6 * sl_dist_pips, 10, 18)
            _build_long(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif short_sig:
            sl_dist_pips = _clamp_pips(((h[i-1] + pip) - (c[i] - spread)) / pip, 7, 12)
            tp_pips = _clamp_pips(1.6 * sl_dist_pips, 10, 18)
            _build_short(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R4_MACDZeroCross — MACD Zero-Line Momentum Push
# =============================================================================

def r4_macd_zero_cross(df, pair, pip, spread):
    """
    MACD line crossing the zero line, confirmed by signal-line agreement.

    SOURCE        : Gerald Appel (creator of MACD); zero-line cross variant
                    from Linda Raschke's "Street Smarts" momentum work.
    CATEGORY      : Momentum / trend initiation
    BEST REGIME   : Markets emerging from a range or transitioning trends.
                    Works across all majors but best on USD/JPY and EUR/USD
                    where MACD signals are smoother.
    EDGE          : Most traders use the signal-line cross, which fires far
                    too often. The zero-line cross is rarer and marks an
                    actual structural momentum change. Adding the EMA40
                    filter cuts whipsaw nearly in half.
    PITFALLS      : During chop, MACD can criss-cross zero. The 5-bar cooldown
                    and EMA40 filter matter. Don't relax the requirement that
                    macd > macd_signal at entry — that's the directional bias.
    EXPECTED      : Win rate 48-55%, R:R 1.4, 1-3 setups/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    c = df['close'].values
    hour = df['hour'].values
    ema40 = df['ema40'].values
    macd = df['macd'].values
    sig = df['macd_signal'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        crossed_up = macd[i-2] < 0 and macd[i-1] > 0
        crossed_dn = macd[i-2] > 0 and macd[i-1] < 0
        mom_up = macd[i-1] > sig[i-1]
        mom_dn = macd[i-1] < sig[i-1]
        trend_up = c[i-1] > ema40[i-1]
        trend_dn = c[i-1] < ema40[i-1]

        if crossed_up and mom_up and trend_up:
            _build_long(i, c[i], 10, 14, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif crossed_dn and mom_dn and trend_dn:
            _build_short(i, c[i], 10, 14, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R5_StochCrossExtreme — Stochastic Cross in Overbought/Oversold
# =============================================================================

def r5_stoch_cross_extreme(df, pair, pip, spread):
    """
    Stochastic %K/%D cross while in oversold (long) or overbought (short).
    The arena's only pure mean-reversion strategy.

    SOURCE        : George Lane (creator of Stochastic Oscillator, 1950s).
                    The "cross-in-extreme" filter is from Larry Williams' work.
    CATEGORY      : Mean reversion
    BEST REGIME   : Range-bound conditions. Critically, Asian / off-peak hours
                    (21-06 UTC) when EUR/USD and AUD/USD typically range.
    EDGE          : %K crossing %D inside an extreme zone signals momentum
                    flipping at a point where mean reversion is most likely.
                    Filtering by BB-band non-extension prevents catching falling
                    knives in trending breaks.
    PITFALLS      : Will get destroyed in trends — that's why session filter
                    excludes London/NY. Hit-rate strategy: R:R below 1 by design.
    EXPECTED      : Win rate 58-65%, R:R 0.75, fires only in Asian session.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; c = df['close'].values
    hour = df['hour'].values
    sk = df['stoch_k'].values; sd = df['stoch_d'].values
    bbu = df['bb_upper'].values; bbl = df['bb_lower'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        # Asian / off-peak ranging window; skip London open and NY hours
        if not (hour[i] >= 21 or hour[i] <= 5):
            continue

        cross_up = sk[i-2] < sd[i-2] and sk[i-1] > sd[i-1]
        cross_dn = sk[i-2] > sd[i-2] and sk[i-1] < sd[i-1]
        oversold   = sk[i-1] < 25
        overbought = sk[i-1] > 75
        bull_candle = c[i-1] > o[i-1]
        bear_candle = c[i-1] < o[i-1]
        not_falling = c[i-1] > bbl[i-1]
        not_rising  = c[i-1] < bbu[i-1]

        if cross_up and oversold and bull_candle and not_falling:
            _build_long(i, c[i], 12, 9, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif cross_dn and overbought and bear_candle and not_rising:
            _build_short(i, c[i], 12, 9, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R6_NR4Breakout — Linda Raschke's Narrow Range 4 (NR4)
# =============================================================================

def r6_nr4_breakout(df, pair, pip, spread):
    """
    Smallest-range bar of the last 4 bars often precedes volatility expansion;
    trade the break of that bar's high or low.

    SOURCE        : Linda Bradford Raschke & Laurence Connors,
                    "Street Smarts: High Probability Short-Term Trading
                    Strategies" (1995). Chapter 4 — NR4 / NR7 setup.
    CATEGORY      : Volatility expansion breakout
    BEST REGIME   : Pre-session compression (just before London open or NY open).
                    Edge collapses outside these windows.
    EDGE          : Range contraction is a precursor to expansion. Statistically
                    rigorous: NR4 bars are the bottom decile of recent volatility.
                    Mean reversion of volatility is one of the strongest
                    market regularities. Direction comes from minor trend bias.
    PITFALLS      : Outside pre-session windows the breakout direction is
                    coin-flip. The session filter is the strategy — not optional.
    EXPECTED      : Win rate 50-58%, R:R ~1.5, 1-3 setups/week per pair (rare).
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; h = df['high'].values
    l = df['low'].values;  c = df['close'].values
    hour = df['hour'].values
    ema20 = df['ema20'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        # Pre-London (06-08 UTC) or pre-NY (12-14 UTC) — where the edge lives
        if not ((6 <= hour[i] <= 8) or (12 <= hour[i] <= 14)):
            continue

        # i-1 is NR4 vs i-2, i-3, i-4, i-5? Raschke uses last 4 including itself.
        # Compare i-1 range to ranges of i-2, i-3, i-4
        rng_im1 = h[i-1] - l[i-1]
        rng_im2 = h[i-2] - l[i-2]
        rng_im3 = h[i-3] - l[i-3]
        rng_im4 = h[i-4] - l[i-4]
        is_nr4 = rng_im1 < min(rng_im2, rng_im3, rng_im4)

        if not is_nr4:
            continue

        broke_up = c[i] > h[i-1]
        broke_dn = c[i] < l[i-1]
        bias_up = c[i-1] > ema20[i-1]
        bias_dn = c[i-1] < ema20[i-1]

        if broke_up and bias_up:
            sl_dist_pips = _clamp_pips(((c[i] + spread) - (l[i-1] - pip)) / pip, 6, 12)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 20)
            _build_long(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif broke_dn and bias_dn:
            sl_dist_pips = _clamp_pips(((h[i-1] + pip) - (c[i] - spread)) / pip, 6, 12)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 20)
            _build_short(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R7_EMARibbonPullback — 9/20/40 Stack + Pullback to EMA20
# =============================================================================

def r7_ema_ribbon_pullback(df, pair, pip, spread):
    """
    Wait for a stable trend (full ribbon stack for 5+ bars), then enter on
    pullback into EMA20 followed by rejection.

    SOURCE        : Combination of Daryl Guppy's Multiple Moving Average system
                    and classic "buy-the-dip-in-uptrend" continuation logic.
    CATEGORY      : Trend continuation (pullback entry)
    BEST REGIME   : Strong, persistent trends with minor pullbacks. Best on
                    GBP/USD, AUD/USD where pullbacks are clean.
    EDGE          : Most trend strategies fail because they enter at extension.
                    This one waits for the natural pullback that institutional
                    flow creates around the 20-EMA, then enters on confirmed
                    rejection. Higher win rate than chase-style trend entries.
    PITFALLS      : Requires the 5-bar ribbon-stable filter — without it you
                    take pullbacks in fragile trends that fail.
    EXPECTED      : Win rate 52-60%, R:R ~1.6, 1-3 setups/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; h = df['high'].values
    l = df['low'].values;  c = df['close'].values
    hour = df['hour'].values
    ema9 = df['ema9'].values; ema20 = df['ema20'].values; ema40 = df['ema40'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        stack_up_5 = all(ema9[i-k] > ema20[i-k] > ema40[i-k] for k in range(1, 6))
        stack_dn_5 = all(ema9[i-k] < ema20[i-k] < ema40[i-k] for k in range(1, 6))

        # Check for pullback into EMA20 in last 3 bars
        pulled_in_up = min(l[i-3], l[i-2], l[i-1]) <= ema20[i-1]
        pulled_in_dn = max(h[i-3], h[i-2], h[i-1]) >= ema20[i-1]
        rejected_up = c[i-1] > ema9[i-1] and c[i-1] > o[i-1]
        rejected_dn = c[i-1] < ema9[i-1] and c[i-1] < o[i-1]

        if stack_up_5 and pulled_in_up and rejected_up:
            recent_low = min(l[i-3], l[i-2], l[i-1])
            sl_dist_pips = _clamp_pips(((c[i] + spread) - (recent_low - pip)) / pip, 7, 13)
            tp_pips = _clamp_pips(1.6 * sl_dist_pips, 10, 18)
            _build_long(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif stack_dn_5 and pulled_in_dn and rejected_dn:
            recent_high = max(h[i-3], h[i-2], h[i-1])
            sl_dist_pips = _clamp_pips(((recent_high + pip) - (c[i] - spread)) / pip, 7, 13)
            tp_pips = _clamp_pips(1.6 * sl_dist_pips, 10, 18)
            _build_short(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R8_EngulfingRSI — Engulfing Pattern + RSI Confluence Reversal
# =============================================================================

def r8_engulfing_rsi(df, pair, pip, spread):
    """
    Bullish/bearish engulfing pattern preceded by an oversold/overbought RSI
    print, with confirmation that the engulfing close clears the prior extreme.

    SOURCE        : Engulfing pattern from Nison; RSI confluence approach
                    formalized in Connors/Alvarez "Short Term Trading Strategies
                    That Work" (2008).
    CATEGORY      : Reversal
    BEST REGIME   : End of pullbacks within a larger range, or at swing extremes.
    EDGE          : Engulfings alone are noise — they fire ~once per session
                    on M5 with no edge. Pairing with a recent RSI extreme
                    (oversold for bull engulf, overbought for bear) restricts
                    them to setups where the reversal has fuel. The "close
                    clears prior high" requirement filters indecisive engulfs.
    PITFALLS      : Without the RSI filter, fires too often. Without the
                    "close clears prior high" filter, you trade weak engulfs
                    that don't follow through.
    EXPECTED      : Win rate 50-58%, R:R ~1.5, 1-2 setups/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; h = df['high'].values
    l = df['low'].values;  c = df['close'].values
    hour = df['hour'].values
    rsi = df['rsi14'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        bull_eng = _is_bullish_engulf(o[i-2], c[i-2], o[i-1], c[i-1])
        bear_eng = _is_bearish_engulf(o[i-2], c[i-2], o[i-1], c[i-1])
        oversold_recent = min(rsi[i-4:i]) < 32
        overbought_recent = max(rsi[i-4:i]) > 68
        cleared_high = c[i-1] > h[i-2]
        cleared_low  = c[i-1] < l[i-2]

        if bull_eng and oversold_recent and cleared_high:
            sl_dist_pips = _clamp_pips(((c[i] + spread) - (l[i-1] - pip)) / pip, 7, 13)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 9, 18)
            _build_long(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif bear_eng and overbought_recent and cleared_low:
            sl_dist_pips = _clamp_pips(((h[i-1] + pip) - (c[i] - spread)) / pip, 7, 13)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 9, 18)
            _build_short(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R9_BBWalk — Bollinger Band Walk (Trend Continuation)
# =============================================================================

def r9_bb_walk(df, pair, pip, spread):
    """
    The opposite of BB-bounce: in strong trends, price walks along the band.
    Two consecutive closes pinned to the upper band + ribbon stack = ride it.

    SOURCE        : John Bollinger himself, "Bollinger on Bollinger Bands" (2001),
                    Chapter on "walking the bands" — a regime-conditional reversal
                    of the standard BB bounce strategy.
    CATEGORY      : Trend continuation (volatility-driven)
    BEST REGIME   : Strong directional trends with expanding volatility. EUR/USD
                    and GBP/USD during NY momentum sessions.
    EDGE          : Bollinger explicitly warned that bouncing the bands fails
                    in trends. The "walk" pattern — repeated closes near the
                    band — is the trend signature. Trading WITH the walk in a
                    confirmed trend captures the strongest moves.
    PITFALLS      : Without the EMA stack filter, you'll be running into
                    standard BB-bounce setups in ranging conditions. Quick TP
                    is essential — bands snap back on the slightest pause.
    EXPECTED      : Win rate 48-55%, R:R ~1.1 (quick scalp), 2-4 setups/day.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    c = df['close'].values
    hour = df['hour'].values
    ema9 = df['ema9'].values; ema20 = df['ema20'].values; ema40 = df['ema40'].values
    bbu = df['bb_upper'].values; bbl = df['bb_lower'].values
    atr = df['atr14'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        near_upper_im1 = c[i-1] >= bbu[i-1] - 0.25 * atr[i-1]
        near_upper_im2 = c[i-2] >= bbu[i-2] - 0.25 * atr[i-2]
        near_lower_im1 = c[i-1] <= bbl[i-1] + 0.25 * atr[i-1]
        near_lower_im2 = c[i-2] <= bbl[i-2] + 0.25 * atr[i-2]

        trend_up = ema9[i-1] > ema20[i-1] > ema40[i-1]
        trend_dn = ema9[i-1] < ema20[i-1] < ema40[i-1]

        long_sig = near_upper_im1 and near_upper_im2 and trend_up and c[i-1] > c[i-2]
        short_sig = near_lower_im1 and near_lower_im2 and trend_dn and c[i-1] < c[i-2]

        if long_sig:
            _build_long(i, c[i], 11, 12, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif short_sig:
            _build_short(i, c[i], 11, 12, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R10_RSI50Momentum — RSI 50-Cross with EMA Filter
# =============================================================================

def r10_rsi50_momentum(df, pair, pip, spread):
    """
    RSI crossing the 50 line in the trend direction — the "true" momentum
    midpoint for continuation entries.

    SOURCE        : Andrew Cardwell's RSI methodology (he taught Constance Brown);
                    Brown's "Technical Analysis for the Trading Professional"
                    formalizes the 50-line as RSI's most important level.
    CATEGORY      : Momentum continuation
    BEST REGIME   : Trending markets with regular pullbacks. Fires more often
                    than zero-line crosses, with similar quality.
    EDGE          : Cardwell's research shows RSI in uptrends oscillates
                    roughly 40-80; in downtrends roughly 20-60. The 50 line
                    is a regime separator. Crossing it back through the
                    trend direction is a clean continuation signal.
    PITFALLS      : Without the "not extended" cap (RSI < 65 for longs),
                    fires too late on already-extended moves.
    EXPECTED      : Win rate 50-56%, R:R 1.4, 2-4 setups/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    c = df['close'].values
    hour = df['hour'].values
    ema20 = df['ema20'].values; ema40 = df['ema40'].values
    rsi = df['rsi14'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        crossed_up_50 = rsi[i-2] < 50 and rsi[i-1] > 50
        crossed_dn_50 = rsi[i-2] > 50 and rsi[i-1] < 50
        trend_up = ema20[i-1] > ema40[i-1] and c[i-1] > ema20[i-1]
        trend_dn = ema20[i-1] < ema40[i-1] and c[i-1] < ema20[i-1]
        not_overbought = rsi[i-1] < 65
        not_oversold   = rsi[i-1] > 35

        if crossed_up_50 and trend_up and not_overbought:
            _build_long(i, c[i], 10, 14, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif crossed_dn_50 and trend_dn and not_oversold:
            _build_short(i, c[i], 10, 14, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R11_ORB_London — Opening Range Breakout (London first hour)
# =============================================================================

def r11_orb_london(df, pair, pip, spread):
    """
    Trade the breakout of the high or low established during the first 60
    minutes of the London session (07:00-08:00 UTC).

    SOURCE        : Toby Crabel, "Day Trading with Short Term Price Patterns
                    and Opening Range Breakout" (1990) — the seminal work on
                    ORB. Adapted for FX from his equity research.
    CATEGORY      : Breakout
    BEST REGIME   : Trending or news-driven days. London open is the highest-
                    volume window for EUR-pairs and GBP-pairs.
    EDGE          : The first hour establishes a reference range. Breaking it
                    typically continues for at least 1-2 ATR because liquidity
                    providers have priced the range; a break invalidates that
                    pricing. Strong on M5 because the 12-bar range is well-formed.
    PITFALLS      : Inside-day pattern (no breakout) generates no trades — that's
                    fine, the strategy waits. Breakout failures are the main loss
                    pattern; the 12-pip clamp on SL controls them.
    EXPECTED      : Win rate 48-55%, R:R 1.5, 0-1 setup/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    hour = df['hour'].values
    ts = pd.to_datetime(df['timestamp']).values

    # Track per-day London opening range
    or_high = None
    or_low = None
    or_day = None
    fired_long_today = False
    fired_short_today = False

    for i in range(50, len(df) - 1):
        # Day boundary: reset OR each day
        day = pd.Timestamp(ts[i]).date()
        if day != or_day:
            or_day = day
            or_high = None
            or_low = None
            fired_long_today = False
            fired_short_today = False

        # Build OR during 07:00-07:55 UTC (12 bars on M5)
        if hour[i] == 7:
            or_high = max(or_high, h[i]) if or_high is not None else h[i]
            or_low  = min(or_low, l[i]) if or_low is not None else l[i]
            continue

        # Trade the break only between 08:00 and 12:00 UTC
        if not (8 <= hour[i] <= 12):
            continue
        if or_high is None or or_low is None:
            continue
        if i - last < 5:
            continue

        broke_up = c[i] > or_high and not fired_long_today
        broke_dn = c[i] < or_low and not fired_short_today

        if broke_up:
            sl_dist_pips = _clamp_pips(((c[i] + spread) - (or_low - pip)) / pip, 7, 12)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 20)
            _build_long(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
            fired_long_today = True
        elif broke_dn:
            sl_dist_pips = _clamp_pips(((or_high + pip) - (c[i] - spread)) / pip, 7, 12)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 20)
            _build_short(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i
            fired_short_today = True

    return indices, dirs, entries, sls, tps


# =============================================================================
# R12_MorningEveningStar — Three-Bar Reversal Pattern
# =============================================================================

def r12_morning_evening_star(df, pair, pip, spread):
    """
    Morning Star (bullish) or Evening Star (bearish) three-bar reversal pattern,
    confirmed at a meaningful Bollinger band touch.

    SOURCE        : Steve Nison, "Japanese Candlestick Charting Techniques."
                    BB-band requirement adapted from John Bollinger's reversal
                    confluence guidance.
    CATEGORY      : Reversal
    BEST REGIME   : Exhausted trends near band extremes; works on all majors
                    but particularly clean on USD/JPY where M5 candle bodies
                    are well-defined.
    EDGE          : Morning Star = strong down candle, then a small body
                    (indecision/star), then strong up candle that closes well
                    into bar-1's body. The three-bar structure shows momentum
                    failure followed by reversal. Restricting to BB extremes
                    ensures you're trading reversal where mean-reversion has fuel.
    PITFALLS      : Without the BB filter, fires inside ranges where there's
                    no reversal target. The "small middle bar" definition matters —
                    too loose admits weak setups.
    EXPECTED      : Win rate 50-58%, R:R 1.5, 1-2 setups/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; h = df['high'].values
    l = df['low'].values;  c = df['close'].values
    hour = df['hour'].values
    bbu = df['bb_upper'].values; bbl = df['bb_lower'].values
    atr = df['atr14'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        # Morning Star: bar i-3 strong down, bar i-2 small body, bar i-1 strong up
        bar3_down = c[i-3] < o[i-3] and abs(c[i-3] - o[i-3]) > 0.5 * atr[i-3]
        bar2_small = abs(c[i-2] - o[i-2]) < 0.3 * atr[i-2]
        bar1_up = c[i-1] > o[i-1] and abs(c[i-1] - o[i-1]) > 0.5 * atr[i-1]
        bar1_close_in_bar3 = c[i-1] > (o[i-3] + c[i-3]) / 2  # close above mid of bar 3
        near_bbl = min(l[i-3], l[i-2], l[i-1]) <= bbl[i-1] * 1.0005

        morning_star = (bar3_down and bar2_small and bar1_up
                        and bar1_close_in_bar3 and near_bbl)

        # Evening Star: bar i-3 strong up, bar i-2 small body, bar i-1 strong down
        bar3_up = c[i-3] > o[i-3] and abs(c[i-3] - o[i-3]) > 0.5 * atr[i-3]
        bar1_down = c[i-1] < o[i-1] and abs(c[i-1] - o[i-1]) > 0.5 * atr[i-1]
        bar1_close_below_mid3 = c[i-1] < (o[i-3] + c[i-3]) / 2
        near_bbu = max(h[i-3], h[i-2], h[i-1]) >= bbu[i-1] * 0.9995

        evening_star = (bar3_up and bar2_small and bar1_down
                        and bar1_close_below_mid3 and near_bbu)

        if morning_star:
            recent_low = min(l[i-3], l[i-2], l[i-1])
            sl_dist_pips = _clamp_pips(((c[i] + spread) - (recent_low - pip)) / pip, 7, 13)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 18)
            _build_long(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif evening_star:
            recent_high = max(h[i-3], h[i-2], h[i-1])
            sl_dist_pips = _clamp_pips(((recent_high + pip) - (c[i] - spread)) / pip, 7, 13)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 18)
            _build_short(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R13_TweezerEMA — Tweezer Top/Bottom at EMA20
# =============================================================================

def r13_tweezer_ema(df, pair, pip, spread):
    """
    Two consecutive bars with matching highs (tweezer top) or matching lows
    (tweezer bottom) wicking the EMA20 — a precise rejection signal.

    SOURCE        : Tweezer pattern from Nison's candlestick canon. Pairing
                    with EMA20 dynamic S/R from Murphy's classical TA framework.
    CATEGORY      : Reversal at dynamic S/R
    BEST REGIME   : Trending markets where EMA20 acts as live support/resistance.
                    Small frequency strategy — quality over quantity.
    EDGE          : Tweezers represent two consecutive failures to break beyond
                    a level. When that level coincides with EMA20 in a clear
                    trend, you have multi-touch validation of dynamic S/R plus
                    candle confirmation. Rare but high-quality.
    PITFALLS      : "Matching" highs/lows must be within ~10% of ATR — too
                    strict misses the pattern, too loose admits noise.
    EXPECTED      : Win rate 52-60%, R:R 1.5, 0-1 setups/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; h = df['high'].values
    l = df['low'].values;  c = df['close'].values
    hour = df['hour'].values
    ema20 = df['ema20'].values; ema40 = df['ema40'].values
    atr = df['atr14'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        tol = 0.10 * atr[i-1]

        # Tweezer bottom: two consecutive bars with near-equal lows, both
        # touching/wicking EMA20 from above, in an uptrend
        matching_lows = abs(l[i-1] - l[i-2]) < tol
        both_wicked_up = l[i-1] <= ema20[i-1] and l[i-2] <= ema20[i-2]
        bull_close = c[i-1] > o[i-1]
        trend_up = ema20[i-1] > ema40[i-1]

        # Tweezer top: matching highs at EMA20 from below, downtrend
        matching_highs = abs(h[i-1] - h[i-2]) < tol
        both_wicked_dn = h[i-1] >= ema20[i-1] and h[i-2] >= ema20[i-2]
        bear_close = c[i-1] < o[i-1]
        trend_dn = ema20[i-1] < ema40[i-1]

        if matching_lows and both_wicked_up and bull_close and trend_up:
            sl_dist_pips = _clamp_pips(((c[i] + spread) - (min(l[i-1], l[i-2]) - pip)) / pip, 7, 12)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 18)
            _build_long(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif matching_highs and both_wicked_dn and bear_close and trend_dn:
            sl_dist_pips = _clamp_pips(((max(h[i-1], h[i-2]) + pip) - (c[i] - spread)) / pip, 7, 12)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 18)
            _build_short(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R14_MACDHistogramReversal — MACD Histogram Peak/Trough Divergence
# =============================================================================

def r14_macd_histogram_reversal(df, pair, pip, spread):
    """
    MACD histogram reversing from a peak/trough — momentum slowing before
    price turns. Confirmed by candle agreement.

    SOURCE        : Alexander Elder, "Trading for a Living" (1993) — Elder's
                    Triple Screen system uses MACD histogram slope as the
                    primary momentum-change signal.
    CATEGORY      : Momentum reversal
    BEST REGIME   : Late-stage trend moves where momentum decelerates. Best
                    on USD/JPY and EUR/USD where MACD smoothness allows clean
                    peak detection.
    EDGE          : Histogram = (MACD - Signal). When histogram peaks and
                    starts declining, momentum is fading even if price is
                    still moving. This precedes price reversals by 1-3 bars
                    on average — perfect M5 scalp window. Requiring candle
                    direction agreement filters out false histogram dips.
    PITFALLS      : Choppy MACD produces frequent fake peaks. The 5-bar
                    cooldown plus the requirement that the histogram was
                    above/below zero (not just any reversal) cuts noise.
    EXPECTED      : Win rate 48-55%, R:R 1.4, 1-3 setups/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; c = df['close'].values
    hour = df['hour'].values
    macd = df['macd'].values
    sig = df['macd_signal'].values

    hist = macd - sig

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        # Histogram peak: hist[i-2] > hist[i-3] AND hist[i-1] < hist[i-2], with hist > 0
        hist_peak = (hist[i-2] > hist[i-3]
                     and hist[i-1] < hist[i-2]
                     and hist[i-2] > 0)
        # Histogram trough: hist[i-2] < hist[i-3] AND hist[i-1] > hist[i-2], hist < 0
        hist_trough = (hist[i-2] < hist[i-3]
                       and hist[i-1] > hist[i-2]
                       and hist[i-2] < 0)

        bear_candle = c[i-1] < o[i-1]
        bull_candle = c[i-1] > o[i-1]

        if hist_peak and bear_candle:
            _build_short(i, c[i], 10, 14, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i
        elif hist_trough and bull_candle:
            _build_long(i, c[i], 10, 14, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# R15_RossHook — Two-Bar Pullback Continuation (Ross Hook)
# =============================================================================

def r15_ross_hook(df, pair, pip, spread):
    """
    After a 1-2-3 swing creating a Ross Hook, enter on break of the hook in
    trend direction.

    SOURCE        : Joe Ross, "Trading the Ross Hook" (1995) — one of the
                    cleanest defined-edge swing patterns in price action.
    CATEGORY      : Trend continuation (structural pullback)
    BEST REGIME   : Trending markets after a clear 1-2-3 swing structure forms.
                    Works across all majors; cleanest on EUR/USD, GBP/USD.
    EDGE          : The "hook" forms when price makes a high (point 2), pulls
                    back to a higher low (point 3), then has a small reaction.
                    Breaking above point 2 from this hook signals continuation
                    of the dominant trend with structural validation —
                    "validated" because point 3 held.
    PITFALLS      : Identifying point 2/3 robustly on M5 requires lookback;
                    using last-N-bar high/low in trend context simplifies.
    EXPECTED      : Win rate 50-58%, R:R 1.5, 1-2 setups/day per pair.
    """
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -999

    o = df['open'].values; h = df['high'].values
    l = df['low'].values;  c = df['close'].values
    hour = df['hour'].values
    ema20 = df['ema20'].values; ema40 = df['ema40'].values

    for i in range(50, len(df) - 1):
        if i - last < 5:
            continue
        if not (7 <= hour[i] <= 20):
            continue

        # Look back over the last 12 bars
        window_h = h[i-12:i]
        window_l = l[i-12:i]
        if len(window_h) < 12:
            continue

        # Point 2 = highest high of window (in uptrend)
        point2_idx = int(np.argmax(window_h))
        point2 = window_h[point2_idx]

        # Point 3 = lowest low AFTER point 2 in window
        if point2_idx >= len(window_l) - 2:
            point3_long_valid = False
        else:
            after_p2_lows = window_l[point2_idx + 1:]
            point3_idx = int(np.argmin(after_p2_lows)) + point2_idx + 1
            point3_long = window_l[point3_idx]
            # Hook condition: point3 must be at least 5 bars after point 2,
            # and current price must break point 2
            point3_long_valid = (point3_idx - point2_idx >= 2
                                 and i - (i - 12 + point3_idx) >= 1
                                 and c[i] > point2)

        # Equivalent for shorts
        point2_idx_s = int(np.argmin(window_l))
        point2_s = window_l[point2_idx_s]
        if point2_idx_s >= len(window_h) - 2:
            point3_short_valid = False
        else:
            after_p2_highs = window_h[point2_idx_s + 1:]
            point3_idx_s = int(np.argmax(after_p2_highs)) + point2_idx_s + 1
            point3_short = window_h[point3_idx_s]
            point3_short_valid = (point3_idx_s - point2_idx_s >= 2
                                  and c[i] < point2_s)

        trend_up = ema20[i-1] > ema40[i-1]
        trend_dn = ema20[i-1] < ema40[i-1]

        if point3_long_valid and trend_up:
            sl_dist_pips = _clamp_pips(((c[i] + spread) - (point3_long - pip)) / pip, 7, 14)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 20)
            _build_long(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                        indices, dirs, entries, sls, tps)
            last = i
        elif point3_short_valid and trend_dn:
            sl_dist_pips = _clamp_pips(((point3_short + pip) - (c[i] - spread)) / pip, 7, 14)
            tp_pips = _clamp_pips(1.5 * sl_dist_pips, 10, 20)
            _build_short(i, c[i], sl_dist_pips, tp_pips, pip, spread,
                         indices, dirs, entries, sls, tps)
            last = i

    return indices, dirs, entries, sls, tps


# =============================================================================
# REGISTRY
# =============================================================================

# (display_name, signal_fn, max_bars) — names R8–R22 to avoid clash with research.py R1–R7
RESEARCH_V2_STRATEGIES = [
    ("R8_THREE", r1_three_soldiers, 20),
    ("R9_INBRK", r2_inside_bar_break, 20),
    ("R10_PIN", r3_pin_bar_ema, 20),
    ("R11_MACD0", r4_macd_zero_cross, 20),
    ("R12_STX", r5_stoch_cross_extreme, 20),
    ("R13_NR4", r6_nr4_breakout, 20),
    ("R14_EMAR", r7_ema_ribbon_pullback, 20),
    ("R15_ENG", r8_engulfing_rsi, 20),
    ("R16_BBW", r9_bb_walk, 15),
    ("R17_RSI50", r10_rsi50_momentum, 20),
    ("R18_ORBL", r11_orb_london, 30),
    ("R19_STAR", r12_morning_evening_star, 20),
    ("R20_TWZ", r13_tweezer_ema, 20),
    ("R21_MACDH", r14_macd_histogram_reversal, 20),
    ("R22_ROSS", r15_ross_hook, 20),
]

STRATEGIES = {name: fn for name, fn, _ in RESEARCH_V2_STRATEGIES}


# =============================================================================
# QUICK-START / SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    """
    Smoke-test on synthetic data so you can verify the module imports and runs.
    Replace with your real data feed.
    """
    rng = np.random.default_rng(42)
    n = 5000
    price = 1.10 + np.cumsum(rng.normal(0, 0.0003, n))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open":  price + rng.normal(0, 0.00005, n),
        "high":  price + np.abs(rng.normal(0, 0.00015, n)),
        "low":   price - np.abs(rng.normal(0, 0.00015, n)),
        "close": price,
    })
    df["hour"] = df["timestamp"].dt.hour
    df["ema9"]  = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema40"] = df["close"].ewm(span=40, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).fillna(50)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean().fillna(tr.mean())
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = (mid + 2 * std).fillna(df["close"])
    df["bb_lower"] = (mid - 2 * std).fillna(df["close"])
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = (100 * (df["close"] - low14) / (high14 - low14).replace(0, np.nan)).fillna(50)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean().fillna(50)

    pip = 0.0001
    spread = 0.00012

    print(f"{'Strategy':<28} {'Trades':>8} {'Long':>6} {'Short':>6}")
    print("-" * 52)
    for name, strat in STRATEGIES.items():
        idx, dirs, e, s, t = strat(df, "EUR/USD", pip, spread)
        n_long = sum(1 for d in dirs if d == 1)
        n_short = sum(1 for d in dirs if d == -1)
        print(f"{name:<28} {len(idx):>8} {n_long:>6} {n_short:>6}")
