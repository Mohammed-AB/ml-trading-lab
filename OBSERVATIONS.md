# Observations Log

Running log of behavioral observations during live paper/live trading.
Separate from `CHANGELOG.md` (which tracks code changes) — this file tracks
**what the agents actually do** and where architecture creaks show up.

---

## 2026-04-20 — Skip-loop during repeat-signal 48h observation

**Context.** Day 1 of the 48h observation window for the repeat-signal
bleed fix (Waves A–D deployed 2026-04-19). No trades opened in the first
~6 hours of the London session (07:00–13:00 UTC).

**Symptom.** Strategy Agent produced 10+ consecutive SKIP verdicts citing
"solo Model E VWAP-reversion, no confluence." No trades reached Risk
Agent; no duplicate/cooldown/tier rejections fired. The skip-loop is
**Strategy-internal**, not a guardrail side-effect.

**Agent coordination failure.**

- **Chief Agent (11:56 UTC)** correctly diagnosed the loop and issued:
  - Pre-authorize USD_CHF short, conf ≥ 0.55
  - Block AUD_USD + GBP_USD
  - Broaden scan or explicitly propose session-edge continuation
- **Learning Agent (12:03:13 UTC)** wrote three new lessons:
  - "USD_CHF solo Model E short: do not auto-skip" (3/3 wins, +55.8p)
  - "AUD_USD longs: stop taking" (0/2, -18.9p)
  - "10 consecutive skips is over-filtering"
- **Strategy Agent (12:03:19 and 12:06:07 UTC)** SKIPped again, citing
  five older boardroom directives (10:38, 10:54, 11:09, 11:25, 11:40 UTC)
  and made no reference to the 11:56 directive or the 12:03 lessons.

**Root causes (hypotheses, not yet verified).**

1. **No dedicated Chief-directives channel.** `strategy.py` reads
   `lessons`, `daily_reviews`, `pair_stats`, `market_state`, and
   `recent_outcomes` — but nothing labelled "recent Chief directives."
   Chief memos decay into ambient context with no expiry/freshness
   signal.
2. **Possible hallucinated citations.** The specific timestamps Strategy
   quoted (10:38, 10:54, 11:09, 11:25, 11:40) may not correspond to
   actual stored directives. LLMs fabricate precise-looking anchors to
   justify decisions. Needs verification against brain state at those
   timestamps.
3. **Lesson propagation lag.** Learning wrote at 12:03:13; Strategy ran
   at 12:03:19. Either cache timing or lesson-confidence ranking kept
   the fresh lessons out of Strategy's top-40 slice.
4. **Strategy cannot proactively propose.** Chief said "propose USD_CHF
   short on pullback." Strategy only evaluates fired models — it has no
   path to originate a trade from a Chief directive alone.

**Initial decision (later reversed — see addendum below).** Do not
intervene during the 48h window. The bot is not losing money, just
missing edge. Injecting a hand-written lesson or hot directive would
contaminate the repeat-signal fix evaluation. Missed opportunity this
morning is cheaper than invalidating the observation.

**Queued for post-observation roadmap.** Added as **Wave 0** (ahead of
all previously-planned waves):

- W0.1 — Structured directive channel (`data/brain/chief_directives.jsonl`
  with timestamp + expiry). Strategy reads last 5 unexpired at top of
  prompt.
- W0.2 — Directive confirmation. Strategy must follow or explicitly
  override in SKIP reasoning; no silent ignoring.
- W0.3 — Proactive proposal slot. Chief pre-auth (e.g. "USD_CHF short
  ≥0.55") lets Strategy propose without a model fire.
- W0.4 — Lesson freshness boost. Lessons <1h old get a confidence bump
  so they surface in the top 40.

**Open verification items (for post-48h).**

- Grep brain files for the 5 cited directive timestamps — confirm
  whether they exist or were hallucinated.
- Check Strategy prompt length at skip-loop time — is the 11:56 directive
  being truncated before reaching the LLM?
- Count skip reasons over the full 48h window — is "solo Model E,
  no confluence" a repeat pattern across weeks or a Sunday-reopen
  artifact?

### Addendum — 14:00 UTC: hour-edge / risk-tier arithmetic conflict (real root cause)

At 13:54 UTC Strategy **did** produce the directive-backed trade —
USD_CHF short, `conf=0.60`, clean reasoning citing Chief pre-auth,
session edge (3/3 wins), trend alignment, and regime confirmation.
Second Opinion approved with `+0.05`. Risk rejected with
`pair_tier_confidence`.

Tracing the confidence chain revealed an arithmetic dead-zone:

```
Strategy:          conf = 0.60
hour_edge neutral: min(0.60, LOW_CONV_CAP=0.50) = 0.50
second-opinion:    0.50 + 0.05                  = 0.55
Risk min (preferred tier): 0.65  →  REJECTED (0.55 < 0.65)
```

Under the previous `LOW_CONV_CAP = 0.50`, **no trade could be approved
at any of 13 UTC "neutral" hours** (06, 08–18, 21). The maximum
achievable post-filter confidence at a neutral hour was:
`min(any, 0.50) + 0.05 = 0.55`, which falls below every Risk tier
threshold (0.65/0.70/0.80). This was not a Wave C1 side-effect — the
hour_edge_filter neutral-hour cap was set back in the Wave 1b deploy,
and no test existed to catch the interaction with later Risk thresholds.

**Implication for the repeat-signal observation.** The 48h window was
already invalid. We were not measuring whether the repeat-signal fix
suppressed bad clusters — we were measuring a system that couldn't
produce proposals during the busiest trading hours at all. The Wave A–D
fix still needs a valid evaluation window; this one did not count.

**Fix shipped (2026-04-20 ~14:10 UTC).**

- `src/scalp_mode/engine/hour_edge_filter.py`: neutral-hour branch now
  returns `adjusted_conf = model_conf` (pass-through, no cap). Edge-hour
  boost, counter-edge shrink, and dead-hour block are unchanged.
- `tests/test_hour_edge_filter.py`: new file, 43 tests covering all
  hours × both directions, including an explicit regression for the
  13:54 USD_CHF short scenario.
- Deploy: rebuild the Docker container.

**New observation clock.** 48h window restarts from the redeploy
timestamp. Previous Wave 0 items (directive channel, lesson freshness,
proactive proposal) remain queued — they are still real problems — but
none of them were the primary blocker today.

---
