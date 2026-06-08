"""Second Opinion Agent — Technical + News double-check.

Called by the Strategy Agent on borderline setups (confidence < 0.70).
Fast review that returns one of: APPROVE, REJECT, or BOOST (stronger than
model thought) along with a short rationale.

Combines two roles:
- Technical: is the setup clean from a structure standpoint? (support/resistance,
  H1/D1 alignment, no obvious invalidation right above/below entry)
- News: is there a recent headline or calendar event that makes this risky?

Keeps it to ONE Claude call per proposal to save tokens.
"""
import logging
from datetime import datetime
from typing import Optional

from .brain import Brain

_log = logging.getLogger("scalp_mode")


class SecondOpinionAgent:
    """Fast double-check on borderline proposals."""

    def __init__(self, brain: Brain, model: str = "claude-opus-4-20250514"):
        self._brain = brain
        self._model = model

    def review(self, pair: str, direction: str, entry: float,
               sl: float, tp: float, confidence: float,
               strategy_reasoning: str, pair_context: dict,
               utc_now: datetime) -> Optional[dict]:
        """Return {verdict: approve|reject|boost, reason: str} or None on error."""

        mtf = pair_context.get("mtf_context", "")
        news_risk = pair_context.get("news_elevated_risk", "")
        regime = pair_context.get("regime", "?")
        spread = pair_context.get("spread_pips", 0)

        market_state = self._brain.read_market_state()
        pair_brief = (market_state.get("per_pair_briefs") or {}).get(pair, {})
        pair_brief_text = ""
        if pair_brief:
            pair_brief_text = (
                f"Research bias: {pair_brief.get('bias', '?')} "
                f"(conviction {pair_brief.get('conviction', '?')}) — "
                f"{pair_brief.get('key_driver', '')[:150]}")

        prompt = (
            f"SECOND OPINION on a borderline trade proposal.\n\n"
            f"Time: {utc_now.strftime('%H:%M UTC %A')}\n"
            f"Proposal: {pair} {direction} entry={entry:.5f} "
            f"SL={sl:.5f} TP={tp:.5f} (conf={confidence:.2f})\n"
            f"Strategy Agent's reasoning: {strategy_reasoning[:250]}\n\n"
            f"Context:\n"
            f"- Regime: {regime}\n"
            f"- MTF: {mtf}\n"
            f"- Spread: {spread:.1f} pips\n"
            f"- News elevated-risk window: {news_risk or 'none'}\n"
            f"- {pair_brief_text or 'No research brief'}\n\n"
            f"Evaluate:\n"
            f"1. Technical: is the entry level clean? Any obvious S/R "
            f"  right above/below? Does the direction align with H1/D1?\n"
            f"2. News: does recent macro context support or contradict this?\n"
            f"3. Overall conviction: does this deserve the original confidence, "
            f"  higher, or lower?\n\n"
            f"Respond in EXACTLY this format on one line each:\n"
            f"VERDICT: APPROVE | BOOST | REJECT\n"
            f"CONFIDENCE_ADJUST: <signed delta like +0.10 or -0.20>\n"
            f"REASON: <one short sentence>\n"
        )

        try:
            import anthropic
            import httpx
            client = anthropic.Anthropic(
                timeout=httpx.Timeout(30.0, connect=10.0))
            response = client.messages.create(
                model=self._model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            _log.info(f"Second-opinion [{pair} {direction}]:\n{text}")
            return self._parse(text)
        except Exception as e:
            _log.warning(f"Second-opinion error: {e}")
            return None

    def _parse(self, text: str) -> dict:
        out = {"verdict": "approve", "adjust": 0.0, "reason": ""}
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().lower()
                if "reject" in v:
                    out["verdict"] = "reject"
                elif "boost" in v:
                    out["verdict"] = "boost"
                else:
                    out["verdict"] = "approve"
            elif line.upper().startswith("CONFIDENCE_ADJUST:"):
                try:
                    raw = line.split(":", 1)[1].strip()
                    out["adjust"] = float(raw.replace("+", ""))
                except (ValueError, IndexError):
                    out["adjust"] = 0.0
            elif line.upper().startswith("REASON:"):
                out["reason"] = line.split(":", 1)[1].strip()
        return out
