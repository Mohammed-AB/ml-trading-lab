"""Load LightGBM long/short models and score live M1 windows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .bar_features import FEATURE_COLUMNS, add_ml_features

_log = logging.getLogger("scalp_mode")


def _top_feature_summary(row: Any, k: int = 8) -> str:
    """Human-readable snippet: largest |value| features on last bar."""
    pairs = []
    for name in FEATURE_COLUMNS:
        try:
            v = float(row[name])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        pairs.append((abs(v), name, v))
    pairs.sort(key=lambda x: -x[0])
    parts = [f"{n}={val:+.3g}" for _, n, val in pairs[:k]]
    return ", ".join(parts)


class MLGate:
    """ML probability gate: replaces rule-model signal generation when enabled."""

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self._enabled = bool(cfg.get("enabled", True))
        ml_dir = Path(cfg.get("model_dir", "data/ml"))
        self._threshold = float(cfg.get("probability_threshold", 0.65))
        self._min_m1_bars = int(cfg.get("min_m1_bars", 120))
        long_path = ml_dir / "model_long.txt"
        short_path = ml_dir / "model_short.txt"
        if not long_path.exists() or not short_path.exists():
            raise FileNotFoundError(
                f"ML models not found under {ml_dir} "
                f"(expected model_long.txt, model_short.txt). Run ml_train.py."
            )
        import lightgbm as lgb

        self._booster_long = lgb.Booster(model_file=str(long_path))
        self._booster_short = lgb.Booster(model_file=str(short_path))
        _log.info(
            f"MLGate loaded from {ml_dir} "
            f"(threshold={self._threshold}, min_bars={self._min_m1_bars})"
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    def score_pair(self, pair: str, df_m1) -> tuple[float, float, str]:
        """Returns (p_long, p_short, feature_summary)."""
        if df_m1 is None or len(df_m1) < self._min_m1_bars:
            return 0.0, 0.0, ""
        tail = df_m1.tail(max(400, self._min_m1_bars + 60))
        d = add_ml_features(tail, pair)
        row = d.iloc[-1]
        X = np.array([[float(row[c]) for c in FEATURE_COLUMNS]], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pl = float(self._booster_long.predict(X)[0])
        ps = float(self._booster_short.predict(X)[0])
        summary = _top_feature_summary(row)
        return pl, ps, summary

    def best_signal(
        self,
        instruments: list[str],
        candle_data: dict[str, Any],
    ) -> Optional[tuple[str, str, float, float, float, str]]:
        """Best (pair, direction, chosen_prob, p_long, p_short, feature_summary).

        direction is 'long' or 'short'. Returns None if nothing above threshold.
        """
        if not self._enabled:
            return None
        best: tuple[str, str, float, float, float, str] | None = None
        best_score = 0.0
        top_scores: list[str] = []
        for pair in instruments:
            cd = candle_data.get(pair) or {}
            df_m1 = cd.get("m1")
            pl, ps, summ = self.score_pair(pair, df_m1)
            hi = max(pl, ps)
            d = "L" if pl >= ps else "S"
            top_scores.append(f"{pair}:{d}{hi:.3f}")
            if pl >= ps and pl >= self._threshold and pl >= best_score:
                best = (pair, "long", pl, pl, ps, summ)
                best_score = pl
            elif ps > pl and ps >= self._threshold and ps >= best_score:
                best = (pair, "short", ps, pl, ps, summ)
                best_score = ps
        if not best:
            now = __import__("time").time()
            if not hasattr(self, "_last_score_log") or now - self._last_score_log > 900:
                _log.info("ML top scores: %s", " | ".join(top_scores))
                self._last_score_log = now
        return best
