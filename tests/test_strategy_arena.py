"""Strategy arena: synthetic OANDA CSV smoke test."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _write_synthetic_pfx(path: Path, n: int, seed: int = 42) -> None:
    np.random.seed(seed)
    t0 = pd.Timestamp("2025-10-01 00:00:00", tz="UTC")
    ts = [t0 + pd.Timedelta(minutes=i) for i in range(n)]
    p = 1.0
    rets = np.random.normal(0, 0.00004, n)
    close = 1.1 + np.cumsum(rets)
    open_ = np.roll(close, 1)
    open_[0] = 1.1
    r = 0.0001
    high = np.maximum(open_, close) + r * np.random.random(n)
    low = np.minimum(open_, close) - r * np.random.random(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [t.isoformat().replace("+00:00", "Z") for t in ts],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n, dtype=int),
        }
    ).to_csv(path, index=False)


def test_arena_runs_synthetic() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_synthetic_pfx(d / "EUR_USD_M1_12m.csv", 3000)
        from strategy_arena.runner import run_arena  # noqa: WPS433

        res = run_arena(d, oos=False, only_research=True)
        assert "error" not in res
        assert "summary" in res
        assert "strategies" in res["summary"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
