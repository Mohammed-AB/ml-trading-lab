#!/usr/bin/env python3
"""Binary rule-signal features for ML V2 (M5 strategies broadcast to M1 rows).

Each non-S15 book strategy and each research strategy contributes
``rule_{NAME}_L`` and ``rule_{NAME}_S`` (0/1) on the M1 grid. S15 VWAP uses M1
signals directly. ``rule_agree_L`` / ``rule_agree_S`` count concurrent fires on
the same M5 bar (broadcast to its five M1 rows).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_strategies import (  # noqa: E402
    STRATEGIES,
    add_indicators,
    resample_m5,
)
from scalp_mode.ml.bar_features import (  # noqa: E402
    SPREAD_PIPS_DEFAULT,
    pip_for_pair,
)


def _m5_specs():
    out: list[tuple[str, object, int]] = []
    for name, fn, mb in STRATEGIES:
        if name == "S15_VWAP":
            continue
        out.append((name, fn, mb))
    from strategy_arena.research import RESEARCH_STRATEGIES  # noqa: WPS433
    from strategy_arena.research_pdf import RESEARCH_PDF_STRATEGIES  # noqa: WPS433
    from strategy_arena.research_v2 import RESEARCH_V2_STRATEGIES  # noqa: WPS433

    out.extend(RESEARCH_STRATEGIES)
    out.extend(RESEARCH_V2_STRATEGIES)
    out.extend(RESEARCH_PDF_STRATEGIES)
    return out


def rule_m5_column_names() -> list[str]:
    names: list[str] = []
    for name, _, _ in _m5_specs():
        names.append(f"rule_{name}_L")
        names.append(f"rule_{name}_S")
    names.extend(["rule_S15_VWAP_L", "rule_S15_VWAP_S", "rule_agree_L", "rule_agree_S"])
    return names


RULE_M5_COLUMN_NAMES = rule_m5_column_names()


def build_rule_signal_features_for_m1(
    m1: pd.DataFrame,
    pair: str,
) -> pd.DataFrame:
    """Return columns aligned to ``m1`` index (length ``len(m1)``), float32 0/1."""
    pip = pip_for_pair(pair)
    spread_full = SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip
    n = len(m1)
    groups = n // 5
    if groups < 1:
        empty = {c: np.zeros(n, dtype=np.float32) for c in RULE_M5_COLUMN_NAMES}
        return pd.DataFrame(empty, index=m1.index)

    m1t = m1.iloc[: groups * 5].copy().reset_index(drop=True)
    m5 = add_indicators(resample_m5(m1t), pair)
    n5 = len(m5)
    n_m5_strat = len(_m5_specs())
    mat = np.zeros((n5, 2 * n_m5_strat), dtype=np.float32)

    for si, (name, fn, mb) in enumerate(_m5_specs()):
        try:
            idx, dirs, _, _, _ = fn(m5, pair, pip, spread_full)
        except Exception:
            continue
        for k in range(len(idx)):
            g = int(idx[k])
            if g < 0 or g >= n5:
                continue
            if int(dirs[k]) == 1:
                mat[g, 2 * si] = 1.0
            else:
                mat[g, 2 * si + 1] = 1.0

    agree_l_m5 = mat[:, 0::2].sum(axis=1).astype(np.float32)
    agree_s_m5 = mat[:, 1::2].sum(axis=1).astype(np.float32)
    bc = np.repeat(mat, 5, axis=0)
    agree_l_m1 = np.repeat(agree_l_m5, 5).astype(np.float32)
    agree_s_m1 = np.repeat(agree_s_m5, 5).astype(np.float32)

    s15_l_m5 = np.zeros(n5, dtype=np.float32)
    s15_s_m5 = np.zeros(n5, dtype=np.float32)
    s15_fn = next(s for s in STRATEGIES if s[0] == "S15_VWAP")[1]
    try:
        res = s15_fn(m1t, pair, pip, spread_full)
        idx, dirs = res[0], res[1]
        for k in range(len(idx)):
            g = int(idx[k])
            if 0 <= g < n5:
                if int(dirs[k]) == 1:
                    s15_l_m5[g] = 1.0
                else:
                    s15_s_m5[g] = 1.0
    except Exception:
        pass
    s15_l = np.repeat(s15_l_m5, 5).astype(np.float32)
    s15_s = np.repeat(s15_s_m5, 5).astype(np.float32)

    cols: dict[str, np.ndarray] = {}
    ci = 0
    for name, _, _ in _m5_specs():
        cols[f"rule_{name}_L"] = bc[:, ci].copy()
        cols[f"rule_{name}_S"] = bc[:, ci + 1].copy()
        ci += 2
    cols["rule_S15_VWAP_L"] = s15_l
    cols["rule_S15_VWAP_S"] = s15_s
    cols["rule_agree_L"] = agree_l_m1
    cols["rule_agree_S"] = agree_s_m1

    out = pd.DataFrame(cols, index=m1t.index)
    out = out.reindex(m1.index, fill_value=0.0).astype(np.float32)
    return out


def feature_columns_v2() -> list[str]:
    from scalp_mode.ml.bar_features import FEATURE_COLUMNS  # noqa: WPS433

    return list(FEATURE_COLUMNS) + RULE_M5_COLUMN_NAMES
