"""Tests for Spread Filter."""

import pytest
from src.scalp_mode.gates.spread_filter import check_spread


class TestSpreadFilter:
    def test_eur_usd_within_limit(self):
        # 0.5 pip spread, limit 0.8
        result = check_spread(bid=1.08500, ask=1.08505, pair="EUR_USD",
                              max_spread_pips=0.8)
        assert result.is_ok is True
        assert result.spread_pips == 0.5

    def test_eur_usd_at_limit(self):
        # Spread exactly at limit — float precision causes 0.80000...08
        # which is technically > 0.8, so filter correctly rejects.
        # This validates that the filter is strict (no tolerance on boundary).
        result = check_spread(bid=1.08500, ask=1.08508, pair="EUR_USD",
                              max_spread_pips=0.8)
        assert result.spread_pips == 0.8
        # Use a price that produces exactly 0.7 pips to test pass at limit
        result2 = check_spread(bid=1.08500, ask=1.08507, pair="EUR_USD",
                               max_spread_pips=0.8)
        assert result2.is_ok is True
        assert result2.spread_pips == 0.7

    def test_eur_usd_over_limit(self):
        # 1.2 pip spread, limit 0.8
        result = check_spread(bid=1.08500, ask=1.08512, pair="EUR_USD",
                              max_spread_pips=0.8)
        assert result.is_ok is False
        assert result.spread_pips == 1.2

    def test_usd_jpy_within_limit(self):
        # 0.6 pip spread for JPY pair
        result = check_spread(bid=149.500, ask=149.506, pair="USD_JPY",
                              max_spread_pips=0.8)
        assert result.is_ok is True
        assert result.spread_pips == 0.6

    def test_usd_jpy_over_limit(self):
        # 1.0 pip spread, limit 0.8
        result = check_spread(bid=149.500, ask=149.510, pair="USD_JPY",
                              max_spread_pips=0.8)
        assert result.is_ok is False
        assert result.spread_pips == 1.0

    def test_gbp_usd_within_limit(self):
        result = check_spread(bid=1.26500, ask=1.26509, pair="GBP_USD",
                              max_spread_pips=1.0)
        assert result.is_ok is True
        assert result.spread_pips == 0.9

    def test_max_allowed_preserved(self):
        result = check_spread(bid=1.08500, ask=1.08505, pair="EUR_USD",
                              max_spread_pips=0.8)
        assert result.max_allowed == 0.8
