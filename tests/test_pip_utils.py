"""Tests for pip normalization utilities."""

import pytest
from src.scalp_mode.utils.pip_utils import (
    pip_value,
    price_to_pips,
    pips_to_price,
    round_price,
)


class TestPipValue:
    def test_eur_usd(self):
        assert pip_value("EUR_USD") == 0.0001

    def test_gbp_usd(self):
        assert pip_value("GBP_USD") == 0.0001

    def test_usd_jpy(self):
        assert pip_value("USD_JPY") == 0.01

    def test_eur_jpy(self):
        assert pip_value("EUR_JPY") == 0.01

    def test_case_insensitive(self):
        assert pip_value("eur_usd") == 0.0001
        assert pip_value("usd_jpy") == 0.01


class TestPriceToFips:
    def test_eur_usd_1pip(self):
        assert price_to_pips(0.0001, "EUR_USD") == pytest.approx(1.0)

    def test_eur_usd_half_pip(self):
        assert price_to_pips(0.00005, "EUR_USD") == pytest.approx(0.5)

    def test_usd_jpy_1pip(self):
        assert price_to_pips(0.01, "USD_JPY") == pytest.approx(1.0)

    def test_usd_jpy_08pip(self):
        assert price_to_pips(0.008, "USD_JPY") == pytest.approx(0.8)

    def test_negative_diff(self):
        assert price_to_pips(-0.0003, "EUR_USD") == pytest.approx(-3.0)


class TestPipsToPrice:
    def test_eur_usd(self):
        assert pips_to_price(0.8, "EUR_USD") == pytest.approx(0.00008)

    def test_usd_jpy(self):
        assert pips_to_price(0.8, "USD_JPY") == pytest.approx(0.008)

    def test_round_trip(self):
        for pair in ["EUR_USD", "USD_JPY", "GBP_USD"]:
            original_pips = 1.5
            price_diff = pips_to_price(original_pips, pair)
            back_to_pips = price_to_pips(price_diff, pair)
            assert back_to_pips == pytest.approx(original_pips)


class TestRoundPrice:
    def test_eur_usd_5_decimals(self):
        assert round_price(1.08501234, "EUR_USD") == 1.08501

    def test_usd_jpy_3_decimals(self):
        assert round_price(149.5067, "USD_JPY") == 149.507

    def test_gbp_usd(self):
        assert round_price(1.265019999, "GBP_USD") == 1.26502
