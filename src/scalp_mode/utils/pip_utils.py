"""Pip normalization utilities.

All pip calculations in the system MUST go through these functions.
Direct division by 0.0001 is forbidden per the engineering spec (A.2).
"""


def pip_value(pair: str) -> float:
    """Return the pip unit for a currency pair.

    JPY pairs use 0.01, all others use 0.0001.
    """
    if "JPY" in pair.upper():
        return 0.01
    return 0.0001


def pip_value_in_account_ccy(pair: str, account_ccy: str = "USD",
                              rates: dict = None) -> float:
    """Return pip value per unit in account currency.

    For sizing: units = risk_amount / (stop_pips * pip_value_in_account_ccy)

    Args:
        pair: Trading pair (e.g. "EUR_USD")
        account_ccy: Account currency ("GBP", "USD", etc.)
        rates: Dict of live rates, e.g. {"GBP_USD": 1.25, "USD_JPY": 150.0}
               Required when account_ccy != quote currency of the pair.

    Returns:
        Pip value per 1 unit in account currency.

    Examples:
        GBP account, EUR_USD: 0.0001 / 1.25 = 0.00008 GBP per unit per pip
        GBP account, USD_JPY: 0.01 / (150 * 1.25) = 0.0000533 GBP per unit per pip
        USD account, EUR_USD: 0.0001 USD per unit per pip (no conversion)
    """
    pv = pip_value(pair)

    if account_ccy == "USD":
        # JPY-quoted (e.g. USD_JPY): pip in JPY → USD
        if "JPY" in pair.upper():
            if rates and "USD_JPY" in rates and rates["USD_JPY"] > 0:
                return pv / rates["USD_JPY"]
            return pv
        # GBP-quoted (e.g. EUR_GBP): pip in GBP → USD
        if pair.upper().endswith("_GBP") and not pair.upper().startswith("GBP_"):
            if rates and "GBP_USD" in rates and rates["GBP_USD"] > 0:
                return pv * rates["GBP_USD"]
            return pv * 1.25
        # USD-quoted majors (EUR_USD, GBP_USD, AUD_USD, …): pip already in USD
        return pv

    if account_ccy == "GBP":
        if rates and "GBP_USD" in rates and rates["GBP_USD"] > 0:
            gbp_usd = rates["GBP_USD"]
            if "JPY" in pair.upper():
                # JPY-quoted: pip in JPY → convert to GBP via GBP_JPY
                usd_jpy = rates.get("USD_JPY", 150.0)
                gbp_jpy = gbp_usd * usd_jpy
                return pv / gbp_jpy if gbp_jpy > 0 else pv
            else:
                # USD-quoted: pip in USD → convert to GBP
                return pv / gbp_usd
        # Fallback: approximate GBP/USD = 1.25
        if "JPY" in pair.upper():
            return pv / (1.25 * 150.0)
        return pv / 1.25

    # Unknown account currency — return raw pip value
    return pv


def price_to_pips(price_diff: float, pair: str) -> float:
    """Convert a price difference to pips."""
    return price_diff / pip_value(pair)


def pips_to_price(pips: float, pair: str) -> float:
    """Convert pips to a price difference."""
    return pips * pip_value(pair)


def round_price(price: float, pair: str) -> float:
    """Round a price to the appropriate decimal places for the pair.

    JPY pairs: 3 decimals (pipettes), others: 5 decimals.
    """
    decimals = 3 if "JPY" in pair.upper() else 5
    return round(price, decimals)
