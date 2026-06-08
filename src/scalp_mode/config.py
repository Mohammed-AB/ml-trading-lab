"""Configuration loader for Scalp Mode V1.

Loads settings from YAML, resolves environment variables for secrets,
and provides typed access to all configuration values.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${VAR} references in string values."""
    if isinstance(value, str):
        def _replace(match):
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                raise EnvironmentError(
                    f"Environment variable '{var_name}' is not set. "
                    f"See .env.example for required variables."
                )
            return env_val
        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


class Config:
    """Typed configuration wrapper around the YAML settings."""

    def __init__(self, path: Path | str | None = None, resolve_env: bool = True):
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if resolve_env:
            self._data = _resolve_env_vars(raw)
        else:
            self._data = raw

        self._scalp = self._data["scalp_mode"]

    # --- Top-level access ---

    @property
    def raw(self) -> dict:
        return self._data

    @property
    def scalp(self) -> dict:
        return self._scalp

    # --- OANDA ---

    @property
    def oanda_account_id(self) -> str:
        return self._data["oanda"]["account_id"]

    @property
    def oanda_api_token(self) -> str:
        return self._data["oanda"]["api_token"]

    @property
    def oanda_base_url(self) -> str:
        return self._data["oanda"]["base_url"]

    @property
    def oanda_stream_url(self) -> str:
        return self._data["oanda"]["stream_url"]

    # --- Instruments ---

    @property
    def instruments(self) -> list[str]:
        return self._scalp["instruments"]

    # --- Costs ---

    def max_spread_pips(self, pair: str) -> float:
        return self._scalp["costs"]["max_spread_pips"][pair]

    @property
    def max_slippage(self) -> float:
        return self._scalp["costs"]["max_slippage"]

    # --- Risk ---

    @property
    def risk(self) -> dict:
        return self._scalp["risk"]

    # --- Regime ---

    @property
    def regime(self) -> dict:
        return self._scalp["regime"]

    # --- Model A ---

    @property
    def model_a(self) -> dict:
        return self._scalp["model_a"]

    # --- Sessions ---

    @property
    def sessions(self) -> dict:
        return self._scalp["sessions"]

    # --- Orders ---

    @property
    def orders(self) -> dict:
        return self._scalp["orders"]

    # --- Data Quality ---

    @property
    def data_quality(self) -> dict:
        return self._scalp["data_quality"]

    # --- Borderline ---

    @property
    def borderline(self) -> dict:
        return self._scalp["borderline"]

    # --- Logging ---

    @property
    def logging_config(self) -> dict:
        return self._data["logging"]

    # --- Generic getter ---

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Access nested config via dot notation: 'scalp_mode.risk.risk_pct'."""
        keys = dotted_key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val
