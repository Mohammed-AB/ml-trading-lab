"""Barbell sizing — optional 2-bucket position sizer.

Disabled by default (risk.sizing_mode != 'barbell'). When enabled:
- SAFE bucket: 70% of NAV, risks 1% per trade
- YOLO bucket: 30% of NAV, risks 15% per trade
- Model A/B route to SAFE; Model C/D route to YOLO

Backtest result (26y, 1,341 rolling 60d windows): 0% blow-up,
4.4% probability of hitting $10k from $500 in 60 days.

The existing Risk Agent guardrails still apply AFTER this sizer
computes the intended risk, so nothing reckless gets through.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import json
import os
from pathlib import Path


Bucket = Literal["safe", "yolo"]


@dataclass
class BarbellConfig:
    safe_risk_pct: float = 0.010
    yolo_risk_pct: float = 0.15
    yolo_fraction: float = 0.30
    min_trade_usd: float = 3.0
    rescue_floor_usd: float = 50.0
    model_bucket_map: dict = field(default_factory=lambda: {
        "model_a": "safe",
        "model_b": "safe",
        "model_c": "yolo",
        "model_d": "yolo",
    })


@dataclass
class BucketState:
    safe_balance: float
    yolo_balance: float
    initialized: bool = False

    @property
    def total(self) -> float:
        return self.safe_balance + self.yolo_balance


def initialize_buckets(nav: float, cfg: BarbellConfig) -> BucketState:
    return BucketState(
        safe_balance=nav * (1 - cfg.yolo_fraction),
        yolo_balance=nav * cfg.yolo_fraction,
        initialized=True,
    )


@dataclass
class SizingResult:
    units: float
    risk_usd: float
    bucket: Bucket
    bucket_balance_before: float
    blocked: bool
    reason: str


def route_signal_to_bucket(model_name: str, cfg: BarbellConfig) -> Bucket:
    return cfg.model_bucket_map.get(model_name, "safe")


def barbell_size(
    state: BucketState,
    cfg: BarbellConfig,
    bucket: Bucket,
    sl_pips: float,
    pip_value_per_unit: float,
) -> SizingResult:
    balance = state.safe_balance if bucket == "safe" else state.yolo_balance
    if balance < cfg.rescue_floor_usd:
        return SizingResult(
            units=0, risk_usd=0, bucket=bucket,
            bucket_balance_before=balance, blocked=True,
            reason=f"{bucket} bucket below rescue floor ${cfg.rescue_floor_usd:.0f}")
    risk_pct = cfg.safe_risk_pct if bucket == "safe" else cfg.yolo_risk_pct
    risk_usd = balance * risk_pct
    if risk_usd < cfg.min_trade_usd:
        return SizingResult(
            units=0, risk_usd=0, bucket=bucket,
            bucket_balance_before=balance, blocked=True,
            reason=f"computed risk ${risk_usd:.2f} below min ${cfg.min_trade_usd}")
    denom = sl_pips * pip_value_per_unit
    if denom <= 0:
        return SizingResult(
            units=0, risk_usd=0, bucket=bucket,
            bucket_balance_before=balance, blocked=True,
            reason="invalid sl_pips or pip_value")
    units = risk_usd / denom
    return SizingResult(
        units=units, risk_usd=risk_usd, bucket=bucket,
        bucket_balance_before=balance, blocked=False,
        reason=(f"{bucket} bucket ${balance:.2f} * {risk_pct:.1%} = "
                f"${risk_usd:.2f} risk"))


def apply_trade_pnl(state: BucketState, bucket: Bucket,
                    pnl_usd: float) -> None:
    if bucket == "safe":
        state.safe_balance += pnl_usd
    else:
        state.yolo_balance += pnl_usd


def save_state(state: BucketState, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "safe_balance": state.safe_balance,
            "yolo_balance": state.yolo_balance,
            "initialized": state.initialized,
        }, f, indent=2)


def load_state(path: str, default_nav: float, cfg: BarbellConfig) -> BucketState:
    if not os.path.exists(path):
        return initialize_buckets(default_nav, cfg)
    with open(path) as f:
        data = json.load(f)
    return BucketState(
        safe_balance=float(data.get("safe_balance", default_nav * 0.7)),
        yolo_balance=float(data.get("yolo_balance", default_nav * 0.3)),
        initialized=bool(data.get("initialized", True)))


def status_line(state: BucketState) -> str:
    total = state.total
    yolo_pct = state.yolo_balance / total * 100 if total else 0
    return (f"BARBELL safe=${state.safe_balance:8.2f} "
            f"yolo=${state.yolo_balance:8.2f} "
            f"total=${total:8.2f} "
            f"yolo_share={yolo_pct:4.1f}%")
