"""Spot FX Trading & Backtesting Engine with Real Point-in-Time Broker Spread.

Supports Triple-Barrier labeling (TP/SL/Time), dynamic ATR barriers,
trailing stops, and real spread deduction for spot forex.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def resample_m1_to_timeframe(df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """Resample M1 DataFrame to higher timeframe (e.g. 5min, 15min, 1h)."""
    df_indexed = df.set_index("time")
    resampled = df_indexed.resample(timeframe).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "spread": "mean",
        }
    ).dropna(subset=["close"]).reset_index()

    return resampled


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR) in price units."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = pd.Series(tr, index=df.index).rolling(window=period, min_periods=period).mean()
    return atr


def compute_triple_barrier(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    atr: float,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    max_holding_bars: int = 24,
    point_size: float = 1e-5,
    trailing_be: bool = False,
    fixed_pips: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute Triple-Barrier outcome for an individual trade with real spread.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + spread dataframe.
    entry_idx : int
        Index of the bar when signal occurs. Trade executes at entry_idx + 1 open.
    side : str
        'buy' or 'sell'.
    atr : float
        ATR value at entry.
    tp_atr_mult : float
        Take-profit multiplier on ATR.
    sl_atr_mult : float
        Stop-loss multiplier on ATR.
    max_holding_bars : int
        Max holding horizon before forced market exit.
    point_size : float
        Symbol point size (e.g. 1e-5 for EURUSD/GBPUSD, 1e-3 for USDJPY/DXY).
    trailing_be : bool
        If True, moves SL to breakeven once price reaches +1R profit.
    fixed_pips : tuple[float, float] or None
        (tp_pips, sl_pips) if using fixed pip barriers instead of ATR.
    """
    n = len(df)
    exec_idx = entry_idx + 1 if entry_idx + 1 < n else entry_idx

    # Broker spread at entry in price units
    entry_spread_pts = df["spread"].iloc[exec_idx] if "spread" in df.columns else 8.0
    entry_spread = entry_spread_pts * point_size

    pip_size = 10.0 * point_size

    if fixed_pips is not None:
        reward_dist = fixed_pips[0] * pip_size
        risk_dist = fixed_pips[1] * pip_size
    else:
        reward_dist = tp_atr_mult * atr
        risk_dist = sl_atr_mult * atr

    # Ensure risk_dist is strictly positive
    if risk_dist <= 1e-12:
        risk_dist = 10.0 * pip_size

    # Execution prices
    raw_open = df["open"].iloc[exec_idx]
    if side.lower() == "buy":
        entry_price = raw_open + entry_spread  # Buy at Ask
        tp_level = entry_price + reward_dist
        sl_level = entry_price - risk_dist
    else:
        entry_price = raw_open  # Sell at Bid
        tp_level = entry_price - reward_dist
        sl_level = entry_price + risk_dist

    hit_tp = False
    hit_sl = False
    bars_held = 0
    exit_price = entry_price
    pnl_r = 0.0

    end_idx = min(exec_idx + max_holding_bars, n)
    curr_sl = sl_level

    for t in range(exec_idx, end_idx):
        bars_held += 1
        h = df["high"].iloc[t]
        l = df["low"].iloc[t]
        c = df["close"].iloc[t]
        spr = (df["spread"].iloc[t] if "spread" in df.columns else 8.0) * point_size

        if side.lower() == "buy":
            # Check Trailing Breakeven
            if trailing_be and h >= entry_price + risk_dist:
                curr_sl = max(curr_sl, entry_price)

            # Check SL and TP (Bid prices)
            # Conservative rule: if both touched in same candle, SL takes priority
            if l <= curr_sl and h >= tp_level:
                hit_sl = True
                exit_price = curr_sl
                break
            elif l <= curr_sl:
                hit_sl = True
                exit_price = curr_sl
                break
            elif h >= tp_level:
                hit_tp = True
                exit_price = tp_level
                break
        else:
            # Sell position: Exit at Ask = Bid + spread
            high_ask = h + spr
            low_ask = l + spr

            if trailing_be and low_ask <= entry_price - risk_dist:
                curr_sl = min(curr_sl, entry_price)

            if high_ask >= curr_sl and low_ask <= tp_level:
                hit_sl = True
                exit_price = curr_sl
                break
            elif high_ask >= curr_sl:
                hit_sl = True
                exit_price = curr_sl
                break
            elif low_ask <= tp_level:
                hit_tp = True
                exit_price = tp_level
                break

    # If neither TP nor SL hit, time barrier exit at close
    if not (hit_tp or hit_sl):
        last_t = end_idx - 1
        last_spr = (df["spread"].iloc[last_t] if "spread" in df.columns else 8.0) * point_size
        exit_price = df["close"].iloc[last_t] if side.lower() == "buy" else df["close"].iloc[last_t] + last_spr

    # Calculate final PnL
    if side.lower() == "buy":
        raw_pnl = exit_price - entry_price
    else:
        raw_pnl = entry_price - exit_price

    pnl_pips = raw_pnl / pip_size
    pnl_r = raw_pnl / risk_dist

    return {
        "hit_tp": hit_tp,
        "hit_sl": hit_sl,
        "pnl_r": float(pnl_r),
        "pnl_pips": float(pnl_pips),
        "bars_held": bars_held,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
    }
