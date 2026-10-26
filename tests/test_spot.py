import numpy as np
import pandas as pd
import pytest
from src.opx.spot import (
    compute_triple_barrier,
    resample_m1_to_timeframe,
)


def test_resample_m1_to_timeframe():
    # 10 M1 bars -> 2 M5 bars
    dates = pd.date_range("2024-01-01 00:00:00", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "time": dates,
            "open": [1.1000 + i * 0.0001 for i in range(10)],
            "high": [1.1005 + i * 0.0001 for i in range(10)],
            "low": [1.0995 + i * 0.0001 for i in range(10)],
            "close": [1.1002 + i * 0.0001 for i in range(10)],
            "volume": [10] * 10,
            "spread": [8] * 10,
        }
    )

    df_m5 = resample_m1_to_timeframe(df, timeframe="5min")
    assert len(df_m5) == 2
    assert df_m5["open"].iloc[0] == 1.1000
    assert df_m5["volume"].iloc[0] == 50
    assert df_m5["spread"].iloc[0] == 8.0


def test_compute_triple_barrier_buy_tp():
    # Construct synthetic data where BUY hits TP
    dates = pd.date_range("2024-01-01 00:00:00", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "time": dates,
            "open": [1.1000, 1.1000, 1.1025, 1.1030, 1.1030],
            "high": [1.1005, 1.1035, 1.1040, 1.1035, 1.1035],
            "low": [1.0995, 1.0995, 1.1020, 1.1025, 1.1025],
            "close": [1.1000, 1.1025, 1.1030, 1.1030, 1.1030],
            "volume": [100] * 5,
            "spread": [10] * 5,  # 1.0 pip spread (0.00010)
        }
    )

    # Entry at bar 0: entry_price = ask = 1.1000 + 0.00010 = 1.10010
    # ATR = 0.0010 (10 pips). TP = 2 ATR = 0.0020 (TP price = 1.10210). SL = 1 ATR = 0.0010 (SL price = 1.09910).
    # Bar 1 high is 1.1035 (bid), which is > 1.10210. So it hits TP!
    res = compute_triple_barrier(
        df=df,
        entry_idx=0,
        side="buy",
        atr=0.0010,
        tp_atr_mult=2.0,
        sl_atr_mult=1.0,
        max_holding_bars=4,
        point_size=1e-5,
    )

    assert res["hit_tp"] is True
    assert res["hit_sl"] is False
    assert res["bars_held"] == 1
    assert res["pnl_r"] > 1.8  # ~2R minus spread


def test_compute_triple_barrier_buy_sl():
    # Construct synthetic data where BUY hits SL
    dates = pd.date_range("2024-01-01 00:00:00", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "time": dates,
            "open": [1.1000, 1.1000, 1.0980, 1.0980, 1.0980],
            "high": [1.1005, 1.1005, 1.0985, 1.0985, 1.0985],
            "low": [1.0995, 1.0985, 1.0975, 1.0975, 1.0975],
            "close": [1.1000, 1.0985, 1.0980, 1.0980, 1.0980],
            "volume": [100] * 5,
            "spread": [10] * 5,
        }
    )

    res = compute_triple_barrier(
        df=df,
        entry_idx=0,
        side="buy",
        atr=0.0010,
        tp_atr_mult=2.0,
        sl_atr_mult=1.0,
        max_holding_bars=4,
        point_size=1e-5,
    )

    assert res["hit_sl"] is True
    assert res["hit_tp"] is False
    assert res["pnl_r"] < -0.9
