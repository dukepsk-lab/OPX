import numpy as np
import pandas as pd
import pytest
from opx.features_extra import (
    compute_vol_squeeze,
    compute_fracdiff,
    compute_dxy_features,
    compute_cross_returns,
)


@pytest.fixture
def sample_bars():
    idx = pd.date_range("2023-01-01 00:00:00", periods=100, freq="1min", tz="UTC")
    # Synthetic trending/fluctuating price
    np.random.seed(42)
    steps = np.random.normal(0, 0.0002, len(idx))
    close = 1.2000 + np.cumsum(steps)
    high = close + np.random.uniform(0.0001, 0.0005, len(idx))
    low = close - np.random.uniform(0.0001, 0.0005, len(idx))
    open_p = close - np.random.uniform(-0.0002, 0.0002, len(idx))
    vol = np.random.uniform(10, 50, len(idx))
    return pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "tick_volume": vol},
        index=idx,
    )


def test_vol_squeeze_features(sample_bars):
    df = compute_vol_squeeze(sample_bars)
    assert "bb_squeeze" in df.columns
    assert "vol_ratio_5_60" in df.columns
    assert len(df) == len(sample_bars)
    assert not np.isinf(df.to_numpy()).any()


def test_fracdiff_causality():
    idx = pd.date_range("2023-01-01 00:00:00", periods=50, freq="1min", tz="UTC")
    np.random.seed(42)
    s = pd.Series(100.0 + np.cumsum(np.random.randn(50)), index=idx)
    fd = compute_fracdiff(s, d=0.35)
    assert len(fd) == len(s)
    assert not np.isinf(fd.to_numpy()).any()
    # Check causality: modifying future value s.iloc[40] should NOT change fd.iloc[20]
    s_mod = s.copy()
    s_mod.iloc[40] += 50.0
    fd_mod = compute_fracdiff(s_mod, d=0.35)
    np.testing.assert_allclose(fd.iloc[:40].dropna(), fd_mod.iloc[:40].dropna())


def test_dxy_features_alignment(sample_bars):
    # DXY bars with matching index
    dxy_close = 103.0 + np.cumsum(np.random.normal(0, 0.01, len(sample_bars)))
    dxy_bars = pd.DataFrame(
        {
            "open": dxy_close,
            "high": dxy_close + 0.02,
            "low": dxy_close - 0.02,
            "close": dxy_close,
            "tick_volume": [20.0] * len(sample_bars),
        },
        index=sample_bars.index,
    )
    dxy_feat = compute_dxy_features(sample_bars.index, dxy_bars)
    assert "dxy_ret_1" in dxy_feat.columns
    assert "dxy_ret_5" in dxy_feat.columns
    assert "dxy_rsi_5" in dxy_feat.columns
    assert len(dxy_feat) == len(sample_bars)


def test_cross_returns_and_composite_usd(sample_bars):
    other_pairs = {
        "eurusd": sample_bars.copy(),
        "usdjpy": sample_bars.copy(),
    }
    cross_feat = compute_cross_returns(sample_bars.index, other_pairs, base_is_usd={"usdjpy": True, "eurusd": False})
    assert "cross_eurusd_ret_1" in cross_feat.columns
    assert "cross_usdjpy_ret_1" in cross_feat.columns
    assert "composite_usd_ret_1" in cross_feat.columns
    assert len(cross_feat) == len(sample_bars)
