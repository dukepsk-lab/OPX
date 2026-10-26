import numpy as np
import pandas as pd
import pytest
from opx.features import build_features

EXPECTED_COLUMNS = [
    "body_ratio",
    "upper_wick",
    "lower_wick",
    "clv",
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_15",
    "rsi_3",
    "rsi_5",
    "rsi_14",
    "bb_pctb",
    "atr_pct",
    "tickvol_z",
    "kaufman_er",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "sess_london",
    "sess_ny",
    "sess_overlap",
]


def test_columns(bars_factory):
    bars = bars_factory(500)
    features = build_features(bars)
    assert list(features.columns) == EXPECTED_COLUMNS


def test_no_lookahead(bars_factory):
    bars = bars_factory(800)
    full = build_features(bars)
    part = build_features(bars.iloc[:500])
    pd.testing.assert_frame_equal(full.iloc[:500], part)


def test_flat_bars_never_inf(bars_factory):
    bars = bars_factory(300)
    # Set flat bars in slice
    locs = bars.columns.get_indexer(["open", "high", "low", "close"])
    bars.iloc[100:130, locs] = 1.1
    f = build_features(bars)
    assert not np.isinf(f.to_numpy(dtype=float)).any()


def test_session_flags_follow_local_time_across_dst(bars_factory):
    # 2024-01-15 08:00 UTC (winter) is 08:00 London (UTC+0)
    # 2024-07-15 07:00 UTC (summer) is 08:00 London (BST = UTC+1)
    winter_bars = bars_factory(300, start="2024-01-15 07:00")
    summer_bars = bars_factory(300, start="2024-07-15 06:00")

    f_winter = build_features(winter_bars)
    f_summer = build_features(summer_bars)

    winter_ts = pd.Timestamp("2024-01-15 08:00", tz="UTC")
    summer_ts = pd.Timestamp("2024-07-15 07:00", tz="UTC")

    assert f_winter.loc[winter_ts, "sess_london"] == 1
    assert f_summer.loc[summer_ts, "sess_london"] == 1
