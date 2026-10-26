import numpy as np
import pandas as pd
import pytest
from opx.evaluate import select_trades, summarize


@pytest.fixture
def oos_three_rows():
    t0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    times = [t0, t0 + pd.Timedelta(minutes=5), t0 + pd.Timedelta(minutes=10)]
    t1s = [t + pd.Timedelta(minutes=2) for t in times]
    return pd.DataFrame(
        {
            "time": times,
            "side": ["call", "put", "call"],
            "p_cal": [0.53, 0.55, 0.70],
            "y": [0.0, 1.0, 1.0],
            "t1": t1s,
        }
    )


@pytest.fixture
def two_overlapping():
    t0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    t1 = t0 + pd.Timedelta(minutes=3)
    # k=5: t1 = time + 6 min
    return pd.DataFrame(
        {
            "time": [t0, t1],
            "side": ["call", "call"],
            "p_cal": [0.65, 0.65],
            "y": [1.0, 1.0],
            "t1": [t0 + pd.Timedelta(minutes=6), t1 + pd.Timedelta(minutes=6)],
        }
    )


@pytest.fixture
def two_adjacent():
    t0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    t1 = t0 + pd.Timedelta(minutes=5)
    # entry for t1 is 00:06, which equals previous t1 (00:06)
    return pd.DataFrame(
        {
            "time": [t0, t1],
            "side": ["call", "call"],
            "p_cal": [0.65, 0.65],
            "y": [1.0, 1.0],
            "t1": [t0 + pd.Timedelta(minutes=6), t1 + pd.Timedelta(minutes=6)],
        }
    )


@pytest.fixture
def trades_6_wins_of_10():
    times = pd.date_range("2024-01-01 00:00", periods=10, freq="10min", tz="UTC")
    y = [1.0] * 6 + [0.0] * 4
    return pd.DataFrame(
        {
            "time": times,
            "side": ["call"] * 10,
            "p_cal": [0.60] * 10,
            "y": y,
            "t1": [t + pd.Timedelta(minutes=2) for t in times],
        }
    )


def test_threshold_filters(oos_three_rows):
    t = select_trades(oos_three_rows, r=0.85, buffer=0.0)  # p_cal = 0.53, 0.55, 0.70
    assert t["p_cal"].tolist() == [0.55, 0.70]


def test_overlapping_trades_dropped(two_overlapping, two_adjacent):
    assert len(select_trades(two_overlapping, r=0.85, buffer=0.0)) == 1
    assert len(select_trades(two_adjacent, r=0.85, buffer=0.0)) == 2


def test_summarize(trades_6_wins_of_10):
    s = summarize(trades_6_wins_of_10, r=0.85)
    assert s["n"] == 10 and s["win_rate"] == 0.6
    assert s["ev"] == pytest.approx(0.6 * 0.85 - 0.4)
    assert s["breakeven"] == pytest.approx(1 / 1.85)


def test_empty_selection_is_not_an_error(oos_three_rows):
    s = summarize(select_trades(oos_three_rows, r=0.85, buffer=0.9), r=0.85)
    assert s["n"] == 0 and np.isnan(s["lcb"])
