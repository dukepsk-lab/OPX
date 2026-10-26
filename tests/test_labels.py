import numpy as np
import pandas as pd
import pytest
from opx.labels import make_labels


@pytest.fixture
def base_bars():
    times = pd.date_range("2024-01-02 00:00", periods=10, freq="1min", tz="UTC")
    # Set known open/close values
    # t=0: open=1.1000, close=1.1000
    # t=1: open=1.1000, close=1.1002 (for t=0, entry=open[1]=1.1000, close[1]=1.1002 -> up)
    # t=2: open=1.1000, close=1.0998 (for t=1, entry=open[2]=1.1000, close[2]=1.0998 -> down)
    # t=3: open=1.1000, close=1.1000 (for t=2, entry=open[3]=1.1000, close[3]=1.1000 -> tie)
    # t=4..9: filler
    opens = [1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1005, 1.1000, 1.1000, 1.1000, 1.1000]
    closes = [1.1000, 1.1002, 1.0998, 1.1000, 1.1000, 1.1008, 1.1000, 1.1000, 1.1000, 1.1000]
    highs = [max(o, c) + 0.0001 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.0001 for o, c in zip(opens, closes)]
    vols = [100.0] * 10
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "tick_volume": vols},
        index=times,
    )


def test_call_put_and_tie(base_bars):
    lab = make_labels(base_bars, k=1)
    t0, t1, t2 = base_bars.index[0], base_bars.index[1], base_bars.index[2]
    assert lab.loc[t0, ["y_call", "y_put"]].tolist() == [1.0, 0.0]
    assert lab.loc[t1, ["y_call", "y_put"]].tolist() == [0.0, 1.0]
    assert lab.loc[t2, ["y_call", "y_put"]].tolist() == [0.0, 0.0]


def test_slip_turns_small_win_into_loss(base_bars):
    # For t0, open[1]=1.1000, close[1]=1.1002, move is +0.0002.
    # If slip = 0.0003, move <= slip -> y_call becomes 0.0
    t0 = base_bars.index[0]
    assert make_labels(base_bars, k=1, slip=0.0003).loc[t0, "y_call"] == 0.0


def test_k5_uses_close_of_bar_t_plus_5(base_bars):
    lab = make_labels(base_bars, k=5)
    expected_call = float(base_bars["close"].iloc[5] > base_bars["open"].iloc[1])
    assert lab["y_call"].iloc[0] == expected_call
    assert lab["t1"].iloc[0] == base_bars.index[0] + pd.Timedelta(minutes=6)


def test_gap_gives_nan(base_bars):
    # drop bar index 4 -> labels whose window spans it are NaN
    bars_with_gap = base_bars.drop(base_bars.index[4])
    lab = make_labels(bars_with_gap, k=3)
    # for t=1, t+3 would be original index 4 (now dropped), so gap
    # for t=2, t+1 is index 3 (2024-01-02 00:03), t+3 would be 2024-01-02 00:05 (diff is 3 min? wait, let's check exact timestamps)
    # t=1 is 00:01. +1 min is 00:02 (present). +3 min is 00:04 (dropped!). So t=1 has gap.
    # t=2 is 00:02. +1 min is 00:03 (present). +3 min is 00:05 (present in bars_with_gap).
    # t=3 is 00:03. +1 min is 00:04 (dropped!). So t=3 has gap.
    t1 = base_bars.index[1]
    assert lab.loc[t1, ["y_call", "y_put"]].isna().all()
    # t=0 has +1 min = 00:01, +3 min = 00:03. Both exist and are at exact expected offsets!
    t0 = base_bars.index[0]
    assert lab.loc[t0, "y_call"] is not np.nan and not np.isnan(lab.loc[t0, "y_call"])


def test_last_rows_are_nan(base_bars):
    lab = make_labels(base_bars, k=3)
    assert lab["y_call"].iloc[-3:].isna().all()
    assert lab["y_put"].iloc[-3:].isna().all()
    assert lab["t1"].iloc[-3:].isna().all()
