import pandas as pd
import pytest
from opx.evaluate import select_trades, summarize
from opx.walkforward import run_walk_forward

FAST = {"n_estimators": 80, "random_state": 42}


def test_planted_mean_reversion_is_found(bars_factory):
    oos = run_walk_forward(bars_factory(30000, phi=-0.5), k=1, r=0.85, params=FAST)
    s = summarize(select_trades(oos, 0.85, buffer=0.03), 0.85)
    assert s["n"] > 300 and s["win_rate"] > 0.60


def test_pure_noise_has_no_edge(bars_factory):
    oos = run_walk_forward(bars_factory(30000, phi=0.0), k=1, r=0.85, params=FAST)
    s = summarize(select_trades(oos, 0.85, buffer=0.03), 0.85)
    assert s["n"] == 0 or s["lcb"] < s["breakeven"]


def test_oos_rows_are_strictly_after_first_train_block(bars_factory):
    bars = bars_factory(6000, phi=-0.5)
    # skip_illiquid=False so no rows are dropped and the first block is exactly len(bars) // 5
    oos = run_walk_forward(bars, k=1, r=0.85, params=FAST, n_splits=4, skip_illiquid=False)
    assert oos["time"].min() >= bars.index[len(bars) // 5]
    assert set(oos["side"]) == {"call", "put"}


def test_illiquid_hours_are_skipped_by_default(bars_factory):
    oos = run_walk_forward(bars_factory(30000, phi=-0.5), k=1, r=0.85, params=FAST)
    ny = pd.DatetimeIndex(oos["time"]).tz_convert("America/New_York")
    assert not ny.hour.isin([16, 17, 18]).any()
    assert not (ny.day_of_week == 6).any()


def test_skip_illiquid_false_keeps_rollover_and_sunday(bars_factory):
    oos = run_walk_forward(bars_factory(30000, phi=-0.5), k=1, r=0.85, params=FAST,
                           skip_illiquid=False)
    ny = pd.DatetimeIndex(oos["time"]).tz_convert("America/New_York")
    assert (ny.hour == 17).any() and (ny.day_of_week == 6).any()


def test_flat_prices_return_empty_frame(bars_factory):
    flat = bars_factory(3000)
    flat.loc[:, ["open", "high", "low", "close"]] = 1.1
    oos = run_walk_forward(flat, k=1, r=0.85, params=FAST)
    assert oos.empty and list(oos.columns) == ["time", "side", "p_cal", "y", "t1"]
