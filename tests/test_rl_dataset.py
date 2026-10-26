import numpy as np
import pandas as pd
import pytest
from opx.rl.dataset import build_rl_state_matrix


def test_no_raw_price_in_rl_state():
    idx = pd.date_range("2023-01-01 00:00:00", periods=50, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [1.10] * 50,
            "high": [1.11] * 50,
            "low": [1.09] * 50,
            "close": [1.10] * 50,
            "tick_volume": [20.0] * 50,
        },
        index=idx,
    )
    pairs_bars = {"eurusd": bars, "gbpusd": bars}
    pairs_preds = {
        "eurusd": pd.DataFrame({"p_call": [0.55] * 50, "p_put": [0.45] * 50}, index=idx),
        "gbpusd": pd.DataFrame({"p_call": [0.52] * 50, "p_put": [0.48] * 50}, index=idx),
    }

    state_df, outcomes_df = build_rl_state_matrix(pairs_bars, pairs_preds, k=5, slip=0.00002)

    # STRICT ASSERTION: raw prices must NEVER appear in state_df
    for raw_col in ["open", "high", "low", "close"]:
        assert raw_col not in state_df.columns
        assert not any(col.endswith(f"_{raw_col}") for col in state_df.columns)

    # ML signals must be present
    assert "eurusd_p_call" in state_df.columns
    assert "eurusd_p_put" in state_df.columns
    assert "gbpusd_p_call" in state_df.columns

    # Outcomes must have binary results
    assert "eurusd_call_win" in outcomes_df.columns
    assert "eurusd_put_win" in outcomes_df.columns
    assert len(state_df) == len(outcomes_df)
