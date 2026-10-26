import pandas as pd
import pytest
from scripts.ablation_study import evaluate_ablation_oos


def test_evaluate_ablation_oos():
    idx = pd.date_range("2023-01-01 00:00:00", periods=50, freq="1min", tz="UTC")
    oos = pd.DataFrame(
        {
            "time": idx,
            "side": ["call"] * 50,
            "p_cal": [0.65] * 50,
            "y": [1.0] * 35 + [0.0] * 15,
            "t1": idx + pd.Timedelta(minutes=5),
        }
    )
    res = evaluate_ablation_oos(oos, r=0.85, buffers=[0.0, 0.02, 0.04], n_trials=11)
    assert "win_rate" in res
    assert "lcb" in res
    assert "dsr" in res
    assert "best_buffer" in res
    assert abs(res["win_rate"] - 0.6923) < 0.01
