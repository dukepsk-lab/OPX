import numpy as np
import pytest
from opx.walkforward import run_walk_forward

FAST_PARAMS = {
    "lightgbm": {"n_estimators": 30, "verbose": -1},
    "xgboost": {"n_estimators": 30, "verbosity": 0},
    "catboost": {"iterations": 30, "verbose": 0},
    "cnn_lstm": {"epochs": 2, "batch_size": 64},
}


@pytest.mark.parametrize("m_type", ["lightgbm", "xgboost", "catboost", "cnn_lstm"])
def test_walk_forward_multi_model(bars_factory, m_type):
    bars = bars_factory(4000, phi=-0.5)
    params = FAST_PARAMS.get(m_type)
    oos = run_walk_forward(
        bars,
        k=1,
        r=0.85,
        model_type=m_type,
        params=params,
        n_splits=2,
    )
    assert not oos.empty
    assert list(oos.columns) == ["time", "side", "p_cal", "y", "t1"]
    assert set(oos["side"]) == {"call", "put"}
    assert (oos["p_cal"] >= 0.0).all() and (oos["p_cal"] <= 1.0).all()
