import pandas as pd
import pytest
from opx.walkforward import run_walk_forward
from opx.features_extra import compute_vol_squeeze

FAST = {"n_estimators": 50, "random_state": 42}


def test_walk_forward_with_extra_features(bars_factory):
    bars = bars_factory(6000, phi=-0.5)
    extra = compute_vol_squeeze(bars)

    # Run with extra features
    oos = run_walk_forward(
        bars,
        k=1,
        r=0.85,
        params=FAST,
        n_splits=3,
        extra_features=extra,
    )
    assert not oos.empty
    assert list(oos.columns) == ["time", "side", "p_cal", "y", "t1"]
    assert set(oos["side"]) == {"call", "put"}


def test_walk_forward_with_selected_feature_cols(bars_factory):
    bars = bars_factory(6000, phi=-0.5)
    extra = compute_vol_squeeze(bars)
    chosen_cols = ["body_ratio", "ret_1", "bb_squeeze"]

    oos = run_walk_forward(
        bars,
        k=1,
        r=0.85,
        params=FAST,
        n_splits=3,
        extra_features=extra,
        feature_cols=chosen_cols,
    )
    assert not oos.empty
