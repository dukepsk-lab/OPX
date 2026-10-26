import numpy as np
import pandas as pd
import pytest
from opx.cv import purged_walk_forward


@pytest.fixture
def cv_data():
    t0 = pd.date_range("2024-01-01", periods=1000, freq="1min", tz="UTC")
    t1 = pd.Series(t0 + pd.Timedelta(minutes=11), index=t0)
    return t0, t1


def test_test_blocks_are_contiguous_and_ordered(cv_data):
    t0, t1 = cv_data
    folds = list(purged_walk_forward(t0, t1, n_splits=4))
    assert len(folds) == 4
    starts = [te[0] for _, te in folds]
    assert starts == sorted(starts) and all(np.all(np.diff(te) == 1) for _, te in folds)


def test_no_training_label_reaches_test_window(cv_data):
    t0, t1 = cv_data
    for tr, te in purged_walk_forward(t0, t1, n_splits=4):
        assert (t1.iloc[tr] < t0[te[0]]).all()
        assert tr.max() < te[0]


def test_purge_drops_exactly_the_overlapping_rows(cv_data):
    t0, t1 = cv_data
    tr, te = next(iter(purged_walk_forward(t0, t1, n_splits=4)))
    assert len(tr) == te[0] - 11  # rows within 11 minutes of the test start are purged
