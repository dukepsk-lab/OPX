"""Purged expanding walk-forward cross-validation splits."""

from typing import Iterator
import numpy as np
import pandas as pd


def purged_walk_forward(
    t0: pd.DatetimeIndex,
    t1: pd.Series | pd.DatetimeIndex,
    n_splits: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield positional (train_idx, test_idx) for expanding walk-forward validation.
    
    Splits positions into n_splits + 1 contiguous blocks.
    Block 0 is initial training data.
    Fold i (0 <= i < n_splits) tests block i + 1.
    Training positions are all positions p < test_start where t1[p] < t0[test_start].
    """
    n = len(t0)
    if n_splits <= 0:
        raise ValueError(f"n_splits must be positive, got {n_splits}")
    if n < n_splits + 1:
        raise ValueError(f"Not enough samples ({n}) for {n_splits + 1} blocks")

    blocks = np.array_split(np.arange(n), n_splits + 1)
    t1_dt = pd.DatetimeIndex(t1)

    for i in range(n_splits):
        test_idx = blocks[i + 1]
        test_start_pos = test_idx[0]
        test_start_time = t0[test_start_pos]

        cand = np.arange(test_start_pos)
        mask = t1_dt[:test_start_pos] < test_start_time
        train_idx = cand[mask]

        yield train_idx, test_idx
