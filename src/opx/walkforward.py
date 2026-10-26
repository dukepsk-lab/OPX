"""Purged walk-forward runner producing out-of-sample prediction frame."""

from typing import Any
import numpy as np
import pandas as pd
from opx.cv import purged_walk_forward
from opx.features import build_features
from opx.labels import make_labels
from opx.model import calibrate, fit_calibrator
from opx.model_registry import fit_model_dispatch, predict_win_prob_dispatch

COLUMNS = ["time", "side", "p_cal", "y", "t1"]


def run_walk_forward(
    bars: pd.DataFrame,
    k: int,
    r: float,
    slip: float = 0.0,
    n_splits: int = 6,
    calib_frac: float = 0.3,
    params: dict[str, Any] | None = None,
    skip_illiquid: bool = True,
    extra_features: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    model_type: str = "lightgbm",
) -> pd.DataFrame:
    """Run expanding purged walk-forward cross-validation.

    With skip_illiquid, signal bars in New York 16:00-18:59 (daily rollover) and on
    New York Sundays (week open) are dropped from training and testing: bid-only quotes
    jump there as spreads widen, which looks like a predictable edge (see docs/audit-eurusd-k15.md).

    Returns an out-of-sample DataFrame with columns:
        time, side ('call'|'put'), p_cal, y, t1
    """
    empty_df = pd.DataFrame(columns=COLUMNS)
    if bars.empty:
        return empty_df

    features = build_features(bars)
    if extra_features is not None:
        features = pd.concat([features, extra_features], axis=1)
    if feature_cols is not None:
        features = features[feature_cols]

    labels = make_labels(bars, k=k, slip=slip)

    # Drop rows with any NaN feature or label
    valid_mask = (
        features.notna().all(axis=1)
        & labels["y_call"].notna()
        & labels["y_put"].notna()
        & labels["t1"].notna()
    )
    if skip_illiquid:
        ny = bars.index.tz_convert("America/New_York")
        valid_mask &= ~(ny.hour.isin([16, 17, 18]) | (ny.day_of_week == 6))

    if not valid_mask.any():
        return empty_df

    feat_valid = features[valid_mask]
    labels_valid = labels[valid_mask]

    t0 = feat_valid.index
    t1_series = labels_valid["t1"]

    if len(t0) < n_splits + 1:
        return empty_df

    X_arr = feat_valid.to_numpy(dtype=float)
    t1_dt = pd.DatetimeIndex(t1_series)

    dfs: list[pd.DataFrame] = []

    for train_idx, test_idx in purged_walk_forward(t0, t1_series, n_splits=n_splits):
        n_tr = len(train_idx)
        if n_tr == 0:
            continue

        n_model = int(np.floor(n_tr * (1.0 - calib_frac)))
        if n_model == 0 or n_model == n_tr:
            continue

        cand_train = train_idx[:n_model]
        calib_idx = train_idx[n_model:]
        calib_start_time = t0[calib_idx[0]]

        # Purge model training rows whose t1 reaches or exceeds calibration start time
        model_mask = t1_dt[cand_train] < calib_start_time
        model_idx = cand_train[model_mask]

        if len(model_idx) == 0 or len(calib_idx) == 0:
            continue

        X_model = X_arr[model_idx]
        X_calib = X_arr[calib_idx]
        X_test = X_arr[test_idx]

        for side in ["call", "put"]:
            col_name = f"y_{side}"
            y_model = labels_valid[col_name].iloc[model_idx].to_numpy(dtype=float)
            y_calib = labels_valid[col_name].iloc[calib_idx].to_numpy(dtype=float)
            y_test = labels_valid[col_name].iloc[test_idx].to_numpy(dtype=float)

            # Skip a side for a fold when model or calibrator training labels have a single class
            if len(np.unique(y_model)) < 2 or len(np.unique(y_calib)) < 2:
                continue

            model = fit_model_dispatch(model_type, X_model, y_model, r=r, params=params)
            p_calib_uncal = predict_win_prob_dispatch(model_type, model, X_calib, r=r)
            cal = fit_calibrator(p_calib_uncal, y_calib)

            p_test_uncal = predict_win_prob_dispatch(model_type, model, X_test, r=r)
            p_cal = calibrate(cal, p_test_uncal)

            fold_side_df = pd.DataFrame(
                {
                    "time": t0[test_idx],
                    "side": side,
                    "p_cal": p_cal,
                    "y": y_test,
                    "t1": pd.DatetimeIndex(t1_series.iloc[test_idx]),
                }
            )
            dfs.append(fold_side_df)

    if not dfs:
        return empty_df

    oos = pd.concat(dfs, ignore_index=True)
    oos = oos.sort_values(["time", "side"]).reset_index(drop=True)
    return oos[COLUMNS]
