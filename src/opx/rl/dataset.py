"""Multi-pair signal & non-raw-price state matrix builder for RL."""

from typing import Any
import numpy as np
import pandas as pd

from opx.features import build_features
from opx.features_extra import compute_dxy_features, compute_vol_squeeze
from opx.labels import make_labels

EXCLUDED_RAW_COLUMNS = {"open", "high", "low", "close", "tick_volume"}


def build_rl_state_matrix(
    pairs_bars: dict[str, pd.DataFrame],
    pairs_preds: dict[str, pd.DataFrame],
    dxy_bars: pd.DataFrame | None = None,
    k: int = 5,
    slip: float = 0.00002,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct causal state matrix with ML probabilities and non-price features, plus trade outcomes."""
    if not pairs_bars:
        raise ValueError("pairs_bars cannot be empty")

    first_pair = next(iter(pairs_bars))
    common_idx = pairs_bars[first_pair].index

    # Find common timestamp index across all pairs
    for p_name, p_bars in pairs_bars.items():
        common_idx = common_idx.intersection(p_bars.index)
        if p_name in pairs_preds:
            common_idx = common_idx.intersection(pairs_preds[p_name].index)

    state_dict: dict[str, pd.Series] = {}
    outcome_dict: dict[str, pd.Series] = {}

    for p_name, p_bars in pairs_bars.items():
        aligned_bars = p_bars.loc[common_idx]

        # 1. ML Probability Signals
        if p_name in pairs_preds:
            preds = pairs_preds[p_name].loc[common_idx]
            state_dict[f"{p_name}_p_call"] = preds["p_call"]
            state_dict[f"{p_name}_p_put"] = preds["p_put"]

        # 2. Non-price technical features
        feat = build_features(aligned_bars)
        for col in feat.columns:
            # Exclude any raw price if present
            if col not in EXCLUDED_RAW_COLUMNS:
                state_dict[f"{p_name}_{col}"] = feat[col]

        # 3. Vol Squeeze features
        squeeze = compute_vol_squeeze(aligned_bars)
        for col in squeeze.columns:
            state_dict[f"{p_name}_{col}"] = squeeze[col]

        # 4. Binary outcomes for reward calculation (Win = 1.0, Loss = 0.0)
        labels = make_labels(aligned_bars, k=k, slip=slip)
        outcome_dict[f"{p_name}_call_win"] = labels["y_call"]
        outcome_dict[f"{p_name}_put_win"] = labels["y_put"]

    # 5. DXY Features
    if dxy_bars is not None:
        dxy_feat = compute_dxy_features(common_idx, dxy_bars)
        for col in dxy_feat.columns:
            state_dict[col] = dxy_feat[col]

    state_df = pd.DataFrame(state_dict, index=common_idx)
    outcomes_df = pd.DataFrame(outcome_dict, index=common_idx)

    # Valid mask: drop any row with NaN in state or outcome
    valid = state_df.notna().all(axis=1) & outcomes_df.notna().all(axis=1)

    state_clean = state_df[valid].copy()
    outcomes_clean = outcomes_df[valid].copy()

    # Double check: ensure no raw price columns slipped in
    for col in state_clean.columns:
        col_lower = col.lower()
        for raw in ["open", "high", "low", "close"]:
            if col_lower == raw or col_lower.endswith(f"_{raw}"):
                state_clean = state_clean.drop(columns=[col])

    return state_clean, outcomes_clean
