"""CatBoost model adapter with EV-weighted objective."""

from typing import Any
import catboost as cb
import numpy as np
from opx.econ import _check


class CatBoostEVObjective:
    """Cost-weighted EV objective for CatBoost.
    
    CatBoost expects negative first derivative:
    der1 = - (q * w - r * y)
    der2 = q * (1 - q) * w
    """

    def __init__(self, r: float):
        _check(r)
        self.r = float(r)

    def calc_ders_range(self, approxes, targets, weights):
        approx_arr = np.asarray(approxes, dtype=float)
        y_arr = np.asarray(targets, dtype=float)

        q = 1.0 / (1.0 + np.exp(-np.clip(approx_arr, -30.0, 30.0)))
        w = self.r * y_arr + (1.0 - y_arr)

        # In CatBoost, der1 = -grad
        der1 = -(q * w - self.r * y_arr)
        der2 = np.maximum(q * (1.0 - q) * w, 1e-12)

        if weights is not None:
            w_arr = np.asarray(weights, dtype=float)
            der1 *= w_arr
            der2 *= w_arr

        return list(zip(der1, der2))


DEFAULT_PARAMS: dict[str, Any] = {
    "iterations": 300,
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 5.0,
    "eval_metric": "Logloss",
    "random_seed": 42,
    "verbose": 0,
    "thread_count": -1,
}


def fit_catboost(
    X: np.ndarray,
    y: np.ndarray,
    r: float,
    params: dict[str, Any] | None = None,
) -> cb.CatBoostClassifier:
    """Train a CatBoost classifier with the EV-weighted objective."""
    combined_params = {**DEFAULT_PARAMS, **(params or {})}
    obj = CatBoostEVObjective(r)
    model = cb.CatBoostClassifier(loss_function=obj, **combined_params)
    model.fit(X, y)
    return model


def predict_catboost_prob(model: cb.CatBoostClassifier, X: np.ndarray, r: float) -> np.ndarray:
    """Predict uncalibrated win probabilities from the CatBoost raw margin."""
    raw = model.predict(X, prediction_type="RawFormulaVal")
    q = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
    denom = q + r * (1.0 - q)
    return q / denom
