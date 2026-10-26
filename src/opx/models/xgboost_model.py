"""XGBoost model adapter with EV-weighted objective."""

from typing import Any
import numpy as np
import xgboost as xgb
from opx.econ import _check


class XGBoostEVObjective:
    """Cost-weighted EV objective for XGBoost."""

    def __init__(self, r: float):
        _check(r)
        self.r = float(r)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_arr = np.asarray(y_true, dtype=float)
        raw_arr = np.asarray(y_pred, dtype=float)

        w = self.r * y_arr + (1.0 - y_arr)
        q = 1.0 / (1.0 + np.exp(-np.clip(raw_arr, -30.0, 30.0)))

        grad = q * w - self.r * y_arr
        hess = np.maximum(q * (1.0 - q) * w, 1e-12)

        return grad, hess


DEFAULT_PARAMS: dict[str, Any] = {
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}


def fit_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    r: float,
    params: dict[str, Any] | None = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with the EV-weighted objective."""
    combined_params = {**DEFAULT_PARAMS, **(params or {})}
    obj = XGBoostEVObjective(r)
    model = xgb.XGBClassifier(objective=obj, **combined_params)
    model.fit(X, y)
    return model


def predict_xgboost_prob(model: xgb.XGBClassifier, X: np.ndarray, r: float) -> np.ndarray:
    """Predict uncalibrated win probabilities from the XGBoost raw margin."""
    raw = model.predict(X, output_margin=True)
    q = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
    denom = q + r * (1.0 - q)
    return q / denom
