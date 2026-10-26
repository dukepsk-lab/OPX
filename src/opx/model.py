"""LightGBM training with EV-weighted objective and Platt probability calibration."""

from typing import Any
import lightgbm as lgb
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from opx.objective import EVObjective

DEFAULT_PARAMS: dict[str, Any] = {
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 100,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbose": -1,
}


def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    r: float,
    params: dict[str, Any] | None = None,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier with the EV-weighted objective."""
    combined_params = {**DEFAULT_PARAMS, **(params or {})}
    obj = EVObjective(r)
    model = lgb.LGBMClassifier(objective=obj, **combined_params)
    model.fit(X, y)
    return model


def predict_win_prob(model: lgb.LGBMClassifier, X: np.ndarray, r: float) -> np.ndarray:
    """Predict uncalibrated win probabilities from the booster's raw scores."""
    # Custom callable objective in LightGBM outputs raw margin via booster_.predict
    raw = model.booster_.predict(X, raw_score=True)
    obj = EVObjective(r)
    return obj.to_win_prob(raw)


def fit_calibrator(p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit a Platt calibrator (logistic regression on logit-transformed probabilities)."""
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    clipped = np.clip(p_arr, 1e-6, 1.0 - 1e-6)
    z = logit(clipped).reshape(-1, 1)

    cal = LogisticRegression(solver="lbfgs")
    cal.fit(z, y_arr)
    return cal


def calibrate(cal: LogisticRegression, p: np.ndarray) -> np.ndarray:
    """Calibrate probabilities using a fitted Platt calibrator."""
    p_arr = np.asarray(p, dtype=float)
    clipped = np.clip(p_arr, 1e-6, 1.0 - 1e-6)
    z = logit(clipped).reshape(-1, 1)
    probs = cal.predict_proba(z)[:, 1]
    return probs
