"""Unified model registry supporting LightGBM, CatBoost, XGBoost, and PyTorch CNN+LSTM."""

from typing import Any
import numpy as np

from opx.model import fit_model as fit_lgb, predict_win_prob as predict_lgb_prob
from opx.models.catboost_model import fit_catboost, predict_catboost_prob
from opx.models.nn_model import fit_nn, predict_nn_prob
from opx.models.xgboost_model import fit_xgboost, predict_xgboost_prob

SUPPORTED_MODELS = ["lightgbm", "catboost", "xgboost", "cnn_lstm"]


def fit_model_dispatch(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    r: float,
    params: dict[str, Any] | None = None,
) -> Any:
    """Dispatch training to the appropriate model architecture."""
    m_type = model_type.lower()
    if m_type == "lightgbm":
        return fit_lgb(X, y, r=r, params=params)
    elif m_type == "catboost":
        return fit_catboost(X, y, r=r, params=params)
    elif m_type == "xgboost":
        return fit_xgboost(X, y, r=r, params=params)
    elif m_type in ("cnn_lstm", "nn"):
        p = params or {}
        epochs = p.get("epochs", 5)
        batch_size = p.get("batch_size", 256)
        lr = p.get("lr", 1e-3)
        return fit_nn(X, y, r=r, epochs=epochs, batch_size=batch_size, lr=lr)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: {SUPPORTED_MODELS}")


def predict_win_prob_dispatch(
    model_type: str,
    model: Any,
    X: np.ndarray,
    r: float,
) -> np.ndarray:
    """Dispatch raw probability prediction to the appropriate model architecture."""
    m_type = model_type.lower()
    if m_type == "lightgbm":
        return predict_lgb_prob(model, X, r=r)
    elif m_type == "catboost":
        return predict_catboost_prob(model, X, r=r)
    elif m_type == "xgboost":
        return predict_xgboost_prob(model, X, r=r)
    elif m_type in ("cnn_lstm", "nn"):
        return predict_nn_prob(model, X, r=r)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: {SUPPORTED_MODELS}")
