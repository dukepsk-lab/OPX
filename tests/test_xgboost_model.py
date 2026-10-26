import numpy as np
import pytest
from opx.models.xgboost_model import fit_xgboost, predict_xgboost_prob


def test_xgboost_ev_objective():
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.binomial(1, 0.5, 200).astype(float)
    model = fit_xgboost(X, y, r=0.85, params={"n_estimators": 30, "verbosity": 0})
    probs = predict_xgboost_prob(model, X, r=0.85)
    assert len(probs) == 200
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # High logit test
    high_X = np.ones((5, 10)) * 5.0
    probs_high = predict_xgboost_prob(model, high_X, r=0.85)
    assert len(probs_high) == 5
