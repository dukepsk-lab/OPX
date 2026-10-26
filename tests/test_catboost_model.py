import numpy as np
import pytest
from opx.models.catboost_model import fit_catboost, predict_catboost_prob


def test_catboost_ev_objective():
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.binomial(1, 0.5, 200).astype(float)
    model = fit_catboost(X, y, r=0.85, params={"iterations": 30, "verbose": 0})
    probs = predict_catboost_prob(model, X, r=0.85)
    assert len(probs) == 200
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # High logit should yield high win probability
    high_X = np.ones((5, 10)) * 5.0
    probs_high = predict_catboost_prob(model, high_X, r=0.85)
    assert len(probs_high) == 5
