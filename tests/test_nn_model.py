import numpy as np
import pytest
import torch
from opx.models.nn_model import fit_nn, predict_nn_prob, CNNLSTMNet


def test_nn_forward_and_fit():
    np.random.seed(42)
    torch.manual_seed(42)
    X = np.random.randn(150, 20).astype(np.float32)
    y = np.random.binomial(1, 0.5, 150).astype(np.float32)
    model = fit_nn(X, y, r=0.85, epochs=3, batch_size=32, lr=1e-3)
    probs = predict_nn_prob(model, X, r=0.85)
    assert len(probs) == 150
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # Check that high input does not explode
    high_X = np.ones((5, 20), dtype=np.float32) * 5.0
    probs_high = predict_nn_prob(model, high_X, r=0.85)
    assert len(probs_high) == 5
