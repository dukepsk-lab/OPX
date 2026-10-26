"""EV-weighted logloss objective for LightGBM."""

import numpy as np
from opx.econ import _check


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class EVObjective:
    """Cost-weighted binary logloss objective.
    
    Weights wins (y=1) by r and losses (y=0) by 1.
    The decision threshold (zero margin) corresponds to the breakeven probability p* = 1 / (1 + r).
    """

    def __init__(self, r: float):
        _check(r)
        self.r = float(r)

    def __call__(self, y_true: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute first and second derivatives w.r.t. the raw margin.
        
        Compatible with LightGBM's scikit-learn API objective callable.
        """
        y_arr = np.asarray(y_true, dtype=float)
        raw_arr = np.asarray(raw, dtype=float)

        w = self.r * y_arr + (1.0 - y_arr)
        q = _sigmoid(raw_arr)

        grad = q * w - self.r * y_arr
        hess = np.maximum(q * (1.0 - q) * w, 1e-12)

        return grad, hess

    def to_win_prob(self, raw: np.ndarray) -> np.ndarray:
        """Invert the cost-weighting to recover the true underlying win probability p.
        
        p = q / (q + r * (1 - q)) where q = sigmoid(raw).
        """
        raw_arr = np.asarray(raw, dtype=float)
        q = _sigmoid(raw_arr)
        denom = q + self.r * (1.0 - q)
        return q / denom
