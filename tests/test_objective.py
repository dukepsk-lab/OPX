import numpy as np
import pytest
from opx.objective import EVObjective


def test_hessian_positive_everywhere():
    z = np.linspace(-20, 20, 81)
    for y in (0.0, 1.0):
        _, h = EVObjective(0.85)(np.full_like(z, y), z)
        assert (h > 0).all()


def test_gradient_zero_at_ev_boundary():
    r, p = 0.85, 0.60
    q = r * p / (r * p + 1 - p)
    z = np.array([np.log(q / (1 - q))])
    g1, _ = EVObjective(r)(np.array([1.0]), z)
    g0, _ = EVObjective(r)(np.array([0.0]), z)
    assert p * g1[0] + (1 - p) * g0[0] == pytest.approx(0.0, abs=1e-12)


def test_zero_margin_means_breakeven_probability():
    # at raw=0, q=0.5, so the recovered win prob is the breakeven prob
    assert EVObjective(0.85).to_win_prob(np.array([0.0]))[0] == pytest.approx(1 / 1.85)


def test_to_win_prob_inverts_the_weighting():
    r, p = 0.85, 0.60
    q = r * p / (r * p + 1 - p)
    assert EVObjective(r).to_win_prob(np.array([np.log(q / (1 - q))]))[0] == pytest.approx(p)
