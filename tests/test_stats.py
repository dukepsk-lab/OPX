import numpy as np
import pytest
from opx.stats import wilson_lcb, deflated_sharpe, permutation_pvalue, pbo


def test_wilson_lcb():
    assert wilson_lcb(60, 100) == pytest.approx(0.5178, abs=1e-3)  # z = 1.645
    assert np.isnan(wilson_lcb(0, 0))
    assert wilson_lcb(600, 1000) > wilson_lcb(60, 100)


def test_dsr_rewards_edge_and_punishes_many_trials():
    rng = np.random.default_rng(0)
    edge = rng.normal(0.15, 1.0, 3000)
    noise = rng.normal(0.0, 1.0, 3000)
    assert deflated_sharpe(edge, n_trials=1) > 0.99
    assert deflated_sharpe(noise, n_trials=100) < 0.95
    assert deflated_sharpe(edge, n_trials=1) > deflated_sharpe(edge, n_trials=500)


def test_permutation_pvalue():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000).astype(float)
    take_good = (y == 1) | (rng.random(4000) < 0.3)  # selection correlated with outcome
    take_none = rng.random(4000) < 0.3
    assert permutation_pvalue(y, take_good, 0.85) < 0.01
    assert permutation_pvalue(y, take_none, 0.85) > 0.05


def test_permutation_pvalue_matches_exact_hypergeometric_tail():
    from scipy.stats import hypergeom
    y = np.r_[np.ones(100), np.zeros(100)]
    take = np.zeros(200, dtype=bool)
    take[:13] = True            # 13 wins among 13 taken
    take[100:107] = True        # plus 7 losses -> K=20, 13 wins
    exact = hypergeom.sf(12, 200, 100, 20)
    got = permutation_pvalue(y, take, 0.85, n_perm=40000, seed=3)
    assert got == pytest.approx(exact, abs=0.01)


def test_pbo():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, (800, 10))
    assert 0.3 < pbo(noise) < 0.7
    dominated = noise.copy()
    dominated[:, 3] += 0.5
    assert pbo(dominated) < 0.1
