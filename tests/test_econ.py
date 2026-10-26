import pytest
from opx.econ import breakeven, ev_per_trade, kelly_fraction

def test_breakeven():
    assert breakeven(0.85) == pytest.approx(1 / 1.85)
    assert breakeven(0.80) == pytest.approx(0.5556, abs=1e-4)

def test_ev_per_trade():
    assert ev_per_trade(0.55, 0.85) == pytest.approx(0.0175)
    assert ev_per_trade(0.50, 0.85) == pytest.approx(-0.075)

def test_kelly_fraction():
    assert kelly_fraction(0.55, 0.85) == pytest.approx(0.55 - 0.45 / 0.85)
    assert kelly_fraction(0.50, 0.85) == 0.0

@pytest.mark.parametrize("r", [0, -0.1, 1.5])
def test_invalid_payout(r):
    with pytest.raises(ValueError):
        breakeven(r)
