"""Payout maths and betting fractions."""


def _check(r: float) -> None:
    if not (0.0 < r <= 1.0):
        raise ValueError(f"Payout r must satisfy 0 < r <= 1, got {r}")


def breakeven(r: float) -> float:
    """Breakeven win rate p* = 1 / (1 + r)."""
    _check(r)
    return 1.0 / (1.0 + r)


def ev_per_trade(p: float, r: float) -> float:
    """Expected value per dollar traded: p * r - (1 - p)."""
    _check(r)
    return p * r - (1.0 - p)


def kelly_fraction(p: float, r: float) -> float:
    """Full Kelly fraction max(0, p - (1 - p) / r)."""
    _check(r)
    return max(0.0, float(p - (1.0 - p) / r))
