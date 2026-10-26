import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def bars_factory():
    def _create(n: int = 5000, phi: float = 0.0, seed: int = 0, start: str = "2024-01-02") -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        times = pd.date_range(start, periods=n, freq="1min", tz="UTC")

        eps = rng.normal(0, 1, n)
        log_ret = np.zeros(n)
        for t in range(1, n):
            log_ret[t] = phi * log_ret[t - 1] + 1e-4 * eps[t]

        # Starting price 1.1000
        cum_ret = np.cumsum(log_ret)
        closes = 1.1000 * np.exp(cum_ret)
        opens = np.empty(n)
        opens[0] = 1.1000
        opens[1:] = closes[:-1]

        noise_high = np.abs(rng.normal(0, 2e-5, n))
        noise_low = np.abs(rng.normal(0, 2e-5, n))
        highs = np.maximum(opens, closes) + noise_high
        lows = np.minimum(opens, closes) - noise_low
        volumes = rng.poisson(100, n).astype(float)

        return pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "tick_volume": volumes,
            },
            index=times,
        )

    return _create
