"""Trade selection above payout breakeven buffer and performance summary."""

from typing import Any
import numpy as np
import pandas as pd
from opx.econ import breakeven, ev_per_trade
from opx.stats import wilson_lcb


def select_trades(oos: pd.DataFrame, r: float, buffer: float) -> pd.DataFrame:
    """Filter out-of-sample predictions to non-overlapping trades exceeding threshold.
    
    Threshold is breakeven(r) + buffer.
    Trades are sorted chronologically. A trade is kept only if its entry time
    (time + 1 minute) is >= the previous kept trade's expiry (t1).
    """
    if oos.empty:
        return oos.copy()

    thresh = breakeven(r) + buffer
    filtered = oos[oos["p_cal"] > thresh].sort_values("time")

    if filtered.empty:
        return filtered.copy()

    times = filtered["time"].to_numpy()
    t1s = filtered["t1"].to_numpy()
    kept_mask = np.zeros(len(filtered), dtype=bool)

    last_t1 = None
    one_min = pd.Timedelta(minutes=1)

    for i in range(len(filtered)):
        entry_time = times[i] + one_min
        if last_t1 is None or entry_time >= last_t1:
            kept_mask[i] = True
            last_t1 = t1s[i]

    return filtered[kept_mask].copy()


def summarize(trades: pd.DataFrame, r: float) -> dict[str, Any]:
    """Calculate summary statistics for selected trades."""
    be = breakeven(r)
    n = len(trades)

    if n == 0:
        return {
            "n": 0,
            "win_rate": np.nan,
            "ev": np.nan,
            "lcb": np.nan,
            "breakeven": be,
        }

    y_vals = trades["y"].to_numpy(dtype=float)
    wins = int(np.sum(y_vals == 1.0))
    win_rate = float(np.mean(y_vals))
    ev = float(ev_per_trade(win_rate, r))
    lcb = float(wilson_lcb(wins, n))

    return {
        "n": n,
        "win_rate": win_rate,
        "ev": ev,
        "lcb": lcb,
        "breakeven": be,
    }
