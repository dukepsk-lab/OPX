"""Settlement-consistent binary trade labels with gap guard."""

import numpy as np
import pandas as pd


def make_labels(bars: pd.DataFrame, k: int, slip: float = 0.0) -> pd.DataFrame:
    """Generate Call and Put binary labels and expiration timestamps (t1).
    
    Parameters
    ----------
    bars : pd.DataFrame
        M1 bars with UTC DatetimeIndex and columns including 'open' and 'close'.
    k : int
        Expiry horizon in minutes (e.g. 1, 5, 15).
    slip : float
        Slippage threshold. Move must strictly exceed slip to win.
        
    Returns
    -------
    pd.DataFrame with columns:
        y_call : float (1.0, 0.0, or NaN)
        y_put : float (1.0, 0.0, or NaN)
        t1 : pd.Timestamp (index + (k+1) minutes, or NaT)
    """
    n = len(bars)
    idx = bars.index

    # Pre-allocate results with NaN / NaT
    y_call = pd.Series(np.nan, index=idx, dtype=float)
    y_put = pd.Series(np.nan, index=idx, dtype=float)
    t1 = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")

    if n < k + 1:
        return pd.DataFrame({"y_call": y_call, "y_put": y_put, "t1": t1}, index=idx)

    # Positional shifts
    open_next = bars["open"].shift(-1)
    close_k = bars["close"].shift(-k)

    idx_series = pd.Series(idx, index=idx)
    idx_next = idx_series.shift(-1)
    idx_k = idx_series.shift(-k)

    expected_next = idx + pd.Timedelta(minutes=1)
    expected_k = idx + pd.Timedelta(minutes=k)

    # Valid mask: next bar is exactly +1 min, and k-th bar is exactly +k min
    valid = (idx_next == expected_next) & (idx_k == expected_k)

    call_win = (close_k - open_next) > slip
    put_win = (open_next - close_k) > slip

    y_call[valid] = np.where(call_win[valid], 1.0, 0.0)
    y_put[valid] = np.where(put_win[valid], 1.0, 0.0)
    t1[valid] = idx[valid] + pd.Timedelta(minutes=k + 1)

    return pd.DataFrame({"y_call": y_call, "y_put": y_put, "t1": t1}, index=idx)
