"""Advanced feature extractors: Volatility Squeeze, FracDiff, DXY, and Cross-Asset Returns."""

import numpy as np
import pandas as pd


def compute_vol_squeeze(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute Bollinger Band squeeze and short/long volatility ratios."""
    close_p = bars["close"]
    high_p = bars["high"]
    low_p = bars["low"]
    vol = bars["tick_volume"] if "tick_volume" in bars.columns else bars["volume"]

    # 1. Bollinger Band width
    roll20 = close_p.rolling(20)
    std20 = roll20.std(ddof=0)
    bb_width = 4.0 * std20

    # 2. True Range & ATR14
    prev_close = close_p.shift(1)
    c1 = high_p - low_p
    c2 = (high_p - prev_close).abs()
    c3 = (low_p - prev_close).abs()
    tr = pd.concat([c1, c2, c3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().replace(0.0, np.nan)

    # Squeeze ratio: BB width / ATR14 (< 1.5 indicates tight squeeze)
    bb_squeeze = (bb_width / atr14).replace([np.inf, -np.inf], np.nan)

    # 3. Realized volatility ratio: 5-bar rolling std / 60-bar rolling std
    log_ret = np.log(close_p / close_p.shift(1))
    std5 = log_ret.rolling(5).std(ddof=0)
    std60 = log_ret.rolling(60).std(ddof=0).replace(0.0, np.nan)
    vol_ratio_5_60 = (std5 / std60).replace([np.inf, -np.inf], np.nan)

    # 4. Volume ratio: 5-bar mean / 60-bar mean
    vol5 = vol.rolling(5).mean()
    vol60 = vol.rolling(60).mean().replace(0.0, np.nan)
    tickvol_ratio = (vol5 / vol60).replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame(
        {
            "bb_squeeze": bb_squeeze,
            "vol_ratio_5_60": vol_ratio_5_60,
            "tickvol_ratio": tickvol_ratio,
        },
        index=bars.index,
    )


def compute_fracdiff(series: pd.Series, d: float = 0.35, threshold: float = 1e-4) -> pd.Series:
    """Compute causal fractional differentiation of a series with weight thresholding.
    
    Weights: w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k.
    Only past values (<= t) are used (zero lookahead).
    """
    # Calculate weights up to threshold cutoff
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1.0) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    weights_arr = np.array(weights[::-1], dtype=float)  # reverse for convolution
    window_size = len(weights_arr)

    vals = series.to_numpy(dtype=float)
    n = len(vals)
    res = np.full(n, np.nan, dtype=float)

    if n >= window_size:
        # Causal convolution
        # Valid convolution aligns window to end at index i
        conv = np.convolve(vals, weights_arr, mode="valid")
        res[window_size - 1 :] = conv

    return pd.Series(res, index=series.index, name=f"fracdiff_{d}")


def _compute_rsi(series: pd.Series, n: int = 5) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    denom = avg_gain + avg_loss
    return pd.Series(
        np.where(denom == 0.0, 50.0, 100.0 * avg_gain / denom),
        index=series.index,
    )


def compute_dxy_features(target_index: pd.DatetimeIndex, dxy_bars: pd.DataFrame) -> pd.DataFrame:
    """Compute DXY momentum and relative state aligned causal to target_index."""
    # Reindex to target_index with forward-fill up to 5 minutes to prevent looking ahead
    # If DXY bar is not available, ffill ensures row t uses DXY at or before t
    dxy_aligned = dxy_bars.reindex(target_index, method="ffill", limit=5)
    close_dxy = dxy_aligned["close"]

    dxy_ret_1 = np.log(close_dxy / close_dxy.shift(1))
    dxy_ret_5 = np.log(close_dxy / close_dxy.shift(5))
    dxy_ret_15 = np.log(close_dxy / close_dxy.shift(15))

    dxy_rsi_5 = _compute_rsi(close_dxy, n=5)

    roll20 = close_dxy.rolling(20)
    mid = roll20.mean()
    std = roll20.std(ddof=0)
    bb_band = (4.0 * std).replace(0.0, np.nan)
    lower_band = mid - 2.0 * std
    dxy_bb_pctb = ((close_dxy - lower_band) / bb_band).replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame(
        {
            "dxy_ret_1": dxy_ret_1,
            "dxy_ret_5": dxy_ret_5,
            "dxy_ret_15": dxy_ret_15,
            "dxy_rsi_5": dxy_rsi_5,
            "dxy_bb_pctb": dxy_bb_pctb,
        },
        index=target_index,
    )


def compute_cross_returns(
    target_index: pd.DatetimeIndex,
    other_pairs: dict[str, pd.DataFrame],
    base_is_usd: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Compute normalized returns of other pairs and composite USD return."""
    if base_is_usd is None:
        base_is_usd = {
            "usdjpy": True,
            "usdcad": True,
            "eurusd": False,
            "gbpusd": False,
            "audusd": False,
        }

    feat_dict: dict[str, pd.Series] = {}
    usd_ret_1_components: list[pd.Series] = []
    usd_ret_5_components: list[pd.Series] = []

    for name, p_bars in other_pairs.items():
        aligned = p_bars.reindex(target_index, method="ffill", limit=5)
        p_close = aligned["close"]
        r1 = np.log(p_close / p_close.shift(1))
        r5 = np.log(p_close / p_close.shift(5))

        feat_dict[f"cross_{name}_ret_1"] = r1
        feat_dict[f"cross_{name}_ret_5"] = r5

        # Directional USD contribution
        is_usd_base = base_is_usd.get(name.lower(), False)
        # If USD is base (e.g. USDJPY), price up = USD up (+r)
        # If USD is quote (e.g. EURUSD), price up = USD down (-r)
        factor = 1.0 if is_usd_base else -1.0
        usd_ret_1_components.append(factor * r1)
        usd_ret_5_components.append(factor * r5)

    if usd_ret_1_components:
        feat_dict["composite_usd_ret_1"] = pd.concat(usd_ret_1_components, axis=1).mean(axis=1)
        feat_dict["composite_usd_ret_5"] = pd.concat(usd_ret_5_components, axis=1).mean(axis=1)

    df = pd.DataFrame(feat_dict, index=target_index)
    return df.replace([np.inf, -np.inf], np.nan)
