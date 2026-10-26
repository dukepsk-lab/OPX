"""Causal technical indicator feature engineering."""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "body_ratio",
    "upper_wick",
    "lower_wick",
    "clv",
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_15",
    "rsi_3",
    "rsi_5",
    "rsi_14",
    "bb_pctb",
    "atr_pct",
    "tickvol_z",
    "kaufman_er",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "sess_london",
    "sess_ny",
    "sess_overlap",
]


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build causal feature matrix from M1 OHLCV bars.
    
    Row t uses only bars <= t.
    Zero-range and flat periods produce NaN or finite values, never inf.
    """
    idx = bars.index
    open_p = bars["open"]
    high_p = bars["high"]
    low_p = bars["low"]
    close_p = bars["close"]
    vol = bars["tick_volume"]

    # 1. Candle geometry
    bar_range = (high_p - low_p).replace(0.0, np.nan)
    body = (close_p - open_p).abs()
    body_ratio = body / bar_range
    upper_wick = (high_p - np.maximum(open_p, close_p)) / bar_range
    lower_wick = (np.minimum(open_p, close_p) - low_p) / bar_range
    clv = (2.0 * close_p - high_p - low_p) / bar_range

    # 2. True Range & ATR14
    prev_close = close_p.shift(1)
    c1 = high_p - low_p
    c2 = (high_p - prev_close).abs()
    c3 = (low_p - prev_close).abs()
    tr = pd.concat([c1, c2, c3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_pct = (atr14 / close_p).replace(0.0, np.nan)

    # 3. ATR-normalized returns
    ret_dict = {}
    for l in [1, 2, 3, 5, 10, 15]:
        log_ret = np.log(close_p / close_p.shift(l))
        ret_dict[f"ret_{l}"] = log_ret / (atr_pct * np.sqrt(l))

    # 4. RSI (Wilder smoothing: alpha = 1 / n, adjust=False)
    delta = close_p.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rsi_dict = {}
    for n in [3, 5, 14]:
        avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()
        denom = avg_gain + avg_loss
        rsi = pd.Series(
            np.where(denom == 0.0, 50.0, 100.0 * avg_gain / denom),
            index=idx,
        )
        rsi_dict[f"rsi_{n}"] = rsi

    # 5. Bollinger Bands %B (20-bar, +/-2 sigma)
    roll20 = close_p.rolling(20)
    mid = roll20.mean()
    std = roll20.std(ddof=0)
    bb_band = (4.0 * std).replace(0.0, np.nan)
    lower_band = mid - 2.0 * std
    bb_pctb = (close_p - lower_band) / bb_band

    # 6. Tick volume z-score (60-bar)
    roll60 = vol.rolling(60)
    vol_mean = roll60.mean()
    vol_std = roll60.std(ddof=0).replace(0.0, np.nan)
    tickvol_z = (vol - vol_mean) / vol_std

    # 7. Kaufman Efficiency Ratio (10-bar)
    direction = (close_p - close_p.shift(10)).abs()
    volatility = (close_p - close_p.shift(1)).abs().rolling(10).sum().replace(0.0, np.nan)
    kaufman_er = direction / volatility

    # 8. Cyclical time features
    minute_of_day = idx.hour * 60 + idx.minute
    tod_sin = pd.Series(np.sin(2.0 * np.pi * minute_of_day / 1440.0), index=idx)
    tod_cos = pd.Series(np.cos(2.0 * np.pi * minute_of_day / 1440.0), index=idx)
    dow = idx.day_of_week if hasattr(idx, "day_of_week") else idx.dayofweek
    dow_sin = pd.Series(np.sin(2.0 * np.pi * dow / 7.0), index=idx)
    dow_cos = pd.Series(np.cos(2.0 * np.pi * dow / 7.0), index=idx)

    # 9. Session flags (London 08-16, NY 08-16 local exchange time)
    lon_idx = idx.tz_convert("Europe/London")
    sess_london = pd.Series(((lon_idx.hour >= 8) & (lon_idx.hour < 17)).astype(float), index=idx)

    ny_idx = idx.tz_convert("America/New_York")
    sess_ny = pd.Series(((ny_idx.hour >= 8) & (ny_idx.hour < 17)).astype(float), index=idx)

    sess_overlap = pd.Series(((sess_london == 1.0) & (sess_ny == 1.0)).astype(float), index=idx)

    # Assemble DataFrame
    features = pd.DataFrame(
        {
            "body_ratio": body_ratio,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "clv": clv,
            **ret_dict,
            **rsi_dict,
            "bb_pctb": bb_pctb,
            "atr_pct": atr_pct,
            "tickvol_z": tickvol_z,
            "kaufman_er": kaufman_er,
            "tod_sin": tod_sin,
            "tod_cos": tod_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "sess_london": sess_london,
            "sess_ny": sess_ny,
            "sess_overlap": sess_overlap,
        },
        index=idx,
    )

    # Ensure no inf values exist
    features = features.replace([np.inf, -np.inf], np.nan)
    return features[FEATURE_COLUMNS]
