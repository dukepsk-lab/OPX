"""Validated bar loading for CSV and Parquet files."""

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
TARGET_COLUMNS = ["open", "high", "low", "close", "tick_volume"]


def load_bars(path: str | Path) -> pd.DataFrame:
    """Load and validate M1 bars from a CSV or Parquet file.
    
    Returns a DataFrame with a sorted, unique, UTC DatetimeIndex and
    columns: open, high, low, close, tick_volume.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    # Parse timestamps
    ts_col = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts_col):
        index = pd.to_datetime(ts_col, unit="ms", utc=True)
    else:
        index = pd.to_datetime(ts_col, utc=True)

    df = df.set_index(index)
    df = df.rename(columns={"volume": "tick_volume"})
    df = df[TARGET_COLUMNS]

    df = df.sort_index()

    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps found in bars")

    return df
