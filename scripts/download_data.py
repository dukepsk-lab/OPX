"""Helper script to download 1-year M1 data directly from Dukascopy into data/."""

from datetime import datetime
from pathlib import Path
import dukascopy_python as dp
import pandas as pd

PAIRS = {
    "eurusd": "EUR/USD",
    "gbpusd": "GBP/USD",
    "usdjpy": "USD/JPY",
}


def download_pair(pair_name: str, symbol: str, start: datetime, end: datetime, out_dir: Path):
    print(f"Downloading {symbol} ({start.date()} to {end.date()})...")
    df = dp.fetch(
        instrument=symbol,
        interval=dp.INTERVAL_MIN_1,
        offer_side=dp.OFFER_SIDE_BID,
        start=start,
        end=end,
    )
    if df.empty:
        print(f"Warning: No data fetched for {symbol}")
        return

    # Reset index so 'timestamp' is a column with UTC Datetime
    df = df.reset_index().rename(columns={"index": "timestamp"})
    # Keep exact columns expected by load_bars: timestamp, open, high, low, close, volume
    target_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[target_cols]

    out_file = out_dir / f"{pair_name}_m1.parquet"
    df.to_parquet(out_file, index=False)
    print(f"Saved {len(df):,} bars to {out_file}")


def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    start = datetime(2023, 1, 1)
    end = datetime(2023, 12, 31, 23, 59)

    for name, symbol in PAIRS.items():
        download_pair(name, symbol, start, end, out_dir)


if __name__ == "__main__":
    main()
