"""Bulk downloader for 2015-2026 M1 FX data from Dukascopy.

Supports resumable year-by-year downloading and automatic parquet consolidation.
"""

from datetime import datetime
from pathlib import Path
import dukascopy_python as dp
import pandas as pd

PAIRS = {
    "eurusd": "EUR/USD",
    "gbpusd": "GBP/USD",
    "usdjpy": "USD/JPY",
    "audusd": "AUD/USD",
    "usdcad": "USD/CAD",
}

START_YEAR = 2015
END_YEAR = 2026


def download_year(symbol: str, year: int, temp_file: Path) -> bool:
    if temp_file.exists():
        print(f"  [Cache hit] {symbol} {year} already downloaded.")
        return True

    now = datetime.utcnow()
    start = datetime(year, 1, 1)
    if year == now.year:
        end = now
    else:
        end = datetime(year, 12, 31, 23, 59)

    print(f"  Fetching {symbol} {year} ({start.date()} to {end.date()})...")
    try:
        df = dp.fetch(
            instrument=symbol,
            interval=dp.INTERVAL_MIN_1,
            offer_side=dp.OFFER_SIDE_BID,
            start=start,
            end=end,
        )
    except Exception as e:
        print(f"  Error fetching {symbol} {year}: {e}")
        return False

    if df.empty:
        print(f"  No bars returned for {symbol} {year}")
        return False

    # Format dataframe
    df = df.reset_index().rename(columns={"index": "timestamp"})
    target_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[target_cols]

    temp_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(temp_file, index=False)
    print(f"  Saved {len(df):,} bars to {temp_file.name}")
    return True


def consolidate_pair(pair_name: str, cache_dir: Path, out_dir: Path):
    year_files = sorted(cache_dir.glob(f"{pair_name}_*.parquet"))
    if not year_files:
        print(f"No files found to consolidate for {pair_name}")
        return

    print(f"Consolidating {len(year_files)} years for {pair_name}...")
    dfs = [pd.read_parquet(f) for f in year_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out_file = out_dir / f"{pair_name}_m1.parquet"
    combined.to_parquet(out_file, index=False)
    print(f"==> Successfully generated {out_file} with {len(combined):,} total M1 bars ({combined['timestamp'].min()} to {combined['timestamp'].max()})")


def process_pair(pair_name: str, symbol: str, cache_dir: Path, out_dir: Path):
    print(f"\n==========================================")
    print(f"Processing {pair_name.upper()} ({symbol}) from {START_YEAR} to {END_YEAR}")
    print(f"==========================================")
    for year in range(START_YEAR, END_YEAR + 1):
        temp_file = cache_dir / f"{pair_name}_{year}.parquet"
        download_year(symbol, year, temp_file)

    consolidate_pair(pair_name, cache_dir, out_dir)


def main():
    cache_dir = Path("data/cache")
    out_dir = Path("data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair_name, symbol in PAIRS.items():
        process_pair(pair_name, symbol, cache_dir, out_dir)


if __name__ == "__main__":
    main()
