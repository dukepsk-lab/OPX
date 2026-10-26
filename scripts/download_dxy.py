"""Dukascopy DXY (DOLLAR.IDX/USD) M1 data downloader.

Downloads historical M1 bars for the US Dollar Index (DOLLAR.IDX/USD)
from Dukascopy, formats to the OPX schema, and consolidates to data/dxy_m1.parquet.
"""

from datetime import datetime, timezone
from pathlib import Path
import dukascopy_python as dp
import pandas as pd

DXY_SYMBOL = "DOLLAR.IDX/USD"
START_YEAR = 2014
END_YEAR = 2026


def format_dxy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Format Dukascopy DXY raw output to OPX standard bar dataframe schema."""
    out = df.copy()
    if "index" in out.columns and "timestamp" not in out.columns:
        out = out.rename(columns={"index": "timestamp"})

    if "timestamp" not in out.columns:
        raise ValueError("DataFrame missing timestamp column")

    # Ensure UTC Datetime
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    elif out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize("UTC")
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert("UTC")

    target_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in target_cols:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")

    out = out[target_cols].sort_values("timestamp").reset_index(drop=True)
    return out


def download_year(year: int, temp_file: Path) -> bool:
    if temp_file.exists():
        print(f"  [Cache hit] DXY {year} already downloaded.")
        return True

    now = datetime.now(timezone.utc)
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    if year == now.year:
        end = now
    else:
        end = datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc)

    print(f"  Fetching {DXY_SYMBOL} {year} ({start.date()} to {end.date()})...")
    try:
        df = dp.fetch(
            instrument=DXY_SYMBOL,
            interval=dp.INTERVAL_MIN_1,
            offer_side=dp.OFFER_SIDE_BID,
            start=start.replace(tzinfo=None),
            end=end.replace(tzinfo=None),
        )
    except Exception as e:
        print(f"  Error fetching {DXY_SYMBOL} {year}: {e}")
        return False

    if df is None or df.empty:
        print(f"  No bars returned for {DXY_SYMBOL} {year}")
        return False

    df = df.reset_index().rename(columns={"index": "timestamp"})
    formatted = format_dxy_dataframe(df)

    temp_file.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_parquet(temp_file, index=False)
    print(f"  Saved {len(formatted):,} bars to {temp_file.name}")
    return True


def consolidate_dxy(cache_dir: Path, out_dir: Path):
    year_files = sorted(cache_dir.glob("dxy_*.parquet"))
    if not year_files:
        print("No cache files found to consolidate for DXY")
        return

    print(f"Consolidating {len(year_files)} years for DXY...")
    dfs = [pd.read_parquet(f) for f in year_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out_file = out_dir / "dxy_m1.parquet"
    combined.to_parquet(out_file, index=False)
    print(
        f"==> Successfully generated {out_file} with {len(combined):,} total M1 bars "
        f"({combined['timestamp'].min()} to {combined['timestamp'].max()})"
    )


def main():
    cache_dir = Path("data/cache")
    out_dir = Path("data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print(f"Processing DXY ({DXY_SYMBOL}) from {START_YEAR} to {END_YEAR}")
    print("==========================================")
    for year in range(START_YEAR, END_YEAR + 1):
        temp_file = cache_dir / f"dxy_{year}.parquet"
        download_year(year, temp_file)

    consolidate_dxy(cache_dir, out_dir)


if __name__ == "__main__":
    main()
