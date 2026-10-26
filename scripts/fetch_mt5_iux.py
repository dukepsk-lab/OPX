"""Fetch historical M1 data from IUX Markets MT5 terminal and save to Parquet.

Exports clean M1 OHLCV + spread bars for DXY. and FX pairs into data/iux/.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

DEFAULT_MT5_PATH = r"C:\Program Files\IUX Markets MT5 Terminal\terminal64.exe"
DEFAULT_SYMBOLS = [
    "DXY.",
    "EURUSD.",
    "GBPUSD.",
    "USDJPY.",
    "AUDUSD.",
    "USDCAD.",
    "USDCHF.",
    "NZDUSD.",
    "XAUUSD.",
]


def fetch_symbol_m1(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame | None:
    """Fetch M1 bars for a symbol from MT5."""
    if not mt5.symbol_select(symbol, True):
        print(f"  [ERROR] Failed to select symbol {symbol}: {mt5.last_error()}")
        return None

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        print(f"  [ERROR] No rates returned for {symbol}: {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    # Convert POSIX timestamp to UTC DatetimeIndex
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})

    # Reorder columns
    keep_cols = ["time", "open", "high", "low", "close", "volume", "spread"]
    cols = [c for c in keep_cols if c in df.columns]
    df = df[cols].sort_values("time").reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch M1 data from IUX MT5")
    parser.add_argument(
        "--path",
        default=DEFAULT_MT5_PATH,
        help="Path to IUX Markets MT5 terminal64.exe",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbols to fetch",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2023,
        help="Start year (default: 2023)",
    )
    parser.add_argument(
        "--outdir",
        default="data/iux",
        help="Directory to save Parquet files (default: data/iux)",
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    start_dt = datetime(args.start_year, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc)

    print("==========================================")
    print("Fetching M1 Data from IUX Markets MT5")
    print(f"Terminal Path: {args.path}")
    print(f"Symbols ({len(symbols)}): {symbols}")
    print(f"Date Range: {start_dt.date()} to {end_dt.date()}")
    print(f"Output Directory: {out_dir.resolve()}")
    print("==========================================")

    if not mt5.initialize(path=args.path):
        print(f"[FATAL] Failed to initialize MT5 at {args.path}: {mt5.last_error()}")
        sys.exit(1)

    acc = mt5.account_info()
    if acc:
        print(f"Connected to Account: {acc.login} | Server: {acc.server} | Company: {acc.company}")
    print("------------------------------------------")

    results = []
    for sym in symbols:
        clean_name = sym.rstrip(".").lower()
        print(f"Fetching {sym} (M1)...")
        df = fetch_symbol_m1(sym, start_dt, end_dt)
        if df is not None:
            out_file = out_dir / f"{clean_name}_m1.parquet"
            df.to_parquet(out_file, index=False, engine="pyarrow", compression="zstd")
            file_mb = out_file.stat().st_size / (1024 * 1024)
            t_min = df["time"].iloc[0].strftime("%Y-%m-%d %H:%M")
            t_max = df["time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
            print(f"  -> Saved {len(df):,} bars to {out_file.name} ({file_mb:.2f} MB)")
            print(f"     Time Range: {t_min} -> {t_max}")
            results.append({
                "symbol": sym,
                "clean_name": clean_name,
                "bars": len(df),
                "start": t_min,
                "end": t_max,
                "file_mb": file_mb,
            })
        else:
            print(f"  -> Failed to fetch {sym}")

    mt5.shutdown()
    print("==========================================")
    print(f"Summary: Successfully fetched and saved {len(results)}/{len(symbols)} symbols.")
    total_bars = sum(r["bars"] for r in results)
    total_mb = sum(r["file_mb"] for r in results)
    print(f"Total M1 Bars: {total_bars:,} | Total Size: {total_mb:.2f} MB")
    print("==========================================")


if __name__ == "__main__":
    main()
