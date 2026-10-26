"""Benchmark runner comparing LightGBM, CatBoost, XGBoost, and PyTorch 1D-CNN+BiLSTM."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from opx.data import load_bars
from opx.features_extra import compute_vol_squeeze
from opx.walkforward import run_walk_forward
from scripts.ablation_study import evaluate_ablation_oos

MODELS_TO_BENCHMARK = [
    ("lightgbm", {}),
    ("xgboost", {}),
    ("catboost", {}),
    ("cnn_lstm", {"epochs": 3, "batch_size": 512}),
]


def run_benchmark(
    pair: str = "gbpusd",
    k: int = 5,
    r: float = 0.85,
    slip: float = 0.00002,
    start_date: str = "2018-01-01",
    max_bars: int | None = None,
    data_dir: Path = Path("data"),
    out_dir: Path = Path("results"),
) -> dict[str, Any]:
    pair_file = data_dir / f"{pair.lower()}_m1.parquet"
    if not pair_file.exists():
        raise FileNotFoundError(f"Target pair data not found: {pair_file}")

    print(f"\n==========================================")
    print(f"Phase 2 Model Benchmark: {pair.upper()} k={k} (payout={r})")
    print(f"Comparing: LightGBM vs CatBoost vs XGBoost vs CNN+BiLSTM")
    print(f"==========================================", flush=True)

    bars = load_bars(pair_file)
    if start_date is not None:
        bars = bars[bars.index >= pd.Timestamp(start_date, tz="UTC")]
    if max_bars is not None and len(bars) > max_bars:
        bars = bars.iloc[-max_bars:]

    print(f"Loaded {len(bars):,} bars for {pair.upper()} ({bars.index.min()} to {bars.index.max()})", flush=True)

    print("Computing Vol Squeeze feature set (Phase 1 Winner)...", flush=True)
    extra_features = compute_vol_squeeze(bars)

    results: dict[str, Any] = {}

    for m_type, m_params in MODELS_TO_BENCHMARK:
        print(f"\n---> Running Model: {m_type.upper()}", flush=True)
        t_start = time.time()

        oos = run_walk_forward(
            bars=bars,
            k=k,
            r=r,
            slip=slip,
            n_splits=6,
            skip_illiquid=True,
            extra_features=extra_features,
            model_type=m_type,
            params=m_params,
        )

        elapsed = time.time() - t_start
        metrics = evaluate_ablation_oos(oos, r=r, n_trials=4)
        metrics["duration_sec"] = round(elapsed, 1)
        results[m_type] = metrics

        lcb_pct = metrics["lcb"] * 100.0
        win_pct = metrics["win_rate"] * 100.0
        print(
            f"     Result {m_type:12s}: Win={win_pct:.2f}% | LCB={lcb_pct:.2f}% | "
            f"DSR={metrics['dsr']:.3f} | PBO={metrics['pbo']:.3f} | "
            f"Trades={metrics['n_trades']:,} | Time={elapsed:.1f}s",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"benchmark_models_{pair.lower()}_k{k}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_file}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-model benchmark runner")
    parser.add_argument("--pair", type=str, default="gbpusd", help="Pair to test")
    parser.add_argument("--k", type=int, default=5, help="Expiry horizon in minutes")
    parser.add_argument("--r", type=float, default=0.85, help="Payout ratio")
    parser.add_argument("--slip", type=float, default=0.00002, help="Slippage threshold")
    parser.add_argument("--start-date", type=str, default="2018-01-01", help="Start date filter")
    parser.add_argument("--max-bars", type=int, default=None, help="Optional max bars for speed testing")
    args = parser.parse_args()

    run_benchmark(
        pair=args.pair,
        k=args.k,
        r=args.r,
        slip=args.slip,
        start_date=args.start_date,
        max_bars=args.max_bars,
    )


if __name__ == "__main__":
    main()
