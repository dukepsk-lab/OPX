"""Ablation study runner for single and pairwise feature combinations.

Evaluates baseline vs:
  - +dxy
  - +cross
  - +squeeze
  - +fracdiff
  - Pairwise combinations
Measures win rate, Wilson 95% LCB, DSR, PBO, and expected value lift on OOS predictions.
"""

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from opx.data import load_bars
from opx.evaluate import select_trades, summarize
from opx.features_extra import (
    compute_cross_returns,
    compute_dxy_features,
    compute_fracdiff,
    compute_vol_squeeze,
)
from opx.stats import deflated_sharpe, pbo
from opx.walkforward import run_walk_forward


def evaluate_ablation_oos(
    oos: pd.DataFrame,
    r: float = 0.85,
    buffers: list[float] | None = None,
    n_trials: int = 11,
) -> dict[str, Any]:
    """Evaluate out-of-sample prediction frame across buffers and return best metrics."""
    if buffers is None:
        buffers = [0.0, 0.02, 0.04, 0.06]

    best_res: dict[str, Any] | None = None
    best_score = -999.0
    buffer_returns: list[np.ndarray] = []

    for buf in buffers:
        trades = select_trades(oos, r=r, buffer=buf)
        s = summarize(trades, r=r)

        # Build trade-by-trade returns for Sharpe / DSR: win = +r, loss = -1.0
        if not trades.empty:
            ret_series = np.where(trades["y"].to_numpy() == 1.0, r, -1.0)
        else:
            ret_series = np.array([], dtype=float)
        buffer_returns.append(ret_series)

        # Ranking score: Wilson LCB
        score = s["lcb"] if s["n"] > 0 else 0.0
        if score > best_score:
            best_score = score
            best_res = {
                "best_buffer": buf,
                "n_trades": s["n"],
                "win_rate": s["win_rate"],
                "lcb": s["lcb"],
                "breakeven": s["breakeven"],
                "ev": s["ev"],
            }

    if best_res is None or best_res["n_trades"] == 0:
        return {
            "best_buffer": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "lcb": 0.0,
            "breakeven": 1.0 / (1.0 + r),
            "ev": -1.0,
            "dsr": 0.0,
            "pbo": 1.0,
        }

    # Deflated Sharpe Ratio calculation
    best_ret = buffer_returns[buffers.index(best_res["best_buffer"])]
    dsr_val = deflated_sharpe(best_ret, n_trials=n_trials)

    # PBO calculation
    # Align returns into a 2D matrix for non-empty buffers
    valid_rets = [r_arr for r_arr in buffer_returns if len(r_arr) >= 16]
    if len(valid_rets) >= 2:
        min_len = min(len(r_arr) for r_arr in valid_rets)
        M = np.column_stack([r_arr[:min_len] for r_arr in valid_rets])
        pbo_val = pbo(M)
    else:
        pbo_val = 0.5

    best_res["dsr"] = dsr_val
    best_res["pbo"] = pbo_val
    return best_res


def prepare_extra_feature_cache(
    bars: pd.DataFrame,
    data_dir: Path,
    target_pair: str,
) -> dict[str, pd.DataFrame]:
    """Precompute each extra feature group once to allow fast combinatorial ablation."""
    print("Precomputing extra feature cache...")
    cache: dict[str, pd.DataFrame] = {}

    # 1. Vol Squeeze
    print("  Computing Vol Squeeze features...")
    cache["squeeze"] = compute_vol_squeeze(bars)

    # 2. FracDiff
    print("  Computing Fractional Differentiation (d=0.35)...")
    fd_series = compute_fracdiff(bars["close"], d=0.35)
    cache["fracdiff"] = pd.DataFrame({"fracdiff_0.35": fd_series}, index=bars.index)

    # 3. DXY
    dxy_path = data_dir / "dxy_m1.parquet"
    if dxy_path.exists():
        print("  Computing DXY Dollar Index features...")
        dxy_bars = load_bars(dxy_path)
        cache["dxy"] = compute_dxy_features(bars.index, dxy_bars)
    else:
        print("  Notice: DXY data not found, skipping DXY feature cache.")

    # 4. Cross-pair returns
    pair_names = ["eurusd", "gbpusd", "usdjpy", "audusd", "usdcad"]
    other_pairs: dict[str, pd.DataFrame] = {}
    for p in pair_names:
        if p == target_pair.lower():
            continue
        p_path = data_dir / f"{p}_m1.parquet"
        if p_path.exists():
            other_pairs[p] = load_bars(p_path)

    if other_pairs:
        print(f"  Computing Cross-pair features for {list(other_pairs.keys())}...")
        cache["cross"] = compute_cross_returns(bars.index, other_pairs)

    print("Extra feature precomputation complete.")
    return cache


CONFIGURATIONS = [
    ("baseline", []),
    ("+dxy", ["dxy"]),
    ("+cross", ["cross"]),
    ("+squeeze", ["squeeze"]),
    ("+fracdiff", ["fracdiff"]),
    ("+dxy+cross", ["dxy", "cross"]),
    ("+dxy+squeeze", ["dxy", "squeeze"]),
    ("+dxy+fracdiff", ["dxy", "fracdiff"]),
    ("+cross+squeeze", ["cross", "squeeze"]),
    ("+cross+fracdiff", ["cross", "fracdiff"]),
    ("+squeeze+fracdiff", ["squeeze", "fracdiff"]),
]


def run_ablation(
    pair: str,
    k: int = 5,
    r: float = 0.85,
    slip: float = 0.00002,
    data_dir: Path = Path("data"),
    out_dir: Path = Path("results"),
    max_bars: int | None = None,
    start_date: str | None = "2018-01-01",
) -> dict[str, Any]:
    pair_file = data_dir / f"{pair.lower()}_m1.parquet"
    if not pair_file.exists():
        raise FileNotFoundError(f"Target pair data not found: {pair_file}")

    print(f"\n==========================================")
    print(f"Starting Phase 1 Feature Ablation: {pair.upper()} k={k} (payout={r})")
    print(f"==========================================")

    bars = load_bars(pair_file)
    if start_date is not None:
        bars = bars[bars.index >= pd.Timestamp(start_date, tz="UTC")]
    if max_bars is not None and len(bars) > max_bars:
        bars = bars.iloc[-max_bars:]

    print(f"Loaded {len(bars):,} bars for {pair.upper()} ({bars.index.min()} to {bars.index.max()})")

    cache = prepare_extra_feature_cache(bars, data_dir, target_pair=pair)

    results: dict[str, Any] = {}

    for name, groups in CONFIGURATIONS:
        # Check if all required groups exist in cache
        missing = [g for g in groups if g not in cache]
        if missing:
            print(f"\nSkipping {name}: missing cached features {missing}", flush=True)
            continue

        print(f"\n---> Evaluating Config: {name} (groups: {groups})", flush=True)
        if groups:
            extra_dfs = [cache[g] for g in groups]
            extra_features = pd.concat(extra_dfs, axis=1)
        else:
            extra_features = None

        oos = run_walk_forward(
            bars=bars,
            k=k,
            r=r,
            slip=slip,
            n_splits=6,
            skip_illiquid=True,
            extra_features=extra_features,
        )

        metrics = evaluate_ablation_oos(oos, r=r, n_trials=len(CONFIGURATIONS))
        metrics["n_features"] = 24 + (extra_features.shape[1] if extra_features is not None else 0)
        results[name] = metrics

        lcb_pct = metrics["lcb"] * 100.0
        win_pct = metrics["win_rate"] * 100.0
        print(
            f"     Result {name:18s}: Win={win_pct:.2f}% | LCB={lcb_pct:.2f}% | "
            f"DSR={metrics['dsr']:.3f} | PBO={metrics['pbo']:.3f} | Trades={metrics['n_trades']:,}",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ablation_{pair.lower()}_k{k}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved ablation study results to {out_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Feature ablation runner")
    parser.add_argument("--pair", type=str, default="gbpusd", help="Pair to test")
    parser.add_argument("--k", type=int, default=5, help="Expiry horizon in minutes")
    parser.add_argument("--r", type=float, default=0.85, help="Payout ratio")
    parser.add_argument("--slip", type=float, default=0.00002, help="Slippage threshold")
    parser.add_argument("--start-date", type=str, default="2018-01-01", help="Start date filter for common period")
    parser.add_argument("--max-bars", type=int, default=None, help="Optional max bars for speed testing")
    args = parser.parse_args()

    run_ablation(
        pair=args.pair,
        k=args.k,
        r=args.r,
        slip=args.slip,
        max_bars=args.max_bars,
        start_date=args.start_date,
    )


if __name__ == "__main__":
    main()
