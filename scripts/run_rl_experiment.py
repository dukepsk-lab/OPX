"""Multi-pair RL Meta-Agent experiment runner using PPO on non-raw-price state features."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from opx.data import load_bars
from opx.model import calibrate, fit_calibrator, fit_model, predict_win_prob
from opx.rl.dataset import build_rl_state_matrix
from opx.rl.env import MultiPairBinaryOptionsEnv
from opx.rl.train import evaluate_agent_oos, train_ppo_agent
from opx.labels import make_labels
from opx.features import build_features

PAIRS = ["eurusd", "gbpusd", "usdjpy", "audusd", "usdcad"]


def generate_pair_ml_predictions(
    bars: pd.DataFrame,
    k: int = 5,
    r: float = 0.85,
    slip: float = 0.00002,
    train_frac: float = 0.7,
) -> pd.DataFrame:
    """Train LightGBM on train_frac and predict out-of-sample calibrated probabilities."""
    feat = build_features(bars)
    labels = make_labels(bars, k=k, slip=slip)

    valid = feat.notna().all(axis=1) & labels["y_call"].notna() & labels["y_put"].notna()
    f_valid = feat[valid]
    l_valid = labels[valid]

    n = len(f_valid)
    n_tr = int(n * train_frac)
    n_model = int(n_tr * 0.7)

    X = f_valid.to_numpy(dtype=float)
    X_m, X_cal, X_test = X[:n_model], X[n_model:n_tr], X[n_tr:]

    res_dict: dict[str, np.ndarray] = {}

    for side in ["call", "put"]:
        y = l_valid[f"y_{side}"].to_numpy(dtype=float)
        y_m, y_cal, y_test = y[:n_model], y[n_model:n_tr], y[n_tr:]

        model = fit_model(X_m, y_m, r=r, params={"n_estimators": 100, "verbose": -1})
        p_calib_raw = predict_win_prob(model, X_cal, r=r)
        cal = fit_calibrator(p_calib_raw, y_cal)

        # Predict for all
        p_all_raw = predict_win_prob(model, X, r=r)
        p_all_cal = calibrate(cal, p_all_raw)
        res_dict[f"p_{side}"] = p_all_cal

    df = pd.DataFrame(res_dict, index=f_valid.index)
    return df


def run_experiment(
    k: int = 5,
    r: float = 0.85,
    slip: float = 0.00002,
    timesteps: int = 100000,
    max_bars: int = 300000,
    data_dir: Path = Path("data"),
    out_dir: Path = Path("results"),
) -> dict[str, Any]:
    print("\n==========================================")
    print("Phase 3: Multi-Pair RL Meta-Agent Training")
    print(f"Pairs: {PAIRS}")
    print(f"Horizon: k={k}m | Payout: {r} | Timesteps: {timesteps:,}")
    print("==========================================", flush=True)

    # 1. Load bars
    pairs_bars: dict[str, pd.DataFrame] = {}
    for p in PAIRS:
        p_file = data_dir / f"{p}_m1.parquet"
        if not p_file.exists():
            raise FileNotFoundError(f"Missing data file: {p_file}")
        bars = load_bars(p_file)
        if max_bars is not None and len(bars) > max_bars:
            bars = bars.iloc[-max_bars:]
        pairs_bars[p] = bars
        print(f"Loaded {p.upper()}: {len(bars):,} bars", flush=True)

    dxy_file = data_dir / "dxy_m1.parquet"
    dxy_bars = load_bars(dxy_file) if dxy_file.exists() else None
    if dxy_bars is not None and max_bars is not None and len(dxy_bars) > max_bars:
        dxy_bars = dxy_bars.iloc[-max_bars:]

    # 2. Generate ML predictions for all pairs
    print("\nGenerating ML probability signals for all 5 pairs...", flush=True)
    pairs_preds: dict[str, pd.DataFrame] = {}
    for p, bars in pairs_bars.items():
        print(f"  Fitting ML model for {p.upper()}...", flush=True)
        pairs_preds[p] = generate_pair_ml_predictions(bars, k=k, r=r, slip=slip)

    # 3. Assemble RL State Matrix (Excluding raw price)
    print("\nAssembling RL state matrix (strictly excluding raw OHLC prices)...", flush=True)
    state_df, outcomes_df = build_rl_state_matrix(
        pairs_bars=pairs_bars,
        pairs_preds=pairs_preds,
        dxy_bars=dxy_bars,
        k=k,
        slip=slip,
    )
    print(f"State Matrix shape: {state_df.shape} (Features: {state_df.shape[1]})", flush=True)
    print(f"Outcomes shape: {outcomes_df.shape}", flush=True)

    # Verify no raw price columns
    for col in state_df.columns:
        c_low = col.lower()
        for raw in ["open", "high", "low", "close"]:
            assert c_low != raw and not c_low.endswith(f"_{raw}"), f"Found raw price column {col} in state!"

    # 4. Train/Test Split (70% train, 30% test)
    n = len(state_df)
    n_tr = int(n * 0.70)
    print(f"\nDataset Split: Train={n_tr:,} bars (70%) | Out-of-Sample Test={n - n_tr:,} bars (30%)", flush=True)

    train_states = state_df.iloc[:n_tr].to_numpy(dtype=np.float32)
    train_outcomes = outcomes_df.iloc[:n_tr]

    test_states = state_df.iloc[n_tr:].to_numpy(dtype=np.float32)
    test_outcomes = outcomes_df.iloc[n_tr:]

    train_env = MultiPairBinaryOptionsEnv(
        states=train_states,
        outcomes=train_outcomes,
        pairs=PAIRS,
        k=k,
        r=r,
    )
    test_env = MultiPairBinaryOptionsEnv(
        states=test_states,
        outcomes=test_outcomes,
        pairs=PAIRS,
        k=k,
        r=r,
    )

    # 5. Train PPO Agent
    print(f"\nTraining PPO Agent for {timesteps:,} timesteps...", flush=True)
    t0 = time.time()
    agent = train_ppo_agent(train_env, total_timesteps=timesteps, lr=3e-4, seed=42)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s", flush=True)

    # 6. Evaluate on Out-of-Sample Test Environment
    print("\nEvaluating Agent Out-of-Sample (Unseen Market Data)...", flush=True)
    metrics = evaluate_agent_oos(agent, test_env)
    metrics["train_time_sec"] = round(train_time, 1)
    metrics["n_features"] = state_df.shape[1]
    metrics["timesteps"] = timesteps

    # Print summary
    win_pct = metrics["win_rate"] * 100.0
    lcb_pct = metrics["lcb"] * 100.0
    print("\n==========================================")
    print("RL META-AGENT OUT-OF-SAMPLE RESULTS")
    print("==========================================")
    print(f"Total Steps Evaluated: {metrics['total_steps']:,}")
    print(f"Trades Taken:          {metrics['total_trades']:,} (Wins: {metrics['wins']:,}, Losses: {metrics['losses']:,})")
    print(f"Win Rate:              {win_pct:.2f}% (Breakeven: {100.0/(1.0+r):.2f}%)")
    print(f"Wilson 95% LCB:        {lcb_pct:.2f}%")
    print(f"Cumulative PnL:        {metrics['cumulative_pnl']:+.2f} R")
    print(f"Sharpe Ratio:          {metrics['sharpe']:.3f}")
    print(f"Action Distribution:   {metrics['action_distribution']}")
    print("==========================================", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "rl_meta_agent_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved results to {out_file}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-pair RL Meta-Agent experiment")
    parser.add_argument("--k", type=int, default=5, help="Expiry horizon in minutes")
    parser.add_argument("--r", type=float, default=0.85, help="Payout ratio")
    parser.add_argument("--slip", type=float, default=0.00002, help="Slippage threshold")
    parser.add_argument("--timesteps", type=int, default=100000, help="PPO total timesteps")
    parser.add_argument("--max-bars", type=int, default=200000, help="Max bars per pair for experiment")
    args = parser.parse_args()

    run_experiment(
        k=args.k,
        r=args.r,
        slip=args.slip,
        timesteps=args.timesteps,
        max_bars=args.max_bars,
    )


if __name__ == "__main__":
    main()
