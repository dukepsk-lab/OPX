"""Go/no-go validation checklist and CLI reporting."""

import argparse
from dataclasses import dataclass
import sys
from typing import Any, Sequence
import numpy as np
import pandas as pd
from opx.data import load_bars
from opx.econ import breakeven
from opx.evaluate import select_trades, summarize
from opx.stats import deflated_sharpe, pbo, permutation_pvalue
from opx.walkforward import run_walk_forward

CHECK_KEYS = [
    "lcb_vs_breakeven",
    "dsr",
    "pbo",
    "permutation",
    "payout_stress",
    "lag_stress",
    "calibration",
    "quarters",
]


@dataclass(frozen=True)
class GateResult:
    checks: dict[str, bool]
    details: dict[str, float]
    go: bool


def go_nogo(
    oos: pd.DataFrame,
    r: float,
    buffers: Sequence[float],
    n_other_trials: int = 1,
    oos_stress: pd.DataFrame | None = None,
    min_quarters: int = 4,
) -> GateResult:
    """Evaluate statistical go/no-go checklist on out-of-sample trade predictions."""
    checks: dict[str, bool] = {k: False for k in CHECK_KEYS}
    details: dict[str, float] = {k: np.nan for k in CHECK_KEYS}

    if oos.empty or not buffers:
        return GateResult(checks=checks, details=details, go=False)

    # 1. Primary buffer selection: highest total PnL
    best_buf = buffers[0]
    best_pnl = -float("inf")
    best_trades = pd.DataFrame(columns=oos.columns)

    for buf in buffers:
        t_buf = select_trades(oos, r, buf)
        if len(t_buf) > 0:
            pnl = float(np.sum(np.where(t_buf["y"] == 1.0, r, -1.0)))
        else:
            pnl = 0.0

        if pnl > best_pnl:
            best_pnl = pnl
            best_buf = buf
            best_trades = t_buf

    primary_buffer = best_buf
    primary_trades = best_trades

    if len(primary_trades) == 0:
        return GateResult(checks=checks, details=details, go=False)

    s = summarize(primary_trades, r)
    lcb = s["lcb"]
    be = s["breakeven"]

    # Check 1: lcb_vs_breakeven
    checks["lcb_vs_breakeven"] = bool(lcb > be) if not np.isnan(lcb) else False
    details["lcb_vs_breakeven"] = float(lcb)

    # Check 2: dsr
    n_trials = len(buffers) * n_other_trials
    returns = np.where(primary_trades["y"] == 1.0, r, -1.0)
    dsr_val = deflated_sharpe(returns, n_trials=n_trials) if len(returns) >= 2 else 0.0
    checks["dsr"] = bool(dsr_val >= 0.95)
    details["dsr"] = float(dsr_val)

    # Check 3: pbo
    if len(buffers) == 1:
        checks["pbo"] = True
        details["pbo"] = np.nan
    else:
        daily_pnls: dict[float, pd.Series] = {}
        all_dates: set[Any] = set()
        for buf in buffers:
            tb = select_trades(oos, r, buf)
            if len(tb) > 0:
                dates = pd.to_datetime(tb["time"]).dt.date
                pnl_vals = np.where(tb["y"] == 1.0, r, -1.0)
                s_pnl = pd.Series(pnl_vals, index=dates).groupby(level=0).sum()
                daily_pnls[buf] = s_pnl
                all_dates.update(s_pnl.index)
            else:
                daily_pnls[buf] = pd.Series(dtype=float)

        if len(all_dates) < 16:
            checks["pbo"] = False
            details["pbo"] = np.nan
        else:
            sorted_dates = sorted(all_dates)
            mat = np.zeros((len(sorted_dates), len(buffers)))
            for c_idx, buf in enumerate(buffers):
                s_series = daily_pnls[buf]
                mat[:, c_idx] = [s_series.get(d, 0.0) for d in sorted_dates]
            pbo_val = pbo(mat, n_blocks=16)
            checks["pbo"] = bool(pbo_val <= 0.2)
            details["pbo"] = float(pbo_val)

    # Check 4: permutation
    take_mask = oos.index.isin(primary_trades.index)
    perm_p = permutation_pvalue(oos["y"].to_numpy(dtype=float), take_mask, r=r)
    checks["permutation"] = bool(perm_p < 0.01) if not np.isnan(perm_p) else False
    details["permutation"] = float(perm_p)

    # Check 5: payout_stress (r - 0.05)
    r_stress = r - 0.05
    if r_stress > 0:
        be_stress = breakeven(r_stress)
        checks["payout_stress"] = bool(lcb > be_stress) if not np.isnan(lcb) else False
        details["payout_stress"] = float(lcb - be_stress)
    else:
        checks["payout_stress"] = False
        details["payout_stress"] = np.nan

    # Check 6: lag_stress
    if oos_stress is None:
        checks["lag_stress"] = False
        details["lag_stress"] = np.nan
    else:
        trades_stress = select_trades(oos_stress, r, primary_buffer)
        if len(trades_stress) == 0:
            checks["lag_stress"] = False
            details["lag_stress"] = np.nan
        else:
            s_stress = summarize(trades_stress, r)
            lcb_stress = s_stress["lcb"]
            checks["lag_stress"] = bool(lcb_stress > be) if not np.isnan(lcb_stress) else False
            details["lag_stress"] = float(lcb_stress)

    # Check 7: calibration
    calib_err = abs(float(primary_trades["p_cal"].mean()) - float(primary_trades["y"].mean()))
    checks["calibration"] = bool(calib_err < 0.02)
    details["calibration"] = float(calib_err)

    # Check 8: quarters
    trade_times = pd.to_datetime(primary_trades["time"])
    quarters_series = trade_times.dt.tz_localize(None).dt.to_period("Q")
    eligible_quarters = []
    positive_quarters = []

    for q, idxs in primary_trades.groupby(quarters_series).groups.items():
        q_trades = primary_trades.loc[idxs]
        if len(q_trades) >= 30:
            eligible_quarters.append(q)
            q_s = summarize(q_trades, r)
            if q_s["ev"] > 0:
                positive_quarters.append(q)

    n_eligible = len(eligible_quarters)
    if n_eligible < min_quarters:
        checks["quarters"] = False
        details["quarters"] = float(n_eligible)
    else:
        pos_frac = len(positive_quarters) / n_eligible
        checks["quarters"] = bool(pos_frac >= 0.75)
        details["quarters"] = float(pos_frac)

    go = bool(all(checks.values()))
    return GateResult(checks=checks, details=details, go=go)


def main(argv: list[str] | None = None) -> int:
    """CLI runner for OPX gate validation report."""
    parser = argparse.ArgumentParser(description="OPX Go/No-Go Gate")
    parser.add_argument("--bars", required=True, help="Path to bars CSV or Parquet")
    parser.add_argument("--k", required=True, type=int, help="Expiry horizon in minutes")
    parser.add_argument("--r", required=True, type=float, help="Payout fraction")
    parser.add_argument("--buffers", required=True, type=float, nargs="+", help="Buffer fractions")
    parser.add_argument("--slip", type=float, default=0.0, help="Slippage")
    parser.add_argument("--n-other-trials", type=int, default=1, help="Other trials multiplier")

    args = parser.parse_args(argv)

    bars = load_bars(args.bars)
    oos = run_walk_forward(bars, k=args.k, r=args.r, slip=args.slip)
    oos_stress = run_walk_forward(bars, k=args.k, r=args.r, slip=2.0 * args.slip)

    result = go_nogo(
        oos,
        r=args.r,
        buffers=args.buffers,
        n_other_trials=args.n_other_trials,
        oos_stress=oos_stress,
    )

    for check in CHECK_KEYS:
        status = "PASS" if result.checks[check] else "FAIL"
        detail = result.details.get(check, float("nan"))
        print(f"{status} [{check}]: detail={detail}")

    print("MANUAL: lawful in your jurisdiction; forward demo test of 500-1000 non-overlapping trades; settlement-feed basis measured")

    if result.go:
        print("VERDICT: GO")
        return 0
    else:
        print("VERDICT: NO-GO")
        return 1


if __name__ == "__main__":
    sys.exit(main())
