"""Batch runner for OPX gate validation across 5 pairs and horizons {1, 5, 15}."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import pandas as pd
from opx.data import load_bars
from opx.gate import CHECK_KEYS, go_nogo
from opx.walkforward import run_walk_forward


JOBS = [
    # pair, path, k, slip
    {"pair": "EURUSD", "path": "data/eurusd_m1.parquet", "k": 1, "slip": 0.00002},
    {"pair": "EURUSD", "path": "data/eurusd_m1.parquet", "k": 5, "slip": 0.00002},
    {"pair": "EURUSD", "path": "data/eurusd_m1.parquet", "k": 15, "slip": 0.00002},
    {"pair": "GBPUSD", "path": "data/gbpusd_m1.parquet", "k": 1, "slip": 0.00002},
    {"pair": "GBPUSD", "path": "data/gbpusd_m1.parquet", "k": 5, "slip": 0.00002},
    {"pair": "GBPUSD", "path": "data/gbpusd_m1.parquet", "k": 15, "slip": 0.00002},
    {"pair": "USDJPY", "path": "data/usdjpy_m1.parquet", "k": 1, "slip": 0.002},
    {"pair": "USDJPY", "path": "data/usdjpy_m1.parquet", "k": 5, "slip": 0.002},
    {"pair": "USDJPY", "path": "data/usdjpy_m1.parquet", "k": 15, "slip": 0.002},
    {"pair": "AUDUSD", "path": "data/audusd_m1.parquet", "k": 1, "slip": 0.00002},
    {"pair": "AUDUSD", "path": "data/audusd_m1.parquet", "k": 5, "slip": 0.00002},
    {"pair": "AUDUSD", "path": "data/audusd_m1.parquet", "k": 15, "slip": 0.00002},
    {"pair": "USDCAD", "path": "data/usdcad_m1.parquet", "k": 1, "slip": 0.00002},
    {"pair": "USDCAD", "path": "data/usdcad_m1.parquet", "k": 5, "slip": 0.00002},
    {"pair": "USDCAD", "path": "data/usdcad_m1.parquet", "k": 15, "slip": 0.00002},
]


def execute_job(job: dict[str, Any], r: float = 0.85, buffers: list[float] | None = None, n_other_trials: int = 15) -> dict[str, Any]:
    """Execute gate validation for one pair and horizon."""
    if buffers is None:
        buffers = [0.00, 0.02, 0.04, 0.06]

    pair = job["pair"]
    k = job["k"]
    path = job["path"]
    slip = job["slip"]

    print(f"[{pair} k={k}] Loading bars from {path}...", flush=True)
    t0 = time.time()
    bars = load_bars(path)
    n_bars = len(bars)
    start_date = str(bars.index[0])
    end_date = str(bars.index[-1])
    print(f"[{pair} k={k}] Loaded {n_bars:,} bars ({start_date} to {end_date}) in {time.time()-t0:.1f}s", flush=True)

    print(f"[{pair} k={k}] Running walk-forward (slip={slip})...", flush=True)
    t_wf = time.time()
    oos = run_walk_forward(bars, k=k, r=r, slip=slip)
    print(f"[{pair} k={k}] Base walk-forward done in {time.time()-t_wf:.1f}s (oos trades: {len(oos):,})", flush=True)

    print(f"[{pair} k={k}] Running stress walk-forward (slip={2.0*slip})...", flush=True)
    t_stress = time.time()
    oos_stress = run_walk_forward(bars, k=k, r=r, slip=2.0 * slip)
    print(f"[{pair} k={k}] Stress walk-forward done in {time.time()-t_stress:.1f}s", flush=True)

    print(f"[{pair} k={k}] Running go/no-go gate...", flush=True)
    t_gate = time.time()
    res = go_nogo(
        oos,
        r=r,
        buffers=buffers,
        n_other_trials=n_other_trials,
        oos_stress=oos_stress,
    )
    print(f"[{pair} k={k}] Gate completed in {time.time()-t_gate:.1f}s. Verdict: {'GO' if res.go else 'NO-GO'}", flush=True)

    # Find which primary buffer had highest total PnL
    best_buf = buffers[0]
    best_pnl = -float("inf")
    for buf in buffers:
        from opx.evaluate import select_trades
        tb = select_trades(oos, r, buf)
        pnl = float(np.sum(np.where(tb["y"] == 1.0, r, -1.0))) if len(tb) > 0 else 0.0
        if pnl > best_pnl:
            best_pnl = pnl
            best_buf = buf

    record = {
        "pair": pair,
        "k": k,
        "n_bars": n_bars,
        "start_date": start_date,
        "end_date": end_date,
        "slip": slip,
        "r": r,
        "primary_buffer": best_buf,
        "checks": res.checks,
        "details": res.details,
        "go": res.go,
        "verdict": "GO" if res.go else "NO-GO",
        "elapsed_s": round(time.time() - t0, 1),
    }

    # Print check details
    for chk in CHECK_KEYS:
        status = "PASS" if res.checks[chk] else "FAIL"
        det = res.details.get(chk, float("nan"))
        print(f"[{pair} k={k}] {status} [{chk}]: detail={det}", flush=True)

    return record


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "gate_results.json"

    existing_results: list[dict[str, Any]] = []
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []

    done_keys = {(r["pair"], r["k"]) for r in existing_results}
    pending_jobs = [j for j in JOBS if (j["pair"], j["k"]) not in done_keys]

    print(f"Total jobs: {len(JOBS)}, already completed: {len(done_keys)}, pending: {len(pending_jobs)}")
    if not pending_jobs:
        print("All jobs completed!")
        return

    workers = min(3, len(pending_jobs))
    print(f"Launching batch execution with {workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute_job, job): job for job in pending_jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                rec = fut.result()
                existing_results.append(rec)
                with open(results_file, "w") as f:
                    json.dump(existing_results, f, indent=2)
                print(f">>> SAVED result for {rec['pair']} k={rec['k']} (verdict: {rec['verdict']})")
            except Exception as e:
                print(f"ERROR running {job['pair']} k={job['k']}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
