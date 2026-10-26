# OPX Architecture

Companion to [summary.md](summary.md). This is the spec the [implementation plan](superpowers/plans/2026-10-26-opx-pipeline.md) builds from.

## Goal and non-goals

**Goal:** a Python research harness that takes M1 FX bars, trains cost-weighted GBDT models for Call and Put at k ∈ {1, 5, 15} minutes, and produces an honest **go/no-go report** against the validation checklist in R1.

**Non-goals (v1):**
- Auto-execution on any broker, including unofficial APIs. See "Risks" in the summary.
- OTC or synthetic feeds.
- A broker settlement-feed recorder. It needs a reverse-engineered API. v1 uses interbank M1 bars (Dukascopy-style CSV) and treats the basis to the settlement feed as an open risk.
- Sequence models, CatBoost/XGBoost, Optuna, TA-Lib. Add them only if the baseline shows a real signal.

## Data flow

```
bars CSV/parquet ──load_bars──► bars (UTC M1 OHLC + tick_volume)
                                   │
              ┌────────────────────┴─────────────┐
              ▼                                  ▼
        make_labels(k, slip)              build_features
        y_call, y_put, t1                 causal feature rows
              └────────────────┬─────────────────┘
                               ▼
              purged_walk_forward splits (train rows with t1 < test start)
                               ▼
        per fold: fit EV-weighted LightGBM (train)
                  fit Platt calibrator (chronological tail of train)
                  predict calibrated p on test fold
                               ▼
        oos frame: time, side, p_cal, y, k          (one row per bar and side)
                               ▼
        select trades: p_cal > breakeven(r) + buffer
                               ▼
        gate: Wilson lower bound, DSR, PBO, permutation test,
              payout stress, lag stress, quarterly stability
                               ▼
        report (text checklist)  ──► live.decide + Telegram alert (only if GO)
```

## Modules

All under `src/opx/`. One responsibility per file.

| Module | Responsibility | Key public API |
|---|---|---|
| `econ.py` | Payout maths | `breakeven(r)`, `ev_per_trade(p, r)`, `kelly_fraction(p, r)` |
| `data.py` | Load and validate bars | `load_bars(path) -> DataFrame` |
| `labels.py` | Settlement-consistent labels | `make_labels(bars, k, slip) -> DataFrame` |
| `features.py` | Causal feature matrix | `build_features(bars) -> DataFrame` |
| `cv.py` | Purged walk-forward splits | `purged_walk_forward(t0, t1, n_splits) -> Iterator` |
| `objective.py` | EV-weighted logloss for LightGBM | `EVObjective(r)` |
| `model.py` | Train, calibrate, predict | `fit_model`, `fit_calibrator`, `predict_win_prob` |
| `evaluate.py` | Trade selection and basic stats | `select_trades`, `summarize` |
| `stats.py` | Multiple-testing and significance | `wilson_lcb`, `deflated_sharpe`, `permutation_pvalue`, `pbo` |
| `walkforward.py` | Glue: bars to out-of-sample frame | `run_walk_forward(bars, k, r, ...) -> DataFrame` |
| `gate.py` | Go/no-go checklist | `go_nogo(...) -> GateResult`, CLI `python -m opx.gate` |
| `live.py` | Alert-only decision and sizing | `decide(...) -> Decision \| None`, `send_telegram` |

## Data contracts

**bars:** `DatetimeIndex` (UTC, sorted, unique), columns `open, high, low, close, tick_volume`. One row per minute. Gaps are allowed and must not be silently filled.

**labels:** same index as bars. Columns `y_call, y_put` (1.0, 0.0 or NaN) and `t1` (label end time). A row is NaN when bar t+1 or bar t+k is not exactly the expected number of minutes later, which covers weekends and holidays.
- Entry is `open[t+1]`. Expiry is `close[t+k]`.
- Call wins when `close[t+k] − open[t+1] > slip`. Put wins when `open[t+1] − close[t+k] > slip`. Ties and sub-slip moves are losses for both.

**features:** same index as bars. Row t uses only bars ≤ t. Warm-up rows are NaN and are dropped by the caller.

**oos frame:** one row per (bar, side) with columns `time, side ('call'|'put'), p_cal, y, t1`. This is the only thing `evaluate`, `stats` and `gate` consume.

## Key design decisions

1. **Cost-weighted logloss**, not a custom EV loss. It is convex with a positive Hessian, and `p* = 1/(1+r)` becomes the zero-logit boundary. A direct EV loss has a Hessian that changes sign.
2. **Separate Call and Put models.** A Put model trained on `1 − y_call` would be paid for ties it actually loses.
3. **Calibrate on the chronological tail of each training window**, with the last `k` rows purged, so the calibrator never sees the model's training rows.
4. **Threshold is `breakeven(r) + buffer`.** The buffer is a parameter. Every buffer tried is logged as a trial for the Deflated Sharpe Ratio.
5. **The gate is code, not judgement.** `go_nogo` returns a pass/fail per checklist item, and any failure means no-go. PBO compares the buffers actually tried. With a single config there is no selection to overfit, so PBO is reported as N/A and passes. Over near-identical configs it sits near 0.5 whatever the edge, so treat it as meaningful only when the configs are genuinely different.
6. **Planted-signal sanity test.** The pipeline must reject pure noise and find a deliberately planted mean-reversion signal. If it fails either, the pipeline is broken regardless of the real-data result.
7. **Pandas and numpy only** for indicators. TA-Lib's Windows install is painful and the indicators are a few lines each.

## Stack

Python 3.11, pandas, numpy, pyarrow, lightgbm, scikit-learn, scipy, httpx (Telegram), pytest. No other runtime dependencies.

## Directory layout

```
OPX/
  pyproject.toml
  docs/{summary.md, architecture.md, research/, superpowers/plans/}
  src/opx/{__init__,econ,data,labels,features,cv,objective,model,
           evaluate,stats,walkforward,gate,live}.py
  tests/{conftest.py, test_<module>.py}
  data/            # git-ignored: bars, logs
```

## Open risks

- **Feed basis:** settlement is on the broker's quotes, v1 trains on interbank bars. A positive result is necessary, not sufficient. The next step after a GO is recording platform quotes for a demo forward test.
- **Sample size:** to show p > 0.5405 at 95% confidence, R1 estimates about 760 trades at a true p of 0.57 and about 1,750 at 0.56. Budget at least a year of M1 data per pair.
- **Overlap:** M5/M15 labels overlap, so the effective N is smaller than the row count. The gate uses non-overlapping trades only (one trade per k minutes).
