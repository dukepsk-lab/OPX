# Phase 1: Feature Engineering Suite (DXY, Cross-Asset, Volatility Squeeze, FracDiff) & Ablation Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement new feature extractors (Dukascopy DXY Dollar Index, Cross-Asset returns, Volatility Squeeze, and Fractional Differentiation), integrate them into the OPX pipeline with zero data leakage, and execute a fast ablation study (individual and pairwise combinations) on GBPUSD k=5 to report the performance lift against the baseline.

**Architecture:** A modular feature extension package (`src/opx/features_extra.py`) computes causal signals aligned on UTC M1 timestamps. A flexible feature registry allows dynamic feature ablation in `run_walk_forward`, and an automated ablation runner (`scripts/ablation_study.py`) runs walk-forward validation across single and pairwise feature combinations, recording win rates, Wilson LCB, DSR, and EV metrics.

**Tech Stack:** Python 3.11, pandas, numpy, lightgbm, scikit-learn, scipy, dukascopy-python, pytest.

**Spec:** Section 9 of `docs/full-report.md` and user requirements (Approach A: DXY + Cross-Asset + Squeeze + FracDiff with individual & pair ablation).

## Global Constraints

- **Causality & Zero Leakage:** Row $t$ must use strictly information $\le t$. When merging DXY and other pairs, prices must be lagged so that bar $t$ only uses completed bars at or before $t$.
- **Illiquidity Filtering:** All walk-forward runs must respect `skip_illiquid=True` (drop NY 16:00–18:59 and NY Sundays) to prevent the rollover artifact discovered in `docs/audit-eurusd-k15.md`.
- **Settlement Invariant:** Entry remains strictly at `open[t+1]` and expiry at `close[t+k]`, tie is loss.
- **No Infinite or Unhandled NaNs:** All features must produce finite floats or NaN, never `+inf` or `-inf`.

## Review Focus

1. **DXY timestamp misalignment or weekend gaps:** DXY bars might open/close slightly differently from FX pairs; merging must use forward-fill up to a strict safe tolerance (e.g. max 5 minutes) without looking ahead.
2. **Fractional differentiation memory window:** FracDiff convolution weights must truncate cleanly without leaking future data.
3. **Cross-pair lookahead:** Merging 5 pairs must be done strictly on closed bars index `t`.
4. **Computational speed during ablation:** Ablation runs multiple feature sets; feature matrices should be precomputed or cached once per pair rather than recalculated inside every cross-validation fold.
5. **DSR trial penalty:** Every tested feature combination must be logged so that multiple-testing penalty is accurately maintained.

---

### Task 1: DXY Data Download Script (`DOLLAR.IDX/USD`)

**Files:**
- Create: `scripts/download_dxy.py`
- Test: `tests/test_download_dxy.py`

**Interfaces:**
- Consumes: `dukascopy_python` API with instrument `DOLLAR.IDX/USD`.
- Produces: `data/dxy_m1.parquet` with columns `["timestamp", "open", "high", "low", "close", "volume"]` in UTC.

- [ ] **Step 1: Write unit test to verify DXY parser and schema formatting**

```python
# tests/test_download_dxy.py
import pandas as pd
from scripts.download_dxy import format_dxy_dataframe

def test_format_dxy_dataframe():
    raw = pd.DataFrame({
        "timestamp": [pd.Timestamp("2023-01-01 00:00:00", tz="UTC")],
        "open": [103.5],
        "high": [103.6],
        "low": [103.4],
        "close": [103.55],
        "volume": [12.0],
    })
    formatted = format_dxy_dataframe(raw)
    assert list(formatted.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert formatted["timestamp"].dt.tz is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_download_dxy.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `scripts/download_dxy.py`**

Fetch `DOLLAR.IDX/USD` from Dukascopy year-by-year (2015–2026), format to standard schema, and consolidate into `data/dxy_m1.parquet`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_download_dxy.py -v`
Expected: PASS

- [ ] **Step 5: Execute DXY download to generate `data/dxy_m1.parquet`**

Run: `.venv\Scripts\python scripts/download_dxy.py`
Expected: Download and save `data/dxy_m1.parquet`

- [ ] **Step 6: Commit**

```bash
git add scripts/download_dxy.py tests/test_download_dxy.py
git commit -m "feat: add DXY data download script and tests"
```

---

### Task 2: Implement New Feature Modules (`features_extra.py`)

**Files:**
- Create: `src/opx/features_extra.py`
- Test: `tests/test_features_extra.py`

**Interfaces:**
- Consumes: Target pair `bars` (pd.DataFrame), `dxy_bars` (optional pd.DataFrame), `other_pairs` (optional dict[str, pd.DataFrame]).
- Produces:
  - `compute_dxy_features(target_index, dxy_bars) -> pd.DataFrame`
  - `compute_cross_returns(target_index, other_pairs) -> pd.DataFrame`
  - `compute_vol_squeeze(bars) -> pd.DataFrame`
  - `compute_fracdiff(close_series, d=0.35, threshold=1e-4) -> pd.Series`

- [ ] **Step 1: Write failing tests for feature calculation and causality**

```python
# tests/test_features_extra.py
import numpy as np
import pandas as pd
from opx.features_extra import (
    compute_vol_squeeze,
    compute_fracdiff,
    compute_dxy_features,
    compute_cross_returns,
)

def test_vol_squeeze_causality_and_bounds():
    # Verify no future leakage and finite values
    ...

def test_fracdiff_stationarity_and_causality():
    # Verify causal weights (only past values used)
    ...

def test_dxy_alignment_no_lookahead():
    # Verify DXY merged with target bars does not look forward
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_features_extra.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/opx/features_extra.py`**

Implement:
1. `compute_dxy_features`: returns DXY log returns (1, 5, 15), RSI, and BB %B aligned strictly $\le t$.
2. `compute_cross_returns`: normalized returns of the other 4 pairs (e.g. for GBPUSD: EURUSD, USDJPY, AUDUSD, USDCAD).
3. `compute_vol_squeeze`: Bollinger Band width / ATR14 and Realized Volatility ratio (5-min / 60-min).
4. `compute_fracdiff`: Fractional differentiation of log price using expanding window weights with cutoff threshold.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_features_extra.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/opx/features_extra.py tests/test_features_extra.py
git commit -m "feat: implement DXY, cross-pair, vol squeeze, and fracdiff feature extractors"
```

---

### Task 3: Feature Matrix Builder & Integration with Walk-Forward

**Files:**
- Modify: `src/opx/features.py`
- Modify: `src/opx/walkforward.py`
- Test: `tests/test_walkforward_features.py`

**Interfaces:**
- Consumes: `feature_groups: list[str]` parameter (e.g. `['baseline', 'dxy', 'cross', 'squeeze', 'fracdiff']`).
- Produces: Feature matrix with requested feature sets cleanly merged on the valid bar index.

- [ ] **Step 1: Write test verifying walk-forward with dynamic feature sets**

```python
# tests/test_walkforward_features.py
from opx.walkforward import run_walk_forward_extended

def test_walk_forward_with_extra_features():
    # verify run_walk_forward_extended executes with baseline + squeeze without errors
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_walkforward_features.py -v`
Expected: FAIL

- [ ] **Step 3: Implement extended walk-forward runner**

Add `run_walk_forward_extended` supporting optional extra feature tables.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_walkforward_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/opx/features.py src/opx/walkforward.py tests/test_walkforward_features.py
git commit -m "feat: add extended walk-forward support for modular feature sets"
```

---

### Task 4: Fast Ablation Experiment Runner

**Files:**
- Create: `scripts/ablation_study.py`
- Test: `tests/test_ablation_study.py`

**Interfaces:**
- Consumes: Target pair parquet (`data/gbpusd_m1.parquet`), DXY parquet (`data/dxy_m1.parquet`), and other pairs.
- Produces: Execution of:
  - Baseline (24 features)
  - Single features: `+dxy`, `+cross`, `+squeeze`, `+fracdiff`
  - Pairwise combinations: `+dxy+cross`, `+dxy+squeeze`, `+dxy+fracdiff`, `+cross+squeeze`, `+cross+fracdiff`, `+squeeze+fracdiff`
  - Writes structured results to `results/ablation_gbpusd_k5.json`.

- [ ] **Step 1: Write unit test for ablation result aggregator**

```python
# tests/test_ablation_study.py
from scripts.ablation_study import evaluate_feature_set_metrics

def test_evaluate_feature_set_metrics():
    # verify win rate, Wilson LCB, and DSR calculation on dummy prediction dataframe
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_ablation_study.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `scripts/ablation_study.py`**

Implement parallel/cached feature extraction and sequential evaluation of all 11 configurations on `GBPUSD k=5`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_ablation_study.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ablation_study.py tests/test_ablation_study.py
git commit -m "feat: add ablation study runner for single and pairwise feature combinations"
```

---

### Task 5: Run Ablation Study, Analyze Lift, and Report

**Files:**
- Output: `results/ablation_gbpusd_k5.json`
- Output: `docs/ablation-report-gbpusd-k5.md`

- [ ] **Step 1: Execute `scripts/ablation_study.py` on GBPUSD k=5**

Run: `.venv\Scripts\python scripts/ablation_study.py --pair gbpusd --k 5`

- [ ] **Step 2: Compile comparative table of all 11 configurations**

Metrics to tabulate:
- Number of features
- Win rate (%)
- Wilson 95% LCB (vs 0.5405 breakeven)
- Deflated Sharpe Ratio (DSR)
- Probability of Backtest Overfitting (PBO)
- Net Expected Value per trade

- [ ] **Step 3: Generate and commit ablation report**

```bash
git add results/ablation_gbpusd_k5.json docs/ablation-report-gbpusd-k5.md
git commit -m "docs: record Phase 1 feature ablation study results on GBPUSD k=5"
```
