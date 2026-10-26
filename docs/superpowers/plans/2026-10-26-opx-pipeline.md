# OPX Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python harness that trains cost-weighted LightGBM models on M1 FX bars for Call/Put at k ∈ {1, 5, 15} minutes and emits a code-enforced go/no-go report.

**Architecture:** Bars → settlement-consistent labels and causal features → purged walk-forward training with Platt calibration → out-of-sample frame → trade selection at `breakeven(r) + buffer` → statistical gate (Wilson bound, Deflated Sharpe Ratio, PBO, permutation, stress tests). Live use is alert-only and only after a GO.

**Tech Stack:** Python 3.11, pandas, numpy, pyarrow, lightgbm, scikit-learn, scipy, httpx, pytest.

**Spec:** [docs/architecture.md](../../architecture.md), with context in [docs/summary.md](../../summary.md).

## Global Constraints

- Python `>=3.11`. Runtime deps only: `pandas numpy pyarrow lightgbm scikit-learn scipy httpx`. Dev dep: `pytest`. No TA-Lib, Optuna, CatBoost, XGBoost or polars in v1.
- Package `opx` in `src/opx/`; tests in `tests/`; all timestamps tz-aware UTC.
- Payout `r` is the net win fraction, valid range `0 < r <= 1`. Breakeven `p* = 1/(1+r)`.
- Label: entry `open[t+1]`, expiry `close[t+k]`, Call wins iff `close[t+k] - open[t+1] > slip`, Put wins iff `open[t+1] - close[t+k] > slip`. Ties and sub-slip moves lose. `t1 = index[t] + (k+1) minutes`.
- Loss: weighted logloss, weight `r` on `y=1`, `1` on `y=0`. Win probability `p = q/(q + r(1-q))` where `q = sigmoid(raw)`.
- Trade threshold: `p_cal > breakeven(r) + buffer`. Trades are non-overlapping: a trade is kept only if `time + 1 minute >= previous kept t1`.
- Gate (all must pass): one-sided 95% Wilson lower bound of win rate `> breakeven(r)`; DSR `>= 0.95`; PBO `<= 0.2` (N/A for a single config); permutation p `< 0.01`; lower bound still `> breakeven(r - 0.05)`; passes again at double slippage; `|mean(p_cal) - mean(y)| < 0.02` on selected trades; at least 4 quarters with `>= 30` trades each and `>= 75%` of them with EV `> 0`.
- Sizing: fixed 1% of equity. Payout floor 0.80. No Martingale. No OTC data. No broker automation or API.

## Review Focus

Inputs and conditions the spec implies but no feature task exercises; each has a test in the named task.

1. **Weekend or holiday gap inside a label window:** the label is NaN, never computed from a stale price (Task 3).
2. **Flat or zero-range bars (high == low):** features are finite or NaN, never `inf` (Task 4).
3. **Empty or degenerate selection, or a single-class training slice:** no crash; `summarize` reports `n=0` and the gate says no-go (Tasks 9, 10, 11).
4. **Bad input file (duplicate timestamps, missing column):** clear `ValueError` (Task 2).
5. **DST change:** session flags follow local exchange time, so London 08:00 is flagged in both summer and winter (Task 4).

---

### Task 1: Scaffold and payout maths

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/opx/__init__.py`, `src/opx/econ.py`
- Test: `tests/test_econ.py`

**Interfaces:**
- Produces: `breakeven(r: float) -> float`, `ev_per_trade(p: float, r: float) -> float`, `kelly_fraction(p: float, r: float) -> float`; all raise `ValueError` unless `0 < r <= 1`.

- [ ] **Step 1: Initialise the project**

```bash
git init
python -m venv .venv
```
`pyproject.toml`: setuptools `src` layout, `requires-python = ">=3.11"`, the runtime deps from Global Constraints, `[project.optional-dependencies] dev = ["pytest"]`, pytest `pythonpath = ["src"]`, `testpaths = ["tests"]`. `.gitignore`: `.venv/`, `data/`, `__pycache__/`, `*.egg-info/`.
Run: `.venv\Scripts\pip install -e ".[dev]"`. Expected: installs without error.

- [ ] **Step 2: Write the failing test**

```python
import pytest
from opx.econ import breakeven, ev_per_trade, kelly_fraction

def test_breakeven():
    assert breakeven(0.85) == pytest.approx(1 / 1.85)
    assert breakeven(0.80) == pytest.approx(0.5556, abs=1e-4)

def test_ev_per_trade():
    assert ev_per_trade(0.55, 0.85) == pytest.approx(0.0175)
    assert ev_per_trade(0.50, 0.85) == pytest.approx(-0.075)

def test_kelly_fraction():
    assert kelly_fraction(0.55, 0.85) == pytest.approx(0.55 - 0.45 / 0.85)
    assert kelly_fraction(0.50, 0.85) == 0.0

@pytest.mark.parametrize("r", [0, -0.1, 1.5])
def test_invalid_payout(r):
    with pytest.raises(ValueError):
        breakeven(r)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv\Scripts\pytest tests/test_econ.py -v`. Expected: FAIL with `ModuleNotFoundError: opx.econ`.

- [ ] **Step 4: Implement the three functions in `src/opx/econ.py`**

`kelly_fraction` is `max(0, p - (1-p)/r)`. Validation lives in one private `_check(r)` called by all three.

- [ ] **Step 5: Run it to verify it passes**

Run: `.venv\Scripts\pytest tests/test_econ.py -v`. Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: scaffold project and payout maths"
```

---

### Task 2: Bar loader

**Files:**
- Create: `src/opx/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `load_bars(path: str | Path) -> pd.DataFrame` with a sorted, unique, UTC `DatetimeIndex` and columns `open, high, low, close, tick_volume`.
- Input format: CSV or parquet with columns `timestamp, open, high, low, close, volume`. `timestamp` is epoch milliseconds (numeric) or an ISO string. `volume` becomes `tick_volume`.

- [ ] **Step 1: Write the failing tests** (write CSVs to `tmp_path`)

```python
def test_loads_epoch_ms_and_sorts(tmp_path):
    # rows given out of order: ts 120000, 0, 60000
    bars = load_bars(csv)
    assert list(bars.columns) == ["open", "high", "low", "close", "tick_volume"]
    assert bars.index.is_monotonic_increasing and str(bars.index.tz) == "UTC"
    assert bars.index[0] == pd.Timestamp("1970-01-01", tz="UTC")

def test_iso_timestamps(tmp_path):
    assert load_bars(csv).index[0] == pd.Timestamp("2024-01-02 00:00", tz="UTC")

def test_duplicate_timestamps_raise(tmp_path):
    with pytest.raises(ValueError, match="duplicate"): load_bars(csv)

def test_missing_column_raises(tmp_path):   # file has no 'close'
    with pytest.raises(ValueError, match="close"): load_bars(csv)
```

- [ ] **Step 2: Run to verify failure.** Run: `.venv\Scripts\pytest tests/test_data.py -v`. Expected: FAIL, `opx.data` missing.

- [ ] **Step 3: Implement `load_bars`.** Choose the reader by suffix (`.parquet` vs CSV). Numeric `timestamp` uses `pd.to_datetime(unit="ms", utc=True)`, strings use `utc=True`. Sort, then raise `ValueError("duplicate timestamps")` if the index has duplicates. Raise `ValueError` naming any missing required column.

- [ ] **Step 4: Run to verify pass.** Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add validated bar loader"
```

---

### Task 3: Labels

**Files:**
- Create: `src/opx/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: bars DataFrame from Task 2.
- Produces: `make_labels(bars: pd.DataFrame, k: int, slip: float = 0.0) -> pd.DataFrame` with the same index as `bars` and columns `y_call, y_put` (float 0/1/NaN) and `t1` (Timestamp, `index + (k+1) minutes`).

- [ ] **Step 1: Write the failing tests** (build a 10-bar frame by hand with 1-minute spacing)

```python
def test_call_put_and_tie():
    # open[t+1]=1.1000; close[t+1]= 1.1002 (up), 1.0998 (down), 1.1000 (tie)
    lab = make_labels(bars, k=1)
    assert lab.loc[t_up, ["y_call", "y_put"]].tolist() == [1.0, 0.0]
    assert lab.loc[t_down, ["y_call", "y_put"]].tolist() == [0.0, 1.0]
    assert lab.loc[t_tie, ["y_call", "y_put"]].tolist() == [0.0, 0.0]

def test_slip_turns_small_win_into_loss():
    # move of +0.00001 with slip=0.00002
    assert make_labels(bars, k=1, slip=0.00002).loc[t, "y_call"] == 0.0

def test_k5_uses_close_of_bar_t_plus_5():
    lab = make_labels(bars, k=5)
    assert lab["y_call"].iloc[0] == float(bars["close"].iloc[5] > bars["open"].iloc[1])
    assert lab["t1"].iloc[0] == bars.index[0] + pd.Timedelta(minutes=6)

def test_gap_gives_nan():           # drop bar index 4 -> labels whose window spans it are NaN
    lab = make_labels(bars_with_gap, k=3)
    assert lab.loc[window_spanning_gap, ["y_call", "y_put"]].isna().all().all()
    assert lab.loc[window_clear_of_gap, "y_call"].notna().all()

def test_last_rows_are_nan():
    assert make_labels(bars, k=3)["y_call"].iloc[-3:].isna().all()
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, `opx.labels` missing.

- [ ] **Step 3: Implement `make_labels`.** Use positional shifts of `open` by -1 and `close` by -k. A row is valid only if the index at `+1` equals `t + 1 min` and the index at `+k` equals `t + k min`; otherwise set both labels NaN. Compare with strict `>`.

- [ ] **Step 4: Run to verify pass.** Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: settlement-consistent labels with gap guard"
```

---

### Task 4: Features

**Files:**
- Create: `src/opx/features.py`, `tests/conftest.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `build_features(bars: pd.DataFrame) -> pd.DataFrame` (same index; row t uses bars ≤ t).
- Produces (test fixture): `bars_factory(n: int = 5000, phi: float = 0.0, seed: int = 0, start: str = "2024-01-02") -> pd.DataFrame`. Contiguous 1-minute UTC bars. Log returns follow `r_t = phi*r_{t-1} + 1e-4*eps`, `open[t] = close[t-1]`, high/low extend the body by small positive noise, `tick_volume ~ Poisson(100)`.
- Columns of `build_features` (exact names): `body_ratio, upper_wick, lower_wick, clv, ret_1, ret_2, ret_3, ret_5, ret_10, ret_15, rsi_3, rsi_5, rsi_14, bb_pctb, atr_pct, tickvol_z, kaufman_er, tod_sin, tod_cos, dow_sin, dow_cos, sess_london, sess_ny, sess_overlap`.
  - `ret_l` = `ln(close/close.shift(l)) / ((ATR14/close) * sqrt(l))`.
  - ATR14 is the rolling mean of true range (14).
  - RSI uses Wilder smoothing (`ewm(alpha=1/n, adjust=False)`).
  - `bb_pctb` uses a 20-bar mean and ±2σ.
  - `tickvol_z` uses a 60-bar window.
  - `sess_*` use `tz_convert` to `Europe/London` and `America/New_York`, flag local hours 8–16 inclusive of 8, exclusive of 17; `sess_overlap` is both.

- [ ] **Step 1: Write the conftest fixture and failing tests**

```python
def test_columns(bars_factory):
    assert list(build_features(bars_factory(500)).columns) == EXPECTED_COLUMNS

def test_no_lookahead(bars_factory):
    bars = bars_factory(800)
    full = build_features(bars)
    part = build_features(bars.iloc[:500])
    pd.testing.assert_frame_equal(full.iloc[:500], part)

def test_flat_bars_never_inf(bars_factory):
    bars = bars_factory(300)
    bars.iloc[100:130, bars.columns.get_indexer(["open", "high", "low", "close"])] = 1.1
    f = build_features(bars)
    assert not np.isinf(f.to_numpy(dtype=float)).any()

def test_session_flags_follow_local_time_across_dst():
    # 2024-01-15 08:00 UTC (winter) and 2024-07-15 07:00 UTC (summer) are both 08:00 London
    f = build_features(bars_at([winter_ts, summer_ts]))   # frames padded so the rows exist
    assert f.loc[winter_ts, "sess_london"] == 1 and f.loc[summer_ts, "sess_london"] == 1
```
For the DST test build two small contiguous frames (300 bars) that each contain the target timestamp; read the flag at that timestamp.

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, `opx.features` missing.

- [ ] **Step 3: Implement `build_features`.** Replace zero range and zero ATR with NaN before dividing, then `replace([inf, -inf], NaN)` on the result. Session flags depend only on the index.

- [ ] **Step 4: Run to verify pass.** Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: causal feature matrix and synthetic bar fixture"
```

---

### Task 5: Purged walk-forward splits

**Files:**
- Create: `src/opx/cv.py`
- Test: `tests/test_cv.py`

**Interfaces:**
- Produces: `purged_walk_forward(t0: pd.DatetimeIndex, t1: pd.Series, n_splits: int) -> Iterator[tuple[np.ndarray, np.ndarray]]`, yielding positional `(train_idx, test_idx)`. `t1` is positionally aligned with `t0`.

- [ ] **Step 1: Write the failing tests** (`t0` = 1000 minutes, `t1 = t0 + 11 min`)

```python
def test_test_blocks_are_contiguous_and_ordered():
    folds = list(purged_walk_forward(t0, t1, n_splits=4))
    assert len(folds) == 4
    starts = [te[0] for _, te in folds]
    assert starts == sorted(starts) and all(np.all(np.diff(te) == 1) for _, te in folds)

def test_no_training_label_reaches_test_window():
    for tr, te in purged_walk_forward(t0, t1, n_splits=4):
        assert (t1.iloc[tr] < t0[te[0]]).all()
        assert tr.max() < te[0]

def test_purge_drops_exactly_the_overlapping_rows():
    tr, te = next(iter(purged_walk_forward(t0, t1, n_splits=4)))
    assert len(tr) == te[0] - 11     # rows within 11 minutes of the test start are purged
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, `opx.cv` missing.

- [ ] **Step 3: Implement.** Split positions into `n_splits + 1` equal blocks; block 0 is train-only; fold i tests block i+1. Train is every position before the test start whose `t1 < t0[test_start]` (expanding window).

- [ ] **Step 4: Run to verify pass.** Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: purged expanding walk-forward splits"
```

---

### Task 6: EV-weighted objective

**Files:**
- Create: `src/opx/objective.py`
- Test: `tests/test_objective.py`

**Interfaces:**
- Produces: `class EVObjective(r: float)` with `__call__(y_true: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (gradient, Hessian w.r.t. the raw margin; usable as a LightGBM sklearn `objective`) and `to_win_prob(raw: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
def test_hessian_positive_everywhere():
    z = np.linspace(-20, 20, 81)
    for y in (0.0, 1.0):
        _, h = EVObjective(0.85)(np.full_like(z, y), z)
        assert (h > 0).all()

def test_gradient_zero_at_ev_boundary():
    r, p = 0.85, 0.60
    q = r * p / (r * p + 1 - p)
    z = np.array([np.log(q / (1 - q))])
    g1, _ = EVObjective(r)(np.array([1.0]), z)
    g0, _ = EVObjective(r)(np.array([0.0]), z)
    assert p * g1[0] + (1 - p) * g0[0] == pytest.approx(0.0, abs=1e-12)

def test_zero_margin_means_breakeven_probability():
    # at raw=0, q=0.5, so the recovered win prob is the breakeven prob
    assert EVObjective(0.85).to_win_prob(np.array([0.0]))[0] == pytest.approx(1 / 1.85)

def test_to_win_prob_inverts_the_weighting():
    r, p = 0.85, 0.60
    q = r * p / (r * p + 1 - p)
    assert EVObjective(r).to_win_prob(np.array([np.log(q / (1 - q))]))[0] == pytest.approx(p)
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, `opx.objective` missing.

- [ ] **Step 3: Implement.** With `w = r*y + (1-y)` and `q = sigmoid(clip(raw, -30, 30))`: gradient `q*w - r*y`, Hessian `max(q*(1-q)*w, 1e-12)`. `to_win_prob`: `q/(q + r*(1-q))`.

- [ ] **Step 4: Run to verify pass.** Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: EV-weighted logloss objective"
```

---

### Task 7: Model and calibration

**Files:**
- Create: `src/opx/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `EVObjective` (Task 6).
- Produces:
  - `DEFAULT_PARAMS: dict` — `num_leaves=15, max_depth=5, min_child_samples=100, learning_rate=0.03, n_estimators=300, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=42, verbose=-1`.
  - `fit_model(X: np.ndarray, y: np.ndarray, r: float, params: dict | None = None) -> lgb.LGBMClassifier` (params override the defaults).
  - `predict_win_prob(model, X: np.ndarray, r: float) -> np.ndarray` (uncalibrated; uses `model.booster_.predict(X, raw_score=True)` then `EVObjective.to_win_prob`).
  - `fit_calibrator(p: np.ndarray, y: np.ndarray) -> LogisticRegression` (Platt: logistic regression on `logit(clip(p, 1e-6, 1-1e-6))`).
  - `calibrate(cal, p: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests** (synthetic: `X ~ N(0,1)` with 5 columns, `y = Bernoulli(sigmoid(1.2*X[:,0]))`, 20k train rows, 10k test rows)

```python
def test_learns_planted_signal():
    m = fit_model(Xtr, ytr, r=0.85, params={"n_estimators": 100})
    p = predict_win_prob(m, Xte, r=0.85)
    assert roc_auc_score(yte, p) > 0.65

def test_calibrated_mean_matches_base_rate():
    p_cal = calibrate(fit_calibrator(p_val, y_val), p_test)
    assert abs(p_cal.mean() - yte.mean()) < 0.03

def test_noise_gives_no_confident_predictions():
    # y independent of X
    m = fit_model(Xtr, ynoise, r=0.85, params={"n_estimators": 100})
    p = predict_win_prob(m, Xte, r=0.85)
    assert abs(p.mean() - ynoise.mean()) < 0.05
```
`p_val` comes from a held-out slice of the training data, as in the walk-forward.

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, `opx.model` missing.

- [ ] **Step 3: Implement the four functions.** `LGBMClassifier(objective=EVObjective(r), **{**DEFAULT_PARAMS, **(params or {})})`. A callable objective makes `predict_proba` return raw scores, which is why prediction goes through `booster_`.

- [ ] **Step 4: Run to verify pass.** Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: LightGBM EV model with Platt calibration"
```

---

### Task 8: Statistics

**Files:**
- Create: `src/opx/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Produces:
  - `wilson_lcb(wins: int, n: int, conf: float = 0.95) -> float` (one-sided lower bound; `nan` when `n == 0`).
  - `deflated_sharpe(returns: np.ndarray, n_trials: int) -> float` (probability in [0, 1]).
  - `permutation_pvalue(y: np.ndarray, take: np.ndarray, r: float, n_perm: int = 2000, seed: int = 0) -> float`.
  - `pbo(perf: np.ndarray, n_blocks: int = 16) -> float` (`perf` is shape `(T, n_configs)` of per-period returns).

- [ ] **Step 1: Write the failing tests**

```python
def test_wilson_lcb():
    assert wilson_lcb(60, 100) == pytest.approx(0.5178, abs=1e-3)   # z = 1.645
    assert np.isnan(wilson_lcb(0, 0))
    assert wilson_lcb(600, 1000) > wilson_lcb(60, 100)

def test_dsr_rewards_edge_and_punishes_many_trials():
    rng = np.random.default_rng(0)
    edge = rng.normal(0.15, 1.0, 3000)
    noise = rng.normal(0.0, 1.0, 3000)
    assert deflated_sharpe(edge, n_trials=1) > 0.99
    assert deflated_sharpe(noise, n_trials=100) < 0.95
    assert deflated_sharpe(edge, n_trials=1) > deflated_sharpe(edge, n_trials=500)

def test_permutation_pvalue():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000).astype(float)
    take_good = (y == 1) | (rng.random(4000) < 0.3)   # selection correlated with outcome
    take_none = rng.random(4000) < 0.3
    assert permutation_pvalue(y, take_good, 0.85) < 0.01
    assert permutation_pvalue(y, take_none, 0.85) > 0.05

def test_pbo():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, (800, 10))
    assert 0.3 < pbo(noise) < 0.7
    dominated = noise.copy(); dominated[:, 3] += 0.5
    assert pbo(dominated) < 0.1
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, `opx.stats` missing.

- [ ] **Step 3: Implement.** Wilson: `z = norm.ppf(conf)`; `(ph + z²/2n − z·sqrt(ph(1−ph)/n + z²/4n²)) / (1 + z²/n)`.

DSR algorithm (Bailey and López de Prado), with `T = len(returns)`, `SR = mean/std`, `γ3`/`γ4` the sample skew and non-excess kurtosis, `γe = 0.5772`:
`SR0 = sqrt(1/(T-1)) * ((1-γe)*Φ⁻¹(1-1/N) + γe*Φ⁻¹(1-1/(N*e)))`, `SR0 = 0` when `N == 1`;
`DSR = Φ((SR - SR0)*sqrt(T-1) / sqrt(1 - γ3*SR + (γ4-1)/4 * SR²))`.

Permutation: observed EV is the mean of `where(y[take]==1, r, -1)`; shuffle all of `y` `n_perm` times, recompute on the same `take`; `p = (1 + count(perm >= obs)) / (1 + n_perm)`; `nan` if `take` is empty.

PBO (CSCV): split rows into `n_blocks`; for every combination of `n_blocks/2` blocks as in-sample, pick the config with the best in-sample mean, take its out-of-sample rank among all configs, `w = rank/(n_configs+1)`, `λ = ln(w/(1-w))`; `PBO = mean(λ <= 0)`.

- [ ] **Step 4: Run to verify pass.** Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Wilson bound, DSR, permutation test and PBO"
```

---

### Task 9: Trade selection and summary

**Files:**
- Create: `src/opx/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `breakeven`, `ev_per_trade` (Task 1), `wilson_lcb` (Task 8).
- Produces:
  - `select_trades(oos: pd.DataFrame, r: float, buffer: float) -> pd.DataFrame`. `oos` has columns `time, side, p_cal, y, t1`. Keeps rows with `p_cal > breakeven(r) + buffer`, sorted by `time`, dropping any trade whose `time + 1 minute` is earlier than the previous kept trade's `t1`.
  - `summarize(trades: pd.DataFrame, r: float) -> dict` with keys `n, win_rate, ev, lcb, breakeven` (`win_rate, ev, lcb` are `nan` when `n == 0`).

- [ ] **Step 1: Write the failing tests**

```python
def test_threshold_filters():
    t = select_trades(oos_three_rows, r=0.85, buffer=0.0)    # p_cal = 0.53, 0.55, 0.70
    assert t["p_cal"].tolist() == [0.55, 0.70]

def test_overlapping_trades_dropped():
    # times 00:00 and 00:03, t1 = time + 6 min (k=5): the second overlaps and is dropped
    assert len(select_trades(two_overlapping, r=0.85, buffer=0.0)) == 1
    # times 00:00 and 00:05: entry 00:06 >= t1 00:06, kept
    assert len(select_trades(two_adjacent, r=0.85, buffer=0.0)) == 2

def test_summarize():
    s = summarize(trades_6_wins_of_10, r=0.85)
    assert s["n"] == 10 and s["win_rate"] == 0.6
    assert s["ev"] == pytest.approx(0.6 * 0.85 - 0.4)
    assert s["breakeven"] == pytest.approx(1 / 1.85)

def test_empty_selection_is_not_an_error():
    s = summarize(select_trades(oos_three_rows, r=0.85, buffer=0.9), r=0.85)
    assert s["n"] == 0 and np.isnan(s["lcb"])
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, `opx.evaluate` missing.

- [ ] **Step 3: Implement both functions.** Use a simple loop over the sorted rows to enforce non-overlap.

- [ ] **Step 4: Run to verify pass.** Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: trade selection and summary"
```

---

### Task 10: Walk-forward runner

**Files:**
- Create: `src/opx/walkforward.py`
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `make_labels`, `build_features`, `purged_walk_forward`, `fit_model`, `predict_win_prob`, `fit_calibrator`, `calibrate`.
- Produces: `run_walk_forward(bars: pd.DataFrame, k: int, r: float, slip: float = 0.0, n_splits: int = 6, calib_frac: float = 0.3, params: dict | None = None) -> pd.DataFrame` returning the oos frame with columns `time, side, p_cal, y, t1`. `side` is `"call"` or `"put"`.
- Per fold and per side: fit the model on the first `1 - calib_frac` of the training rows; fit the calibrator on the remaining rows after dropping those with `t1 >= ` the first calibration row's time; predict on the test block. Skip a side for a fold when the model's training labels have a single class. Rows with any NaN feature or label are dropped before splitting.

- [ ] **Step 1: Write the failing tests** (`params={"n_estimators": 80}` for speed)

```python
def test_planted_mean_reversion_is_found(bars_factory):
    oos = run_walk_forward(bars_factory(30000, phi=-0.5), k=1, r=0.85, params=FAST)
    s = summarize(select_trades(oos, 0.85, buffer=0.03), 0.85)
    assert s["n"] > 300 and s["win_rate"] > 0.60

def test_pure_noise_has_no_edge(bars_factory):
    oos = run_walk_forward(bars_factory(30000, phi=0.0), k=1, r=0.85, params=FAST)
    s = summarize(select_trades(oos, 0.85, buffer=0.03), 0.85)
    assert s["n"] == 0 or s["lcb"] < s["breakeven"]

def test_oos_rows_are_strictly_after_first_train_block(bars_factory):
    bars = bars_factory(6000, phi=-0.5)
    oos = run_walk_forward(bars, k=1, r=0.85, params=FAST, n_splits=4)
    assert oos["time"].min() >= bars.index[len(bars) // 5]
    assert set(oos["side"]) == {"call", "put"}

def test_flat_prices_return_empty_frame(bars_factory):
    flat = bars_factory(3000); flat[["open", "high", "low", "close"]] = 1.1
    oos = run_walk_forward(flat, k=1, r=0.85, params=FAST)
    assert oos.empty and list(oos.columns) == ["time", "side", "p_cal", "y", "t1"]
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, `opx.walkforward` missing.

- [ ] **Step 3: Implement `run_walk_forward`** as specified above.

- [ ] **Step 4: Run to verify pass.** Run: `.venv\Scripts\pytest tests/test_walkforward.py -v`. Expected: PASS (4 tests). The planted-signal test is the pipeline's integrity check: if it fails, fix the pipeline before touching real data.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: purged walk-forward runner"
```

---

### Task 11: Go/no-go gate and CLI

**Files:**
- Create: `src/opx/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `select_trades`, `summarize`, `wilson_lcb`, `deflated_sharpe`, `permutation_pvalue`, `pbo`, `breakeven`, `run_walk_forward`, `load_bars`.
- Produces:
  - `@dataclass(frozen=True) GateResult` with `checks: dict[str, bool]`, `details: dict[str, float]`, `go: bool` (`go` is `all(checks.values())`).
  - `go_nogo(oos: pd.DataFrame, r: float, buffers: Sequence[float], n_other_trials: int = 1, oos_stress: pd.DataFrame | None = None, min_quarters: int = 4) -> GateResult`.
  - `main(argv: list[str] | None = None) -> int` for `python -m opx.gate --bars PATH --k INT --r FLOAT --buffers FLOAT [FLOAT ...] [--slip FLOAT] [--n-other-trials INT]`. Exit code 0 on GO, 1 on NO-GO.
- Check keys: `lcb_vs_breakeven, dsr, pbo, permutation, payout_stress, lag_stress, calibration, quarters`.
- Rules:
  - The primary buffer is the one in `buffers` with the highest total PnL.
  - `n_trials = len(buffers) * n_other_trials`.
  - DSR runs on per-trade PnL (`r` or `-1`) of the primary selection.
  - PBO uses a daily-PnL matrix with one column per buffer, and is N/A (check true, detail `nan`) when `len(buffers) == 1`.
  - `lag_stress` is false when `oos_stress` is `None`; otherwise the same buffer on `oos_stress` must pass the lower-bound check.
  - Payout stress, calibration and quarters follow Global Constraints.
- CLI behaviour: runs `run_walk_forward` at `slip` and at `2*slip`, calls `go_nogo`, prints one `PASS`/`FAIL` line per check with its detail value, then prints `MANUAL: lawful in your jurisdiction; forward demo test of 500-1000 non-overlapping trades; settlement-feed basis measured`.

- [ ] **Step 1: Write the failing tests** (a module-scoped fixture computes the planted and noise oos frames once, using `FAST` params)

```python
def test_planted_signal_passes_all_checks(planted_oos):
    g = go_nogo(planted_oos, 0.85, buffers=[0.03], oos_stress=planted_oos, min_quarters=1)
    assert g.go, g.checks

def test_noise_fails(noise_oos):
    g = go_nogo(noise_oos, 0.85, buffers=[0.03], oos_stress=noise_oos, min_quarters=1)
    assert not g.go and not g.checks["lcb_vs_breakeven"]

def test_default_min_quarters_blocks_short_history(planted_oos):
    g = go_nogo(planted_oos, 0.85, buffers=[0.03], oos_stress=planted_oos)
    assert not g.checks["quarters"] and not g.go

def test_missing_stress_run_fails_lag_check(planted_oos):
    assert not go_nogo(planted_oos, 0.85, buffers=[0.03], min_quarters=1).checks["lag_stress"]

def test_empty_oos_is_no_go_not_a_crash():
    empty = pd.DataFrame(columns=["time", "side", "p_cal", "y", "t1"])
    assert not go_nogo(empty, 0.85, buffers=[0.03]).go

def test_cli_exit_codes(tmp_path, bars_factory):
    path = tmp_path / "noise.parquet"   # bars written in the loader's input format
    assert main(["--bars", str(path), "--k", "1", "--r", "0.85", "--buffers", "0.03"]) == 1
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, `opx.gate` missing.

- [ ] **Step 3: Implement `go_nogo` and `main`** per the rules above. The CLI writes nothing to disk.

- [ ] **Step 4: Run to verify pass.** Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: go/no-go gate and CLI report"
```

---

### Task 12: Alert-only decision and Telegram sender

**Files:**
- Create: `src/opx/live.py`
- Test: `tests/test_live.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Decision` with `side: str`, `p_win: float`, `stake: float`, `k: int`.
  - `decide(p_call: float, p_put: float, r: float, buffer: float, equity: float, k: int, min_payout: float = 0.80, stake_frac: float = 0.01) -> Decision | None`. Returns `None` when `r < min_payout` or the better side's probability is `<= breakeven(r) + buffer`. Otherwise returns the better side with `stake = round(equity * stake_frac, 2)`.
  - `send_telegram(token: str, chat_id: str, text: str, client: httpx.Client | None = None) -> None`. POSTs JSON `{"chat_id", "text"}` to `https://api.telegram.org/bot{token}/sendMessage` and raises on a non-2xx status.

- [ ] **Step 1: Write the failing tests**

```python
def test_picks_better_side_and_fixed_stake():
    d = decide(0.58, 0.51, r=0.85, buffer=0.02, equity=1000.0, k=5)
    assert d == Decision("call", 0.58, 10.0, 5)

def test_below_threshold_returns_none():
    assert decide(0.55, 0.50, r=0.85, buffer=0.02, equity=1000.0, k=5) is None   # 0.55 < 0.5405 + 0.02

def test_payout_floor_vetoes():
    assert decide(0.70, 0.40, r=0.75, buffer=0.0, equity=1000.0, k=5) is None

def test_send_telegram_posts_expected_request():
    seen = []
    transport = httpx.MockTransport(lambda req: seen.append(req) or httpx.Response(200, json={"ok": True}))
    send_telegram("TOKEN", "42", "hi", client=httpx.Client(transport=transport))
    assert seen[0].url.path == "/botTOKEN/sendMessage"
    assert json.loads(seen[0].content) == {"chat_id": "42", "text": "hi"}

def test_send_telegram_raises_on_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(401))
    with pytest.raises(httpx.HTTPStatusError):
        send_telegram("T", "1", "x", client=httpx.Client(transport=transport))
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, `opx.live` missing.

- [ ] **Step 3: Implement `decide`, `Decision` and `send_telegram`.** Without a client, create and close one inside the function.

- [ ] **Step 4: Run to verify pass.** Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: alert-only decision rule and Telegram sender"
```

---

### Task 13: Run on real data and record the verdict

**Files:**
- Create: `docs/results.md`

**Interfaces:**
- Consumes: the CLI from Task 11. Needs EURUSD, GBPUSD and USDJPY M1 history, at least one year each, saved in `data/` as CSV or parquet in the Task 2 format. The user supplies these (for example from Dukascopy). They are never committed.

- [ ] **Step 1: Run the full test suite.**

Run: `.venv\Scripts\pytest -q`. Expected: all tests pass.

- [ ] **Step 2: Run the gate for each pair and horizon.**

```bash
.venv\Scripts\python -m opx.gate --bars data/eurusd_m1.parquet --k 5 --r 0.85 --buffers 0.00 0.02 0.04 0.06 --slip 0.00002 --n-other-trials 9
```
Repeat for `k` in 1, 5, 15 and for each pair, so `--n-other-trials` is the honest count of pair-and-horizon combinations tried (9 here). Expected: a PASS/FAIL line per check, exit code 0 or 1.

- [ ] **Step 3: Record the results in `docs/results.md`**: one table row per pair and horizon with every check's outcome, then the cross-pair rule (edge in at least 2 of 3 pairs). State plainly whether any configuration reached GO. A NO-GO is a valid and expected outcome.

- [ ] **Step 4: Commit**

```bash
git add docs/results.md && git commit -m "docs: record go/no-go results on real data"
```

---

## Deferred (do not start unless Task 13 yields a GO)

- Meta-label model (second GBDT on regime features plus the primary probability, trained on a 70/30 chronological split with purge).
- Extra features: Hawkes order-flow imbalance, fractional differentiation, squeeze state, fractal support/resistance distance.
- A settlement-feed quote recorder and a demo forward-test log. This needs a reverse-engineered broker API, so decide on the terms-of-service and legal risk first.
- Optuna tuning (every trial must be counted in `n_other_trials`), CatBoost and XGBoost.
