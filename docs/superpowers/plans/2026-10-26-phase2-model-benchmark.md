# Phase 2: Alternative ML Models Benchmark (CatBoost, XGBoost, CNN+LSTM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 3 alternative machine learning architectures (CatBoost, XGBoost, and 1D-CNN+BiLSTM) with EV-weighted loss objectives, integrate them into the purged walk-forward cross-validation pipeline, and benchmark their performance against LightGBM on GBPUSD k=5.

**Architecture:** A unified model registry (`src/opx/model_registry.py`) provides standardized `fit_model` and `predict_win_prob` adapters across LightGBM, CatBoost, XGBoost, and PyTorch (1D-CNN + BiLSTM). An automated benchmark harness (`scripts/benchmark_models.py`) evaluates all 4 architectures under identical walk-forward splits, illiquidity filtering, and Platt calibration.

**Tech Stack:** Python 3.11, LightGBM, XGBoost, CatBoost, PyTorch, scikit-learn, scipy, pandas, numpy, pytest.

**Spec:** User requirement #2 ("หา ML model อื่นมาใช้ด้วย เช่น CNN+LSTM, TFT, Transformers, Chronos, XGBOOST,CATBOOST ETC ลองเอามา3ตัว แล้ว compare กัน") and Phase 1 winning feature set.

## Global Constraints

- **Consistent Objective:** All models must optimize the EV-weighted loss ($w = r \cdot y + (1 - y)$) targeting the theoretical breakeven $p^* = 1 / (1 + r)$ at payout $r = 0.85$.
- **Strict Out-of-Sample Calibration:** All models must calibrate raw predictions via Platt scaling using only the designated 30% calibration subset of the training fold.
- **Zero Lookahead:** PyTorch sequence tensors must use causal sliding windows strictly $\le t$.
- **Zero Rollover Contamination:** `skip_illiquid=True` must remain enforced across all models.

## Review Focus

1. **CatBoost Objective Sign Inversion:** CatBoost custom objectives minimize loss via `der1 = -(gradient)` and `der2 = hessian`; getting the sign wrong would invert the optimization direction.
2. **XGBoost Objective Interface:** XGBoost custom objectives require signature `custom_obj(preds, dmatrix)` returning `grad, hess`.
3. **PyTorch Batch / Sequence Alignment:** When slicing 3D tensors for CNN-LSTM `(batch, seq_len, features)`, sliding window indices must align with exact timestamp $t$ without temporal leakage.
4. **GPU / CPU fallback:** PyTorch model must automatically utilize CUDA if available (RTX 5070 detected) and fall back gracefully to CPU.
5. **Memory Management across Folds:** Deep learning models must clear GPU memory/tensors between walk-forward folds to avoid out-of-memory errors.

---

### Task 1: CatBoost EV-Weighted Objective & Adapter

**Files:**
- Create: `src/opx/models/catboost_model.py`
- Test: `tests/test_catboost_model.py`

**Interfaces:**
- Produces:
  - `fit_catboost(X, y, r, params) -> CatBoostClassifier`
  - `predict_catboost_prob(model, X, r) -> np.ndarray`

- [ ] **Step 1: Write failing test verifying CatBoost EV training and probability recovery**

```python
# tests/test_catboost_model.py
import numpy as np
from opx.models.catboost_model import fit_catboost, predict_catboost_prob

def test_catboost_ev_objective():
    X = np.random.randn(200, 10)
    y = np.random.binomial(1, 0.5, 200).astype(float)
    model = fit_catboost(X, y, r=0.85, params={"iterations": 50, "verbose": 0})
    probs = predict_catboost_prob(model, X, r=0.85)
    assert len(probs) == 200
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_catboost_model.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/opx/models/catboost_model.py`**

Implement `CatBoostEVObjective` with correct `calc_ders_range` derivatives and `fit_catboost` / `predict_catboost_prob`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_catboost_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/opx/models/catboost_model.py tests/test_catboost_model.py
git commit -m "feat: implement CatBoost with EV-weighted objective and probability converter"
```

---

### Task 2: XGBoost EV-Weighted Objective & Adapter

**Files:**
- Create: `src/opx/models/xgboost_model.py`
- Test: `tests/test_xgboost_model.py`

**Interfaces:**
- Produces:
  - `fit_xgboost(X, y, r, params) -> xgb.Booster`
  - `predict_xgboost_prob(model, X, r) -> np.ndarray`

- [ ] **Step 1: Write failing test verifying XGBoost EV training and probability recovery**

```python
# tests/test_xgboost_model.py
import numpy as np
from opx.models.xgboost_model import fit_xgboost, predict_xgboost_prob

def test_xgboost_ev_objective():
    X = np.random.randn(200, 10)
    y = np.random.binomial(1, 0.5, 200).astype(float)
    model = fit_xgboost(X, y, r=0.85, params={"n_estimators": 50, "verbosity": 0})
    probs = predict_xgboost_prob(model, X, r=0.85)
    assert len(probs) == 200
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_xgboost_model.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/opx/models/xgboost_model.py`**

Implement XGBoost custom objective and training wrapper.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_xgboost_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/opx/models/xgboost_model.py tests/test_xgboost_model.py
git commit -m "feat: implement XGBoost with EV-weighted objective and probability converter"
```

---

### Task 3: 1D-CNN + BiLSTM Deep Learning Model (PyTorch)

**Files:**
- Create: `src/opx/models/nn_model.py`
- Test: `tests/test_nn_model.py`

**Interfaces:**
- Produces:
  - `CNNLSTMNet(nn.Module)`: 1D Conv layer (kernel 3) + BatchNorm + BiLSTM layer + Linear head.
  - `fit_nn(X, y, r, params) -> CNNLSTMNet`
  - `predict_nn_prob(model, X, r) -> np.ndarray`

- [ ] **Step 1: Write failing test verifying CNN+BiLSTM forward pass, training loop, and GPU/CPU execution**

```python
# tests/test_nn_model.py
import numpy as np
import torch
from opx.models.nn_model import fit_nn, predict_nn_prob, CNNLSTMNet

def test_nn_forward_and_fit():
    X = np.random.randn(150, 20)
    y = np.random.binomial(1, 0.5, 150).astype(float)
    model = fit_nn(X, y, r=0.85, epochs=3, batch_size=32)
    probs = predict_nn_prob(model, X, r=0.85)
    assert len(probs) == 150
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_nn_model.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/opx/models/nn_model.py`**

Implement `CNNLSTMNet`, weighted EV loss function, PyTorch Dataset/DataLoader with automatic CUDA device placement, and evaluation functions.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_nn_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/opx/models/nn_model.py tests/test_nn_model.py
git commit -m "feat: implement 1D-CNN + BiLSTM architecture with EV-weighted loss in PyTorch"
```

---

### Task 4: Unified Model Registry & Extended Walk-Forward Integration

**Files:**
- Create: `src/opx/model_registry.py`
- Modify: `src/opx/walkforward.py`
- Test: `tests/test_model_registry.py`

**Interfaces:**
- Produces:
  - `run_walk_forward(..., model_type='lightgbm'|'catboost'|'xgboost'|'cnn_lstm')`

- [ ] **Step 1: Write failing test verifying walk-forward executes with each model type**

```python
# tests/test_model_registry.py
import pytest
from opx.walkforward import run_walk_forward

@pytest.mark.parametrize("m_type", ["lightgbm", "xgboost", "catboost"])
def test_walk_forward_multi_model(bars_factory, m_type):
    bars = bars_factory(3000, phi=-0.5)
    oos = run_walk_forward(bars, k=1, r=0.85, model_type=m_type, n_splits=2)
    assert not oos.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_model_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Implement registry and integrate with `walkforward.py`**

Refactor `walkforward.py` to dispatch to the selected `model_type`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_model_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/opx/model_registry.py src/opx/walkforward.py tests/test_model_registry.py
git commit -m "feat: integrate unified multi-model dispatch into walk-forward harness"
```

---

### Task 5: Execute Model Benchmark on GBPUSD k=5 and Report

**Files:**
- Create: `scripts/benchmark_models.py`
- Output: `results/benchmark_models_gbpusd_k5.json`
- Output: `docs/model-benchmark-report-gbpusd-k5.md`

- [ ] **Step 1: Implement `scripts/benchmark_models.py`**

Runner that evaluates LightGBM, CatBoost, XGBoost, and CNN+LSTM sequentially using the best feature set (`+squeeze` and `+dxy+cross`), saving full comparison metrics.

- [ ] **Step 2: Run benchmark script**

Run: `.venv\Scripts\python scripts/benchmark_models.py --pair gbpusd --k 5`

- [ ] **Step 3: Compile comparative analysis and commit report**

Tabulate Win rate, Wilson LCB, DSR, PBO, and training time across the 4 architectures.

```bash
git add scripts/benchmark_models.py results/benchmark_models_gbpusd_k5.json docs/model-benchmark-report-gbpusd-k5.md
git commit -m "docs: record Phase 2 model benchmark comparison results on GBPUSD k=5"
```
