# Phase 3: Reinforcement Learning Meta-Agent (Multi-Pair Portfolio Allocator) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train a Reinforcement Learning Meta-Agent (using Gymnasium and Stable-Baselines3 PPO) that ingests calibrated ML signals across all 5 FX pairs along with non-raw-price features (excluding raw OHLC) to learn optimal trade timing, pair selection, and trade omission under the 85% payout structure.

**Architecture:** A multi-pair signal dataset builder (`src/opx/rl/dataset.py`) synchronizes out-of-sample calibrated ML probabilities ($p_{call}, p_{put}$) for EURUSD, GBPUSD, USDJPY, AUDUSD, and USDCAD alongside non-price state features. A custom Gymnasium environment (`src/opx/rl/env.py`) simulates binary options execution with non-overlapping position locks and EV-based rewards. A PPO policy is trained and evaluated out-of-sample against the baseline fixed-threshold heuristics.

**Tech Stack:** Python 3.11, Gymnasium, Stable-Baselines3, PyTorch (CUDA), pandas, numpy, pytest.

**Spec:** User requirement #3 ("นำ RL มาฝึก โดยให้ signal จาก ML ของทุกคู่เงินมาฝึก และ feature ยกเว้น raw price").

## Global Constraints

- **Strict Exclusion of Raw Price:** Observation state vectors must NOT contain raw prices (`open`, `high`, `low`, `close`). Only normalized statistics, ratios, indicators, and ML probability signals are permitted.
- **Settlement Invariant:** Expiry remains fixed at $k$ minutes, entry at `open[t+1]` with slippage, tie is loss (Win: $+r$, Loss: $-1.0$).
- **Zero Future Leakage:** State at step $t$ must only expose data $\le t$. When an action is taken at $t$, its outcome is determined strictly at $t+k$.
- **Illiquidity Mask:** NY Rollover hours (16:00–18:59 NY) and Sundays remain non-tradable.

## Review Focus

1. **Timestamp Synchronization across all 5 pairs:** Inner join or forward-fill across all 5 pairs must be causal and aligned to the common trading minutes.
2. **Action Space & Active Position Lock:** The environment must prevent taking overlapping trades on the same pair while a previous trade is active.
3. **Reward Scaling & Variance Penalty:** Pure binary rewards (+0.85 / -1.0) with high frequency of 0 can have sparse gradients; reward shaping should support risk-adjusted return (e.g. differential Sharpe or bounded EV).
4. **Out-of-Sample Train/Test Split:** RL agent must train on historical walk-forward train folds and be evaluated strictly on unseen out-of-sample test blocks.
5. **No Lookahead in State Vector:** Ensure ML probability signals in state $t$ were predicted using models trained on data strictly prior to $t$.

---

### Task 1: Multi-Pair Signal & Non-Price State Matrix Builder

**Files:**
- Create: `src/opx/rl/dataset.py`
- Test: `tests/test_rl_dataset.py`

**Interfaces:**
- Produces:
  - `build_rl_state_matrix(pairs_bars, pairs_oos_preds, dxy_bars) -> pd.DataFrame`
  - Columns:
    - 10 ML signals: `{pair}_p_call`, `{pair}_p_put` for EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD
    - Non-price features: Squeeze, Vol ratios, Normalized returns, RSI, DXY momentum, Cyclical time, Sessions
    - Strictly excludes: `open`, `high`, `low`, `close`

- [ ] **Step 1: Write failing test verifying no raw price in state columns and correct multi-pair alignment**

```python
# tests/test_rl_dataset.py
import pandas as pd
from opx.rl.dataset import build_rl_state_matrix, NON_PRICE_COLUMNS

def test_no_raw_price_in_rl_state():
    # verify that none of 'open', 'high', 'low', 'close' appear in the generated state matrix
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_rl_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/opx/rl/dataset.py`**

Assemble and align multi-pair ML probability predictions and causal normalized feature matrices.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_rl_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/opx/rl/dataset.py tests/test_rl_dataset.py
git commit -m "feat: implement multi-pair signal and non-price state matrix builder for RL"
```

---

### Task 2: Gymnasium Binary Options Multi-Pair Environment

**Files:**
- Create: `src/opx/rl/env.py`
- Test: `tests/test_rl_env.py`

**Interfaces:**
- Produces:
  - `MultiPairBinaryOptionsEnv(gymnasium.Env)`
  - Action space: Discrete(11) -> 0: Hold, 1-2: EURUSD (Call/Put), 3-4: GBPUSD, 5-6: USDJPY, 7-8: AUDUSD, 9-10: USDCAD
  - Reward: Win $+r$, Loss $-1.0$, Hold $0.0$

- [ ] **Step 1: Write failing test verifying Gym environment step, reset, observation space, and reward calculation**

```python
# tests/test_rl_env.py
from opx.rl.env import MultiPairBinaryOptionsEnv

def test_env_step_and_reward():
    # test step with action 0 (hold) and action 1 (trade)
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_rl_env.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/opx/rl/env.py`**

Implement `MultiPairBinaryOptionsEnv` complying with Gymnasium standard API.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_rl_env.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/opx/rl/env.py tests/test_rl_env.py
git commit -m "feat: implement Gymnasium multi-pair binary options portfolio environment"
```

---

### Task 3: PPO Agent Training & Evaluation Pipeline

**Files:**
- Create: `src/opx/rl/train.py`
- Test: `tests/test_rl_train.py`

**Interfaces:**
- Produces:
  - `train_ppo_agent(train_env, total_timesteps=100000, lr=3e-4) -> PPO`
  - `evaluate_agent_oos(agent, test_env) -> dict[str, Any]`

- [ ] **Step 1: Write failing test verifying PPO training loop and policy inference**

```python
# tests/test_rl_train.py
from opx.rl.train import train_ppo_agent, evaluate_agent_oos

def test_ppo_training_and_evaluation():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest tests/test_rl_train.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/opx/rl/train.py`**

Train PPO policy network with Stable-Baselines3, support GPU acceleration, and evaluate out-of-sample metrics.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest tests/test_rl_train.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/opx/rl/train.py tests/test_rl_train.py
git commit -m "feat: implement PPO training and out-of-sample evaluation pipeline"
```

---

### Task 4: Execute RL Experiment on Historical Multi-Pair Data and Report

**Files:**
- Create: `scripts/run_rl_experiment.py`
- Output: `results/rl_meta_agent_results.json`
- Output: `docs/rl-meta-agent-report.md`

- [ ] **Step 1: Implement `scripts/run_rl_experiment.py`**

End-to-end experiment script that generates state matrices across all 5 pairs, trains PPO on train window, and evaluates on out-of-sample test window.

- [ ] **Step 2: Run the experiment script**

Run: `.venv\Scripts\python scripts/run_rl_experiment.py --timesteps 200000`

- [ ] **Step 3: Compile comparison metrics and commit report**

Compare RL Meta-Agent Win Rate, Trade Frequency, EV, and Sharpe vs. Individual Fixed-Threshold Heuristics.

```bash
git add scripts/run_rl_experiment.py results/rl_meta_agent_results.json docs/rl-meta-agent-report.md
git commit -m "docs: record Phase 3 RL meta-agent results and comparative analysis"
```
