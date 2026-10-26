# Phase 3: Reinforcement Learning Meta-Agent Report

**Date:** 2026-10-26  
**Environment:** Multi-Pair Binary Options (`MultiPairBinaryOptionsEnv`)  
**Assets Covered:** EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD  
**Horizon:** $k = 5$ minutes  
**Payout Rate:** $r = 0.85$ (Breakeven Hurdle: $p^* = \frac{1}{1 + 0.85} \approx 54.05\%$)  
**Agent Architecture:** Proximal Policy Optimization (PPO, Stable-Baselines3)  
**Hardware:** NVIDIA GeForce RTX 5070 / Multi-core CPU  

---

## 1. Executive Summary

| Metric | Out-of-Sample Result | Benchmark / Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Out-of-Sample Bars** | 47,802 bars | 30% held-out unseen market data | Validated |
| **Trades Taken** | 8,012 trades | Sample size $> 250$ | Validated |
| **Win Rate** | **45.16%** | **> 54.05%** (Breakeven Hurdle) | **FAILED (-8.89% gap)** |
| **Wilson 95% LCB** | **44.24%** | **> 54.05%** | **FAILED** |
| **Cumulative PnL** | **-1,493.50 R** | $> 0$ | **FAILED** |
| **Sharpe Ratio** | **-0.179** | $> 1.0$ | **FAILED** |
| **Action Collapse** | USDCAD dominated (52.0%) | Balanced across 5 pairs | **Degenerate Policy** |
| **Production Recommendation** | **STRICT NO-GO** | — | **FAILED** |

The Reinforcement Learning Meta-Agent trained on 150 engineered non-price features and multi-pair ML probability signals **failed to achieve profitability**, posting a **45.16% win rate** and losing **-1,493.5 R** over 8,012 trades. 

This confirms that chaining an RL meta-agent on top of weakly predictive or negative-EV base signals on M1 binary options does not extract alpha; instead, it amplifies noise and suffers policy collapse.

---

## 2. Experimental Setup & Protocol

### 2.1 State Space (150 Dimensions, Strictly Zero Raw Price)
To satisfy the strict constraint of **zero raw OHLC prices**, each bar across all 5 pairs was encoded using stationary and normalized features:
- **ML Probability Signals:** Calibrated probability estimates $P(\text{Up}_{t+5} \mid X_t)$ from base gradient boosted models trained on each respective currency pair.
- **Normalized Returns:** Log returns at horizons $\Delta t \in \{1, 2, 3, 5, 10, 15, 30\}$ minutes.
- **Oscillators & Momentum:** RSI(7), RSI(14), MACD histogram, Stochastic %K/%D.
- **Volatility & Bands:** Realized volatility (15m, 60m), Bollinger Band Squeeze ratio, Normalized ATR.
- **Candle Geometry:** Relative body size, upper wick ratio, lower wick ratio (normalized by candle range, no absolute price levels).
- **Cross-Pair & Macro Context:** DXY momentum (1m, 5m, 15m returns, RSI), cross-asset normalized returns relative to USD.

Total state vector size: $5 \text{ pairs} \times 30 \text{ features} = 150 \text{ continuous features}$.

### 2.2 Action Space: Discrete(11)
- `Action 0`: **HOLD** (Do not trade any pair on this bar)
- `Action 1`: EURUSD CALL | `Action 2`: EURUSD PUT
- `Action 3`: GBPUSD CALL | `Action 4`: GBPUSD PUT
- `Action 5`: USDJPY CALL | `Action 6`: USDJPY PUT
- `Action 7`: AUDUSD CALL | `Action 8`: AUDUSD PUT
- `Action 9`: USDCAD CALL | `Action 10`: USDCAD PUT

### 2.3 Reward & Cooldown Function
$$\text{Reward}(a_t) = \begin{cases} 0.0 & \text{if } a_t = 0 \text{ (HOLD)} \\ +0.85 & \text{if Trade Wins } (S_{t+k} > S_t \text{ for CALL or } S_{t+k} < S_t \text{ for PUT}) \\ -1.00 & \text{if Trade Loses or Ties } (S_{t+k} \le S_t \text{ for CALL}) \end{cases}$$

- **Position Lock:** When a pair is traded, a 5-bar cooldown lock is engaged for that pair to prevent overlapping trades that distort exposure.
- **Split:** Chronological out-of-sample split: 111,538 bars train (70%) vs 47,802 bars test (30%).

---

## 3. Results & Empirical Breakdown

### 3.1 Performance Summary
- **Total Decision Steps:** 47,801 bars evaluated chronologically
- **Total Executed Trades:** 8,012 trades (executed on average every ~6 bars)
- **Wins:** 3,618
- **Losses:** 4,394 (includes ties as losses per binary options payout structure)
- **Win Rate:** **45.16%** (Breakeven requirement: 54.05%)
- **Wilson 95% Confidence Interval:** $[44.24\%, 46.08\%]$
- **Net Cumulative PnL:** $3,618 \times 0.85 - 4,394 \times 1.0 = 3,075.3 - 4,394.0 = -1,493.50 \text{ R}$

### 3.2 Action Distribution
| Action ID | Meaning | Count | Percentage |
| :--- | :--- | :--- | :--- |
| **0** | **HOLD** | **22,309** | **46.67%** |
| 1 | EURUSD CALL | 0 | 0.00% |
| 2 | EURUSD PUT | 3 | 0.01% |
| 3 | GBPUSD CALL | 1 | 0.00% |
| 4 | GBPUSD PUT | 83 | 0.17% |
| 5 | USDJPY CALL | 51 | 0.11% |
| 6 | USDJPY PUT | 517 | 1.08% |
| 7 | AUDUSD CALL | 1 | 0.00% |
| 8 | AUDUSD PUT | 0 | 0.00% |
| **9** | **USDCAD CALL** | **8,967** | **18.76%** |
| **10** | **USDCAD PUT** | **15,869** | **33.20%** |

---

## 4. Root Cause Analysis

### 4.1 Policy Collapse into High-Frequency USDCAD Flipping
The action distribution clearly illustrates a classic Reinforcement Learning pathology known as **policy collapse / local optima capture**:
- Over 98% of the agent's trades were concentrated entirely in **USDCAD** (Actions 9 & 10), alternating between Call and Put.
- Why did this occur? In a noisy multi-agent discrete action space where no single pair possesses a strong linear predictive advantage, the PPO policy gradient quickly zeroes in on the asset with the highest variance of rewards during initial rollouts. Once USDCAD parameters received slight initial gradient updates, the actor network biased towards USDCAD actions, reducing exploration in EURUSD, GBPUSD, AUDUSD, and USDJPY to near zero.

### 4.2 The "Garbage In, Garbage Out" Law of Meta-Agents
- A meta-agent's theoretical role is to learn *when* to trust base models and *which* market regimes warrant taking risk.
- However, as demonstrated in Phase 2 benchmarks, single-model M1 directional predictors on retail FX post-2022 hover around 47% to 49.7% accuracy.
- When an RL agent receives base signals whose true underlying probability is $p < 0.50$, it cannot construct a positive expectation policy unless it selectively filters down to an infinitesimal fraction of bars (acting almost entirely as HOLD).
- Instead, because PPO balances entropy exploration against expected reward, the agent found it impossible to discover a sparse threshold that cleared 54.05% out-of-sample, resulting in aggressive over-trading (8,012 trades).

### 4.3 Why Binary Options Mathematical Asymmetry Punishes RL
In standard financial RL (e.g. equities or futures), an agent can win by having a 40% win rate if the reward-to-risk ratio is $3:1$ (cutting losses short and letting winners run).  
In retail binary options:
- **Reward:** $+0.85$
- **Risk:** $-1.00$
- **R/R Ratio:** $0.85:1$ (Inverted risk/reward)

This requires the RL policy to be fundamentally an **ultra-high precision classifier** ($>54.05\%$). Standard policy gradient algorithms (PPO) optimized via MSE value estimators are notoriously weak at sparse high-precision classification under negative-sum conditions.

---

## 5. Holistic Synthesis Across All Three Tasks

Let us step back and look at the empirical truth across all three tasks requested by the user:

```
+-----------------------------------------------------------------------------------+
|                        THE COMPREHENSIVE EXPERIMENTAL SUITE                       |
+-----------------------------------------------------------------------------------+
| 1. FEATURE ABLATION (Phase 1, 3.28M bars, GBPUSD k=5)                             |
|    - Standard Baseline: 52.88% (Loss)                                             |
|    - Vol Squeeze (squeeze): 54.19% - 54.93% Win Rate | PBO 0.04-0.10 | POSITIVE EV |
|    - Macro Filter (dxy+cross): 65.96% Win Rate at buffer 0.06 (Ultra-rare trades) |
+-----------------------------------------------------------------------------------+
| 2. MODEL BENCHMARK (Phase 2, 1M bars, 2024-2026)                                  |
|    - LightGBM: 49.71% Win Rate                                                    |
|    - XGBoost:  47.34% Win Rate                                                    |
|    - CatBoost: 0 trades (Calibrated p < 54.05%)                                   |
|    - CNN+BiLSTM: 0 trades (Calibrated p < 54.05%)                                  |
|    * Conclusion: Post-2022 M1 market efficiency makes model type irrelevant.      |
+-----------------------------------------------------------------------------------+
| 3. RL META-AGENT (Phase 3, 5 Pairs, 150 Features, Zero Price)                     |
|    - PPO Multi-Pair Agent: 45.16% Win Rate | -1,493.5 R PnL                       |
|    - Policy Collapsed into USDCAD Flipping                                        |
|    * Conclusion: RL cannot turn negative EV inputs into positive EV outputs.     |
+-----------------------------------------------------------------------------------+
```

### The Definitive Scientific Answer
1. **Did Advanced Models Solve the Problem?** No. 1D-CNN+BiLSTM, CatBoost, and XGBoost performed equal to or worse than LightGBM. Deep Learning cannot extract non-existent signal from M1 noise.
2. **Did RL Meta-Agent Solve the Problem?** No. Unconstrained PPO over-traded and collapsed to 45.16% win rate.
3. **What ACTUALLY Worked?**  
   The **ONLY** component across the entire investigation that demonstrated legitimate, reproducible positive expected value was **Phase 1 Feature Engineering with Volatility Squeeze and DXY Alignment**:
   - `+squeeze`: Filtered trades to breakout expansions, achieving **54.2% – 54.9% win rate** across 3.28M bars.
   - `+dxy+cross`: Filtered trades to rare macro-synchronized momentum bursts, achieving up to **65.96% win rate**.

---

## 6. Project Recommendation

- **Do NOT deploy the RL Meta-Agent.** The policy is unstable and loses money rapidly.
- **Do NOT pursue Complex Deep Learning architectures (CNN/LSTM/Transformers) on 1-minute FX candles.** The signal-to-noise ratio is too low for deep parameterization.
- **IF trading binary options is mandatory:** The only empirically grounded path is an **ultra-selective rule-based / LightGBM filter** incorporating:
  1. Strict Volatility Squeeze condition (`squeeze_ratio < 0.75` transitioning to expansion).
  2. DXY and Cross-Asset directional confluence.
  3. Strict probability buffer ($p \ge 0.58$) taking fewer than 5–10 high-conviction trades per week rather than continuous trading.
