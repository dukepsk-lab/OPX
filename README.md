# OPX: Can Machine Learning Beat 1-Minute FX Binary Options?

> **Spoiler:** No. Not even with Deep Learning or Reinforcement Learning. Here is the math, the 20 million bars of Dukascopy tick data, and the story of what actually happened.

---

## 💡 The Premise

If you spend any time on trading forums or YouTube, you will see people selling "95% win rate binary options bots" or showing off color-coded indicators that allegedly print money on 1-minute candles.

As engineers and quants, we wanted to answer one straightforward question with zero bullshit:

> **Can modern machine learning (LightGBM, XGBoost, CatBoost, 1D-CNN+BiLSTM) or Reinforcement Learning (PPO) extract a statistically genuine edge over the broker's breakeven hurdle on 1-minute / 5-minute FX contracts?**

No Martingale. No compounding lottery bets. No unregulated broker "OTC" data feeds. Just clean tick data from Dukascopy, institutional-grade validation rules, and cold, hard statistics.

---

## 🧮 The Brutal Math of Binary Options

Before touching a single line of Python, you have to understand the math of retail binary options.

When a broker offers an **85% payout** ($r = 0.85$):
- If you win: you get **+$0.85$**
- If you lose: you lose **-$1.00$**
- If the price doesn't move (a tie): you lose your entire stake (**-$1.00$**)

To break even, your win rate $p^*$ must clear:

$$p^* = \frac{1}{1 + r} = \frac{1}{1 + 0.85} \approx 54.05\%$$

In professional quantitative finance, achieving a **52% directional edge** on liquid currencies like EUR/USD is considered world-class and can make a hedge fund billions.

To survive binary options, you have to hit **over 54.05% out-of-sample consistently**, after accounting for ties and commissions. That is an enormous statistical hurdle.

---

## 📓 The Quant Research Journal: Hypothesis, Trial, Failure, Repeat

The real value of this project wasn't finding a magic money machine—it was the quantitative discipline forged through rigorous trial, failure, and iteration. Here is the actual scientific journal of our research cycles:

### 🔄 Cycle 1: Directional Forecasting & Payout-Weighted GBDT
* **Hypothesis:** Standard machine learning uses symmetric cross-entropy loss, treating false positives and false negatives equally. If we weight the gradient objective directly by the binary options payout ratio ($r = 0.85$) and train LightGBM using forward-expiry labels, the model will identify directional trades with positive mathematical expectation.
* **Trial:** Implemented custom EV-weighted first- and second-order loss derivatives. Built purged 6-fold walk-forward cross-validation with a 5-minute embargo across 5 major pairs and 3 time horizons ($k \in \{1, 5, 15\}$ min) over 11.8 years of Dukascopy tick data (~20 million bars).
* **The "Success" (Trap):** The pipeline reported a profitable **GO** on `EURUSD k=15` (55.4% Win Rate) and `GBPUSD k=15` (55.8% Win Rate), beating the 54.05% breakeven hurdle!
* **The Audit & Failure:** We ran forensic audits on trade timestamps ([`docs/audit-report-eurusd-k15.md`](docs/audit-report-eurusd-k15.md)). 88% of the profitable trades occurred between 16:00 and 18:59 NY time (New York Rollover) and Sunday market opens. Spreads blow out from 0.2 to 5+ pips during rollover, and quote feeds freeze. Once illiquid hours were purged, the win rate plummeted to 52.88%. **Full NO-GO across all 15 setups.**
* **💡 Quant Lesson & Skill Acquired:** 
  - **Market Microstructure Trumps Pure Math:** If a backtest produces anomalous alpha, check execution feasibility first. Historical OHLC bars during illiquid bank rollover hours contain artificial spread artifacts that real traders cannot execute.
  - **Realistic Forward Expiry Contracts:** Real binary options lock entry at $O_{t+1}$ and expire at $C_{t+k}$, with price ties counting as total losses. Ignoring ties or entering at $C_t$ artificially inflates win rates by 2–4%.

---

### 🔄 Cycle 2: Probability Calibration & The Overconfidence Illusion
* **Hypothesis:** GBDT decision tree raw leaf scores are not true probabilities. If we fit a Platt Scaling (logistic sigmoid) model out-of-sample, we can filter for only ultra-high-conviction trades where calibrated confidence clears $P(\text{Win}) > 0.5405 + \text{buffer}$.
* **Trial:** Implemented out-of-sample Platt calibration strictly fitted on validation folds and evaluated on held-out test folds.
* **Result & Failure:** The raw LightGBM output frequently boasted $P > 0.65$. But once Platt calibration was applied on out-of-sample data, the true empirical probabilities collapsed into a narrow, flat band between $0.48$ and $0.52$. Virtually no bars cleared the required $54.05\%$ threshold. Lowering the buffer forced the model to trade, but it reverted directly to an unprofitable 49–51% win rate.
* **💡 Quant Lesson & Skill Acquired:** 
  - **Tree Models Suffer from Severe Probability Distortion:** GBDT algorithms are notorious for outputting overconfident probabilities on noisy financial data. 
  - **Out-of-Sample Calibration is Non-Negotiable:** Without out-of-sample Platt or isotonic calibration, confidence thresholds in trading algorithms are an illusion that guarantees live blowups.

---

### 🔄 Cycle 3: Feature Engineering & Regime Shifts (The Only Real Alpha)
* **Hypothesis:** Single-pair technical indicators (RSI, MACD, Bollinger Bands) are memoryless and noisy. If we add Macro Dollar Index (DXY) momentum, Cross-Asset relative returns against the USD basket, Volatility Squeeze ratios (Bollinger Bands compressing inside Realized Volatility), and Fractional Differentiation ($d=0.35$ to preserve memory while achieving stationarity), the model can detect structural breakout regimes.
* **Trial:** Downloaded 2.72 million bars of M1 DXY data. Built an 11-configuration feature ablation matrix and evaluated across 3.28 million bars of GBPUSD $k=5$ (2018–2026).
* **Result & Discovery:**
  - Standard baseline: 52.88% (Failed)
  - `+squeeze`: **54.19% to 54.93% Win Rate across 3.28M bars** with positive EV, solid trade counts (1,400–1,900 trades), and negligible overfitting (PBO 0.04 to 0.10).
  - `+dxy+cross`: **65.96% Win Rate** at buffer 0.06 (EV +0.22 R), but trade volume collapsed to only 47 trades in 8 years.
* **💡 Quant Lesson & Skill Acquired:** 
  - **Regimes > Point Predictions:** You cannot predict directional noise in a choppy market. The only verifiable alpha came from identifying *volatility transition regimes* (compression into expansion) confirmed by macro currency index momentum.
  - **The Precision vs. Opportunity Tradeoff:** A 66% win rate that triggers 5 times a year cannot pay the server bills. Trading systems must balance edge precision against statistical opportunity frequency.

---

### 🔄 Cycle 4: Model Complexity Escalation (Deep Learning & SOTA Trees)
* **Hypothesis:** LightGBM's shallow orthogonal splits may fail to capture complex temporal patterns and feature interactions. Advanced architectures—CatBoost (symmetric oblivious trees), XGBoost (exact Hessian), and a PyTorch 1D-CNN + BiLSTM network—will extract latent sequential representations that GBDTs miss.
* **Trial:** Implemented custom EV-weighted objectives in CatBoost and XGBoost. Built a deep PyTorch neural network combining 1D convolutions (spatial feature extraction across indicators) with Bidirectional LSTM (temporal memory over lookback windows) on an RTX 5070 GPU. Evaluated on recent market data (1,000,000 bars, 2024–2026).
* **Result & Failure:**
  - XGBoost: 47.34% Win Rate (Overfit to noise).
  - LightGBM: 49.71% Win Rate.
  - CatBoost: 0 trades taken.
  - 1D-CNN + BiLSTM: 0 trades taken.
* **💡 Quant Lesson & Skill Acquired:** 
  - **The Complexity Penalty in Low SNR Environments:** In financial time series where signal-to-noise ratio is near zero, increasing model capacity (more layers, bi-directional attention, recurrent units) simply gives the model more ropes to hang itself. The model memorizes training noise rather than discovering fundamental laws.
  - **Post-2022 Market Efficiency:** Modern retail FX on M1 is closer to a martingale than it was a decade ago. If the underlying data has no linear or non-linear signal, no deep learning architecture can manufacture it.

---

### 🔄 Cycle 5: Multi-Pair Reinforcement Learning Meta-Agent
* **Hypothesis:** Instead of training individual models to guess direction every minute, an RL Meta-Agent observing 150 non-price features across 5 currency pairs can act as a high-level portfolio conductor. It can learn when to stay in cash (HOLD = 0 reward) and only execute CALL or PUT on specific pairs when cross-market alignment is overwhelmingly favorable.
* **Trial:** Designed a Gymnasium environment (`MultiPairBinaryOptionsEnv`) with Discrete(11) action space, $+0.85$ win / $-1.00$ loss / $0.00$ hold rewards, and 5-bar position lockouts. Strictly excluded all raw OHLC prices to eliminate non-stationarity. Trained PPO for 100,000 timesteps across 111,538 training bars and tested on 47,801 unseen bars.
* **Result & Failure:** Out-of-sample Win Rate: **45.16%** (-8.89% below breakeven). Net PnL: **-1,493.50 R**. The policy suffered severe **action collapse**, spending 52% of all bars repeatedly flipping USDCAD CALL and PUT (actions 9 & 10) while ignoring other pairs.
* **💡 Quant Lesson & Skill Acquired:** 
  - **The "Garbage In, Garbage Out" Law of Financial RL:** RL agents optimize for cumulative reward. When the underlying feature signals have negative or zero expected value, the policy gradient cannot discover an optimal sparse policy; it latches onto the asset with the highest reward variance during initial exploration and spirals into catastrophic over-trading.
  - **Discrete Policy Degeneracy:** In continuous state spaces with negative-sum payoffs, policy gradient methods require heavy entropy tuning and explicit trade transaction costs, or they will rapidly degenerate into gambling loops.

---

## 🎓 The Quant Skill Matrix Acquired

Working through this project from start to finish built a complete toolkit of institutional quantitative finance skills:

| Quant Domain | Method / Tool Implemented | Practical Engineering Skill |
| :--- | :--- | :--- |
| **Validation Rigor** | Purged Walk-Forward CV + Embargo (`opx.cv`) | Eliminates serial correlation and train/test leakage in overlapping forward labels. |
| **Microstructure Audit** | Time-of-Day Liquidity Filters (`opx.data`) | Prevents illusory alpha caused by bank rollover spreads and market open gaps. |
| **Probability Calibration** | Out-of-Sample Platt Scaling (`opx.evaluate`) | Neutralizes tree overconfidence and provides true empirical probability estimates. |
| **Custom Loss Functions** | EV-Weighted First & Second Derivatives (`opx.objective`) | Aligns optimization loss surfaces with asymmetric financial payouts ($+r$ vs $-1$). |
| **Statistical Gating** | Wilson 95% LCB, Deflated Sharpe, PBO (`opx.stats`) | Distinguishes genuine statistical edge from multiple-testing data snooping. |
| **Regime Detection** | Volatility Squeeze & Cross-Asset DXY (`opx.features_extra`) | Identifies structural market compression and macro-driven breakout momentum. |
| **Deep Learning Diagnostics** | 1D-CNN + BiLSTM PyTorch GPU (`opx.models.nn_model`) | Diagnoses when model complexity overfits noise and hurts out-of-sample generalization. |
| **RL Environment Design** | Multi-Pair Gymnasium Env (`opx.rl.env`) | Models discrete portfolio actions, non-price stationary states, and position cooldown locks. |

---

## 🧱 What's in this Repository

Even though the final verdict for automated binary options trading is **STRICT NO-GO**, the codebase contains a complete, battle-tested quant research framework you can repurpose for spot FX, futures, or equities:

- [`src/opx/data.py`](src/opx/data.py) — Dukascopy parquet tick/M1 processor with illiquid hour filters.
- [`src/opx/labels.py`](src/opx/labels.py) — Purged binary options forward-expiry labeler (ties = loss).
- [`src/opx/features.py`](src/opx/features.py) & [`features_extra.py`](src/opx/features_extra.py) — Technical features, fractional differentiation, Bollinger squeeze, and cross-asset relative momentum.
- [`src/opx/objective.py`](src/opx/objective.py) — Custom asymmetric loss functions for LightGBM, XGBoost, and CatBoost weighted by payout ratios.
- [`src/opx/cv.py`](src/opx/cv.py) — Purged walk-forward cross-validation with embargo periods to prevent lookahead bias.
- [`src/opx/stats.py`](src/opx/stats.py) — Quantitative metrics: Wilson 95% Score interval, Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO), and Monte Carlo permutation tests.
- [`src/opx/rl/`](src/opx/rl/) — Complete Gymnasium environment for multi-pair fixed-time options portfolio management and PPO training pipelines.
- [`docs/`](docs/) — Full audit reports, ablation benchmarks, and experimental logs.

---

## 🚀 Getting Started & Running Tests

Clone the repository and install dependencies:

```bash
git clone https://github.com/dukepsk-lab/OPX.git
cd OPX
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run the complete test suite (75 unit & integration tests):

```bash
pytest
```

---

## 📜 Final Word & Ethics

We built this project with rigorous scientific methodology and published all of it—including every failure, dead-end, and negative result—as open source. 

If this repository saves even one person from paying $500 for a fake "AI Binary Options Robot" or losing their savings to a Martingale bot on Telegram, it did its job.

Trade safe, test your assumptions out-of-sample, and respect the math.
