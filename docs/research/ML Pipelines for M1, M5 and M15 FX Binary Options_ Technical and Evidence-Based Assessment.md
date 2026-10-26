# ML Pipelines for M1/M5/M15 FX Binary Options (IQ Option-style Fixed-Time Trades): A Technical and Evidence-Based Assessment

You can build a technically sound ML pipeline for M1/M5/M15 binary-option direction prediction, but no credible published evidence shows that a sustained out-of-sample win rate above the 54–56% payout breakeven holds after execution lag, honest multiple-testing correction and the broker's built-in edge. Treat this as a no-go for real capital by default. Change that only if a strategy passes a strict validation gate on settlement-consistent data, and never on OTC synthetic feeds.

## TL;DR
- **The maths is unforgiving.** At payout r, breakeven is p* = 1/(1+r): 52.6% at 90%, 54.1% at 85%, 55.6% at 80%. A coin-flip trader at an 85% payout loses 7.5% of every stake. Rigorous intraday studies, by contrast, find raw directional accuracy of about 50–51% out of sample. Mathias Mesfin's 2026 arXiv study (2605.17724) of gradient boosting and LSTM on 944 MNQ futures trading days, for example, reached at most 50.89% against a 51.8% base rate.
- **The method matters more than the model.** Use gradient-boosted trees on hand-built features. Label from the actual entry quote after execution lag, with ties scored as losses. Train with cost-weighted logloss (positive weight r, which moves the Bayes threshold to p*). Calibrate, add a meta-label filter, and validate with purged walk-forward/CPCV plus the Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO). Size with fractional Kelly on the lower confidence bound of p. Never use Martingale.
- **Structural and legal risks dominate the modelling risks.** OTC weekend pairs are broker-generated prices from the platform's own algorithms, and the broker is also your counterparty. Binary options are banned for retail clients in the EU and UK. In Thailand they are not a licensed product, and the Thai SEC has warned about iqoption.com by name. IQ Option's terms let it close accounts and withhold profits over abusive high-frequency software, and the community Python API says "Don't use it on your real Account!!!"\[1\]

## Key Findings

| Question | Evidence-based answer |
|---|---|
| Breakeven win rate | p* = 1/(1+r). 80% → 55.56%, 85% → 54.05%, 90% → 52.63%. Scoring ties as losses raises the effective bar. |
| Realistic OOS accuracy, M1–M15 | Typically 50–52% raw. Claims of 58–63%+ are usually daily horizons, in-sample results, leaky or unreplicated. |
| Best model family | GBDT (LightGBM/CatBoost/XGBoost) on engineered features. Sequence models rarely beat it on noisy, low signal-to-noise tabular data. |
| Is a fixed P > 0.60 filter right? | Only if probabilities are calibrated. The EV-optimal threshold is p* plus an uncertainty buffer, set on purged out-of-sample folds. |
| Martingale | Does not change EV per dollar staked. It raises turnover and puts ruin risk in the tail. With 1,000 units and a 2.18× ladder at 55% win rate, ruin over 5,000 trades is about 60% (approximate). |
| Sizing | Kelly for binaries: f* = p − (1−p)/r. Use ¼-Kelly on a lower confidence bound of p. Fixed 0.5–1% is a sane fallback. |
| OTC weekend pairs | Proprietary synthetic feed. Models trained on interbank data do not transfer, and any "edge" sits on a process controlled by your counterparty. |
| Automation | Only unofficial, reverse-engineered WebSocket APIs exist.\[2\] Account-termination clauses apply. Prefer signal alerting with manual execution, or move the edge to a regulated venue. |

## 1. Mathematical Formulation & Target Labeling

### 1.1 Target definition (with execution lag and ties)
Let P(τ) be the platform's settlement quote, not the interbank mid. The signal fires at the close of bar t. The contract opens at τ_e = t + δ, where δ is the execution delay. It expires at τ_x = τ_e + k·Δ, with Δ ∈ {1m, 5m, 15m}. Define two directional labels so that ties count as losses on both sides:

- y_call = 1{P(τ_x) > P(τ_e)}
- y_put = 1{P(τ_x) < P(τ_e)}
- tie = 1{P(τ_x) = P(τ_e)}, which is a loss for both under your conservative convention.

The textbook version, Y = 1 if Close(t+k) > Close(t), has two quiet flaws. It assumes entry at the close of t (zero lag). It also folds ties into the "down" class, which means a Put model trained on 1 − Y would be paid for ties it would actually lose. Train separate Call and Put heads, or use a three-class target. A "doji" (open = close inside a bar) is a feature, not a label. The label tie is entry quote = expiry quote, and its frequency depends on quote precision (5-digit vs 3-digit for JPY pairs), so round to the platform's tick size before comparing. Whether ties are refunded or lost varies by platform and contract type. Scoring them as losses is the conservative choice; verify the actual rule in the contract specs.

```python
import numpy as np, pandas as pd

def make_labels(quotes: pd.Series, k_bars: int, lag_bars: int = 1, tick: float = 1e-5) -> pd.DataFrame:
    """quotes: settlement-feed prices sampled on a fine grid (e.g., 1s); k/lag in grid units."""
    q = (quotes / tick).round().astype("int64")            # integer pipettes → well-defined ties
    entry, expiry = q.shift(-lag_bars), q.shift(-(lag_bars + k_bars))
    out = pd.DataFrame(index=quotes.index)
    out["y_call"] = (expiry > entry).astype(float)
    out["y_put"]  = (expiry < entry).astype(float)
    out["tie"]    = (expiry == entry).astype(float)
    out.loc[entry.isna() | expiry.isna()] = np.nan
    out["t1"] = quotes.index.to_series().shift(-(lag_bars + k_bars))  # label end time, for purging
    return out
```

### 1.2 Breakeven derivation
Stake 1 unit. A win pays +r and a loss (including a tie) costs −1. For win probability p, EV = p·r − (1−p) = p(1+r) − 1, so EV > 0 ⇔ p > p* = 1/(1+r).

| Payout r | p* | EV at p = 0.50 | EV at p = 0.55 | EV at p = 0.58 |
|---|---|---|---|---|
| 0.80 | 55.56% | −10.0% | −1.0% | +4.4% |
| 0.85 | 54.05% | −7.5% | +1.75% | +7.3% |
| 0.90 | 52.63% | −5.0% | +4.5% | +10.2% |

Note that the ">55.6%" figure in your spec applies only at an 80% payout. The broker also sets payouts per asset and per moment (they fall in volatile or illiquid conditions), so r is a random variable you must log for every trade.

### 1.3 Why accuracy and precision are the wrong objectives
- **The payoff is asymmetric.** Accuracy weighs a correct "no trade" the same as a correct trade and ignores that a loss costs 1/r times as much as a win earns.
- **You choose when to trade.** What matters is EV over the subset you trade, which depends on coverage × (precision − p*). High precision on 0.1% of bars may be statistically meaningless. Moderate precision on 30% of bars may lose money.
- **Empirical confirmation.** In a 2024 EUR/USD daily-direction study, a decision-tree meta-estimator with 55.31% accuracy returned 32.48%, while an SVM with higher accuracy (55.63%) returned only 4.25%.\[3\] Ranking by accuracy picks the wrong model.

### 1.4 EV-consistent training objective
**Cost-weighted logloss (recommended, convex).** Weight positives by r and negatives by 1. Minimising this weighted logloss gives the score q = r·p / (r·p + 1 − p), and q > 0.5 ⇔ p > 1/(1+r). The Bayes decision boundary of the weighted model therefore sits at the payout breakeven. You can recover the true probability with p = q / (q + r(1 − q)).

```python
import numpy as np, lightgbm as lgb

R = 0.85  # net payout on win (log the realised payout per trade in production)

def ev_weighted_logloss(y_true, raw):
    p = 1.0 / (1.0 + np.exp(-raw))
    w = np.where(y_true == 1, R, 1.0)
    return w * (p - y_true), np.maximum(w * p * (1.0 - p), 1e-6)   # grad, hess

def ev_per_trade(y_true, raw):
    q = 1.0 / (1.0 + np.exp(-raw))
    take = q > 0.5                                   # == p > 1/(1+R) under the weighting
    if take.sum() < 50:
        return "ev_per_trade", -1.0, True            # punish degenerate low-coverage solutions
    pnl = np.where(y_true[take] == 1, R, -1.0)
    return "ev_per_trade", float(pnl.mean()), True

clf = lgb.LGBMClassifier(objective=ev_weighted_logloss, n_estimators=3000, learning_rate=0.02,
                         num_leaves=15, min_child_samples=500, subsample=0.7, subsample_freq=1,
                         colsample_bytree=0.7, reg_lambda=5.0)
clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric=ev_per_trade,
        callbacks=[lgb.early_stopping(200)])
q = 1 / (1 + np.exp(-clf.predict(X_te, raw_score=True)))     # custom objective → raw margins
p = q / (q + R * (1 - q))                                     # back out P(win)
```

XGBoost takes the same function as `obj=` in `xgb.train` (signature `(predt, dtrain)`, label from `dtrain.get_label()`). The shortcut `scale_pos_weight=R` gives the same weighting.

**Direct "soft-EV" loss (use as metric, not objective).** L(z) = −σ(z)·[r·y − (1−y)] maximises expected profit when the decision is relaxed into a soft one. However, its Hessian, −σ(1−σ)(1−2σ)·[r·y − (1−y)], is not positive semi-definite, and the optimum drifts toward all-or-nothing trading. Use it for early stopping and Optuna objectives, not for tree-split gain.

**Recommended practice.** Train with plain or cost-weighted logloss, calibrate with isotonic or Platt scaling on purged out-of-sample folds, then decide with an explicit EV rule. Calibration and threshold choice matter more than the loss.

## 2. High-Frequency Feature Engineering (M1/M5)

**What the microstructure literature implies.**
- **Order flow imbalance (OFI).** Cont, Kukanov and Stoikov (J. Financial Econometrics 2014; NYSE TAQ, 50 stocks) show that over short intervals price changes are mainly driven by OFI, with a linear relation whose slope is inversely proportional to market depth.\[4\] Retail FX platforms do not expose a consolidated limit order book, so you can only approximate OFI. Use a tick-rule signed tick count (upticks − downticks) per bar, or quote-update imbalance if the feed carries bid and ask.
- **Tick volume.** FX has no consolidated volume. An often-quoted industry study (Casper Marney, 2011, republished by Varianse) reports about 90% correlation between tick volume and actual ECN traded volume for EUR/USD, GBP/USD, USD/JPY and EUR/CHF.\[5\]\[6\]\[7\] It is industry, not peer-reviewed. Tick volume also differs by broker, and the platform's own tick count partly reflects its own quote generation.\[6\]\[8\]
- **Negative first-order autocorrelation at the highest frequencies** is a long-documented stylised fact of FX (Goodhart and Figliuoli, 1991; Dacorogna et al., 2001).\[9\]\[10\] Much of it is bid-ask bounce (Roll, 1984), which produces negative autocorrelation even when the efficient price is a random walk.\[11\]\[12\] An M1 mean-reversion signal can therefore be a quote-construction artefact that you cannot monetise once you add the entry lag. Test it on the exact settlement series.
- **Intraday periodicity is strong in volatility, not in direction.** Andersen and Bollerslev's 5-minute DM/$ work found average absolute returns ranging from about 0.01% around 04:00 GMT to about 0.05% at peaks. They also found that the time series of volatility cannot be modelled meaningfully without first removing this periodicity.\[13\]\[14\] Session encodings mostly help as volatility and regime conditioners, not as directional predictors.
- **Intraday momentum** is documented at half-hour granularity in equity ETFs (Gao, Han, Li and Zhou, JFE 2018: the first half-hour return predicts the last half-hour return in SPY over 1993–2013).\[15\] It is stronger when liquidity is low and volatility high (12 of 16 developed markets in a global follow-up).\[16\]\[17\] This is weak support for M15 momentum features in FX around session opens and closes. It says little about M1.

**Feature set (all computed only from information at or before t).**

| Family | Features |
|---|---|
| Candle geometry | body/range = abs(C−O)/(H−L); upper_wick/range; lower_wick/range; close location value (C−L)/(H−L); doji flag (abs(C−O) ≤ 1 tick) |
| Returns | log-return lags r_{t−1..t−n}; multi-scale sums (1, 3, 5, 15, 60 bars); ATR-normalised returns r_t / ATR_14; signed run length |
| Volatility/regime | ATR percentile vs a 20-day same-time-of-day baseline; realised variance; Parkinson/Garman-Klass range vol; variance ratio VR(q); Kaufman efficiency ratio; ADX |
| Mean reversion | RSI(2/7/14) and distance from 30/70; Bollinger %B and pierce flags (close beyond ±2σ then re-entry); z-score of close vs VWAP proxy |
| Micro S/R exhaustion | distance to rolling N-bar high/low in ATR units; number of touches within ε; failed-breakout flag (wick beyond level, close inside) |
| Flow proxies | tick count z-score vs time-of-day norm; signed tick imbalance; bid/ask spread and its z-score if available |
| Calendar | sin/cos of minute-of-day (UTC); sin/cos of day-of-week; session flags (Tokyo, London, NY, London–NY overlap) built from local exchange time so DST is handled; minutes to/from scheduled macro releases |
| Cross-asset | DXY-proxy return (basket of USD pairs); EUR/USD–GBP/USD spread return; USD/JPY vs risk proxy where available |

```python
import numpy as np, pandas as pd, talib

def features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (df[x] for x in ["open", "high", "low", "close", "tick_volume"])
    rng = (h - l).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    f["body_ratio"] = (c - o).abs() / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["clv"] = (c - l) / rng
    lr = np.log(c).diff()
    atr = pd.Series(talib.ATR(h, l, c, 14), index=df.index)
    for k in (1, 2, 3, 5, 15):
        f[f"lr_{k}"] = lr.rolling(k).sum()
        f[f"lr_{k}_atr"] = f[f"lr_{k}"] / (atr / c)
    f["rsi7"] = talib.RSI(c, 7)
    up, mid, lo = talib.BBANDS(c, 20, 2, 2)
    f["bb_pctb"] = (c - lo) / (up - lo)
    f["bb_pierce_dn"] = ((l < lo) & (c > lo)).astype(int)      # wick pierce + close back inside
    f["bb_pierce_up"] = ((h > up) & (c < up)).astype(int)
    f["tickvol_z"] = (v - v.rolling(240).mean()) / v.rolling(240).std()
    m = df.index.hour * 60 + df.index.minute                  # index must be tz-aware UTC
    f["tod_sin"], f["tod_cos"] = np.sin(2*np.pi*m/1440), np.cos(2*np.pi*m/1440)
    dow = df.index.dayofweek
    f["dow_sin"], f["dow_cos"] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
    ldn = df.index.tz_convert("Europe/London").hour
    ny = df.index.tz_convert("America/New_York").hour
    f["ldn_ny_overlap"] = ((ldn >= 8) & (ldn < 17) & (ny >= 8) & (ny < 17)).astype(int)
    eff = (c - c.shift(20)).abs() / c.diff().abs().rolling(20).sum()
    f["kaufman_er"] = eff
    return f
```

Discipline matters here. Every indicator with a lookback creates overlap between adjacent samples. Normalise per time of day so the model is not simply learning the volatility clock. Prune with clustered MDA/SFI importance on purged folds rather than in-sample gain.

## 3. Model Selection & Regime Filtering

### 3.1 GBDT vs sequence models
- **Benchmark evidence.** Grinsztajn, Oyallon and Varoquaux (NeurIPS 2022, 45 datasets) found tree-based models remain state of the art on medium-sized tabular data (about 10K samples), partly because neural nets are hurt by uninformative features.\[18\] Low-SNR financial features are the extreme case of that.
- **Direct intraday evidence.** Mathias Mesfin's 2026 walk-forward study (arXiv:2605.17724, 944 MNQ trading days, 2021–2025) found that no configuration beat the 51.80% intraday base rate. Out-of-sample accuracy was GB-Daily 50.00%, GB-Intraday 50.45% (permutation p = 0.135), GB-VolAdj 50.89% (p = 0.390) and LSTM 50.59% (p = 0.515), and the LSTM "converges to random prediction". On minute-level SPY, the 13 random-forest configurations in arXiv:2412.15448 went from 80.5–86.7% training directional accuracy to 47.6–49.5% test accuracy, with R² falling from 0.749–0.812 in training to negative values in testing, a textbook overfitting gap.
- **Positive results do exist, at lower frequencies.** The ACM paper "Predicting Foreign Exchange EUR/USD Direction Using Machine Learning" reports 58.52% for one-day-ahead direction (histogram gradient boosting, 2022 test year). The Duke "factor zoo" paper (Aleti, Bollerslev, Siggaard) finds genuine 15-minute predictability in the aggregate equity market, but only from a wide cross-section of high-frequency factor returns, where a simple AR(1) on 15-minute returns gives an R² of 0.00008%.\[19\] Studies reporting 60–70%+ M1 "accuracy" with LSTMs usually predict multi-bar trend classes, use in-sample validation, or ignore costs. One example is the MDPI Applied Sciences paper on a deep LSTM with an RL layer, which reports 69.25–73.98% yearly EUR/USD M1 accuracy (about 70% on average) for a 3-class trend over the next 100 quotes. Treat them as hypotheses, not benchmarks.

**Verdict.** Use LightGBM or CatBoost as the primary model (CatBoost's ordered boosting is useful with small effective sample sizes). A 1D-CNN or small Transformer on raw tick or quote sequences is worth trying only if you have true tick data, then stacked as a feature into the GBDT. Expect it to add variance, not edge.

### 3.2 Meta-labeling and confidence thresholds
1. **Primary model (side).** The GBDT or a simple rule (e.g., Bollinger pierce + RSI extreme) proposes Call or Put.
2. **Meta-label.** y_meta = 1 if the primary's bet would have won after lag and tie handling, else 0. Train a second classifier on features describing regime and context: primary score, ATR percentile, spread z-score, Kaufman ER, VR(q), session, minutes to news, recent primary hit rate, and the current payout r.
3. **Calibrate** the meta probabilities (isotonic on purged out-of-sample predictions).
4. **Decision rule.** Trade if p_meta,LCB > p*(r_t) + buffer, where p_LCB is a lower confidence bound (e.g., from bootstrap or conformal calibration within the score bin). Stay flat otherwise, and always within ±N minutes of scheduled high-impact releases and during the daily rollover and illiquid hours.

A fixed 0.60 cut-off is arbitrary. At r = 0.85 the true breakeven is 0.5405. If your probabilities are well calibrated, 0.60 gives up coverage. If they are poorly calibrated (the usual case), 0.60 offers no real protection. Choose the buffer by maximising the deflated out-of-sample EV across thresholds, and count each threshold you try as a trial for DSR.

## 4. Risk Management & Position Sizing

### 4.1 Martingale
To recover all prior losses plus one base profit after a loss at payout r, the next stake must be s_{n} = s_0·m^n with m = (1+r)/r (≈2.18 at r = 0.85). Cumulative loss after n straight losses is s_0·r·(m^n − 1). After 8 losses that is ≈427 base units, and the 9th stake alone is ≈503 units.

- **EV does not change.** Each trade's EV is stake × (p(1+r) − 1), so total EV = (p(1+r) − 1) × total turnover. In a negative-EV game Martingale scales the loss by inflating turnover. In a marginally positive one it swaps a steady edge for rare total ruin. Platform maximum-stake caps also cut the ladder off early.
- **Losing streaks are guaranteed at scale.** The probability of at least one run of L losses in N trades is approximately 1 − exp(−N·p·q^L). With a 1,000-unit bankroll (survives 9 losses at m ≈ 2.18), ruin requires 10 straight losses: about 61% over 5,000 trades at p = 0.55, and about 82% at p = 0.52. These are my approximations; confirm by simulation.

```python
def kelly_binary(p, r):           # growth-optimal fraction for +r / -1 payoff
    return max(0.0, p - (1 - p) / r)

def martingale_ruin(p=0.55, r=0.85, bank=1000.0, base=1.0, n=5000, sims=20000, seed=0):
    rng, m, ruined = np.random.default_rng(seed), (1 + r) / r, 0
    for _ in range(sims):
        B, s = bank, base
        for _ in range(n):
            if s > B: ruined += 1; break
            if rng.random() < p: B += s * r; s = base
            else: B -= s; s *= m
    return ruined / sims
```

### 4.2 Kelly vs fixed-fractional
Kelly maximises g(f) = p·ln(1 + f·r) + (1−p)·ln(1 − f), giving f* = p − (1−p)/r. At r = 0.85: p = 0.55 → 2.1%, p = 0.57 → 6.4%, p = 0.59 → 10.8%. A ±2-percentage-point error in p swings the "optimal" bet five-fold, and overbetting beyond 2f* has negative growth. Use κ·f*(p_LCB, r) with κ = 0.25, where p_LCB is a lower confidence bound on the live or out-of-sample win rate, and cap at 1–2% per trade. Fixed-percentage sizing (0.5–1%) cannot hit literal zero and is the right default until there are at least 1,000 live trades. Add a hard daily loss limit and a kill switch when the live win rate drops below p* at 90% one-sided confidence.

## 5. Backtesting Pitfalls in Binary Options

### 5.1 Leakage control
- **Purging.** Each sample's information set spans [t − lookback, t1], where t1 = t + lag + k. Remove from training any sample whose t1 is at or after the test-fold start. Add an embargo after each test fold (e.g., 1% of samples, or at least the longest feature lookback) when training data follows the test fold, as in CPCV.
- **Walk-forward (expanding or rolling) with nested tuning.** Optuna runs inside the inner purged folds only. The outer folds are touched once.
- **CPCV** (combinatorial purged CV) produces a distribution of out-of-sample paths, which you need for PBO.
- **Point-in-time features.** Calendar and news features must use release timestamps as they were known, and candle aggregation must match the platform's (bar boundaries, timezone, which side of the quote).

```python
def purged_walk_forward(t0: pd.DatetimeIndex, t1: pd.Series, n_splits=8, gap=pd.Timedelta("30min")):
    idx = np.arange(len(t0)); blocks = np.array_split(idx, n_splits + 1)
    for b in blocks[1:]:
        test_start = t0[b[0]]
        train = idx[: b[0]]
        train = train[t1.iloc[train].values < (test_start - gap)]   # purge overlapping labels
        yield train, b
```

### 5.2 Multiple testing and overfitting
- **Deflated Sharpe Ratio** (Bailey and López de Prado, JPM 40(5), 2014).\[20\] Compare the observed Sharpe to the expected maximum Sharpe from N unskilled trials, SR₀ ≈ √V[SR]·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))], and correct for skewness and kurtosis: DSR = Φ((SR̂ − SR₀)√(T−1) / √(1 − γ₃SR̂ + (γ₄−1)/4·SR̂²)). Binary P&L (+r/−1) is strongly non-normal, so the correction matters.
- **PBO via CSCV** (Bailey, Borwein, López de Prado and Zhu, J. Computational Finance 20(4), 2017)\[21\] estimates the probability that the in-sample-best configuration underperforms the median out of sample. Log every configuration you try (features, thresholds, horizons, pairs, sessions), because each one counts as a trial.
- **Sample size needed just to clear breakeven** (one-sided 95%, independent trades): to show p > 0.5405 (r = 0.85) you need about 760 trades if the true p is 0.57, and about 1,750 if it is 0.56. At r = 0.80 with p = 0.58 you need about 1,100. Serial correlation between overlapping trades and multiple testing both push these numbers up materially.

### 5.3 Execution lag and feed mismatch
- **Lag.** Measure the full chain: bar close, feature computation, inference, alert or API call, platform acceptance, and the strike timestamp. Simulate δ from the empirical distribution (including the tail) and relabel with the strike at t + δ. Any M1 edge built on bounce or reversal often disappears with a lag of 1–3 seconds.
- **Expiry mechanics.** Platforms often align expiries to clock boundaries with purchase cut-offs, so the realised holding period ≠ k·Δ. Model the actual contract specification.
- **Settlement feed ≠ interbank feed.** You are paid on the broker's quote stream. Train and validate on recorded platform quotes. Use interbank data (e.g., Dukascopy) only for pre-training and robustness checks, and measure the basis between the two feeds.
- **Payout drift.** Log r at the moment of entry. Backtests that assume a constant 85–90% overstate EV whenever payouts fall in the volatile regimes your model likes.

### 5.4 OTC weekend pairs
- **How the prices are made.** IQ Option's own blog says OTC prices "are generated through the platform's proprietary algorithms" and that the platform "uses internal pricing models to simulate market conditions". It describes the result as "a closed ecosystem" with "less transparency".\[22\] Independent descriptions agree that EUR/USD-OTC does not follow the interbank rate on a Saturday.\[23\]
- **Conflict of interest.** In a retail binary the provider is the counterparty. ESMA's 2018 intervention cited binary options' "structural expected negative return and embedded conflict of interest between providers and their clients".\[24\] With OTC the counterparty also produces the price process you are trying to predict. The CFTC/SEC investor alert on binary options lists, as a fraud category in some platforms (not a finding about any specific broker), "manipulation of the binary options trading software to generate losing trades".\[25\]
- **Why models don't transfer.** There is no order flow, no news, no cross-venue arbitrage, and no shared microstructure. The generator's parameters can change without notice, so non-stationarity is built in. A pattern you "discover" in OTC data is at best a property of a pseudo-random generator, which the owner can recalibrate once it becomes unprofitable. No external historical OTC data exists, so you must record your own.
- **Position.** Exclude OTC from capital deployment. If you research it at all, run randomness tests first (runs test, variance ratio, autocorrelation of signs, entropy) and treat it as a separate asset with its own validation.

## 6. Python Tech Stack & Automation Architecture

| Layer | Recommendation | Notes |
|---|---|---|
| Interbank history | Dukascopy tick/M1 (e.g., via `dukascopy-node` or `duka`), HistData, broker MT5 via `MetaTrader5` package | For pre-training and robustness only |
| Settlement-feed capture | Recorder of platform candles and quotes | The only valid source for labels; unofficial APIs (below) carry ToS risk |
| Storage | Parquet + DuckDB/Polars; tz-aware UTC indices | Keep raw quotes immutable |
| Features | pandas/Polars, TA-Lib (C-backed, fast), pandas-ta (pure Python, convenient), numba for custom loops | Unit-test every feature for look-ahead |
| Models | LightGBM, CatBoost, XGBoost; PyTorch only for optional sequence encoders | GBDT first |
| Validation | Own purged-WF/CPCV implementation; DSR/PBO code from the papers | Log all trials |
| Tuning | Optuna (TPE + MedianPruner) with objective = purged OOS deflated EV, not accuracy | Count trials for DSR |
| Calibration | scikit-learn `CalibratedClassifierCV` (isotonic) on held-out purged folds | Recalibrate on a schedule |
| Tracking | MLflow (params, data hashes, trial counts) | Required for honest DSR |
| Serving/alerts | APScheduler or asyncio loop triggered at bar close; python-telegram-bot / Discord webhook; Prometheus/Grafana for latency and win-rate drift | Alert-first design |
| Execution | Manual on alert, or a regulated venue's official API | See risks below |

**Unofficial IQ Option APIs.** The community `iqoptionapi` project is a reverse-engineered WebSocket client. It pins old dependencies (e.g., `websocket-client==0.56`) and its README says "ONLY FOR STUDY - Don't use it on your real Account!!!".\[1\]\[2\]\[26\] IQ Option's General Terms (the Digital Invest Ltd / Seychelles version, clause 16.4.b.xiv) allow immediate termination, closing of positions and blocking without notice where a client uses "a high frequency trading software with the purpose of manipulating the Company's systems". Clause 16.5 adds that in such cases the Company has "no obligation to pay the profit of the Client".\[27\] Other editions number this clause differently (15.4.B.xiii is cited secondhand). The terms do not ban robots as such, but a profitable bot is exactly what invites an "abuse" determination,\[28\] and broker reviews describe the platform as not supporting algorithmic trading.\[29\] Architect for alerting, and assume any automated real-money account can be closed with its profits forfeited.

## Regulatory Status (affects whether to build at all)
- **EU.** ESMA agreed a retail binary-option prohibition on 27 March 2018, applied it from 2 July 2018 and renewed it every three months. It let the prohibition lapse on 1 July 2019 because most national regulators had adopted permanent measures at least as strict.\[24\]\[30\]\[31\]
- **UK.** FCA PS19/11 permanently bans selling, marketing or distributing binary options to retail consumers from 2 April 2019, including securitised binaries. The FCA said: "Binary options are gambling products dressed up as financial instruments."\[32\]\[33\]
- **US.** The joint CFTC/SEC investor alert warns that binary options offered off designated contract markets or without registration may be illegal: "All other entities offering binary options that are commodity options transactions are doing so illegally."\[25\]
- **Thailand.**
  - The Thai SEC (ก.ล.ต.) said in an April 2017 article that binary options "are not under SEC supervision".\[34\] They are an unregulated, unlicensed product rather than one explicitly banned for individuals.\[35\]
  - SEC press release No. 133/2561 (31 October 2018) named iqoption.com as a foreign entity with no Thai address that was soliciting Thai investors (for CFDs, with Thai-translated content) without a licence, and warned that investors may lack legal protection.\[36\]
  - The Derivatives Act B.E. 2546 penalises unlicensed derivatives business (s.16/s.125: up to 3 years' imprisonment and/or a THB 300,000 fine, plus up to THB 10,000 per day).\[37\]\[38\] This targets operators, not traders.
  - Bank of Thailand circular 2674/2567 (6 December 2024) says running or supporting FOREX businesses, including payment services for FOREX transactions, is an offence under exchange-control law, and tells payment providers not to process such payments.\[39\] Expect funding and withdrawal friction.
  - I found no Thai primary source classifying binary options as gambling. Claims to that effect come from unsourced affiliate sites.
- **Elsewhere.** In 2021 Indonesia's Bappebti, working with the Ministry of Communication and Informatics, blocked 1,222 domains. They included 92 binary-option domains such as Binomo, IQ Option, Olymptrade and Quotex. In a 2 February 2022 release, acting head Indrasari Wisnu Wardhana called binary option trading "a gambling scheme disguised as investment" (reported by The Jakarta Post).

## Recommended End-to-End Pipeline
1. **Ingest.** Record platform settlement quotes and payouts (tick or 1s) for EUR/USD, GBP/USD and USD/JPY. Pull Dukascopy interbank ticks for the same period, and an economic calendar with point-in-time timestamps.
2. **Measure lag.** Log signal → strike latency for at least 2 weeks, using demo trades, and fit the δ distribution.
3. **Label.** Separate y_call and y_put from the strike at t + δ, with ties as losses. Store the label end time t1 and the payout r_t.
4. **Features.** Section 2 set, normalised by time of day, with look-ahead unit tests.
5. **Primary model.** LightGBM/CatBoost per horizon (M1, M5, M15) with cost-weighted logloss, tuned by Optuna inside purged inner folds.
6. **Meta-model and calibration.** Isotonic calibration on purged out-of-sample folds. Decision rule p_LCB > p*(r_t) + buffer, plus news, rollover and spread filters.
7. **Evaluation.** CPCV path distribution, deflated EV and DSR, PBO, permutation test against shuffled labels, a lag-stress test (δ × 2), and a payout-stress test (r − 5 percentage points).
8. **Sizing.** Fixed 0.5–1%. Move to ¼-Kelly on p_LCB only after the live gate passes.
9. **Deployment.** Alert-only (Telegram) → demo forward test → minimal live stake, with drift monitoring (rolling win rate vs p*, calibration error, latency) and automatic kill switches.

## Go/No-Go Validation Checklist (all must pass)
- [ ] Labels use the platform settlement feed, the strike at t + δ (empirical lag) and ties as losses. The OTC series is excluded.
- [ ] Purged walk-forward and CPCV out-of-sample win rate on traded signals beats p*(r) at a one-sided 95% lower bound, using effective (autocorrelation-adjusted) N.
- [ ] DSR ≥ 0.95, with N equal to all logged trials. PBO ≤ 0.2.
- [ ] Label-permutation test p < 0.01. Edge survives doubled lag and a payout 5 percentage points lower.
- [ ] Edge holds in at least 2 of 3 pairs and at least 3 of 4 calendar quarters, not in a single session or month.
- [ ] Calibration: reliability-curve error < 2 percentage points in the traded probability bins.
- [ ] Forward demo test of at least 500–1,000 trades (non-overlapping) meets the same bound. Live micro-stake results agree with demo within sampling error.
- [ ] Legal: the product is lawful for you to use in your jurisdiction, and you accept that funding, withdrawal and terms-of-service risks may override any statistical edge.
- [ ] Sizing ≤ 1% fixed or ¼-Kelly on p_LCB. No progression systems. Daily loss limit and kill switch are tested.

If any box fails, the answer is no-go. If they all pass, ask why the edge should not be traded on a regulated spot FX or futures venue, where breakeven is about 50% plus spread and the venue is not your counterparty.

## Caveats
- Most published intraday ML-forecasting papers use daily data, equities, or protocols that cannot be audited. Direct, rigorous evidence on M1/M5 FX direction under binary-option settlement is scarce, so the 50–52% realistic range is an inference from adjacent studies and market-efficiency stylised facts, not a single definitive measurement.
- Several studies cited here (the 2026 arXiv MNQ and SPY papers) are preprints. The tick-volume correlation figure comes from industry sources.\[5\]
- The ruin and sample-size figures are analytical approximations assuming independent trades. Simulate with your own streak structure.
- Details of IQ Option's terms, tie handling and expiry mechanics differ across entities and change over time. Read the version tied to your account.
- Nothing here is legal advice. Thai enforcement mainly targets operators and payment channels, and the legal position of individual traders is a grey area.

## Sources

1. [GitHub - iqoptionapi/iqoptionapi: IQ Option API (Python 3.7.2) - supported by community - ONLY FOR STUDY - Don't use it on your real Account!!! · GitHub](https://github.com/iqoptionapi/iqoptionapi)
2. [IQ Option API with Python: iqoptionapi Guide](https://botbinaryoptions.com/en/iq-option-api-python.html)
3. [Predicting Foreign Exchange EUR/USD Direction Using Machine Learning](https://dl.acm.org/doi/10.1145/3696271.3696272)
4. [EconPapers: The Price Impact of Order Book Events](https://econpapers.repec.org/RePEc:oup:jfinec:v:12:y:2014:i:1:p:47-88.)
5. [Trading Volume Requires the Right Broker - VARIANSE](https://varianse.com/en/forex-education/broker-knowledge/trading-volume-requires-the-right-broker/)
6. [What is Forex Trade 'Volume'?](https://www.fpmarkets.com/education/forex-trading/forex-trading-volume/)
7. [7 Volume Analysis Strategies That Actually Work](https://forextester.com/blog/volume-analysis-trading/)
8. [Tick Volume in Forex](https://www.earnforex.com/guides/tick-volume-in-forex/)
9. [Research Note RN/11/01 Characterization of Financial Time Series](http://www.cs.ucl.ac.uk/fileadmin/UCL-CS/research/Research_Notes/RN_11_01.pdf)
10. [martinsewell.com: Finance: Stylized Facts: Dependence](https://finance.martinsewell.com/stylized-facts/dependence/)
11. [ORDERFLOW IMBALANCE AND HIGH FREQUENCY ...](https://tesi.luiss.it/27169/1/701851_PECCHIARI_MATTEO.pdf)
12. [The Bounce Has No Direction Sign, Magnitude, and the Microstructureof Equity Return Predictability Fourier-Residue Identities, Fejér Sums, and Evidencefrom US Equity and Cross-Asset Markets, 1993–2026](https://arxiv.org/html/2606.29591)
13. [Deutsche Mark–Dollar Volatility: Intraday Activity Patterns, ...](https://public.econ.duke.edu/~boller/Published_Papers/jf_98.pdf)
14. [Intraday periodicity and volatility persistence in financial ...](http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/c9d1a098610563d0c12581f60072a23a/$FILE/AndersenBollerslev1997b.pdf)
15. [Market intraday momentum - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351)
16. [Intraday time series momentum: global evidence and links to market](https://centaur.reading.ac.uk/95566/1/Accepted-Version.pdf)
17. [Intraday time series momentum: Global evidence and links to market characteristics - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S138641812100001X)
18. [Léo GRINSZTAJN on alphaXiv](https://www.alphaxiv.org/@leo-grinsztajn)
19. [Intraday Market Return Predictability Culled from the Factor Zoo](https://public.econ.duke.edu/~boller/Papers/HFML.pdf)
20. [The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality by David H. Bailey, Marcos Lopez de Prado :: SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
21. [Probability of Backtest Overfitting (PBO): Bailey et al.](https://aifinhub.io/learn/bailey-lopez-pbo/)
22. [OTC Trading on IQ Option: How to Trade on Weekends - IQ Option Official Blog](https://blog.iqoption.com/en/otc-trading-on-iq-option-how-to-trade-securities-over-the-counter/)
23. [IQ OTC Trading Bot - Your Weekend Playbook](https://www.earnforex.com/forum/threads/iq-otc-trading-bot-your-weekend-playbook.54849/)
24. [ESMA agrees to prohibit binary options and restrict CFDs to protect retail investors](https://www.esma.europa.eu/press-news/esma-news/esma-agrees-prohibit-binary-options-and-restrict-cfds-protect-retail-investors)
25. [CFTC/SEC Investor Alert: Binary Options and Fraud](https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/fraudadv_binaryoptions.html)
26. [iqoptionapi/setup.py at master · Lu-Yi-Hsun/iqoptionapi · GitHub](https://github.com/Lu-Yi-Hsun/iqoptionapi/blob/master/setup.py)
27. <https://static.cdnroute.io/files/storage/public/cc/i5/g32cpif6r6n5vlrg.pdf>
28. [IQ Option Account Blocked: Reasons and What to Do (2026)](https://iqoptionwiki.com/iq-option-account-blocked/)
29. [IQ Option Review](https://fxscouts.com/broker/iqoption/)
30. [ESMA adopts final product intervention measures on CFDs and binary options](https://www.esma.europa.eu/press-news/esma-news/esma-adopts-final-product-intervention-measures-cfds-and-binary-options)
31. [ESMA ceases renewal of product intervention measure relating to binary options](https://www.esma.europa.eu/node/88538)
32. [PS19/11: Product intervention measures for retail binary options](https://www.fca.org.uk/publications/policy-statements/ps19-11-product-intervention-measures-retail-binary-options)
33. [FCA confirms permanent ban on the sale of binary options to retail consumers](https://www.fca.org.uk/news/statements/fca-confirms-permanent-ban-sale-binary-options-retail-consumers)
34. <https://www.sec.or.th/TH/Template3/Articles/2560/ac-post-25600408-binary-obtions.pdf>
35. [Binary Options Regulations by Country - Legal or Restricted?](https://www.binaryoptions.net/regulation)
36. [Thai ข่าว ก.ล.ต.](https://www.sec.or.th/TH/Pages/News_Detail.aspx?SECID=7206)
37. [ตะกร้าข่าว - ก.ล.ต.กล่าวโทษผู้กระทำผิด 5 ราย ต่อ ปอศ. กรณีประกอบธุรกิจสัญญาซื้อขายล่วงหน้าไม่ได้รับใบอนุญาต](https://www.isranews.org/content-page/item/69434-sec_69434.html)
38. [ก.ล.ต.กล่าวโทษผู้กระทำผิดรวม5ราย ต่อ ปอศ.กรณีประกอบธุรกิจสัญญาซื้อขายล่วงหน้าโดยไม่ได้รับใบอนุญาต](https://www.prachachat.net/finance/news-218595)
39. <https://www.bot.or.th/content/dam/bot/fipcs/documents/FOG/2567/ThaiPDF/25670207.pdf>
