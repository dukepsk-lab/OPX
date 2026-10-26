# Research Summary: OPX (ML for M1/M5/M15 FX Fixed-Time Trades)

Sources (all in `docs/research/`):
- **R1** `ML Pipelines for M1, M5 and M15 FX Binary Options_ Technical and Evidence-Based Assessment.md` (evidence-based, cited, sceptical)
- **R2** `ML Pipeline for Binary Options.docx` (long technical blueprint with formulas and code)
- **R3** `เอกสารสรุปข้อมูลเชิงลึก...docx` (Thai digest of 4 web sources: a quant-research GitHub repo, a commercial binary-options API vendor, a HankoX vs TakeProfit comparison, and custom-loss articles for GBDTs)

## Verdict

All three agree on the method. They differ on how much to believe the results.

- Breakeven win rate is `p* = 1/(1+r)`: 55.6% at an 80% payout, 54.1% at 85%, 52.6% at 90%. Ties count as losses.
- R1 cites rigorous out-of-sample accuracy of about 50–52% at these horizons. R2 claims +1–7% lifts from a meta-model filter, with Sharpe 1.4, but cites no auditable study. Treat R2's numbers as hypotheses.
- R1's default answer is **no-go for real capital** unless a strict validation gate is passed on settlement-consistent data.
- So OPX is a **research and validation harness that ends in a go/no-go report**. It does not auto-trade.

## What all sources agree on

| Topic | Decision |
|---|---|
| Label | Entry at the next bar's open plus slippage, expiry at the close of bar t+k, strict `>` so ties lose. Separate Call and Put targets. |
| Loss | Cost-weighted logloss: weight `r` on wins, 1 on losses. Gradient `p·w − r·y`, Hessian `p(1−p)·w`. The decision boundary moves to `p*`. Recover the true probability with `p = q/(q + r(1−q))`. |
| Model | GBDT (LightGBM) on engineered features. Sequence models are not worth it on noisy tabular data. |
| Filter | Calibrate probabilities (Platt), then trade only above a threshold. A meta-label model is a second stage. |
| Validation | Purged walk-forward, so no training label overlaps the test window. Add the Deflated Sharpe Ratio, Probability of Backtest Overfitting, and a permutation test. |
| Sizing | No Martingale. It leaves EV per dollar unchanged and raises ruin risk. Fixed 0.5–1%, or ¼-Kelly on a lower bound of p. |
| OTC weekend pairs | Broker-generated prices with a counterparty conflict. Never train or trade on them. |
| Features | Candle geometry, ATR-normalised returns, RSI/Bollinger, tick-volume z-score, cyclical time and session flags. Later: Hawkes order-flow imbalance, fractional differentiation, squeeze states. |

## Where sources conflict, and what OPX does

| Conflict | Resolution |
|---|---|
| R2 uses a fixed `P(win) > 0.60` gate. R1 says the threshold should be `p* + buffer`, tuned out of sample. | The threshold is `breakeven(r) + buffer`. The buffer is a parameter, and 0.60 is only a baseline to compare against. Each threshold tried counts as a trial for the Deflated Sharpe Ratio. |
| R2 allows OTC at M5/M15 with a sequential test. R1 excludes OTC. | OTC is excluded from v1. |
| R2 uses ¼-Kelly capped at 3%. R1 uses fixed 0.5–1% until there are at least 1,000 live trades. | Fixed 1% by default. ¼-Kelly on the lower bound of p comes only after the gate passes. |
| R2 says Platt scaling is preferred below 5,000 calibration samples. R1 says isotonic. | Platt by default, since it has two parameters and overfits less. |
| R2 says Kelly sizing gives "zero probability of ruin". | This is wrong. No sizing rule guarantees that, and R1 shows ¼-Kelly still depends on how well p is estimated. |
| R2 assumes 250 ms median latency. R1 says an M1 reversal edge often vanishes at 1–3 s. | Lag is a stress parameter: run with slippage and with double slippage. |
| R2's plan trains many tuned models. R1 counts every trial. | Log every configuration tried and feed the count into the Deflated Sharpe Ratio. |

## What R3 adds

Most of R3 is not about this problem.

- **Useful:**
  - The `qbacktest` design (T+1 fill invariants, accounting invariants, bootstrap Sharpe confidence intervals) is a good pattern for OPX's own tests.
  - The LightGBM objective takes `(grad, hess)`. CatBoost wants the **negative** gradient and Hessian. This matters only if CatBoost is added.
  - The drill-wear experiment shows that a cost-weighted loss cut critical errors from 1,150 to 724 but dropped overall accuracy from 64% to 56%. That supports optimising EV rather than accuracy.
  - TakeProfit can send alerts to Telegram or Discord in under 100 ms. It is optional tooling.
- **Do not use:** the CB Traders BD API. It is a paid source-code product that bypasses Cloudflare Turnstile, supports Martingale and claims a 90%+ win rate it doesn't evidence. Bypassing bot protection and the unverifiable claim both rule it out.
- **Ignore:** the HankoX vs TakeProfit comparison, the C++ HFT engines, and the XGBoost drill-bit features. None of it bears on OPX.

## Risks that outrank modelling

- Binary options are banned for retail clients in the EU and UK. In Thailand they are unlicensed, and the Thai SEC has warned about iqoption.com by name.
- The Bank of Thailand has told payment providers not to process forex payments, so expect funding and withdrawal friction.
- IQ Option's terms allow account closure and withheld profits over "high-frequency" software. The community Python API is labelled "for study only".
- A pipeline that passes every check could still be unusable for legal or terms-of-service reasons. That is why execution is alert-only.
