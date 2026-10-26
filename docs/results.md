# OPX Results: Go/No-Go on Real Data (2026-10-26)

**Verdict: NO-GO. 0 of 15 configurations pass the gate.** Do not trade fixed-time contracts on these pairs on the strength of this pipeline.

## Setup
- **Data:** Dukascopy M1 **bid** bars, 5 pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD), 2014-12-31 to 2026-10-23, about 4.4 million bars per pair. Not the broker's settlement quotes.
- **Pipeline:** `opx` at commit `eef438a` or later. Expanding purged walk-forward, 6 folds. LightGBM with the payout-weighted loss, Platt calibration on the chronological tail of each training window.
- **Labels:** entry `open[t+1]`, expiry `close[t+k]`, ties lose, slippage 0.2 pip (0.002 on JPY pairs), lag stress at 0.4 pip.
- **Payout:** r = 0.85, breakeven 54.05%. Buffer grid 0.00, 0.02, 0.04, 0.06 above breakeven.
- **Signal bars skipped:** New York 16:00–18:59 (daily rollover) and New York Sundays, in both training and testing.
- **Trials counted for DSR:** 4 buffers × 15 configurations = 60.
- **Gate:** the 8 checks in `docs/superpowers/plans/2026-10-26-opx-pipeline.md`; any failure is a NO-GO. Raw numbers: `results/gate_results.json`.

## History: why the first run was wrong
The first batch, without the rollover filter, reported GO for EURUSD k=15 and GBPUSD k=15. The audit in [audit-eurusd-k15.md](audit-eurusd-k15.md) showed the whole edge came from trades in the New York 17:00 hour (win rate 73.5%). Without them the win rate was 53.75%, below breakeven. Those results are archived as `results/gate_results_contaminated.json` and must not be used. After the filter, both k=15 results fail.

## Results (15 configurations, all NO-GO)

`lcb` is the one-sided 95% Wilson lower bound of the win rate on non-overlapping trades (needs > 0.5405). Failed checks: `lcb`, `dsr` (≥ 0.95), `pbo` (≤ 0.2), `perm` (p < 0.01), `pay` (still above breakeven at payout 0.80), `lag` (still above at double slippage), `cal` (calibration error < 0.02), `qtr` (≥ 75% of quarters positive).

| Pair | k | Buffer | lcb | DSR | PBO | Failed checks |
|---|---|---|---|---|---|---|
| EURUSD | 1 | 0.04 | 0.5509 | 0.663 | 0.211 | dsr, pbo, pay, lag, cal, qtr |
| EURUSD | 5 | 0.02 | 0.5475 | 0.710 | 0.118 | dsr, pay, lag, qtr |
| EURUSD | 15 | 0.06 | 0.5366 | 0.189 | 0.673 | lcb, dsr, pbo, pay, lag, cal, qtr |
| GBPUSD | 1 | 0.02 | 0.5471 | 0.646 | 0.062 | dsr, pay, cal, qtr |
| **GBPUSD** | **5** | 0.04 | **0.5648** | 0.908 | 0.212 | **dsr, pbo** (closest miss) |
| GBPUSD | 15 | 0.02 | 0.5278 | 0.015 | 0.960 | lcb, dsr, pbo, pay, lag, cal, qtr |
| USDJPY | 1 | 0.02 | 0.5475 | 0.615 | 0.023 | dsr, pay, lag, qtr |
| USDJPY | 5 | 0.04 | 0.5557 | 0.746 | 0.209 | dsr, pbo, lag, qtr |
| USDJPY | 15 | 0.02 | 0.5309 | 0.026 | 0.797 | lcb, dsr, pbo, pay, lag, cal, qtr |
| AUDUSD | 1 | 0.06 | 0.5441 | 0.309 | 0.061 | dsr, pay, lag, cal, qtr |
| AUDUSD | 5 | 0.02 | 0.5485 | 0.624 | 0.270 | dsr, pbo, pay, lag, qtr |
| AUDUSD | 15 | 0.06 | 0.4795 | 0.191 | 0.625 | all 8 |
| USDCAD | 1 | 0.02 | 0.5247 | 0.036 | 0.499 | lcb, dsr, pbo, pay, lag, cal, qtr |
| USDCAD | 5 | 0.02 | 0.5085 | 0.004 | 0.899 | lcb, dsr, pbo, pay, lag, cal, qtr |
| USDCAD | 15 | 0.02 | 0.4963 | 0.022 | 0.250 | lcb, dsr, pbo, perm, pay, lag, qtr |

## Reading the results
- **k = 15 has no edge** on any pair once rollover is excluded: all five k=15 configurations fail the lower-bound check.
- **k = 1 and k = 5 show a raw win rate near 55%** on EURUSD, GBPUSD, USDJPY and AUDUSD (lower bound 0.54–0.56). That is thin: 13 of the 15 configurations fail the payout-stress check (breakeven at an 80% payout is 55.56%), and all 15 fail the deflated Sharpe ratio after 60 trials.
- **Closest miss:** GBPUSD k=5 passes everything except DSR (0.908 against 0.95) and PBO (0.212 against 0.20), and it survives both stress tests. USDJPY k=5 is next but fails lag stress. The plan's cross-pair rule (an edge in at least 2 of 3 core pairs) is not met at any horizon.
- **The k=1/5 edge is not audited.** The k=15 edge turned out to be an artifact, and the same kind of check has not been run on k=1 and k=5. Bid-only quotes, bid-ask bounce and the `open[t+1]` entry assumption could all produce a short-horizon pattern that a real settlement feed would not pay.

## Caveats
- Dukascopy interbank bid bars are not the broker's settlement quotes. A pass here would still be necessary, not sufficient.
- The buffer is chosen by total PnL on the same out-of-sample data. DSR and PBO correct for this, but they are the checks that fail.
- Manual checks remain undone: lawful in your jurisdiction, a forward demo of 500–1,000 non-overlapping trades, and measuring the feed difference.

## What this means
No configuration reaches GO, so no capital should be committed. GBPUSD k=5, the closest miss, was audited in [audit-gbpusd-k5.md](audit-gbpusd-k5.md): it is not a rollover artifact, but it passes only at one of five buffers, has produced almost no trades since 2022, and loses its edge with a one-minute delay. The other k=1 and k=5 configurations remain unaudited, and they sit further from passing.
