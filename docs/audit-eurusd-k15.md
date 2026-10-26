# Audit: EURUSD k=15 "GO" (2026-10-26)

**Verdict: the GO is a rollover artifact, not an edge.** All of the profit comes from trades in the New York 17:00 hour. With that hour removed, the win rate is below breakeven.

Setup: the same out-of-sample predictions the gate used (EURUSD, k=15, r=0.85, slip 0.00002, buffer 0.02, non-overlapping trades), recomputed and sliced. Dukascopy M1 bid bars, 2016–2026.

## What the gate said
8 of 8 checks passed: Wilson lower bound 0.5583 against breakeven 0.5405, DSR 0.9998, PBO 0.005, 14,605 trades, 86% of quarters positive.

## What the audit found

| Test | Result | Reading |
|---|---|---|
| Placebo: same signals against labels from unrelated times | win 0.472–0.476 (base rate 0.473) | No label or signal alignment leak. |
| Signals only in NY 17:00–17:59 (rollover) | n=2,145, win **0.735** | Almost all of the edge. |
| All hours except NY 17:00–17:59 | n=12,771, win **0.5375**, lower bound 0.5303 | **Below breakeven 0.5405.** |
| All hours except NY 16:00–18:59 | n=10,842, win 0.5369 | Same. |
| All hours except NY 16:00–19:59 | n=10,170, win 0.5396 | Same. |
| Share of trades in NY 16–18h | 14% in 2016, 86% in 2024 | The model increasingly learned the rollover pattern. |
| Delayed entry, same signals | 1 bar late: win 0.5455 (EV +0.009); 2 bars: 0.5403 (EV −0.0005); 5 bars: 0.5210 | Edge is gone with a 1–2 minute delay, even with rollover included. |
| By side | Call win 0.593, put win 0.542 | The call-side edge is the rollover artifact. Outside rollover: call 0.5406, put 0.5349. |
| Sunday trades (week-open hours) | 1,034 trades, win 0.608 | Same type of artifact. |

## Why this is an artifact
At the daily rollover the Dukascopy bid quote jumps as spreads widen and liquidity thins. A bid-only bar series makes that move look predictable. A broker's fixed-time contract is not guaranteed to be priced or settled on the same quote at that hour, and many brokers cut payouts or suspend assets then. R1 already says to stay flat at rollover and in illiquid hours; the pipeline did not enforce it.

## Consequence
- The 14-job batch was run without a rollover exclusion, so every job carries the same contamination. Its results cannot support a GO.
- Fix before any further run: drop signal bars in NY 16:00–18:59 (and the week-open hours) from both training and testing, then rerun.
- The delayed-entry result is a second warning: even a clean edge must be tested at realistic latency before any real money.
