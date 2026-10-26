# Audit: GBPUSD k=5 (closest miss) (2026-10-26)

**Verdict: no credible edge. It is not an artifact like the k=15 case, but it is fragile, short-lived and has stopped producing trades.** Consistent with the gate's NO-GO (DSR 0.908, PBO 0.212).

Setup: the same out-of-sample predictions as the gate (GBPUSD, k=5, r=0.85, slip 0.00002, buffer 0.04, non-overlapping trades, rollover and Sunday bars skipped), sliced further. Dukascopy M1 bid bars, 2016–2026. Overall: 1,723 trades, win 0.5844, Wilson lower bound 0.5648 against breakeven 0.5405.

| Test | Result | Reading |
|---|---|---|
| Placebo: same signals against unrelated labels | win 0.453–0.480 (base rate 0.458) | No label or signal leak. |
| Hour of day (NY) | Spread across hours 0–15 and 19–23; no rollover concentration | Not the k=15 artifact. NY hours 8–9 are the strongest (n=340, win about 0.65); without them win 0.5676, lower bound 0.5456, barely above breakeven. |
| Entry gap | Entry at close[t] gives the same win rate (0.5844); mean gap 0.2 pip | The edge is not a bar-gap or bid-ask-bounce artifact. |
| **Buffer sensitivity** | Lower bound is above breakeven only at buffer 0.04: 0.00 → 0.5294, 0.02 → 0.5392, **0.04 → 0.5648**, 0.06 → 0.5365, 0.08 → 0.4785 | The passing point was picked by total PnL on the same data: a best-of-grid effect. Only 1 of 5 buffers clears breakeven. |
| Confidence within selection | Five confidence bands win 0.583, 0.596, 0.559, 0.611, 0.574 | A higher predicted probability does not mean more wins. The ranking carries no information past the threshold. |
| **Over time** | 2016–2021: 74 / 161 / 231 / 269 / 709 / 193 trades, win 0.52–0.76. 2022–2025: 63 / 14 / 6 / 3 trades, win about 0.50 combined | The signal has largely stopped. Almost no trades since 2022, and the few that exist do not win. |
| **Delayed entry** (same signals) | 1 bar late: win 0.5572 (lower bound 0.5374, below breakeven). 2 bars: 0.5360. 5 bars: 0.5000 | The edge decays within one to two minutes. A rough estimate of about 0.5 point of win rate per 10 seconds is an interpolation, not a measurement: the data has no sub-minute entries. |
| By side | Call win 0.5626, put win 0.6139 | No pattern that would make this credible. |

## Reading
- **Why it is not a GO:** passing needed a lucky buffer choice (a best-of-grid pick among 15 configurations and 4 buffers each), the edge is gone since 2021, and one minute of delay erases it.
- **Realistic execution:** alert-and-click entry takes seconds to tens of seconds. This data cannot measure that, but the decay rate suggests most of a thin 2.4-point margin would go.
- **What would change the verdict:** a signal that stays positive across buffers, still trades after 2022, and survives a measured sub-minute delay. None of the 15 configurations show that.
