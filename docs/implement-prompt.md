You are implementing the OPX project: a Python research harness that trains cost-weighted LightGBM models on M1 FX bars (Call/Put at k = 1, 5, 15 minutes) and emits a code-enforced go/no-go report. It is NOT a trading bot.

## Read first, in this order
1. `docs/summary.md` — what the research concluded and which conflicts were resolved.
2. `docs/architecture.md` — the spec: modules, data contracts, design decisions.
3. `docs/superpowers/plans/2026-10-26-opx-pipeline.md` — the task list you will execute.

The three files in `docs/research/` are background only. Do not re-derive decisions already made in the summary.

## How to work
- Execute the plan's tasks in order, Task 1 to Task 12. Each task is test-first: write the failing test, run it and see it fail, implement, run it and see it pass, commit. Do not skip the "see it fail" step.
- Use the exact file paths, function names, signatures, column names and numeric values in the plan. If two tasks disagree on a name, stop and ask rather than guessing.
- Work in the project root `C:\Users\swing\Desktop\TRADING\OPX`. Shell is PowerShell on Windows 11. Python 3.11 and git are installed. Use a `.venv` and run tests with `.venv\Scripts\pytest`.
- One commit per task, with the message given in the task.
- After Task 12, run the full suite. Do NOT start Task 13 until the user has put real M1 data in `data/`.

## Hard constraints (do not break, even if it looks simpler)
- Runtime deps only: pandas, numpy, pyarrow, lightgbm, scikit-learn, scipy, httpx. Dev: pytest. No TA-Lib, Optuna, CatBoost, XGBoost or polars.
- Ties are losses: strict `>` in labels. Label entry is `open[t+1]`, expiry is `close[t+k]`, `t1 = index + (k+1) minutes`.
- Loss is logloss with weight `r` on wins and 1 on losses. Win probability is `q / (q + r(1-q))` with `q = sigmoid(raw)`.
- Features may use only bars up to and including t. Keep the no-look-ahead test passing.
- Threshold is `breakeven(r) + buffer`, never a hard-coded 0.60.
- No Martingale, no OTC data, no broker API or automation, no code that bypasses bot protection. `live.py` only decides and sends Telegram alerts.
- Do not add features, options or modules that are not in the plan. The "Deferred" section of the plan is out of scope.

## Quality bar
- All tests pass before each commit. Never weaken a test to make it pass; if a plan test value looks wrong (for example a statistical test that is seed-sensitive), fix the seed or the number, explain why in the commit message, and tell the user.
- The planted-signal and pure-noise tests in Task 10 are the integrity check. If either fails, fix the pipeline before continuing.
- Handle bad inputs as the plan's Review Focus lists them: gaps, flat bars, empty selections, bad files, DST.

## Report back
When Task 12 is done, or if you get blocked, tell the user:
- which tasks are committed and the final `pytest -q` result,
- any plan value or interface you had to change, and why,
- anything you could not verify.
