import numpy as np
import pandas as pd
import pytest
from opx.rl.env import MultiPairBinaryOptionsEnv


def test_env_lifecycle_and_rewards():
    n_steps = 100
    features = np.random.randn(n_steps, 20).astype(np.float32)
    # Binary outcomes: 1 win, 0 loss
    outcomes = {
        "eurusd_call_win": np.ones(n_steps, dtype=np.float32),  # always wins
        "eurusd_put_win": np.zeros(n_steps, dtype=np.float32),   # always loses
        "gbpusd_call_win": np.ones(n_steps, dtype=np.float32),
        "gbpusd_put_win": np.zeros(n_steps, dtype=np.float32),
    }
    outcomes_df = pd.DataFrame(outcomes)

    env = MultiPairBinaryOptionsEnv(
        states=features,
        outcomes=outcomes_df,
        pairs=["eurusd", "gbpusd"],
        k=5,
        r=0.85,
    )

    obs, info = env.reset()
    assert obs.shape == (20,)

    # 1. Action 0: Hold -> reward 0.0
    next_obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0

    # 2. Action 1: Call EURUSD -> wins -> reward +0.85
    next_obs, reward, terminated, truncated, info = env.step(1)
    assert abs(reward - 0.85) < 1e-4

    # 3. Action 2: Put EURUSD while locked in cooldown -> penalty or 0
    next_obs, reward, terminated, truncated, info = env.step(2)
    assert reward <= 0.0
