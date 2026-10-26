import numpy as np
import pandas as pd
import pytest
from opx.rl.env import MultiPairBinaryOptionsEnv
from opx.rl.train import evaluate_agent_oos, train_ppo_agent


def test_ppo_training_and_evaluation():
    n_steps = 200
    np.random.seed(42)
    features = np.random.randn(n_steps, 15).astype(np.float32)
    # EURUSD call always wins, put always loses
    outcomes = {
        "eurusd_call_win": np.ones(n_steps, dtype=np.float32),
        "eurusd_put_win": np.zeros(n_steps, dtype=np.float32),
    }
    outcomes_df = pd.DataFrame(outcomes)

    train_env = MultiPairBinaryOptionsEnv(
        states=features[:150],
        outcomes=outcomes_df.iloc[:150],
        pairs=["eurusd"],
        k=5,
        r=0.85,
    )
    test_env = MultiPairBinaryOptionsEnv(
        states=features[150:],
        outcomes=outcomes_df.iloc[150:],
        pairs=["eurusd"],
        k=5,
        r=0.85,
    )

    agent = train_ppo_agent(train_env, total_timesteps=1000, lr=1e-3)
    metrics = evaluate_agent_oos(agent, test_env)

    assert "total_trades" in metrics
    assert "win_rate" in metrics
    assert "cumulative_pnl" in metrics
    assert metrics["total_steps"] > 0
