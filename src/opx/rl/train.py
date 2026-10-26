"""PPO reinforcement learning training and evaluation on multi-pair binary options env."""

from typing import Any
import numpy as np
from stable_baselines3 import PPO
from opx.rl.env import MultiPairBinaryOptionsEnv
from opx.stats import wilson_lcb


def train_ppo_agent(
    env: MultiPairBinaryOptionsEnv,
    total_timesteps: int = 50000,
    lr: float = 3e-4,
    seed: int = 42,
    device: str = "auto",
) -> PPO:
    """Train a PPO policy on the binary options portfolio environment."""
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
        device=device,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_agent_oos(
    agent: PPO,
    env: MultiPairBinaryOptionsEnv,
) -> dict[str, Any]:
    """Evaluate trained agent policy on out-of-sample environment."""
    obs, info = env.reset()
    terminated = False
    truncated = False

    rewards: list[float] = []
    trade_outcomes: list[bool] = []
    action_counts: dict[int, int] = {}

    while not (terminated or truncated):
        action, _states = agent.predict(obs, deterministic=True)
        act = int(action)
        action_counts[act] = action_counts.get(act, 0) + 1

        obs, reward, terminated, truncated, step_info = env.step(act)
        rewards.append(reward)

        if step_info.get("win") is not None:
            trade_outcomes.append(step_info["win"])

    total_steps = len(rewards)
    total_trades = len(trade_outcomes)
    wins = int(sum(trade_outcomes))
    losses = total_trades - wins
    win_rate = float(wins / total_trades) if total_trades > 0 else 0.0
    lcb = float(wilson_lcb(wins, total_trades)) if total_trades > 0 else 0.0
    cumulative_pnl = float(sum(rewards))

    # Sharpe on trade returns
    trade_rewards = [r for r in rewards if r in (env.r, -1.0)]
    if len(trade_rewards) > 1 and np.std(trade_rewards) > 1e-12:
        sharpe = float(np.mean(trade_rewards) / np.std(trade_rewards, ddof=1))
    else:
        sharpe = 0.0

    return {
        "total_steps": total_steps,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "lcb": lcb,
        "cumulative_pnl": cumulative_pnl,
        "sharpe": sharpe,
        "action_distribution": action_counts,
    }
