"""Gymnasium environment for multi-pair binary options portfolio trading."""

from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from opx.econ import _check


class MultiPairBinaryOptionsEnv(gym.Env):
    """Multi-pair binary options portfolio execution environment.
    
    Action Space: Discrete(2 * n_pairs + 1)
        0: Hold (do nothing)
        1: Call Pair 0, 2: Put Pair 0
        3: Call Pair 1, 4: Put Pair 1
        ...
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        states: np.ndarray,
        outcomes: pd.DataFrame,
        pairs: list[str],
        k: int = 5,
        r: float = 0.85,
    ):
        super().__init__()
        _check(r)
        self.r = float(r)
        self.k = int(k)
        self.pairs = [p.lower() for p in pairs]
        self.n_pairs = len(self.pairs)

        self.states = np.asarray(states, dtype=np.float32)
        self.outcomes = outcomes
        self.n_steps = len(self.states)

        # Observation space: 1D feature vector
        n_features = self.states.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Action space: 1 (Hold) + 2 * n_pairs
        self.action_space = spaces.Discrete(2 * self.n_pairs + 1)

        # Cooldown locks per pair (expiry step index)
        self.locks: dict[str, int] = {p: -1 for p in self.pairs}
        self.current_step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.locks = {p: -1 for p in self.pairs}
        obs = self.states[self.current_step]
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_step >= self.n_steps - 1:
            return self.states[-1], 0.0, True, False, {}

        reward = 0.0
        info: dict[str, Any] = {"action": action, "win": None}

        if action > 0:
            pair_idx = (action - 1) // 2
            side = "call" if (action - 1) % 2 == 0 else "put"
            pair = self.pairs[pair_idx]

            # Check if this pair is locked in an active trade
            if self.locks[pair] > self.current_step:
                # Penalty for attempting to re-enter locked pair
                reward = -0.01
                info["status"] = "locked"
            else:
                col = f"{pair}_{side}_win"
                win = self.outcomes[col].iloc[self.current_step]
                if win == 1.0:
                    reward = self.r
                    info["win"] = True
                else:
                    reward = -1.0
                    info["win"] = False

                # Set lock until current_step + k
                self.locks[pair] = self.current_step + self.k
        else:
            info["status"] = "hold"

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        obs = self.states[self.current_step] if not terminated else self.states[-1]

        return obs, float(reward), terminated, truncated, info
