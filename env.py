"""
HealthEnv - local Python wrapper around the FastAPI server.
Used by inference.py so it can run without a live HTTP server.

Reward values are STRICTLY within (0, 1) — never 0.0 or 1.0:
  correct action  → 0.95
  off-by-one      → 0.50
  wrong action    → 0.05
"""

import random
import uuid
from typing import Any, Dict, Tuple


class HealthEnv:
    """Lightweight in-process health monitoring environment."""

    def __init__(self):
        self._episode_id: str = ""
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._heart_rates: list = []
        self._state: Dict[str, Any] = {}

    def _random_state(self) -> Dict[str, Any]:
        return {
            "heart_rate": random.randint(55, 160),
            "temperature": round(random.uniform(36.0, 40.5), 1),
        }

    def _compute_reward(self, heart_rate: int, temperature: float, action: int) -> float:
        """
        Returns a reward STRICTLY in (0, 1) — never 0.0 or 1.0.
        correct → 0.95 | off-by-one → 0.50 | wrong → 0.05
        """
        if heart_rate > 120 or temperature > 39.0:
            correct = 2
        elif heart_rate > 100 or temperature > 38.0:
            correct = 1
        else:
            correct = 0

        if action == correct:
            return 0.95        # perfect — but strictly < 1
        elif abs(action - correct) == 1:
            return 0.50        # partial credit
        else:
            return 0.05        # wrong — but strictly > 0

    # ── OpenEnv-style API ──────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._state = self._random_state()
        self._heart_rates = [self._state["heart_rate"]]
        return dict(self._state)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        if self._done:
            return dict(self._state), 0.05, True, {"message": "Episode done"}

        reward = self._compute_reward(
            self._state["heart_rate"], self._state["temperature"], action
        )
        self._total_reward += reward
        self._step_count += 1
        self._state = self._random_state()
        self._heart_rates.append(self._state["heart_rate"])
        self._done = self._step_count >= 20

        return dict(self._state), round(reward, 4), self._done, {"step": self._step_count}

    def state(self) -> Dict[str, Any]:
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "total_reward": round(self._total_reward, 4),
            "done": self._done,
        }
