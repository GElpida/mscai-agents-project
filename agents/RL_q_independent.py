# agents/independent_q.py
from __future__ import annotations
import numpy as np


class IndependentQLearner:
    """
    Independent Q-learning baseline.

    Learns Q[s,a] from reward samples, treating the opponent as part of the environment.
    Useful for RL vs FP / FP vs RL comparisons (to show pros/cons).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.2,
        gamma: float = 0.95,
        eps: float = 0.1,
        seed: int | None = None,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((self.n_states, self.n_actions), dtype=float)

    def reset(self) -> None:
        pass

    def act(self, s: int) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[int(s)]))

    def update(self, s: int, a: int, r: float, s_next: int) -> None:
        s = int(s); a = int(a); s_next = int(s_next)
        target = float(r) + self.gamma * float(np.max(self.Q[s_next]))
        self.Q[s, a] = (1.0 - self.alpha) * self.Q[s, a] + self.alpha * target

    def greedy_action(self, s: int) -> int:
        return int(np.argmax(self.Q[int(s)]))
