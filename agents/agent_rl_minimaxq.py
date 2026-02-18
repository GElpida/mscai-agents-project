# agents/agent_rl_minimaxq.py
from __future__ import annotations
import numpy as np


def _solve_maximin_2x2(M: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, float]:
    """
    Solve: max_{pi in Δ2} min_{b in {0,1}} sum_a pi[a] * M[a,b]
    M is 2x2 payoff matrix for this agent.
    Returns (pi, v).
    """
    m00, m01 = float(M[0, 0]), float(M[0, 1])
    m10, m11 = float(M[1, 0]), float(M[1, 1])

    denom = (m00 - m10) - (m01 - m11)

    candidates = [0.0, 1.0]
    if abs(denom) >= eps:
        p_star = (m11 - m10) / denom
        candidates.append(float(np.clip(p_star, 0.0, 1.0)))

    best_p, best_v = 0.0, -1e18
    for p in candidates:
        f0 = p * m00 + (1.0 - p) * m10  # vs column 0
        f1 = p * m01 + (1.0 - p) * m11  # vs column 1
        v = min(f0, f1)
        if v > best_v:
            best_v, best_p = v, p

    pi = np.array([best_p, 1.0 - best_p], dtype=float)
    return pi, float(best_v)


class MinimaxQLearner:
    """
    Minimax-Q learner for zero-sum stochastic games.

    Learns Q[s, a, b] for *this agent's* payoff.
    Works out-of-the-box for 2 actions (2x2 minimax solved analytically).

    In self-play:
      - Player 1 updates with r
      - Player 2 updates with -r and swaps (a,b) in update call
        (because for P2, 'my action' is b, opponent is a).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int = 2,
        alpha: float = 0.2,
        gamma: float = 0.95,
        eps: float = 0.1,
        seed: int | None = None,
    ):
        if n_actions != 2:
            raise NotImplementedError("This version supports n_actions=2 only.")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((self.n_states, 2, 2), dtype=float)

    def reset(self) -> None:
        pass

    def policy(self, s: int) -> np.ndarray:
        """Minimax mixed strategy pi(a|s)."""
        pi, _ = _solve_maximin_2x2(self.Q[int(s)])
        return pi

    def value(self, s: int) -> float:
        """V(s) = max_pi min_b E[Q]."""
        _, v = _solve_maximin_2x2(self.Q[int(s)])
        return v

    def act(self, s: int) -> int:
        """ε-greedy over minimax policy."""
        if self.rng.random() < self.eps:
            return int(self.rng.integers(2))
        pi = self.policy(s)
        return int(self.rng.choice([0, 1], p=pi))

    def update(self, s: int, a: int, b: int, r: float, s_next: int) -> None:
        s = int(s); a = int(a); b = int(b); s_next = int(s_next)
        target = float(r) + self.gamma * self.value(s_next)
        self.Q[s, a, b] = (1.0 - self.alpha) * self.Q[s, a, b] + self.alpha * target
