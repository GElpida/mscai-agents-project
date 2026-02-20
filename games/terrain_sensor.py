# terrain_game.py
# Zero-sum stochastic terrain game
# Player A = sensor placement
# Player B = path choice

import numpy as np
import random
import math

# -------------------------
# Utility
# -------------------------

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def enumerate_paths(n):
    """All monotone (right/down) paths from (0,0) to (n-1,n-1)."""
    goal = (n - 1, n - 1)
    paths = []

    def rec(r, c, path):
        if (r, c) == goal:
            paths.append(path.copy())
            return
        if r < n - 1:
            path.append((r + 1, c))
            rec(r + 1, c, path)
            path.pop()
        if c < n - 1:
            path.append((r, c + 1))
            rec(r, c + 1, path)
            path.pop()

    rec(0, 0, [(0, 0)])
    return paths


# -------------------------
# Game
# -------------------------

class TerrainGame:
    """
    One-shot zero-sum game.

    A chooses sensor location.
    B chooses a path.
    Detection is stochastic.
    """

    def __init__(self, heights, fog=0.25, k_diff=0.9, k_height=1.0, seed=0):
        self.heights = np.array(heights, dtype=float)
        self.n = self.heights.shape[0]
        self.paths = enumerate_paths(self.n)

        self.sensor_actions = [(r, c) for r in range(self.n) for c in range(self.n)]
        self.path_actions = list(range(len(self.paths)))

        self.fog = fog
        self.k_diff = k_diff
        self.k_height = k_height
        self.rng = random.Random(seed)

    # ----- detection model -----

    def p_detect_cell(self, sensor, cell):
        hs = self.heights[sensor]
        hc = self.heights[cell]
        diff = abs(hs - hc)
        # Higher sensor elevation increases detection probability.
        logit = (self.k_diff * diff) - self.fog + (self.k_height * hs)
        return float(np.clip(sigmoid(logit), 0.05, 0.95))

    def p_detect_path(self, sensor, path_idx):
        p_not = 1.0
        for cell in self.paths[path_idx]:
            p_not *= (1.0 - self.p_detect_cell(sensor, cell))
        return 1.0 - p_not

    # ----- environment step -----

    def step(self, sensor_idx, path_idx):
        """
        sensor_idx : int  (action by Player A)
        path_idx   : int  (action by Player B)

        returns dict with rewards and info
        """
        sensor = self.sensor_actions[sensor_idx]
        p = self.p_detect_path(sensor, path_idx)
        detected = self.rng.random() < p

        reward_A = 1 if detected else -1
        reward_B = -reward_A

        return {
            "sensor_idx": sensor_idx,
            "path_idx": path_idx,
            "sensor": sensor,
            "path": self.paths[path_idx],
            "p_detect": p,
            "detected": detected,
            "reward_A": reward_A,
            "reward_B": reward_B,
        }


# -------------------------
# Example agent interfaces
# -------------------------

class RandomSensorAgent:
    def act(self, game):
        return random.randrange(len(game.sensor_actions))


class RandomPathAgent:
    def act(self, game):
        return random.randrange(len(game.path_actions))


# -------------------------
# Generic match runner
# -------------------------

def play_game(game, agent_A, agent_B, rounds=100):
    """
    agent_A.act(game) -> sensor_idx
    agent_B.act(game) -> path_idx
    """
    history = []

    for t in range(rounds):
        a = agent_A.act(game)
        b = agent_B.act(game)
        outcome = game.step(a, b)
        history.append(outcome)

    return history
