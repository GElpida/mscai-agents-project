from __future__ import annotations

import inspect
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TwoPlayerGame:
    """
    Minimal adapter interface used by experiment entrypoints.

    Conventions:
      - Player 1 chooses action a in [0, n_actions_p1)
      - Player 2 chooses action b in [0, n_actions_p2)
      - step(...) returns (next_state, r1, r2)
    """

    name: str
    n_states: int
    n_actions_p1: int
    n_actions_p2: int
    is_zero_sum: bool

    def reset(self) -> int:  # pragma: no cover - interface stub
        raise NotImplementedError

    def step(self, state: int, a: int, b: int) -> tuple[int, float, float]:  # pragma: no cover - interface stub
        raise NotImplementedError

    def payoff_matrix_p1(self, state: int) -> np.ndarray:  # pragma: no cover - interface stub
        raise NotImplementedError

    def payoff_matrix_p2(self, state: int) -> np.ndarray:  # pragma: no cover - interface stub
        raise NotImplementedError


class _MatrixMarkovGame(TwoPlayerGame):
    def __init__(self, *, game_class, is_zero_sum: bool):
        A = np.asarray(game_class.get_payoff_matrix(), dtype=float)
        if A.ndim != 2:
            raise ValueError(f"{game_class.__name__}.get_payoff_matrix() must return 2D array, got {A.shape}")

        self._A = A
        if is_zero_sum:
            # r2(a,b) = -r1(a,b) but our B is indexed as B[b,a], so B = -A.T.
            self._B = -A.T
        else:
            # Symmetric matrix-game convention used throughout this repo:
            # Player 2's payoff for (b,a) is the same payoff function with swapped arguments,
            # i.e. r2(a,b) = A[b,a]. Therefore the (rows=b, cols=a) matrix is B = A.
            if A.shape[0] != A.shape[1]:
                raise ValueError(
                    f"{game_class.__name__} is non-zero-sum but has non-square payoff matrix {A.shape}; "
                    "cannot infer Player 2 payoffs from Player 1 matrix."
                )
            self._B = A.copy()

        action_names = getattr(game_class, "ACTION_NAMES", None)
        n1, n2 = A.shape
        super().__init__(
            name=game_class.__name__,
            n_states=1,
            n_actions_p1=int(n1),
            n_actions_p2=int(n2),
            is_zero_sum=bool(is_zero_sum),
        )
        self.action_names_p1 = list(action_names) if action_names is not None else None
        self.action_names_p2 = list(action_names) if action_names is not None else None

    def reset(self) -> int:
        return 0

    def step(self, state: int, a: int, b: int) -> tuple[int, float, float]:
        a = int(a)
        b = int(b)
        r1 = float(self._A[a, b])
        r2 = float(-r1) if self.is_zero_sum else float(self._B[b, a])
        return 0, r1, r2

    def payoff_matrix_p1(self, state: int) -> np.ndarray:
        return self._A.copy()

    def payoff_matrix_p2(self, state: int) -> np.ndarray:
        return self._B.copy()


class _StochasticSwitchingDominance(TwoPlayerGame):
    def __init__(self, *, switch_p: float, seed: int):
        from games.stochastic_switching_dominance import StochasticSwitchingDominanceGame

        self._game = StochasticSwitchingDominanceGame(switch_p=float(switch_p), seed=int(seed))
        super().__init__(
            name="StochasticSwitchingDominanceGame",
            n_states=int(self._game.get_num_states()),
            n_actions_p1=int(self._game.get_num_actions()),
            n_actions_p2=int(self._game.get_num_actions()),
            is_zero_sum=True,
        )

    def reset(self) -> int:
        return int(self._game.reset())

    def step(self, state: int, a: int, b: int) -> tuple[int, float, float]:
        s_next, r1 = self._game.step(int(state), int(a), int(b))
        r1 = float(r1)
        return int(s_next), r1, float(-r1)

    def payoff_matrix_p1(self, state: int) -> np.ndarray:
        return np.asarray(self._game.get_payoff_matrix(int(state)), dtype=float)

    def payoff_matrix_p2(self, state: int) -> np.ndarray:
        A = self.payoff_matrix_p1(state)
        return -A.T


class _TerrainSensor(TwoPlayerGame):
    def __init__(self, *, n: int, fog: float, k_diff: float, k_height: float, seed: int):
        from games.terrain_sensor import TerrainGame

        heights = _default_terrain_heights(n=int(n), seed=int(seed))
        self._game = TerrainGame(
            heights=heights,
            fog=float(fog),
            k_diff=float(k_diff),
            k_height=float(k_height),
            seed=int(seed),
        )
        self._A_exp = _terrain_expected_payoff_matrix_p1(self._game)
        super().__init__(
            name="TerrainGame",
            n_states=1,
            n_actions_p1=int(self._A_exp.shape[0]),
            n_actions_p2=int(self._A_exp.shape[1]),
            is_zero_sum=True,
        )

    def reset(self) -> int:
        return 0

    def step(self, state: int, a: int, b: int) -> tuple[int, float, float]:
        out = self._game.step(int(a), int(b))
        r1 = float(out["reward_A"])
        r2 = float(out["reward_B"])
        return 0, r1, r2

    def payoff_matrix_p1(self, state: int) -> np.ndarray:
        return self._A_exp.copy()

    def payoff_matrix_p2(self, state: int) -> np.ndarray:
        return -self._A_exp.T


def _default_terrain_heights(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    heights = rng.normal(loc=0.0, scale=1.0, size=(int(n), int(n)))
    heights = heights - float(np.min(heights))
    heights = heights / float(np.max(heights) + 1e-12)
    return heights


def _terrain_expected_payoff_matrix_p1(game) -> np.ndarray:
    m = len(game.sensor_actions)
    n = len(game.path_actions)
    A = np.zeros((m, n), dtype=float)
    for sensor_idx in range(m):
        sensor = game.sensor_actions[sensor_idx]
        for path_idx in range(n):
            p = float(game.p_detect_path(sensor, path_idx))
            A[sensor_idx, path_idx] = 2.0 * p - 1.0
    return A


def discover_games(
    *,
    seed: int,
    switch_p: float,
    terrain_n: int = 4,
    terrain_fog: float = 0.25,
    terrain_k_diff: float = 0.9,
    terrain_k_height: float = 1.0,
    include_terrain: bool = True,
) -> list[TwoPlayerGame]:
    """
    Discover all games under `games/` and return adapters usable by experiments.

    Notes:
      - Matrix games are wrapped as a single-state Markov game.
      - `switching_dominance.py` is a compatibility shim and is ignored.
    """
    root = Path(__file__).resolve().parents[1]
    games_dir = root / "games"

    zero_sum_names = {
        "MatchingPennies",
        "StochasticSwitchingDominanceGame",
        "TerrainGame",
    }

    discovered: list[TwoPlayerGame] = []

    for py in sorted(games_dir.glob("*.py")):
        if py.name in {"__init__.py", "switching_dominance.py"}:
            continue

        module = importlib.import_module(f"games.{py.stem}")
        for _, obj in vars(module).items():
            if not inspect.isclass(obj):
                continue
            if obj.__module__ != module.__name__:
                continue

            if obj.__name__ == "StochasticSwitchingDominanceGame":
                discovered.append(_StochasticSwitchingDominance(switch_p=switch_p, seed=seed))
                continue

            if obj.__name__ == "TerrainGame":
                if include_terrain:
                    discovered.append(
                        _TerrainSensor(
                            n=terrain_n,
                            fog=terrain_fog,
                            k_diff=terrain_k_diff,
                            k_height=terrain_k_height,
                            seed=seed,
                        )
                    )
                continue

            if hasattr(obj, "get_payoff_matrix") and hasattr(obj, "NUM_ACTIONS"):
                try:
                    A = obj.get_payoff_matrix()
                except TypeError:
                    continue
                if isinstance(A, np.ndarray) or hasattr(A, "shape"):
                    is_zero_sum = obj.__name__ in zero_sum_names
                    discovered.append(_MatrixMarkovGame(game_class=obj, is_zero_sum=is_zero_sum))

    # De-duplicate by name (defensive against accidental duplicates)
    unique: dict[str, TwoPlayerGame] = {}
    for g in discovered:
        unique.setdefault(g.name, g)

    return sorted(unique.values(), key=lambda g: g.name.lower())


def filter_games(games: Iterable[TwoPlayerGame], only: list[str] | None) -> list[TwoPlayerGame]:
    if not only:
        return list(games)
    only_set = {s.strip() for s in only if s.strip()}
    return [g for g in games if g.name in only_set]
