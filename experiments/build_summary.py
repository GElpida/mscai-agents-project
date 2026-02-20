from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_RUN_DIR_RE = re.compile(r"^(?P<game>.+?)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_\d+)?$")


GAME_NAME_TO_KEY: dict[str, str] = {
    "MatchingPennies": "matching_pennies",
    "AntiCoordination": "anti_coordination",
    "AlmostRockPaperScissors": "almost_rock_paper_scissors",
    "PrisonersDilemma": "prisoners_dilemma",
    "StochasticSwitchingDominanceGame": "switching_dominance",
    "TerrainGame": "terrain_sensor",
}


ACTION_LABELS: dict[str, dict[int, str] | None] = {
    "matching_pennies": {0: "Heads", 1: "Tails"},
    "anti_coordination": {0: "Strategy A", 1: "Strategy B"},
    "almost_rock_paper_scissors": {0: "Rock", 1: "Paper", 2: "Scissors"},
    "prisoners_dilemma": {0: "Cooperate", 1: "Defect"},
    "switching_dominance": {0: "A0", 1: "A1"},
    "terrain_sensor": None,
}


@dataclass(frozen=True)
class PayoffSpec:
    kind: str  # "zero_sum" | "general_sum" | "zero_sum_stochastic" | "matrix_only" | "env_no_matrix"
    A: np.ndarray | None = None
    B: np.ndarray | None = None
    A_by_state: np.ndarray | None = None  # shape (S, m, n)

_GAME_KEY_TO_MATRIX_CLASS: dict[str, tuple[str, str]] = {
    "matching_pennies": ("games.matching_pennies", "MatchingPennies"),
    "anti_coordination": ("games.anti_coordination", "AntiCoordination"),
    "prisoners_dilemma": ("games.prisoners_dilemma", "PrisonersDilemma"),
    "almost_rock_paper_scissors": ("games.almost_rock_paper_scissors", "AlmostRockPaperScissors"),
}


def _load_payoff_spec(game_key: str) -> PayoffSpec:
    if game_key == "matching_pennies":
        mod_name, cls_name = _GAME_KEY_TO_MATRIX_CLASS[game_key]
        game_cls = getattr(importlib.import_module(mod_name), cls_name)
        A = np.asarray(game_cls.get_payoff_matrix(), dtype=float)
        return PayoffSpec(kind="zero_sum", A=A)

    if game_key in {"anti_coordination", "prisoners_dilemma"}:
        mod_name, cls_name = _GAME_KEY_TO_MATRIX_CLASS[game_key]
        game_cls = getattr(importlib.import_module(mod_name), cls_name)
        A = np.asarray(game_cls.get_payoff_matrix(), dtype=float)
        return PayoffSpec(kind="general_sum", A=A, B=A.T)

    if game_key == "almost_rock_paper_scissors":
        mod_name, cls_name = _GAME_KEY_TO_MATRIX_CLASS[game_key]
        game_cls = getattr(importlib.import_module(mod_name), cls_name)
        A = np.asarray(game_cls.get_payoff_matrix(), dtype=float)
        return PayoffSpec(kind="matrix_only", A=A)

    if game_key == "switching_dominance":
        from games.stochastic_switching_dominance import StochasticSwitchingDominanceGame

        g = StochasticSwitchingDominanceGame()
        A_by_state = np.asarray(g.A, dtype=float)
        return PayoffSpec(kind="zero_sum_stochastic", A_by_state=A_by_state)

    return PayoffSpec(kind="env_no_matrix")


def _action_label(game_key: str, a: int) -> str:
    mapping = ACTION_LABELS.get(game_key)
    if not mapping:
        return str(int(a))
    return mapping.get(int(a), str(int(a)))


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    s = float(np.sum(p))
    if not math.isfinite(s) or s <= 0:
        return float("nan")
    p = p / s
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(-np.sum(p * np.log(p)))


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    s = float(np.sum(p))
    if s <= 0:
        return np.full_like(p, 1.0 / float(p.size))
    return p / s


def _u_xy(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    x = _normalize(x)
    y = _normalize(y)
    return float(x @ A @ y)


def _exploit_zero_sum(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    x = _normalize(x)
    y = _normalize(y)
    best_p1 = float(np.max(A @ y))
    worst_vs_p2 = float(np.min(x @ A))
    return best_p1 - worst_vs_p2


def _exploit_general_sum(A: np.ndarray, B: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = _normalize(x)
    y = _normalize(y)
    u1 = _u_xy(A, x, y)
    u2 = _u_xy(B, x, y)
    br1 = float(np.max(A @ y))
    br2 = float(np.max(x @ B))
    e1 = br1 - u1
    e2 = br2 - u2
    return {"exploit1": e1, "exploit2": e2, "exploit_total": e1 + e2, "u1": u1, "u2": u2}


def _read_results_csv(path: Path) -> dict[str, np.ndarray]:
    rounds: list[int] = []
    states: list[int] = []
    a1: list[int] = []
    a2: list[int] = []
    r1: list[float] = []
    r2: list[float] = []
    reg1: list[float] = []
    reg2: list[float] = []
    has_state = False
    has_regret = False

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(float(row["Round"])))
            if "State" in row and row["State"] != "" and row["State"] is not None:
                states.append(int(float(row["State"])))
                has_state = True
            a1.append(int(float(row["Agent1_Action"])))
            a2.append(int(float(row["Agent2_Action"])))
            r1.append(float(row["Agent1_Payoff"]))
            r2.append(float(row["Agent2_Payoff"]))
            if "Agent1_Regret" in row and "Agent2_Regret" in row and row["Agent1_Regret"] != "" and row["Agent2_Regret"] != "":
                reg1.append(float(row["Agent1_Regret"]))
                reg2.append(float(row["Agent2_Regret"]))
                has_regret = True

    out: dict[str, np.ndarray] = {
        "t": np.asarray(rounds, dtype=int),
        "a1": np.asarray(a1, dtype=int),
        "a2": np.asarray(a2, dtype=int),
        "r1": np.asarray(r1, dtype=float),
        "r2": np.asarray(r2, dtype=float),
    }
    if has_state and len(states) == len(rounds):
        out["state"] = np.asarray(states, dtype=int)
    if has_regret and len(reg1) == len(rounds) and len(reg2) == len(rounds):
        out["regret1"] = np.asarray(reg1, dtype=float)
        out["regret2"] = np.asarray(reg2, dtype=float)
    return out


def _try_load_states_from_npz(run_dir: Path, expected_len: int) -> np.ndarray | None:
    npz_path = run_dir / "data.npz"
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path) as d:
            if "states" not in d.files:
                return None
            states = np.asarray(d["states"], dtype=int).reshape(-1)
    except Exception:
        return None
    return states if int(states.size) == int(expected_len) else None


def _reconstruct_switching_states(*, Tn: int, switch_p: float, seed: int) -> np.ndarray:
    """
    Reconstruct the per-round state sequence for StochasticSwitchingDominanceGame.

    This game’s state transition depends only on RNG draws (and `switch_p`), not on actions.
    We treat the "state at round t" as the current state *before* calling step(...),
    matching what experiment CSVs log when they include a State column.
    """
    Tn = int(Tn)
    rng = np.random.default_rng(int(seed))
    states = np.zeros(Tn, dtype=int)
    s = 0
    for i in range(Tn):
        states[i] = int(s)
        if rng.random() < float(switch_p):
            s = 1 - int(s)
    return states


def _empirical_dist(actions: np.ndarray, n_actions: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=int).reshape(-1)
    if actions.size == 0:
        return np.ones(int(n_actions), dtype=float) / float(n_actions)
    counts = np.bincount(actions, minlength=int(n_actions)).astype(float)
    return counts / float(actions.size)


def _rolling_dists(actions: np.ndarray, n_actions: int, window: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=int).reshape(-1)
    T = int(actions.size)
    W = int(min(max(1, window), T)) if T > 0 else 1
    dists = np.zeros((T, int(n_actions)), dtype=float)
    counts = np.zeros(int(n_actions), dtype=int)

    buf = np.empty(T, dtype=int) if T > 0 else np.empty(0, dtype=int)
    head = 0
    size = 0

    for i, a in enumerate(actions.tolist()):
        counts[int(a)] += 1
        buf[(head + size) % T] = int(a)
        size += 1
        if size > W:
            old = int(buf[head])
            counts[old] -= 1
            head = (head + 1) % T
            size -= 1
        denom = float(np.sum(counts))
        dists[i, :] = (counts / denom) if denom > 0 else (np.ones(int(n_actions)) / float(n_actions))

    return dists


def _infer_game_name(run_dir_name: str) -> str:
    m = _RUN_DIR_RE.match(run_dir_name)
    return m.group("game") if m else run_dir_name


def _try_parse_action_spaces_from_report(run_dir: Path) -> tuple[int, int] | None:
    report_path = run_dir / "report.txt"
    if not report_path.exists():
        return None

    p1: int | None = None
    p2: int | None = None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                if p1 is None:
                    m1 = re.search(r"Action space \(P1\):\s*(\d+)", line)
                    if m1:
                        p1 = int(m1.group(1))
                if p2 is None:
                    m2 = re.search(r"Action space \(P2\):\s*(\d+)", line)
                    if m2:
                        p2 = int(m2.group(1))
                if p1 is not None and p2 is not None:
                    break
    except Exception:
        return None

    if p1 is None or p2 is None:
        return None
    return int(p1), int(p2)


def _try_parse_args_from_report(run_dir: Path) -> dict[str, float]:
    """
    Best-effort parse of scalar args from `report.txt`.

    Some older runs may not have `args.json`; the report includes lines like `seed: 0`.
    """
    report_path = run_dir / "report.txt"
    if not report_path.exists():
        return {}

    out: dict[str, float] = {}
    patterns: dict[str, re.Pattern[str]] = {
        "seed": re.compile(r"^\s*seed:\s*(?P<v>[-+]?\d+)\s*$"),
        "switch_p": re.compile(r"^\s*switch_p:\s*(?P<v>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"),
        "terrain_n": re.compile(r"^\s*terrain_n:\s*(?P<v>[-+]?\d+)\s*$"),
        "terrain_fog": re.compile(r"^\s*terrain_fog:\s*(?P<v>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"),
        "terrain_k_diff": re.compile(r"^\s*terrain_k_diff:\s*(?P<v>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"),
        "terrain_k_height": re.compile(r"^\s*terrain_k_height:\s*(?P<v>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"),
    }
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                for k, pat in patterns.items():
                    if k in out:
                        continue
                    m = pat.match(line)
                    if m:
                        out[k] = float(m.group("v"))
                if len(out) == len(patterns):
                    break
    except Exception:
        return {}
    return out


def _enumerate_monotone_paths(n: int) -> list[list[tuple[int, int]]]:
    """
    All monotone (down/right) paths from (0,0) to (n-1,n-1).

    Ordering matches `games/terrain_sensor.py` (down-recursion before right-recursion).
    """
    n = int(n)
    goal = (n - 1, n - 1)
    paths: list[list[tuple[int, int]]] = []

    def rec(r: int, c: int, path: list[tuple[int, int]]) -> None:
        if (r, c) == goal:
            paths.append(list(path))
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


def _terrain_path_cell_masks(n: int) -> np.ndarray:
    """
    Returns mask matrix of shape (n_paths, n*n) where entry is 1 if the path visits the cell.
    """
    paths = _enumerate_monotone_paths(int(n))
    mask = np.zeros((len(paths), int(n) * int(n)), dtype=float)
    for pi, path in enumerate(paths):
        for (r, c) in path:
            idx = int(r) * int(n) + int(c)
            mask[int(pi), int(idx)] = 1.0
    return mask


def _default_terrain_heights(n: int, seed: int) -> np.ndarray:
    """
    Deterministic terrain heightfield used by the TerrainGame adapter.

    Must match `experiments/game_registry.py::_default_terrain_heights`.
    """
    rng = np.random.default_rng(int(seed))
    heights = rng.normal(loc=0.0, scale=1.0, size=(int(n), int(n)))
    heights = heights - float(np.min(heights))
    heights = heights / float(np.max(heights) + 1e-12)
    return heights


def _try_load_args_json(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "args.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            out = json.load(f)
    except Exception:
        return None
    return out if isinstance(out, dict) else None


_TERRAIN_A_CACHE: dict[tuple[int, int, float, float, float], np.ndarray] = {}


def _terrain_expected_payoff_matrix_p1(*, n: int, seed: int, fog: float, k_diff: float, k_height: float) -> np.ndarray:
    """
    Expected payoff matrix for TerrainGame (Player 1 = sensor placement).

    We reconstruct it from run parameters to define exploitability in the summary.
    Expected payoff matches `experiments/game_registry.py::_terrain_expected_payoff_matrix_p1`:
      A[sensor_idx, path_idx] = 2 * p_detect_path - 1
    """
    n = int(n)
    seed = int(seed)
    fog = float(fog)
    k_diff = float(k_diff)
    k_height = float(k_height)

    cache_key = (n, seed, float(fog), float(k_diff), float(k_height))
    cached = _TERRAIN_A_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    heights = _default_terrain_heights(n=int(n), seed=int(seed))
    n_cells = int(n) * int(n)
    path_masks = _terrain_path_cell_masks(int(n))  # (n_paths, n_cells)
    n_paths = int(path_masks.shape[0])

    h_flat = np.asarray(heights, dtype=float).reshape(-1)
    A = np.zeros((n_cells, n_paths), dtype=float)

    for sensor_idx in range(n_cells):
        hs = float(h_flat[int(sensor_idx)])
        diff = np.abs(hs - h_flat)
        logit = (k_diff * diff) - fog + (k_height * hs)
        p_cells = 1.0 / (1.0 + np.exp(-logit))
        p_cells = np.clip(p_cells, 0.05, 0.95)
        log_1mp = np.log(np.clip(1.0 - p_cells, 1e-12, 1.0))
        log_p_not = path_masks @ log_1mp
        p_detect = 1.0 - np.exp(log_p_not)
        A[int(sensor_idx), :] = 2.0 * p_detect - 1.0

    _TERRAIN_A_CACHE[cache_key] = A.copy()
    return A


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_payoff_running_mean(out_path: Path, t: np.ndarray, r1: np.ndarray, r2: np.ndarray, *, title: str) -> None:
    t = np.asarray(t, dtype=int)
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    rm1 = np.cumsum(r1) / np.arange(1, r1.size + 1)
    rm2 = np.cumsum(r2) / np.arange(1, r2.size + 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=100)
    ax.plot(t, rm1, linewidth=2, label="Player 1")
    ax.plot(t, rm2, linewidth=2, linestyle="--", label="Player 2")
    ax.set_xlabel("t (round)")
    ax.set_ylabel("Running mean payoff")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_regret(out_path: Path, t: np.ndarray, reg1: np.ndarray, reg2: np.ndarray, *, title: str) -> None:
    t = np.asarray(t, dtype=int)
    reg1 = np.asarray(reg1, dtype=float)
    reg2 = np.asarray(reg2, dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=100)
    ax.plot(t, reg1, linewidth=2, label="Player 1")
    ax.plot(t, reg2, linewidth=2, linestyle="--", label="Player 2")
    ax.set_xlabel("t (round)")
    ax.set_ylabel("Cumulative regret")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _joint_action_matrix(a1: np.ndarray, a2: np.ndarray, n1: int, n2: int) -> np.ndarray:
    a1 = np.asarray(a1, dtype=int).reshape(-1)
    a2 = np.asarray(a2, dtype=int).reshape(-1)
    n1 = int(n1)
    n2 = int(n2)
    M = np.zeros((n1, n2), dtype=float)
    if int(a1.size) == 0:
        return M
    for i in range(int(a1.size)):
        x = int(a1[i])
        y = int(a2[i])
        if 0 <= x < n1 and 0 <= y < n2:
            M[x, y] += 1.0
    s = float(np.sum(M))
    return (M / s) if s > 0 else M


def _plot_joint_action_heatmap(
    out_path: Path,
    P: np.ndarray,
    *,
    title: str,
    x_label: str = "P2 action",
    y_label: str = "P1 action",
) -> None:
    P = np.asarray(P, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=110)
    im = ax.imshow(P, origin="lower", aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    c = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    c.set_label("Probability")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_action_proportions(
    out_path: Path,
    t: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    *,
    game_key: str,
    n_actions_p1: int,
    n_actions_p2: int,
    window: int,
    title: str,
) -> None:
    t = np.asarray(t, dtype=int)
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)
    props1 = _rolling_dists(a1, int(n_actions_p1), int(window))
    props2 = _rolling_dists(a2, int(n_actions_p2), int(window))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9.5, 7.4), dpi=100, sharex=True)
    W_eff = min(int(window), int(a1.size)) if int(a1.size) > 0 else int(window)

    for a in range(int(n_actions_p1)):
        ax1.plot(t, props1[:, a], linewidth=2, label=_action_label(game_key, a))
    ax1.set_ylabel(f"P1 rolling prop (W={W_eff})")
    ax1.legend(loc="upper right", frameon=False, ncol=1)

    for b in range(int(n_actions_p2)):
        ax2.plot(t, props2[:, b], linewidth=2, label=_action_label(game_key, b))
    ax2.set_xlabel("t (round)")
    ax2.set_ylabel(f"P2 rolling prop (W={W_eff})")
    ax2.legend(loc="upper right", frameon=False, ncol=1)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_exploitability(out_path: Path, t: np.ndarray, expl: np.ndarray, *, title: str) -> None:
    t = np.asarray(t, dtype=int)
    expl = np.asarray(expl, dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=100)
    ax.plot(t, expl, linewidth=2)
    ax.axhline(0.1, linestyle="--", linewidth=1)
    ax.axhline(0.05, linestyle=":", linewidth=1)
    ax.set_xlabel("t (round)")
    ax.set_ylabel("Exploitability (rolling)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_terrain_movement_gif(
    out_path: Path,
    t: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    *,
    heights: np.ndarray,
    n: int,
    n_paths: int,
    fog: float,
    k_diff: float,
    window: int,
    title: str,
    max_frames: int = 90,
    fps: int = 12,
) -> None:
    from matplotlib import animation
    from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
    from matplotlib.patches import Circle

    t = np.asarray(t, dtype=int).reshape(-1)
    a1 = np.asarray(a1, dtype=int).reshape(-1)
    a2 = np.asarray(a2, dtype=int).reshape(-1)

    n = int(n)
    heights = np.asarray(heights, dtype=float)
    if heights.shape != (n, n):
        raise ValueError(f"TerrainGame: expected heights shape ({n},{n}), got {tuple(heights.shape)}")
    fog = float(fog)
    k_diff = float(k_diff)
    if not math.isfinite(fog) or not math.isfinite(k_diff):
        raise ValueError(f"TerrainGame: fog/k_diff must be finite, got fog={fog!r}, k_diff={k_diff!r}")
    n_cells = int(n) * int(n)
    if int(a1.size) == 0:
        return
    if int(np.max(a1)) >= n_cells:
        raise ValueError(f"TerrainGame: sensor action index out of range for n={n}: max(a1)={int(np.max(a1))}")
    if int(np.max(a2)) >= int(n_paths):
        raise ValueError(f"TerrainGame: path action index out of range: max(a2)={int(np.max(a2))}, n_paths={int(n_paths)}")

    paths = _enumerate_monotone_paths(n)
    if int(len(paths)) != int(n_paths):
        raise ValueError(f"TerrainGame: expected n_paths={int(n_paths)} from report, but enumerated {len(paths)} for n={n}")

    Tn = int(t.size)
    frames = int(min(int(max_frames), Tn))
    frame_idxs = np.linspace(0, Tn - 1, num=frames, dtype=int).tolist() if frames > 1 else [Tn - 1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.8, 6.2), dpi=95)

    elev_cmap = LinearSegmentedColormap.from_list(
        "elevation_green_to_red",
        ["#1a9850", "#ffffbf", "#d73027"],  # green -> yellow -> red
        N=256,
    )
    elev_norm = Normalize(vmin=float(np.min(heights)), vmax=float(np.max(heights)))

    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(heights, cmap=elev_cmap, norm=elev_norm, vert_exag=1.6, blend_mode="soft")
    im = ax.imshow(shaded, origin="lower", interpolation="nearest")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=elev_norm, cmap=elev_cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Elevation")

    ax.set_xlabel("col")
    ax.set_ylabel("row")

    sensor_scatter = ax.scatter([], [], s=70, c="white", edgecolors="black", linewidths=1.2, zorder=4)
    (path_line,) = ax.plot([], [], color="#00bcd4", linewidth=2.2, zorder=4)
    sensor_range = Circle(
        (0.0, 0.0),
        radius=0.0,
        facecolor="#00bcd4",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.5,
        zorder=3,
    )
    ax.add_patch(sensor_range)

    text = fig.text(0.5, 0.02, "", ha="center", va="bottom")

    def _expected_cells_from_elevation(sr: int, sc: int) -> float:
        """
        "Sensor range" for visualization only (not part of TerrainGame dynamics).

        Requirement:
          - Max range = 4 cells when the sensor is at the highest elevation.

        We implement:
          E[cells] = 4 * normalize(h_sensor)  in [0, 4]
        """
        hmin = float(np.min(heights))
        hmax = float(np.max(heights))
        hs = float(heights[int(sr), int(sc)])
        if not math.isfinite(hmin) or not math.isfinite(hmax) or abs(hmax - hmin) < 1e-12:
            return 0.0
        hs_norm = (hs - hmin) / (hmax - hmin)
        return float(np.clip(4.0 * float(hs_norm), 0.0, 4.0))

    def update(i: int):
        k = int(i)

        sensor_idx = int(a1[k])
        sr = sensor_idx // n
        sc = sensor_idx % n
        sensor_scatter.set_offsets(np.array([[sc, sr]], dtype=float))
        sensor_range.center = (float(sc), float(sr))
        # Visual range is derived from elevation (max 4 cells at highest point).
        expected_cells = _expected_cells_from_elevation(int(sr), int(sc))
        sensor_range.radius = math.sqrt(max(0.0, expected_cells) / math.pi)

        path_idx = int(a2[k])
        coords = paths[path_idx]
        xs = [c for (_, c) in coords]
        ys = [r for (r, _) in coords]
        path_line.set_data(xs, ys)

        text.set_text(f"{title}    t={int(t[k])}    E[cells]={expected_cells:.2f}")
        return (im, sensor_scatter, path_line, sensor_range, text)

    anim = animation.FuncAnimation(fig, update, frames=frame_idxs, interval=int(1000 / max(1, int(fps))), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=int(fps)))
    plt.close(fig)


def _plot_switching_dominance_gif(
    out_path: Path,
    t: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    *,
    states: np.ndarray | None,
    A_by_state: np.ndarray,
    title: str,
    max_frames: int = 140,
    fps: int = 12,
) -> None:
    from matplotlib import animation
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    t = np.asarray(t, dtype=int).reshape(-1)
    a1 = np.asarray(a1, dtype=int).reshape(-1)
    a2 = np.asarray(a2, dtype=int).reshape(-1)
    if states is not None:
        states = np.asarray(states, dtype=int).reshape(-1)

    Tn = int(t.size)
    if Tn <= 0:
        return

    A_by_state = np.asarray(A_by_state, dtype=float)
    if A_by_state.ndim != 3:
        raise ValueError(f"switching_dominance: expected A_by_state ndim=3, got {A_by_state.ndim}")
    nS, nA1, nA2 = (int(A_by_state.shape[0]), int(A_by_state.shape[1]), int(A_by_state.shape[2]))
    if nA1 <= 0 or nA2 <= 0:
        return

    if int(np.max(a1)) >= nA1:
        raise ValueError(f"switching_dominance: max(a1)={int(np.max(a1))} out of range for nA1={nA1}")
    if int(np.max(a2)) >= nA2:
        raise ValueError(f"switching_dominance: max(a2)={int(np.max(a2))} out of range for nA2={nA2}")

    frames = int(min(int(max_frames), Tn))
    frame_idxs = np.linspace(0, Tn - 1, num=frames, dtype=int).tolist() if frames > 1 else [Tn - 1]

    # Swap roles when players "catch": interpret as landing on same grid cell (a1 == a2).
    swapped_role = np.zeros(Tn, dtype=bool)
    swapped = False
    for i in range(Tn):
        swapped_role[i] = swapped
        if int(a1[i]) == int(a2[i]):
            swapped = not swapped

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.8, 5.8), dpi=110)
    vmin = float(np.min(A_by_state))
    vmax = float(np.max(A_by_state))
    norm = Normalize(vmin=vmin, vmax=vmax)

    init_state = int(states[0]) if states is not None and states.size else 0
    init_state = max(0, min(nS - 1, int(init_state)))
    im = ax.imshow(
        A_by_state[init_state],
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
        norm=norm,
        extent=(-0.5, nA1 - 0.5, -0.5, nA2 - 0.5),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Payoff (P1)")

    # Grid lines
    for x in range(nA1 + 1):
        ax.axvline(x - 0.5, color="black", linewidth=0.8, alpha=0.5)
    for y in range(nA2 + 1):
        ax.axhline(y - 0.5, color="black", linewidth=0.8, alpha=0.5)

    ax.set_xlim(-0.5, nA1 - 0.5)
    ax.set_ylim(-0.5, nA2 - 0.5)
    ax.set_xticks(list(range(nA1)))
    ax.set_yticks(list(range(nA2)))
    ax.set_xticklabels([_action_label("switching_dominance", x) for x in range(nA1)])
    ax.set_yticklabels([_action_label("switching_dominance", y) for y in range(nA2)])
    ax.set_xlabel("P1 action")
    ax.set_ylabel("P2 action")

    # Player markers; symbols swap on role swap.
    p1, = ax.plot([], [], linestyle="", marker="o", markersize=11, markerfacecolor="white", markeredgecolor="black", zorder=5)
    p2, = ax.plot([], [], linestyle="", marker="^", markersize=11, markerfacecolor="#00bcd4", markeredgecolor="black", zorder=5)

    highlight = Rectangle((0.0, 0.0), 1.0, 1.0, fill=False, linewidth=2.2, edgecolor="#ffd54f", zorder=6)
    ax.add_patch(highlight)

    text = fig.text(0.5, 0.02, "", ha="center", va="bottom")

    def update(i: int):
        k = int(i)
        st = int(states[k]) if states is not None and int(states.size) > k else 0
        st = max(0, min(nS - 1, int(st)))
        im.set_data(A_by_state[st])

        x = int(a1[k])
        y = int(a2[k])
        caught = bool(x == y)
        swapped_now = bool(swapped_role[k])

        # Offsets so both points are visible even when they coincide.
        p1_off = (-0.14, 0.14)
        p2_off = (0.14, -0.14)

        if swapped_now:
            p1.set_marker("^")
            p1.set_markerfacecolor("#00bcd4")
            p2.set_marker("o")
            p2.set_markerfacecolor("white")
        else:
            p1.set_marker("o")
            p1.set_markerfacecolor("white")
            p2.set_marker("^")
            p2.set_markerfacecolor("#00bcd4")

        p1.set_data([x + p1_off[0]], [y + p1_off[1]])
        p2.set_data([x + p2_off[0]], [y + p2_off[1]])

        highlight.set_xy((x - 0.5, y - 0.5))
        highlight.set_visible(bool(caught))

        text.set_text(f"{title}    t={int(t[k])}    state={st}    catch={int(caught)}    swapped={int(swapped_now)}")
        return (im, p1, p2, highlight, text)

    anim = animation.FuncAnimation(fig, update, frames=frame_idxs, interval=int(1000 / max(1, int(fps))), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=int(fps)))
    plt.close(fig)


def build_summary(results_dir: Path, out_dir: Path, *, window: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_for_global_summary: list[dict[str, Any]] = []
    index_lines: list[str] = [
        "# Results Summary\n",
        "\n",
        "Auto-generated. See `README.md` in this folder for metric definitions.\n",
        "\n",
    ]

    experiment_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir() and p.name != "summary"], key=lambda p: p.name)
    for exp_dir in experiment_dirs:
        run_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
        for run_dir in run_dirs:
            results_csv = run_dir / "results.csv"
            if not results_csv.exists():
                continue

            game_name = _infer_game_name(run_dir.name)
            game_key = GAME_NAME_TO_KEY.get(game_name, game_name)
            payoff = _load_payoff_spec(game_key)

            dat = _read_results_csv(results_csv)
            t = dat["t"]
            a1 = dat["a1"]
            a2 = dat["a2"]
            r1 = dat["r1"]
            r2 = dat["r2"]
            states = dat.get("state", None)
            states_source: str | None = "csv" if states is not None else None
            if states is None:
                states = _try_load_states_from_npz(run_dir, int(t.size))
                if states is not None:
                    states_source = "npz"
            if states is None and game_key == "switching_dominance":
                args_json = _try_load_args_json(run_dir) or {}
                if "seed" not in args_json or "switch_p" not in args_json:
                    args_json = {**_try_parse_args_from_report(run_dir), **args_json}
                # When runs don't log State, we can reconstruct deterministically from seed+switch_p.
                # This is specific to StochasticSwitchingDominanceGame (state transitions are action-independent).
                if "seed" in args_json and "switch_p" in args_json and int(t.size) > 0:
                    try:
                        states = _reconstruct_switching_states(
                            Tn=int(t.size),
                            switch_p=float(args_json["switch_p"]),
                            seed=int(args_json["seed"]),
                        )
                        states_source = "reconstructed"
                    except Exception:
                        states = None

            Tn = int(t.size)
            n_actions_p1 = int(np.max(a1) + 1) if a1.size else 1
            n_actions_p2 = int(np.max(a2) + 1) if a2.size else 1

            if payoff.A is not None:
                n_actions_p1 = int(payoff.A.shape[0])
                n_actions_p2 = int(payoff.A.shape[1])
            if payoff.B is not None:
                n_actions_p1 = int(payoff.B.shape[0])
                n_actions_p2 = int(payoff.B.shape[1])
            if payoff.A_by_state is not None:
                n_actions_p1 = int(payoff.A_by_state.shape[1])
                n_actions_p2 = int(payoff.A_by_state.shape[2])

            if game_key == "terrain_sensor":
                parsed = _try_parse_action_spaces_from_report(run_dir)
                if parsed is not None:
                    n_actions_p1, n_actions_p2 = int(parsed[0]), int(parsed[1])

            x_emp = _empirical_dist(a1, n_actions_p1)
            y_emp = _empirical_dist(a2, n_actions_p2)
            Hx = _entropy(x_emp)
            Hy = _entropy(y_emp)

            out_run_dir = out_dir / exp_dir.name / run_dir.name
            out_run_dir.mkdir(parents=True, exist_ok=True)
            for legacy_name in ("plot_p1_action_proportions.png", "plot_p2_action_proportions.png"):
                try:
                    (out_run_dir / legacy_name).unlink()
                except FileNotFoundError:
                    pass
            for legacy_name in ("plot_action_proportions.png", "plot_terrain_movement.gif", "plot_switching_dominance.gif"):
                try:
                    (out_run_dir / legacy_name).unlink()
                except FileNotFoundError:
                    pass
            for legacy_name in (
                "plot_regret.png",
                "plot_joint_action_heatmap.png",
                "action_probabilities.csv",
                "joint_action_probabilities.csv",
            ):
                try:
                    (out_run_dir / legacy_name).unlink()
                except FileNotFoundError:
                    pass

            emp_rows: list[dict[str, Any]] = []
            for a in range(int(n_actions_p1)):
                emp_rows.append(
                    {"player": "P1", "action": int(a), "action_label": _action_label(game_key, a), "prob": float(x_emp[a])}
                )
            for b in range(int(n_actions_p2)):
                emp_rows.append(
                    {"player": "P2", "action": int(b), "action_label": _action_label(game_key, b), "prob": float(y_emp[b])}
                )
            _write_csv(out_run_dir / "empirical_strategies.csv", emp_rows)
            _write_csv(out_run_dir / "action_probabilities.csv", emp_rows)

            joint = _joint_action_matrix(a1, a2, n_actions_p1, n_actions_p2)
            joint_rows: list[dict[str, Any]] = []
            for a in range(int(n_actions_p1)):
                for b in range(int(n_actions_p2)):
                    joint_rows.append(
                        {
                            "action_p1": int(a),
                            "action_p1_label": _action_label(game_key, a),
                            "action_p2": int(b),
                            "action_p2_label": _action_label(game_key, b),
                            "prob": float(joint[a, b]),
                        }
                    )
            _write_csv(out_run_dir / "joint_action_probabilities.csv", joint_rows)
            _plot_joint_action_heatmap(
                out_run_dir / "plot_joint_action_heatmap.png",
                joint,
                title=f"Joint action probabilities: {game_name} ({exp_dir.name})",
            )

            running_mean_r1 = (np.cumsum(r1) / np.arange(1, r1.size + 1)) if r1.size else np.zeros(0, dtype=float)
            running_mean_r2 = (np.cumsum(r2) / np.arange(1, r2.size + 1)) if r2.size else np.zeros(0, dtype=float)
            regret1 = dat.get("regret1", None)
            regret2 = dat.get("regret2", None)

            expl_curve: np.ndarray | None = None
            expl_final: float | None = None
            expl_kind: str = "NA"
            exploit_detail: dict[str, float] | None = None
            t_lt_01: int | None = None
            t_lt_005: int | None = None
            exploit_note: str | None = None
            x_roll: np.ndarray | None = None
            y_roll: np.ndarray | None = None

            if game_key == "terrain_sensor" and payoff.kind == "env_no_matrix":
                args_json = _try_load_args_json(run_dir) or {}
                if any(k not in args_json for k in ("seed", "terrain_n", "terrain_fog", "terrain_k_diff", "terrain_k_height")):
                    args_json = {**_try_parse_args_from_report(run_dir), **args_json}
                try:
                    n_side = int(args_json.get("terrain_n", 0)) or int(round(math.sqrt(float(n_actions_p1))))
                    seed = int(args_json.get("seed", 0))
                    fog = float(args_json.get("terrain_fog", 0.25))
                    k_diff = float(args_json.get("terrain_k_diff", 0.9))
                    # Backward-compatible default: older runs didn't have elevation-boost dynamics.
                    k_height = float(args_json.get("terrain_k_height", 0.0))
                    A_terrain = _terrain_expected_payoff_matrix_p1(
                        n=n_side,
                        seed=seed,
                        fog=fog,
                        k_diff=k_diff,
                        k_height=k_height,
                    )
                    if A_terrain.shape == (int(n_actions_p1), int(n_actions_p2)):
                        payoff = PayoffSpec(kind="zero_sum", A=A_terrain)
                except Exception:
                    pass

            if payoff.kind == "zero_sum" and payoff.A is not None:
                expl_kind = "zero_sum_minimax_gap"
                expl_final = _exploit_zero_sum(payoff.A, x_emp, y_emp)
                W = int(min(int(window), Tn)) if Tn > 0 else 1
                x_roll = _rolling_dists(a1, n_actions_p1, W)
                y_roll = _rolling_dists(a2, n_actions_p2, W)
                expl_curve = np.asarray([_exploit_zero_sum(payoff.A, x_roll[i], y_roll[i]) for i in range(Tn)], dtype=float)
            elif game_key == "almost_rock_paper_scissors" and payoff.A is not None:
                # Use the RPS-style definition shown in the assignment screenshot:
                # exploitability(pi) = V* - BR(pi) = V* - min(pi^T A).
                # For this game's payoff matrix (win=1, else=0), V* = 1/3 under the uniform equilibrium.
                expl_kind = "rps_vstar_minus_best_response"
                V_star = 1.0 / 3.0
                br = float(np.min(_normalize(x_emp) @ payoff.A))
                expl_final = float(V_star - br)
                W = int(min(int(window), Tn)) if Tn > 0 else 1
                x_roll = _rolling_dists(a1, n_actions_p1, W)
                expl_curve = np.asarray(
                    [float(V_star - float(np.min(_normalize(x_roll[i]) @ payoff.A))) for i in range(Tn)],
                    dtype=float,
                )
            elif payoff.kind == "general_sum" and payoff.A is not None and payoff.B is not None:
                expl_kind = "general_sum_exploitability_sum"
                exploit_detail = _exploit_general_sum(payoff.A, payoff.B, x_emp, y_emp)
                expl_final = float(exploit_detail["exploit_total"])
                W = int(min(int(window), Tn)) if Tn > 0 else 1
                x_roll = _rolling_dists(a1, n_actions_p1, W)
                y_roll = _rolling_dists(a2, n_actions_p2, W)
                expl_curve = np.asarray(
                    [_exploit_general_sum(payoff.A, payoff.B, x_roll[i], y_roll[i])["exploit_total"] for i in range(Tn)],
                    dtype=float,
                )
            elif payoff.kind == "zero_sum_stochastic" and payoff.A_by_state is not None:
                if states is None:
                    expl_kind = "stochastic_weighted_over_states"
                    exploit_note = "Missing per-round State column in results.csv; cannot compute per-state exploitability."
                else:
                    expl_kind = "stochastic_weighted_over_states"
                    per_state_rows: list[dict[str, Any]] = []
                    total_visits = 0
                    weighted = 0.0
                    # Also compute rolling exploitability curve for thresholds/plots (windowed over time).
                    W = int(min(int(window), Tn)) if Tn > 0 else 1
                    expl_curve = np.zeros(Tn, dtype=float) if Tn > 0 else np.zeros(0, dtype=float)
                    for ss in sorted(set(states.tolist())):
                        idx = np.where(states == int(ss))[0]
                        visits = int(idx.size)
                        if visits <= 0:
                            continue
                        xs = _empirical_dist(a1[idx], n_actions_p1)
                        ys = _empirical_dist(a2[idx], n_actions_p2)
                        A_s = np.asarray(payoff.A_by_state[int(ss)], dtype=float)
                        u_emp = _u_xy(A_s, xs, ys)
                        e = _exploit_zero_sum(A_s, xs, ys)
                        per_state_rows.append(
                            {"state": int(ss), "visits": visits, "u_emp": float(u_emp), "exploitability": float(e)}
                        )
                        total_visits += visits
                        weighted += float(e) * float(visits)
                    if total_visits > 0:
                        expl_final = weighted / float(total_visits)
                    _write_csv(out_run_dir / "per_state.csv", per_state_rows)

                    # Rolling curve: for each time i, compute state-weighted exploitability on the last W rounds.
                    if int(Tn) > 0:
                        for i in range(Tn):
                            end = i + 1
                            start = max(0, end - W)
                            st_win = states[start:end]
                            total = int(st_win.size)
                            if total <= 0:
                                expl_curve[i] = float("nan")
                                continue
                            e_w = 0.0
                            for ss in sorted(set(st_win.tolist())):
                                idx_local = np.where(st_win == int(ss))[0]
                                v = int(idx_local.size)
                                if v <= 0:
                                    continue
                                idx_global = idx_local + start
                                xs = _empirical_dist(a1[idx_global], n_actions_p1)
                                ys = _empirical_dist(a2[idx_global], n_actions_p2)
                                A_s = np.asarray(payoff.A_by_state[int(ss)], dtype=float)
                                e_w += float(_exploit_zero_sum(A_s, xs, ys)) * (float(v) / float(total))
                            expl_curve[i] = float(e_w)
            else:
                exploit_note = "Exploitability not defined for this game under the available payoff spec."

            if expl_curve is not None and expl_curve.size > 0:
                idx01 = np.where(expl_curve < 0.1)[0]
                idx005 = np.where(expl_curve < 0.05)[0]
                t_lt_01 = int(t[int(idx01[0])]) if idx01.size > 0 else None
                t_lt_005 = int(t[int(idx005[0])]) if idx005.size > 0 else None

            # Policy convergence (single scalar): Δt = ||π_t - π_{t-1}||_1 where π_t is the
            # concatenation of rolling action distributions for P1 and P2 (window=W).
            W_pol = int(min(int(window), Tn)) if Tn > 0 else 1
            x_pol = x_roll if x_roll is not None else _rolling_dists(a1, n_actions_p1, W_pol)
            y_pol = y_roll if y_roll is not None else _rolling_dists(a2, n_actions_p2, W_pol)
            if int(Tn) >= 2:
                policy_convergence = float(
                    np.sum(np.abs(x_pol[-1] - x_pol[-2])) + np.sum(np.abs(y_pol[-1] - y_pol[-2]))
                )
            else:
                policy_convergence = 0.0

            timeseries_rows: list[dict[str, Any]] = []
            for i in range(Tn):
                timeseries_rows.append(
                    {
                        "t": int(t[i]),
                        "r1": float(r1[i]),
                        "r2": float(r2[i]),
                        "running_mean_r1": float(running_mean_r1[i]) if running_mean_r1.size else float("nan"),
                        "running_mean_r2": float(running_mean_r2[i]) if running_mean_r2.size else float("nan"),
                        "exploit_roll": float(expl_curve[i]) if expl_curve is not None else "",
                        "regret_p1": float(regret1[i]) if regret1 is not None else "",
                        "regret_p2": float(regret2[i]) if regret2 is not None else "",
                    }
                )
            _write_csv(out_run_dir / "timeseries.csv", timeseries_rows)

            _plot_payoff_running_mean(
                out_run_dir / "plot_payoff_running_mean.png",
                t,
                r1,
                r2,
                title=f"Running mean payoff: {game_name} ({exp_dir.name})",
            )

            action_plot_name = "plot_action_proportions.png"
            if game_key == "terrain_sensor":
                action_plot_name = "plot_terrain_movement.gif"
                n_side = int(round(math.sqrt(float(n_actions_p1))))
                if int(n_side) * int(n_side) != int(n_actions_p1):
                    raise ValueError(
                        f"TerrainGame: expected square sensor action space, got n_actions_p1={int(n_actions_p1)}"
                    )
                args_json = _try_load_args_json(run_dir) or {}
                seed = int(args_json.get("seed", 0))
                fog = float(args_json.get("terrain_fog", 0.25))
                k_diff = float(args_json.get("terrain_k_diff", 0.9))
                _plot_terrain_movement_gif(
                    out_run_dir / action_plot_name,
                    t,
                    a1,
                    a2,
                    heights=_default_terrain_heights(n=int(n_side), seed=int(seed)),
                    n=int(n_side),
                    n_paths=int(n_actions_p2),
                    fog=float(fog),
                    k_diff=float(k_diff),
                    window=int(window),
                    title=f"TerrainGame ({exp_dir.name})",
                )
            elif game_key == "switching_dominance":
                action_plot_name = "plot_switching_dominance.gif"
                if payoff.A_by_state is None:
                    raise ValueError("switching_dominance: missing A_by_state payoff spec for GIF")
                if states is None:
                    states_for_plot = np.zeros(int(t.size), dtype=int)
                else:
                    states_for_plot = np.asarray(states, dtype=int).reshape(-1)
                _plot_switching_dominance_gif(
                    out_run_dir / action_plot_name,
                    t,
                    a1,
                    a2,
                    states=states_for_plot,
                    A_by_state=np.asarray(payoff.A_by_state, dtype=float),
                    title=f"SwitchingDominance ({exp_dir.name})",
                )
            else:
                _plot_action_proportions(
                    out_run_dir / action_plot_name,
                    t,
                    a1,
                    a2,
                    game_key=game_key,
                    n_actions_p1=n_actions_p1,
                    n_actions_p2=n_actions_p2,
                    window=int(window),
                    title=f"Action proportions (P1 & P2): {game_name} ({exp_dir.name})",
                )
            if expl_curve is not None and expl_curve.size > 0:
                _plot_exploitability(
                    out_run_dir / "plot_exploitability.png",
                    t,
                    expl_curve,
                    title=f"Exploitability: {game_name} ({exp_dir.name})",
                )

            if regret1 is not None and regret2 is not None:
                _plot_regret(
                    out_run_dir / "plot_regret.png",
                    t,
                    regret1,
                    regret2,
                    title=f"Regret over time: {game_name} ({exp_dir.name})",
                )

            final_regret_p1 = float(regret1[-1]) if regret1 is not None and int(regret1.size) > 0 else None
            final_regret_p2 = float(regret2[-1]) if regret2 is not None and int(regret2.size) > 0 else None

            metrics: dict[str, Any] = {
                "experiment": exp_dir.name,
                "run": run_dir.name,
                "game_name": game_name,
                "game_key": game_key,
                "n_rounds": Tn,
                "window": int(min(int(window), Tn)) if Tn > 0 else 0,
                "state_source": states_source,
                "payoff_total_p1": float(np.sum(r1)) if r1.size else 0.0,
                "payoff_total_p2": float(np.sum(r2)) if r2.size else 0.0,
                "payoff_mean_p1": float(np.mean(r1)) if r1.size else 0.0,
                "payoff_mean_p2": float(np.mean(r2)) if r2.size else 0.0,
                "entropy_p1": float(Hx),
                "entropy_p2": float(Hy),
                "final_regret_p1": final_regret_p1,
                "final_regret_p2": final_regret_p2,
                "exploitability_kind": expl_kind,
                "exploitability_final": float(expl_final) if expl_final is not None else None,
                "t_exploit_lt_0_10": t_lt_01,
                "t_exploit_lt_0_05": t_lt_005,
                "notes": exploit_note,
            }
            if exploit_detail is not None:
                metrics.update({k: float(v) for k, v in exploit_detail.items()})

            with open(out_run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            rows_for_global_summary.append(
                {
                    "experiment": exp_dir.name,
                    "game": game_name,
                    "n_rounds": Tn,
                    "entropy_p1": float(Hx),
                    "entropy_p2": float(Hy),
                    "final_regret_p1": float(final_regret_p1) if final_regret_p1 is not None else "-",
                    "final_regret_p2": float(final_regret_p2) if final_regret_p2 is not None else "-",
                    "exploitability_final": float(expl_final) if expl_final is not None else "-",
                    "t_exploit_lt_0_10": int(t_lt_01) if t_lt_01 is not None else "-",
                    "payoff_mean_p1": float(np.mean(r1)) if r1.size else 0.0,
                    "payoff_mean_p2": float(np.mean(r2)) if r2.size else 0.0,
                    "policy_convergence": float(policy_convergence),
                    "out_dir": str(out_run_dir.relative_to(out_dir)).replace("\\", "/"),
                }
            )

            rel = out_run_dir.relative_to(out_dir).as_posix()
            index_lines.extend(
                [
                    f"## {exp_dir.name} / {run_dir.name}\n",
                    "\n",
                    f"- Folder: `{rel}`\n",
                    f"- Metrics: `{rel}/metrics.json`\n",
                    f"- Empirical strategies: `{rel}/empirical_strategies.csv`\n",
                    f"- Action probs (CSV): `{rel}/action_probabilities.csv`\n",
                    f"- Joint move probs (CSV): `{rel}/joint_action_probabilities.csv`\n",
                    f"- Plots: `{rel}/plot_payoff_running_mean.png`, `{rel}/{action_plot_name}`, `{rel}/plot_joint_action_heatmap.png`"
                    + (f", `{rel}/plot_exploitability.png`" if (expl_curve is not None and expl_curve.size > 0) else "")
                    + (f", `{rel}/plot_regret.png`" if (regret1 is not None and regret2 is not None) else "")
                    + "\n",
                    "\n",
                ]
            )

    _write_csv(out_dir / "summary.csv", rows_for_global_summary)
    with open(out_dir / "INDEX.md", "w", encoding="utf-8") as f:
        f.writelines(index_lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build `results/summary` with metrics + plots for all experiments/games.")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("results") / "summary")
    ap.add_argument("--window", type=int, default=500, help="Rolling window size (W).")
    args = ap.parse_args()

    build_summary(args.results_dir, args.out_dir, window=int(args.window))
    print(f"Wrote summary to: {args.out_dir}")


if __name__ == "__main__":
    main()
