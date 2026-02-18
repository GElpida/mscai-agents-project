from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CSV_COLUMNS = [
    "Round",
    "Agent1_Type",
    "Agent2_Type",
    "Agent1_ExpPayoff",
    "Agent2_ExpPayoff",
    "Agent1_Payoff",
    "Agent2_Payoff",
    "Agent1_Action",
    "Agent2_Action",
    "Agent1_Regret",
    "Agent2_Regret",
    "Agent1_Belief",
    "Agent2_Belief",
    "Agent1_Visits",
    "Agent2_Visits",
]


def make_unique_dir(parent: Path, name: str) -> Path:
    candidate = parent / name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    for i in range(1, 10_000):
        candidate = parent / f"{name}_{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate

    raise RuntimeError(f"Could not create a unique directory under: {parent}")


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _vector_to_json(v: Any, *, decimals: int = 4) -> str:
    arr = np.asarray(v, dtype=float).reshape(-1)
    rounded = [round(float(x), int(decimals)) for x in arr.tolist()]
    return json.dumps(rounded, ensure_ascii=False)


def _action_names_or_default(names: list[str] | None, prefix: str, n: int) -> list[str]:
    if names is not None and len(names) >= int(n):
        return list(names)
    return [f"{prefix}{i}" for i in range(int(n))]


def _format_matrix(M: np.ndarray, *, max_cells: int = 100) -> str:
    M = np.asarray(M, dtype=float)
    if int(M.size) <= int(max_cells):
        return str(M)
    return (
        f"<matrix shape={M.shape}, min={float(np.min(M)):.3f}, "
        f"max={float(np.max(M)):.3f}, mean={float(np.mean(M)):.3f}>"
    )


def write_results_csv(run_dir: Path, rows: Iterable[dict[str, Any]]) -> Path:
    out = run_dir / "results.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out


@dataclass(frozen=True)
class ReportInputs:
    experiment_name: str
    game: Any
    num_rounds: int
    agent1_type: str
    agent2_type: str
    args: dict[str, Any]
    states: list[int]
    actions_p1: list[int]
    actions_p2: list[int]
    payoffs_p1: list[float]
    payoffs_p2: list[float]
    final_beliefs_by_state_p1: list[np.ndarray] | None = None
    final_beliefs_by_state_p2: list[np.ndarray] | None = None
    final_regret_p1: float | None = None
    final_regret_p2: float | None = None


def write_report_txt(run_dir: Path, r: ReportInputs) -> Path:
    out = run_dir / "report.txt"

    game = r.game
    game_name = getattr(game, "name", type(game).__name__)
    timestamp_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    action_names_p1 = _action_names_or_default(getattr(game, "action_names_p1", None), "a", game.n_actions_p1)
    action_names_p2 = _action_names_or_default(getattr(game, "action_names_p2", None), "b", game.n_actions_p2)

    counts_p1 = np.bincount(np.asarray(r.actions_p1, dtype=int), minlength=game.n_actions_p1) if r.actions_p1 else np.zeros(game.n_actions_p1)
    counts_p2 = np.bincount(np.asarray(r.actions_p2, dtype=int), minlength=game.n_actions_p2) if r.actions_p2 else np.zeros(game.n_actions_p2)
    probs_p1 = (counts_p1 / float(np.sum(counts_p1))) if float(np.sum(counts_p1)) > 0 else (np.ones(game.n_actions_p1) / game.n_actions_p1)
    probs_p2 = (counts_p2 / float(np.sum(counts_p2))) if float(np.sum(counts_p2)) > 0 else (np.ones(game.n_actions_p2) / game.n_actions_p2)

    state_counts = np.bincount(np.asarray(r.states, dtype=int), minlength=game.n_states) if r.states else np.zeros(game.n_states, dtype=int)

    with open(out, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"{r.experiment_name.upper()} GAME REPORT: {game_name}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {timestamp_human}\n\n")

        f.write("GAME CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of rounds: {int(r.num_rounds)}\n")
        f.write(f"Number of states: {int(game.n_states)}\n")
        f.write(f"Action space (P1): {int(game.n_actions_p1)}\n")
        f.write(f"Action space (P2): {int(game.n_actions_p2)}\n")
        f.write(f"Zero-sum: {bool(game.is_zero_sum)}\n")
        f.write(f"Agent 1 type: {r.agent1_type}\n")
        f.write(f"Agent 2 type: {r.agent2_type}\n")
        for k, v in sorted(r.args.items()):
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("PAYOFF MATRIX\n")
        f.write("-" * 70 + "\n")
        for s in range(int(game.n_states)):
            A_s = np.asarray(game.payoff_matrix_p1(s), dtype=float)
            # Convention:
            # - A_s is indexed as A_s[a, b]  (rows=P1 action a, cols=P2 action b)
            # - B_s is indexed as B_s[b, a]  (rows=P2 action b, cols=P1 action a)
            # For readability in the report we print P2 payoffs aligned to (a, b),
            # i.e. as B_s.T where B_s.T[a, b] = B_s[b, a] = r2(a,b).
            B_s = np.asarray(game.payoff_matrix_p2(s), dtype=float)
            if int(game.n_states) > 1:
                f.write(f"State {s}:\n")
            f.write("P1 payoff (rows=a, cols=b):\n")
            f.write(_format_matrix(A_s) + "\n")
            f.write("P2 payoff (rows=a, cols=b):\n")
            f.write(_format_matrix(B_s.T) + "\n\n")

        f.write("RESULTS SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Agent 1 total payoff: {float(np.sum(r.payoffs_p1)):.4f}\n")
        f.write(f"Agent 2 total payoff: {float(np.sum(r.payoffs_p2)):.4f}\n")
        f.write(f"Agent 1 average payoff: {float(np.mean(r.payoffs_p1)) if r.payoffs_p1 else 0.0:.4f}\n")
        f.write(f"Agent 2 average payoff: {float(np.mean(r.payoffs_p2)) if r.payoffs_p2 else 0.0:.4f}\n")
        if r.final_regret_p1 is not None:
            f.write(f"Agent 1 final regret: {float(r.final_regret_p1):.6f}\n")
        if r.final_regret_p2 is not None:
            f.write(f"Agent 2 final regret: {float(r.final_regret_p2):.6f}\n")
        if int(game.n_states) > 1:
            f.write("\nState visitation counts:\n")
            for s in range(int(game.n_states)):
                f.write(f"  state {s}: {int(state_counts[s])}\n")
        f.write("\n")

        f.write("ACTION PROBABILITY DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write("Agent 1 Action Probabilities:\n")
        for a in range(int(game.n_actions_p1)):
            f.write(f"  {action_names_p1[a]}: {float(probs_p1[a]):.4f} ({int(counts_p1[a])} plays)\n")
        f.write("\nAgent 2 Action Probabilities:\n")
        for b in range(int(game.n_actions_p2)):
            f.write(f"  {action_names_p2[b]}: {float(probs_p2[b]):.4f} ({int(counts_p2[b])} plays)\n")
        f.write("\n")

        f.write("BELIEF CONVERGENCE\n")
        f.write("-" * 70 + "\n")
        if r.final_beliefs_by_state_p1 is not None and r.final_beliefs_by_state_p2 is not None:
            if int(game.n_states) == 1:
                f.write(f"Agent 1 final belief (over P2 actions): {np.asarray(r.final_beliefs_by_state_p1[0]).round(4)}\n")
                f.write(f"Agent 2 final belief (over P1 actions): {np.asarray(r.final_beliefs_by_state_p2[0]).round(4)}\n")
            else:
                for s in range(int(game.n_states)):
                    f.write(f"State {s}:\n")
                    f.write(f"  Agent 1 belief (over P2 actions): {np.asarray(r.final_beliefs_by_state_p1[s]).round(4)}\n")
                    f.write(f"  Agent 2 belief (over P1 actions): {np.asarray(r.final_beliefs_by_state_p2[s]).round(4)}\n")
        else:
            f.write("<not available>\n")
        f.write("\n")

        f.write("ACTION HISTORY\n")
        f.write("-" * 70 + "\n")
        f.write("Agent 1: " + ", ".join(action_names_p1[a] if a < len(action_names_p1) else str(a) for a in r.actions_p1) + "\n")
        f.write("Agent 2: " + ", ".join(action_names_p2[b] if b < len(action_names_p2) else str(b) for b in r.actions_p2) + "\n\n")

        f.write("PAYOFF HISTORY\n")
        f.write("-" * 70 + "\n")
        for i, (p1, p2) in enumerate(zip(r.payoffs_p1, r.payoffs_p2), 1):
            f.write(f"Round {i:4d}: Agent1={float(p1): .6f}, Agent2={float(p2): .6f}\n")

    return out


def build_csv_row(
    *,
    round_idx: int,
    agent1_type: str,
    agent2_type: str,
    agent1_exp_payoff: float,
    agent2_exp_payoff: float,
    agent1_payoff: float,
    agent2_payoff: float,
    agent1_action: int,
    agent2_action: int,
    agent1_regret: float,
    agent2_regret: float,
    agent1_belief: Any,
    agent2_belief: Any,
    agent1_visits: int,
    agent2_visits: int,
) -> dict[str, Any]:
    return {
        "Round": int(round_idx),
        "Agent1_Type": agent1_type,
        "Agent2_Type": agent2_type,
        "Agent1_ExpPayoff": float(agent1_exp_payoff),
        "Agent2_ExpPayoff": float(agent2_exp_payoff),
        "Agent1_Payoff": float(agent1_payoff),
        "Agent2_Payoff": float(agent2_payoff),
        "Agent1_Action": int(agent1_action),
        "Agent2_Action": int(agent2_action),
        "Agent1_Regret": float(agent1_regret),
        "Agent2_Regret": float(agent2_regret),
        "Agent1_Belief": _vector_to_json(agent1_belief),
        "Agent2_Belief": _vector_to_json(agent2_belief),
        "Agent1_Visits": int(agent1_visits),
        "Agent2_Visits": int(agent2_visits),
    }
