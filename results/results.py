# results.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np

# ---- FIX THESE IMPORTS TO MATCH YOUR PROJECT ----
from games.StochasticSwitchingDominanceGame import StochasticSwitchingDominanceGame
from agents.RL_minimaxQ import MinimaxQLearner
from agents.RL_q_independent import IndependentQLearner
from agents.agent_fp import FictitousPlayAgent


# =========================
# Helpers: 2x2 game metrics
# =========================

def best_response_row(A: np.ndarray, y: np.ndarray) -> int:
    """Row player's best response against column mixed strategy y."""
    pay = A @ y
    return int(np.argmax(pay))

def best_response_col(A: np.ndarray, x: np.ndarray) -> int:
    """Column player's best response (minimizer) against row mixed strategy x."""
    pay = x @ A  # shape (2,)
    return int(np.argmin(pay))

def payoff(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ A @ y)

def exploitability_2x2_zero_sum(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """
    Exploitability for zero-sum matrix game with payoff A for Player 1.
    exploitability(x,y) = (max_x' u(x',y) - u(x,y)) + (u(x,y) - min_y' u(x,y'))
    """
    u_xy = payoff(A, x, y)

    # Row best response value:
    i = best_response_row(A, y)
    u_br_row = float(A[i, :] @ y)

    # Column best response value (minimizer):
    j = best_response_col(A, x)
    u_br_col = float(x @ A[:, j])

    return (u_br_row - u_xy) + (u_xy - u_br_col)

def freq_to_mixed(counts: np.ndarray) -> np.ndarray:
    """Convert action counts to a mixed strategy."""
    s = float(np.sum(counts))
    if s <= 0:
        return np.ones_like(counts) / len(counts)
    return counts / s


# =========================
# Run logs
# =========================

@dataclass
class RunLog:
    states: np.ndarray          # (T,)
    a_actions: np.ndarray       # (T,) Player 1 actions
    b_actions: np.ndarray       # (T,) Player 2 actions
    rewards: np.ndarray         # (T,) reward for Player 1
    next_states: np.ndarray     # (T,)


# =========================
# Experiments
# =========================

def run_rl_vs_rl(game, steps: int, seed: int, alpha: float, gamma: float, eps: float) -> Tuple[RunLog, Dict[str, Any]]:
    nS = game.get_num_states()
    nA = game.get_num_actions()
    if nA != 2:
        raise ValueError("This results.py currently expects 2 actions (2x2).")

    p1 = MinimaxQLearner(n_states=nS, n_actions=nA, alpha=alpha, gamma=gamma, eps=eps, seed=seed + 1)
    p2 = MinimaxQLearner(n_states=nS, n_actions=nA, alpha=alpha, gamma=gamma, eps=eps, seed=seed + 2)

    s = game.reset()

    states = np.zeros(steps, dtype=int)
    next_states = np.zeros(steps, dtype=int)
    a_actions = np.zeros(steps, dtype=int)
    b_actions = np.zeros(steps, dtype=int)
    rewards = np.zeros(steps, dtype=float)

    for t in range(steps):
        a = p1.act(s)
        b = p2.act(s)

        s_next, r1 = game.step(s, a, b)

        # zero-sum: P2 gets -r1
        p1.update(s, a, b, r1, s_next)
        p2.update(s, b, a, -r1, s_next)

        states[t] = s
        next_states[t] = s_next
        a_actions[t] = a
        b_actions[t] = b
        rewards[t] = r1

        s = s_next

    log = RunLog(states, a_actions, b_actions, rewards, next_states)
    ctx = {"p1": p1, "p2": p2, "mode": "rl_vs_rl"}
    return log, ctx


def run_rl_vs_fp(game, steps: int, seed: int, alpha: float, gamma: float, eps: float, fp_strategy: str) -> Tuple[RunLog, Dict[str, Any]]:
    nS = game.get_num_states()
    nA = game.get_num_actions()
    if nA != 2:
        raise ValueError("This results.py currently expects 2 actions (2x2).")

    rl = IndependentQLearner(n_states=nS, n_actions=nA, alpha=alpha, gamma=gamma, eps=eps, seed=seed + 10)

    # Keep FP as-is: one FP per state with fixed matrix A_s
    fp_agents: List[FictitousPlayAgent] = []
    for s in range(nS):
        A_s = game.get_payoff_matrix(s)
        fp_agents.append(
            FictitousPlayAgent(
                payoff_matrix=A_s,
                action_space=nA,
                opponent_action_space=nA,
                strategy_type=fp_strategy,
            )
        )

    s = game.reset()

    states = np.zeros(steps, dtype=int)
    next_states = np.zeros(steps, dtype=int)
    a_actions = np.zeros(steps, dtype=int)  # RL action (we treat RL as Player 1)
    b_actions = np.zeros(steps, dtype=int)  # FP action (Player 2)
    rewards = np.zeros(steps, dtype=float)

    for t in range(steps):
        fp = fp_agents[s]

        a = rl.act(s)     # RL action
        b = fp.play()     # FP action

        s_next, r1 = game.step(s, a, b)  # reward for Player 1 (RL)

        rl.update(s, a, r1, s_next)

        # FP observes opponent action (RL)
        fp.observe(opponent_action=a)

        states[t] = s
        next_states[t] = s_next
        a_actions[t] = a
        b_actions[t] = b
        rewards[t] = r1

        s = s_next

    log = RunLog(states, a_actions, b_actions, rewards, next_states)
    ctx = {"rl": rl, "fp_agents": fp_agents, "mode": "rl_vs_fp"}
    return log, ctx


# =========================
# Metrics + display
# =========================

def display_run(game, log: RunLog, n: int = 30):
    n = int(min(n, len(log.states)))
    print("\n=== Display of a run (first steps) ===")
    print(" t |  s | a(P1) | b(P2) |  r1 | s'")
    print("---+----+-------+-------+-----+----")
    for t in range(n):
        print(f"{t:2d} | {log.states[t]:2d} |   {log.a_actions[t]:1d}   |   {log.b_actions[t]:1d}   | {log.rewards[t]:>3.0f} | {log.next_states[t]:2d}")


def compute_metrics(game, log: RunLog, ctx: Dict[str, Any]) -> Dict[str, Any]:
    nS = game.get_num_states()
    nA = game.get_num_actions()

    out: Dict[str, Any] = {}

    out["T"] = int(len(log.states))
    out["mean_reward"] = float(np.mean(log.rewards))
    out["std_reward"] = float(np.std(log.rewards))

    # Action frequencies per state (empirical strategies)
    p1_counts = np.zeros((nS, nA), dtype=float)
    p2_counts = np.zeros((nS, nA), dtype=float)
    state_counts = np.zeros(nS, dtype=float)

    for s, a, b in zip(log.states, log.a_actions, log.b_actions):
        state_counts[s] += 1.0
        p1_counts[s, a] += 1.0
        p2_counts[s, b] += 1.0

    out["state_visits"] = state_counts.copy()

    # Per-state exploitability on the *stage* payoff matrix A_s (nice, interpretable).
    # For RLvsRL we can also report learned minimax policies from the agents.
    per_state = []
    for s in range(nS):
        A_s = game.get_payoff_matrix(s)
        x_emp = freq_to_mixed(p1_counts[s])
        y_emp = freq_to_mixed(p2_counts[s])
        expl = exploitability_2x2_zero_sum(A_s, x_emp, y_emp)

        per_state.append({
            "state": int(s),
            "visits": int(state_counts[s]),
            "x_emp": x_emp.copy(),
            "y_emp": y_emp.copy(),
            "exploitability_emp": float(expl),
            "u_emp": float(payoff(A_s, x_emp, y_emp)),
        })

    out["per_state"] = per_state

    # Optional: policies implied by agents (if available)
    if ctx.get("mode") == "rl_vs_rl":
        p1 = ctx["p1"]
        p2 = ctx["p2"]
        pols = []
        for s in range(nS):
            A_s = game.get_payoff_matrix(s)

            x_pi = p1.policy(s)  # minimax policy wrt learned Q
            y_qi = p2.policy(s)  # minimax policy for P2 wrt its learned Q (over its own actions)

            # Note: y_qi is a distribution over P2 actions already.
            expl_pi = exploitability_2x2_zero_sum(A_s, x_pi, y_qi)

            pols.append({
                "state": int(s),
                "p1_minimax_policy": x_pi.copy(),
                "p2_minimax_policy": y_qi.copy(),
                "exploitability_policies": float(expl_pi),
                "u_policies": float(payoff(A_s, x_pi, y_qi)),
                "p1_V_est": float(p1.value(s)),
                "p2_V_est": float(p2.value(s)),
            })
        out["agent_policies"] = pols

    return out


def print_metrics(metrics: Dict[str, Any]):
    print("\n=== Metrics summary ===")
    print(f"T = {metrics['T']}")
    print(f"Mean reward (P1): {metrics['mean_reward']:.4f}")
    print(f"Std  reward (P1): {metrics['std_reward']:.4f}")

    print("\nPer-state (empirical frequencies):")
    for row in metrics["per_state"]:
        x = row["x_emp"]; y = row["y_emp"]
        print(
            f"  state {row['state']}: visits={row['visits']}, "
            f"x_emp={x.round(3)}, y_emp={y.round(3)}, "
            f"u_emp={row['u_emp']:.4f}, exploit={row['exploitability_emp']:.4f}"
        )

    if "agent_policies" in metrics:
        print("\nPer-state (agent minimax policies from learned Q):")
        for row in metrics["agent_policies"]:
            print(
                f"  state {row['state']}: "
                f"p1_pi={row['p1_minimax_policy'].round(3)}, "
                f"p2_pi={row['p2_minimax_policy'].round(3)}, "
                f"u={row['u_policies']:.4f}, exploit={row['exploitability_policies']:.4f}, "
                f"V1={row['p1_V_est']:.4f}, V2={row['p2_V_est']:.4f}"
            )


def save_metrics_csv(metrics: Dict[str, Any], path: str):
    # Save per-state as rows
    rows = []
    for r in metrics["per_state"]:
        rows.append({
            "state": r["state"],
            "visits": r["visits"],
            "x_emp_0": float(r["x_emp"][0]),
            "x_emp_1": float(r["x_emp"][1]),
            "y_emp_0": float(r["y_emp"][0]),
            "y_emp_1": float(r["y_emp"][1]),
            "u_emp": r["u_emp"],
            "exploitability_emp": r["exploitability_emp"],
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["state"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSaved per-state metrics CSV -> {path}")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rl_vs_rl", "rl_vs_fp"], required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--switch_p", type=float, default=0.2)

    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=0.1)

    ap.add_argument("--fp_strategy", choices=["pure", "mixed"], default="pure")
    ap.add_argument("--display_n", type=int, default=25)

    ap.add_argument("--csv_out", type=str, default="")
    args = ap.parse_args()

    game = StochasticSwitchingDominanceGame(switch_p=args.switch_p, seed=args.seed)

    if args.mode == "rl_vs_rl":
        log, ctx = run_rl_vs_rl(game, args.steps, args.seed, args.alpha, args.gamma, args.eps)
    else:
        log, ctx = run_rl_vs_fp(game, args.steps, args.seed, args.alpha, args.gamma, args.eps, args.fp_strategy)

    display_run(game, log, n=args.display_n)

    metrics = compute_metrics(game, log, ctx)
    print_metrics(metrics)

    if args.csv_out:
        save_metrics_csv(metrics, args.csv_out)


if __name__ == "__main__":
    main()
