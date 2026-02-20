# experiments/main_rl_vs_rl.py
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.agent_rl_minimaxq import MinimaxQLearner
from agents.agent_rl_q import IndependentQLearner

from experiments.game_registry import discover_games, filter_games
from experiments.output_artifacts import (
    ReportInputs,
    build_csv_row,
    make_unique_dir,
    now_timestamp,
    write_report_txt,
    write_results_csv,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "rl_vs_rl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_minimaxq_selfplay(game, steps: int, seed: int, alpha: float, gamma: float, eps: float):
    if not game.is_zero_sum:
        raise ValueError("Minimax-Q self-play requires a zero-sum game adapter.")
    if game.n_actions_p1 != 2 or game.n_actions_p2 != 2:
        raise ValueError("This Minimax-Q implementation currently supports 2 actions only.")

    p1 = MinimaxQLearner(
        n_states=game.n_states,
        n_actions=2,
        alpha=alpha,
        gamma=gamma,
        eps=eps,
        seed=seed + 1,
    )
    p2 = MinimaxQLearner(
        n_states=game.n_states,
        n_actions=2,
        alpha=alpha,
        gamma=gamma,
        eps=eps,
        seed=seed + 2,
    )

    s = game.reset()

    rewards_p1 = np.zeros(steps, dtype=float)
    rewards_p2 = np.zeros(steps, dtype=float)
    states = np.zeros(steps, dtype=int)
    actions_p1 = np.zeros(steps, dtype=int)
    actions_p2 = np.zeros(steps, dtype=int)

    # empirical beliefs per state (with pseudocounts)
    counts_p2_by_state = [np.ones(game.n_actions_p2, dtype=float) for _ in range(game.n_states)]
    counts_p1_by_state = [np.ones(game.n_actions_p1, dtype=float) for _ in range(game.n_states)]

    action_visits_p1 = np.zeros(game.n_actions_p1, dtype=int)
    action_visits_p2 = np.zeros(game.n_actions_p2, dtype=int)
    regret_p1 = 0.0
    regret_p2 = 0.0
    csv_rows = []

    for t in range(steps):
        A_s = np.asarray(game.payoff_matrix_p1(s), dtype=float)
        B_s = np.asarray(game.payoff_matrix_p2(s), dtype=float)

        belief_p1 = counts_p2_by_state[s] / float(np.sum(counts_p2_by_state[s]))  # over P2 actions
        belief_p2 = counts_p1_by_state[s] / float(np.sum(counts_p1_by_state[s]))  # over P1 actions

        a = p1.act(s)
        b = p2.act(s)

        s_next, r1, r2 = game.step(s, a, b)  # rewards for (p1, p2)

        exp1 = float(A_s[int(a)] @ belief_p1)
        exp2 = float(B_s[int(b)] @ belief_p2)

        best1 = float(np.max(A_s @ belief_p1))
        best2 = float(np.max(B_s @ belief_p2))
        regret_p1 += best1 - float(r1)
        regret_p2 += best2 - float(r2)

        counts_p2_by_state[s][int(b)] += 1.0
        counts_p1_by_state[s][int(a)] += 1.0

        action_visits_p1[int(a)] += 1
        action_visits_p2[int(b)] += 1
        v1 = int(action_visits_p1[int(a)])
        v2 = int(action_visits_p2[int(b)])

        csv_rows.append(
            build_csv_row(
                round_idx=t + 1,
                state=int(s),
                agent1_type="MinimaxQLearner",
                agent2_type="MinimaxQLearner",
                agent1_exp_payoff=exp1,
                agent2_exp_payoff=exp2,
                agent1_payoff=float(r1),
                agent2_payoff=float(r2),
                agent1_action=int(a),
                agent2_action=int(b),
                agent1_regret=float(regret_p1),
                agent2_regret=float(regret_p2),
                agent1_belief=belief_p1,
                agent2_belief=belief_p2,
                agent1_visits=v1,
                agent2_visits=v2,
            )
        )

        # Update P1: my action=a, opp=b
        p1.update(s, a, b, r1, s_next)

        # Update P2: for P2, "my action" is b, opponent is a
        p2.update(s, b, a, r2, s_next)

        rewards_p1[t] = float(r1)
        rewards_p2[t] = float(r2)
        states[t] = s
        actions_p1[t] = int(a)
        actions_p2[t] = int(b)
        s = s_next

    print("RL vs RL finished.")
    print(f"Mean reward P1: {rewards_p1.mean():.4f}  (should be near minimax value over time)")
    print(f"Std reward  P1: {rewards_p1.std():.4f}")

    final_beliefs_p1 = [c / float(np.sum(c)) for c in counts_p2_by_state]
    final_beliefs_p2 = [c / float(np.sum(c)) for c in counts_p1_by_state]

    return {
        "algo": "minimax_q_selfplay",
        "rewards_p1": rewards_p1,
        "rewards_p2": rewards_p2,
        "states": states,
        "actions_p1": actions_p1,
        "actions_p2": actions_p2,
        "csv_rows": csv_rows,
        "final_beliefs_p1": final_beliefs_p1,
        "final_beliefs_p2": final_beliefs_p2,
        "final_regret_p1": float(regret_p1),
        "final_regret_p2": float(regret_p2),
        "p1": p1,
        "p2": p2,
    }


def run_independentq_selfplay(game, steps: int, seed: int, alpha: float, gamma: float, eps: float):
    p1 = IndependentQLearner(
        n_states=game.n_states,
        n_actions=game.n_actions_p1,
        alpha=alpha,
        gamma=gamma,
        eps=eps,
        seed=seed + 101,
    )
    p2 = IndependentQLearner(
        n_states=game.n_states,
        n_actions=game.n_actions_p2,
        alpha=alpha,
        gamma=gamma,
        eps=eps,
        seed=seed + 202,
    )

    s = game.reset()
    rewards_p1 = np.zeros(steps, dtype=float)
    rewards_p2 = np.zeros(steps, dtype=float)
    states = np.zeros(steps, dtype=int)
    actions_p1 = np.zeros(steps, dtype=int)
    actions_p2 = np.zeros(steps, dtype=int)

    counts_p2_by_state = [np.ones(game.n_actions_p2, dtype=float) for _ in range(game.n_states)]
    counts_p1_by_state = [np.ones(game.n_actions_p1, dtype=float) for _ in range(game.n_states)]

    action_visits_p1 = np.zeros(game.n_actions_p1, dtype=int)
    action_visits_p2 = np.zeros(game.n_actions_p2, dtype=int)
    regret_p1 = 0.0
    regret_p2 = 0.0
    csv_rows = []

    for t in range(steps):
        A_s = np.asarray(game.payoff_matrix_p1(s), dtype=float)
        B_s = np.asarray(game.payoff_matrix_p2(s), dtype=float)

        belief_p1 = counts_p2_by_state[s] / float(np.sum(counts_p2_by_state[s]))
        belief_p2 = counts_p1_by_state[s] / float(np.sum(counts_p1_by_state[s]))

        a = p1.act(s)
        b = p2.act(s)
        s_next, r1, r2 = game.step(s, a, b)

        exp1 = float(A_s[int(a)] @ belief_p1)
        exp2 = float(B_s[int(b)] @ belief_p2)

        best1 = float(np.max(A_s @ belief_p1))
        best2 = float(np.max(B_s @ belief_p2))
        regret_p1 += best1 - float(r1)
        regret_p2 += best2 - float(r2)

        counts_p2_by_state[s][int(b)] += 1.0
        counts_p1_by_state[s][int(a)] += 1.0

        action_visits_p1[int(a)] += 1
        action_visits_p2[int(b)] += 1
        v1 = int(action_visits_p1[int(a)])
        v2 = int(action_visits_p2[int(b)])

        csv_rows.append(
            build_csv_row(
                round_idx=t + 1,
                state=int(s),
                agent1_type="IndependentQLearner",
                agent2_type="IndependentQLearner",
                agent1_exp_payoff=exp1,
                agent2_exp_payoff=exp2,
                agent1_payoff=float(r1),
                agent2_payoff=float(r2),
                agent1_action=int(a),
                agent2_action=int(b),
                agent1_regret=float(regret_p1),
                agent2_regret=float(regret_p2),
                agent1_belief=belief_p1,
                agent2_belief=belief_p2,
                agent1_visits=v1,
                agent2_visits=v2,
            )
        )

        p1.update(s, a, r1, s_next)
        p2.update(s, b, r2, s_next)
        rewards_p1[t] = r1
        rewards_p2[t] = r2
        states[t] = s
        actions_p1[t] = int(a)
        actions_p2[t] = int(b)
        s = s_next

    print("Independent-Q vs Independent-Q finished.")
    print(f"Mean reward P1: {rewards_p1.mean():.4f}")
    print(f"Mean reward P2: {rewards_p2.mean():.4f}")

    final_beliefs_p1 = [c / float(np.sum(c)) for c in counts_p2_by_state]
    final_beliefs_p2 = [c / float(np.sum(c)) for c in counts_p1_by_state]

    return {
        "algo": "independent_q_selfplay",
        "rewards_p1": rewards_p1,
        "rewards_p2": rewards_p2,
        "states": states,
        "actions_p1": actions_p1,
        "actions_p2": actions_p2,
        "csv_rows": csv_rows,
        "final_beliefs_p1": final_beliefs_p1,
        "final_beliefs_p2": final_beliefs_p2,
        "final_regret_p1": float(regret_p1),
        "final_regret_p2": float(regret_p2),
        "p1": p1,
        "p2": p2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--switch_p", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--only", action="append", default=None, help="Run only these game names (repeatable).")
    ap.add_argument("--terrain_n", type=int, default=4)
    ap.add_argument("--terrain_fog", type=float, default=0.25)
    ap.add_argument("--terrain_k_diff", type=float, default=0.9)
    ap.add_argument("--terrain_k_height", type=float, default=1.0)
    args = ap.parse_args()

    games = discover_games(
        seed=args.seed,
        switch_p=args.switch_p,
        terrain_n=args.terrain_n,
        terrain_fog=args.terrain_fog,
        terrain_k_diff=args.terrain_k_diff,
        terrain_k_height=args.terrain_k_height,
        include_terrain=True,
    )
    games = filter_games(games, args.only)

    for game in games:
        print("\n" + "=" * 72)
        print(f"Game: {game.name} (states={game.n_states}, actions=({game.n_actions_p1},{game.n_actions_p2}))")
        print("=" * 72)

        if game.is_zero_sum and game.n_actions_p1 == 2 and game.n_actions_p2 == 2:
            out = run_minimaxq_selfplay(
                game=game,
                steps=args.steps,
                seed=args.seed,
                alpha=args.alpha,
                gamma=args.gamma,
                eps=args.eps,
            )
            agent1_type = "MinimaxQLearner"
            agent2_type = "MinimaxQLearner"
        else:
            out = run_independentq_selfplay(
                game=game,
                steps=args.steps,
                seed=args.seed,
                alpha=args.alpha,
                gamma=args.gamma,
                eps=args.eps,
            )
            agent1_type = "IndependentQLearner"
            agent2_type = "IndependentQLearner"

        timestamp = now_timestamp()
        run_dir = make_unique_dir(RESULTS_DIR, f"{game.name}_{timestamp}")

        np.savez_compressed(
            run_dir / "data.npz",
            rewards_p1=out["rewards_p1"],
            rewards_p2=out["rewards_p2"],
            states=out["states"],
            actions_p1=out["actions_p1"],
            actions_p2=out["actions_p2"],
        )
        args_dict = dict(vars(args))
        args_dict["algo"] = out["algo"]
        (run_dir / "args.json").write_text(json.dumps(args_dict, indent=2, sort_keys=True), encoding="utf-8")

        write_report_txt(
            run_dir,
            ReportInputs(
                experiment_name="RL vs RL",
                game=game,
                num_rounds=int(args.steps),
                agent1_type=agent1_type,
                agent2_type=agent2_type,
                args=args_dict,
                states=out["states"].astype(int).tolist(),
                actions_p1=out["actions_p1"].astype(int).tolist(),
                actions_p2=out["actions_p2"].astype(int).tolist(),
                payoffs_p1=out["rewards_p1"].astype(float).tolist(),
                payoffs_p2=out["rewards_p2"].astype(float).tolist(),
                final_beliefs_by_state_p1=out["final_beliefs_p1"],
                final_beliefs_by_state_p2=out["final_beliefs_p2"],
                final_regret_p1=out["final_regret_p1"],
                final_regret_p2=out["final_regret_p2"],
            ),
        )
        write_results_csv(run_dir, out["csv_rows"])
        print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()
