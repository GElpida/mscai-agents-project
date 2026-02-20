# experiments/main_fp_vs_rl.py
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.agent_fp import FictitousPlayAgent
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

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "fp_vs_rl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_fp_vs_rl(game, steps: int, seed: int, alpha: float, gamma: float, eps: float, fp_strategy: str):
    nS = game.n_states
    nA_fp = game.n_actions_p1
    nA_rl = game.n_actions_p2

    # RL (Player 2): simple independent Q-learning (baseline)
    rl = IndependentQLearner(n_states=nS, n_actions=nA_rl, alpha=alpha, gamma=gamma, eps=eps, seed=seed + 10)

    # FP (Player 1): one agent per state, each tied to that state's payoff matrix
    fp_agents = []
    for s in range(nS):
        A_s = game.payoff_matrix_p1(s)  # payoff matrix for player 1, rows=a, cols=b
        fp_agents.append(
            FictitousPlayAgent(
                payoff_matrix=A_s,
                action_space=nA_fp,
                opponent_action_space=nA_rl,
                strategy_type=fp_strategy,
            )
        )

    s = game.reset()

    rewards_p1 = np.zeros(steps, dtype=float)
    rewards_p2 = np.zeros(steps, dtype=float)
    states = np.zeros(steps, dtype=int)
    actions_p1 = np.zeros(steps, dtype=int)
    actions_p2 = np.zeros(steps, dtype=int)

    # Belief for RL (P2): empirical distribution of FP (P1) actions, per state (with pseudocounts)
    counts_fp_by_state = [np.ones(nA_fp, dtype=float) for _ in range(nS)]

    action_visits_p1 = np.zeros(nA_fp, dtype=int)
    action_visits_p2 = np.zeros(nA_rl, dtype=int)
    regret_p1 = 0.0
    regret_p2 = 0.0
    csv_rows = []

    for t in range(steps):
        fp = fp_agents[s]  # FP agent for current state

        A_s = np.asarray(game.payoff_matrix_p1(s), dtype=float)
        B_s = np.asarray(game.payoff_matrix_p2(s), dtype=float)

        belief_fp_over_rl = fp.get_belief()
        belief_rl_over_fp = counts_fp_by_state[s] / float(np.sum(counts_fp_by_state[s]))

        # FP chooses action a
        a = fp.play()

        # RL chooses action b
        b = rl.act(s)

        exp1 = float(A_s[int(a)] @ belief_fp_over_rl)
        exp2 = float(B_s[int(b)] @ belief_rl_over_fp)

        # environment step: r1 is FP payoff (P1), r2 is RL payoff (P2)
        s_next, r1, r2 = game.step(s, a, b)

        # update RL from sample
        rl.update(s, b, r2, s_next)

        # update FP belief with observed opponent action (RL's action b)
        fp.observe(opponent_action=b)

        # Update RL belief counts for FP (P1) actions
        counts_fp_by_state[s][int(a)] += 1.0

        # Regret (cumulative): best expected response vs current belief minus realized payoff
        best1 = float(np.max(A_s @ belief_fp_over_rl))
        best2 = float(np.max(B_s @ belief_rl_over_fp))
        regret_p1 += best1 - float(r1)
        regret_p2 += best2 - float(r2)

        action_visits_p1[int(a)] += 1
        action_visits_p2[int(b)] += 1
        v1 = int(action_visits_p1[int(a)])
        v2 = int(action_visits_p2[int(b)])

        csv_rows.append(
            build_csv_row(
                round_idx=t + 1,
                state=int(s),
                agent1_type="FictitiousPlayAgent",
                agent2_type="IndependentQLearner",
                agent1_exp_payoff=exp1,
                agent2_exp_payoff=exp2,
                agent1_payoff=float(r1),
                agent2_payoff=float(r2),
                agent1_action=int(a),
                agent2_action=int(b),
                agent1_regret=float(regret_p1),
                agent2_regret=float(regret_p2),
                agent1_belief=belief_fp_over_rl,
                agent2_belief=belief_rl_over_fp,
                agent1_visits=v1,
                agent2_visits=v2,
            )
        )

        rewards_p1[t] = float(r1)
        rewards_p2[t] = float(r2)
        states[t] = s
        actions_p1[t] = int(a)
        actions_p2[t] = int(b)
        s = s_next

    print("FP vs RL finished.")
    print(f"Mean reward (FP as P1): {rewards_p1.mean():.4f}")
    print(f"Std reward  (FP as P1): {rewards_p1.std():.4f}")

    final_beliefs_p1 = [fp_agents[s].get_belief() for s in range(nS)]
    final_beliefs_p2 = [c / float(np.sum(c)) for c in counts_fp_by_state]

    return {
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
        "rl": rl,
        "fp_agents": fp_agents,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--switch_p", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--fp_strategy", type=str, default="pure", choices=["pure", "mixed"])
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

        out = run_fp_vs_rl(
            game=game,
            steps=args.steps,
            seed=args.seed,
            alpha=args.alpha,
            gamma=args.gamma,
            eps=args.eps,
            fp_strategy=args.fp_strategy,
        )

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
        (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

        write_report_txt(
            run_dir,
            ReportInputs(
                experiment_name="FP vs RL",
                game=game,
                num_rounds=int(args.steps),
                agent1_type="FictitiousPlayAgent",
                agent2_type="IndependentQLearner",
                args=vars(args),
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
