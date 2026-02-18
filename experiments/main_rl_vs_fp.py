# experiments/main_rl_vs_fp.py
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

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "rl_vs_fp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_rl_vs_fp(game, steps: int, seed: int, alpha: float, gamma: float, eps: float, fp_strategy: str):
    nS = game.n_states
    nA_rl = game.n_actions_p1
    nA_fp = game.n_actions_p2

    # RL: simple independent Q-learning (baseline)
    rl = IndependentQLearner(n_states=nS, n_actions=nA_rl, alpha=alpha, gamma=gamma, eps=eps, seed=seed + 10)

    # FP: one agent per state, each tied to that state's payoff matrix
    fp_agents = []
    for s in range(nS):
        B_s = game.payoff_matrix_p2(s)  # payoff matrix for player 2, rows=b, cols=a
        fp_agents.append(
            FictitousPlayAgent(
                payoff_matrix=B_s,
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

    # Belief for RL: empirical distribution of FP actions, per state (with pseudocounts)
    counts_fp_by_state = [np.ones(nA_fp, dtype=float) for _ in range(nS)]

    action_visits_p1 = np.zeros(nA_rl, dtype=int)
    action_visits_p2 = np.zeros(nA_fp, dtype=int)
    regret_p1 = 0.0
    regret_p2 = 0.0
    csv_rows = []

    for t in range(steps):
        fp = fp_agents[s]  # FP agent for current state

        A_s = np.asarray(game.payoff_matrix_p1(s), dtype=float)
        B_s = np.asarray(game.payoff_matrix_p2(s), dtype=float)

        belief_rl_over_fp = counts_fp_by_state[s] / float(np.sum(counts_fp_by_state[s]))
        belief_fp_over_rl = fp.get_belief()

        # RL chooses action a
        a = rl.act(s)

        # FP chooses action b (note: FP agent thinks it's "player 1" in its own code,
        # but for action selection this doesn't matter; it just outputs an action index)
        b = fp.play()

        exp1 = float(A_s[int(a)] @ belief_rl_over_fp)
        exp2 = float(B_s[int(b)] @ belief_fp_over_rl)

        # environment step: reward is for Player 1 (we choose who is Player 1 here)
        # We'll treat RL as Player 1 => RL reward is r1
        s_next, r1, r2 = game.step(s, a, b)

        # update RL from sample
        rl.update(s, a, r1, s_next)

        # update FP belief with observed opponent action (RL's action)
        # FP is "observing opponent", so pass RL action a
        fp.observe(opponent_action=a)

        # Update RL belief counts for FP actions
        counts_fp_by_state[s][int(b)] += 1.0

        # Regret (cumulative): best expected response vs current belief minus realized payoff
        best1 = float(np.max(A_s @ belief_rl_over_fp))
        best2 = float(np.max(B_s @ belief_fp_over_rl))
        regret_p1 += best1 - float(r1)
        regret_p2 += best2 - float(r2)

        action_visits_p1[int(a)] += 1
        action_visits_p2[int(b)] += 1
        v1 = int(action_visits_p1[int(a)])
        v2 = int(action_visits_p2[int(b)])

        csv_rows.append(
            build_csv_row(
                round_idx=t + 1,
                agent1_type="IndependentQLearner",
                agent2_type="FictitiousPlayAgent",
                agent1_exp_payoff=exp1,
                agent2_exp_payoff=exp2,
                agent1_payoff=float(r1),
                agent2_payoff=float(r2),
                agent1_action=int(a),
                agent2_action=int(b),
                agent1_regret=float(regret_p1),
                agent2_regret=float(regret_p2),
                agent1_belief=belief_rl_over_fp,
                agent2_belief=belief_fp_over_rl,
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

    print("RL vs FP finished.")
    print(f"Mean reward (RL as P1): {rewards_p1.mean():.4f}")
    print(f"Std reward  (RL as P1): {rewards_p1.std():.4f}")

    final_beliefs_p1 = [c / float(np.sum(c)) for c in counts_fp_by_state]
    final_beliefs_p2 = [fp_agents[s].get_belief() for s in range(nS)]

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
    args = ap.parse_args()

    games = discover_games(
        seed=args.seed,
        switch_p=args.switch_p,
        terrain_n=args.terrain_n,
        terrain_fog=args.terrain_fog,
        terrain_k_diff=args.terrain_k_diff,
        include_terrain=True,
    )
    games = filter_games(games, args.only)

    for game in games:
        print("\n" + "=" * 72)
        print(f"Game: {game.name} (states={game.n_states}, actions=({game.n_actions_p1},{game.n_actions_p2}))")
        print("=" * 72)

        out = run_rl_vs_fp(
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
                experiment_name="RL vs FP",
                game=game,
                num_rounds=int(args.steps),
                agent1_type="IndependentQLearner",
                agent2_type="FictitiousPlayAgent",
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
