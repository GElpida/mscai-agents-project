# experiments/main_rl_vs_fp.py
from __future__ import annotations

import argparse
import numpy as np

from agents.independent_q import IndependentQLearner
from agents.fictitious_play import FictitousPlayAgent  # CHANGE to your filename

# CHANGE THIS import to your actual game file/class
from games.switching_dominance import StochasticSwitchingDominanceGame


def run_rl_vs_fp(game, steps: int, seed: int, alpha: float, gamma: float, eps: float, fp_strategy: str):
    nS = game.get_num_states()
    nA = game.get_num_actions()

    # RL: simple independent Q-learning (baseline)
    rl = IndependentQLearner(n_states=nS, n_actions=nA, alpha=alpha, gamma=gamma, eps=eps, seed=seed + 10)

    # FP: one agent per state, each tied to that state's payoff matrix
    fp_agents = []
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

    rewards = np.zeros(steps, dtype=float)
    states = np.zeros(steps, dtype=int)

    for t in range(steps):
        fp = fp_agents[s]  # FP agent for current state

        # RL chooses action a
        a = rl.act(s)

        # FP chooses action b (note: FP agent thinks it's "player 1" in its own code,
        # but for action selection this doesn't matter; it just outputs an action index)
        b = fp.play()

        # environment step: reward is for Player 1 (we choose who is Player 1 here)
        # We'll treat RL as Player 1 => RL reward is r1
        s_next, r1 = game.step(s, a, b)

        # update RL from sample
        rl.update(s, a, r1, s_next)

        # update FP belief with observed opponent action (RL's action)
        # FP is "observing opponent", so pass RL action a
        fp.observe(opponent_action=a)

        rewards[t] = r1
        states[t] = s
        s = s_next

    print("RL vs FP finished.")
    print(f"Mean reward (RL as P1): {rewards.mean():.4f}")
    print(f"Std reward  (RL as P1): {rewards.std():.4f}")
    return {"rewards": rewards, "states": states, "rl": rl, "fp_agents": fp_agents}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--switch_p", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--fp_strategy", type=str, default="pure", choices=["pure", "mixed"])
    args = ap.parse_args()

    game = StochasticSwitchingDominanceGame(switch_p=args.switch_p, seed=args.seed)

    run_rl_vs_fp(
        game=game,
        steps=args.steps,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        eps=args.eps,
        fp_strategy=args.fp_strategy,
    )


if __name__ == "__main__":
    main()
