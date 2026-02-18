# experiments/main_rl_vs_rl.py
from __future__ import annotations

import argparse
import numpy as np

from agents.minimax_q import MinimaxQLearner

# CHANGE THIS import to your actual game file/class
from games.switching_dominance import StochasticSwitchingDominanceGame
# from games.matching_pennies_env import MatchingPenniesEnv  # if you wrap pennies as env


def run_rl_vs_rl(game, steps: int, seed: int, alpha: float, gamma: float, eps: float):
    nS = game.get_num_states()
    nA = game.get_num_actions()

    if nA != 2:
        raise ValueError("This Minimax-Q implementation currently supports 2 actions only.")

    p1 = MinimaxQLearner(n_states=nS, n_actions=nA, alpha=alpha, gamma=gamma, eps=eps, seed=seed + 1)
    p2 = MinimaxQLearner(n_states=nS, n_actions=nA, alpha=alpha, gamma=gamma, eps=eps, seed=seed + 2)

    s = game.reset()

    rewards = np.zeros(steps, dtype=float)
    states = np.zeros(steps, dtype=int)

    for t in range(steps):
        a = p1.act(s)
        b = p2.act(s)

        s_next, r1 = game.step(s, a, b)  # reward for player 1
        r2 = -r1                         # zero-sum

        # Update P1: my action=a, opp=b
        p1.update(s, a, b, r1, s_next)

        # Update P2: for P2, "my action" is b, opponent is a
        p2.update(s, b, a, r2, s_next)

        rewards[t] = r1
        states[t] = s
        s = s_next

    print("RL vs RL finished.")
    print(f"Mean reward P1: {rewards.mean():.4f}  (should be near minimax value over time)")
    print(f"Std reward  P1: {rewards.std():.4f}")
    return {"rewards": rewards, "states": states, "p1": p1, "p2": p2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--switch_p", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=0.1)
    args = ap.parse_args()

    game = StochasticSwitchingDominanceGame(switch_p=args.switch_p, seed=args.seed)

    run_rl_vs_rl(
        game=game,
        steps=args.steps,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        eps=args.eps,
    )


if __name__ == "__main__":
    main()
