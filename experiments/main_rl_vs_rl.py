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
from games.stochastic_switching_dominance import StochasticSwitchingDominanceGame
# from games.matching_pennies_env import MatchingPenniesEnv  # if you wrap pennies as env

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "rl_vs_rl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _make_unique_dir(parent: Path, name: str) -> Path:
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


def _save_run(
    *,
    game,
    args: argparse.Namespace,
    rewards: np.ndarray,
    states: np.ndarray,
) -> Path:
    game_name = type(game).__name__
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _make_unique_dir(RESULTS_DIR, f"{game_name}_{timestamp}")

    np.savez_compressed(run_dir / "data.npz", rewards=rewards, states=states)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    report_path = run_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RL vs RL experiment\n")
        f.write(f"Game: {game_name}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write("\n")
        f.write(f"Mean reward P1: {float(np.mean(rewards)):.6f}\n")
        f.write(f"Std reward  P1: {float(np.std(rewards)):.6f}\n")

    return run_dir


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

    out = run_rl_vs_rl(
        game=game,
        steps=args.steps,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        eps=args.eps,
    )

    run_dir = _save_run(game=game, args=args, rewards=out["rewards"], states=out["states"])
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()
