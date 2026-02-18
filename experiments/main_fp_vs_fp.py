"""
Script to play Fictitious Play agents against each other.

Two FictitousPlayAgent instances play a game while updating
their beliefs about each other's strategy. Generates reports.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_fp import FictitousPlayAgent
from experiments.game_registry import discover_games, filter_games
from experiments.output_artifacts import (
    ReportInputs,
    build_csv_row,
    make_unique_dir,
    now_timestamp,
    write_report_txt,
    write_results_csv,
)

# Create results directory if it doesn't exist
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "fp_vs_fp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _default_action_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(int(n))]


def _format_matrix_for_report(M: np.ndarray, *, max_cells: int = 100) -> str:
    M = np.asarray(M, dtype=float)
    cells = int(M.size)
    if cells <= max_cells:
        return str(M)
    return (
        f"<matrix shape={M.shape}, min={float(np.min(M)):.3f}, "
        f"max={float(np.max(M)):.3f}, mean={float(np.mean(M)):.3f}>"
    )


def get_pure_nash_equilibria(A_p1: np.ndarray, B_p2: np.ndarray) -> list[tuple[int, int]]:
    """
    Pure-strategy Nash equilibria for a 2-player normal-form game.

    A_p1: shape (m, n) with payoff to player 1 for (a, b)
    B_p2: shape (n, m) with payoff to player 2 for (b, a)
    Returns list of (a, b) indices.
    """
    A_p1 = np.asarray(A_p1, dtype=float)
    B_p2 = np.asarray(B_p2, dtype=float)
    m, n = A_p1.shape
    if B_p2.shape != (n, m):
        raise ValueError(f"Expected B_p2 shape {(n, m)} for A_p1 shape {(m, n)}, got {B_p2.shape}")

    pure_nashes: list[tuple[int, int]] = []
    for a in range(m):
        for b in range(n):
            p1_best = A_p1[a, b] >= np.max(A_p1[:, b])
            p2_best = B_p2[b, a] >= np.max(B_p2[:, a])
            if p1_best and p2_best:
                pure_nashes.append((a, b))
    return pure_nashes


def calculate_regret(payoff_matrix: np.ndarray, agent_payoffs, opponent_history, opponent_action_space: int):
    """
    Cumulative regret for an agent with payoff_matrix (rows: agent actions, cols: opponent actions).
    """
    payoff_matrix = np.asarray(payoff_matrix, dtype=float)
    cumulative_regret = []
    total_regret = 0.0

    for round_num in range(len(agent_payoffs)):
        if round_num == 0:
            opponent_counts = np.ones(opponent_action_space, dtype=float)
        else:
            opponent_counts = np.bincount(opponent_history[:round_num], minlength=opponent_action_space).astype(float)
            opponent_counts = opponent_counts + 1.0  # pseudocount
        opponent_dist = opponent_counts / float(np.sum(opponent_counts))
        best_payoff = float(np.max(payoff_matrix @ opponent_dist))
        round_regret = best_payoff - float(agent_payoffs[round_num])
        total_regret += round_regret
        cumulative_regret.append(total_regret)

    return cumulative_regret


def calculate_empirical_distribution(history, num_actions):
    """
    Calculate empirical distribution of actions over time.
    
    Args:
        history: List of actions
        num_actions: Number of possible actions
        
    Returns:
        Array of shape (len(history)+1, num_actions) showing distribution evolution
    """
    distributions = []
    
    for i in range(len(history) + 1):
        if i == 0:
            dist = np.ones(num_actions) / num_actions
        else:
            counts = np.bincount(history[:i], minlength=num_actions)
            counts = counts + 1  # Add pseudocount
            dist = counts / np.sum(counts)
        distributions.append(dist)
    
    return np.array(distributions)


def play_game(game, num_rounds: int = 20, verbose: bool = True, strategy_type: str = "mixed"):
    """
    Play two Fictitious Play agents against each other.
    
    Args:
        game: Two-player game adapter to play
        num_rounds: Number of rounds to play
        verbose: Whether to print round-by-round details
        strategy_type: "pure" for best response, "mixed" for mixed strategy
    """
    
    game_name = game.name
    num_actions_p1 = game.n_actions_p1
    num_actions_p2 = game.n_actions_p2

    agent1_by_state = []
    agent2_by_state = []
    for s in range(game.n_states):
        A_s = game.payoff_matrix_p1(s)
        B_s = game.payoff_matrix_p2(s)
        agent1_by_state.append(
            FictitousPlayAgent(
                payoff_matrix=A_s,
                action_space=num_actions_p1,
                opponent_action_space=num_actions_p2,
                strategy_type=strategy_type,
            )
        )
        agent2_by_state.append(
            FictitousPlayAgent(
                payoff_matrix=B_s,
                action_space=num_actions_p2,
                opponent_action_space=num_actions_p1,
                strategy_type=strategy_type,
            )
        )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Playing {game_name} ({strategy_type.upper()} strategy)")
        print(f"{'='*60}")
        print(f"Number of rounds: {num_rounds}\n")
    
    if verbose:
        print(f"Initial beliefs:")
        print(f"  Agent 1: {agent1_by_state[0].get_belief()}")
        print(f"  Agent 2: {agent2_by_state[0].get_belief()}\n")
    
    agent1_payoffs = []
    agent2_payoffs = []
    actions_p1 = []
    actions_p2 = []
    states = []
    csv_rows = []

    action_visits_p1 = np.zeros(game.n_actions_p1, dtype=int)
    action_visits_p2 = np.zeros(game.n_actions_p2, dtype=int)
    regret_p1 = 0.0
    regret_p2 = 0.0

    s = game.reset()
    
    # Play rounds
    for round_num in range(1, num_rounds + 1):
        agent1 = agent1_by_state[s]
        agent2 = agent2_by_state[s]

        A_s = np.asarray(game.payoff_matrix_p1(s), dtype=float)
        B_s = np.asarray(game.payoff_matrix_p2(s), dtype=float)
        belief1 = agent1.get_belief()  # over P2 actions
        belief2 = agent2.get_belief()  # over P1 actions

        # Both agents choose actions based on current beliefs
        action1 = agent1.play()
        action2 = agent2.play()
        
        # Record actions in history
        agent1.play_history.append(action1)
        agent2.play_history.append(action2)
        
        # Expected payoff of chosen actions (vs current beliefs)
        exp1 = float(A_s[int(action1)] @ belief1)
        exp2 = float(B_s[int(action2)] @ belief2)

        # Environment step
        s_next, payoff1, payoff2 = game.step(s, action1, action2)
        
        agent1_payoffs.append(payoff1)
        agent2_payoffs.append(payoff2)
        actions_p1.append(int(action1))
        actions_p2.append(int(action2))
        states.append(int(s))

        # Regret (cumulative): best expected response vs current belief minus realized payoff
        best1 = float(np.max(A_s @ belief1))
        best2 = float(np.max(B_s @ belief2))
        regret_p1 += best1 - float(payoff1)
        regret_p2 += best2 - float(payoff2)

        # Visits (how many times this chosen action has been played so far)
        action_visits_p1[int(action1)] += 1
        action_visits_p2[int(action2)] += 1
        v1 = int(action_visits_p1[int(action1)])
        v2 = int(action_visits_p2[int(action2)])

        csv_rows.append(
            build_csv_row(
                round_idx=round_num,
                agent1_type="FictitiousPlayAgent",
                agent2_type="FictitiousPlayAgent",
                agent1_exp_payoff=exp1,
                agent2_exp_payoff=exp2,
                agent1_payoff=float(payoff1),
                agent2_payoff=float(payoff2),
                agent1_action=int(action1),
                agent2_action=int(action2),
                agent1_regret=float(regret_p1),
                agent2_regret=float(regret_p2),
                agent1_belief=belief1,
                agent2_belief=belief2,
                agent1_visits=v1,
                agent2_visits=v2,
            )
        )
        
        # Update beliefs based on observed actions
        agent1.observe(action2)
        agent2.observe(action1)

        s = int(s_next)
        
        if verbose:
            action_names_p1 = getattr(game, "action_names_p1", None) or _default_action_names("a", num_actions_p1)
            action_names_p2 = getattr(game, "action_names_p2", None) or _default_action_names("b", num_actions_p2)
            action_name1 = action_names_p1[action1] if action1 < len(action_names_p1) else str(action1)
            action_name2 = action_names_p2[action2] if action2 < len(action_names_p2) else str(action2)
            print(f"Round {round_num:2d}: "
                  f"Agent1={action_name1:15s} (payoff={payoff1:2}) | "
                  f"Agent2={action_name2:15s} (payoff={payoff2:2}) | "
                  f"Beliefs: A1={agent1.get_belief().round(3)} | A2={agent2.get_belief().round(3)} | "
                  f"State={states[-1]}")
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("Game Summary:")
        print(f"{'='*60}")
        print(f"Total rounds played: {num_rounds}")
        print(f"Agent 1 total payoff: {sum(agent1_payoffs)}")
        print(f"Agent 2 total payoff: {sum(agent2_payoffs)}")
        print(f"Agent 1 average payoff: {sum(agent1_payoffs)/num_rounds:.3f}")
        print(f"Agent 2 average payoff: {sum(agent2_payoffs)/num_rounds:.3f}")
        
        print(f"\nFinal beliefs:")
        print(f"  Agent 1: {agent1.get_belief().round(4)}")
        print(f"  Agent 2: {agent2.get_belief().round(4)}")
        
        # Get action histories
        history1, _ = agent1_by_state[0].get_history()
        history2, _ = agent2_by_state[0].get_history()
        
        print(f"\nAction histories:")
        action_names_p1 = getattr(game, "action_names_p1", None) or _default_action_names("a", num_actions_p1)
        action_names_p2 = getattr(game, "action_names_p2", None) or _default_action_names("b", num_actions_p2)
        print(f"  Agent 1 played: {[action_names_p1[a] if a < len(action_names_p1) else str(a) for a in history1]}")
        print(f"  Agent 2 played: {[action_names_p2[a] if a < len(action_names_p2) else str(a) for a in history2]}")
        
        print(f"Generating report...")
    
    timestamp = now_timestamp()
    run_dir = make_unique_dir(RESULTS_DIR, f"{game_name}_{timestamp}")

    report_path = write_report_txt(
        run_dir,
        ReportInputs(
            experiment_name="Fictitious Play",
            game=game,
            num_rounds=int(num_rounds),
            agent1_type="FictitiousPlayAgent",
            agent2_type="FictitiousPlayAgent",
            args={"strategy_type": strategy_type},
            states=states,
            actions_p1=actions_p1,
            actions_p2=actions_p2,
            payoffs_p1=[float(x) for x in agent1_payoffs],
            payoffs_p2=[float(x) for x in agent2_payoffs],
            final_beliefs_by_state_p1=[a.get_belief() for a in agent1_by_state],
            final_beliefs_by_state_p2=[a.get_belief() for a in agent2_by_state],
            final_regret_p1=float(regret_p1),
            final_regret_p2=float(regret_p2),
        ),
    )
    write_results_csv(run_dir, csv_rows)
    
    if verbose:
        print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--switch_p", type=float, default=0.2)
    ap.add_argument("--strategy_type", type=str, default="mixed", choices=["pure", "mixed"])
    ap.add_argument("--only", action="append", default=None, help="Run only these game names (repeatable).")
    ap.add_argument("--terrain_n", type=int, default=4)
    ap.add_argument("--terrain_fog", type=float, default=0.25)
    ap.add_argument("--terrain_k_diff", type=float, default=0.9)
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("FICTITIOUS PLAY: Agent vs Agent")
    print(f"{args.rounds} Rounds per Game")
    print("=" * 60)

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
        play_game(game, num_rounds=args.rounds, strategy_type=args.strategy_type)

    print("\n" + "=" * 60)
    print("Experiments completed!")
    print("=" * 60 + "\n")
