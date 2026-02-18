"""
Script to play Fictitious Play agents against each other.

Two FictitousPlayAgent instances play a game while updating
their beliefs about each other's strategy. Generates reports and visualizations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_fp import FictitousPlayAgent
from games import MatchingPennies, PrisonersDilemma, AntiCoordination, AlmostRockPaperScissors

# Create results directory if it doesn't exist
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_report(game_class, agent1_payoffs, agent2_payoffs, agent1, agent2, num_rounds):
    """
    Save a text report of the game results.
    
    Args:
        game_class: The game class played
        agent1_payoffs: List of agent 1's payoffs per round
        agent2_payoffs: List of agent 2's payoffs per round
        agent1: Agent 1 instance
        agent2: Agent 2 instance
        num_rounds: Number of rounds played
    """
    game_name = game_class.__name__
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = RESULTS_DIR / f"{game_name}_{timestamp}.txt"
    
    history1, opponent_history1 = agent1.get_history()
    history2, opponent_history2 = agent2.get_history()
    
    # Calculate action probabilities
    action_counts1 = np.zeros(game_class.NUM_ACTIONS)
    action_counts2 = np.zeros(game_class.NUM_ACTIONS)
    
    if history1:
        action_counts1 = np.bincount(history1, minlength=game_class.NUM_ACTIONS)
        action_probs1 = action_counts1 / np.sum(action_counts1)
    else:
        action_probs1 = np.ones(game_class.NUM_ACTIONS) / game_class.NUM_ACTIONS
    
    if history2:
        action_counts2 = np.bincount(history2, minlength=game_class.NUM_ACTIONS)
        action_probs2 = action_counts2 / np.sum(action_counts2)
    else:
        action_probs2 = np.ones(game_class.NUM_ACTIONS) / game_class.NUM_ACTIONS
    
    with open(report_path, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"FICTITIOUS PLAY GAME REPORT: {game_name}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"GAME CONFIGURATION\n")
        f.write(f"{'-'*70}\n")
        f.write(f"Number of rounds: {num_rounds}\n")
        f.write(f"Number of actions: {game_class.NUM_ACTIONS}\n")
        f.write(f"Action names: {', '.join(game_class.ACTION_NAMES)}\n\n")
        
        f.write(f"PAYOFF MATRIX\n")
        f.write(f"{'-'*70}\n")
        payoff_matrix = game_class.get_payoff_matrix()
        f.write(str(payoff_matrix) + "\n\n")
        
        f.write(f"RESULTS SUMMARY\n")
        f.write(f"{'-'*70}\n")
        f.write(f"Agent 1 total payoff: {sum(agent1_payoffs)}\n")
        f.write(f"Agent 2 total payoff: {sum(agent2_payoffs)}\n")
        f.write(f"Agent 1 average payoff: {sum(agent1_payoffs)/num_rounds:.4f}\n")
        f.write(f"Agent 2 average payoff: {sum(agent2_payoffs)/num_rounds:.4f}\n\n")
        
        f.write(f"ACTION PROBABILITY DISTRIBUTION\n")
        f.write(f"{'-'*70}\n")
        f.write(f"Agent 1 Action Probabilities:\n")
        for action in range(game_class.NUM_ACTIONS):
            count = int(action_counts1[action]) if len(action_counts1) > action else 0
            prob = action_probs1[action]
            f.write(f"  {game_class.ACTION_NAMES[action]}: {prob:.4f} ({count} plays)\n")
        f.write(f"\nAgent 2 Action Probabilities:\n")
        for action in range(game_class.NUM_ACTIONS):
            count = int(action_counts2[action]) if len(action_counts2) > action else 0
            prob = action_probs2[action]
            f.write(f"  {game_class.ACTION_NAMES[action]}: {prob:.4f} ({count} plays)\n\n")
        
        f.write(f"BELIEF CONVERGENCE\n")
        f.write(f"{'-'*70}\n")
        f.write(f"Agent 1 final belief: {agent1.get_belief().round(4)}\n")
        f.write(f"Agent 2 final belief: {agent2.get_belief().round(4)}\n\n")
        
        f.write(f"ACTION HISTORY\n")
        f.write(f"{'-'*70}\n")
        action_names1 = [game_class.ACTION_NAMES[a] for a in history1]
        action_names2 = [game_class.ACTION_NAMES[a] for a in history2]
        f.write(f"Agent 1: {', '.join(action_names1)}\n")
        f.write(f"Agent 2: {', '.join(action_names2)}\n\n")
        
        f.write(f"PAYOFF HISTORY\n")
        f.write(f"{'-'*70}\n")
        for round_num, (p1, p2) in enumerate(zip(agent1_payoffs, agent2_payoffs), 1):
            f.write(f"Round {round_num:2d}: Agent1={p1:3}, Agent2={p2:3}\n")
    
    return report_path


def get_nash_equilibria(game_class):
    """
    Identify Nash equilibria for the game.
    
    Args:
        game_class: The game class
        
    Returns:
        Dictionary with Nash equilibria information
    """
    payoff_matrix = game_class.get_payoff_matrix()
    num_actions = game_class.NUM_ACTIONS
    
    # Check for pure strategy Nash equilibria
    pure_nashes = []
    for action1 in range(num_actions):
        for action2 in range(num_actions):
            payoff1 = payoff_matrix[action1, action2]
            payoff2 = payoff_matrix[action2, action1]
            
            # Check if it's a Nash equilibrium
            is_nash = True
            for alt_action1 in range(num_actions):
                if payoff_matrix[alt_action1, action2] > payoff1:
                    is_nash = False
                    break
            
            if is_nash:
                for alt_action2 in range(num_actions):
                    if payoff_matrix[action2, alt_action2] > payoff2:
                        is_nash = False
                        break
            
            if is_nash:
                pure_nashes.append((action1, action2))
    
    return {
        'pure_nash': pure_nashes,
        'game_name': game_class.__name__
    }


def calculate_regret(game_class, agent_payoffs, agent_history, opponent_history):
    """
    Calculate cumulative regret for an agent.
    
    Regret at round t = (best payoff against opponent's empirical distribution) - (actual payoff)
    
    Args:
        game_class: The game class
        agent_payoffs: List of agent's payoffs per round
        agent_history: List of agent's actions
        opponent_history: List of opponent's actions
        
    Returns:
        List of cumulative regrets
    """
    payoff_matrix = game_class.get_payoff_matrix()
    num_actions = game_class.NUM_ACTIONS
    
    cumulative_regret = []
    total_regret = 0
    
    for round_num in range(len(agent_payoffs)):
        # Empirical distribution of opponent's actions up to this round
        if round_num == 0:
            opponent_counts = np.ones(num_actions)
        else:
            opponent_counts = np.bincount(opponent_history[:round_num], minlength=num_actions)
            opponent_counts = opponent_counts + 1  # Add pseudocount
        
        opponent_dist = opponent_counts / np.sum(opponent_counts)
        
        # Best payoff against this distribution
        best_payoff = np.max(payoff_matrix @ opponent_dist)
        
        # Regret for this round
        round_regret = best_payoff - agent_payoffs[round_num]
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



    return graph_path


def play_game(game_class, num_rounds: int = 20, verbose: bool = True, strategy_type: str = "mixed"):
    """
    Play two Fictitious Play agents against each other.
    
    Args:
        game_class: The game class to play (e.g., MatchingPennies)
        num_rounds: Number of rounds to play
        verbose: Whether to print round-by-round details
        strategy_type: "pure" for best response, "mixed" for mixed strategy
    """
    
    game_name = game_class.__name__
    payoff_matrix = game_class.get_payoff_matrix()
    num_actions = game_class.NUM_ACTIONS
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Playing {game_name} ({strategy_type.upper()} strategy)")
        print(f"{'='*60}")
        print(f"Number of rounds: {num_rounds}\n")
    
    # Create two agents playing the same game
    # Agent 1 and Agent 2 (Agent 2's payoffs are from their perspective)
    agent1 = FictitousPlayAgent(
        payoff_matrix=payoff_matrix,
        action_space=num_actions,
        opponent_action_space=num_actions,
        strategy_type=strategy_type
    )
    
    agent2 = FictitousPlayAgent(
        payoff_matrix=payoff_matrix,
        action_space=num_actions,
        opponent_action_space=num_actions,
        strategy_type=strategy_type
    )
    
    if verbose:
        print(f"Initial beliefs:")
        print(f"  Agent 1: {agent1.get_belief()}")
        print(f"  Agent 2: {agent2.get_belief()}\n")
    
    agent1_payoffs = []
    agent2_payoffs = []
    
    # Play rounds
    for round_num in range(1, num_rounds + 1):
        # Both agents choose actions based on current beliefs
        action1 = agent1.play()
        action2 = agent2.play()
        
        # Record actions in history
        agent1.play_history.append(action1)
        agent2.play_history.append(action2)
        
        # Get payoffs
        payoff1 = game_class.get_payoff(action1, action2)
        payoff2 = game_class.get_payoff(action2, action1)
        
        agent1_payoffs.append(payoff1)
        agent2_payoffs.append(payoff2)
        
        # Update beliefs based on observed actions
        agent1.observe(action2)
        agent2.observe(action1)
        
        if verbose:
            action_name1 = game_class.ACTION_NAMES[action1] if hasattr(game_class, 'ACTION_NAMES') else str(action1)
            action_name2 = game_class.ACTION_NAMES[action2] if hasattr(game_class, 'ACTION_NAMES') else str(action2)
            print(f"Round {round_num:2d}: "
                  f"Agent1={action_name1:15s} (payoff={payoff1:2}) | "
                  f"Agent2={action_name2:15s} (payoff={payoff2:2}) | "
                  f"Beliefs: A1={agent1.get_belief().round(3)} | A2={agent2.get_belief().round(3)}")
    
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
        history1, history1_opponent = agent1.get_history()
        history2, history2_opponent = agent2.get_history()
        
        print(f"\nAction histories:")
        print(f"  Agent 1 played: {[game_class.ACTION_NAMES[a] if hasattr(game_class, 'ACTION_NAMES') else str(a) for a in history1]}")
        print(f"  Agent 2 played: {[game_class.ACTION_NAMES[a] if hasattr(game_class, 'ACTION_NAMES') else str(a) for a in history2]}")
        
        print(f"Generating report...")
    
    # Save report
    report_path = save_report(game_class, agent1_payoffs, agent2_payoffs, agent1, agent2, num_rounds)
    
    if verbose:
        print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    # Play multiple games with mixed strategy for 1000 rounds
    print("\n" + "="*60)
    print("FICTITIOUS PLAY: Agent vs Agent (Mixed Strategy)")
    print("1000 Rounds per Game")
    print("="*60)
    
    # Example 1: Matching Pennies (zero-sum game)
    play_game(MatchingPennies, num_rounds=1000, strategy_type="mixed")
    
    # Example 2: Prisoner's Dilemma
    play_game(PrisonersDilemma, num_rounds=1000, strategy_type="mixed")
    
    # Example 3: Anti-Coordination
    play_game(AntiCoordination, num_rounds=1000, strategy_type="mixed")
    
    # Example 4: Almost Rock-Paper-Scissors
    play_game(AlmostRockPaperScissors, num_rounds=1000, strategy_type="mixed")
    
    print("\n" + "="*60)
    print("Experiments completed!")
    print("="*60 + "\n")
