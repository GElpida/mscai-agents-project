"""Almost Rock-Paper-Scissors game implementation."""

import numpy as np


class AlmostRockPaperScissors:
    """
    Almost Rock-Paper-Scissors - a 3-action game with mixed strategy equilibria.
    
    A variant of rock-paper-scissors with slightly modified payoffs
    to create interesting learning dynamics.
    
    Actions:
    - 0: Rock
    - 1: Paper
    - 2: Scissors
    """
    
    # Payoff matrix for player 1 (3x3)
    # Rock beats Scissors, Paper beats Rock, Scissors beats Paper
    # Win: 2, Tie: 1, Loss: 0
    PAYOFF_MATRIX = np.array([
        [0, 0, 1],    # Player 1 plays Rock vs (Rock, Paper, Scissors)
        [1, 0, 0],    # Player 1 plays Paper vs (Rock, Paper, Scissors)
        [0, 1, 0]     # Player 1 plays Scissors vs (Rock, Paper, Scissors)
    ])
    
    ACTION_NAMES = ["Rock", "Paper", "Scissors"]
    NUM_ACTIONS = 3
    
    @classmethod
    def get_payoff_matrix(cls):
        """Get the payoff matrix for player 1."""
        return cls.PAYOFF_MATRIX.copy()
    
    @classmethod
    def get_payoff(cls, player1_action: int, player2_action: int) -> int:
        """
        Get player 1's payoff for given action pair.
        
        Args:
            player1_action: Player 1's action (0: Rock, 1: Paper, 2: Scissors)
            player2_action: Player 2's action (0: Rock, 1: Paper, 2: Scissors)
            
        Returns:
            Player 1's payoff
        """
        return cls.PAYOFF_MATRIX[player1_action, player2_action]
    
    @classmethod
    def get_action_name(cls, action: int) -> str:
        """Get the name of an action."""
        return cls.ACTION_NAMES[action]
