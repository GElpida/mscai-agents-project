"""Matching Pennies game implementation."""

import numpy as np


class MatchingPennies:
    """
    Matching Pennies - a zero-sum game.
    
    Player 1 wins if both play the same action, loses if different.
    Player 2 wants different actions.
    
    Actions:
    - 0: Heads
    - 1: Tails
    """
    
    # Payoff matrix for player 1 (rows: player 1 actions, columns: player 2 actions)
    # Player 1 wins (+1) if matching, loses (-1) if not
    PAYOFF_MATRIX = np.array([
        [1, -1],   # Player 1 plays Heads
        [-1, 1]    # Player 1 plays Tails
    ])
    
    ACTION_NAMES = ["Heads", "Tails"]
    NUM_ACTIONS = 2
    
    @classmethod
    def get_payoff_matrix(cls):
        """Get the payoff matrix for player 1."""
        return cls.PAYOFF_MATRIX.copy()
    
    @classmethod
    def get_payoff(cls, player1_action: int, player2_action: int) -> int:
        """
        Get player 1's payoff for given action pair.
        
        Args:
            player1_action: Player 1's action (0: Heads, 1: Tails)
            player2_action: Player 2's action (0: Heads, 1: Tails)
            
        Returns:
            Player 1's payoff
        """
        return cls.PAYOFF_MATRIX[player1_action, player2_action]
    
    @classmethod
    def get_action_name(cls, action: int) -> str:
        """Get the name of an action."""
        return cls.ACTION_NAMES[action]