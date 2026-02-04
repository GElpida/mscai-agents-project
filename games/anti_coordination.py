"""Anti-Coordination game implementation."""

import numpy as np


class AntiCoordination:
    """
    Anti-Coordination (also called Hawk-Dove or Mismatch) game.
    
    Players prefer to play different actions from each other.
    Each player gets rewarded when taking a different action than opponent.
    
    Actions:
    - 0: Strategy A
    - 1: Strategy B
    """
    
    # Payoff matrix for player 1
    # Player 1 gets 1 if different from opponent, 0 if same
    PAYOFF_MATRIX = np.array([
        [0, 1],    # Player 1 plays Strategy A
        [1, 0]     # Player 1 plays Strategy B
    ])
    
    ACTION_NAMES = ["Strategy A", "Strategy B"]
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
            player1_action: Player 1's action (0: Strategy A, 1: Strategy B)
            player2_action: Player 2's action (0: Strategy A, 1: Strategy B)
            
        Returns:
            Player 1's payoff
        """
        return cls.PAYOFF_MATRIX[player1_action, player2_action]
    
    @classmethod
    def get_action_name(cls, action: int) -> str:
        """Get the name of an action."""
        return cls.ACTION_NAMES[action]
