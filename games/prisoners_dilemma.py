"""Prisoner's Dilemma game implementation."""

import numpy as np


class PrisonersDilemma:
    """
    Prisoner's Dilemma - a canonical game theory problem.
    
    Two players can either Cooperate or Defect.
    Mutual cooperation yields moderate payoffs.
    Defecting against a cooperator yields high payoff.
    Mutual defection yields low payoffs.
    
    Actions:
    - 0: Cooperate
    - 1: Defect
    """
    
    # Payoff matrix for player 1 (rows: player 1 actions, columns: player 2 actions)
    # Standard payoffs: Temptation=5, Reward=3, Punishment=1, Sucker=0
    PAYOFF_MATRIX = np.array([
        [3, 0],  # Player 1 Cooperates vs Player 2 (Coop, Defect)
        [4, 1]   # Player 1 Defects vs Player 2 (Coop, Defect)
    ])
    
    ACTION_NAMES = ["Cooperate", "Defect"]
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
            player1_action: Player 1's action (0: Cooperate, 1: Defect)
            player2_action: Player 2's action (0: Cooperate, 1: Defect)
            
        Returns:
            Player 1's payoff
        """
        return cls.PAYOFF_MATRIX[player1_action, player2_action]
    
    @classmethod
    def get_action_name(cls, action: int) -> str:
        """Get the name of an action."""
        return cls.ACTION_NAMES[action]
