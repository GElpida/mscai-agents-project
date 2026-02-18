"""
Fictitious Play Agent

An agent that plays fictitious play, a learning algorithm where:
1. Initialize beliefs about the opponent's strategy
2. Play a best response to the assessed strategy
3. Observe the opponent's actual play and update beliefs
4. Repeat steps 2-3
"""

import numpy as np
from typing import Tuple, Dict, Optional


class FictitousPlayAgent:
    """
    A game-playing agent using the Fictitious Play algorithm.
    
    The agent maintains beliefs about the opponent's strategy distribution
    and plays best responses based on these beliefs, updating them after
    each observation of the opponent's play.
    """
    
    def __init__(
        self,
        payoff_matrix: np.ndarray,
        action_space: int,
        opponent_action_space: int,
        initial_belief: Optional[np.ndarray] = None,
        strategy_type: str = "pure"
    ):
        """
        Initialize the Fictitious Play agent.
        
        Args:
            payoff_matrix: 2D array where payoff_matrix[i, j] is the agent's payoff
                          when agent plays action i and opponent plays action j
            action_space: Number of actions available to the agent
            opponent_action_space: Number of actions available to the opponent
            initial_belief: Initial belief distribution over opponent's actions
                           (default: uniform distribution)
            strategy_type: "pure" for best response, "mixed" for mixed strategy
                          (default: "pure")
        """
        self.payoff_matrix = payoff_matrix
        self.action_space = action_space
        self.opponent_action_space = opponent_action_space
        self.strategy_type = strategy_type
        
        # Initialize beliefs (step 1)
        if initial_belief is None:
            self.belief = np.ones(opponent_action_space) / opponent_action_space
        else:
            self.belief = initial_belief / np.sum(initial_belief)
        
        # Track observation counts for updating beliefs
        self.opponent_action_counts = np.ones(opponent_action_space)
        self.play_history = []
        self.opponent_history = []
        
        # Temperature parameter for softmax mixed strategy
        self.temperature = 0.1
        
    def _compute_best_response(self) -> int:
        """
        Compute best response action to current belief about opponent strategy (step 2).
        
        Returns:
            The action index that maximizes expected payoff against current belief
        """
        # Expected payoff for each action: payoff_matrix @ belief
        expected_payoffs = self.payoff_matrix @ self.belief
        best_action = np.argmax(expected_payoffs)
        return best_action
    
    def _compute_mixed_strategy(self) -> np.ndarray:
        """
        Compute mixed strategy using softmax over expected payoffs.
        
        Uses temperature-scaled softmax: p(action) ∝ exp(payoff / temperature)
        
        Returns:
            Probability distribution over actions
        """
        expected_payoffs = self.payoff_matrix @ self.belief
        
        # Shift payoffs for numerical stability
        shifted_payoffs = expected_payoffs - np.max(expected_payoffs)
        
        # Compute softmax with temperature
        exp_payoffs = np.exp(shifted_payoffs / self.temperature)
        mixed_strategy = exp_payoffs / np.sum(exp_payoffs)
        
        return mixed_strategy
    
    def play(self) -> int:
        """
        Select action to play using fictitious play.
        
        Returns:
            The action to play
        """
        if self.strategy_type == "pure":
            action = self._compute_best_response()
        elif self.strategy_type == "mixed":
            mixed_strategy = self._compute_mixed_strategy()
            action = np.random.choice(self.action_space, p=mixed_strategy)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
        
        return action
    
    def observe(self, opponent_action: int) -> None:
        """
        Observe the opponent's action and update beliefs (step 3).
        
        Args:
            opponent_action: The action index the opponent played (0 to opponent_action_space-1)
        """
        if not (0 <= opponent_action < self.opponent_action_space):
            raise ValueError(
                f"opponent_action must be in [0, {self.opponent_action_space-1}]"
            )
        
        # Update belief based on observed action
        self.opponent_action_counts[opponent_action] += 1
        self.belief = self.opponent_action_counts / np.sum(self.opponent_action_counts)
        self.opponent_history.append(opponent_action)
    
    def round(self, opponent_action: int) -> int:
        """
        Execute a complete round: play and observe opponent's response (steps 2-3).
        
        Args:
            opponent_action: The action the opponent is playing in this round
            
        Returns:
            The action this agent played
        """
        # Step 2: Play best response
        action = self.play()
        self.play_history.append(action)
        
        # Step 3: Observe and update
        self.observe(opponent_action)
        
        return action
    
    def reset(self, initial_belief: Optional[np.ndarray] = None) -> None:
        """
        Reset the agent's beliefs and history for a new game.
        
        Args:
            initial_belief: New initial belief distribution (default: uniform)
        """
        if initial_belief is None:
            self.belief = np.ones(self.opponent_action_space) / self.opponent_action_space
        else:
            self.belief = initial_belief / np.sum(initial_belief)
        
        self.opponent_action_counts = np.ones(self.opponent_action_space)
        self.play_history = []
        self.opponent_history = []
    
    def get_belief(self) -> np.ndarray:
        """Get current belief distribution over opponent's actions."""
        return self.belief.copy()
    
    def get_mixed_strategy(self) -> np.ndarray:
        """Get current mixed strategy probability distribution."""
        return self._compute_mixed_strategy()
    
    def get_history(self) -> Tuple[list, list]:
        """Get the play history of both agent and opponent."""
        return self.play_history.copy(), self.opponent_history.copy()


# Backwards/typo-compatible alias (more conventional spelling)
FictitiousPlayAgent = FictitousPlayAgent
