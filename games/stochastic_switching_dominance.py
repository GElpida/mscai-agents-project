import numpy as np


class StochasticSwitchingDominanceGame:

    ACTION_NAMES = ["A", "B"]
    NUM_ACTIONS = 2
    NUM_STATES = 2

    def __init__(self, switch_p=0.2, seed=None):
        self.rng = np.random.default_rng(seed)
        self.switch_p = switch_p

        # Payoff matrices per state (Player 1)
        self.A = np.array([
            [[ 2, -1],
             [-1,  1]],

            [[ 1, -2],
             [-2,  2]],
        ], dtype=float)

    # ----- General API -----

    def get_num_states(self):
        return self.NUM_STATES

    def get_num_actions(self):
        return self.NUM_ACTIONS

    def get_payoff_matrix(self, state):
        return self.A[state].copy()

    def get_payoff(self, state, a, b):
        return self.A[state, a, b]

    # ----- RL API -----

    def reset(self):
        return 0  # always start in state 0

    def step(self, state, a, b):
        r1 = self.A[state, a, b]

        # stochastic switching
        if self.rng.random() < self.switch_p:
            next_state = 1 - state
        else:
            next_state = state

        return next_state, r1
