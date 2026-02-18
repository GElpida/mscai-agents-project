"""Game definitions for game theory experiments."""

from .prisoners_dilemma import PrisonersDilemma
from .matching_pennies import MatchingPennies
from .anti_coordination import AntiCoordination
from .almost_rock_paper_scissors import AlmostRockPaperScissors
from .stochastic_switching_dominance import StochasticSwitchingDominanceGame

__all__ = [
    "PrisonersDilemma",
    "MatchingPennies",
    "AntiCoordination",
    "AlmostRockPaperScissors",
    "StochasticSwitchingDominanceGame",
]
