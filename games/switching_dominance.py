"""
Compatibility shim for older imports.

Prefer importing from `games/stochastic_switching_dominance.py`, but keep this
module so code that expects `games.switching_dominance` continues to work.
"""

from .stochastic_switching_dominance import StochasticSwitchingDominanceGame

__all__ = ["StochasticSwitchingDominanceGame"]

