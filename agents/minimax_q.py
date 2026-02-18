"""
Compatibility shim for older imports.

Prefer importing from `agents/agent_rl_minimaxq.py`, but keep this module so code
that expects `agents.minimax_q` continues to work.
"""

from .agent_rl_minimaxq import MinimaxQLearner

__all__ = ["MinimaxQLearner"]

