"""
Compatibility shim for older imports.

Prefer importing from `agents/agent_rl_q.py`, but keep this module so code that
expects `agents.independent_q` continues to work.
"""

from .agent_rl_q import IndependentQLearner

__all__ = ["IndependentQLearner"]

