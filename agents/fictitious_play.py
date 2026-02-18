"""
Compatibility shim for older imports.

Prefer importing from `agents/agent_fp.py`, but keep this module so code that
expects `agents.fictitious_play` continues to work.
"""

from .agent_fp import FictitousPlayAgent, FictitiousPlayAgent

__all__ = ["FictitousPlayAgent", "FictitiousPlayAgent"]

