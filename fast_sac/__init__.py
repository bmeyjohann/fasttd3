"""
Fast SAC is a high-performance implementation of Soft Actor-Critic (SAC)
for reinforcement learning.
"""

# Core model components
# Use relative imports so the package works when accessed as
# `fasttd3.fast_sac` from the thesis repo without requiring a separate
# editable install of the standalone `fast_sac` package.
from .fast_sac import Actor, Critic
from .fast_sac_utils import EmpiricalNormalization, SimpleReplayBuffer

__all__ = [
    # Core model components
    "Actor",
    "Critic",
    "EmpiricalNormalization",
    "SimpleReplayBuffer",
]
