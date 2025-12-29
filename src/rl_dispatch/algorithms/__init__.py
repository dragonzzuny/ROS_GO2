"""
Reinforcement learning algorithms and neural networks.

This module provides:
- PPO actor-critic network architecture
- PPO training algorithm
- Baseline policies for comparison
- Experience buffer and GAE computation
"""

from rl_dispatch.algorithms.networks import ActorCriticNetwork
from rl_dispatch.algorithms.ppo import PPOAgent

__all__ = [
    "ActorCriticNetwork",
    "PPOAgent",
]
