"""
Reward calculation for the RL Dispatch system.

This module implements the multi-component reward function that guides
the RL policy's learning. The reward balances four objectives:

1. Event Response (R^evt): Encourage timely event handling
2. Patrol Coverage (R^pat): Maintain regular patrol coverage
3. Safety (R^safe): Avoid collisions and navigation failures
4. Efficiency (R^eff): Minimize unnecessary travel

The total reward is a weighted sum: R = w1*R^evt + w2*R^pat + w3*R^safe + w4*R^eff
"""

from rl_dispatch.rewards.reward_calculator import RewardCalculator

__all__ = ["RewardCalculator"]
