"""
Patrol route planning and candidate generation.

This module provides candidate generation strategies for patrol route rescheduling.
Each strategy implements a different heuristic for determining patrol point visit order.

The candidate-based approach reduces the action space from M! (all permutations) to
K (number of strategies), making the problem tractable while maintaining solution quality.
"""

from rl_dispatch.planning.candidate_generator import (
    CandidateGenerator,
    KeepOrderGenerator,
    NearestFirstGenerator,
    MostOverdueFirstGenerator,
    OverdueETABalanceGenerator,
    RiskWeightedGenerator,
    BalancedCoverageGenerator,
    CandidateFactory,
)

__all__ = [
    "CandidateGenerator",
    "KeepOrderGenerator",
    "NearestFirstGenerator",
    "MostOverdueFirstGenerator",
    "OverdueETABalanceGenerator",
    "RiskWeightedGenerator",
    "BalancedCoverageGenerator",
    "CandidateFactory",
]
