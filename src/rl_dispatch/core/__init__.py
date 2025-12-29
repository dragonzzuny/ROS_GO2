"""
Core data structures and type definitions for the RL Dispatch system.

This module provides the fundamental building blocks for representing:
- Robot states and observations
- Actions (dispatch decisions and rescheduling strategies)
- Events detected by CCTV systems
- Patrol points and routes
- Environment configurations
"""

from rl_dispatch.core.types import (
    Event,
    PatrolPoint,
    RobotState,
    Action,
    ActionMode,
    State,
    Observation,
    Candidate,
    RewardComponents,
    EpisodeMetrics,
)
from rl_dispatch.core.config import EnvConfig, TrainingConfig, RewardConfig

__all__ = [
    # Data structures
    "Event",
    "PatrolPoint",
    "RobotState",
    "Action",
    "ActionMode",
    "State",
    "Observation",
    "Candidate",
    "RewardComponents",
    "EpisodeMetrics",
    # Configuration
    "EnvConfig",
    "TrainingConfig",
    "RewardConfig",
]
