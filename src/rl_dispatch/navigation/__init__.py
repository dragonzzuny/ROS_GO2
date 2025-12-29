"""
Navigation interfaces for patrol robots.

Provides abstraction layer for navigation systems, supporting both
simulated navigation (for training) and real Nav2 integration (for deployment).
"""

from .nav2_interface import (
    NavigationInterface,
    SimulatedNav2,
    NavigationResult,
)

__all__ = [
    "NavigationInterface",
    "SimulatedNav2",
    "NavigationResult",
]
