"""
Common interfaces and utilities for simulation.

Provides base classes and configuration for all simulation interfaces.
"""

from .robot_interface import (
    RobotInterface,
    RobotState,
    NavigationGoal,
    NavigationFeedback,
    SimulationRobotInterface,
    create_robot_interface,
)
from .config import (
    DeploymentConfig,
    DeploymentMode,
    SimulationConfig,
    GazeboConfig,
    RealRobotConfig,
    load_deployment_config,
)

__all__ = [
    # Interface classes
    "RobotInterface",
    "RobotState",
    "NavigationGoal",
    "NavigationFeedback",
    "SimulationRobotInterface",
    "create_robot_interface",
    # Config classes
    "DeploymentConfig",
    "DeploymentMode",
    "SimulationConfig",
    "GazeboConfig",
    "RealRobotConfig",
    "load_deployment_config",
]
