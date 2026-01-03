"""
RL Dispatch Deployment Module.

This module provides interfaces for deploying trained policies to:
- Simulation environments (Gazebo, Isaac Sim)
- Real Unitree Go2 robots via ROS2

Usage:
    # Simulation mode
    from rl_dispatch.deployment import create_robot_interface
    robot = create_robot_interface(mode="simulation", config=sim_config)

    # Real robot mode
    robot = create_robot_interface(mode="real", config=real_config)
"""

from .robot_interface import RobotInterface, create_robot_interface
from .config import DeploymentConfig, SimulationConfig, RealRobotConfig

__all__ = [
    "RobotInterface",
    "create_robot_interface",
    "DeploymentConfig",
    "SimulationConfig",
    "RealRobotConfig",
]
