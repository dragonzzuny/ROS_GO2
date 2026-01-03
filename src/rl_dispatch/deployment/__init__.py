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

    # Run trained policy
    from rl_dispatch.deployment import PolicyRunner
    runner = PolicyRunner(model_path="checkpoints/best_model.pth", mode="simulation")
    runner.connect()
    runner.run_episode()
"""

from .robot_interface import (
    RobotInterface,
    RobotState,
    NavigationGoal,
    NavigationFeedback,
    SimulationRobotInterface,
    create_robot_interface,
)
from .config import DeploymentConfig, SimulationConfig, RealRobotConfig
from .inference import PolicyRunner

__all__ = [
    # Interfaces
    "RobotInterface",
    "RobotState",
    "NavigationGoal",
    "NavigationFeedback",
    "SimulationRobotInterface",
    "create_robot_interface",
    # Config
    "DeploymentConfig",
    "SimulationConfig",
    "RealRobotConfig",
    # Inference
    "PolicyRunner",
]
