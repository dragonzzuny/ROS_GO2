"""
Deployment configuration for simulation and real robot.

Provides unified configuration interface for both simulation and real deployment.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum
import yaml
from pathlib import Path


class DeploymentMode(Enum):
    """Deployment mode enumeration."""
    SIMULATION = "simulation"
    GAZEBO = "gazebo"
    REAL = "real"


@dataclass
class DeploymentConfig:
    """
    Base deployment configuration.

    Attributes:
        mode: Deployment mode (simulation, gazebo, real)
        model_path: Path to trained policy checkpoint
        map_config_path: Path to map configuration YAML
        device: Torch device (cuda/cpu)
        log_level: Logging level
    """
    mode: DeploymentMode = DeploymentMode.SIMULATION
    model_path: str = "checkpoints/best_model.pth"
    map_config_path: str = "configs/default.yaml"
    device: str = "cpu"
    log_level: str = "INFO"

    # Inference settings
    deterministic: bool = True  # Use argmax instead of sampling
    observation_normalization: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "DeploymentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if 'mode' in data:
            data['mode'] = DeploymentMode(data['mode'])

        return cls(**data)


@dataclass
class SimulationConfig(DeploymentConfig):
    """
    Simulation-specific configuration.

    Attributes:
        sim_time_step: Simulation time step (seconds)
        render: Whether to render visualization
        record_video: Whether to record episode videos
        video_path: Path to save recorded videos
    """
    mode: DeploymentMode = DeploymentMode.SIMULATION

    # Simulation settings
    sim_time_step: float = 0.1
    render: bool = False
    record_video: bool = False
    video_path: str = "videos/"

    # Environment overrides
    max_episode_steps: int = 400
    event_generation_rate: float = 20.0


@dataclass
class GazeboConfig(DeploymentConfig):
    """
    Gazebo simulation configuration.

    Attributes:
        gazebo_world: Path to Gazebo world file
        robot_model: Robot URDF/SDF model name
        use_sim_time: Use Gazebo simulation time
    """
    mode: DeploymentMode = DeploymentMode.GAZEBO

    # Gazebo settings
    gazebo_world: str = "worlds/patrol_environment.world"
    robot_model: str = "go2_description"
    use_sim_time: bool = True

    # ROS2 topics
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"


@dataclass
class RealRobotConfig(DeploymentConfig):
    """
    Real Go2 robot configuration.

    Attributes:
        robot_ip: Go2 robot IP address
        robot_port: Communication port
        nav2_namespace: Nav2 namespace
        safety_enabled: Enable safety constraints
        max_linear_velocity: Maximum linear velocity (m/s)
        max_angular_velocity: Maximum angular velocity (rad/s)
    """
    mode: DeploymentMode = DeploymentMode.REAL

    # Robot connection
    robot_ip: str = "192.168.123.161"  # Default Go2 IP
    robot_port: int = 8080

    # ROS2 Nav2 settings
    nav2_namespace: str = ""
    navigate_to_pose_action: str = "/navigate_to_pose"
    get_path_service: str = "/compute_path_to_pose"

    # Safety constraints
    safety_enabled: bool = True
    max_linear_velocity: float = 1.5  # m/s
    max_angular_velocity: float = 1.0  # rad/s
    min_obstacle_distance: float = 0.5  # m
    emergency_stop_distance: float = 0.3  # m

    # Battery management
    low_battery_threshold: float = 0.20  # 20%
    critical_battery_threshold: float = 0.10  # 10%
    auto_return_on_low_battery: bool = True

    # Watchdog
    watchdog_timeout: float = 5.0  # seconds
    heartbeat_interval: float = 1.0  # seconds

    # Topics
    battery_topic: str = "/battery_state"
    imu_topic: str = "/imu/data"
    joint_states_topic: str = "/joint_states"

    # Patrol points (loaded from map config)
    patrol_points: List[Tuple[float, float]] = field(default_factory=list)
    charging_station: Tuple[float, float] = (0.0, 0.0)


def load_deployment_config(config_path: str) -> DeploymentConfig:
    """
    Load deployment configuration from YAML file.

    Automatically selects appropriate config class based on mode.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Appropriate DeploymentConfig subclass instance
    """
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    mode = data.get('mode', 'simulation')

    if mode == 'simulation':
        return SimulationConfig(**{k: v for k, v in data.items()
                                   if k in SimulationConfig.__dataclass_fields__})
    elif mode == 'gazebo':
        return GazeboConfig(**{k: v for k, v in data.items()
                               if k in GazeboConfig.__dataclass_fields__})
    elif mode == 'real':
        return RealRobotConfig(**{k: v for k, v in data.items()
                                  if k in RealRobotConfig.__dataclass_fields__})
    else:
        raise ValueError(f"Unknown deployment mode: {mode}")
