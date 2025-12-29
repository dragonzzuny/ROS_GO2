"""
Configuration management for the RL Dispatch system.

This module provides dataclass-based configuration for all system components:
- Environment parameters (map size, patrol points, event generation)
- Training hyperparameters (PPO settings, learning rates, batch sizes)
- Reward function weights and normalization

All configurations support YAML serialization for experiment tracking.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import yaml
from pathlib import Path
import numpy as np


@dataclass
class RewardConfig:
    """
    Configuration for multi-component reward function.

    Defines weights for each reward component and normalization parameters.
    These weights are critical hyperparameters that should be tuned via
    ablation studies.

    Attributes:
        w_event: Weight for event response reward (R^evt)
        w_patrol: Weight for patrol coverage reward (R^pat)
        w_safety: Weight for safety reward (R^safe)
        w_efficiency: Weight for efficiency reward (R^eff)

        event_response_bonus: Reward for successfully reaching event (+)
        event_delay_penalty_rate: Penalty per second of delay (-)
        event_max_delay: Maximum delay before event is considered failed (seconds)

        patrol_gap_penalty_rate: Penalty per second of coverage gap (-)
        patrol_visit_bonus: Small reward for visiting patrol point (+)

        collision_penalty: Large penalty for collision (-)
        nav_failure_penalty: Penalty for Nav2 planning failure (-)

        distance_penalty_rate: Small penalty per meter traveled (-)

    Note:
        Default weights are based on initial testing. Run ablation studies
        to optimize for your specific deployment scenario.

    Example:
        >>> config = RewardConfig(w_event=1.0, w_patrol=0.5)
        >>> config.save_yaml("configs/reward_default.yaml")
    """
    # Component weights
    w_event: float = 1.0
    w_patrol: float = 0.5
    w_safety: float = 2.0
    w_efficiency: float = 0.1

    # Event response parameters
    event_response_bonus: float = 50.0
    event_delay_penalty_rate: float = 0.5  # per second
    event_max_delay: float = 120.0  # seconds

    # Patrol coverage parameters
    patrol_gap_penalty_rate: float = 0.1  # per second of gap
    patrol_visit_bonus: float = 2.0

    # Safety parameters
    collision_penalty: float = -100.0
    nav_failure_penalty: float = -20.0

    # Efficiency parameters
    distance_penalty_rate: float = 0.01  # per meter

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str) -> 'RewardConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle both formats:
        # 1. Top-level keys are RewardConfig params: {w_event: 1.0, ...}
        # 2. Nested under 'reward' key: {reward: {w_event: 1.0, ...}, env: {...}}
        if 'reward' in data:
            config_dict = data['reward']
        else:
            config_dict = data

        return cls(**config_dict)


@dataclass
class EnvConfig:
    """
    Environment configuration for PatrolEnv.

    Defines all parameters for the patrol environment including map bounds,
    patrol point locations, event generation, and episode settings.

    Attributes:
        map_width: Map width in meters
        map_height: Map height in meters
        patrol_points: List of (x, y) coordinates for patrol points
        patrol_point_priorities: Priority weight for each point (0.0-1.0)
        charging_station_position: (x, y) coordinates of charging station

        max_episode_steps: Maximum SMDP steps per episode
        max_episode_time: Maximum episode duration in seconds
        timestep: Simulation timestep in seconds (for physics)

        event_generation_rate: Average events per episode
        event_min_urgency: Minimum event urgency value
        event_max_urgency: Maximum event urgency value
        event_min_confidence: Minimum event confidence value

        robot_max_velocity: Maximum robot linear velocity (m/s)
        robot_max_angular_velocity: Maximum robot angular velocity (rad/s)
        robot_battery_capacity: Battery capacity in Wh (for depletion modeling)
        robot_battery_drain_rate: Battery drain rate in W

        lidar_num_channels: Number of LiDAR range channels
        lidar_max_range: Maximum LiDAR detection range (meters)
        lidar_min_range: Minimum LiDAR detection range (meters)

        num_candidates: Number of replan candidates to generate (K)
        candidate_strategies: List of strategy names to use

    Example:
        >>> config = EnvConfig(
        ...     map_width=50.0,
        ...     map_height=50.0,
        ...     patrol_points=[(10, 10), (40, 10), (40, 40), (10, 40)]
        ... )
        >>> env = PatrolEnv(config)
    """
    # Map configuration
    map_width: float = 50.0
    map_height: float = 50.0
    patrol_points: List[Tuple[float, float]] = field(default_factory=lambda: [
        (10.0, 10.0),
        (40.0, 10.0),
        (40.0, 40.0),
        (10.0, 40.0),
    ])
    patrol_point_priorities: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    charging_station_position: Tuple[float, float] = (5.0, 5.0)  # Default: near corner

    # Episode configuration
    max_episode_steps: int = 200
    max_episode_time: float = 600.0  # 10 minutes
    timestep: float = 0.1  # 100ms simulation timestep

    # Event generation
    event_generation_rate: float = 5.0  # average per episode
    event_min_urgency: float = 0.3
    event_max_urgency: float = 1.0
    event_min_confidence: float = 0.7

    # Robot dynamics
    robot_max_velocity: float = 1.5  # m/s
    robot_max_angular_velocity: float = 1.0  # rad/s
    robot_battery_capacity: float = 100.0  # Wh
    robot_battery_drain_rate: float = 20.0  # W (5 hours nominal)

    # LiDAR configuration
    lidar_num_channels: int = 64
    lidar_max_range: float = 10.0
    lidar_min_range: float = 0.1

    # Candidate configuration
    num_candidates: int = 6
    candidate_strategies: List[str] = field(default_factory=lambda: [
        "keep_order",
        "nearest_first",
        "most_overdue_first",
        "overdue_eta_balance",
        "risk_weighted",
        "balanced_coverage",
    ])

    # Rendering (optional)
    render_mode: Optional[str] = None  # "human", "rgb_array", or None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert len(self.patrol_points) >= 2, "Must have at least 2 patrol points"
        assert len(self.patrol_point_priorities) == len(self.patrol_points), \
            "Must have priority for each patrol point"
        assert self.num_candidates == len(self.candidate_strategies), \
            "num_candidates must match number of strategies"
        assert self.lidar_num_channels == 64, "Only 64-channel LiDAR supported currently"

    @property
    def num_patrol_points(self) -> int:
        """Returns the number of patrol points."""
        return len(self.patrol_points)

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str) -> 'EnvConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle both formats:
        # 1. Top-level keys are EnvConfig params: {map_width: 50.0, ...}
        # 2. Nested under 'env' key: {env: {map_width: 50.0, ...}, reward: {...}}
        if 'env' in data:
            config_dict = data['env']
        else:
            config_dict = data

        return cls(**config_dict)


@dataclass
class NetworkConfig:
    """
    Neural network architecture configuration.

    Defines the PPO actor-critic network structure including layer sizes,
    activation functions, and initialization strategies.

    Attributes:
        encoder_hidden_dims: Hidden layer dimensions for shared encoder
        activation: Activation function name ("relu", "tanh", "elu")
        use_layer_norm: Whether to use layer normalization
        orthogonal_init: Whether to use orthogonal weight initialization
        init_scale: Scaling factor for weight initialization

    Example:
        >>> config = NetworkConfig(encoder_hidden_dims=[256, 256])
    """
    # Encoder architecture
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    use_layer_norm: bool = False
    orthogonal_init: bool = True
    init_scale: float = np.sqrt(2)

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str) -> 'NetworkConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle both formats:
        # 1. Top-level keys: {encoder_hidden_dims: [256, 256], ...}
        # 2. Nested under 'network' key: {network: {...}}
        if 'network' in data:
            config_dict = data['network']
        else:
            config_dict = data

        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """
    PPO training hyperparameters.

    Comprehensive configuration for PPO algorithm including learning rates,
    batch sizes, clipping parameters, and training schedule.

    Attributes:
        # PPO hyperparameters
        learning_rate: Initial learning rate for optimizer
        gamma: Discount factor for future rewards
        gae_lambda: Lambda parameter for GAE advantage estimation
        clip_epsilon: PPO clipping parameter (epsilon)
        value_loss_coef: Coefficient for value loss in total loss
        entropy_coef: Coefficient for entropy bonus
        max_grad_norm: Maximum gradient norm for clipping

        # Training schedule
        total_timesteps: Total environment timesteps to train
        num_steps: Steps per rollout before update
        num_epochs: PPO epochs per update
        batch_size: Minibatch size for SGD
        num_minibatches: Number of minibatches per epoch

        # Normalization
        normalize_obs: Whether to normalize observations
        normalize_rewards: Whether to normalize rewards
        clip_obs: Maximum absolute value for normalized obs
        clip_rewards: Maximum absolute value for normalized rewards

        # Logging and checkpointing
        log_interval: Log metrics every N updates
        save_interval: Save checkpoint every N updates
        eval_interval: Run evaluation every N updates
        eval_episodes: Number of episodes for evaluation

    Note:
        Default values are based on PPO best practices. May require tuning
        for this specific task.

    Example:
        >>> config = TrainingConfig(
        ...     learning_rate=3e-4,
        ...     total_timesteps=10_000_000
        ... )
        >>> trainer = PPOTrainer(env, policy, config)
    """
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training schedule
    total_timesteps: int = 10_000_000
    num_steps: int = 2048  # Steps per rollout
    num_epochs: int = 10  # PPO epochs per update
    batch_size: int = 256  # Minibatch size
    num_minibatches: int = 8  # batch_size * num_minibatches = num_steps

    # Learning rate schedule
    anneal_lr: bool = True
    lr_schedule: str = "linear"  # "linear" or "constant"

    # Normalization
    normalize_obs: bool = True
    normalize_rewards: bool = True
    clip_obs: float = 10.0
    clip_rewards: float = 10.0

    # Logging and checkpointing
    log_interval: int = 10  # updates
    save_interval: int = 100  # updates
    eval_interval: int = 50  # updates
    eval_episodes: int = 10

    # Experiment tracking
    experiment_name: str = "ppo_patrol"
    seed: int = 42
    cuda: bool = True
    num_envs: int = 1  # Parallel environments (if supported)

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.num_steps % self.num_minibatches == 0, \
            "num_steps must be divisible by num_minibatches"
        self.batch_size = self.num_steps // self.num_minibatches

    @property
    def total_updates(self) -> int:
        """Calculate total number of PPO updates."""
        return self.total_timesteps // self.num_steps

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle both formats:
        # 1. Top-level keys: {learning_rate: 3e-4, ...}
        # 2. Nested under 'training' key: {training: {...}}
        if 'training' in data:
            config_dict = data['training']
        else:
            config_dict = data

        return cls(**config_dict)
