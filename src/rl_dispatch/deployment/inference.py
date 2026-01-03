"""
Policy Inference for Deployment.

This module provides the inference pipeline for running trained
policies on both simulation and real robots.

Usage:
    from rl_dispatch.deployment import PolicyRunner

    # Create runner
    runner = PolicyRunner(
        model_path="checkpoints/best_model.pth",
        mode="simulation"  # or "real"
    )

    # Run single episode
    metrics = runner.run_episode()

    # Run continuous patrol
    runner.run_continuous(max_episodes=100)
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
import torch

from .config import DeploymentConfig, SimulationConfig, RealRobotConfig
from .robot_interface import (
    RobotInterface,
    RobotState,
    NavigationFeedback,
    create_robot_interface
)

logger = logging.getLogger(__name__)


class PolicyRunner:
    """
    Runs trained RL policy for patrol robot deployment.

    This class:
    1. Loads trained PPO policy
    2. Connects to robot (simulation or real)
    3. Runs inference loop with observation processing
    4. Logs metrics and handles events

    Attributes:
        model_path: Path to trained model checkpoint
        config: Deployment configuration
        robot: Robot interface instance
        policy: Loaded policy network
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[DeploymentConfig] = None,
        mode: str = "simulation",
        device: str = "cpu"
    ):
        """
        Initialize policy runner.

        Args:
            model_path: Path to trained .pth checkpoint
            config: Deployment configuration
            mode: "simulation", "gazebo", or "real"
            device: Torch device for inference
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.mode = mode

        # Create config if not provided
        if config is None:
            if mode == "real":
                config = RealRobotConfig()
            else:
                config = SimulationConfig()
        self.config = config

        # Load policy
        self.policy = None
        self.observation_processor = None
        self._load_policy()

        # Robot interface (created on connect)
        self.robot: Optional[RobotInterface] = None

        # State tracking
        self.patrol_points: List[Tuple[float, float]] = []
        self.current_target_idx: int = 0
        self.episode_count: int = 0
        self.total_distance: float = 0.0
        self.events_handled: int = 0

    def _load_policy(self):
        """Load trained policy from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading policy from {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Import policy network
        from rl_dispatch.algorithms.networks import ActorCritic

        # Get network config from checkpoint or use default
        if 'network_config' in checkpoint:
            network_config = checkpoint['network_config']
        else:
            from rl_dispatch.core.config import NetworkConfig
            network_config = NetworkConfig()

        # Create network
        obs_dim = checkpoint.get('obs_dim', 88)  # Phase 4: 88D
        action_dim = checkpoint.get('action_dim', 10)  # 10 candidates

        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=network_config.hidden_dims,
            activation=network_config.activation,
        ).to(self.device)

        # Load weights
        self.policy.load_state_dict(checkpoint['actor_critic'])
        self.policy.eval()

        logger.info(f"Policy loaded: obs_dim={obs_dim}, action_dim={action_dim}")

        # Load observation normalizer if available
        if 'obs_normalizer' in checkpoint:
            from rl_dispatch.utils.observation import ObservationProcessor
            self.observation_processor = ObservationProcessor(
                use_normalization=True
            )
            self.observation_processor.normalizer.mean = checkpoint['obs_normalizer']['mean']
            self.observation_processor.normalizer.var = checkpoint['obs_normalizer']['var']
            self.observation_processor.normalizer.count = checkpoint['obs_normalizer']['count']
            logger.info("Observation normalizer loaded")

    def connect(self) -> bool:
        """
        Connect to robot.

        Returns:
            True if connection successful
        """
        logger.info(f"Connecting to robot in {self.mode} mode...")

        self.robot = create_robot_interface(
            mode=self.mode,
            config=self.config
        )

        if not self.robot.connect():
            logger.error("Failed to connect to robot")
            return False

        logger.info("Connected to robot successfully")
        return True

    def disconnect(self):
        """Disconnect from robot."""
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None
        logger.info("Disconnected from robot")

    def set_patrol_points(self, points: List[Tuple[float, float]]):
        """
        Set patrol points for the mission.

        Args:
            points: List of (x, y) patrol point coordinates
        """
        self.patrol_points = points
        self.current_target_idx = 0
        logger.info(f"Set {len(points)} patrol points")

    def get_observation(self, robot_state: RobotState) -> np.ndarray:
        """
        Convert robot state to observation vector.

        Args:
            robot_state: Current robot state

        Returns:
            88D observation vector
        """
        # Build observation vector (88D for Phase 4)
        obs = np.zeros(88, dtype=np.float32)

        # Goal relative (would need current target)
        if self.patrol_points and self.current_target_idx < len(self.patrol_points):
            target = self.patrol_points[self.current_target_idx]
            dx = target[0] - robot_state.position[0]
            dy = target[1] - robot_state.position[1]
            max_dist = 50.0
            obs[0] = np.clip(dx / max_dist, -1, 1)
            obs[1] = np.clip(dy / max_dist, -1, 1)

        # Heading
        obs[2] = np.sin(robot_state.heading)
        obs[3] = np.cos(robot_state.heading)

        # Velocity
        obs[4] = np.clip(robot_state.velocity / 2.0, -1, 1)
        obs[5] = np.clip(robot_state.angular_velocity / 1.0, -1, 1)

        # Battery
        obs[6] = robot_state.battery_level

        # LiDAR (64D)
        obs[7:71] = robot_state.lidar_ranges

        # Event features (4D) - would come from event detector
        obs[71:75] = 0.0

        # Patrol features (2D)
        obs[75] = 0.5  # Distance to next
        obs[76] = 0.5  # Coverage ratio

        # Phase 4 features
        obs[77] = 0.0  # Event risk
        obs[78:81] = [0.5, 0.3, 0.4]  # Patrol crisis
        obs[81:87] = [0.8] * 6  # Candidate feasibility
        obs[87] = 0.0  # Urgency-risk combined

        # Apply normalization if available
        if self.observation_processor is not None:
            obs = self.observation_processor.normalizer.normalize(obs)

        return obs

    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """
        Select action using policy.

        Args:
            observation: 88D observation vector
            deterministic: If True, use argmax; else sample

        Returns:
            Selected action index (0-9 for 10 candidates)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_probs, _ = self.policy(obs_tensor)

            if deterministic:
                action = action_probs.argmax(dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs=action_probs)
                action = dist.sample().item()

        return action

    def run_episode(
        self,
        max_steps: int = 400,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Run a single patrol episode.

        Args:
            max_steps: Maximum steps per episode
            render: Whether to render visualization

        Returns:
            Episode metrics dictionary
        """
        if self.robot is None or not self.robot.is_connected:
            raise RuntimeError("Robot not connected")

        logger.info(f"Starting episode {self.episode_count + 1}")

        start_time = time.time()
        total_reward = 0.0
        step_count = 0
        events_in_episode = 0

        # Get initial state
        state = self.robot.get_state()
        start_position = state.position

        for step in range(max_steps):
            # Get observation
            obs = self.get_observation(state)

            # Select action
            action = self.select_action(obs, deterministic=self.config.deterministic)

            # Execute action (navigate to selected candidate/target)
            if self.patrol_points:
                # Simple: action selects which patrol point to visit
                target_idx = action % len(self.patrol_points)
                target = self.patrol_points[target_idx]

                # Navigate
                feedback = self.robot.navigate_to(target, timeout=60.0)

                if feedback.success:
                    logger.info(f"Step {step}: Reached patrol point {target_idx}")
                    self.current_target_idx = (target_idx + 1) % len(self.patrol_points)
                else:
                    logger.warning(f"Step {step}: Navigation failed - {feedback.error_message}")

            # Get new state
            state = self.robot.get_state()
            step_count += 1

            # Check battery
            if state.battery_level < 0.1:
                logger.warning("Low battery - ending episode")
                break

        # Calculate metrics
        elapsed_time = time.time() - start_time
        end_position = state.position
        distance = np.sqrt(
            (end_position[0] - start_position[0])**2 +
            (end_position[1] - start_position[1])**2
        )

        self.episode_count += 1
        self.total_distance += distance

        metrics = {
            "episode": self.episode_count,
            "steps": step_count,
            "time": elapsed_time,
            "distance": distance,
            "events_handled": events_in_episode,
            "final_battery": state.battery_level,
            "success": step_count >= max_steps or state.battery_level < 0.1
        }

        logger.info(f"Episode complete: {metrics}")
        return metrics

    def run_continuous(
        self,
        max_episodes: int = -1,
        stop_on_low_battery: bool = True
    ):
        """
        Run continuous patrol operation.

        Args:
            max_episodes: Maximum episodes (-1 for unlimited)
            stop_on_low_battery: Stop when battery is critical
        """
        logger.info("Starting continuous patrol operation")

        episode = 0
        while max_episodes < 0 or episode < max_episodes:
            try:
                metrics = self.run_episode()
                episode += 1

                # Check battery
                if stop_on_low_battery and metrics['final_battery'] < 0.1:
                    logger.info("Stopping due to low battery")
                    break

            except KeyboardInterrupt:
                logger.info("Stopped by user")
                break
            except Exception as e:
                logger.error(f"Episode failed: {e}")
                time.sleep(1.0)

        logger.info(f"Continuous operation complete: {episode} episodes")


def main():
    """Example usage of PolicyRunner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run trained patrol policy")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="simulation", choices=["simulation", "gazebo", "real"])
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create runner
    runner = PolicyRunner(
        model_path=args.model,
        mode=args.mode,
        device=args.device
    )

    # Connect
    if not runner.connect():
        logger.error("Failed to connect")
        return

    # Set example patrol points
    runner.set_patrol_points([
        (5.0, 5.0),
        (10.0, 5.0),
        (10.0, 10.0),
        (5.0, 10.0),
    ])

    # Run episodes
    try:
        for i in range(args.episodes):
            metrics = runner.run_episode()
            print(f"Episode {i+1}: {metrics}")
    finally:
        runner.disconnect()


if __name__ == "__main__":
    main()
