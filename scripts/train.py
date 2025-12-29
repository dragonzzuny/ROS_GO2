#!/usr/bin/env python3
# 검수자: 박용준
"""
Training script for PPO agent on PatrolEnv.

This script implements the complete training loop including:
- Environment creation
- PPO agent initialization
- Experience collection
- Policy updates
- TensorBoard logging
- Model checkpointing
- Periodic evaluation

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --env-config configs/env.yaml --reward-config configs/reward.yaml
    python scripts/train.py --seed 42 --cuda
"""

import argparse
import time
from pathlib import Path
from typing import Optional
import yaml

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rl_dispatch.env import PatrolEnv
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.core.config import EnvConfig, RewardConfig, TrainingConfig, NetworkConfig
from rl_dispatch.utils import ObservationProcessor


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on patrol dispatch task"
    )

    # Configuration files
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the main configuration YAML file",
    )

    # Training parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments (not implemented yet)",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for model checkpoints",
    )

    return parser.parse_args()


def make_env(
    env_config: Optional[EnvConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    seed: int = 42,
) -> PatrolEnv:
    """
    Create and configure environment.

    Args:
        env_config: Environment configuration
        reward_config: Reward configuration
        seed: Random seed

    Returns:
        Configured PatrolEnv
    """
    env = PatrolEnv(
        env_config=env_config,
        reward_config=reward_config,
    )
    env.reset(seed=seed)
    return env


def evaluate(
    agent: PPOAgent,
    env: PatrolEnv,
    num_episodes: int = 10,
) -> dict:
    """
    Evaluate agent performance.

    Args:
        agent: PPO agent to evaluate
        env: Environment
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary of evaluation metrics
    """
    agent.eval_mode()

    episode_returns = []
    episode_lengths = []
    events_detected = []
    events_responded = []
    events_successful = []
    avg_delays = []
    coverage_ratios = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0
        done = False

        while not done:
            # Select action deterministically
            action, _, _ = agent.get_action(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated

        # Collect episode metrics
        if "episode" in info:
            episode_returns.append(info["episode"]["r"])
            episode_lengths.append(info["episode"]["l"])
            events_detected.append(info["episode"]["events_detected"])
            events_responded.append(info["episode"]["events_responded"])
            events_successful.append(info["episode"]["events_successful"])
            avg_delays.append(info["episode"]["avg_event_delay"])
            coverage_ratios.append(info["episode"]["patrol_coverage"])

    agent.train_mode()

    return {
        "eval/mean_return": np.mean(episode_returns),
        "eval/mean_length": np.mean(episode_lengths),
        "eval/events_detected": np.mean(events_detected),
        "eval/events_responded": np.mean(events_responded),
        "eval/events_successful": np.mean(events_successful),
        "eval/event_response_rate": np.mean(events_responded) / max(np.mean(events_detected), 1),
        "eval/event_success_rate": np.mean(events_successful) / max(np.mean(events_detected), 1),
        "eval/avg_event_delay": np.mean(avg_delays),
        "eval/patrol_coverage": np.mean(coverage_ratios),
    }


def train(args: argparse.Namespace) -> None:
    """
    Main training loop.

    Args:
        args: Command line arguments
    """
    # Load master configuration
    with open(args.config, 'r') as f:
        master_config = yaml.safe_load(f)

    env_config = EnvConfig(**master_config.get("env", {}))
    reward_config = RewardConfig(**master_config.get("reward", {}))
    training_config = TrainingConfig(**master_config.get("training", {}))
    network_config = NetworkConfig(**master_config.get("network", {}))

    # Update training config from args
    training_config.seed = args.seed
    training_config.cuda = args.cuda

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create run name
    run_name = args.run_name or f"ppo_{int(time.time())}"

    # Create directories
    log_dir = Path(args.log_dir) / run_name
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save configurations
    env_config.save_yaml(str(checkpoint_dir / "env_config.yaml"))
    reward_config.save_yaml(str(checkpoint_dir / "reward_config.yaml"))
    training_config.save_yaml(str(checkpoint_dir / "training_config.yaml"))
    network_config.save_yaml(str(checkpoint_dir / "network_config.yaml"))

    # Initialize TensorBoard writer
    writer = SummaryWriter(str(log_dir))

    # Create environment
    env = make_env(env_config, reward_config, args.seed)

    # Create agent
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    agent = PPOAgent(
        obs_dim=77,
        num_replan_strategies=env_config.num_candidates,
        training_config=training_config,
        network_config=network_config,
        device=device,
    )

    print(f"Training PPO agent on {device}")
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {training_config.total_timesteps:,}")
    print(f"Updates: {training_config.total_updates:,}")
    print("-" * 80)

    # Training loop
    obs, info = env.reset()
    episode_return = 0
    episode_length = 0
    episode_count = 0
    start_time = time.time()

    for global_step in range(training_config.total_timesteps):
        # Collect experience
        action, log_prob, value = agent.get_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store in buffer
        agent.buffer.add(obs, action, log_prob, reward, value, done)

        # Update for next step
        episode_return += reward
        episode_length += 1
        obs = next_obs

        # Handle episode end
        if done:
            episode_count += 1

            # Log episode metrics
            if "episode" in info:
                writer.add_scalar("train/episode_return", info["episode"]["r"], global_step)
                writer.add_scalar("train/episode_length", info["episode"]["l"], global_step)
                writer.add_scalar("train/events_detected", info["episode"]["events_detected"], global_step)
                writer.add_scalar("train/events_responded", info["episode"]["events_responded"], global_step)
                writer.add_scalar("train/events_successful", info["episode"]["events_successful"], global_step)
                writer.add_scalar("train/avg_event_delay", info["episode"]["avg_event_delay"], global_step)
                writer.add_scalar("train/patrol_coverage", info["episode"]["patrol_coverage"], global_step)
                writer.add_scalar("train/safety_violations", info["episode"]["safety_violations"], global_step)

                print(
                    f"Episode {episode_count} | "
                    f"Steps: {global_step:,} | "
                    f"Return: {info['episode']['r']:.2f} | "
                    f"Length: {info['episode']['l']} | "
                    f"Events: {info['episode']['events_detected']}/{info['episode']['events_successful']} | "
                    f"Coverage: {info['episode']['patrol_coverage']:.2%}"
                )

            # Reset environment
            obs, info = env.reset()
            episode_return = 0
            episode_length = 0

        # Perform update when buffer is full
        if agent.buffer.is_full:
            # Get last value for bootstrapping
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
                last_value = agent.network.get_value(obs_t).item()

            # Compute returns and perform update
            agent.buffer.compute_returns_and_advantages(last_value, done)
            stats = agent.update()

            # Log training statistics
            for key, value in stats.items():
                writer.add_scalar(f"train/{key}", value, global_step)

            # Print update stats
            if agent.update_count % training_config.log_interval == 0:
                elapsed_time = time.time() - start_time
                fps = global_step / elapsed_time
                print(
                    f"Update {agent.update_count} | "
                    f"Steps: {global_step:,} | "
                    f"FPS: {fps:.0f} | "
                    f"Policy Loss: {stats['policy_loss']:.4f} | "
                    f"Value Loss: {stats['value_loss']:.4f} | "
                    f"Entropy: {stats['entropy']:.4f} | "
                    f"KL: {stats['approx_kl']:.4f} | "
                    f"LR: {stats['learning_rate']:.6f}"
                )

            # Save checkpoint
            if agent.update_count % training_config.save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{agent.update_count}.pth"
                agent.save(str(checkpoint_path))
                print(f"Saved checkpoint: {checkpoint_path}")

            # Evaluation
            if agent.update_count % training_config.eval_interval == 0:
                print("Running evaluation...")
                eval_metrics = evaluate(agent, env, training_config.eval_episodes)
                for key, value in eval_metrics.items():
                    writer.add_scalar(key, value, global_step)
                print(
                    f"Eval | "
                    f"Mean Return: {eval_metrics['eval/mean_return']:.2f} | "
                    f"Success Rate: {eval_metrics['eval/event_success_rate']:.2%} | "
                    f"Coverage: {eval_metrics['eval/patrol_coverage']:.2%}"
                )

    # Save final model
    final_path = checkpoint_dir / "final_model.pth"
    agent.save(str(final_path))
    print(f"Training complete! Saved final model: {final_path}")

    # Close
    writer.close()
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
