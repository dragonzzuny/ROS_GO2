#!/usr/bin/env python3
"""
Multi-Map Training Script

여러 맵에서 동시에 학습하여 일반화된 정책을 만듭니다.
에피소드마다 다른 맵이 선택되어 다양한 시나리오에서 학습합니다.

Features:
- Random/Sequential/Curriculum map selection
- Per-map performance tracking
- Coverage heatmap visualization
- Generalization evaluation across unseen maps
"""

import argparse
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_dispatch.env import create_multi_map_env
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.core.config import NetworkConfig, TrainingConfig, RewardConfig


def train_multi_map(
    env_wrapper,
    agent: PPOAgent,
    config: TrainingConfig,
    log_dir: Path,
) -> None:
    """
    Train PPO agent on multiple maps.

    Args:
        env_wrapper: MultiMapPatrolEnv instance
        agent: PPO agent to train
        config: Training configuration
        log_dir: Directory for logs and checkpoints
    """
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    obs, info = env_wrapper.reset()
    episode_return = 0.0
    episode_length = 0
    current_map = info["map_name"]

    num_updates = config.total_timesteps // config.num_steps
    global_step = 0

    print("\n" + "=" * 80)
    print("Starting Multi-Map Training")
    print("=" * 80)
    print(f"Maps: {env_wrapper.map_names}")
    print(f"Selection Mode: {env_wrapper.map_selection_mode}")
    print(f"Total Timesteps: {config.total_timesteps:,}")
    print(f"Updates: {num_updates:,}")
    print(f"Steps per Update: {config.num_steps}")
    print("=" * 80 + "\n")

    start_time = time.time()

    for update in range(1, num_updates + 1):
        # Collect rollout
        for step in range(config.num_steps):
            global_step += 1

            # Select action
            action, log_prob, value = agent.get_action(obs)

            # Environment step
            next_obs, reward, terminated, truncated, info = env_wrapper.step(action)

            episode_return += reward
            episode_length += 1

            # Store transition in agent's buffer
            agent.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=terminated,
            )

            obs = next_obs

            # Episode finished
            if terminated or truncated:
                # Log episode
                writer.add_scalar("episode/return", episode_return, global_step)
                writer.add_scalar("episode/length", episode_length, global_step)
                writer.add_scalar(f"episode_per_map/{current_map}/return",
                                episode_return, global_step)

                if "map_episode_metrics" in info:
                    metrics = info["map_episode_metrics"]
                    writer.add_scalar(
                        f"episode_per_map/{current_map}/event_success_rate",
                        metrics.event_success_rate, global_step
                    )
                    writer.add_scalar(
                        f"episode_per_map/{current_map}/patrol_coverage",
                        metrics.patrol_coverage_ratio, global_step
                    )

                # Reset
                obs, info = env_wrapper.reset()
                current_map = info["map_name"]
                episode_return = 0.0
                episode_length = 0

        # Compute final value for bootstrapping
        final_action, final_log_prob, final_value = agent.get_action(obs)

        # PPO update with bootstrapping from final state
        train_stats = agent.update(
            last_value=final_value,
            last_done=False  # Not done since we're mid-episode
        )

        # Log training stats
        for key, value in train_stats.items():
            writer.add_scalar(f"train/{key}", value, global_step)

        # Logging
        if update % config.log_interval == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed

            print(f"\nUpdate {update}/{num_updates}:")
            print(f"  Global Step: {global_step:,}")
            print(f"  FPS: {fps:.1f}")
            print(f"  Current Map: {current_map}")

            # Map statistics
            stats = env_wrapper.get_map_statistics()
            print(f"\n  Per-Map Performance:")
            for map_name in env_wrapper.map_names:
                s = stats[map_name]
                if s["episodes"] > 0:
                    print(f"    {map_name}:")
                    print(f"      Episodes: {s['episodes']}")
                    print(f"      Return: {s['mean_return']:.1f} ± {s['std_return']:.1f}")
                    print(f"      Success: {100*s['mean_event_success']:.1f}%")
                    print(f"      Coverage: {100*s['mean_patrol_coverage']:.1f}%")

            # Training stats
            print(f"\n  Training:")
            for key, value in train_stats.items():
                print(f"    {key}: {value:.4f}")

        # Save checkpoint
        if update % config.save_interval == 0:
            checkpoint_path = log_dir / "checkpoints" / f"update_{update}.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(str(checkpoint_path))
            print(f"\n  ✅ Saved checkpoint: {checkpoint_path}")

        # Periodic evaluation (coverage visualization)
        if update % config.eval_interval == 0:
            print(f"\n  Running coverage evaluation...")

            # Save coverage heatmaps
            coverage_dir = log_dir / "coverage" / f"update_{update}"
            coverage_dir.mkdir(parents=True, exist_ok=True)

            for map_name in env_wrapper.map_names:
                heatmap = env_wrapper.get_coverage_heatmap(map_name)
                if heatmap is not None:
                    np.save(
                        coverage_dir / f"{map_name}_heatmap.npy",
                        heatmap
                    )

            print(f"  ✅ Saved coverage heatmaps to {coverage_dir}")

    # Final save
    final_path = log_dir / "checkpoints" / "final.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(final_path))
    print(f"\n✅ Training complete! Final model saved to: {final_path}")

    # Print final statistics
    env_wrapper.print_statistics()

    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO policy on multiple map configurations"
    )

    # Map selection
    parser.add_argument(
        "--maps",
        type=str,
        nargs="+",
        default=None,
        help="Specific map config files (uses all diverse maps if not specified)",
    )
    parser.add_argument(
        "--map-mode",
        type=str,
        default="random",
        choices=["random", "sequential", "curriculum"],
        help="Map selection strategy",
    )

    # Training config
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        help="Rollout steps per update",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="PPO epochs per update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size",
    )

    # Experiment
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="multi_map_ppo",
        help="Experiment name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA",
    )

    # Logging
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N updates",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save checkpoint every N updates",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Evaluate coverage every N updates",
    )

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    print("Creating multi-map environment...")
    env = create_multi_map_env(
        map_configs=args.maps,
        mode=args.map_mode,
        track_coverage=True,
    )

    print(f"✅ Loaded {len(env.map_names)} maps:")
    for name in env.map_names:
        config = env.env_configs[name]
        print(f"  - {name}: {config.map_width}×{config.map_height}m, "
              f"{config.num_patrol_points} points")

    # Create PPO agent
    print("\nCreating PPO agent...")
    obs_dim = env.observation_space.shape[0]
    num_replan_strategies = env.action_space.nvec[1]  # Second dimension of MultiDiscrete

    # Training config
    training_config = TrainingConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        experiment_name=args.experiment_name,
        seed=args.seed,
        cuda=(args.cuda and torch.cuda.is_available()),
    )

    network_config = NetworkConfig()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    agent = PPOAgent(
        obs_dim=obs_dim,
        num_replan_strategies=num_replan_strategies,
        training_config=training_config,
        network_config=network_config,
        device=device,
    )

    print(f"✅ PPO agent created on {device}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Num strategies: {num_replan_strategies}")

    # Log directory
    log_dir = Path("runs") / args.experiment_name / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save configs
    training_config.save_yaml(str(log_dir / "training_config.yaml"))
    print(f"\n✅ Logs will be saved to: {log_dir}")

    # Train!
    train_multi_map(env, agent, training_config, log_dir)

    env.close()


if __name__ == "__main__":
    main()
