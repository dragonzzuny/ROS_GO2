#!/usr/bin/env python3
"""
Multi-Map Training Script - FIXED VERSION

주요 개선사항:
1. Reward normalization 제대로 구현
2. 상세한 diagnostic logging (action mask, entropy, advantage, etc.)
3. Entropy annealing으로 탐색 붕괴 방지
4. Improved hyperparameters (낮은 LR, 높은 entropy_coef)
5. Action masking 검증 로그

작성자: Reviewer 박용준
수정일: 2025-12-30
"""

import argparse
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_dispatch.env import create_multi_map_env
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.core.config import NetworkConfig, TrainingConfig, RewardConfig


class RunningMeanStd:
    """Running mean and std for reward normalization."""

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        """Update with batch of values."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def train_multi_map(
    env_wrapper,
    agent: PPOAgent,
    config: TrainingConfig,
    log_dir: Path,
) -> None:
    """
    Train PPO agent on multiple maps with improved diagnostics.

    Args:
        env_wrapper: MultiMapPatrolEnv instance
        agent: PPO agent to train
        config: Training configuration
        log_dir: Directory for logs and checkpoints
    """
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    # Reviewer 박용준: Reward normalization (Welford's online algorithm)
    reward_normalizer = RunningMeanStd(shape=())

    obs, info = env_wrapper.reset()
    episode_return = 0.0
    episode_length = 0
    current_map = info["map_name"]

    # Episode tracking for diagnostics
    raw_rewards = []
    action_mode_counts = {"patrol": 0, "dispatch": 0}
    valid_action_counts = []

    num_updates = config.total_timesteps // config.num_steps
    global_step = 0

    print("\n" + "=" * 80)
    print("Starting Multi-Map Training (FIXED VERSION)")
    print("=" * 80)
    print(f"Maps: {env_wrapper.map_names}")
    print(f"Selection Mode: {env_wrapper.map_selection_mode}")
    print(f"Total Timesteps: {config.total_timesteps:,}")
    print(f"Updates: {num_updates:,}")
    print(f"Steps per Update: {config.num_steps}")
    print(f"\n🔧 FIXES APPLIED:")
    print(f"  - Reward normalization: ENABLED")
    print(f"  - Entropy annealing: 0.1 → 0.01 over training")
    print(f"  - Learning rate: 1e-4 (reduced from 3e-4)")
    print(f"  - Diagnostic logging: EXTENSIVE")
    print("=" * 80 + "\n")

    start_time = time.time()

    for update in range(1, num_updates + 1):
        # Reviewer 박용준: Entropy annealing (start high, decay to low)
        # 초반에는 탐색, 후반에는 exploitation
        progress = update / num_updates
        current_entropy_coef = 0.1 * (1.0 - progress) + 0.01 * progress
        agent.training_config.entropy_coef = current_entropy_coef

        # Collect rollout
        rollout_raw_rewards = []
        rollout_action_modes = []
        rollout_valid_actions = []

        for step in range(config.num_steps):
            global_step += 1

            # Reviewer 박용준: Get action_mask from info (from previous step or reset)
            action_mask = info.get("action_mask", None)

            # Select action with action masking
            action, log_prob, value = agent.get_action(obs, action_mask=action_mask)

            # Reviewer 박용준: Track action mode
            action_mode = int(action[0])  # 0=patrol, 1=dispatch
            rollout_action_modes.append(action_mode)

            # Reviewer 박용준: Track valid action count from mask
            if "action_mask" in info:
                mask = info["action_mask"]
                valid_count = np.sum(mask > 0.5)
                rollout_valid_actions.append(valid_count)

            # Environment step
            next_obs, reward, terminated, truncated, info = env_wrapper.step(action)

            # Reviewer 박용준: Store raw reward for normalization
            rollout_raw_rewards.append(reward)
            raw_rewards.append(reward)

            # Reviewer 박용준: Normalize reward using running statistics
            if config.normalize_rewards:
                # Update normalizer
                reward_normalizer.update(np.array([reward]))

                # Normalize: (r - mean) / sqrt(var + eps)
                normalized_reward = (reward - reward_normalizer.mean) / np.sqrt(reward_normalizer.var + 1e-8)

                # Clip to prevent extreme values
                normalized_reward = np.clip(normalized_reward, -config.clip_rewards, config.clip_rewards)
            else:
                normalized_reward = reward

            episode_return += reward  # Track raw return for logging
            episode_length += 1

            # Store transition in agent's buffer
            nav_time = info.get("nav_time", 1.0)
            action_mask = info.get("action_mask", None)

            agent.buffer.add(
                obs=obs,
                action=action,
                reward=normalized_reward,  # Use normalized reward!
                value=value,
                log_prob=log_prob,
                done=terminated,
                nav_time=nav_time,
                action_mask=action_mask,
            )

            obs = next_obs

            # Episode finished
            if terminated or truncated:
                # Log episode
                writer.add_scalar("episode/return", episode_return, global_step)
                writer.add_scalar("episode/length", episode_length, global_step)
                writer.add_scalar(f"episode_per_map/{current_map}/return",
                                episode_return, global_step)

                # Reviewer 박용준: Log action distribution
                patrol_ratio = action_mode_counts["patrol"] / max(1, episode_length)
                dispatch_ratio = action_mode_counts["dispatch"] / max(1, episode_length)
                writer.add_scalar(f"episode_per_map/{current_map}/patrol_ratio",
                                patrol_ratio, global_step)
                writer.add_scalar(f"episode_per_map/{current_map}/dispatch_ratio",
                                dispatch_ratio, global_step)

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

                # Reset counters
                action_mode_counts = {"patrol": 0, "dispatch": 0}
                obs, info = env_wrapper.reset()
                current_map = info["map_name"]
                episode_return = 0.0
                episode_length = 0

        # Reviewer 박용준: Diagnostic logging for rollout
        if len(rollout_raw_rewards) > 0:
            writer.add_scalar("diagnostics/reward_mean_raw", np.mean(rollout_raw_rewards), global_step)
            writer.add_scalar("diagnostics/reward_std_raw", np.std(rollout_raw_rewards), global_step)
            writer.add_scalar("diagnostics/reward_mean_normalized", reward_normalizer.mean, global_step)
            writer.add_scalar("diagnostics/reward_std_normalized", np.sqrt(reward_normalizer.var), global_step)

        if len(rollout_action_modes) > 0:
            patrol_count = sum(1 for m in rollout_action_modes if m == 0)
            dispatch_count = sum(1 for m in rollout_action_modes if m == 1)
            writer.add_scalar("diagnostics/action_patrol_ratio", patrol_count / len(rollout_action_modes), global_step)
            writer.add_scalar("diagnostics/action_dispatch_ratio", dispatch_count / len(rollout_action_modes), global_step)

        if len(rollout_valid_actions) > 0:
            writer.add_scalar("diagnostics/valid_action_count_mean", np.mean(rollout_valid_actions), global_step)
            writer.add_scalar("diagnostics/valid_action_count_min", np.min(rollout_valid_actions), global_step)

        # Compute final value for bootstrapping
        final_action, final_log_prob, final_value = agent.get_action(obs)

        # PPO update with bootstrapping from final state
        train_stats = agent.update(
            last_value=final_value,
            last_done=False
        )

        # Reviewer 박용준: Enhanced diagnostic logging
        # Advantage statistics
        if hasattr(agent.buffer, 'advantages'):
            advantages = agent.buffer.advantages
            writer.add_scalar("diagnostics/advantage_mean", np.mean(advantages), global_step)
            writer.add_scalar("diagnostics/advantage_std", np.std(advantages), global_step)
            writer.add_scalar("diagnostics/advantage_max", np.max(advantages), global_step)
            writer.add_scalar("diagnostics/advantage_min", np.min(advantages), global_step)

        # Value/return statistics
        if hasattr(agent.buffer, 'values') and hasattr(agent.buffer, 'returns'):
            values = agent.buffer.values
            returns = agent.buffer.returns
            writer.add_scalar("diagnostics/value_mean", np.mean(values), global_step)
            writer.add_scalar("diagnostics/return_mean", np.mean(returns), global_step)
            writer.add_scalar("diagnostics/value_return_gap", np.mean(returns - values), global_step)

        # Log training stats
        for key, value in train_stats.items():
            writer.add_scalar(f"train/{key}", value, global_step)

        # Reviewer 박용준: Log current entropy coefficient (annealing)
        writer.add_scalar("train/entropy_coef", current_entropy_coef, global_step)

        # Logging
        if update % config.log_interval == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed

            print(f"\nUpdate {update}/{num_updates}:")
            print(f"  Global Step: {global_step:,}")
            print(f"  FPS: {fps:.1f}")
            print(f"  Current Map: {current_map}")
            print(f"  Entropy Coef: {current_entropy_coef:.4f}")

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

            # Reviewer 박용준: Diagnostic stats
            print(f"\n  Diagnostics:")
            if len(raw_rewards) > 0:
                recent_rewards = raw_rewards[-config.num_steps:]
                print(f"    Reward (raw): {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
                print(f"    Reward (normalized): mean={reward_normalizer.mean:.2f}, std={np.sqrt(reward_normalizer.var):.2f}")
            if len(rollout_valid_actions) > 0:
                print(f"    Valid actions: {np.mean(rollout_valid_actions):.1f} (avg), {np.min(rollout_valid_actions)} (min)")

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
        description="Train PPO policy on multiple maps (FIXED VERSION)"
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
        default=1e-4,  # Reviewer 박용준: 3e-4 → 1e-4 (안정성)
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
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.1,  # Reviewer 박용준: 0.01 → 0.1 (탐색 강화, annealing 적용)
        help="Initial entropy coefficient (will anneal to 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=1.0,  # Reviewer 박용준: 0.5 → 1.0 (critic 학습 강화)
        help="Value loss coefficient",
    )

    # Experiment
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="multi_map_ppo_fixed",
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
        default=50,  # Reviewer 박용준: 100 → 50 (더 자주 저장)
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
    num_replan_strategies = env.action_space.nvec[1]

    # Training config with FIXED hyperparameters
    training_config = TrainingConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,  # Will be annealed
        value_loss_coef=args.value_loss_coef,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        experiment_name=args.experiment_name,
        seed=args.seed,
        cuda=(args.cuda and torch.cuda.is_available()),
        normalize_rewards=True,  # Reviewer 박용준: Enable reward normalization
        clip_rewards=5.0,  # Reviewer 박용준: 10.0 → 5.0 (더 보수적인 clipping)
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
    print(f"\n🔧 FIXED Hyperparameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Entropy coef: {args.entropy_coef} (annealing to 0.01)")
    print(f"  Value loss coef: {args.value_loss_coef}")
    print(f"  Reward normalization: ENABLED")
    print(f"  Clip rewards: 5.0")

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
