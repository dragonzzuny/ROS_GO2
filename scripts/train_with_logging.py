#!/usr/bin/env python3
"""
Multi-Map Training Script with COMPREHENSIVE LOGGING

모든 학습 데이터를 빠짐없이 기록:
- TensorBoard (실시간 모니터링)
- CSV files (상세 분석용)
- JSON files (설정 및 최종 결과)
- Console output (실시간 확인)

작성자: Reviewer 박용준
작성일: 2025-12-30
"""

import argparse
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import csv
from collections import defaultdict, deque
from datetime import datetime

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
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
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


class ComprehensiveLogger:
    """
    포괄적인 학습 로깅 시스템.

    기록하는 데이터:
    1. Step-level: 매 스텝 데이터 (action, reward, etc.)
    2. Episode-level: 에피소드 단위 통계
    3. Update-level: PPO 업데이트 단위 메트릭
    4. Map-level: 맵별 성능
    """

    def __init__(self, log_dir: Path, map_names: list):
        self.log_dir = log_dir
        self.map_names = map_names

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

        # CSV files
        self.csv_dir = log_dir / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        # Step-level CSV
        self.step_csv = self.csv_dir / "steps.csv"
        self.step_writer = None
        self.step_file = None

        # Episode-level CSV
        self.episode_csv = self.csv_dir / "episodes.csv"
        self.episode_writer = None
        self.episode_file = None

        # Update-level CSV
        self.update_csv = self.csv_dir / "updates.csv"
        self.update_writer = None
        self.update_file = None

        # Map-level CSV (per map)
        self.map_csv_writers = {}
        self.map_csv_files = {}
        for map_name in map_names:
            map_csv = self.csv_dir / f"map_{map_name}.csv"
            self.map_csv_writers[map_name] = None
            self.map_csv_files[map_name] = None

        # JSON for final results
        self.results_json = log_dir / "results.json"

        # In-memory buffers for analysis
        self.step_buffer = []
        self.episode_buffer = []
        self.update_buffer = []

        # Episode tracking
        self.current_episode_data = defaultdict(list)

        # Initialize CSV files
        self._init_csv_files()

    def _init_csv_files(self):
        """Initialize CSV files with headers."""

        # Step-level CSV
        self.step_file = open(self.step_csv, 'w', newline='')
        self.step_writer = csv.writer(self.step_file)
        self.step_writer.writerow([
            'global_step', 'update', 'episode', 'map_name',
            'action_mode', 'action_replan_idx',
            'reward_total', 'reward_event', 'reward_patrol', 'reward_safety', 'reward_efficiency',
            'has_event', 'nav_time', 'battery_level',
            'valid_action_count', 'patrol_valid', 'dispatch_valid',
            'collision', 'nav_success', 'event_resolved'
        ])

        # Episode-level CSV
        self.episode_file = open(self.episode_csv, 'w', newline='')
        self.episode_writer = csv.writer(self.episode_file)
        self.episode_writer.writerow([
            'episode', 'global_step', 'map_name',
            'return', 'length', 'duration',
            'event_success_rate', 'patrol_coverage', 'safety_violations',
            'avg_reward_event', 'avg_reward_patrol', 'avg_reward_safety', 'avg_reward_efficiency',
            'patrol_ratio', 'dispatch_ratio',
            'avg_nav_time', 'final_battery',
            'events_detected', 'events_responded', 'events_successful'
        ])

        # Update-level CSV
        self.update_file = open(self.update_csv, 'w', newline='')
        self.update_writer = csv.writer(self.update_file)
        self.update_writer.writerow([
            'update', 'global_step',
            'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clipfrac', 'explained_variance',
            'entropy_coef', 'learning_rate',
            'advantage_mean', 'advantage_std', 'advantage_min', 'advantage_max',
            'value_mean', 'return_mean', 'value_return_gap',
            'reward_raw_mean', 'reward_raw_std', 'reward_normalized_mean', 'reward_normalized_std',
            'grad_norm', 'fps'
        ])

        # Map-level CSVs
        for map_name in self.map_names:
            map_file = open(self.csv_dir / f"map_{map_name}.csv", 'w', newline='')
            map_writer = csv.writer(map_file)
            map_writer.writerow([
                'episode', 'global_step',
                'return', 'length',
                'event_success_rate', 'patrol_coverage',
                'patrol_ratio', 'dispatch_ratio'
            ])
            self.map_csv_writers[map_name] = map_writer
            self.map_csv_files[map_name] = map_file

    def log_step(self, global_step, update, episode, map_name, step_data):
        """Log single step data."""

        # TensorBoard
        self.writer.add_scalar(f"step/{map_name}/reward", step_data['reward_total'], global_step)
        self.writer.add_scalar(f"step/{map_name}/battery", step_data['battery_level'], global_step)

        # CSV
        self.step_writer.writerow([
            global_step, update, episode, map_name,
            step_data['action_mode'], step_data['action_replan_idx'],
            step_data['reward_total'], step_data['reward_event'],
            step_data['reward_patrol'], step_data['reward_safety'], step_data['reward_efficiency'],
            step_data['has_event'], step_data['nav_time'], step_data['battery_level'],
            step_data['valid_action_count'], step_data['patrol_valid'], step_data['dispatch_valid'],
            step_data['collision'], step_data['nav_success'], step_data['event_resolved']
        ])

        # Flush every 100 steps
        if global_step % 100 == 0:
            self.step_file.flush()

        # Buffer for episode aggregation
        self.current_episode_data['rewards'].append(step_data['reward_total'])
        self.current_episode_data['reward_event'].append(step_data['reward_event'])
        self.current_episode_data['reward_patrol'].append(step_data['reward_patrol'])
        self.current_episode_data['reward_safety'].append(step_data['reward_safety'])
        self.current_episode_data['reward_efficiency'].append(step_data['reward_efficiency'])
        self.current_episode_data['action_modes'].append(step_data['action_mode'])
        self.current_episode_data['nav_times'].append(step_data['nav_time'])

    def log_episode(self, episode, global_step, map_name, episode_metrics, duration):
        """Log episode-level data."""

        # Calculate episode statistics
        patrol_count = sum(1 for m in self.current_episode_data['action_modes'] if m == 0)
        dispatch_count = sum(1 for m in self.current_episode_data['action_modes'] if m == 1)
        total_actions = len(self.current_episode_data['action_modes'])

        patrol_ratio = patrol_count / total_actions if total_actions > 0 else 0
        dispatch_ratio = dispatch_count / total_actions if total_actions > 0 else 0

        avg_nav_time = np.mean(self.current_episode_data['nav_times']) if self.current_episode_data['nav_times'] else 0

        # TensorBoard
        self.writer.add_scalar("episode/return", episode_metrics.episode_return, global_step)
        self.writer.add_scalar("episode/length", episode_metrics.episode_length, global_step)
        self.writer.add_scalar("episode/event_success_rate", episode_metrics.event_success_rate, global_step)
        self.writer.add_scalar("episode/patrol_coverage", episode_metrics.patrol_coverage_ratio, global_step)
        self.writer.add_scalar("episode/safety_violations", episode_metrics.safety_violations, global_step)
        self.writer.add_scalar("episode/patrol_ratio", patrol_ratio, global_step)
        self.writer.add_scalar("episode/dispatch_ratio", dispatch_ratio, global_step)

        # Map-specific
        self.writer.add_scalar(f"episode_per_map/{map_name}/return", episode_metrics.episode_return, global_step)
        self.writer.add_scalar(f"episode_per_map/{map_name}/event_success_rate", episode_metrics.event_success_rate, global_step)
        self.writer.add_scalar(f"episode_per_map/{map_name}/patrol_coverage", episode_metrics.patrol_coverage_ratio, global_step)

        # CSV - Episode
        self.episode_writer.writerow([
            episode, global_step, map_name,
            episode_metrics.episode_return, episode_metrics.episode_length, duration,
            episode_metrics.event_success_rate, episode_metrics.patrol_coverage_ratio, episode_metrics.safety_violations,
            np.mean(self.current_episode_data['reward_event']),
            np.mean(self.current_episode_data['reward_patrol']),
            np.mean(self.current_episode_data['reward_safety']),
            np.mean(self.current_episode_data['reward_efficiency']),
            patrol_ratio, dispatch_ratio,
            avg_nav_time, episode_metrics.final_battery,
            episode_metrics.events_detected, episode_metrics.events_responded, episode_metrics.events_successful
        ])
        self.episode_file.flush()

        # CSV - Map-specific
        if map_name in self.map_csv_writers:
            self.map_csv_writers[map_name].writerow([
                episode, global_step,
                episode_metrics.episode_return, episode_metrics.episode_length,
                episode_metrics.event_success_rate, episode_metrics.patrol_coverage_ratio,
                patrol_ratio, dispatch_ratio
            ])
            self.map_csv_files[map_name].flush()

        # Clear episode buffer
        self.current_episode_data = defaultdict(list)

    def log_update(self, update, global_step, train_stats, diagnostics, fps):
        """Log PPO update data."""

        # TensorBoard - Training stats
        for key, value in train_stats.items():
            self.writer.add_scalar(f"train/{key}", value, global_step)

        # TensorBoard - Diagnostics
        for key, value in diagnostics.items():
            self.writer.add_scalar(f"diagnostics/{key}", value, global_step)

        # CSV
        self.update_writer.writerow([
            update, global_step,
            train_stats.get('policy_loss', 0),
            train_stats.get('value_loss', 0),
            train_stats.get('entropy', 0),
            train_stats.get('approx_kl', 0),
            train_stats.get('clipfrac', 0),
            train_stats.get('explained_variance', 0),
            diagnostics.get('entropy_coef', 0),
            diagnostics.get('learning_rate', 0),
            diagnostics.get('advantage_mean', 0),
            diagnostics.get('advantage_std', 0),
            diagnostics.get('advantage_min', 0),
            diagnostics.get('advantage_max', 0),
            diagnostics.get('value_mean', 0),
            diagnostics.get('return_mean', 0),
            diagnostics.get('value_return_gap', 0),
            diagnostics.get('reward_raw_mean', 0),
            diagnostics.get('reward_raw_std', 0),
            diagnostics.get('reward_normalized_mean', 0),
            diagnostics.get('reward_normalized_std', 0),
            diagnostics.get('grad_norm', 0),
            fps
        ])
        self.update_file.flush()

    def save_final_results(self, final_stats):
        """Save final training results to JSON."""
        with open(self.results_json, 'w') as f:
            json.dump(final_stats, f, indent=2)

    def close(self):
        """Close all files."""
        self.writer.close()
        self.step_file.close()
        self.episode_file.close()
        self.update_file.close()
        for f in self.map_csv_files.values():
            f.close()


def train_multi_map(
    env_wrapper,
    agent: PPOAgent,
    config: TrainingConfig,
    log_dir: Path,
) -> None:
    """Train PPO agent with comprehensive logging."""

    logger = ComprehensiveLogger(log_dir, env_wrapper.map_names)

    # Reward normalization
    reward_normalizer = RunningMeanStd(shape=())

    obs, info = env_wrapper.reset()
    episode_return = 0.0
    episode_length = 0
    episode_start_time = time.time()
    current_map = info["map_name"]
    episode_count = 0

    num_updates = config.total_timesteps // config.num_steps
    global_step = 0

    print("\n" + "=" * 80)
    print("🚀 Multi-Map Training with COMPREHENSIVE LOGGING")
    print("=" * 80)
    print(f"Maps: {env_wrapper.map_names}")
    print(f"Total Timesteps: {config.total_timesteps:,}")
    print(f"Updates: {num_updates:,}")
    print(f"\n📁 Logs saved to: {log_dir}")
    print(f"  - TensorBoard: {log_dir / 'tensorboard'}")
    print(f"  - CSV: {log_dir / 'csv'}")
    print(f"  - JSON: {log_dir / 'results.json'}")
    print("=" * 80 + "\n")

    start_time = time.time()

    for update in range(1, num_updates + 1):
        # Entropy annealing
        progress = update / num_updates
        current_entropy_coef = 0.1 * (1.0 - progress) + 0.01 * progress
        agent.training_config.entropy_coef = current_entropy_coef

        # Collect rollout
        rollout_raw_rewards = []

        for step in range(config.num_steps):
            global_step += 1

            # Get action with masking
            action_mask = info.get("action_mask", None)
            action, log_prob, value = agent.get_action(obs, action_mask=action_mask)

            # Environment step
            next_obs, reward, terminated, truncated, info = env_wrapper.step(action)

            # Extract step data for logging
            step_data = {
                'action_mode': int(action[0]),
                'action_replan_idx': int(action[1]),
                'reward_total': reward,
                'reward_event': info.get('reward_components', {}).get('event', 0),
                'reward_patrol': info.get('reward_components', {}).get('patrol', 0),
                'reward_safety': info.get('reward_components', {}).get('safety', 0),
                'reward_efficiency': info.get('reward_components', {}).get('efficiency', 0),
                'has_event': info.get('has_event', False),
                'nav_time': info.get('nav_time', 1.0),
                'battery_level': getattr(env_wrapper.current_state.robot if hasattr(env_wrapper, 'current_state') else env_wrapper.env.current_state.robot, 'battery_level', 1.0),
                'valid_action_count': np.sum(action_mask > 0.5) if action_mask is not None else 0,
                'patrol_valid': action_mask[:len(action_mask)//2].max() > 0.5 if action_mask is not None else True,
                'dispatch_valid': action_mask[len(action_mask)//2:].max() > 0.5 if action_mask is not None else True,
                'collision': info.get('collision', False),
                'nav_success': info.get('nav_success', True),
                'event_resolved': info.get('event_resolved', False),
            }

            # Log step
            logger.log_step(global_step, update, episode_count, current_map, step_data)

            # Store raw reward
            rollout_raw_rewards.append(reward)

            # Normalize reward
            if config.normalize_rewards:
                reward_normalizer.update(np.array([reward]))
                normalized_reward = (reward - reward_normalizer.mean) / np.sqrt(reward_normalizer.var + 1e-8)
                normalized_reward = np.clip(normalized_reward, -config.clip_rewards, config.clip_rewards)
            else:
                normalized_reward = reward

            episode_return += reward
            episode_length += 1

            # Store in buffer
            nav_time = info.get("nav_time", 1.0)
            agent.buffer.add(
                obs=obs,
                action=action,
                reward=normalized_reward,
                value=value,
                log_prob=log_prob,
                done=terminated,
                nav_time=nav_time,
                action_mask=action_mask,
            )

            obs = next_obs

            # Episode finished
            if terminated or truncated:
                episode_duration = time.time() - episode_start_time

                # Log episode
                if "map_episode_metrics" in info:
                    logger.log_episode(
                        episode_count,
                        global_step,
                        current_map,
                        info["map_episode_metrics"],
                        episode_duration
                    )

                # Reset
                episode_count += 1
                obs, info = env_wrapper.reset()
                current_map = info["map_name"]
                episode_return = 0.0
                episode_length = 0
                episode_start_time = time.time()

        # Compute final value
        action_mask = info.get("action_mask", None)
        final_action, final_log_prob, final_value = agent.get_action(obs, action_mask=action_mask)

        # PPO update
        train_stats = agent.update(last_value=final_value, last_done=False)

        # Collect diagnostics
        diagnostics = {
            'entropy_coef': current_entropy_coef,
            'learning_rate': agent.training_config.learning_rate,
            'reward_raw_mean': np.mean(rollout_raw_rewards),
            'reward_raw_std': np.std(rollout_raw_rewards),
            'reward_normalized_mean': reward_normalizer.mean,
            'reward_normalized_std': np.sqrt(reward_normalizer.var),
        }

        # Add advantage/value stats
        if hasattr(agent.buffer, 'advantages'):
            advantages = agent.buffer.advantages
            diagnostics.update({
                'advantage_mean': np.mean(advantages),
                'advantage_std': np.std(advantages),
                'advantage_min': np.min(advantages),
                'advantage_max': np.max(advantages),
            })

        if hasattr(agent.buffer, 'values') and hasattr(agent.buffer, 'returns'):
            values = agent.buffer.values
            returns = agent.buffer.returns
            diagnostics.update({
                'value_mean': np.mean(values),
                'return_mean': np.mean(returns),
                'value_return_gap': np.mean(returns - values),
            })

        # Calculate FPS
        elapsed = time.time() - start_time
        fps = global_step / elapsed

        # Log update
        logger.log_update(update, global_step, train_stats, diagnostics, fps)

        # Console logging
        if update % config.log_interval == 0:
            print(f"\nUpdate {update}/{num_updates} (Step {global_step:,}):")
            print(f"  FPS: {fps:.1f}")
            print(f"  Entropy coef: {current_entropy_coef:.4f}")

            # Map statistics
            stats = env_wrapper.get_map_statistics()
            print(f"\n  Per-Map Performance:")
            for map_name in env_wrapper.map_names:
                s = stats[map_name]
                if s["episodes"] > 0:
                    print(f"    {map_name}: Return={s['mean_return']:.1f}±{s['std_return']:.1f}, "
                          f"Success={100*s['mean_event_success']:.1f}%, Cov={100*s['mean_patrol_coverage']:.1f}%")

            print(f"\n  Training:")
            for key in ['policy_loss', 'value_loss', 'entropy', 'approx_kl', 'explained_variance']:
                if key in train_stats:
                    print(f"    {key}: {train_stats[key]:.6f}")

        # Save checkpoint
        if update % config.save_interval == 0:
            checkpoint_path = log_dir / "checkpoints" / f"update_{update}.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(str(checkpoint_path))
            print(f"\n  ✅ Saved: {checkpoint_path}")

        # Coverage evaluation
        if update % config.eval_interval == 0:
            coverage_dir = log_dir / "coverage" / f"update_{update}"
            coverage_dir.mkdir(parents=True, exist_ok=True)
            for map_name in env_wrapper.map_names:
                heatmap = env_wrapper.get_coverage_heatmap(map_name)
                if heatmap is not None:
                    np.save(coverage_dir / f"{map_name}_heatmap.npy", heatmap)

    # Final save
    final_path = log_dir / "checkpoints" / "final.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(final_path))

    # Save final results
    final_stats = {
        'training_config': config.__dict__,
        'total_steps': global_step,
        'total_updates': num_updates,
        'final_map_stats': env_wrapper.get_map_statistics(),
        'training_time': time.time() - start_time,
        'timestamp': datetime.now().isoformat(),
    }
    logger.save_final_results(final_stats)

    print(f"\n✅ Training complete!")
    print(f"  Final model: {final_path}")
    print(f"  Results: {logger.results_json}")

    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-Map Training with Comprehensive Logging")

    # Map selection
    parser.add_argument("--maps", type=str, nargs="+", default=None)
    parser.add_argument("--map-mode", type=str, default="random", choices=["random", "sequential", "curriculum"])

    # Training config
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.1)
    parser.add_argument("--value-loss-coef", type=float, default=1.0)

    # Experiment
    parser.add_argument("--experiment-name", type=str, default="multi_map_logged")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")

    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=50)

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env = create_multi_map_env(
        map_configs=args.maps,
        mode=args.map_mode,
        track_coverage=True,
    )

    # Create PPO agent
    obs_dim = env.observation_space.shape[0]
    num_replan_strategies = env.action_space.nvec[1]

    training_config = TrainingConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        experiment_name=args.experiment_name,
        seed=args.seed,
        cuda=(args.cuda and torch.cuda.is_available()),
        normalize_rewards=True,
        clip_rewards=5.0,
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    agent = PPOAgent(
        obs_dim=obs_dim,
        num_replan_strategies=num_replan_strategies,
        training_config=training_config,
        network_config=NetworkConfig(),
        device=device,
    )

    # Log directory
    log_dir = Path("runs") / args.experiment_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    training_config.save_yaml(str(log_dir / "training_config.yaml"))

    # Train!
    train_multi_map(env, agent, training_config, log_dir)

    env.close()


if __name__ == "__main__":
    main()
