#!/usr/bin/env python3
"""
Quick demo script to test the RL Dispatch system.

Runs a short training session and visualizes results.
Perfect for testing installation and understanding the system.

Usage:
    python scripts/demo.py
    python scripts/demo.py --steps 10000 --visualize
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from rl_dispatch.env import PatrolEnv
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.algorithms.baselines import B2_ThresholdDispatch
from rl_dispatch.core.config import EnvConfig, RewardConfig, TrainingConfig, NetworkConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quick demo of RL Dispatch")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save model")
    return parser.parse_args()


def run_demo(args):
    """Run quick demo."""
    print("=" * 80)
    print("RL DISPATCH MVP - QUICK DEMO")
    print("=" * 80)
    print()

    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create simple environment
    env_config = EnvConfig(
        map_width=50.0,
        map_height=50.0,
        max_episode_steps=100,
        event_generation_rate=3.0,
    )
    reward_config = RewardConfig()
    training_config = TrainingConfig(
        total_timesteps=args.steps,
        num_steps=512,
        learning_rate=3e-4,
    )
    network_config = NetworkConfig()

    print(f"Environment: {env_config.map_width}x{env_config.map_height}m")
    print(f"Training steps: {args.steps:,}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    # Create environment and agent
    env = PatrolEnv(env_config=env_config, reward_config=reward_config)
    agent = PPOAgent(
        obs_dim=77,
        num_replan_strategies=6,
        training_config=training_config,
        network_config=network_config,
    )

    print("Starting training...")
    print("-" * 80)

    # Training loop
    obs, _ = env.reset(seed=args.seed)
    episode_returns = []
    episode_lengths = []
    current_return = 0
    current_length = 0

    for step in range(args.steps):
        # Collect experience
        action, log_prob, value = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.buffer.add(obs, action, log_prob, reward, value, done)
        current_return += reward
        current_length += 1
        obs = next_obs

        # Episode end
        if done:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)

            if len(episode_returns) % 5 == 0:
                print(f"Episode {len(episode_returns):3d} | "
                      f"Return: {current_return:7.2f} | "
                      f"Length: {current_length:3d}")

            obs, _ = env.reset()
            current_return = 0
            current_length = 0

        # Update when buffer full
        if agent.buffer.is_full:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
                last_value = agent.network.get_value(obs_t).item()

            agent.buffer.compute_returns_and_advantages(last_value, done)
            stats = agent.update()

            if agent.update_count % 5 == 0:
                print(f"\nUpdate {agent.update_count:2d} | "
                      f"Policy Loss: {stats['policy_loss']:.4f} | "
                      f"Value Loss: {stats['value_loss']:.4f} | "
                      f"Entropy: {stats['entropy']:.4f}\n")

    print("-" * 80)
    print(f"Training complete! {len(episode_returns)} episodes")
    print(f"Mean return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print()

    # Evaluate trained agent
    print("Evaluating trained agent...")
    agent.eval_mode()
    eval_returns = []

    for _ in range(10):
        obs, _ = env.reset()
        episode_return = 0
        done = False

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        eval_returns.append(episode_return)

    print(f"Evaluation (10 episodes): {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")
    print()

    # Compare with baseline
    print("Comparing with baseline policy (B2_ThresholdDispatch)...")
    baseline = B2_ThresholdDispatch()
    baseline_returns = []

    for _ in range(10):
        obs, _ = env.reset()
        episode_return = 0
        done = False

        while not done:
            state = env.current_state
            action = baseline.select_action(state)
            action_array = np.array([action.mode, action.replan_idx])
            obs, reward, terminated, truncated, _ = env.step(action_array)
            episode_return += reward
            done = terminated or truncated

        baseline_returns.append(episode_return)

    print(f"Baseline (10 episodes): {np.mean(baseline_returns):.2f} ± {np.std(baseline_returns):.2f}")
    print()

    # Summary
    print("=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    print(f"Training episodes: {len(episode_returns)}")
    print(f"Mean training return: {np.mean(episode_returns[-10:]):.2f}")
    print(f"Learned agent eval: {np.mean(eval_returns):.2f}")
    print(f"Baseline eval: {np.mean(baseline_returns):.2f}")
    improvement = ((np.mean(eval_returns) - np.mean(baseline_returns)) / abs(np.mean(baseline_returns))) * 100
    print(f"Improvement over baseline: {improvement:+.1f}%")
    print()

    # Save model
    if args.save_model:
        agent.save(args.save_model)
        print(f"Model saved to: {args.save_model}")
        print()

    # Visualization
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            from rl_dispatch.utils.visualization import TrainingVisualizer

            print("Creating visualizations...")
            viz = TrainingVisualizer()

            # Learning curve
            fig = viz.plot_learning_curve(
                episode_returns,
                title="Demo Learning Curve",
                window=max(1, len(episode_returns) // 10)
            )
            plt.savefig("demo_learning_curve.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Baseline comparison
            comparison_data = {
                "Learned Agent": {"mean_return": np.mean(eval_returns)},
                "Baseline B2": {"mean_return": np.mean(baseline_returns)},
            }
            fig = viz.plot_baseline_comparison(
                comparison_data,
                metrics=["mean_return"],
                title="Agent vs Baseline"
            )
            plt.savefig("demo_comparison.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            print("Visualizations saved:")
            print("  - demo_learning_curve.png")
            print("  - demo_comparison.png")
            print()

        except ImportError:
            print("Matplotlib not available, skipping visualization")
            print()

    print("=" * 80)
    print("Demo complete! 🎉")
    print()
    print("Next steps:")
    print("  - Full training: python scripts/train.py --cuda")
    print("  - See QUICK_START.md for more examples")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args)
