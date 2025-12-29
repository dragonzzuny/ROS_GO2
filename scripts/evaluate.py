#!/usr/bin/env python3
"""
Evaluation script for trained PPO agent.

This script loads a trained model and evaluates its performance across
multiple episodes, computing detailed statistics and comparisons.

Usage:
    python scripts/evaluate.py --model checkpoints/run_name/final_model.pth --episodes 100
    python scripts/evaluate.py --model models/best.pth --render --episodes 10
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from rl_dispatch.env import PatrolEnv
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.core.config import EnvConfig, RewardConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO agent"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (None for random)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes (not implemented)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    return parser.parse_args()


def evaluate_agent(
    agent: PPOAgent,
    env: PatrolEnv,
    num_episodes: int,
    deterministic: bool = True,
    render: bool = False,
) -> Dict[str, any]:
    """
    Evaluate agent over multiple episodes.

    Args:
        agent: Trained PPO agent
        env: Evaluation environment
        num_episodes: Number of episodes
        deterministic: Use deterministic policy
        render: Render episodes

    Returns:
        Dictionary of evaluation statistics
    """
    agent.eval_mode()

    # Episode-level metrics
    episode_returns = []
    episode_lengths = []
    events_detected_list = []
    events_responded_list = []
    events_successful_list = []
    avg_delays_list = []
    coverage_ratios_list = []
    safety_violations_list = []
    total_distances_list = []

    # Step-level metrics
    all_reward_components = {
        "event": [],
        "patrol": [],
        "safety": [],
        "efficiency": [],
    }

    print(f"Evaluating for {num_episodes} episodes...")
    print("-" * 80)

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0
        done = False

        while not done:
            # Select action
            action, _, _ = agent.get_action(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated

            # Collect reward components
            if "reward_components" in info:
                for key in all_reward_components.keys():
                    all_reward_components[key].append(info["reward_components"][key])

            if render:
                env.render()

        # Collect episode metrics
        if "episode" in info:
            episode_returns.append(info["episode"]["r"])
            episode_lengths.append(info["episode"]["l"])
            events_detected_list.append(info["episode"]["events_detected"])
            events_responded_list.append(info["episode"]["events_responded"])
            events_successful_list.append(info["episode"]["events_successful"])
            avg_delays_list.append(info["episode"]["avg_event_delay"])
            coverage_ratios_list.append(info["episode"]["patrol_coverage"])
            safety_violations_list.append(info["episode"]["safety_violations"])
            total_distances_list.append(info["episode"]["total_distance"])

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes")

    print("-" * 80)
    print("Evaluation complete!")
    print()

    # Compute statistics
    results = {
        # Returns and lengths
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "min_return": np.min(episode_returns),
        "max_return": np.max(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),

        # Event handling
        "mean_events_detected": np.mean(events_detected_list),
        "mean_events_responded": np.mean(events_responded_list),
        "mean_events_successful": np.mean(events_successful_list),
        "event_response_rate": np.mean(events_responded_list) / max(np.mean(events_detected_list), 1),
        "event_success_rate": np.mean(events_successful_list) / max(np.mean(events_detected_list), 1),
        "mean_event_delay": np.mean(avg_delays_list),
        "std_event_delay": np.std(avg_delays_list),

        # Patrol coverage
        "mean_patrol_coverage": np.mean(coverage_ratios_list),
        "std_patrol_coverage": np.std(coverage_ratios_list),
        "min_patrol_coverage": np.min(coverage_ratios_list),

        # Safety
        "mean_safety_violations": np.mean(safety_violations_list),
        "episodes_with_violations": np.sum(np.array(safety_violations_list) > 0),

        # Efficiency
        "mean_total_distance": np.mean(total_distances_list),
        "std_total_distance": np.std(total_distances_list),

        # Reward components (averaged across all steps)
        "mean_event_reward": np.mean(all_reward_components["event"]),
        "mean_patrol_reward": np.mean(all_reward_components["patrol"]),
        "mean_safety_reward": np.mean(all_reward_components["safety"]),
        "mean_efficiency_reward": np.mean(all_reward_components["efficiency"]),

        # Raw data
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
    }

    return results


def print_results(results: Dict[str, any]) -> None:
    """Print evaluation results in formatted table."""
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()

    print("Episode Statistics:")
    print(f"  Mean Return:        {results['mean_return']:8.2f} ± {results['std_return']:.2f}")
    print(f"  Return Range:       [{results['min_return']:.2f}, {results['max_return']:.2f}]")
    print(f"  Mean Length:        {results['mean_length']:8.1f} ± {results['std_length']:.1f} steps")
    print()

    print("Event Handling:")
    print(f"  Events Detected:    {results['mean_events_detected']:8.2f} per episode")
    print(f"  Response Rate:      {results['event_response_rate']:8.1%}")
    print(f"  Success Rate:       {results['event_success_rate']:8.1%}")
    print(f"  Avg Delay:          {results['mean_event_delay']:8.2f} ± {results['std_event_delay']:.2f} seconds")
    print()

    print("Patrol Coverage:")
    print(f"  Mean Coverage:      {results['mean_patrol_coverage']:8.1%} ± {results['std_patrol_coverage']:.1%}")
    print(f"  Min Coverage:       {results['min_patrol_coverage']:8.1%}")
    print()

    print("Safety:")
    print(f"  Mean Violations:    {results['mean_safety_violations']:8.2f} per episode")
    print(f"  Episodes w/ Viol:   {results['episodes_with_violations']:8d}")
    print()

    print("Efficiency:")
    print(f"  Mean Distance:      {results['mean_total_distance']:8.2f} ± {results['std_total_distance']:.2f} meters")
    print()

    print("Reward Components (mean per step):")
    print(f"  Event Reward:       {results['mean_event_reward']:8.4f}")
    print(f"  Patrol Reward:      {results['mean_patrol_reward']:8.4f}")
    print(f"  Safety Reward:      {results['mean_safety_reward']:8.4f}")
    print(f"  Efficiency Reward:  {results['mean_efficiency_reward']:8.4f}")
    print()
    print("=" * 80)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load checkpoint
    print(f"Loading model from: {args.model}")
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load configurations from checkpoint
    training_config = checkpoint.get("training_config")
    network_config = checkpoint.get("network_config")

    # Try to load env and reward configs from checkpoint directory
    checkpoint_dir = checkpoint_path.parent
    env_config_path = checkpoint_dir / "env_config.yaml"
    reward_config_path = checkpoint_dir / "reward_config.yaml"

    if env_config_path.exists():
        env_config = EnvConfig.load_yaml(str(env_config_path))
        print(f"Loaded env config from: {env_config_path}")
    else:
        env_config = EnvConfig()
        print("Using default env config")

    if reward_config_path.exists():
        reward_config = RewardConfig.load_yaml(str(reward_config_path))
        print(f"Loaded reward config from: {reward_config_path}")
    else:
        reward_config = RewardConfig()
        print("Using default reward config")

    # Create environment
    env = PatrolEnv(env_config=env_config, reward_config=reward_config)
    if args.seed is not None:
        env.reset(seed=args.seed)

    # Create agent and load weights
    agent = PPOAgent(
        obs_dim=77,
        num_replan_strategies=env_config.num_candidates,
        training_config=training_config,
        network_config=network_config,
        device="cpu",
    )
    agent.load(str(checkpoint_path))
    print("Model loaded successfully!")
    print()

    # Evaluate
    results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render,
    )

    # Print results
    print_results(results)

    # Save results if requested
    if args.save_results:
        import json
        results_path = Path(args.save_results)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove raw data for JSON serialization
        save_results = {k: v for k, v in results.items()
                       if not isinstance(v, (list, np.ndarray))}

        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    env.close()


if __name__ == "__main__":
    main()
