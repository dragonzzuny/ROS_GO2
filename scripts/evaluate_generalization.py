#!/usr/bin/env python3
"""
Generalization Evaluation Script

학습된 정책의 일반화 성능을 평가합니다.
- 학습에 사용된 맵들에서의 성능
- 학습에 사용되지 않은 새로운 맵에서의 성능
- 맵 간 성능 분산 분석

Usage:
    python scripts/evaluate_generalization.py --model checkpoints/final.pth --episodes 50
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_dispatch.env import create_multi_map_env, PatrolEnv
from rl_dispatch.algorithms.ppo import PPOPolicy
from rl_dispatch.core.config import EnvConfig


def evaluate_on_map(
    env,
    policy: PPOPolicy,
    map_name: str,
    episodes: int,
) -> Dict[str, float]:
    """
    Evaluate policy on a specific map.

    Returns:
        metrics: Dictionary of performance metrics
    """
    returns = []
    event_success_rates = []
    patrol_coverages = []
    response_times = []
    episode_lengths = []

    for ep in range(episodes):
        obs, info = env.reset(options={"map_name": map_name})
        episode_return = 0.0
        done = False
        steps = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated
            steps += 1

        returns.append(episode_return)
        episode_lengths.append(steps)

        if "map_episode_metrics" in info:
            metrics = info["map_episode_metrics"]
            event_success_rates.append(metrics.event_success_rate)
            patrol_coverages.append(metrics.patrol_coverage)
            response_times.append(metrics.avg_response_time)

    return {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_event_success": np.mean(event_success_rates),
        "std_event_success": np.std(event_success_rates),
        "mean_patrol_coverage": np.mean(patrol_coverages),
        "std_patrol_coverage": np.std(patrol_coverages),
        "mean_response_time": np.mean(response_times),
        "std_response_time": np.std(response_times),
        "mean_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "returns": returns,
    }


def create_comparison_plots(
    results: Dict[str, Dict[str, float]],
    output_dir: Path,
):
    """Create visualization comparing performance across maps."""

    map_names = list(results.keys())
    n_maps = len(map_names)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Return comparison
    ax = axes[0, 0]
    means = [results[m]["mean_return"] for m in map_names]
    stds = [results[m]["std_return"] for m in map_names]

    x = np.arange(n_maps)
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(map_names, rotation=45, ha='right')
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title('Return per Map', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 2. Event success rate
    ax = axes[0, 1]
    means = [100 * results[m]["mean_event_success"] for m in map_names]
    stds = [100 * results[m]["std_event_success"] for m in map_names]

    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='forestgreen')
    ax.set_xticks(x)
    ax.set_xticklabels(map_names, rotation=45, ha='right')
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Event Success Rate per Map', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    # 3. Patrol coverage
    ax = axes[1, 0]
    means = [100 * results[m]["mean_patrol_coverage"] for m in map_names]
    stds = [100 * results[m]["std_patrol_coverage"] for m in map_names]

    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels(map_names, rotation=45, ha='right')
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Patrol Coverage per Map', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    # 4. Response time
    ax = axes[1, 1]
    means = [results[m]["mean_response_time"] for m in map_names]
    stds = [results[m]["std_response_time"] for m in map_names]

    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='crimson')
    ax.set_xticks(x)
    ax.set_xticklabels(map_names, rotation=45, ha='right')
    ax.set_ylabel('Response Time (s)', fontsize=12)
    ax.set_title('Avg Event Response Time per Map', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        'Policy Generalization Performance Across Maps',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()

    # Save
    plot_path = output_dir / "generalization_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison plot: {plot_path}")

    # Additional: Return distribution violin plot
    fig, ax = plt.subplots(figsize=(14, 6))

    returns_data = [results[m]["returns"] for m in map_names]

    parts = ax.violinplot(
        returns_data,
        positions=x,
        widths=0.7,
        showmeans=True,
        showmedians=True,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(map_names, rotation=45, ha='right')
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title('Return Distribution per Map (Violin Plot)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()

    violin_path = output_dir / "return_distribution.png"
    plt.savefig(violin_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved distribution plot: {violin_path}")

    plt.show()


def print_results_table(results: Dict[str, Dict[str, float]]):
    """Print formatted results table."""

    print("\n" + "=" * 120)
    print("Generalization Evaluation Results")
    print("=" * 120)
    print(f"{'Map':<25} {'Return':<20} {'Event Success':<20} {'Patrol Coverage':<20} {'Response Time':<15}")
    print("-" * 120)

    for map_name, metrics in results.items():
        return_str = f"{metrics['mean_return']:.1f} ± {metrics['std_return']:.1f}"
        success_str = f"{100*metrics['mean_event_success']:.1f}% ± {100*metrics['std_event_success']:.1f}%"
        coverage_str = f"{100*metrics['mean_patrol_coverage']:.1f}% ± {100*metrics['std_patrol_coverage']:.1f}%"
        time_str = f"{metrics['mean_response_time']:.1f}s ± {metrics['std_response_time']:.1f}s"

        print(f"{map_name:<25} {return_str:<20} {success_str:<20} {coverage_str:<20} {time_str:<15}")

    print("-" * 120)

    # Overall statistics
    all_returns = [m["mean_return"] for m in results.values()]
    all_success = [m["mean_event_success"] for m in results.values()]
    all_coverage = [m["mean_patrol_coverage"] for m in results.values()]

    print(f"\n{'Overall Statistics':<25}")
    print(f"  Average Return: {np.mean(all_returns):.1f} (std across maps: {np.std(all_returns):.1f})")
    print(f"  Average Event Success: {100*np.mean(all_success):.1f}% (std: {100*np.std(all_success):.1f}%)")
    print(f"  Average Patrol Coverage: {100*np.mean(all_coverage):.1f}% (std: {100*np.std(all_coverage):.1f}%)")

    print(f"\n{'Generalization Metrics':<25}")
    print(f"  Return Variance: {np.var(all_returns):.1f} (lower is better)")
    print(f"  Worst Map Return: {np.min(all_returns):.1f}")
    print(f"  Best Map Return: {np.max(all_returns):.1f}")
    print(f"  Return Range: {np.max(all_returns) - np.min(all_returns):.1f}")

    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate policy generalization across multiple maps"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained PPO model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes to run per map",
    )
    parser.add_argument(
        "--maps",
        type=str,
        nargs="+",
        default=None,
        help="Specific maps to evaluate on (uses all if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/generalization",
        help="Directory for output files",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Load policy
    print(f"Loading policy from {args.model}...")
    policy = PPOPolicy.load(args.model)
    print("✅ Policy loaded!")

    # Create environment
    print("\nCreating multi-map environment...")
    env = create_multi_map_env(
        map_configs=args.maps,
        track_coverage=False,  # Don't need coverage for evaluation
    )

    print(f"✅ Loaded {len(env.map_names)} maps:")
    for name in env.map_names:
        config = env.env_configs[name]
        print(f"  - {name}: {config.map_width}×{config.map_height}m, "
              f"{config.num_patrol_points} points")

    # Evaluate on each map
    results = {}

    print("\n" + "=" * 80)
    print("Running Evaluation")
    print("=" * 80)

    for map_name in env.map_names:
        print(f"\nEvaluating on {map_name}... ({args.episodes} episodes)")

        metrics = evaluate_on_map(env, policy, map_name, args.episodes)
        results[map_name] = metrics

        print(f"  Return: {metrics['mean_return']:.1f} ± {metrics['std_return']:.1f}")
        print(f"  Event Success: {100*metrics['mean_event_success']:.1f}%")
        print(f"  Patrol Coverage: {100*metrics['mean_patrol_coverage']:.1f}%")

    # Print results table
    print_results_table(results)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    if args.save_json:
        # Convert numpy arrays to lists for JSON
        json_results = {}
        for map_name, metrics in results.items():
            json_results[map_name] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in metrics.items()
            }

        json_path = output_dir / "generalization_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"✅ Saved results to: {json_path}")

    # Create visualizations
    create_comparison_plots(results, output_dir)

    env.close()

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
