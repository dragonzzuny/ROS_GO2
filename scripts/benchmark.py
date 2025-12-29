#!/usr/bin/env python3
"""
Benchmark script to compare all baseline policies.

Runs comprehensive evaluation of all 5 baseline policies
and creates comparison report.

Usage:
    python scripts/benchmark.py --episodes 100
    python scripts/benchmark.py --episodes 100 --save-results results/benchmark.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from rl_dispatch.env import PatrolEnv
from rl_dispatch.algorithms.baselines import (
    B0_AlwaysPatrol,
    B1_AlwaysDispatch,
    B2_ThresholdDispatch,
    B3_UrgencyBased,
    B4_HeuristicPolicy,
)
from rl_dispatch.core.config import EnvConfig, RewardConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark baseline policies")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-results", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    return parser.parse_args()


def evaluate_policy(policy, env, num_episodes: int) -> Dict:
    """Evaluate single policy."""
    episode_returns = []
    episode_lengths = []
    events_detected = []
    events_responded = []
    events_successful = []
    avg_delays = []
    coverage_ratios = []
    safety_violations = []
    total_distances = []

    print(f"  Evaluating {policy.name}...", end=" ", flush=True)
    start_time = time.time()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0
        done = False

        while not done:
            state = env.current_state
            action = policy.select_action(state)
            action_array = np.array([action.mode, action.replan_idx])
            obs, reward, terminated, truncated, info = env.step(action_array)
            episode_return += reward
            done = terminated or truncated

        if "episode" in info:
            episode_returns.append(info["episode"]["r"])
            episode_lengths.append(info["episode"]["l"])
            events_detected.append(info["episode"]["events_detected"])
            events_responded.append(info["episode"]["events_responded"])
            events_successful.append(info["episode"]["events_successful"])
            avg_delays.append(info["episode"]["avg_event_delay"])
            coverage_ratios.append(info["episode"]["patrol_coverage"])
            safety_violations.append(info["episode"]["safety_violations"])
            total_distances.append(info["episode"]["total_distance"])

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s")

    return {
        "policy_name": policy.name,
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "min_return": float(np.min(episode_returns)),
        "max_return": float(np.max(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_events_detected": float(np.mean(events_detected)),
        "mean_events_responded": float(np.mean(events_responded)),
        "mean_events_successful": float(np.mean(events_successful)),
        "event_response_rate": float(np.mean(events_responded) / max(np.mean(events_detected), 1)),
        "event_success_rate": float(np.mean(events_successful) / max(np.mean(events_detected), 1)),
        "mean_event_delay": float(np.mean(avg_delays)),
        "mean_patrol_coverage": float(np.mean(coverage_ratios)),
        "std_patrol_coverage": float(np.std(coverage_ratios)),
        "mean_safety_violations": float(np.mean(safety_violations)),
        "mean_total_distance": float(np.mean(total_distances)),
    }


def print_results_table(results: List[Dict]) -> None:
    """Print formatted results table."""
    print()
    print("=" * 120)
    print("BASELINE POLICY BENCHMARK RESULTS")
    print("=" * 120)
    print()

    # Header
    print(f"{'Policy':<25} {'Return':<18} {'Success Rate':<15} {'Coverage':<15} {'Violations':<12}")
    print("-" * 120)

    # Rows
    for result in results:
        print(
            f"{result['policy_name']:<25} "
            f"{result['mean_return']:>8.2f} ± {result['std_return']:<6.2f} "
            f"{result['event_success_rate']:>13.1%} "
            f"{result['mean_patrol_coverage']:>13.1%} "
            f"{result['mean_safety_violations']:>10.2f}"
        )

    print("=" * 120)
    print()

    # Detailed breakdown
    print("DETAILED METRICS")
    print("-" * 120)
    for result in results:
        print(f"\n{result['policy_name']}:")
        print(f"  Return: {result['mean_return']:.2f} (range: [{result['min_return']:.2f}, {result['max_return']:.2f}])")
        print(f"  Episode Length: {result['mean_length']:.1f} steps")
        print(f"  Events per episode: {result['mean_events_detected']:.1f}")
        print(f"  Response Rate: {result['event_response_rate']:.1%}")
        print(f"  Success Rate: {result['event_success_rate']:.1%}")
        print(f"  Avg Event Delay: {result['mean_event_delay']:.1f}s")
        print(f"  Patrol Coverage: {result['mean_patrol_coverage']:.1%} ± {result['std_patrol_coverage']:.1%}")
        print(f"  Safety Violations: {result['mean_safety_violations']:.2f} per episode")
        print(f"  Total Distance: {result['mean_total_distance']:.1f}m")

    print()
    print("=" * 120)


def main():
    """Main benchmark function."""
    args = parse_args()

    print("=" * 120)
    print("BASELINE POLICY BENCHMARK")
    print("=" * 120)
    print(f"Episodes per policy: {args.episodes}")
    print(f"Random seed: {args.seed}")
    print()

    # Create environment
    env_config = EnvConfig()
    reward_config = RewardConfig()
    env = PatrolEnv(env_config=env_config, reward_config=reward_config)
    env.reset(seed=args.seed)

    # Create all baseline policies
    policies = [
        B0_AlwaysPatrol(),
        B1_AlwaysDispatch(),
        B2_ThresholdDispatch(),
        B3_UrgencyBased(seed=args.seed),
        B4_HeuristicPolicy(),
    ]

    print(f"Evaluating {len(policies)} baseline policies...")
    print()

    # Evaluate all policies
    results = []
    for policy in policies:
        result = evaluate_policy(policy, env, args.episodes)
        results.append(result)

    # Print results
    print_results_table(results)

    # Identify best policies
    best_return_idx = np.argmax([r["mean_return"] for r in results])
    best_coverage_idx = np.argmax([r["mean_patrol_coverage"] for r in results])
    best_success_idx = np.argmax([r["event_success_rate"] for r in results])

    print("BEST PERFORMERS")
    print("-" * 120)
    print(f"  Best Return: {results[best_return_idx]['policy_name']} ({results[best_return_idx]['mean_return']:.2f})")
    print(f"  Best Coverage: {results[best_coverage_idx]['policy_name']} ({results[best_coverage_idx]['mean_patrol_coverage']:.1%})")
    print(f"  Best Success Rate: {results[best_success_idx]['policy_name']} ({results[best_success_idx]['event_success_rate']:.1%})")
    print()

    # Save results
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {save_path}")
        print()

    # Visualization
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            from rl_dispatch.utils.visualization import TrainingVisualizer

            print("Creating visualizations...")

            viz = TrainingVisualizer()
            baseline_dict = {r["policy_name"]: r for r in results}

            # Comparison plot
            fig = viz.plot_baseline_comparison(
                baseline_dict,
                metrics=["mean_return", "event_success_rate", "mean_patrol_coverage"],
                title="Baseline Policy Comparison"
            )
            plt.savefig("baseline_comparison.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            print("Visualization saved: baseline_comparison.png")
            print()

        except ImportError:
            print("Matplotlib not available, skipping visualization")
            print()

    env.close()

    print("=" * 120)
    print("Benchmark complete! 🎉")
    print("=" * 120)


if __name__ == "__main__":
    main()
