#!/usr/bin/env python3
"""
Quick Phase 2 Validation - Run 50 episodes and analyze reward statistics
Reviewer 박용준: Fast validation without full training
"""

import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_dispatch.core.config import EnvConfig, RewardConfig
from rl_dispatch.env import PatrolEnv


def quick_validate():
    """Run 50 random episodes and collect statistics."""
    print("="*80)
    print("Quick Phase 2 Validation (50 Episodes)")
    print("="*80)

    config = EnvConfig.load_yaml("configs/map_campus.yaml")
    reward_config = RewardConfig()

    env = PatrolEnv(env_config=config, reward_config=reward_config)

    print(f"\n📍 Map: {config.map_width}m × {config.map_height}m campus")
    print(f"   Patrol points: {len(config.patrol_points)}")
    print(f"   Max steps: {config.max_episode_steps}")

    # Collect statistics
    episode_returns = []
    episode_lengths = []
    component_stats = defaultdict(list)
    nav_failures = []
    nav_times = []

    print(f"\n🚀 Running 50 episodes with random policy...")

    for episode in range(50):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0.0
        steps = 0
        ep_nav_failures = 0

        for step in range(config.max_episode_steps):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            # Collect component stats
            comp = info.get('reward_components', {})
            for key, val in comp.items():
                component_stats[key].append(val)

            # Nav stats
            nav_time = info.get('nav_time', 0.0)
            nav_times.append(nav_time)
            if nav_time < 1.0:
                ep_nav_failures += 1

            if terminated or truncated:
                break

        episode_returns.append(episode_reward)
        episode_lengths.append(steps)
        nav_failures.append(ep_nav_failures / max(steps, 1))

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/50: Return={episode_reward:.1f}, Steps={steps}")

    # Analysis
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Episode statistics
    print(f"\n📊 Episode Statistics (50 episodes):")
    print(f"   Return mean:  {np.mean(episode_returns):>10.1f}")
    print(f"   Return std:   {np.std(episode_returns):>10.1f}")
    print(f"   Return min:   {np.min(episode_returns):>10.1f}")
    print(f"   Return max:   {np.max(episode_returns):>10.1f}")
    print(f"   Length mean:  {np.mean(episode_lengths):>10.1f}")

    # Component statistics
    print(f"\n📊 Reward Component Statistics:")
    print(f"{'Component':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)

    for name in ['event', 'patrol', 'safety', 'efficiency']:
        if name in component_stats:
            values = component_stats[name]
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"{name:<15} {mean:>11.2f} {std:>11.2f} {min_val:>11.2f} {max_val:>11.2f}")

    # Nav statistics
    print(f"\n📊 Navigation Statistics:")
    print(f"   Nav failure rate: {np.mean(nav_failures)*100:.1f}% (target: <10%)")
    print(f"   Avg nav time:     {np.mean(nav_times):.1f}s")

    # Comparison
    print(f"\n📊 Phase 2 Improvement Assessment:")
    print(f"   {'Metric':<30} {'Target':<15} {'Actual':<15} {'Status':<10}")
    print("-" * 80)

    return_std = np.std(episode_returns)
    nav_fail_rate = np.mean(nav_failures) * 100

    # Check targets
    std_ok = "✅ PASS" if return_std < 40000 else "❌ FAIL"
    nav_ok = "✅ PASS" if nav_fail_rate < 10.0 else "❌ FAIL"

    # Component balance (check if std < 50 for all)
    balance_ok = all(np.std(component_stats[k]) < 50 for k in ['event', 'patrol', 'efficiency'])
    balance_status = "✅ PASS" if balance_ok else "❌ FAIL"

    print(f"   {'Return std':<30} {'<40k':<15} {f'{return_std:.0f}':<15} {std_ok}")
    print(f"   {'Nav failure rate':<30} {'<10%':<15} {f'{nav_fail_rate:.1f}%':<15} {nav_ok}")
    print(f"   {'Component balance':<30} {'Std<50':<15} {'Yes/No':<15} {balance_status}")

    print("\n" + "="*80)

    all_pass = (return_std < 40000) and (nav_fail_rate < 10.0) and balance_ok

    if all_pass:
        print("✅ VALIDATION SUCCESSFUL - Phase 2 improvements confirmed!")
    else:
        print("⚠️  Some targets not met - further tuning may be needed")
        if return_std >= 40000:
            print(f"   → Return std still high ({return_std:.0f}), target <40k")
            print("   → Suggestion: Check reward component balance in training")
        if nav_fail_rate >= 10.0:
            print(f"   → Nav failure rate {nav_fail_rate:.1f}%, target <10%")
            print("   → Suggestion: Phase 1 may need refinement")

    print("="*80)

    return all_pass


if __name__ == "__main__":
    success = quick_validate()
    sys.exit(0 if success else 1)
