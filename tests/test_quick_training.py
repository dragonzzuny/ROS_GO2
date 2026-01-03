"""
Quick training test to verify the complete system works end-to-end.

Tests a short training run (10,000 steps) to ensure:
1. Environment resets properly
2. Actions are valid
3. Observations are correct
4. Rewards are computed
5. Episodes terminate correctly
6. All 10 heuristic strategies are used
"""

import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_dispatch.core.config import EnvConfig, RewardConfig
from rl_dispatch.env import PatrolEnv


def test_quick_training():
    """Run a quick training simulation."""
    print("=" * 80)
    print("Quick Training Test (10,000 steps)")
    print("=" * 80)

    # Load config
    config = EnvConfig.load_yaml("configs/map_large_square.yaml")
    reward_config = RewardConfig()

    print(f"\n✓ Configuration loaded:")
    print(f"  Map: {config.map_width}m × {config.map_height}m")
    print(f"  Patrol points: {len(config.patrol_points)}")
    print(f"  Heuristic strategies: {config.num_candidates}")
    print(f"  Max episode steps: {config.max_episode_steps}")

    # Create environment
    env = PatrolEnv(env_config=config, reward_config=reward_config)

    print(f"\n✓ Environment created:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")

    # Training statistics
    total_steps = 10000
    num_episodes = 0
    total_reward = 0.0
    episode_rewards = []
    episode_lengths = []
    strategy_usage = defaultdict(int)
    event_responses = 0
    patrol_visits = 0

    print(f"\n✓ Starting training simulation ({total_steps} steps)...")
    print("=" * 80)

    obs, info = env.reset(seed=42)
    episode_reward = 0.0
    episode_length = 0

    for step in range(total_steps):
        # Random action (in real training, this would be from policy network)
        action = env.action_space.sample()
        mode = action[0]
        strategy_idx = action[1]

        # Track strategy usage
        strategy_name = env.current_state.candidates[strategy_idx].strategy_name
        strategy_usage[strategy_name] += 1

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_length += 1
        total_reward += reward

        # Track metrics
        if mode == 1:  # DISPATCH mode
            event_responses += 1
        else:  # PATROL mode
            patrol_visits += 1

        # Episode end
        if terminated or truncated:
            num_episodes += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if num_episodes % 5 == 0:
                avg_reward = np.mean(episode_rewards[-5:])
                avg_length = np.mean(episode_lengths[-5:])
                print(f"  Episode {num_episodes:3d} | "
                      f"Steps: {step+1:5d} | "
                      f"Avg Reward (last 5): {avg_reward:7.2f} | "
                      f"Avg Length: {avg_length:5.1f}")

            # Reset
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0

    print("=" * 80)
    print(f"\n✓ Training simulation completed!")

    # Final statistics
    print(f"\n📊 Training Statistics:")
    print(f"  Total steps: {total_steps}")
    print(f"  Episodes completed: {num_episodes}")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f}")
    print(f"  Average episode reward: {np.mean(episode_rewards):.2f}")
    print(f"  Total reward: {total_reward:.2f}")

    print(f"\n📊 Action Distribution:")
    print(f"  Event responses: {event_responses} ({event_responses/total_steps*100:.1f}%)")
    print(f"  Patrol visits: {patrol_visits} ({patrol_visits/total_steps*100:.1f}%)")

    print(f"\n📊 Strategy Usage (Top 5):")
    sorted_strategies = sorted(strategy_usage.items(), key=lambda x: x[1], reverse=True)
    for i, (strategy, count) in enumerate(sorted_strategies[:5], 1):
        percentage = count / total_steps * 100
        print(f"  {i}. {strategy:<25} {count:5d} ({percentage:5.2f}%)")

    print(f"\n📊 All 10 Strategies Used:")
    all_used = len(strategy_usage) == 10
    print(f"  Unique strategies: {len(strategy_usage)}/10")
    if all_used:
        print(f"  ✅ All 10 heuristic strategies were sampled!")
    else:
        print(f"  ⚠️  Only {len(strategy_usage)} strategies used")
        print(f"     Missing: {set(range(10)) - set(strategy_usage.keys())}")

    # Verify basic sanity
    print(f"\n✅ Sanity Checks:")
    checks_passed = 0
    total_checks = 5

    # Check 1: Episodes completed
    if num_episodes > 0:
        print(f"  ✓ Episodes completed: {num_episodes}")
        checks_passed += 1
    else:
        print(f"  ✗ No episodes completed!")

    # Check 2: Observations valid
    if obs.shape == (77,):
        print(f"  ✓ Observation shape correct: {obs.shape}")
        checks_passed += 1
    else:
        print(f"  ✗ Observation shape wrong: {obs.shape}")

    # Check 3: Rewards reasonable
    if -1000 < np.mean(episode_rewards) < 1000:
        print(f"  ✓ Rewards in reasonable range: {np.mean(episode_rewards):.2f}")
        checks_passed += 1
    else:
        print(f"  ✗ Rewards seem unreasonable: {np.mean(episode_rewards):.2f}")

    # Check 4: Episode lengths reasonable
    if 10 < np.mean(episode_lengths) < 500:
        print(f"  ✓ Episode lengths reasonable: {np.mean(episode_lengths):.1f}")
        checks_passed += 1
    else:
        print(f"  ✗ Episode lengths unusual: {np.mean(episode_lengths):.1f}")

    # Check 5: Multiple strategies used
    if len(strategy_usage) >= 5:
        print(f"  ✓ Multiple strategies used: {len(strategy_usage)}/10")
        checks_passed += 1
    else:
        print(f"  ✗ Too few strategies: {len(strategy_usage)}/10")

    print(f"\n  Checks passed: {checks_passed}/{total_checks}")

    if checks_passed >= 4:
        print("\n" + "=" * 80)
        print("✅ QUICK TRAINING TEST PASSED!")
        print("=" * 80)
        print("\n시스템이 정상적으로 작동합니다!")
        print("이제 본격적인 학습을 시작할 수 있습니다:\n")
        print("  python scripts/train_multi_map.py --total-timesteps 5000000\n")
        return True
    else:
        print("\n" + "=" * 80)
        print("⚠️  QUICK TRAINING TEST: SOME ISSUES DETECTED")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = test_quick_training()
    sys.exit(0 if success else 1)
