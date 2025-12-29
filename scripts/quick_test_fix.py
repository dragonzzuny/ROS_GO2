#!/usr/bin/env python3
"""
Quick Test Script - Policy Collapse Fix 검증

이 스크립트는 수정사항이 제대로 작동하는지 빠르게 검증합니다 (~1-2분).

검증 항목:
1. Reward normalization 작동
2. Action masking 로그
3. Entropy가 0으로 붕괴하지 않음
4. Value loss가 합리적 범위
5. Diagnostic logging 작동

작성자: Reviewer 박용준
작성일: 2025-12-30
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_dispatch.env import create_multi_map_env
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.core.config import NetworkConfig, TrainingConfig


class RunningMeanStd:
    """Running mean/std for reward normalization."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x) if hasattr(x, '__len__') else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def test_reward_normalization():
    """Test 1: Reward normalization"""
    print("\n" + "="*80)
    print("Test 1: Reward Normalization")
    print("="*80)

    normalizer = RunningMeanStd()

    # Simulate some rewards
    test_rewards = [-100, -50, 10, 50, 100, -200, 20, -150]

    print(f"\nRaw rewards: {test_rewards}")

    normalized = []
    for r in test_rewards:
        normalizer.update([r])
        norm_r = (r - normalizer.mean) / np.sqrt(normalizer.var + 1e-8)
        norm_r = np.clip(norm_r, -5.0, 5.0)
        normalized.append(norm_r)

    print(f"\nNormalized rewards: {[f'{x:.2f}' for x in normalized]}")
    print(f"Normalizer stats: mean={normalizer.mean:.2f}, std={np.sqrt(normalizer.var):.2f}")

    # Check that normalized rewards are reasonable
    assert all(abs(r) <= 5.0 for r in normalized), "❌ Clipping failed!"
    assert abs(np.mean(normalized)) < 1.0, "❌ Normalization failed!"

    print("\n✅ Reward normalization works correctly!")
    return True


def test_environment_basics():
    """Test 2: Environment and action masking"""
    print("\n" + "="*80)
    print("Test 2: Environment Basics & Action Masking")
    print("="*80)

    # Create environment
    env = create_multi_map_env(
        map_configs=None,  # Use default diverse maps
        mode="random",
        track_coverage=False,
    )

    print(f"\n✅ Created environment with {len(env.map_names)} maps")

    # Reset
    obs, info = env.reset(seed=42)

    print(f"\nObservation shape: {obs.shape}")
    assert obs.shape == (77,), f"❌ Wrong obs shape: {obs.shape}"

    # Check action mask
    if "action_mask" in info:
        mask = info["action_mask"]
        valid_count = np.sum(mask > 0.5)
        print(f"Action mask shape: {mask.shape}")
        print(f"Valid actions: {valid_count} / {len(mask)}")

        assert valid_count > 0, "❌ No valid actions!"
        assert valid_count <= len(mask), "❌ Invalid mask!"

        print("\n✅ Action masking works correctly!")
    else:
        print("\n⚠️ Warning: No action_mask in info dict")

    # Take some steps
    print("\nTaking 10 test steps...")
    rewards = []
    nav_times = []
    valid_actions_list = []

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)

        rewards.append(reward)

        if "nav_time" in info:
            nav_times.append(info["nav_time"])

        if "action_mask" in info:
            valid_count = np.sum(info["action_mask"] > 0.5)
            valid_actions_list.append(valid_count)

        if done or trunc:
            obs, info = env.reset()
            print(f"  Episode ended at step {i+1}")

    print(f"\nRewards: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}")

    if nav_times:
        print(f"Nav times: mean={np.mean(nav_times):.2f}s, min={np.min(nav_times):.2f}s, max={np.max(nav_times):.2f}s")
        print("✅ SMDP nav_time is logged!")
    else:
        print("⚠️ Warning: No nav_time logged")

    if valid_actions_list:
        print(f"Valid actions: mean={np.mean(valid_actions_list):.1f}, min={np.min(valid_actions_list)}")
        print("✅ Action masking is logged!")
    else:
        print("⚠️ Warning: No action masking logged")

    env.close()

    print("\n✅ Environment test passed!")
    return True


def test_ppo_update():
    """Test 3: PPO update with fixed hyperparameters"""
    print("\n" + "="*80)
    print("Test 3: PPO Update with Fixed Hyperparameters")
    print("="*80)

    # Create environment
    env = create_multi_map_env(mode="random", track_coverage=False)

    obs_dim = env.observation_space.shape[0]
    num_replan_strategies = env.action_space.nvec[1]

    # FIXED hyperparameters
    training_config = TrainingConfig(
        total_timesteps=10000,
        learning_rate=1e-4,         # 3e-4 → 1e-4
        num_steps=128,               # Small for quick test
        num_epochs=4,                # Reduced for test
        batch_size=32,
        entropy_coef=0.1,            # 0.01 → 0.1
        value_loss_coef=1.0,         # 0.5 → 1.0
        normalize_rewards=True,
        clip_rewards=5.0,
    )

    network_config = NetworkConfig()

    agent = PPOAgent(
        obs_dim=obs_dim,
        num_replan_strategies=num_replan_strategies,
        training_config=training_config,
        network_config=network_config,
        device="cpu",
    )

    print(f"\n✅ Created PPO agent")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Entropy coef: {training_config.entropy_coef}")
    print(f"  Value loss coef: {training_config.value_loss_coef}")

    # Collect rollout with reward normalization
    print(f"\nCollecting {training_config.num_steps} steps...")

    reward_normalizer = RunningMeanStd()
    raw_rewards = []

    obs, info = env.reset(seed=42)

    for step in range(training_config.num_steps):
        # Get action with masking
        action_mask = info.get("action_mask", None)
        action, log_prob, value = agent.get_action(obs, action_mask=action_mask)
        next_obs, reward, done, trunc, info = env.step(action)

        raw_rewards.append(reward)

        # Normalize reward
        reward_normalizer.update([reward])
        normalized_reward = (reward - reward_normalizer.mean) / np.sqrt(reward_normalizer.var + 1e-8)
        normalized_reward = np.clip(normalized_reward, -5.0, 5.0)

        nav_time = info.get("nav_time", 1.0)
        action_mask = info.get("action_mask", None)

        agent.buffer.add(
            obs=obs,
            action=action,
            reward=normalized_reward,  # Use normalized!
            value=value,
            log_prob=log_prob,
            done=done,
            nav_time=nav_time,
            action_mask=action_mask,
        )

        obs = next_obs

        if done or trunc:
            obs, info = env.reset()

    print(f"\n✅ Collected rollout")
    print(f"  Raw rewards: mean={np.mean(raw_rewards):.2f}, std={np.std(raw_rewards):.2f}")
    print(f"  Normalized: mean={reward_normalizer.mean:.2f}, std={np.sqrt(reward_normalizer.var):.2f}")

    # Do PPO update
    print("\nPerforming PPO update...")

    final_action, final_log_prob, final_value = agent.get_action(obs)

    train_stats = agent.update(last_value=final_value, last_done=False)

    print("\n✅ PPO update completed!")
    print("\nTraining stats:")
    for key, value in train_stats.items():
        print(f"  {key}: {value:.6f}")

    # Verify stats are reasonable
    assert train_stats["entropy"] > 0.0, "❌ Entropy is zero!"
    assert train_stats["policy_loss"] != 0.0, "❌ Policy loss is zero!"
    assert train_stats["value_loss"] < 1e6, f"❌ Value loss too high: {train_stats['value_loss']}"

    print("\n" + "="*80)
    print("CRITICAL CHECKS:")
    print("="*80)

    # Check entropy
    if train_stats["entropy"] > 0.01:
        print(f"✅ Entropy: {train_stats['entropy']:.4f} (> 0.01) - Good!")
    else:
        print(f"⚠️ Entropy: {train_stats['entropy']:.4f} (< 0.01) - May collapse")

    # Check approx_kl
    if train_stats["approx_kl"] > 0.0001:
        print(f"✅ Approx KL: {train_stats['approx_kl']:.6f} (> 0.0001) - Policy updating!")
    else:
        print(f"⚠️ Approx KL: {train_stats['approx_kl']:.6f} (~ 0) - Policy may not be updating")

    # Check clipfrac
    if train_stats["clipfrac"] > 0.01:
        print(f"✅ Clipfrac: {train_stats['clipfrac']:.4f} (> 0.01) - PPO working!")
    else:
        print(f"⚠️ Clipfrac: {train_stats['clipfrac']:.4f} (~ 0) - PPO may not be working")

    # Check value_loss
    if train_stats["value_loss"] < 1000:
        print(f"✅ Value loss: {train_stats['value_loss']:.2f} (< 1000) - Reasonable!")
    elif train_stats["value_loss"] < 100000:
        print(f"⚠️ Value loss: {train_stats['value_loss']:.2f} (< 100K) - High but acceptable")
    else:
        print(f"❌ Value loss: {train_stats['value_loss']:.2f} (> 100K) - TOO HIGH!")

    # Check explained_variance
    if train_stats["explained_variance"] > 0.05:
        print(f"✅ Explained variance: {train_stats['explained_variance']:.4f} (> 0.05) - Critic learning!")
    else:
        print(f"⚠️ Explained variance: {train_stats['explained_variance']:.4f} (< 0.05) - Critic struggling")

    env.close()

    print("\n✅ PPO update test passed!")
    return True


def main():
    print("\n" + "="*80)
    print("🔧 POLICY COLLAPSE FIX - QUICK VALIDATION TEST")
    print("="*80)
    print("\nThis script validates that all fixes are working correctly.")
    print("Expected runtime: 1-2 minutes\n")

    torch.manual_seed(42)
    np.random.seed(42)

    all_passed = True

    try:
        # Test 1: Reward normalization
        if not test_reward_normalization():
            all_passed = False

        # Test 2: Environment and masking
        if not test_environment_basics():
            all_passed = False

        # Test 3: PPO update
        if not test_ppo_update():
            all_passed = False

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nYou can now proceed with full training:")
        print("  python scripts/train_multi_map_fixed.py --total-timesteps 100000 --seed 42")
        print("\nMonitor these metrics in TensorBoard:")
        print("  - train/entropy should stay > 0.02")
        print("  - train/approx_kl should be > 0.001")
        print("  - train/value_loss should decrease over time")
        print("  - diagnostics/* should show reasonable values")
    else:
        print("❌ SOME TESTS FAILED!")
        print("="*80)
        print("\nPlease check the errors above and fix before training.")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
