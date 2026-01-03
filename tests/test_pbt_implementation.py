#!/usr/bin/env python3
"""
Test PBT Implementation

Tests core PBT components:
1. HyperparameterSpace sampling and perturbation
2. PopulationMember creation and weight copying
3. PBTManager exploit/explore logic
4. Short training run
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_dispatch.env import create_multi_map_env
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.core.config import NetworkConfig, TrainingConfig
from rl_dispatch.pbt import PopulationMember, HyperparameterSpace, PBTManager


def test_hyperparameter_sampling():
    """Test 1: Hyperparameter sampling and perturbation."""
    print("=" * 80)
    print("Test 1: Hyperparameter Sampling and Perturbation")
    print("=" * 80)

    # Sample random hyperparameters
    print("\n1. Sampling random hyperparameters (5 samples):")
    for i in range(5):
        hyperparams = HyperparameterSpace.sample_random()
        print(f"\n  Sample {i + 1}:")
        print(f"    learning_rate: {hyperparams['learning_rate']:.2e}")
        print(f"    batch_size:    {hyperparams['batch_size']}")
        print(f"    num_epochs:    {hyperparams['num_epochs']}")
        print(f"    clip_range:    {hyperparams['clip_range']:.3f}")
        print(f"    entropy_coef:  {hyperparams['entropy_coef']:.4f}")

        # Validate
        assert HyperparameterSpace.validate(hyperparams), \
            f"Sample {i + 1} failed validation!"

    # Test perturbation
    print("\n2. Testing perturbation:")
    base_hyperparams = HyperparameterSpace.get_default()
    print(f"\n  Base hyperparams:")
    print(f"    learning_rate: {base_hyperparams['learning_rate']:.2e}")
    print(f"    batch_size:    {base_hyperparams['batch_size']}")
    print(f"    num_epochs:    {base_hyperparams['num_epochs']}")

    print(f"\n  Perturbed versions:")
    for i in range(5):
        perturbed = HyperparameterSpace.perturb(base_hyperparams)
        print(f"\n    Perturb {i + 1}:")
        print(f"      learning_rate: {perturbed['learning_rate']:.2e}")
        print(f"      batch_size:    {perturbed['batch_size']}")
        print(f"      num_epochs:    {perturbed['num_epochs']}")

        # Validate
        assert HyperparameterSpace.validate(perturbed), \
            f"Perturbed {i + 1} failed validation!"

    print("\n" + "=" * 80)
    print("✅ Test 1 PASSED: Sampling and perturbation work correctly")
    print("=" * 80 + "\n")
    return True


def test_population_member():
    """Test 2: PopulationMember creation and operations."""
    print("=" * 80)
    print("Test 2: PopulationMember Creation and Operations")
    print("=" * 80)

    # Create environment to get dimensions
    config_path = Path(__file__).parent / "configs" / "map_corridor.yaml"
    env = create_multi_map_env(
        map_configs=[str(config_path)],
        mode="random",
    )
    obs_dim = env.observation_space.shape[0]
    num_strategies = env.action_space.nvec[1]
    env.close()

    print(f"\n  Observation dim: {obs_dim}")
    print(f"  Num strategies: {num_strategies}")

    # Create two members
    print("\n1. Creating two population members...")

    hyperparams1 = HyperparameterSpace.get_default()
    agent1 = PPOAgent(
        obs_dim=obs_dim,
        num_replan_strategies=num_strategies,
        training_config=TrainingConfig(),
        network_config=NetworkConfig(),
        device="cpu",
    )
    member1 = PopulationMember(
        member_id=0,
        agent=agent1,
        hyperparams=hyperparams1,
    )

    hyperparams2 = HyperparameterSpace.sample_random()
    agent2 = PPOAgent(
        obs_dim=obs_dim,
        num_replan_strategies=num_strategies,
        training_config=TrainingConfig(),
        network_config=NetworkConfig(),
        device="cpu",
    )
    member2 = PopulationMember(
        member_id=1,
        agent=agent2,
        hyperparams=hyperparams2,
    )

    print(f"\n  Member 0: {member1}")
    print(f"  Member 1: {member2}")

    # Test weight copying
    print("\n2. Testing weight copying...")

    # Get initial weights
    initial_weights_0 = member1.agent.network.state_dict()["encoder.0.weight"].clone()
    initial_weights_1 = member2.agent.network.state_dict()["encoder.0.weight"].clone()

    print(f"\n  Initial weight norm (Member 0): {torch.norm(initial_weights_0):.4f}")
    print(f"  Initial weight norm (Member 1): {torch.norm(initial_weights_1):.4f}")

    # Member 1 copies from Member 0
    member2.copy_weights_from(member1)

    copied_weights_1 = member2.agent.network.state_dict()["encoder.0.weight"]

    print(f"\n  After copy, weight norm (Member 1): {torch.norm(copied_weights_1):.4f}")

    # Check they match
    assert torch.allclose(copied_weights_1, initial_weights_0), \
        "Weight copying failed!"

    print(f"\n  ✅ Weights successfully copied!")

    # Test performance tracking
    print("\n3. Testing performance tracking...")

    # Add some returns
    returns = [100.0, 150.0, 200.0, 180.0, 220.0]
    for ret in returns:
        member1.add_return(ret)

    print(f"\n  Added returns: {returns}")
    print(f"  Performance (mean): {member1.performance:.1f}")
    print(f"  Performance std: {member1.performance_std:.1f}")

    expected_mean = np.mean(returns)
    assert abs(member1.performance - expected_mean) < 0.1, \
        "Performance calculation incorrect!"

    print("\n" + "=" * 80)
    print("✅ Test 2 PASSED: PopulationMember works correctly")
    print("=" * 80 + "\n")
    return True


def test_pbt_manager():
    """Test 3: PBTManager exploit/explore logic."""
    print("=" * 80)
    print("Test 3: PBTManager Exploit/Explore Logic")
    print("=" * 80)

    # Create environment
    config_path = Path(__file__).parent / "configs" / "map_corridor.yaml"
    env = create_multi_map_env(
        map_configs=[str(config_path)],
        mode="random",
    )
    obs_dim = env.observation_space.shape[0]
    num_strategies = env.action_space.nvec[1]
    env.close()

    # Create PBT manager
    print("\n1. Creating PBT manager with 4 members...")

    pbt_manager = PBTManager(
        population_size=4,
        eval_interval=1000,
        exploit_threshold=0.5,  # Top/bottom 50% for small population
        log_dir=None,  # No logging for test
    )

    # Create 4 members with different performance
    for member_id in range(4):
        hyperparams = HyperparameterSpace.sample_random()
        agent = PPOAgent(
            obs_dim=obs_dim,
            num_replan_strategies=num_strategies,
            training_config=TrainingConfig(),
            network_config=NetworkConfig(),
            device="cpu",
        )
        member = PopulationMember(
            member_id=member_id,
            agent=agent,
            hyperparams=hyperparams,
        )

        # Simulate different performance levels
        if member_id == 0:
            # Best performer
            for _ in range(20):
                member.add_return(300.0 + np.random.randn() * 20)
        elif member_id == 1:
            # Good performer
            for _ in range(20):
                member.add_return(250.0 + np.random.randn() * 20)
        elif member_id == 2:
            # Poor performer
            for _ in range(20):
                member.add_return(150.0 + np.random.randn() * 20)
        else:
            # Worst performer
            for _ in range(20):
                member.add_return(100.0 + np.random.randn() * 20)

        pbt_manager.add_member(member)

    # Print initial status
    print("\n2. Initial population status:")
    pbt_manager.print_status()

    # Get best/worst
    best = pbt_manager.get_best_member()
    worst = pbt_manager.get_worst_member()

    print(f"\n  Best member: {best.member_id} (performance: {best.performance:.1f})")
    print(f"  Worst member: {worst.member_id} (performance: {worst.performance:.1f})")

    assert best.performance > worst.performance, \
        "Best performer should have higher performance than worst!"

    # Test exploit/explore
    print("\n3. Running exploit/explore cycle...")

    exploit_info = pbt_manager.exploit_and_explore()

    print(f"\n  Exploits performed: {len(exploit_info['exploits'])}")
    print(f"  Pairs:")
    for bottom_id, top_id in exploit_info['exploits']:
        print(f"    Member {bottom_id} ← Member {top_id}")

    # Check that bottom performers were exploited
    worst_after = pbt_manager.get_worst_member()
    print(f"\n  Worst member after exploit: {worst_after.member_id}")
    print(f"  Exploited count: {worst_after.exploited_count}")

    assert worst_after.exploited_count > 0 or len(exploit_info['exploits']) > 0, \
        "Bottom performers should have been exploited!"

    print("\n" + "=" * 80)
    print("✅ Test 3 PASSED: PBTManager exploit/explore works correctly")
    print("=" * 80 + "\n")
    return True


def test_short_training():
    """Test 4: Short training run."""
    print("=" * 80)
    print("Test 4: Short Training Run (100 steps)")
    print("=" * 80)

    # This just tests that training can run without errors
    print("\n  Running short training test...")
    print("  (This is a smoke test, not checking performance)")

    # Create environment
    config_path = Path(__file__).parent / "configs" / "map_corridor.yaml"
    env = create_multi_map_env(
        map_configs=[str(config_path)],
        mode="random",
    )
    obs_dim = env.observation_space.shape[0]
    num_strategies = env.action_space.nvec[1]

    # Create simple member
    hyperparams = HyperparameterSpace.get_default()
    agent = PPOAgent(
        obs_dim=obs_dim,
        num_replan_strategies=num_strategies,
        training_config=TrainingConfig(),
        network_config=NetworkConfig(),
        device="cpu",
    )
    member = PopulationMember(
        member_id=0,
        agent=agent,
        hyperparams=hyperparams,
    )
    member.apply_hyperparams()

    # Run 100 steps
    obs, info = env.reset()
    episode_return = 0.0

    for step in range(100):
        action, log_prob, value = member.agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        member.agent.buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done,
        )

        obs = next_obs
        episode_return += reward

        if done:
            member.add_return(episode_return)
            obs, info = env.reset()
            episode_return = 0.0

    env.close()

    print(f"\n  ✅ Completed 100 steps without errors")
    print(f"  Episodes completed: {len(member.recent_returns)}")
    if member.recent_returns:
        print(f"  Mean return: {np.mean(member.recent_returns):.1f}")

    print("\n" + "=" * 80)
    print("✅ Test 4 PASSED: Short training run works")
    print("=" * 80 + "\n")
    return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "PBT Implementation Test Suite" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")

    results = {}

    # Test 1
    try:
        results['hyperparameter_sampling'] = test_hyperparameter_sampling()
    except Exception as e:
        print(f"\n❌ Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['hyperparameter_sampling'] = False

    # Test 2
    try:
        results['population_member'] = test_population_member()
    except Exception as e:
        print(f"\n❌ Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['population_member'] = False

    # Test 3
    try:
        results['pbt_manager'] = test_pbt_manager()
    except Exception as e:
        print(f"\n❌ Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['pbt_manager'] = False

    # Test 4
    try:
        results['short_training'] = test_short_training()
    except Exception as e:
        print(f"\n❌ Test 4 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['short_training'] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<35} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - PBT implementation ready!")
        print("\nNext steps:")
        print("  1. Run: python scripts/train_pbt.py --population-size 8 --total-timesteps 50000")
        print("  2. Monitor TensorBoard: tensorboard --logdir runs/")
        print("  3. Check exploit/explore cycles in logs")
    else:
        print("⚠️  SOME TESTS FAILED - Review implementation")

    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
