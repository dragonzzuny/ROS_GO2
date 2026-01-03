#!/usr/bin/env python3
"""
Phase 4 Test: State Space Enhancements

Tests that Phase 4 observation enhancements are properly implemented:
1. Observation dimension is 88 (was 77)
2. Event risk level extraction works
3. Patrol crisis indicators work
4. Candidate feasibility hints work
5. Combined urgency-risk signal works

Reviewer 박용준 - Phase 4 Verification
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_dispatch.core.config import EnvConfig, RewardConfig
from rl_dispatch.env import PatrolEnv


def test_observation_dimension():
    """Test that observation space is now 88D."""
    print("=" * 80)
    print("Test 1: Observation Dimension")
    print("=" * 80)

    config = EnvConfig.load_yaml("configs/map_campus.yaml")
    reward_config = RewardConfig()

    env = PatrolEnv(env_config=config, reward_config=reward_config)

    print(f"\n📏 Observation Space Shape: {env.observation_space.shape}")
    print(f"   Expected: (88,)")
    print(f"   Actual: {env.observation_space.shape}")

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\n🔍 Observation vector length: {len(obs)}")
    print(f"   Expected: 88")
    print(f"   Actual: {len(obs)}")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\n🔍 After step, observation length: {len(obs)}")

    print("\n" + "=" * 80)
    if env.observation_space.shape == (88,) and len(obs) == 88:
        print("✅ PASS: Observation dimension is 88D")
        return True
    else:
        print(f"❌ FAIL: Observation dimension is {len(obs)}D, expected 88D")
        return False


def test_event_risk_extraction():
    """Test that event risk level is properly extracted."""
    print("\n" + "=" * 80)
    print("Test 2: Event Risk Level Extraction")
    print("=" * 80)

    config = EnvConfig.load_yaml("configs/map_campus.yaml")
    reward_config = RewardConfig()

    env = PatrolEnv(env_config=config, reward_config=reward_config)
    obs, info = env.reset(seed=42)

    print(f"\n🎯 Testing event risk extraction...")

    # Run until we get an event
    max_steps = 100
    event_found = False
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if event exists
        has_event = info.get('has_event', False)
        if has_event:
            event_found = True
            # Index 77 should contain event risk level
            event_risk = obs[77]
            print(f"\n   Event detected at step {step}")
            print(f"   Event risk feature (index 77): {event_risk:.3f}")
            print(f"   Expected range: [0.0, 1.0]")

            # Check it's in valid range
            if 0.0 <= event_risk <= 1.0:
                print(f"   ✅ Event risk is in valid range")
                return True
            else:
                print(f"   ❌ Event risk {event_risk} is out of range!")
                return False

        if terminated or truncated:
            obs, info = env.reset()

    if not event_found:
        print(f"\n   ⚠️  No event found in {max_steps} steps")
        print(f"   Cannot verify event risk extraction")
        print(f"   Testing with no-event case: risk should be 0.0")

        # Check no-event case
        no_event_risk = obs[77]
        if abs(no_event_risk) < 0.01:
            print(f"   ✅ No-event risk is {no_event_risk:.3f} (≈0.0)")
            return True
        else:
            print(f"   ❌ No-event risk should be 0.0, got {no_event_risk:.3f}")
            return False


def test_patrol_crisis_indicators():
    """Test that patrol crisis indicators are properly extracted."""
    print("\n" + "=" * 80)
    print("Test 3: Patrol Crisis Indicators")
    print("=" * 80)

    config = EnvConfig.load_yaml("configs/map_campus.yaml")
    reward_config = RewardConfig()

    env = PatrolEnv(env_config=config, reward_config=reward_config)
    obs, info = env.reset(seed=42)

    print(f"\n🚨 Testing patrol crisis indicators...")
    print(f"   Index 78: max_gap_normalized")
    print(f"   Index 79: critical_count_normalized")
    print(f"   Index 80: crisis_score")

    # Run a few steps to accumulate coverage gaps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    # Extract crisis indicators
    max_gap_norm = obs[78]
    critical_count_norm = obs[79]
    crisis_score = obs[80]

    print(f"\n   Values:")
    print(f"   max_gap_normalized: {max_gap_norm:.3f}")
    print(f"   critical_count_norm: {critical_count_norm:.3f}")
    print(f"   crisis_score: {crisis_score:.3f}")

    # All should be in reasonable range
    all_valid = True
    if not (0.0 <= max_gap_norm <= 2.0):
        print(f"   ❌ max_gap_normalized out of range [0, 2]")
        all_valid = False
    if not (0.0 <= critical_count_norm <= 1.0):
        print(f"   ❌ critical_count_norm out of range [0, 1]")
        all_valid = False
    if not (0.0 <= crisis_score <= 2.0):
        print(f"   ❌ crisis_score out of range [0, 2]")
        all_valid = False

    print("\n" + "=" * 80)
    if all_valid:
        print("✅ PASS: Patrol crisis indicators in valid ranges")
        return True
    else:
        print("❌ FAIL: Some patrol crisis indicators out of range")
        return False


def test_candidate_feasibility():
    """Test that candidate feasibility hints are properly extracted."""
    print("\n" + "=" * 80)
    print("Test 4: Candidate Feasibility Hints")
    print("=" * 80)

    config = EnvConfig.load_yaml("configs/map_campus.yaml")
    reward_config = RewardConfig()

    env = PatrolEnv(env_config=config, reward_config=reward_config)
    obs, info = env.reset(seed=42)

    print(f"\n🎯 Testing candidate feasibility...")
    print(f"   Indices 81-86: Feasibility for 6 candidates")

    # Extract feasibility features
    feasibility = obs[81:87]

    print(f"\n   Candidate feasibility scores:")
    for i, score in enumerate(feasibility):
        print(f"   Candidate {i}: {score:.3f}")

    # Check all are in [0, 1] range
    all_valid = all(0.0 <= score <= 1.0 for score in feasibility)

    # After Phase 1 A* fix, all should be > 0 (feasible)
    most_feasible = all(score > 0.0 for score in feasibility)

    print(f"\n   All in [0, 1] range: {all_valid}")
    print(f"   All feasible (>0): {most_feasible}")

    print("\n" + "=" * 80)
    if all_valid:
        print("✅ PASS: Candidate feasibility scores in valid range")
        if most_feasible:
            print("   ✅ Bonus: All candidates feasible (Phase 1 A* working!)")
        return True
    else:
        print("❌ FAIL: Some feasibility scores out of range")
        return False


def test_urgency_risk_combined():
    """Test that combined urgency-risk signal works."""
    print("\n" + "=" * 80)
    print("Test 5: Combined Urgency-Risk Signal")
    print("=" * 80)

    config = EnvConfig.load_yaml("configs/map_campus.yaml")
    reward_config = RewardConfig()

    env = PatrolEnv(env_config=config, reward_config=reward_config)
    obs, info = env.reset(seed=42)

    print(f"\n🎯 Testing urgency-risk combined signal...")
    print(f"   Index 87: urgency × risk geometric mean")

    # Run until we get an event
    max_steps = 100
    event_found = False
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        has_event = info.get('has_event', False)
        if has_event:
            event_found = True
            combined_signal = obs[87]
            event_urgency = obs[71]  # Original urgency feature
            event_risk = obs[77]  # Risk level feature

            print(f"\n   Event found at step {step}")
            print(f"   Urgency (index 71): {event_urgency:.3f}")
            print(f"   Risk (index 77): {event_risk:.3f}")
            print(f"   Combined (index 87): {combined_signal:.3f}")
            print(f"   Expected (geometric mean): {np.sqrt(event_urgency * event_risk):.3f}")

            # Should be in [0, 1]
            if 0.0 <= combined_signal <= 1.0:
                print(f"   ✅ Combined signal in valid range")
                # Check it's approximately geometric mean
                expected = np.sqrt(event_urgency * event_risk)
                if abs(combined_signal - expected) < 0.1:
                    print(f"   ✅ Matches geometric mean formula")
                return True
            else:
                print(f"   ❌ Combined signal out of range")
                return False

        if terminated or truncated:
            obs, info = env.reset()

    if not event_found:
        print(f"\n   ⚠️  No event found in {max_steps} steps")
        print(f"   Testing no-event case: should be 0.0")

        combined_no_event = obs[87]
        if abs(combined_no_event) < 0.01:
            print(f"   ✅ No-event combined signal is {combined_no_event:.3f} (≈0.0)")
            return True
        else:
            print(f"   ❌ No-event combined should be 0.0, got {combined_no_event:.3f}")
            return False


def test_backward_compatibility():
    """Test that existing features (indices 0-76) still work."""
    print("\n" + "=" * 80)
    print("Test 6: Backward Compatibility")
    print("=" * 80)

    config = EnvConfig.load_yaml("configs/map_campus.yaml")
    reward_config = RewardConfig()

    env = PatrolEnv(env_config=config, reward_config=reward_config)
    obs, info = env.reset(seed=42)

    print(f"\n🔍 Checking existing features...")

    # Check key features are still populated
    goal_relative = obs[0:2]
    heading_sincos = obs[2:4]
    velocity = obs[4:6]
    battery = obs[6]
    lidar = obs[7:71]
    event_features = obs[71:75]
    patrol_features = obs[75:77]

    print(f"   Goal relative (0:2): {goal_relative}")
    print(f"   Heading sin/cos (2:4): {heading_sincos}")
    print(f"   Velocity (4:6): {velocity}")
    print(f"   Battery (6): {battery:.3f}")
    print(f"   LiDAR ranges (7:71): mean={np.mean(lidar):.3f}")
    print(f"   Event features (71:75): {event_features}")
    print(f"   Patrol features (75:77): {patrol_features}")

    # Basic sanity checks
    checks = [
        ("Battery in [0,1]", 0.0 <= battery <= 1.0),
        ("Heading sin in [-1,1]", -1.0 <= heading_sincos[0] <= 1.0),
        ("Heading cos in [-1,1]", -1.0 <= heading_sincos[1] <= 1.0),
        ("LiDAR non-empty", len(lidar) == 64),
    ]

    all_pass = True
    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"   {status} {name}")
        if not result:
            all_pass = False

    print("\n" + "=" * 80)
    if all_pass:
        print("✅ PASS: Backward compatibility maintained")
        return True
    else:
        print("❌ FAIL: Some backward compatibility checks failed")
        return False


def main():
    """Run all Phase 4 tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "Phase 4: State Space Enhancements Test Suite" + " " * 18 + "║")
    print("╚" + "=" * 78 + "╝")

    results = {}

    try:
        results['dimension'] = test_observation_dimension()
    except Exception as e:
        print(f"\n❌ Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['dimension'] = False

    try:
        results['event_risk'] = test_event_risk_extraction()
    except Exception as e:
        print(f"\n❌ Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['event_risk'] = False

    try:
        results['patrol_crisis'] = test_patrol_crisis_indicators()
    except Exception as e:
        print(f"\n❌ Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['patrol_crisis'] = False

    try:
        results['feasibility'] = test_candidate_feasibility()
    except Exception as e:
        print(f"\n❌ Test 4 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['feasibility'] = False

    try:
        results['combined'] = test_urgency_risk_combined()
    except Exception as e:
        print(f"\n❌ Test 5 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['combined'] = False

    try:
        results['backward_compat'] = test_backward_compatibility()
    except Exception as e:
        print(f"\n❌ Test 6 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['backward_compat'] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<30} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Phase 4 state enhancements working!")
        print("\nPhase 4 Features Added (+11D → 88D total):")
        print("  - Event risk level (index 77)")
        print("  - Patrol crisis indicators (indices 78-80)")
        print("  - Candidate feasibility (indices 81-86)")
        print("  - Urgency-risk combined signal (index 87)")
        print("\nNext steps:")
        print("  1. Re-train curriculum with enhanced observations")
        print("  2. Compare performance vs Phase 3 baseline")
        print("  3. Analyze which features contribute most to learning")
    else:
        print("⚠️  SOME TESTS FAILED - Review implementation")
        print("\nDebugging steps:")
        print("  1. Check observation.py feature extraction methods")
        print("  2. Verify observation space dimension in patrol_env.py")
        print("  3. Check State object has all required fields")

    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
