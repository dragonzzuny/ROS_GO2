#!/usr/bin/env python3
"""
Phase 1 Test: Feasible Goal Generation with A* Pathfinding

Tests that candidate generation now uses A* pathfinding to:
1. Generate only feasible patrol routes
2. Accurately estimate distances using actual paths
3. Filter out infeasible goals that would cause immediate nav failure

Expected results:
- Candidates have realistic distance estimates (not Euclidean)
- No candidates with distance=inf (all feasible)
- Nav immediate failure rate should drop from 95% to <10%

Reviewer 박용준 - Phase 1 Verification
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_dispatch.core.config import EnvConfig
from rl_dispatch.env import PatrolEnv


def test_feasible_candidate_generation():
    """Test that all generated candidates are feasible."""
    print("="*80)
    print("Test 1: Feasible Candidate Generation")
    print("="*80)

    # Load campus map (most problematic)
    config = EnvConfig.load_yaml("configs/map_campus.yaml")

    # Create environment
    env = PatrolEnv(env_config=config)
    obs, info = env.reset(seed=42)

    print(f"\n📍 Map: campus")
    print(f"   Size: {config.map_width}m × {config.map_height}m")
    print(f"   Walls: {len(config.walls)} defined")
    print(f"   Nav interface: {type(env.nav_interface).__name__}")
    print(f"   Uses A*: {env.nav_interface.use_astar if hasattr(env.nav_interface, 'use_astar') else 'N/A'}")

    # Check candidate factory has nav_interface
    has_nav = env.candidate_factory.nav_interface is not None
    print(f"\n✅ CandidateFactory has nav_interface: {has_nav}")

    if not has_nav:
        print("❌ FAIL: nav_interface not connected to candidate factory!")
        return False

    # Generate candidates
    robot = env.current_state.robot
    patrol_points = env.current_state.patrol_points
    candidates = env.candidate_factory.generate_all(
        robot, patrol_points, current_time=0.0
    )

    print(f"\n📊 Generated {len(candidates)} candidates:")
    print(f"{'Strategy':<25} {'Distance (m)':<15} {'Max Gap (s)':<15} {'Feasible'}")
    print("-" * 80)

    all_feasible = True
    for c in candidates:
        feasible = c.estimated_total_distance != np.inf
        if not feasible:
            all_feasible = False

        status = "✅" if feasible else "❌"
        dist_str = f"{c.estimated_total_distance:.1f}" if feasible else "INFEASIBLE"
        gap_str = f"{c.max_coverage_gap:.1f}" if feasible else "N/A"

        print(f"{c.strategy_name:<25} {dist_str:<15} {gap_str:<15} {status}")

    print("\n" + "="*80)
    if all_feasible:
        print("✅ PASS: All candidates are feasible")
    else:
        print("❌ FAIL: Some candidates are infeasible (distance=inf)")

    return all_feasible


def test_realistic_distance_estimation():
    """Test that distance estimates are realistic (not just Euclidean)."""
    print("\n" + "="*80)
    print("Test 2: Realistic Distance Estimation")
    print("="*80)

    # Load office building map (complex layout)
    config = EnvConfig.load_yaml("configs/map_office_building.yaml")

    env = PatrolEnv(env_config=config)
    obs, info = env.reset(seed=42)

    # Get first patrol candidate
    robot = env.current_state.robot
    patrol_points = env.current_state.patrol_points
    candidates = env.candidate_factory.generate_all(
        robot, patrol_points, current_time=0.0
    )

    # Compare nearest-first (A*) vs theoretical Euclidean
    nearest_first = candidates[1]  # NearestFirstGenerator

    print(f"\n📐 Distance Comparison:")
    print(f"   Strategy: {nearest_first.strategy_name}")

    # Calculate Euclidean distance for same route
    euclidean_dist = 0.0
    current_pos = robot.position
    for idx in nearest_first.patrol_order:
        point = patrol_points[idx]
        euclidean_dist += np.sqrt(
            (point.x - current_pos[0])**2 + (point.y - current_pos[1])**2
        )
        current_pos = (point.x, point.y)

    astar_dist = nearest_first.estimated_total_distance

    print(f"   Euclidean distance: {euclidean_dist:.1f}m")
    print(f"   A* path distance:   {astar_dist:.1f}m")
    print(f"   Difference:         {astar_dist - euclidean_dist:.1f}m ({(astar_dist/euclidean_dist - 1)*100:.1f}%)")

    # A* distance should be >= Euclidean (due to obstacles)
    is_realistic = astar_dist >= euclidean_dist

    print("\n" + "="*80)
    if is_realistic:
        print("✅ PASS: A* distance >= Euclidean (accounts for obstacles)")
    else:
        print("❌ FAIL: A* distance < Euclidean (should not happen)")

    return is_realistic


def test_nav_immediate_failure_rate():
    """Test that nav immediate failure rate is reduced."""
    print("\n" + "="*80)
    print("Test 3: Nav Immediate Failure Rate Reduction")
    print("="*80)

    # Load campus map (95% immediate failure before Phase 1)
    config = EnvConfig.load_yaml("configs/map_campus.yaml")

    env = PatrolEnv(env_config=config)
    obs, info = env.reset(seed=42)

    print(f"\n🧪 Running 50 steps on campus map...")

    immediate_failures = 0
    total_steps = 0

    for i in range(50):
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Check nav_time
        nav_time = info.get("nav_time", 1.0)
        if nav_time < 1.0:
            immediate_failures += 1

        total_steps += 1

        if terminated or truncated:
            obs, info = env.reset()

    immediate_fail_rate = (immediate_failures / total_steps) * 100

    print(f"\n📊 Results:")
    print(f"   Total steps: {total_steps}")
    print(f"   Immediate failures (nav_time < 1.0s): {immediate_failures}")
    print(f"   Immediate failure rate: {immediate_fail_rate:.1f}%")

    print("\n   Targets:")
    print(f"   ❌ Before Phase 1: 95.29% (campus)")
    print(f"   ✅ After Phase 1:  <10%")

    success = immediate_fail_rate < 10.0

    print("\n" + "="*80)
    if success:
        print(f"✅ PASS: Immediate failure rate reduced to {immediate_fail_rate:.1f}%")
    else:
        print(f"⚠️  WARNING: Immediate failure rate still {immediate_fail_rate:.1f}% (target: <10%)")
        print("    This may improve with more extensive testing or tuning.")

    return success


def main():
    """Run all Phase 1 tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " " * 15 + "Phase 1: Feasible Goal Generation Test Suite" + " " * 17 + "║")
    print("╚" + "="*78 + "╝")

    results = {}

    try:
        results['feasible_candidates'] = test_feasible_candidate_generation()
    except Exception as e:
        print(f"\n❌ Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['feasible_candidates'] = False

    try:
        results['realistic_distance'] = test_realistic_distance_estimation()
    except Exception as e:
        print(f"\n❌ Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['realistic_distance'] = False

    try:
        results['nav_failure_rate'] = test_nav_immediate_failure_rate()
    except Exception as e:
        print(f"\n❌ Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['nav_failure_rate'] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<30} {status}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Phase 1 implementation successful!")
        print("\nNext steps:")
        print("  1. Run training to verify nav_time < 1.0s rate < 10%")
        print("  2. Check campus Return std drops from 83k to <40k")
        print("  3. Proceed to Phase 2: Reward Redesign")
    else:
        print("⚠️  SOME TESTS FAILED - Review implementation")
        print("\nDebugging steps:")
        print("  1. Verify candidate_generator.py was updated")
        print("  2. Verify patrol_env.py calls set_nav_interface()")
        print("  3. Check that A* pathfinding is enabled")

    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
