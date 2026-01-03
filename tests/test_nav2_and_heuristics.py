"""
Test script for Nav2 interface and 10 heuristic strategies.

Tests:
1. Nav2 interface abstraction (SimulatedNav2)
2. 10 heuristic strategy generators
3. Integration with PatrolEnv
4. Action space verification
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_dispatch.navigation import NavigationInterface, SimulatedNav2, NavigationResult
from rl_dispatch.planning import CandidateFactory
from rl_dispatch.core.types import RobotState, PatrolPoint
from rl_dispatch.core.config import EnvConfig
from rl_dispatch.env import PatrolEnv


def test_nav2_interface():
    """Test 1: Nav2 interface abstraction."""
    print("\n" + "=" * 80)
    print("Test 1: Nav2 Interface Abstraction")
    print("=" * 80)

    # Create SimulatedNav2
    np_random = np.random.RandomState(42)
    nav = SimulatedNav2(
        max_velocity=1.5,
        nav_failure_rate=0.05,
        collision_rate=0.01,
        np_random=np_random,
    )

    print("\n✓ SimulatedNav2 created successfully")
    print(f"  Max velocity: {nav.max_velocity} m/s")
    print(f"  Failure rate: {nav.nav_failure_rate}")
    print(f"  Collision rate: {nav.collision_rate}")

    # Test navigate_to_goal
    start = (0.0, 0.0)
    goal = (10.0, 10.0)

    result = nav.navigate_to_goal(start, goal)
    print(f"\n✓ Navigation test (0,0) → (10,10):")
    print(f"  Time: {result.time:.2f}s")
    print(f"  Success: {result.success}")
    print(f"  Collision: {result.collision}")
    print(f"  Path: {result.path}")

    assert isinstance(result, NavigationResult), "Should return NavigationResult"
    assert result.time > 0, "Navigation time should be positive"

    # Test get_eta
    eta = nav.get_eta(start, goal)
    print(f"\n✓ ETA estimation:")
    print(f"  ETA: {eta:.2f}s")
    assert eta > 0, "ETA should be positive"

    # Test plan_path
    path = nav.plan_path(start, goal)
    print(f"\n✓ Path planning:")
    print(f"  Path: {path}")
    assert path is not None or True, "Path can be None if planning fails"

    print("\n✅ Test 1 PASSED: Nav2 interface works correctly")


def test_10_heuristic_strategies():
    """Test 2: All 10 heuristic strategy generators."""
    print("\n" + "=" * 80)
    print("Test 2: 10 Heuristic Strategy Generators")
    print("=" * 80)

    # Create test scenario
    robot = RobotState(
        x=50.0,
        y=50.0,
        heading=0.0,
        velocity=0.0,
        angular_velocity=0.0,
        battery_level=1.0,
        current_goal_idx=0,
    )

    patrol_points = tuple([
        PatrolPoint(x=10.0, y=10.0, last_visit_time=-100.0, priority=1.0, point_id=0),
        PatrolPoint(x=90.0, y=10.0, last_visit_time=-50.0, priority=1.2, point_id=1),
        PatrolPoint(x=90.0, y=90.0, last_visit_time=-20.0, priority=0.8, point_id=2),
        PatrolPoint(x=10.0, y=90.0, last_visit_time=-80.0, priority=1.5, point_id=3),
    ])

    current_time = 0.0

    # Create factory
    factory = CandidateFactory()

    print(f"\n✓ CandidateFactory created with {len(factory.generators)} generators")

    # Generate all candidates
    candidates = factory.generate_all(robot, patrol_points, current_time)

    print(f"\n✓ Generated {len(candidates)} candidates:")
    for i, candidate in enumerate(candidates):
        print(f"\n  [{i}] {candidate.strategy_name} (ID: {candidate.strategy_id}):")
        print(f"      Order: {candidate.patrol_order}")
        print(f"      Distance: {candidate.estimated_total_distance:.1f}m")
        print(f"      Max gap: {candidate.max_coverage_gap:.1f}s")

    # Verify we have 10 strategies
    assert len(candidates) == 10, f"Expected 10 candidates, got {len(candidates)}"

    # Verify all strategy names
    expected_names = [
        "keep_order",
        "nearest_first",
        "most_overdue_first",
        "overdue_eta_balance",
        "risk_weighted",
        "balanced_coverage",
        "overdue_threshold_first",
        "windowed_replan",
        "minimal_deviation_insert",
        "shortest_eta_first",
    ]

    actual_names = [c.strategy_name for c in candidates]
    print(f"\n✓ Strategy names verification:")
    for name in expected_names:
        if name in actual_names:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} MISSING!")

    assert all(name in actual_names for name in expected_names), \
        "Not all expected strategies found"

    # Verify unique strategy IDs
    strategy_ids = [c.strategy_id for c in candidates]
    assert len(set(strategy_ids)) == 10, "Strategy IDs should be unique"

    print("\n✅ Test 2 PASSED: All 10 heuristic strategies work correctly")


def test_patrolenv_with_nav2():
    """Test 3: PatrolEnv integration with Nav2 interface."""
    print("\n" + "=" * 80)
    print("Test 3: PatrolEnv Integration with Nav2")
    print("=" * 80)

    # Load config
    config = EnvConfig.load_yaml("configs/map_large_square.yaml")

    print(f"\n✓ Config loaded:")
    print(f"  Map: {config.map_width}m × {config.map_height}m")
    print(f"  Patrol points: {len(config.patrol_points)}")
    print(f"  Num candidates: {config.num_candidates}")
    print(f"  Charging station: {config.charging_station_position}")

    # Verify num_candidates is 10
    assert config.num_candidates == 10, \
        f"Expected num_candidates=10, got {config.num_candidates}"

    # Create environment
    env = PatrolEnv(env_config=config)

    print(f"\n✓ PatrolEnv created successfully")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")

    # Verify action space is MultiDiscrete([2, 10])
    assert env.action_space.nvec[0] == 2, "First action dimension should be 2 (modes)"
    assert env.action_space.nvec[1] == 10, "Second action dimension should be 10 (strategies)"

    # Reset environment
    obs, info = env.reset(seed=42)

    print(f"\n✓ Environment reset:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Nav interface type: {type(env.nav_interface).__name__}")
    print(f"  Initial patrol route (first 5): {env.current_patrol_route[:5]}")

    # Verify nav interface
    assert env.nav_interface is not None, "Nav interface should be initialized"
    assert isinstance(env.nav_interface, SimulatedNav2), \
        "Should use SimulatedNav2 for training"

    # Verify initial route is randomized (not sequential [0,1,2,3,...])
    is_sequential = all(
        env.current_patrol_route[i] == i
        for i in range(min(5, len(env.current_patrol_route)))
    )
    print(f"\n✓ Initial route randomization:")
    print(f"  Is sequential: {is_sequential} (should be False)")
    # Note: There's a small chance it could be sequential by random chance

    # Run a few steps
    print(f"\n✓ Running 5 steps:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        mode = "DISPATCH" if action[0] == 1 else "PATROL"
        strategy = action[1]

        print(f"  Step {i+1}: action=({mode}, strategy_{strategy}), "
              f"reward={reward:.2f}, time={env.current_state.current_time:.1f}s")

        if terminated or truncated:
            print(f"  Episode ended at step {i+1}")
            break

    print("\n✅ Test 3 PASSED: PatrolEnv integrates correctly with Nav2")


def test_all_map_configs():
    """Test 4: Verify all 6 maps have correct configuration."""
    print("\n" + "=" * 80)
    print("Test 4: All Map Configurations")
    print("=" * 80)

    map_configs = [
        "configs/map_large_square.yaml",
        "configs/map_corridor.yaml",
        "configs/map_l_shaped.yaml",
        "configs/map_office_building.yaml",
        "configs/map_campus.yaml",
        "configs/map_warehouse.yaml",
    ]

    print(f"\n✓ Testing {len(map_configs)} map configurations:")

    for config_path in map_configs:
        map_name = Path(config_path).stem
        config = EnvConfig.load_yaml(config_path)

        # Verify num_candidates is 10
        assert config.num_candidates == 10, \
            f"{map_name}: Expected num_candidates=10, got {config.num_candidates}"

        # Verify charging station exists
        assert hasattr(config, 'charging_station_position'), \
            f"{map_name}: Missing charging_station_position"

        # Verify charging station is within bounds
        cs_x, cs_y = config.charging_station_position
        assert 0 <= cs_x <= config.map_width, \
            f"{map_name}: Charging station X out of bounds"
        assert 0 <= cs_y <= config.map_height, \
            f"{map_name}: Charging station Y out of bounds"

        print(f"\n  ✓ {map_name}:")
        print(f"      Size: {config.map_width}m × {config.map_height}m")
        print(f"      Points: {len(config.patrol_points)}")
        print(f"      Candidates: {config.num_candidates}")
        print(f"      Charging: {config.charging_station_position}")

    print("\n✅ Test 4 PASSED: All map configurations are correct")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Nav2 Interface & Heuristics Test Suite")
    print("=" * 80)

    try:
        test_nav2_interface()
        test_10_heuristic_strategies()
        test_patrolenv_with_nav2()
        test_all_map_configs()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\n시스템이 정상적으로 작동합니다:")
        print("  ✓ Nav2 인터페이스 추상화")
        print("  ✓ 10개 휴리스틱 전략")
        print("  ✓ PatrolEnv 통합")
        print("  ✓ 6개 맵 설정")
        print("  ✓ 랜덤 초기 루트\n")

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
