"""
Test script for industrial safety event system and charging station integration.

Tests:
1. Event type distribution (34 events, risk-weighted sampling)
2. Event creation with risk levels (1-9)
3. Charging station configuration in all maps
4. Integration with PatrolEnv
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_dispatch.core.event_types import (
    INDUSTRIAL_SAFETY_EVENTS,
    get_random_event_name,
    get_event_risk_level,
    get_event_category,
    get_risk_level_statistics,
    get_event_distribution_by_category,
)
from rl_dispatch.core.types_extended import Event, validate_event
from rl_dispatch.core.config import EnvConfig
from rl_dispatch.env import PatrolEnv


def test_event_type_system():
    """Test 1: Industrial safety event type definitions."""
    print("\n" + "=" * 80)
    print("Test 1: Industrial Safety Event Type System")
    print("=" * 80)

    # Check total number of events
    total_events = len(INDUSTRIAL_SAFETY_EVENTS)
    print(f"\n✓ Total industrial safety events defined: {total_events}")
    assert total_events == 34, f"Expected 34 events, got {total_events}"

    # Check risk level statistics
    stats = get_risk_level_statistics()
    print(f"\n위험도별 분포:")
    print(f"  저위험 (1-3): {stats['low_risk']}개")
    print(f"  중위험 (4-6): {stats['medium_risk']}개")
    print(f"  고위험 (7-9): {stats['high_risk']}개")

    # Check category distribution
    cat_stats = get_event_distribution_by_category()
    print(f"\n카테고리별 분포:")
    for cat, count in sorted(cat_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat.value}: {count}개")

    # Sample some events
    print(f"\n샘플 이벤트:")
    for event_name in list(INDUSTRIAL_SAFETY_EVENTS.keys())[:5]:
        risk = get_event_risk_level(event_name)
        category = get_event_category(event_name)
        risk_emoji = "🔴" if risk >= 7 else ("🟡" if risk >= 4 else "🟢")
        print(f"  {risk_emoji} [{risk}] {event_name:<20} - {category.value}")

    print("\n✅ Test 1 PASSED: Event type system is correctly defined")


def test_risk_weighted_sampling():
    """Test 2: Risk-weighted event sampling (higher risk = less frequent)."""
    print("\n" + "=" * 80)
    print("Test 2: Risk-Weighted Event Sampling")
    print("=" * 80)

    np_random = np.random.RandomState(42)
    num_samples = 10000

    # Sample events
    sampled_events = [get_random_event_name(np_random) for _ in range(num_samples)]

    # Count by risk level
    risk_counts = {i: 0 for i in range(1, 10)}
    for event_name in sampled_events:
        risk = get_event_risk_level(event_name)
        risk_counts[risk] += 1

    print(f"\n{num_samples}번 샘플링 결과 (위험도별 빈도):")
    for risk in sorted(risk_counts.keys()):
        count = risk_counts[risk]
        pct = (count / num_samples) * 100
        bar = "█" * int(pct * 2)
        print(f"  위험도 {risk}: {count:4d} ({pct:5.2f}%) {bar}")

    # Verify high-risk events are less frequent
    high_risk_total = sum(risk_counts[r] for r in [7, 8, 9])
    low_risk_total = sum(risk_counts[r] for r in [1, 2, 3])
    print(f"\n고위험 (7-9): {high_risk_total} ({high_risk_total/num_samples*100:.1f}%)")
    print(f"저위험 (1-3): {low_risk_total} ({low_risk_total/num_samples*100:.1f}%)")

    assert low_risk_total > high_risk_total, "저위험 이벤트가 고위험보다 많아야 함"

    print("\n✅ Test 2 PASSED: Risk-weighted sampling works correctly")


def test_extended_event_dataclass():
    """Test 3: Extended Event dataclass with backward compatibility."""
    print("\n" + "=" * 80)
    print("Test 3: Extended Event Dataclass")
    print("=" * 80)

    # Create event with new format
    event = Event(
        x=50.0,
        y=30.0,
        risk_level=8,
        event_name="무단침입",
        confidence=0.92,
        detection_time=120.0,
        event_id=1,
    )

    print(f"\n생성된 이벤트:")
    print(f"  위치: ({event.x}, {event.y})")
    print(f"  위험도: {event.risk_level}/9")
    print(f"  이벤트명: {event.event_name}")
    print(f"  신뢰도: {event.confidence:.2f}")
    print(f"  감지 시간: {event.detection_time}s")

    # Test backward compatibility
    print(f"\n하위 호환성 (urgency):")
    print(f"  event.urgency = {event.urgency:.3f}")
    print(f"  (risk_level {event.risk_level} → urgency {event.risk_level/9:.3f})")
    assert abs(event.urgency - event.risk_level / 9.0) < 0.001

    # Test helper properties
    print(f"\n헬퍼 속성:")
    print(f"  is_critical: {event.is_critical}")
    print(f"  is_high_confidence: {event.is_high_confidence}")
    print(f"  requires_immediate_response: {event.requires_immediate_response}")

    # Test validation
    validate_event(event)
    print(f"\n✓ Event validation passed")

    # Test legacy conversion
    legacy_event = Event.from_legacy(
        x=20.0, y=40.0, urgency=0.9, confidence=0.85, detection_time=50.0, event_id=2
    )
    print(f"\nLegacy 변환 테스트:")
    print(f"  urgency 0.9 → risk_level {legacy_event.risk_level}")
    print(f"  event_name: {legacy_event.event_name}")

    print("\n✅ Test 3 PASSED: Extended Event dataclass works correctly")


def test_charging_station_config():
    """Test 4: Charging station configuration in all maps."""
    print("\n" + "=" * 80)
    print("Test 4: Charging Station Configuration")
    print("=" * 80)

    map_configs = [
        "configs/map_large_square.yaml",
        "configs/map_corridor.yaml",
        "configs/map_l_shaped.yaml",
        "configs/map_office_building.yaml",
        "configs/map_campus.yaml",
        "configs/map_warehouse.yaml",
    ]

    print(f"\n각 맵의 충전 스테이션 위치:")
    for config_path in map_configs:
        try:
            config = EnvConfig.load_yaml(config_path)
            map_name = Path(config_path).stem
            charging_pos = config.charging_station_position
            print(f"  ✓ {map_name:<25} → {charging_pos}")

            # Verify charging station is within map bounds
            assert 0 <= charging_pos[0] <= config.map_width, \
                f"Charging station X out of bounds: {charging_pos[0]}"
            assert 0 <= charging_pos[1] <= config.map_height, \
                f"Charging station Y out of bounds: {charging_pos[1]}"

        except Exception as e:
            print(f"  ✗ {map_name:<25} → ERROR: {e}")
            raise

    print("\n✅ Test 4 PASSED: All maps have valid charging station positions")


def test_patrolenv_integration():
    """Test 5: Integration with PatrolEnv."""
    print("\n" + "=" * 80)
    print("Test 5: PatrolEnv Integration")
    print("=" * 80)

    # Load a map config
    config = EnvConfig.load_yaml("configs/map_large_square.yaml")

    # Create environment
    env = PatrolEnv(env_config=config)

    print(f"\n환경 생성 성공:")
    print(f"  맵 크기: {config.map_width}m × {config.map_height}m")
    print(f"  순찰 포인트: {len(config.patrol_points)}개")
    print(f"  충전 스테이션: {config.charging_station_position}")

    # Reset and run a few steps
    obs, info = env.reset(seed=42)
    print(f"\n초기 관측값 shape: {obs.shape}")

    # Run steps and check for events
    events_generated = []
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if event was generated
        if env.current_state.current_event is not None:
            event = env.current_state.current_event
            if event.event_id not in [e.event_id for e in events_generated]:
                events_generated.append(event)
                risk_emoji = "🔴" if event.risk_level >= 7 else (
                    "🟡" if event.risk_level >= 4 else "🟢"
                )
                print(
                    f"\n  {risk_emoji} 이벤트 생성 (step {i+1}): "
                    f"{event.event_name} [위험도 {event.risk_level}]"
                )

        if terminated or truncated:
            break

    if not events_generated:
        print(f"\n  (10 steps 동안 이벤트 생성 없음 - 정상, 확률적 생성)")

    print(f"\n✅ Test 5 PASSED: PatrolEnv integration works correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("산업안전 이벤트 시스템 및 충전 스테이션 통합 테스트")
    print("=" * 80)

    try:
        test_event_type_system()
        test_risk_weighted_sampling()
        test_extended_event_dataclass()
        test_charging_station_config()
        test_patrolenv_integration()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\n산업안전 이벤트 시스템 (34개 이벤트, 위험도 1-9)과")
        print("충전 스테이션 (6개 맵)이 성공적으로 통합되었습니다.\n")

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
