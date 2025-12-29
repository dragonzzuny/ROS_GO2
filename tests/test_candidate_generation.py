"""
Unit tests for candidate generation strategies.

Tests all 6 heuristic strategies for patrol route planning.
"""

import pytest
import numpy as np

from rl_dispatch.core.types import PatrolPoint, RobotState
from rl_dispatch.planning import (
    KeepOrderGenerator,
    NearestFirstGenerator,
    MostOverdueFirstGenerator,
    OverdueETABalanceGenerator,
    RiskWeightedGenerator,
    BalancedCoverageGenerator,
    CandidateFactory,
)


@pytest.fixture
def sample_robot():
    """Create sample robot state."""
    return RobotState(
        x=5.0,
        y=5.0,
        heading=0.0,
        velocity=0.5,
        angular_velocity=0.0,
        battery_level=0.8,
    )


@pytest.fixture
def sample_patrol_points():
    """Create sample patrol points."""
    return (
        PatrolPoint(x=10.0, y=10.0, last_visit_time=0.0, priority=1.0, point_id=0),
        PatrolPoint(x=40.0, y=10.0, last_visit_time=50.0, priority=1.0, point_id=1),
        PatrolPoint(x=40.0, y=40.0, last_visit_time=100.0, priority=1.0, point_id=2),
        PatrolPoint(x=10.0, y=40.0, last_visit_time=150.0, priority=1.0, point_id=3),
    )


class TestKeepOrderGenerator:
    """Test Keep-Order strategy."""

    def test_maintains_order(self, sample_robot, sample_patrol_points):
        """Test that it maintains original order."""
        generator = KeepOrderGenerator()
        candidate = generator.generate(sample_robot, sample_patrol_points, current_time=200.0)

        assert candidate.patrol_order == (0, 1, 2, 3)
        assert candidate.strategy_name == "keep_order"
        assert candidate.strategy_id == 0


class TestNearestFirstGenerator:
    """Test Nearest-First strategy."""

    def test_visits_nearest(self, sample_robot, sample_patrol_points):
        """Test that it prioritizes nearest points."""
        generator = NearestFirstGenerator()
        candidate = generator.generate(sample_robot, sample_patrol_points, current_time=200.0)

        # Robot at (5, 5), should visit (10, 10) first
        assert candidate.patrol_order[0] == 0
        assert candidate.strategy_name == "nearest_first"

    def test_all_points_visited(self, sample_robot, sample_patrol_points):
        """Test that all points are visited exactly once."""
        generator = NearestFirstGenerator()
        candidate = generator.generate(sample_robot, sample_patrol_points, current_time=200.0)

        assert len(candidate.patrol_order) == 4
        assert set(candidate.patrol_order) == {0, 1, 2, 3}


class TestMostOverdueFirstGenerator:
    """Test Most-Overdue-First strategy."""

    def test_visits_overdue_first(self, sample_patrol_points):
        """Test that it prioritizes most overdue points."""
        robot = RobotState(x=25.0, y=25.0, heading=0.0, velocity=0.5,
                          angular_velocity=0.0, battery_level=0.8)

        generator = MostOverdueFirstGenerator()
        candidate = generator.generate(robot, sample_patrol_points, current_time=200.0)

        # Point 0 has oldest visit time (0.0), should be first
        assert candidate.patrol_order[0] == 0
        assert candidate.strategy_name == "most_overdue_first"


class TestOverdueETABalanceGenerator:
    """Test Overdue-ETA-Balance strategy."""

    def test_generates_valid_route(self, sample_robot, sample_patrol_points):
        """Test that it generates valid route."""
        generator = OverdueETABalanceGenerator()
        candidate = generator.generate(sample_robot, sample_patrol_points, current_time=200.0)

        assert len(candidate.patrol_order) == 4
        assert set(candidate.patrol_order) == {0, 1, 2, 3}
        assert candidate.strategy_name == "overdue_eta_balance"


class TestRiskWeightedGenerator:
    """Test Risk-Weighted strategy."""

    def test_respects_priority(self):
        """Test that it respects patrol point priorities."""
        robot = RobotState(x=25.0, y=25.0, heading=0.0, velocity=0.5,
                          angular_velocity=0.0, battery_level=0.8)

        # Create points with different priorities
        patrol_points = (
            PatrolPoint(x=10.0, y=10.0, last_visit_time=0.0, priority=0.5, point_id=0),
            PatrolPoint(x=40.0, y=10.0, last_visit_time=0.0, priority=2.0, point_id=1),  # High priority
            PatrolPoint(x=40.0, y=40.0, last_visit_time=0.0, priority=1.0, point_id=2),
        )

        generator = RiskWeightedGenerator()
        candidate = generator.generate(robot, patrol_points, current_time=100.0)

        # Point 1 has highest priority * overdue score
        assert candidate.patrol_order[0] == 1
        assert candidate.strategy_name == "risk_weighted"


class TestBalancedCoverageGenerator:
    """Test Balanced-Coverage strategy."""

    def test_generates_valid_route(self, sample_robot, sample_patrol_points):
        """Test that it generates valid route."""
        generator = BalancedCoverageGenerator()
        candidate = generator.generate(sample_robot, sample_patrol_points, current_time=200.0)

        assert len(candidate.patrol_order) == 4
        assert set(candidate.patrol_order) == {0, 1, 2, 3}
        assert candidate.strategy_name == "balanced_coverage"


class TestCandidateFactory:
    """Test CandidateFactory."""

    def test_generates_all_candidates(self, sample_robot, sample_patrol_points):
        """Test that factory generates all 6 candidates."""
        factory = CandidateFactory()
        candidates = factory.generate_all(sample_robot, sample_patrol_points, current_time=200.0)

        assert len(candidates) == 6
        assert factory.num_strategies == 6

    def test_unique_strategies(self, sample_robot, sample_patrol_points):
        """Test that all strategies are unique."""
        factory = CandidateFactory()
        candidates = factory.generate_all(sample_robot, sample_patrol_points, current_time=200.0)

        strategy_names = [c.strategy_name for c in candidates]
        assert len(strategy_names) == len(set(strategy_names))

    def test_get_generator(self):
        """Test getting generator by name."""
        factory = CandidateFactory()
        generator = factory.get_generator("nearest_first")
        assert isinstance(generator, NearestFirstGenerator)

    def test_get_invalid_generator(self):
        """Test that invalid generator name raises error."""
        factory = CandidateFactory()
        with pytest.raises(ValueError):
            factory.get_generator("invalid_strategy")
