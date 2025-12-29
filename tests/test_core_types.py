"""
Unit tests for core data structures.

Tests the fundamental data types including State, Action, Event, etc.
"""

import pytest
import numpy as np

from rl_dispatch.core.types import (
    PatrolPoint,
    Event,
    RobotState,
    Action,
    ActionMode,
    Candidate,
    State,
    Observation,
    RewardComponents,
    EpisodeMetrics,
)


class TestPatrolPoint:
    """Test PatrolPoint dataclass."""

    def test_creation(self):
        """Test basic creation."""
        point = PatrolPoint(x=10.0, y=5.0, last_visit_time=100.0, priority=1.0, point_id=0)
        assert point.x == 10.0
        assert point.y == 5.0
        assert point.position == (10.0, 5.0)

    def test_distance_to(self):
        """Test distance calculation."""
        point1 = PatrolPoint(x=0.0, y=0.0, last_visit_time=0.0, priority=1.0, point_id=0)
        point2 = PatrolPoint(x=3.0, y=4.0, last_visit_time=0.0, priority=1.0, point_id=1)
        assert point1.distance_to(point2) == 5.0

    def test_time_since_visit(self):
        """Test time since visit calculation."""
        point = PatrolPoint(x=0.0, y=0.0, last_visit_time=50.0, priority=1.0, point_id=0)
        assert point.time_since_visit(100.0) == 50.0


class TestEvent:
    """Test Event dataclass."""

    def test_creation(self):
        """Test basic creation."""
        event = Event(
            x=20.0,
            y=15.0,
            urgency=0.8,
            confidence=0.95,
            detection_time=100.0,
            event_id=1,
        )
        assert event.urgency == 0.8
        assert event.confidence == 0.95
        assert event.is_active

    def test_time_elapsed(self):
        """Test time elapsed calculation."""
        event = Event(
            x=0.0,
            y=0.0,
            urgency=1.0,
            confidence=1.0,
            detection_time=50.0,
            event_id=0,
        )
        assert event.time_elapsed(150.0) == 100.0

    def test_distance_to(self):
        """Test distance calculation."""
        event = Event(
            x=3.0,
            y=4.0,
            urgency=1.0,
            confidence=1.0,
            detection_time=0.0,
            event_id=0,
        )
        assert event.distance_to(0.0, 0.0) == 5.0


class TestRobotState:
    """Test RobotState dataclass."""

    def test_creation(self):
        """Test basic creation."""
        robot = RobotState(
            x=5.0,
            y=3.0,
            heading=0.0,
            velocity=0.5,
            angular_velocity=0.0,
            battery_level=0.8,
            current_goal_idx=2,
        )
        assert robot.position == (5.0, 3.0)
        assert robot.battery_level == 0.8

    def test_heading_vector(self):
        """Test heading vector calculation."""
        robot = RobotState(
            x=0.0,
            y=0.0,
            heading=0.0,  # East
            velocity=0.0,
            angular_velocity=0.0,
            battery_level=1.0,
        )
        hx, hy = robot.heading_vector
        assert abs(hx - 1.0) < 1e-6
        assert abs(hy - 0.0) < 1e-6


class TestAction:
    """Test Action dataclass."""

    def test_creation(self):
        """Test basic creation."""
        action = Action(mode=ActionMode.DISPATCH, replan_idx=2)
        assert action.mode == ActionMode.DISPATCH
        assert action.replan_idx == 2

    def test_to_tuple(self):
        """Test conversion to tuple."""
        action = Action(mode=ActionMode.PATROL, replan_idx=3)
        assert action.to_tuple() == (0, 3)

    def test_from_tuple(self):
        """Test creation from tuple."""
        action = Action.from_tuple((1, 2))
        assert action.mode == ActionMode.DISPATCH
        assert action.replan_idx == 2


class TestObservation:
    """Test Observation dataclass."""

    def test_creation(self):
        """Test basic creation."""
        vector = np.zeros(77, dtype=np.float32)
        obs = Observation(vector=vector)
        assert obs.dim == 77
        assert obs.vector.shape == (77,)

    def test_invalid_dimension(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError):
            Observation(vector=np.zeros(50, dtype=np.float32))

    def test_to_dict(self):
        """Test conversion to dictionary."""
        vector = np.random.randn(77).astype(np.float32)
        obs = Observation(vector=vector)
        d = obs.to_dict()
        assert "goal_relative" in d
        assert "lidar" in d
        assert len(d["lidar"]) == 64


class TestRewardComponents:
    """Test RewardComponents dataclass."""

    def test_creation(self):
        """Test basic creation."""
        rewards = RewardComponents(event=10.0, patrol=-5.0, safety=0.0, efficiency=-2.0)
        assert rewards.event == 10.0
        assert rewards.patrol == -5.0

    def test_compute_total(self):
        """Test total computation."""
        from rl_dispatch.core.config import RewardConfig

        rewards = RewardComponents(event=10.0, patrol=-5.0, safety=0.0, efficiency=-2.0)
        config = RewardConfig(w_event=1.0, w_patrol=0.5, w_safety=2.0, w_efficiency=0.1)
        total = rewards.compute_total(config)
        expected = 1.0 * 10.0 + 0.5 * (-5.0) + 2.0 * 0.0 + 0.1 * (-2.0)
        assert abs(total - expected) < 1e-6


class TestEpisodeMetrics:
    """Test EpisodeMetrics dataclass."""

    def test_creation(self):
        """Test basic creation."""
        metrics = EpisodeMetrics(
            episode_return=234.5,
            episode_length=87,
            events_detected=5,
            events_responded=4,
            events_successful=3,
        )
        assert metrics.episode_return == 234.5
        assert metrics.episode_length == 87

    def test_response_rate(self):
        """Test event response rate calculation."""
        metrics = EpisodeMetrics(
            events_detected=5,
            events_responded=4,
        )
        assert metrics.event_response_rate == 0.8

    def test_success_rate(self):
        """Test event success rate calculation."""
        metrics = EpisodeMetrics(
            events_detected=5,
            events_successful=3,
        )
        assert metrics.event_success_rate == 0.6
