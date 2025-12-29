"""
Candidate generation strategies for patrol route planning.

This module implements 6 different heuristic strategies for generating patrol
route candidates:

1. Keep-Order: Maintain current patrol sequence (baseline)
2. Nearest-First: Greedy nearest-neighbor (optimize immediate efficiency)
3. Most-Overdue-First: Visit most overdue points first (coverage recovery)
4. Overdue-ETA-Balance: Balance overdue time and travel time (hybrid)
5. Risk-Weighted: Prioritize high-risk areas (priority-based)
6. Balanced-Coverage: Minimize maximum coverage gap (optimal distribution)

Each generator produces a Candidate with estimated metrics for RL policy selection.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from rl_dispatch.core.types import Candidate, PatrolPoint, RobotState


class CandidateGenerator(ABC):
    """
    Abstract base class for candidate generation strategies.

    All candidate generators must implement the generate() method which takes
    current patrol state and produces a Candidate with visit order and metrics.

    Subclasses:
        - KeepOrderGenerator: Baseline (maintain current order)
        - NearestFirstGenerator: Greedy nearest-neighbor
        - MostOverdueFirstGenerator: Prioritize coverage gaps
        - OverdueETABalanceGenerator: Balance gap and travel time
        - RiskWeightedGenerator: Prioritize high-priority areas
        - BalancedCoverageGenerator: Optimize maximum gap
    """

    def __init__(self, strategy_name: str, strategy_id: int):
        """
        Initialize candidate generator.

        Args:
            strategy_name: Human-readable name for this strategy
            strategy_id: Unique integer ID for this strategy
        """
        self.strategy_name = strategy_name
        self.strategy_id = strategy_id

    @abstractmethod
    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """
        Generate a patrol route candidate.

        Args:
            robot: Current robot state
            patrol_points: All patrol points with visit times
            current_time: Current episode time

        Returns:
            Candidate with patrol order and estimated metrics
        """
        pass

    def _estimate_route_distance(
        self,
        robot_pos: Tuple[float, float],
        patrol_points: Tuple[PatrolPoint, ...],
        visit_order: List[int],
    ) -> float:
        """
        Estimate total distance for a patrol route.

        Calculates Euclidean distance from robot's current position through
        all patrol points in the given order.

        Args:
            robot_pos: Robot's (x, y) position
            patrol_points: All patrol points
            visit_order: Sequence of patrol point indices to visit

        Returns:
            Estimated total distance in meters
        """
        if len(visit_order) == 0:
            return 0.0

        total_distance = 0.0
        current_pos = robot_pos

        for idx in visit_order:
            point = patrol_points[idx]
            dx = point.x - current_pos[0]
            dy = point.y - current_pos[1]
            total_distance += np.sqrt(dx**2 + dy**2)
            current_pos = (point.x, point.y)

        return total_distance

    def _calculate_max_coverage_gap(
        self,
        patrol_points: Tuple[PatrolPoint, ...],
        visit_order: List[int],
        current_time: float,
        robot_pos: Tuple[float, float],
        avg_velocity: float = 1.0,
    ) -> float:
        """
        Calculate maximum coverage gap for a route.

        Estimates the maximum time any patrol point will go unvisited.

        Args:
            patrol_points: All patrol points
            visit_order: Proposed visit sequence
            current_time: Current episode time
            robot_pos: Robot's current position
            avg_velocity: Average robot velocity for ETA estimation

        Returns:
            Maximum coverage gap in seconds
        """
        if len(visit_order) == 0:
            return 0.0

        max_gap = 0.0
        current_pos = robot_pos
        estimated_time = current_time

        for idx in visit_order:
            point = patrol_points[idx]

            # Calculate travel time to this point
            distance = np.sqrt(
                (point.x - current_pos[0])**2 + (point.y - current_pos[1])**2
            )
            travel_time = distance / avg_velocity
            estimated_time += travel_time

            # Calculate gap: time from last visit to estimated arrival
            time_since_last_visit = estimated_time - point.last_visit_time
            max_gap = max(max_gap, time_since_last_visit)

            current_pos = (point.x, point.y)

        return max_gap


class KeepOrderGenerator(CandidateGenerator):
    """
    Baseline strategy: Keep current patrol order.

    Simply maintains the existing patrol sequence without any reordering.
    Serves as the baseline/conservative strategy.

    Example:
        If current order is [0, 1, 2, 3], returns [0, 1, 2, 3].
    """

    def __init__(self):
        super().__init__(strategy_name="keep_order", strategy_id=0)

    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """Generate candidate maintaining current patrol order."""
        # Keep original order (by point_id)
        visit_order = [p.point_id for p in patrol_points]

        total_distance = self._estimate_route_distance(
            robot.position, patrol_points, visit_order
        )
        max_gap = self._calculate_max_coverage_gap(
            patrol_points, visit_order, current_time, robot.position
        )

        return Candidate(
            patrol_order=tuple(visit_order),
            estimated_total_distance=total_distance,
            max_coverage_gap=max_gap,
            strategy_name=self.strategy_name,
            strategy_id=self.strategy_id,
        )


class NearestFirstGenerator(CandidateGenerator):
    """
    Greedy nearest-neighbor strategy.

    Iteratively selects the nearest unvisited patrol point. Optimizes for
    immediate travel efficiency but may create coverage gaps.

    Algorithm:
        1. Start from robot's current position
        2. Repeat: Select nearest unvisited patrol point
        3. Continue until all points visited

    Example:
        Robot at (5, 5), points at (6, 5), (20, 20), (7, 6):
        Returns [0, 2, 1] (visits nearby points first)
    """

    def __init__(self):
        super().__init__(strategy_name="nearest_first", strategy_id=1)

    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """Generate candidate using greedy nearest-neighbor."""
        visit_order = []
        remaining = set(range(len(patrol_points)))
        current_pos = robot.position

        while remaining:
            # Find nearest unvisited point
            nearest_idx = min(
                remaining,
                key=lambda idx: np.sqrt(
                    (patrol_points[idx].x - current_pos[0])**2 +
                    (patrol_points[idx].y - current_pos[1])**2
                )
            )
            visit_order.append(nearest_idx)
            remaining.remove(nearest_idx)
            current_pos = patrol_points[nearest_idx].position

        total_distance = self._estimate_route_distance(
            robot.position, patrol_points, visit_order
        )
        max_gap = self._calculate_max_coverage_gap(
            patrol_points, visit_order, current_time, robot.position
        )

        return Candidate(
            patrol_order=tuple(visit_order),
            estimated_total_distance=total_distance,
            max_coverage_gap=max_gap,
            strategy_name=self.strategy_name,
            strategy_id=self.strategy_id,
        )


class MostOverdueFirstGenerator(CandidateGenerator):
    """
    Prioritize most overdue patrol points.

    Visits patrol points in descending order of time since last visit.
    Optimizes for coverage recovery at the expense of travel efficiency.

    Algorithm:
        1. Calculate time_since_visit for each patrol point
        2. Sort points by time_since_visit (descending)
        3. Return sorted order

    Example:
        Point 0 visited 100s ago, Point 1 visited 50s ago, Point 2 visited 200s ago:
        Returns [2, 0, 1] (most overdue first)
    """

    def __init__(self):
        super().__init__(strategy_name="most_overdue_first", strategy_id=2)

    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """Generate candidate prioritizing overdue points."""
        # Calculate time since visit for each point
        overdue_scores = [
            (idx, current_time - point.last_visit_time)
            for idx, point in enumerate(patrol_points)
        ]

        # Sort by overdue time (descending)
        overdue_scores.sort(key=lambda x: x[1], reverse=True)
        visit_order = [idx for idx, _ in overdue_scores]

        total_distance = self._estimate_route_distance(
            robot.position, patrol_points, visit_order
        )
        max_gap = self._calculate_max_coverage_gap(
            patrol_points, visit_order, current_time, robot.position
        )

        return Candidate(
            patrol_order=tuple(visit_order),
            estimated_total_distance=total_distance,
            max_coverage_gap=max_gap,
            strategy_name=self.strategy_name,
            strategy_id=self.strategy_id,
        )


class OverdueETABalanceGenerator(CandidateGenerator):
    """
    Balance overdue time and estimated travel time (hybrid strategy).

    Combines coverage urgency (time since visit) with travel efficiency (distance).
    Uses weighted scoring: score = urgency_weight * overdue - efficiency_weight * distance.

    Algorithm:
        1. For each point, calculate: score = α * time_since_visit - β * distance
        2. Greedily select point with highest score
        3. Repeat until all points visited

    Attributes:
        urgency_weight: Weight for overdue time (default: 1.0)
        efficiency_weight: Weight for distance (default: 0.1)

    Example:
        Balances between visiting overdue points and minimizing travel.
    """

    def __init__(self, urgency_weight: float = 1.0, efficiency_weight: float = 0.1):
        super().__init__(strategy_name="overdue_eta_balance", strategy_id=3)
        self.urgency_weight = urgency_weight
        self.efficiency_weight = efficiency_weight

    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """Generate candidate balancing overdue and travel time."""
        visit_order = []
        remaining = set(range(len(patrol_points)))
        current_pos = robot.position
        estimated_time = current_time

        while remaining:
            # Calculate score for each remaining point
            best_idx = None
            best_score = -np.inf

            for idx in remaining:
                point = patrol_points[idx]

                # Calculate distance
                distance = np.sqrt(
                    (point.x - current_pos[0])**2 + (point.y - current_pos[1])**2
                )

                # Calculate overdue time
                overdue_time = estimated_time - point.last_visit_time

                # Combined score (higher is better)
                score = (
                    self.urgency_weight * overdue_time -
                    self.efficiency_weight * distance
                )

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # Add best point to route
            visit_order.append(best_idx)
            remaining.remove(best_idx)

            # Update position and time estimate
            point = patrol_points[best_idx]
            distance = np.sqrt(
                (point.x - current_pos[0])**2 + (point.y - current_pos[1])**2
            )
            estimated_time += distance / 1.0  # Assume 1 m/s average speed
            current_pos = point.position

        total_distance = self._estimate_route_distance(
            robot.position, patrol_points, visit_order
        )
        max_gap = self._calculate_max_coverage_gap(
            patrol_points, visit_order, current_time, robot.position
        )

        return Candidate(
            patrol_order=tuple(visit_order),
            estimated_total_distance=total_distance,
            max_coverage_gap=max_gap,
            strategy_name=self.strategy_name,
            strategy_id=self.strategy_id,
        )


class RiskWeightedGenerator(CandidateGenerator):
    """
    Prioritize high-risk/high-priority patrol areas.

    Uses patrol point priority weights to determine visit order. High-priority
    areas (e.g., critical infrastructure, known hotspots) are visited first.

    Algorithm:
        1. Calculate weighted urgency: priority * time_since_visit
        2. Sort points by weighted urgency (descending)
        3. Return sorted order

    Note:
        Priority weights are specified in EnvConfig.patrol_point_priorities.
        Default is 1.0 (equal priority), but can be customized per deployment.

    Example:
        Point 0 (priority=1.5, overdue=100s), Point 1 (priority=0.5, overdue=150s):
        Returns [0, 1] because 1.5*100 > 0.5*150
    """

    def __init__(self):
        super().__init__(strategy_name="risk_weighted", strategy_id=4)

    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """Generate candidate prioritizing high-risk areas."""
        # Calculate risk-weighted urgency
        weighted_scores = [
            (idx, point.priority * (current_time - point.last_visit_time))
            for idx, point in enumerate(patrol_points)
        ]

        # Sort by weighted urgency (descending)
        weighted_scores.sort(key=lambda x: x[1], reverse=True)
        visit_order = [idx for idx, _ in weighted_scores]

        total_distance = self._estimate_route_distance(
            robot.position, patrol_points, visit_order
        )
        max_gap = self._calculate_max_coverage_gap(
            patrol_points, visit_order, current_time, robot.position
        )

        return Candidate(
            patrol_order=tuple(visit_order),
            estimated_total_distance=total_distance,
            max_coverage_gap=max_gap,
            strategy_name=self.strategy_name,
            strategy_id=self.strategy_id,
        )


class BalancedCoverageGenerator(CandidateGenerator):
    """
    Minimize maximum coverage gap (optimal distribution).

    Attempts to minimize the largest gap between visits across all points.
    Uses a greedy approach to balance coverage.

    Algorithm:
        1. While points remain:
        2.   For each remaining point, calculate projected max_gap if visited next
        3.   Select point that minimizes max_gap
        4.   Update visit times and repeat

    This is similar to the "minimax" strategy in optimization.

    Example:
        Produces routes that distribute visits evenly over time.
    """

    def __init__(self):
        super().__init__(strategy_name="balanced_coverage", strategy_id=5)

    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """Generate candidate minimizing maximum gap."""
        visit_order = []
        remaining = set(range(len(patrol_points)))
        current_pos = robot.position
        estimated_time = current_time

        # Track projected visit times
        projected_visit_times = {
            idx: point.last_visit_time
            for idx, point in enumerate(patrol_points)
        }

        while remaining:
            best_idx = None
            best_max_gap = np.inf

            for idx in remaining:
                point = patrol_points[idx]

                # Calculate ETA to this point
                distance = np.sqrt(
                    (point.x - current_pos[0])**2 + (point.y - current_pos[1])**2
                )
                travel_time = distance / 1.0  # Assume 1 m/s
                arrival_time = estimated_time + travel_time

                # Calculate max gap if we visit this point next
                temp_projected = projected_visit_times.copy()
                temp_projected[idx] = arrival_time

                # Max gap across all points (including remaining)
                max_gap = max(
                    estimated_time + travel_time - temp_projected[i]
                    for i in remaining
                )

                if max_gap < best_max_gap:
                    best_max_gap = max_gap
                    best_idx = idx

            # Add best point
            visit_order.append(best_idx)
            remaining.remove(best_idx)

            # Update state
            point = patrol_points[best_idx]
            distance = np.sqrt(
                (point.x - current_pos[0])**2 + (point.y - current_pos[1])**2
            )
            estimated_time += distance / 1.0
            projected_visit_times[best_idx] = estimated_time
            current_pos = point.position

        total_distance = self._estimate_route_distance(
            robot.position, patrol_points, visit_order
        )
        max_gap = self._calculate_max_coverage_gap(
            patrol_points, visit_order, current_time, robot.position
        )

        return Candidate(
            patrol_order=tuple(visit_order),
            estimated_total_distance=total_distance,
            max_coverage_gap=max_gap,
            strategy_name=self.strategy_name,
            strategy_id=self.strategy_id,
        )


class CandidateFactory:
    """
    Factory for creating all candidate generators.

    Provides a single interface to instantiate and manage all 6 strategy generators.
    Used by the environment to generate candidates at each decision point.

    Example:
        >>> factory = CandidateFactory()
        >>> candidates = factory.generate_all(robot, patrol_points, current_time)
        >>> assert len(candidates) == 6
        >>> for c in candidates:
        ...     print(f"{c.strategy_name}: distance={c.estimated_total_distance:.1f}m")
    """

    def __init__(self):
        """Initialize factory with all 6 generators."""
        self.generators = [
            KeepOrderGenerator(),
            NearestFirstGenerator(),
            MostOverdueFirstGenerator(),
            OverdueETABalanceGenerator(),
            RiskWeightedGenerator(),
            BalancedCoverageGenerator(),
        ]

    def generate_all(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Tuple[Candidate, ...]:
        """
        Generate all candidates using all strategies.

        Args:
            robot: Current robot state
            patrol_points: All patrol points
            current_time: Current episode time

        Returns:
            Tuple of 6 Candidate objects (one per strategy)
        """
        candidates = tuple(
            generator.generate(robot, patrol_points, current_time)
            for generator in self.generators
        )
        return candidates

    def get_generator(self, strategy_name: str) -> CandidateGenerator:
        """
        Get a specific generator by name.

        Args:
            strategy_name: Name of strategy (e.g., "nearest_first")

        Returns:
            CandidateGenerator instance

        Raises:
            ValueError: If strategy_name not found
        """
        for generator in self.generators:
            if generator.strategy_name == strategy_name:
                return generator
        raise ValueError(f"Unknown strategy: {strategy_name}")

    @property
    def num_strategies(self) -> int:
        """Returns the number of available strategies."""
        return len(self.generators)
