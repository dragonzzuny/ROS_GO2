"""
Candidate generation strategies for patrol route planning.

This module implements 10 different heuristic strategies for generating patrol
route candidates:

1. Keep-Order: Maintain current patrol sequence (baseline)
2. Nearest-First: Greedy nearest-neighbor (optimize immediate efficiency)
3. Most-Overdue-First: Visit most overdue points first (coverage recovery)
4. Overdue-ETA-Balance: Balance overdue time and travel time (hybrid)
5. Risk-Weighted: Prioritize high-risk areas (priority-based)
6. Balanced-Coverage: Minimize maximum coverage gap (optimal distribution)
7. Overdue-Threshold-First: Prioritize points exceeding threshold (critical handling)
8. Windowed-Replan: Replan first H points only (computational efficiency)
9. Minimal-Deviation-Insert: Insert urgent points with minimal detour (stability)
10. Shortest-ETA-First: Sort by estimated arrival time (Nav2-aware planning)

Each generator produces a Candidate with estimated metrics for RL policy selection.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
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
        self.nav_interface = None  # Reviewer 박용준: Will be set by CandidateFactory

    def set_nav_interface(self, nav_interface):
        """
        Set navigation interface for realistic path planning.

        Reviewer 박용준: Enables A* pathfinding for feasibility checks and ETA.

        Args:
            nav_interface: NavigationInterface instance (SimulatedNav2 or RealNav2)
        """
        self.nav_interface = nav_interface

    def _get_distance_between(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
    ) -> float:
        """
        Get distance between two positions.

        Reviewer 박용준: Uses A* path distance if available, else Euclidean.
        """
        # Try to use A* pathfinder if available
        if self.nav_interface is not None:
            if hasattr(self.nav_interface, 'pathfinder') and self.nav_interface.pathfinder is not None:
                return self.nav_interface.pathfinder.get_distance(pos1, pos2)

        # Fallback to Euclidean distance
        return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

    def _get_eta(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> float:
        """
        Get estimated time of arrival from start to goal.

        Reviewer 박용준: Uses NavigationInterface.get_eta() if available.
        """
        if self.nav_interface is not None:
            return self.nav_interface.get_eta(start, goal)
        else:
            distance = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
            return distance / 1.0

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

        Reviewer 박용준: Now uses A* path distance instead of Euclidean.

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
            next_pos = (point.x, point.y)

            # Reviewer 박용준: Use A* distance
            distance = self._get_distance_between(current_pos, next_pos)

            if distance == np.inf:
                return np.inf  # Infeasible route

            total_distance += distance
            current_pos = next_pos

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

        Reviewer 박용준: Now uses A* ETA instead of Euclidean distance.

        Args:
            patrol_points: All patrol points
            visit_order: Proposed visit sequence
            current_time: Current episode time
            robot_pos: Robot's current position
            avg_velocity: Average robot velocity for ETA estimation (unused if nav_interface available)

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
            next_pos = (point.x, point.y)

            # Reviewer 박용준: Use realistic ETA
            eta = self._get_eta(current_pos, next_pos)

            if eta == np.inf:
                return np.inf  # Infeasible

            estimated_time += eta

            # Calculate gap: time from last visit to estimated arrival
            time_since_last_visit = estimated_time - point.last_visit_time
            max_gap = max(max_gap, time_since_last_visit)

            current_pos = next_pos

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
            # Reviewer 박용준: Find nearest using A* distance
            nearest_idx = None
            nearest_dist = np.inf

            for idx in remaining:
                point = patrol_points[idx]
                dist = self._get_distance_between(current_pos, point.position)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = idx

            if nearest_idx is None:
                nearest_idx = min(remaining)

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

                # Reviewer 박용준: Calculate A* distance
                distance = self._get_distance_between(current_pos, point.position)

                if distance == np.inf:
                    continue  # Skip infeasible

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
            if best_idx is None:
                break  # No feasible points remain

            visit_order.append(best_idx)
            remaining.remove(best_idx)

            # Update position and time estimate
            point = patrol_points[best_idx]
            eta = self._get_eta(current_pos, point.position)
            estimated_time += eta if eta != np.inf else 10.0  # Fallback
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

                # Reviewer 박용준: Calculate ETA using A*
                eta = self._get_eta(current_pos, point.position)

                if eta == np.inf:
                    continue  # Skip infeasible

                arrival_time = estimated_time + eta

                # Calculate max gap if we visit this point next
                temp_projected = projected_visit_times.copy()
                temp_projected[idx] = arrival_time

                # Max gap across all points (including remaining)
                max_gap = max(
                    estimated_time + eta - temp_projected[i]
                    for i in remaining
                )

                if max_gap < best_max_gap:
                    best_max_gap = max_gap
                    best_idx = idx

            # Add best point
            if best_idx is None:
                break  # No feasible points remain

            visit_order.append(best_idx)
            remaining.remove(best_idx)

            # Update state
            point = patrol_points[best_idx]
            eta = self._get_eta(current_pos, point.position)
            estimated_time += eta if eta != np.inf else 10.0
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


class OverdueThresholdFirstGenerator(CandidateGenerator):
    """
    Prioritize points exceeding overdue threshold.

    Visits points with gap > threshold first, then handles remaining points
    by nearest-first strategy.

    Algorithm:
        1. Find all points with (current_time - last_visit) > threshold
        2. Sort overdue points by gap (descending)
        3. Append remaining points sorted by distance

    This strategy ensures critical overdue points are handled immediately
    while maintaining efficiency for non-critical points.

    Args:
        threshold: Overdue threshold in seconds (default: 60s)
    """

    def __init__(self, threshold: float = 60.0):
        super().__init__(strategy_name="overdue_threshold_first", strategy_id=6)
        self.threshold = threshold

    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """Generate candidate prioritizing threshold-exceeding points."""
        # Separate points by threshold
        overdue = []
        normal = []

        for idx, point in enumerate(patrol_points):
            gap = current_time - point.last_visit_time
            if gap > self.threshold:
                overdue.append((idx, gap))
            else:
                normal.append(idx)

        # Sort overdue by gap (descending)
        overdue.sort(key=lambda x: x[1], reverse=True)
        overdue_order = [idx for idx, _ in overdue]

        # Reviewer 박용준: Sort normal by A* distance (nearest first)
        normal_with_dist = [
            (idx, self._get_distance_between(robot.position, patrol_points[idx].position))
            for idx in normal
        ]
        normal_with_dist.sort(key=lambda x: x[1])
        normal_order = [idx for idx, _ in normal_with_dist]

        # Combine: overdue first, then normal
        visit_order = overdue_order + normal_order

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


class WindowedReplanGenerator(CandidateGenerator):
    """
    Replan only the first H points (windowed replanning).

    Reorders only the first H patrol points while keeping the rest unchanged.
    Reduces computational cost and maintains long-term route stability.

    Algorithm:
        1. Take first H points from current route
        2. Reorder them by most-overdue-first within window
        3. Keep remaining points in original order

    This provides a balance between responsiveness and stability.

    Args:
        window_size: Number of points to replan (default: 3)
    """

    def __init__(self, window_size: int = 3):
        super().__init__(strategy_name="windowed_replan", strategy_id=7)
        self.window_size = window_size

    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """Generate candidate with windowed replanning."""
        num_points = len(patrol_points)

        # Default sequential order
        base_order = list(range(num_points))

        # Take first H points for replanning
        window_size = min(self.window_size, num_points)
        window_indices = base_order[:window_size]

        # Reorder window by overdue time (descending)
        window_with_gap = [
            (idx, current_time - patrol_points[idx].last_visit_time)
            for idx in window_indices
        ]
        window_with_gap.sort(key=lambda x: x[1], reverse=True)
        reordered_window = [idx for idx, _ in window_with_gap]

        # Combine reordered window + remaining points
        visit_order = reordered_window + base_order[window_size:]

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


class MinimalDeviationInsertGenerator(CandidateGenerator):
    """
    Insert urgent points into current route with minimal deviation.

    Finds the best insertion position for most-overdue points while
    minimizing deviation from current route.

    Algorithm:
        1. Identify most overdue point
        2. For each position in current route:
           Calculate detour cost if inserting at that position
        3. Insert at position with minimal detour
        4. Repeat for remaining points

    This maintains route stability while handling urgent points.
    """

    def __init__(self):
        super().__init__(strategy_name="minimal_deviation_insert", strategy_id=8)

    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """Generate candidate with minimal-deviation insertion."""
        # Start with sequential order
        visit_order = []
        remaining = set(range(len(patrol_points)))

        # Insert points one by one
        while remaining:
            # Find most overdue remaining point
            best_point = None
            best_gap = -np.inf
            for idx in remaining:
                gap = current_time - patrol_points[idx].last_visit_time
                if gap > best_gap:
                    best_gap = gap
                    best_point = idx

            if not visit_order:
                # First point: just add it
                visit_order.append(best_point)
            else:
                # Find best insertion position
                best_pos = len(visit_order)
                best_detour = np.inf

                for pos in range(len(visit_order) + 1):
                    # Calculate detour cost
                    test_order = visit_order[:pos] + [best_point] + visit_order[pos:]
                    detour = self._estimate_route_distance(
                        robot.position, patrol_points, test_order
                    )

                    if detour < best_detour:
                        best_detour = detour
                        best_pos = pos

                # Insert at best position
                visit_order.insert(best_pos, best_point)

            remaining.remove(best_point)

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


class ShortestETAFirstGenerator(CandidateGenerator):
    """
    Sort patrol points by estimated time of arrival (ETA).

    Uses NavigationInterface.get_eta() to estimate arrival time to each point,
    then visits points in order of shortest ETA first.

    This differs from nearest-first by considering actual path planning costs
    rather than Euclidean distance.

    Note:
        Currently uses Euclidean-based ETA estimation. When Nav2 integration is
        complete, this will use actual Nav2 path planning for ETA.

    Args:
        avg_velocity: Average robot velocity for ETA estimation (m/s)
    """

    def __init__(self, avg_velocity: float = 1.0):
        super().__init__(strategy_name="shortest_eta_first", strategy_id=9)
        self.avg_velocity = avg_velocity

    def generate(
        self,
        robot: RobotState,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> Candidate:
        """Generate candidate sorted by ETA."""
        # Reviewer 박용준: Calculate ETA using A* pathfinding
        etas = []
        for idx, point in enumerate(patrol_points):
            eta = self._get_eta(robot.position, point.position)
            etas.append((idx, eta))

        # Sort by ETA (ascending)
        etas.sort(key=lambda x: x[1])
        visit_order = [idx for idx, _ in etas]

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

    Provides a single interface to instantiate and manage all 10 strategy generators.
    Used by the environment to generate candidates at each decision point.

    Example:
        >>> factory = CandidateFactory()
        >>> candidates = factory.generate_all(robot, patrol_points, current_time)
        >>> assert len(candidates) == 10
        >>> for c in candidates:
        ...     print(f"{c.strategy_name}: distance={c.estimated_total_distance:.1f}m")
    """

    def __init__(self):
        """Initialize factory with all 10 generators."""
        self.generators = [
            KeepOrderGenerator(),
            NearestFirstGenerator(),
            MostOverdueFirstGenerator(),
            OverdueETABalanceGenerator(),
            RiskWeightedGenerator(),
            BalancedCoverageGenerator(),
            OverdueThresholdFirstGenerator(),
            WindowedReplanGenerator(),
            MinimalDeviationInsertGenerator(),
            ShortestETAFirstGenerator(),
        ]
        self.nav_interface = None  # Reviewer 박용준: Will be set by environment

    def set_nav_interface(self, nav_interface):
        """
        Set navigation interface for all generators.

        Reviewer 박용준: CRITICAL - Must be called before generate_all()!

        Args:
            nav_interface: NavigationInterface instance with pathfinder
        """
        self.nav_interface = nav_interface
        for generator in self.generators:
            generator.set_nav_interface(nav_interface)

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
