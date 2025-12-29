# Reviewer: 박용준
"""
Navigation interface abstraction for patrol robots.

Provides unified interface for both simulated navigation (training) and
real Nav2 integration (deployment).

Changes by Reviewer 박용준:
- Integrated A* pathfinding for realistic navigation
- Added occupancy grid support
- Realistic ETA based on actual path length (not Euclidean distance)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np

# Import A* pathfinder
try:
    from rl_dispatch.navigation.pathfinding import AStarPathfinder
except ImportError:
    AStarPathfinder = None  # Fallback for backwards compatibility


@dataclass
class NavigationResult:
    """
    Result of a navigation action.

    Attributes:
        time: Time taken for navigation (seconds)
        success: Whether navigation succeeded
        collision: Whether collision occurred
        path: Optional path taken (list of (x, y) waypoints)
    """
    time: float
    success: bool
    collision: bool
    path: Optional[List[Tuple[float, float]]] = None


class NavigationInterface(ABC):
    """
    Abstract interface for navigation systems.

    This interface abstracts the differences between simulated navigation
    (used during RL training) and real Nav2 navigation (used during deployment).

    The interface supports:
    - Path planning (plan_path)
    - ETA estimation (get_eta)
    - Navigation execution (navigate_to_goal)
    """

    @abstractmethod
    def navigate_to_goal(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> NavigationResult:
        """
        Navigate from start to goal position.

        Args:
            start: Starting (x, y) position in meters
            goal: Goal (x, y) position in meters

        Returns:
            NavigationResult with time, success, collision info
        """
        pass

    @abstractmethod
    def get_eta(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> float:
        """
        Get estimated time of arrival (ETA) from start to goal.

        This is used by heuristic strategies for planning.

        Args:
            start: Starting (x, y) position in meters
            goal: Goal (x, y) position in meters

        Returns:
            Estimated time in seconds
        """
        pass

    @abstractmethod
    def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal.

        Args:
            start: Starting (x, y) position in meters
            goal: Goal (x, y) position in meters

        Returns:
            List of (x, y) waypoints, or None if planning failed
        """
        pass


class SimulatedNav2(NavigationInterface):
    """
    Realistic navigation simulation for RL training with A* pathfinding.

    Changes by Reviewer 박용준:
    - Uses A* pathfinding on occupancy grid (no wall penetration)
    - Realistic ETA based on actual path length
    - Dynamic obstacle support via grid updates

    Args:
        occupancy_grid: 2D occupancy grid (0=free, 1=occupied)
        grid_resolution: Meters per grid cell
        max_velocity: Maximum robot velocity (m/s)
        nav_failure_rate: Probability of navigation failure (0-1)
        collision_rate: Probability of collision (0-1)
        np_random: NumPy random state for reproducibility
    """

    def __init__(
        self,
        occupancy_grid: Optional[np.ndarray] = None,
        grid_resolution: float = 0.5,
        max_velocity: float = 1.5,
        nav_failure_rate: float = 0.05,
        collision_rate: float = 0.01,
        np_random: Optional[np.random.RandomState] = None,
    ):
        self.max_velocity = max_velocity
        self.nav_failure_rate = nav_failure_rate
        self.collision_rate = collision_rate
        self.np_random = np_random if np_random is not None else np.random.RandomState()

        # Reviewer 박용준: A* pathfinder integration
        if occupancy_grid is not None and AStarPathfinder is not None:
            self.pathfinder = AStarPathfinder(occupancy_grid, grid_resolution)
            self.use_astar = True
        else:
            self.pathfinder = None
            self.use_astar = False

    def navigate_to_goal(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> NavigationResult:
        """
        Simulate navigation to goal with A* pathfinding.

        Reviewer 박용준: Uses A* if available, falls back to straight-line

        Returns:
            NavigationResult with simulated time, success, collision
        """
        if self.use_astar and self.pathfinder is not None:
            # A* pathfinding (realistic)
            path_result = self.pathfinder.find_path(start, goal)

            if path_result is None:
                # No path exists
                return NavigationResult(
                    time=0.0,
                    success=False,
                    collision=False,
                    path=None
                )

            path, distance = path_result

            # Time calculation based on actual path length
            avg_velocity = self.max_velocity * 0.7
            base_time = distance / avg_velocity if avg_velocity > 0 else 0.0

            # Add random variation
            time_variation = self.np_random.normal(1.0, 0.1)
            nav_time = max(0.1, base_time * time_variation)

            # Simulate navigation failure and collision
            nav_success = self.np_random.random() > self.nav_failure_rate
            collision = self.np_random.random() < self.collision_rate if nav_success else False

            return NavigationResult(
                time=nav_time,
                success=nav_success and not collision,
                collision=collision,
                path=path if nav_success else None
            )

        else:
            # Fallback: straight-line (original behavior)
            distance = np.sqrt(
                (goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2
            )

            avg_velocity = self.max_velocity * 0.7
            base_time = distance / avg_velocity if avg_velocity > 0 else 0.0

            time_variation = self.np_random.normal(1.0, 0.1)
            nav_time = max(0.1, base_time * time_variation)

            nav_success = self.np_random.random() > self.nav_failure_rate
            collision = self.np_random.random() < self.collision_rate

            if collision:
                nav_success = False

            path = [start, goal] if nav_success else None

            return NavigationResult(
                time=nav_time,
                success=nav_success,
                collision=collision,
                path=path,
            )

    def get_eta(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> float:
        """
        Get estimated time of arrival using A* path length.

        Reviewer 박용준: Uses A* distance instead of Euclidean

        Returns:
            Estimated time in seconds (or np.inf if no path)
        """
        if self.use_astar and self.pathfinder is not None:
            # A* distance (realistic)
            distance = self.pathfinder.get_distance(start, goal)

            if distance == np.inf:
                return np.inf  # No path exists

            avg_velocity = self.max_velocity * 0.7
            return distance / avg_velocity if avg_velocity > 0 else 0.0

        else:
            # Fallback: straight-line
            distance = np.sqrt(
                (goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2
            )
            avg_velocity = self.max_velocity * 0.7
            return distance / avg_velocity if avg_velocity > 0 else 0.0

    def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using A*.

        Reviewer 박용준: Uses A* if available

        Returns:
            List of waypoints or None if no path
        """
        if self.use_astar and self.pathfinder is not None:
            # A* pathfinding
            result = self.pathfinder.find_path(start, goal)
            return result[0] if result else None
        else:
            # Fallback: straight-line
            return [start, goal]

    def update_occupancy_grid(self, new_grid: np.ndarray):
        """
        Update occupancy grid (for dynamic obstacles).

        Reviewer 박용준: Allows updating grid with dynamic obstacles

        Args:
            new_grid: Updated occupancy grid with dynamic obstacles
        """
        if self.use_astar and self.pathfinder is not None:
            self.pathfinder.grid = new_grid
        # Simulate occasional planning failure
        if self.np_random.random() < self.nav_failure_rate:
            return None

        # Simple straight-line path
        return [start, goal]


class RealNav2(NavigationInterface):
    """
    Real Nav2 integration for deployment (placeholder).

    This would interface with actual ROS2 Nav2 stack for production deployment.
    Currently not implemented - requires ROS2 environment.

    Usage in deployment:
        ```python
        import rclpy
        from nav2_msgs.action import NavigateToPose

        nav2 = RealNav2(ros_node)
        result = nav2.navigate_to_goal(start, goal)
        ```

    Args:
        ros_node: ROS2 node for Nav2 action client
    """

    def __init__(self, ros_node=None):
        self.ros_node = ros_node
        # TODO: Initialize Nav2 action clients
        # self.navigate_to_pose_client = ActionClient(
        #     ros_node, NavigateToPose, '/navigate_to_pose'
        # )
        raise NotImplementedError(
            "RealNav2 integration requires ROS2 environment. "
            "Use SimulatedNav2 for training."
        )

    def navigate_to_goal(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> NavigationResult:
        """Send navigation goal to Nav2."""
        # TODO: Implement Nav2 action call
        raise NotImplementedError("RealNav2 not yet implemented")

    def get_eta(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> float:
        """Get ETA from Nav2 planner."""
        # TODO: Query Nav2 planner for path cost
        raise NotImplementedError("RealNav2 not yet implemented")

    def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """Request path from Nav2 planner."""
        # TODO: Call Nav2 path planning service
        raise NotImplementedError("RealNav2 not yet implemented")
