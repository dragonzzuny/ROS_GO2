"""
Unified Robot Interface for Sim2Real deployment.

Provides a common interface for both simulation and real robot control.
The interface abstracts away the differences between:
- Pure simulation (SimulatedNav2)
- Gazebo simulation with ROS2
- Real Unitree Go2 with ROS2 Nav2
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import logging

from .config import DeploymentConfig, DeploymentMode, SimulationConfig, RealRobotConfig


logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    """
    Robot state observation.

    Attributes:
        position: (x, y) position in meters
        heading: Heading angle in radians
        velocity: Linear velocity in m/s
        angular_velocity: Angular velocity in rad/s
        battery_level: Battery level (0.0 to 1.0)
        is_charging: Whether robot is currently charging
        lidar_ranges: LiDAR range readings (64 values)
    """
    position: Tuple[float, float]
    heading: float
    velocity: float
    angular_velocity: float
    battery_level: float
    is_charging: bool
    lidar_ranges: np.ndarray  # Shape: (64,)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "position": self.position,
            "heading": self.heading,
            "velocity": self.velocity,
            "angular_velocity": self.angular_velocity,
            "battery_level": self.battery_level,
            "is_charging": self.is_charging,
            "lidar_ranges": self.lidar_ranges.tolist(),
        }


@dataclass
class NavigationGoal:
    """
    Navigation goal specification.

    Attributes:
        position: Target (x, y) position
        orientation: Optional target orientation (radians)
        timeout: Navigation timeout in seconds
    """
    position: Tuple[float, float]
    orientation: Optional[float] = None
    timeout: float = 60.0


@dataclass
class NavigationFeedback:
    """
    Navigation feedback/result.

    Attributes:
        success: Whether navigation succeeded
        time_elapsed: Time taken for navigation
        distance_traveled: Total distance traveled
        final_position: Final robot position
        error_code: Error code if failed
        error_message: Error message if failed
    """
    success: bool
    time_elapsed: float
    distance_traveled: float
    final_position: Tuple[float, float]
    error_code: int = 0
    error_message: str = ""


class RobotInterface(ABC):
    """
    Abstract robot interface for unified Sim2Real control.

    This interface provides common methods for:
    - Getting robot state (position, velocity, battery, sensors)
    - Sending navigation goals
    - Checking navigation status
    - Emergency stop

    Implementations:
    - SimulationRobotInterface: For pure Python simulation
    - GazeboRobotInterface: For Gazebo simulation with ROS2
    - Go2RobotInterface: For real Unitree Go2 with ROS2
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._is_connected = False
        self._current_goal: Optional[NavigationGoal] = None

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to robot.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from robot."""
        pass

    @abstractmethod
    def get_state(self) -> RobotState:
        """
        Get current robot state.

        Returns:
            Current RobotState observation
        """
        pass

    @abstractmethod
    def send_navigation_goal(self, goal: NavigationGoal) -> bool:
        """
        Send navigation goal to robot.

        Args:
            goal: Navigation goal specification

        Returns:
            True if goal accepted
        """
        pass

    @abstractmethod
    def get_navigation_feedback(self) -> NavigationFeedback:
        """
        Get feedback from current navigation.

        Returns:
            NavigationFeedback with status and metrics
        """
        pass

    @abstractmethod
    def cancel_navigation(self) -> bool:
        """
        Cancel current navigation goal.

        Returns:
            True if cancellation successful
        """
        pass

    @abstractmethod
    def emergency_stop(self) -> bool:
        """
        Execute emergency stop.

        Returns:
            True if stop successful
        """
        pass

    @abstractmethod
    def get_path_to_goal(
        self,
        goal: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Get planned path to goal.

        Args:
            goal: Target (x, y) position

        Returns:
            List of waypoints or None if no path
        """
        pass

    @abstractmethod
    def get_eta(self, goal: Tuple[float, float]) -> float:
        """
        Get estimated time to reach goal.

        Args:
            goal: Target (x, y) position

        Returns:
            Estimated time in seconds (np.inf if unreachable)
        """
        pass

    def navigate_to(
        self,
        goal: Tuple[float, float],
        timeout: float = 60.0,
        blocking: bool = True
    ) -> NavigationFeedback:
        """
        Navigate to goal position.

        Convenience method that combines send_goal and wait.

        Args:
            goal: Target (x, y) position
            timeout: Navigation timeout
            blocking: If True, wait for navigation to complete

        Returns:
            NavigationFeedback with result
        """
        nav_goal = NavigationGoal(position=goal, timeout=timeout)

        if not self.send_navigation_goal(nav_goal):
            return NavigationFeedback(
                success=False,
                time_elapsed=0.0,
                distance_traveled=0.0,
                final_position=self.get_state().position,
                error_code=-1,
                error_message="Failed to send goal"
            )

        if blocking:
            return self._wait_for_navigation()
        else:
            return NavigationFeedback(
                success=True,
                time_elapsed=0.0,
                distance_traveled=0.0,
                final_position=self.get_state().position,
                error_message="Navigation started (non-blocking)"
            )

    def _wait_for_navigation(self) -> NavigationFeedback:
        """Wait for navigation to complete."""
        import time
        start_time = time.time()
        start_position = self.get_state().position

        while True:
            feedback = self.get_navigation_feedback()

            if feedback.success or feedback.error_code != 0:
                return feedback

            if self._current_goal and time.time() - start_time > self._current_goal.timeout:
                self.cancel_navigation()
                current_pos = self.get_state().position
                distance = np.sqrt(
                    (current_pos[0] - start_position[0])**2 +
                    (current_pos[1] - start_position[1])**2
                )
                return NavigationFeedback(
                    success=False,
                    time_elapsed=time.time() - start_time,
                    distance_traveled=distance,
                    final_position=current_pos,
                    error_code=-2,
                    error_message="Navigation timeout"
                )

            time.sleep(0.1)


class SimulationRobotInterface(RobotInterface):
    """
    Robot interface for pure Python simulation.

    Uses SimulatedNav2 for navigation without ROS2 dependency.
    """

    def __init__(self, config: SimulationConfig, nav_interface=None):
        super().__init__(config)
        self.nav_interface = nav_interface
        self._position = (0.0, 0.0)
        self._heading = 0.0
        self._velocity = 0.0
        self._angular_velocity = 0.0
        self._battery_level = 1.0
        self._is_charging = False
        self._lidar_ranges = np.ones(64)

    def connect(self) -> bool:
        """Simulation is always connected."""
        self._is_connected = True
        logger.info("Simulation robot interface connected")
        return True

    def disconnect(self) -> None:
        """Disconnect simulation."""
        self._is_connected = False
        logger.info("Simulation robot interface disconnected")

    def get_state(self) -> RobotState:
        """Get simulated robot state."""
        return RobotState(
            position=self._position,
            heading=self._heading,
            velocity=self._velocity,
            angular_velocity=self._angular_velocity,
            battery_level=self._battery_level,
            is_charging=self._is_charging,
            lidar_ranges=self._lidar_ranges.copy()
        )

    def send_navigation_goal(self, goal: NavigationGoal) -> bool:
        """Send navigation goal to simulation."""
        self._current_goal = goal
        return True

    def get_navigation_feedback(self) -> NavigationFeedback:
        """Get navigation feedback from simulation."""
        if self.nav_interface is None or self._current_goal is None:
            return NavigationFeedback(
                success=False,
                time_elapsed=0.0,
                distance_traveled=0.0,
                final_position=self._position,
                error_code=-1,
                error_message="No navigation interface or goal"
            )

        # Execute navigation in simulation
        result = self.nav_interface.navigate_to_goal(
            self._position,
            self._current_goal.position
        )

        # Update simulated position
        if result.success:
            self._position = self._current_goal.position

        return NavigationFeedback(
            success=result.success,
            time_elapsed=result.time,
            distance_traveled=result.time * 1.0,  # Approximate
            final_position=self._position,
            error_code=0 if result.success else -3,
            error_message="" if result.success else "Navigation failed"
        )

    def cancel_navigation(self) -> bool:
        """Cancel navigation in simulation."""
        self._current_goal = None
        return True

    def emergency_stop(self) -> bool:
        """Emergency stop in simulation."""
        self._velocity = 0.0
        self._angular_velocity = 0.0
        self._current_goal = None
        return True

    def get_path_to_goal(
        self,
        goal: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """Get path from simulation navigator."""
        if self.nav_interface is None:
            return [self._position, goal]
        return self.nav_interface.plan_path(self._position, goal)

    def get_eta(self, goal: Tuple[float, float]) -> float:
        """Get ETA from simulation navigator."""
        if self.nav_interface is None:
            distance = np.sqrt(
                (goal[0] - self._position[0])**2 +
                (goal[1] - self._position[1])**2
            )
            return distance / 1.0  # 1 m/s default
        return self.nav_interface.get_eta(self._position, goal)

    def set_state(
        self,
        position: Tuple[float, float] = None,
        heading: float = None,
        battery_level: float = None,
        lidar_ranges: np.ndarray = None
    ):
        """Set simulation state (for testing)."""
        if position is not None:
            self._position = position
        if heading is not None:
            self._heading = heading
        if battery_level is not None:
            self._battery_level = battery_level
        if lidar_ranges is not None:
            self._lidar_ranges = lidar_ranges


def create_robot_interface(
    mode: str = "simulation",
    config: DeploymentConfig = None,
    **kwargs
) -> RobotInterface:
    """
    Factory function to create appropriate robot interface.

    Args:
        mode: Deployment mode ("simulation", "gazebo", "real")
        config: Deployment configuration
        **kwargs: Additional arguments passed to interface

    Returns:
        Appropriate RobotInterface instance

    Example:
        # Simulation
        robot = create_robot_interface("simulation")

        # Real robot
        robot = create_robot_interface("real", config=RealRobotConfig())
    """
    if mode == "simulation":
        if config is None:
            config = SimulationConfig()
        return SimulationRobotInterface(config, **kwargs)

    elif mode == "gazebo":
        # Import Gazebo interface (requires ROS2)
        try:
            from ..gazebo.gazebo_interface import Go2RobotInterface as GazeboRobotInterface
            if config is None:
                from .config import GazeboConfig
                config = GazeboConfig()
            return GazeboRobotInterface(config, **kwargs)
        except ImportError:
            raise ImportError(
                "Gazebo interface requires ROS2. "
                "Install with: pip install -e .[ros2]"
            )

    elif mode == "real":
        # Import Go2 interface (requires ROS2)
        try:
            from ..gazebo.gazebo_interface import Go2RobotInterface
            if config is None:
                config = RealRobotConfig()
            return Go2RobotInterface(config, **kwargs)
        except ImportError:
            raise ImportError(
                "Go2 interface requires ROS2. "
                "Install with: pip install -e .[ros2]"
            )

    else:
        raise ValueError(f"Unknown deployment mode: {mode}")
