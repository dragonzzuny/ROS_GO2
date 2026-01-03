"""
Unitree Go2 Robot Interface with ROS2 Nav2.

This module provides the real robot interface for deploying trained
policies to a Unitree Go2 quadruped robot via ROS2 and Nav2.

Requirements:
    - ROS2 Humble or later
    - Nav2 stack
    - unitree_ros2 package
    - go2_description package

Usage:
    from rl_dispatch.deployment import create_robot_interface, RealRobotConfig

    config = RealRobotConfig(robot_ip="192.168.123.161")
    robot = create_robot_interface("real", config=config)
    robot.connect()

    # Get state
    state = robot.get_state()
    print(f"Position: {state.position}, Battery: {state.battery_level}")

    # Navigate
    result = robot.navigate_to((5.0, 3.0))
    print(f"Navigation success: {result.success}")
"""

import logging
import time
from typing import Tuple, Optional, List
import numpy as np

from .robot_interface import (
    RobotInterface,
    RobotState,
    NavigationGoal,
    NavigationFeedback,
)
from .config import RealRobotConfig


logger = logging.getLogger(__name__)


# ROS2 imports (optional - only needed for real deployment)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry, Path
    from sensor_msgs.msg import LaserScan, BatteryState, Imu
    from nav2_msgs.action import NavigateToPose, ComputePathToPose
    from nav2_msgs.srv import GetCostmap

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.warning("ROS2 not available. Go2Interface will not work.")


class Go2RobotInterface(RobotInterface):
    """
    Real Unitree Go2 robot interface using ROS2 and Nav2.

    This interface:
    - Subscribes to robot state topics (odom, battery, lidar)
    - Uses Nav2 action clients for navigation
    - Implements safety constraints and watchdog

    Attributes:
        config: RealRobotConfig with robot settings
        node: ROS2 node for communication
        nav_client: Nav2 NavigateToPose action client
    """

    def __init__(self, config: RealRobotConfig):
        """
        Initialize Go2 interface.

        Args:
            config: Real robot configuration
        """
        super().__init__(config)

        if not ROS2_AVAILABLE:
            raise ImportError(
                "ROS2 is required for Go2RobotInterface. "
                "Please install ROS2 and source the workspace."
            )

        self.config: RealRobotConfig = config
        self.node: Optional[Node] = None

        # State variables
        self._position = (0.0, 0.0)
        self._heading = 0.0
        self._velocity = 0.0
        self._angular_velocity = 0.0
        self._battery_level = 1.0
        self._is_charging = False
        self._lidar_ranges = np.ones(64)

        # Navigation state
        self._nav_goal_handle = None
        self._nav_result = None
        self._nav_feedback = None

        # Safety
        self._last_heartbeat = 0.0
        self._emergency_stopped = False

    def connect(self) -> bool:
        """
        Connect to Go2 robot via ROS2.

        Initializes ROS2 node, subscribers, and action clients.

        Returns:
            True if connection successful
        """
        try:
            # Initialize ROS2 if not already done
            if not rclpy.ok():
                rclpy.init()

            # Create node
            self.node = rclpy.create_node('rl_dispatch_go2')
            logger.info("Created ROS2 node: rl_dispatch_go2")

            # QoS for sensor data
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=10
            )

            # Subscribe to odometry
            self._odom_sub = self.node.create_subscription(
                Odometry,
                '/odom',
                self._odom_callback,
                10
            )

            # Subscribe to battery state
            self._battery_sub = self.node.create_subscription(
                BatteryState,
                self.config.battery_topic,
                self._battery_callback,
                10
            )

            # Subscribe to LiDAR
            self._scan_sub = self.node.create_subscription(
                LaserScan,
                '/scan',
                self._scan_callback,
                sensor_qos
            )

            # Nav2 action client
            self._nav_client = ActionClient(
                self.node,
                NavigateToPose,
                self.config.navigate_to_pose_action
            )

            # Path planning client
            self._path_client = ActionClient(
                self.node,
                ComputePathToPose,
                self.config.get_path_service
            )

            # Velocity publisher (for emergency stop)
            self._cmd_vel_pub = self.node.create_publisher(
                Twist,
                '/cmd_vel',
                10
            )

            # Wait for Nav2 to be ready
            logger.info("Waiting for Nav2 action server...")
            if not self._nav_client.wait_for_server(timeout_sec=10.0):
                logger.error("Nav2 action server not available")
                return False

            logger.info("Connected to Nav2 action server")
            self._is_connected = True
            self._last_heartbeat = time.time()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Go2: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Go2 robot."""
        try:
            # Stop robot
            self.emergency_stop()

            # Cancel any active navigation
            if self._nav_goal_handle is not None:
                self.cancel_navigation()

            # Destroy node
            if self.node is not None:
                self.node.destroy_node()
                self.node = None

            self._is_connected = False
            logger.info("Disconnected from Go2")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def get_state(self) -> RobotState:
        """
        Get current robot state from ROS2 topics.

        Spins node once to get latest messages.

        Returns:
            Current RobotState
        """
        if self.node is not None:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        # Check watchdog
        if time.time() - self._last_heartbeat > self.config.watchdog_timeout:
            logger.warning("Watchdog timeout - no recent state updates")

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
        """
        Send navigation goal to Nav2.

        Args:
            goal: Navigation goal specification

        Returns:
            True if goal accepted by Nav2
        """
        if not self._is_connected or self._emergency_stopped:
            logger.warning("Cannot navigate: not connected or emergency stopped")
            return False

        # Check battery
        if self._battery_level < self.config.critical_battery_threshold:
            logger.error("Critical battery - navigation disabled")
            return False

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal.position[0]
        goal_msg.pose.pose.position.y = goal.position[1]

        if goal.orientation is not None:
            # Convert heading to quaternion
            goal_msg.pose.pose.orientation.z = np.sin(goal.orientation / 2)
            goal_msg.pose.pose.orientation.w = np.cos(goal.orientation / 2)
        else:
            goal_msg.pose.pose.orientation.w = 1.0

        # Send goal
        self._current_goal = goal
        self._nav_result = None

        logger.info(f"Sending navigation goal: {goal.position}")
        send_goal_future = self._nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav_feedback_callback
        )
        send_goal_future.add_done_callback(self._goal_response_callback)

        return True

    def get_navigation_feedback(self) -> NavigationFeedback:
        """
        Get current navigation status.

        Returns:
            NavigationFeedback with current status
        """
        if self.node is not None:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        if self._nav_result is not None:
            return self._nav_result

        if self._current_goal is None:
            return NavigationFeedback(
                success=False,
                time_elapsed=0.0,
                distance_traveled=0.0,
                final_position=self._position,
                error_code=-1,
                error_message="No active goal"
            )

        # Navigation in progress
        return NavigationFeedback(
            success=False,
            time_elapsed=0.0,  # Would need to track start time
            distance_traveled=0.0,
            final_position=self._position,
            error_code=0,
            error_message="Navigation in progress"
        )

    def cancel_navigation(self) -> bool:
        """Cancel current navigation goal."""
        if self._nav_goal_handle is not None:
            logger.info("Canceling navigation")
            cancel_future = self._nav_goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self.node, cancel_future, timeout_sec=2.0)
            self._nav_goal_handle = None
            self._current_goal = None
            return True
        return False

    def emergency_stop(self) -> bool:
        """
        Execute emergency stop.

        Publishes zero velocity and sets emergency flag.
        """
        logger.warning("EMERGENCY STOP activated")
        self._emergency_stopped = True

        # Cancel navigation
        self.cancel_navigation()

        # Publish zero velocity
        if self._cmd_vel_pub is not None:
            stop_msg = Twist()  # All zeros
            for _ in range(5):  # Send multiple times
                self._cmd_vel_pub.publish(stop_msg)
                time.sleep(0.02)

        return True

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop flag."""
        logger.info("Emergency stop reset")
        self._emergency_stopped = False

    def get_path_to_goal(
        self,
        goal: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Request path from Nav2 planner.

        Args:
            goal: Target position

        Returns:
            List of waypoints or None
        """
        if not self._is_connected:
            return None

        # Create path request
        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal.header.frame_id = 'map'
        goal_msg.goal.pose.position.x = goal[0]
        goal_msg.goal.pose.position.y = goal[1]
        goal_msg.goal.pose.orientation.w = 1.0

        goal_msg.start.header.frame_id = 'map'
        goal_msg.start.pose.position.x = self._position[0]
        goal_msg.start.pose.position.y = self._position[1]
        goal_msg.start.pose.orientation.w = 1.0

        # Send request
        future = self._path_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)

        if future.result() is None:
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            return None

        # Get result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=5.0)

        if result_future.result() is None:
            return None

        result = result_future.result().result
        path = result.path

        # Convert to list of tuples
        waypoints = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in path.poses
        ]

        return waypoints if waypoints else None

    def get_eta(self, goal: Tuple[float, float]) -> float:
        """
        Estimate time to reach goal.

        Uses path length and average velocity.
        """
        path = self.get_path_to_goal(goal)
        if path is None:
            return np.inf

        # Calculate path length
        total_distance = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            total_distance += np.sqrt(dx*dx + dy*dy)

        # Estimate time (conservative velocity)
        avg_velocity = self.config.max_linear_velocity * 0.6
        return total_distance / avg_velocity if avg_velocity > 0 else np.inf

    # ==================== Callbacks ====================

    def _odom_callback(self, msg: 'Odometry'):
        """Process odometry message."""
        self._position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

        # Extract heading from quaternion
        q = msg.pose.pose.orientation
        self._heading = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        self._velocity = np.sqrt(
            msg.twist.twist.linear.x**2 +
            msg.twist.twist.linear.y**2
        )
        self._angular_velocity = msg.twist.twist.angular.z

        self._last_heartbeat = time.time()

    def _battery_callback(self, msg: 'BatteryState'):
        """Process battery state message."""
        self._battery_level = msg.percentage / 100.0
        self._is_charging = msg.power_supply_status == 1  # CHARGING

        # Low battery warning
        if self._battery_level < self.config.low_battery_threshold:
            logger.warning(f"Low battery: {self._battery_level*100:.1f}%")

            if (self.config.auto_return_on_low_battery and
                self._battery_level < self.config.critical_battery_threshold):
                logger.error("Critical battery - returning to charging station")
                # Would trigger return to charging station

    def _scan_callback(self, msg: 'LaserScan'):
        """Process LiDAR scan message."""
        ranges = np.array(msg.ranges)

        # Resample to 64 rays if needed
        if len(ranges) != 64:
            indices = np.linspace(0, len(ranges)-1, 64, dtype=int)
            ranges = ranges[indices]

        # Normalize and clip
        max_range = msg.range_max
        ranges = np.clip(ranges / max_range, 0.0, 1.0)
        ranges = np.nan_to_num(ranges, nan=1.0, posinf=1.0)

        self._lidar_ranges = ranges.astype(np.float32)

        # Safety check
        min_distance = np.min(ranges) * max_range
        if min_distance < self.config.emergency_stop_distance:
            logger.warning(f"Obstacle too close: {min_distance:.2f}m")
            if self.config.safety_enabled and not self._emergency_stopped:
                self.emergency_stop()

    def _goal_response_callback(self, future):
        """Handle goal response from Nav2."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            logger.warning("Navigation goal rejected")
            self._nav_result = NavigationFeedback(
                success=False,
                time_elapsed=0.0,
                distance_traveled=0.0,
                final_position=self._position,
                error_code=-4,
                error_message="Goal rejected by Nav2"
            )
            return

        logger.info("Navigation goal accepted")
        self._nav_goal_handle = goal_handle

        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result_callback)

    def _nav_result_callback(self, future):
        """Handle navigation result from Nav2."""
        result = future.result().result

        success = result.result == 0  # NavigationResult.SUCCEEDED
        self._nav_result = NavigationFeedback(
            success=success,
            time_elapsed=0.0,  # Would need to track
            distance_traveled=0.0,
            final_position=self._position,
            error_code=0 if success else result.result,
            error_message="" if success else f"Nav2 error: {result.result}"
        )

        self._nav_goal_handle = None
        self._current_goal = None
        logger.info(f"Navigation complete: success={success}")

    def _nav_feedback_callback(self, feedback_msg):
        """Handle navigation feedback from Nav2."""
        feedback = feedback_msg.feedback
        # Could track progress here
        pass
