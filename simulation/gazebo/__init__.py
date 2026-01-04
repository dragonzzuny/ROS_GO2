"""
Gazebo simulation interface for Go2 robot.

Requires:
- ROS2 Humble
- Gazebo Fortress/Harmonic
- Nav2 stack
- unitree-go2-ros2 package

Usage:
    from simulation.gazebo import GazeboGo2Interface
    
    interface = GazeboGo2Interface()
    interface.connect()
    interface.spawn_robot((0, 0, 0.5))
"""

try:
    from .gazebo_interface import Go2RobotInterface as GazeboGo2Interface
    __all__ = ["GazeboGo2Interface"]
except ImportError:
    __all__ = []
