"""
RL Dispatch Simulation Module.

Provides interfaces for simulating Go2 robot in:
- Gazebo (ROS2 Humble + Nav2)
- Isaac Sim (NVIDIA Isaac Lab)

Usage:
    # Gazebo
    from simulation.gazebo import GazeboInterface
    sim = GazeboInterface()
    
    # Isaac Sim
    from simulation.isaac_sim import IsaacInterface
    sim = IsaacInterface()
"""

__all__ = ["gazebo", "isaac_sim", "common"]
