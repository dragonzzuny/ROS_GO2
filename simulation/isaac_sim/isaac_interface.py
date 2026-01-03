"""
Isaac Sim Interface for Go2 Robot.

External Dependencies:
- unitree_sim_isaaclab: https://github.com/unitreerobotics/unitree_sim_isaaclab
- go2_omniverse: https://github.com/abizovnuralem/go2_omniverse  
- isaac-go2-ros2: https://github.com/Zhefan-Xu/isaac-go2-ros2

Requirements:
- Isaac Sim 4.5+
- Isaac Lab 2.1.0+
"""

import logging
from typing import Tuple, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

try:
    from isaacsim import SimulationApp
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False


class IsaacInterface:
    """Isaac Sim interface for Go2 robot simulation."""
    
    def __init__(self, headless: bool = False, gpu_id: int = 0):
        if not ISAAC_AVAILABLE:
            raise ImportError("Isaac Sim not available. Install from NVIDIA.")
        
        self._sim_app = None
        self._world = None
        self._robot = None
        self._headless = headless
        self._gpu_id = gpu_id
        
        # State
        self._position = (0.0, 0.0, 0.0)
        self._orientation = (0.0, 0.0, 0.0, 1.0)
        self._velocity = (0.0, 0.0, 0.0)
        self._lidar_ranges = np.ones(64)
        self._sim_time = 0.0
    
    def initialize(self) -> bool:
        """Initialize Isaac Sim."""
        try:
            self._sim_app = SimulationApp({"headless": self._headless})
            from omni.isaac.core import World
            self._world = World(physics_dt=0.005, rendering_dt=0.033)
            return True
        except Exception as e:
            logger.error(f"Init failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown Isaac Sim."""
        if self._sim_app:
            self._sim_app.close()
    
    def load_environment(self, usd_path: str) -> bool:
        """Load USD environment."""
        try:
            from omni.isaac.core.utils.stage import open_stage
            open_stage(usd_path)
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False
    
    def spawn_robot(self, position: Tuple[float, float, float]) -> bool:
        """Spawn Go2 robot."""
        try:
            from omni.isaac.core.robots import Robot
            self._robot = Robot(
                prim_path="/World/Go2",
                name="go2",
                position=position
            )
            self._world.scene.add(self._robot)
            return True
        except Exception as e:
            logger.error(f"Spawn failed: {e}")
            return False
    
    def step(self, num_steps: int = 1):
        """Step simulation."""
        for _ in range(num_steps):
            self._world.step(render=not self._headless)
            self._sim_time += 0.005
            self._update_state()
    
    def get_robot_state(self) -> Dict:
        """Get robot state."""
        return {
            "position": self._position,
            "orientation": self._orientation,
            "velocity": self._velocity,
            "lidar_ranges": self._lidar_ranges.copy(),
            "sim_time": self._sim_time
        }
    
    def _update_state(self):
        """Update state from simulation."""
        if self._robot:
            pos, ori = self._robot.get_world_pose()
            self._position = tuple(pos) if pos is not None else (0, 0, 0)
            self._orientation = tuple(ori) if ori is not None else (0, 0, 0, 1)

