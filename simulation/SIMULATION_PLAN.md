# Go2 Simulation Integration Plan

## Overview

This document describes the implementation plan for simulating and validating 
trained RL policies on Unitree Go2 robot in Gazebo and Isaac Sim.

## Folder Structure

```
simulation/
├── SIMULATION_PLAN.md      # This document
├── __init__.py             # Main module init
├── common/                 # Shared interfaces
│   ├── __init__.py
│   ├── robot_interface.py  # Base RobotInterface
│   └── config.py           # Configuration classes
├── gazebo/                 # Gazebo ROS2 simulation
│   ├── __init__.py
│   └── gazebo_interface.py # Gazebo interface (ROS2 + Nav2)
├── isaac_sim/              # NVIDIA Isaac Sim
│   ├── __init__.py
│   └── isaac_interface.py  # Isaac Sim interface
├── launch/                 # ROS2 launch files
└── config/                 # Simulation configs
```

## External Dependencies

### Gazebo (ROS2 Humble)

| Package | Description | URL |
|---------|-------------|-----|
| unitree_ros2 | Official Unitree ROS2 SDK | github.com/unitreerobotics/unitree_ros2 |
| unitree-go2-ros2 | Go2 URDF + CHAMP controller | github.com/anujjain-dev/unitree-go2-ros2 |
| go2_ros2_sdk | Unofficial Go2 ROS2 SDK | github.com/abizovnuralem/go2_ros2_sdk |

### Isaac Sim

| Package | Description | URL |
|---------|-------------|-----|
| unitree_sim_isaaclab | Official Unitree Isaac Lab | github.com/unitreerobotics/unitree_sim_isaaclab |
| go2_omniverse | Go2 Isaac Lab integration | github.com/abizovnuralem/go2_omniverse |
| isaac-go2-ros2 | Isaac Sim + ROS2 bridge | github.com/Zhefan-Xu/isaac-go2-ros2 |

## Installation

### Gazebo Setup

```bash
# ROS2 Humble
sudo apt install ros-humble-desktop ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Go2 packages
cd ~/ros2_ws/src
git clone https://github.com/anujjain-dev/unitree-go2-ros2.git
cd ~/ros2_ws && colcon build
```

### Isaac Sim Setup

```bash
# Install Isaac Sim 4.5+ from NVIDIA
# Install Isaac Lab 2.1.0+
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install

# Go2 package
git clone https://github.com/Zhefan-Xu/isaac-go2-ros2.git
```

## Usage

### Gazebo Simulation

```python
from simulation.gazebo import GazeboInterface

# Initialize (requires ROS2 running)
sim = GazeboInterface()
sim.initialize()

# Spawn robot
sim.spawn_robot(position=(0, 0, 0.5))

# Run steps
for _ in range(1000):
    state = sim.step()
    print(f"Position: {state.position}")
```

### Isaac Sim Simulation

```python
from simulation.isaac_sim import IsaacInterface

# Initialize
sim = IsaacInterface(headless=True)
sim.initialize()
sim.load_environment("environments/warehouse.usd")
sim.spawn_robot(position=(0, 0, 0.5))

# Run steps
for _ in range(1000):
    sim.step()
    state = sim.get_robot_state()
    print(f"Position: {state['position']}")
```

## Architecture

```
                    RL Policy (PPO)
                          |
                    PolicyRunner
                          |
            +-------------+-------------+
            |             |             |
      GazeboInterface  IsaacInterface  Go2Interface
            |             |             |
         Gazebo      Isaac Sim       Real Go2
        (ROS2)        (USD)          (ROS2)
```

## Next Steps

1. [ ] Test Gazebo interface with unitree-go2-ros2
2. [ ] Test Isaac Sim interface with go2_omniverse
3. [ ] Create validation pipeline for trained policies
4. [ ] Add visualization and metrics collection
5. [ ] Create launch files for easy deployment

---

Last updated: 2026-01-03
