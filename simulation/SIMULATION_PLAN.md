# Go2 Simulation Integration Plan

## Overview

This module provides simulation interfaces for validating trained RL policies
on the Unitree Go2 robot in Gazebo and Isaac Sim before real-world deployment.

## Folder Structure

```
simulation/
├── SIMULATION_PLAN.md      # This document
├── requirements.txt        # Python dependencies
├── __init__.py
├── common/                 # Shared interfaces
│   ├── __init__.py
│   ├── robot_interface.py  # Base RobotInterface
│   └── config.py           # Configuration classes
├── gazebo/                 # Gazebo ROS2 simulation
│   ├── __init__.py
│   └── gazebo_interface.py # Gazebo + Nav2 interface
├── isaac_sim/              # NVIDIA Isaac Sim
│   ├── __init__.py
│   └── isaac_interface.py  # Isaac Sim interface
├── launch/                 # ROS2 launch files
└── config/                 # Simulation configs
```

## External Dependencies (Cloned in external/)

### 1. unitree-go2-ros2
- **URL**: https://github.com/anujjain-dev/unitree-go2-ros2
- **Purpose**: Go2 URDF model + CHAMP quadruped controller
- **Features**:
  - Go2 robot description (URDF/meshes)
  - CHAMP controller for quadruped locomotion
  - Teleoperation support
  - LiDAR (2D/3D), IMU integration

### 2. go2_ros2_sdk
- **URL**: https://github.com/abizovnuralem/go2_ros2_sdk
- **Purpose**: Unofficial ROS2 SDK for real Go2 robot
- **Features**:
  - WebRTC (Wi-Fi) and CycloneDDS (Ethernet) protocols
  - Real-time robot state access
  - Motion control interface
  - LiDAR and camera streaming

### 3. isaac-go2-ros2
- **URL**: https://github.com/Zhefan-Xu/isaac-go2-ros2
- **Purpose**: Isaac Sim simulation with ROS2 bridge
- **Features**:
  - Pre-trained RL locomotion policy
  - RTX LiDAR simulation
  - RGB/Depth/Semantic camera
  - Multiple environments (warehouse, obstacles)
- **ROS2 Topics**:
  - `/unitree_go2/cmd_vel` - Velocity commands
  - `/unitree_go2/odom` - Odometry
  - `/unitree_go2/lidar/point_cloud` - LiDAR data
  - `/unitree_go2/front_cam/*` - Camera data

## Installation

### Prerequisites

```bash
# Ubuntu 22.04
sudo apt update

# ROS2 Humble
sudo apt install ros-humble-desktop
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install ros-humble-gazebo-ros-pkgs

# Python dependencies
pip install -r simulation/requirements.txt
```

### External Packages (Already Cloned)

```bash
cd rl_dispatch_mvp/external
ls  # unitree-go2-ros2, go2_ros2_sdk, isaac-go2-ros2
```

### Isaac Sim Setup

1. Download Isaac Sim 4.5+ from NVIDIA
2. Install Isaac Lab 2.1.0
3. Follow isaac-go2-ros2 README

## Usage

### Gazebo Simulation

```python
from simulation.gazebo import GazeboGo2Interface

# Connect to Gazebo (requires ROS2 + Gazebo running)
interface = GazeboGo2Interface()
interface.connect()

# Get robot state
state = interface.get_state()
print(f"Position: {state.position}")

# Navigate
interface.navigate_to((5.0, 5.0))
```

### Isaac Sim Simulation

```python
from simulation.isaac_sim import IsaacInterface, IsaacConfig

# Initialize
config = IsaacConfig(headless=True, env_name="warehouse")
interface = IsaacInterface(config)
interface.initialize()
interface.create_environment()
interface.spawn_robot((0, 0, 0.5))

# Run simulation
for _ in range(1000):
    interface.set_velocity_command(0.5, 0, 0)  # Forward
    state = interface.step()
    print(f"Position: {state['position']}")
```

### Policy Validation

```python
from rl_dispatch.deployment import PolicyRunner
from simulation.gazebo import GazeboGo2Interface

# Load trained policy
runner = PolicyRunner(
    model_path="checkpoints/best_model.pth",
    mode="simulation"
)

# Use Gazebo interface
interface = GazeboGo2Interface()
interface.connect()

# Validate policy
for episode in range(100):
    state = interface.reset()
    for step in range(400):
        action = runner.select_action(state)
        state = interface.step(action)
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
    (ROS2 + Nav2)    (Isaac Lab)     (ROS2 Real)
          |             |             |
       Gazebo      Isaac Sim       Real Go2
```

## ROS2 Topics Summary

| Topic | Type | Description |
|-------|------|-------------|
| /odom | Odometry | Robot odometry |
| /scan | LaserScan | 2D LiDAR |
| /cmd_vel | Twist | Velocity command |
| /navigate_to_pose | Action | Nav2 navigation |
| /battery_state | BatteryState | Battery level |

---

Last updated: 2026-01-03
