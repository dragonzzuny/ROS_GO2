# RL Dispatch MVP

**Deep Reinforcement Learning for Autonomous Security Patrol Robot Dispatch System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

This project implements an intelligent dispatch and patrol route rescheduling system for autonomous security robots using **Proximal Policy Optimization (PPO)**. The system learns to balance:

- **Event Response**: Quickly dispatching to CCTV-detected industrial safety incidents
- **Patrol Coverage**: Maintaining regular visits to all patrol points
- **Safety & Efficiency**: Avoiding collisions while minimizing travel distance
- **Battery Management**: Returning to charging stations when needed

Designed for **Unitree Go2 quadruped robots** with ROS2 Nav2 integration and real-world deployment capability.

---

## ✨ Key Features

### 🎯 Core Capabilities
- **Multi-Map Generalization**: Trains on 6 diverse maps (100-150m²) for robust generalization
- **34 Industrial Safety Events**: Risk-weighted sampling (risk levels 1-9) based on Korean safety standards
- **10 Heuristic Strategies**: Diverse rescheduling candidates for efficient exploration
- **Nav2 Integration**: Pluggable navigation interface (simulated for training, real Nav2 for deployment)
- **Charging Station Management**: Autonomous battery monitoring and charging

### 🏗️ Technical Highlights
- **SMDP Formulation**: Semi-Markov Decision Process handling variable-time navigation steps
- **Candidate-Based Action Space**: `MultiDiscrete([2, 10])` - 2 modes × 10 strategies
- **Multi-Objective Rewards**: Balanced event response, patrol coverage, safety, efficiency
- **Production-Ready Code**: Modular, tested, type-annotated, documented

---

## 📦 Installation

### Prerequisites
```bash
# Ubuntu 22.04 recommended
python >= 3.10
```

### Quick Install
```bash
# Clone repository
git clone https://github.com/dragonzzuny/ROS_GO2.git
cd ROS_GO2/rl_dispatch_mvp

# Install basic dependencies
pip install gymnasium numpy pyyaml

# Install training dependencies (PyTorch, TensorBoard)
chmod +x install_training_deps.sh
./install_training_deps.sh
```

### Development Install
```bash
# Install with all development dependencies
pip install -e ".[dev]"
```

---

## 🎮 Quick Start

### 1. Run Test Scripts

Verify the system works correctly:

```bash
# Test industrial events and charging stations
python test_industrial_events.py

# Test Nav2 interface and 10 heuristics
python test_nav2_and_heuristics.py

# Test quick training (10K steps)
python test_quick_training.py
```

All tests should pass with ✅.

### 2. Start Training

**Quick test (100K steps, ~5-10 minutes):**
```bash
python scripts/train_multi_map.py --total-timesteps 100000 --seed 42
```

**Full training (5M steps, ~3-5 hours):**
```bash
python scripts/train_multi_map.py \
    --total-timesteps 5000000 \
    --seed 42 \
    --log-interval 10 \
    --save-interval 100
```

**With curriculum learning:**
```bash
python scripts/train_multi_map.py \
    --total-timesteps 5000000 \
    --map-mode curriculum \
    --cuda
```

### 3. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir runs

# Open browser at http://localhost:6006
```

You'll see:
- Episode returns per map
- Event success rates
- Patrol coverage ratios
- Policy/value losses
- Learning rate schedule

---

## 📊 Training Results

Expected performance after 5M steps:

| Map | Size | Event Success | Patrol Coverage | Avg Return |
|-----|------|---------------|-----------------|------------|
| Large Square | 100×100m | ~90% | ~85% | -3000 |
| Corridor | 120×30m | ~92% | ~90% | -2500 |
| L-Shaped | 80×80m | ~91% | ~88% | -2800 |
| Office Building | 90×70m | ~89% | ~82% | -4000 |
| Campus | 150×120m | ~85% | ~75% | -8000 |
| Warehouse | 140×100m | ~88% | ~80% | -5000 |

*(Initial random policy: ~85% success, ~10% coverage, -15000 return)*

---

## 🏗️ Project Structure

```
rl_dispatch_mvp/
├── src/rl_dispatch/
│   ├── core/                   # Core types and configurations
│   │   ├── types.py            # RobotState, PatrolPoint, Event, State
│   │   ├── types_extended.py   # Extended Event with risk_level
│   │   ├── event_types.py      # 34 industrial safety events
│   │   └── config.py           # EnvConfig, RewardConfig, TrainingConfig
│   │
│   ├── env/                    # Gymnasium environments
│   │   ├── patrol_env.py       # Single-map patrol environment
│   │   └── multi_map_env.py    # Multi-map wrapper for generalization
│   │
│   ├── algorithms/             # Reinforcement learning
│   │   ├── ppo.py              # PPO agent implementation
│   │   ├── networks.py         # Actor-critic neural networks
│   │   ├── buffer.py           # Rollout buffer with GAE
│   │   └── baselines.py        # Heuristic baseline policies
│   │
│   ├── planning/               # Route planning
│   │   └── candidate_generator.py  # 10 heuristic strategies
│   │
│   ├── navigation/             # Nav2 integration
│   │   └── nav2_interface.py   # SimulatedNav2 / RealNav2
│   │
│   ├── rewards/                # Reward calculation
│   │   └── reward_calculator.py
│   │
│   └── utils/                  # Utilities
│       ├── observation.py      # State → Observation encoding
│       ├── math.py             # Geometric utilities
│       └── visualization.py    # Rendering and plots
│
├── scripts/
│   └── train_multi_map.py      # Multi-map PPO training
│
├── configs/                    # Map configurations (6 maps)
│   ├── map_large_square.yaml
│   ├── map_corridor.yaml
│   ├── map_l_shaped.yaml
│   ├── map_office_building.yaml
│   ├── map_campus.yaml
│   └── map_warehouse.yaml
│
├── docs/                       # Documentation
│   ├── heuristic_method.md     # 10 heuristic strategy descriptions
│   └── R&D_Plan_Complete_v4.md # Complete technical specification
│
└── tests/                      # Test scripts
    ├── test_industrial_events.py
    ├── test_nav2_and_heuristics.py
    └── test_quick_training.py
```

---

## 🧠 System Architecture

### State Space (77D)

| Component | Dimension | Description |
|-----------|-----------|-------------|
| **Robot State** | 7D | x, y, heading, velocity, angular_velocity, battery, goal_idx |
| **LiDAR** | 64D | 360° obstacle detection (64 channels) |
| **Event** | 5D | exists, risk_level (1-9), confidence, elapsed_time, distance |
| **Patrol** | 1D | Max coverage gap across all points |

**Observation Normalization:**
- Positions: Normalized by map size
- Distances: Log-scaled for better gradient flow
- Time gaps: Exponentially decayed to emphasize urgency

### Action Space

**MultiDiscrete([2, 10]):**

1. **Mode** (Binary):
   - `0`: PATROL - Continue current patrol route
   - `1`: DISPATCH - Respond to event (only valid if event exists)

2. **Replan Strategy** (Categorical, 10 options):
   - `keep_order`: Maintain current route order
   - `nearest_first`: Greedy nearest neighbor
   - `most_overdue_first`: Prioritize longest-waiting points
   - `overdue_eta_balance`: Balance overdue time and travel time
   - `risk_weighted`: Weight by point importance
   - `balanced_coverage`: Minimize max coverage gap
   - `overdue_threshold_first`: Points exceeding threshold first
   - `windowed_replan`: Replan only first H waypoints
   - `minimal_deviation_insert`: Insert with minimal deviation
   - `shortest_eta_first`: Sort by ETA (Nav2-aware)

### Reward Function

**Total Reward**: `R = R_evt + R_pat + R_safe + R_eff`

| Component | Formula | Weight | Description |
|-----------|---------|--------|-------------|
| **Event Response** | `-α·delay + β·success` | α=10, β=100 | Penalize delays, reward successes |
| **Patrol Coverage** | `-γ·Σ(gap_i²)` | γ=1 | Quadratic penalty for gaps |
| **Safety** | `-δ·collision - ε·failure` | δ=200, ε=50 | Heavily penalize unsafe actions |
| **Efficiency** | `-ζ·distance` | ζ=0.1 | Mild penalty for travel |

---

## 🔧 Configuration

### Map Configuration Example

```yaml
# configs/map_large_square.yaml
env:
  # Map dimensions
  map_width: 100.0
  map_height: 100.0

  # Patrol points (x, y)
  patrol_points:
    - [10.0, 10.0]
    - [90.0, 10.0]
    - [90.0, 90.0]
    # ... 12 points total

  # Charging station location
  charging_station_position: [5.0, 5.0]

  # Event generation
  event_rate_per_hour: 2.0
  max_episode_steps: 500

  # Robot parameters
  robot_max_velocity: 1.0
  robot_max_angular_velocity: 1.57
  battery_drain_rate: 0.001

  # Heuristic candidates
  num_candidates: 10
  candidate_strategies:
    - keep_order
    - nearest_first
    - most_overdue_first
    # ... 10 strategies total
```

### Training Configuration

```yaml
# Training hyperparameters
training:
  total_timesteps: 5000000
  learning_rate: 3e-4
  num_steps: 2048        # Steps per update
  num_epochs: 10         # PPO epochs
  batch_size: 256

  # PPO parameters
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5

  # Learning rate schedule
  anneal_lr: true
```

---

## 🎯 34 Industrial Safety Events

Based on Korean KOSHA/MOEL safety standards:

| Category | Events | Risk Levels |
|----------|--------|-------------|
| **Fire/Explosion** | 화재감지, 연기감지, 전기화재위험, 가스누출, 폭발위험물질 | 7-9 (Critical) |
| **Chemical** | 화학물질누출, 유해가스검출, 부식성물질유출 | 6-9 |
| **Mechanical** | 기계고장, 이상진동, 과열, 압력이상 | 4-8 |
| **Electrical** | 전기시설이상, 정전, 누전 | 4-8 |
| **Structural** | 구조물균열, 천장누수, 바닥침하 | 5-8 |
| **Environment** | 소음초과, 분진발생, 조명이상, 환기불량 | 2-5 |
| **Abnormal Behavior** | 무단침입, 낙상사고, 응급상황 | 6-9 |

**Risk-Weighted Sampling**: `P(risk=r) ∝ 1/r²` (high-risk events are rare but critical)

---

## 📈 Performance Metrics

### Episode Metrics

- **Event Success Rate**: % of events responded within timeout
- **Patrol Coverage Ratio**: % of time all points within threshold
- **Average Response Time**: Mean delay for event dispatch
- **Battery Efficiency**: % of time above critical battery
- **Collision Rate**: Collisions per 1000 steps

### Training Metrics

- **Episode Return**: Cumulative reward (target: > -3000)
- **Policy Loss**: PPO clipped objective
- **Value Loss**: Critic MSE
- **Entropy**: Policy exploration (target: ~2.0-3.0)
- **KL Divergence**: Policy change (target: < 0.02)
- **Clip Fraction**: % of updates clipped (target: 5-15%)
- **Explained Variance**: Value function fit (target: > 0.8)

---

## 🚀 Deployment (Unitree Go2)

### Prerequisites

```bash
# Install ROS2 Humble
sudo apt install ros-humble-desktop

# Install Nav2
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Install Unitree Go2 SDK
git clone https://github.com/unitreerobotics/unitree_ros2.git
cd unitree_ros2 && colcon build
```

### Deploy Trained Policy

```bash
# 1. Load trained model
python scripts/deploy_go2.py \
    --model runs/multi_map_ppo/20251229-230854/checkpoints/final.pth \
    --map configs/real_building.yaml

# 2. Launch ROS2 nodes
ros2 launch rl_dispatch go2_patrol.launch.py

# 3. Monitor via RViz
rviz2 -d config/patrol.rviz
```

The deployment script:
- Loads the trained PPO policy
- Connects to real Nav2 for navigation
- Subscribes to CCTV event detections
- Publishes patrol route visualizations
- Logs performance metrics

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python test_industrial_events.py

# Run with coverage
pytest --cov=rl_dispatch tests/
```

---

## 📚 Documentation

- **[Heuristic Methods](docs/heuristic_method.md)**: Detailed description of 10 planning strategies
- **[R&D Plan](docs/R&D_Plan_Complete_v4.md)**: Complete technical specification
- **[Development Guide](docs/development_guide.md)**: AI development guidelines

---

## 🛣️ Roadmap

### ✅ Completed
- [x] Core environment implementation
- [x] 34 industrial safety events system
- [x] 10 heuristic strategies
- [x] Nav2 interface abstraction
- [x] Multi-map training infrastructure
- [x] PPO agent implementation
- [x] Comprehensive test suite
- [x] Training scripts and configs

### 🚧 In Progress
- [ ] Full 5M-step training run
- [ ] Hyperparameter tuning
- [ ] Baseline comparisons

### 📋 Planned
- [ ] Gazebo simulation validation
- [ ] Real Unitree Go2 deployment
- [ ] Multi-robot coordination
- [ ] Hierarchical RL for long-horizon planning

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_dispatch_mvp_2025,
  title={RL Dispatch MVP: Deep RL for Autonomous Security Patrol Robots},
  author={YJP},
  year={2025},
  url={https://github.com/dragonzzuny/ROS_GO2},
  note={Multi-map PPO training with 34 industrial safety events}
}
```

---

## 📧 Contact

- **Author**: YJP
- **Repository**: [https://github.com/dragonzzuny/ROS_GO2](https://github.com/dragonzzuny/ROS_GO2)
- **Issues**: [GitHub Issues](https://github.com/dragonzzuny/ROS_GO2/issues)

---

## 🙏 Acknowledgments

- **Unitree Robotics** for Go2 quadruped robot platform
- **OpenAI** for PPO algorithm
- **ROS2 Navigation** team for Nav2 stack
- **KOSHA/MOEL** for Korean industrial safety standards

---

**Built with ❤️ for autonomous security systems**
