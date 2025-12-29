# RL Dispatch MVP

**Unified RL-based Dispatch and Rescheduling Policy for Autonomous Patrol Robots**

## Overview

This project implements a single reinforcement learning policy that intelligently handles both:
1. **Dispatch decisions**: Whether to respond to CCTV-detected events
2. **Route rescheduling**: How to reorganize patrol routes after event response

The system uses PPO (Proximal Policy Optimization) trained in simulation (Isaac Sim/Gazebo) and deployed on Unitree Go2 quadruped robots.

## Key Features

- **SMDP Formulation**: Semi-Markov Decision Process handling variable navigation times
- **Candidate-Based Action Space**: Efficient action space with 6 pre-computed rescheduling strategies
- **Multi-Objective Rewards**: Balances event response, patrol coverage, safety, and efficiency
- **Sim2Real Pipeline**: Validated from simulation through real hardware deployment
- **Professional Architecture**: Modular, tested, and production-ready code

## Installation

```bash
# Clone the repository
git clone https://github.com/yjp/rl_dispatch_mvp.git
cd rl_dispatch_mvp

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with ROS2 dependencies (for real robot deployment)
pip install -e ".[ros]"
```

## Quick Start

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --model models/best_model.pth --episodes 100
```

### Deployment

```bash
python scripts/deploy.py --robot go2 --model models/best_model.pth
```

## Project Structure

```
rl_dispatch_mvp/
├── src/
│   └── rl_dispatch/
│       ├── core/           # Core data structures and types
│       ├── env/            # Gymnasium environment
│       ├── algorithms/     # PPO and baseline policies
│       ├── networks/       # Neural network architectures
│       ├── nav/            # Nav2 integration
│       ├── rewards/        # Reward calculation
│       ├── planning/       # Candidate generation
│       └── utils/          # Utilities and helpers
├── tests/                  # Unit and integration tests
├── scripts/                # Training and evaluation scripts
├── configs/                # Configuration files
└── readme/                 # Comprehensive documentation

```

## Architecture

### State Space (77D)
- Robot state: position, heading, velocity, battery
- Perception: 64-channel LiDAR
- Event: existence, urgency, confidence, elapsed time
- Patrol: coverage gaps, next waypoint distance

### Action Space
- **Mode**: Binary (0=continue patrol, 1=dispatch to event)
- **Replan**: Categorical (6 rescheduling strategies)
  - Keep-Order, Nearest-First, Most-Overdue-First, Overdue-ETA Balance, Risk-Weighted, Balanced-Coverage

### Reward Components
1. **Event Response** (R^evt): Delay penalties + success bonuses
2. **Patrol Coverage** (R^pat): Cost of coverage gaps
3. **Safety** (R^safe): Collision and navigation failure penalties
4. **Efficiency** (R^eff): Path length penalties

## Documentation

- [Development Guide](readme/development_guide.md) - AI development guidelines and golden rules
- [Complete R&D Plan](readme/R&D_Plan_Complete_v4.md) - Full technical specification
- [Hardware Guide](readme/Unitree_GO2_PRO_Developer_Guide.md) - Unitree Go2 integration

## Development Roadmap

- [x] Architecture design and specifications
- [x] Core implementation (Phase 1)
- [ ] Simulation environment setup
- [ ] PPO training and validation
- [ ] Gazebo Sim2Real validation
- [ ] Real Go2 hardware deployment

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_dispatch_mvp,
  title={RL Dispatch MVP: Unified Dispatch and Rescheduling for Patrol Robots},
  author={YJP},
  year={2025},
  url={https://github.com/yjp/rl_dispatch_mvp}
}
```
