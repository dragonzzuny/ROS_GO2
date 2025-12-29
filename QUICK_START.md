# Quick Start Guide

**Get started with RL Dispatch MVP in 5 minutes!**

---

## Installation

### 1. Clone or navigate to repository

```bash
cd rl_dispatch_mvp
```

### 2. Install package

```bash
# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### 3. Verify installation

```bash
python -c "from rl_dispatch.env import PatrolEnv; print('✅ Installation successful!')"
```

---

## Quick Test Run

### Test the environment

```python
from rl_dispatch.env import PatrolEnv

# Create environment
env = PatrolEnv()

# Reset environment
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # (77,)

# Take a random action
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward:.2f}")
print(f"Info: {info}")
```

### Test candidate generation

```python
from rl_dispatch.planning import CandidateFactory
from rl_dispatch.core.types import RobotState, PatrolPoint

# Create sample data
robot = RobotState(x=5.0, y=5.0, heading=0.0, velocity=0.5,
                  angular_velocity=0.0, battery_level=0.8)
patrol_points = (
    PatrolPoint(x=10.0, y=10.0, last_visit_time=0.0, priority=1.0, point_id=0),
    PatrolPoint(x=40.0, y=10.0, last_visit_time=50.0, priority=1.0, point_id=1),
)

# Generate all candidates
factory = CandidateFactory()
candidates = factory.generate_all(robot, patrol_points, current_time=100.0)

# Print strategies
for candidate in candidates:
    print(f"{candidate.strategy_name}: {candidate.patrol_order}")
```

---

## Training

### Option A: Single Map Training (Basic)

#### 1. Quick training (1M steps)

```bash
python scripts/train.py --seed 42 --cuda
```

#### 2. Custom configuration

```bash
python scripts/train.py \
    --env-config configs/custom_env.yaml \
    --reward-config configs/custom_reward.yaml \
    --run-name my_experiment \
    --seed 123
```

---

### Option B: Multi-Map Training (⭐ Recommended for Generalization)

**일반화된 정책 학습 - 다양한 맵에서 훈련하여 어떤 환경에서도 잘 작동하는 모델**

#### 1. 기본 멀티맵 학습 (6개 다양한 맵)

```bash
python scripts/train_multi_map.py \
    --total-timesteps 5000000 \
    --seed 42 \
    --cuda
```

자동으로 다음 맵들을 사용합니다:
- `map_large_square` - 큰 정사각형 (100×100m)
- `map_corridor` - 복도형 (120×30m)
- `map_l_shaped` - L자형 건물 (80×80m)
- `map_office_building` - 사무실 (90×70m)
- `map_campus` - 캠퍼스 (150×120m)
- `map_warehouse` - 창고 (140×100m)

#### 2. 특정 맵들만 선택

```bash
python scripts/train_multi_map.py \
    --maps configs/map_large_square.yaml configs/map_office_building.yaml \
    --map-mode random \
    --seed 42
```

#### 3. 커리큘럼 학습 (쉬운 맵 → 어려운 맵)

```bash
python scripts/train_multi_map.py \
    --map-mode curriculum \
    --total-timesteps 10000000 \
    --seed 42 \
    --cuda
```

초반에는 작고 간단한 맵에서 학습하고, 점차 크고 복잡한 맵으로 전환됩니다.

---

### 3. Monitor with TensorBoard

```bash
tensorboard --logdir runs
```

Open browser to http://localhost:6006

**멀티맵 학습 시 TensorBoard에서 확인 가능:**
- 각 맵별 성능 (episode_per_map/)
- 전체 평균 성능 (episode/)
- 맵별 이벤트 성공률, 순찰 커버리지

---

## Evaluation

### Option A: Single Map Evaluation

```bash
python scripts/evaluate.py \
    --model checkpoints/my_experiment/final_model.pth \
    --episodes 100
```

---

### Option B: Multi-Map Generalization Evaluation (⭐ Recommended)

**여러 맵에서 정책의 일반화 성능을 평가합니다.**

#### 1. 일반화 성능 평가

```bash
python scripts/evaluate_generalization.py \
    --model checkpoints/multi_map_ppo/final.pth \
    --episodes 50 \
    --save-json
```

**결과:**
- `outputs/generalization/generalization_results.json` - 상세 데이터
- `outputs/generalization/generalization_comparison.png` - 맵별 성능 비교
- `outputs/generalization/return_distribution.png` - 리턴 분포

**출력 예시:**
```
Map                       Return               Event Success        Patrol Coverage
---------------------------------------------------------------------------------
map_large_square         523.4 ± 45.2         87.3% ± 8.1%        94.2% ± 3.5%
map_office_building      -234.1 ± 67.8        72.1% ± 11.2%       88.6% ± 5.2%
map_campus               312.5 ± 89.3         81.4% ± 9.7%        91.3% ± 4.8%
```

#### 2. 커버리지 히트맵 시각화

```bash
python scripts/visualize_coverage.py \
    --model checkpoints/multi_map_ppo/final.pth \
    --episodes-per-map 20 \
    --output outputs/coverage_heatmaps.png
```

**확인 가능:**
- 각 맵에서 로봇이 방문한 영역 (히트맵)
- 순찰 커버리지 통계
- 사각지대 파악

---

### Quick Code Evaluation

```python
from rl_dispatch.env import PatrolEnv
from rl_dispatch.algorithms import PPOAgent

# Load trained agent
agent = PPOAgent()
agent.load("checkpoints/my_experiment/final_model.pth")

# Evaluate
env = PatrolEnv()
obs, _ = env.reset()

for _ in range(100):
    action, _, _ = agent.get_action(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        print(f"Episode return: {info['episode']['r']:.2f}")
        obs, _ = env.reset()
```

---

## Baseline Comparison

### Test baseline policies

```python
from rl_dispatch.env import PatrolEnv
from rl_dispatch.algorithms.baselines import (
    B0_AlwaysPatrol,
    B1_AlwaysDispatch,
    B2_ThresholdDispatch,
    B3_UrgencyBased,
    B4_HeuristicPolicy,
)

env = PatrolEnv()
policy = B2_ThresholdDispatch()

# Run episode
obs, _ = env.reset()
done = False
episode_return = 0

while not done:
    state = env.current_state  # Get full state
    action = policy.select_action(state)
    action_array = [action.mode, action.replan_idx]
    obs, reward, done, truncated, info = env.step(action_array)
    episode_return += reward
    done = done or truncated

print(f"Baseline return: {episode_return:.2f}")
```

---

## Run Tests

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_env.py -v
```

---

## Project Structure Overview

```
rl_dispatch_mvp/
├── src/rl_dispatch/        # Main package
│   ├── core/               # Data structures & configs
│   ├── env/                # Gymnasium environment
│   ├── algorithms/         # PPO & baselines
│   ├── planning/           # Candidate generation
│   ├── rewards/            # Reward calculation
│   └── utils/              # Utilities
│
├── scripts/                # Training & evaluation
│   ├── train.py           # Main training script
│   └── evaluate.py        # Evaluation script
│
├── configs/                # Configuration files
│   └── default.yaml       # Default config
│
├── tests/                  # Unit tests
│   ├── test_core_types.py
│   ├── test_candidate_generation.py
│   └── test_env.py
│
└── readme/                 # Documentation
    ├── PROGRESS.md        # Development progress
    ├── development_guide.md
    └── R&D_Plan_Complete_v4.md
```

---

## Common Tasks

### 1. Create custom environment config

```yaml
# configs/my_env.yaml
env:
  map_width: 100.0
  map_height: 100.0
  patrol_points:
    - [20.0, 20.0]
    - [80.0, 20.0]
    - [80.0, 80.0]
    - [20.0, 80.0]
  max_episode_steps: 300
  event_generation_rate: 8.0
```

### 2. Tune reward weights

```yaml
# configs/my_reward.yaml
reward:
  w_event: 1.5      # Increase event importance
  w_patrol: 0.8     # Increase patrol importance
  w_safety: 3.0     # Heavily penalize safety violations
  w_efficiency: 0.05  # Reduce efficiency penalty
```

### 3. Adjust training hyperparameters

```yaml
# configs/my_training.yaml
training:
  learning_rate: 0.0001  # Lower LR for stability
  total_timesteps: 20000000  # Longer training
  num_steps: 4096    # Larger batch
  num_epochs: 15     # More epochs per update
```

---

## Troubleshooting

### Issue: Import errors

```bash
# Ensure package is installed
pip install -e .

# Check installation
python -c "import rl_dispatch; print(rl_dispatch.__version__)"
```

### Issue: CUDA out of memory

```bash
# Use CPU instead
python scripts/train.py  # --cuda flag removed

# Or reduce batch size in training config
```

### Issue: Environment not converging

```bash
# Try different random seed
python scripts/train.py --seed 123

# Adjust reward weights
# Edit configs/default.yaml

# Increase learning rate
# Edit training config: learning_rate: 0.001
```

---

## Next Steps

1. **Read full documentation**: See `readme/` folder
2. **Experiment with configurations**: Modify `configs/default.yaml`
3. **Run ablation studies**: Test different reward weights
4. **Compare with baselines**: Use baseline policies
5. **Visualize results**: Use TensorBoard

---

## Getting Help

- **Documentation**: See `readme/` folder
- **Examples**: See `scripts/` folder
- **Tests**: See `tests/` folder for usage examples
- **Issues**: Open GitHub issue

---

**Happy Training! 🚀**
