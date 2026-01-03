# RL Dispatch MVP - Systematic Improvement Plan

**Created:** 2025-12-30
**Based on:** Debug Guide v2.1 & Research Proposal v3.0
**Current Status:** Phase 0 (Planning Complete) → Starting Phase 1

---

## Executive Summary

This plan addresses critical learning instability and generalization failure in the multi-map PPO training system. Analysis of training logs reveals that **while PPO learning mechanics are stable** (KL divergence, clip fraction, entropy all normal), **Return improvement is minimal** with extreme map-wise variance.

**Root Cause Analysis:**
1. **Nav Immediate Failure (Priority #1):** 95.29% of actions in `campus` map fail within 1 second, indicating infeasible goal generation
2. **Reward Signal Distortion (Priority #2):** Patrol penalty (-332/step) drowns out event reward (+0.12/step)
3. **Structural KPI Issues:** Coverage metric unsuitable for large maps, only 5-15% achievable

**Solution Approach:** 4-Phase sequential improvement
- Phase 1: Nav stabilization & feasible goal generation
- Phase 2: Reward function redesign
- Phase 3: Curriculum learning
- Phase 4: State space enhancement

---

## Current Performance Data

### Map-wise Performance Variance

| Map | Nav Immediate Fail Rate | Nav Final Fail Rate | Avg Patrol Penalty | Avg Event Reward | Coverage |
|:----|:----------------------:|:-------------------:|:------------------:|:----------------:|:--------:|
| campus | **95.29%** | 69.93% | -332.12 | +0.12 | 5% |
| office_building | **78.11%** | 59.73% | -354.29 | +1.05 | 8% |
| warehouse | 70.50% | 46.75% | - | - | 12% |
| corridor | 14.56% | 6.15% | -143.46 | -0.46 | 42% |
| l_shaped | 12.92% | 6.02% | - | - | 38% |
| large_square | 9.63% | 6.23% | - | - | 35% |

**Key Observation:** Nav immediate failure rate (`nav_time < 1.0s`) correlates perfectly with low coverage and high Return variance.

### PPO Learning Stability (Confirmed Normal)

- **KL Divergence:** 0.004 - 0.01 (within target)
- **Clip Fraction:** 0.05 - 0.15 (healthy policy updates)
- **Entropy:** 0.01 - 0.10 (proper exploration-exploitation balance)
- **Value Loss:** Increasing but not exploding

**Conclusion:** PPO algorithm is functioning correctly. The problem is in the environment/reward design, not the learning algorithm.

### Action Distribution Anomaly

- **Observed Actions:** Only `PATROL` and `DISPATCH` modes
- **Missing Actions:** `WAIT`, `CHARGE`, `CHARGE_THEN_DISPATCH`
- **Battery Level:** Stays near 1.0 (charging never needed)

**Implication:** Action space may be over-engineered for current task complexity.

---

## Phase 1: Nav Stabilization & Feasible Goal Generation

**Goal:** Reduce nav immediate failure rate from 95% to <10% in campus/office maps.

### Success Criteria
- [ ] `nav_time < 1.0s` rate drops to <10% in campus map
- [ ] `nav_time < 1.0s` rate drops to <15% in office_building map
- [ ] Overall nav_success rate improves to >80% across all maps
- [ ] Return std_dev in campus drops from 83k to <40k

### Implementation Tasks

#### 1.1. Add Feasibility Validation in CandidateGenerator
**File:** `src/rl_dispatch/navigation/candidate_generator.py`

**Changes:**
- Call `SimulatedNav2.get_eta()` for each generated candidate
- Mark candidates as `feasible=True` only if ETA is valid (not inf, not None)
- If all candidates are infeasible, fall back to nearest valid patrol point
- Log infeasible candidate rate for monitoring

**Code Pattern:**
```python
def _validate_candidate(self, candidate: Candidate) -> bool:
    """Check if candidate goal is reachable via A* pathfinding."""
    eta = self.nav2_interface.get_eta(
        start_pos=self.robot_state.position,
        goal_pos=candidate.target_position
    )
    return eta is not None and eta != float('inf') and eta > 0
```

#### 1.2. Implement Safe Zone Logic for Goal Placement
**File:** `src/rl_dispatch/navigation/candidate_generator.py`

**Changes:**
- Before setting goal coordinates, check `occupancy_grid` for obstacles
- If goal is inside obstacle, find nearest free space using BFS
- Add configurable `goal_safety_margin` parameter (default: 0.5m)

**Code Pattern:**
```python
def _find_safe_goal(self, raw_position: tuple) -> tuple:
    """Adjust goal to nearest free space if inside obstacle."""
    if self._is_obstacle(raw_position):
        return self._nearest_free_space(raw_position, margin=0.5)
    return raw_position
```

#### 1.3. Add Nav Failure Recovery Mechanisms
**File:** `src/rl_dispatch/env/patrol_env.py`

**Changes:**
- When nav fails immediately, don't terminate episode
- Execute fallback action: move to nearest patrol point or stay in place
- Apply small penalty (-10) instead of large penalty cascade
- Log recovery events for analysis

**Code Pattern:**
```python
if nav_time < 1.0 and not nav_success:
    # Nav failed immediately - likely infeasible goal
    self.logger.warning(f"Nav immediate failure - using fallback action")
    fallback_action = self._get_fallback_action()
    nav_result = self._execute_fallback(fallback_action)
    step_reward -= 10.0  # Small penalty for nav failure
```

#### 1.4. Add A* Pathfinding Implementation
**File:** `src/rl_dispatch/navigation/pathfinding.py` (already exists as untracked file)

**Changes:**
- Implement efficient A* pathfinding on occupancy grid
- Use Manhattan distance heuristic for grid-based search
- Cache recent path queries for performance
- Return None if no path exists (instead of crashing)

### Validation Methods

**Quick Test:**
```bash
# Run single-episode test on campus map
python3 scripts/test_nav_feasibility.py --map campus --num_episodes 10
```

**Full Training Test:**
```bash
# Train for 1M steps and check nav metrics
./train_fixed.sh --total_timesteps 1000000 --log_interval 10000
# Check tensorboard: nav_time distribution, nav_success_rate
```

**Expected Results:**
- `nav_time < 1.0s` rate should drop dramatically in first 100k steps
- Return variance should decrease
- Coverage should improve as robots can actually reach goals

---

## Phase 2: Reward Function Redesign

**Goal:** Normalize reward signals so event success learning is not drowned out by patrol penalties.

### Success Criteria
- [ ] Patrol reward magnitude comparable to event reward magnitude
- [ ] Coverage increases from 5% to 15%+ in campus map
- [ ] Event success rate maintains or improves
- [ ] Return variance continues to decrease

### Implementation Tasks

#### 2.1. Change Patrol Reward to Delta Coverage
**File:** `src/rl_dispatch/rewards/reward_calculator.py`

**Current Issue:** Patrol penalty accumulates every step regardless of robot action, becoming -332/step in large maps.

**Solution:** Reward coverage improvement, not absolute coverage.

**Changes:**
```python
def _calculate_patrol_reward(
    self,
    prev_coverage: float,
    current_coverage: float
) -> float:
    """Reward based on coverage change, not absolute value."""
    coverage_delta = current_coverage - prev_coverage

    # w_patrol_delta: 50.0 ~ 100.0 (needs high weight as delta is small)
    return self.config.w_patrol_delta * coverage_delta
```

**Config Addition:**
```yaml
# configs/map_campus.yaml
reward:
  w_patrol_delta: 75.0  # Weight for coverage improvement
  # Remove old w_patrol penalty
```

#### 2.2. Implement Per-Component Reward Normalization
**File:** `src/rl_dispatch/rewards/reward_calculator.py`

**Current Issue:** Event reward (+0.12) vs patrol penalty (-332) are on completely different scales.

**Solution:** Normalize each reward component separately with running statistics.

**Changes:**
```python
class RewardNormalizer:
    """Online normalization using Welford's algorithm."""
    def __init__(self):
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0

    def normalize(self, value: float) -> float:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (value - self.mean)

        if self.count < 2:
            return value

        variance = self.M2 / (self.count - 1)
        std = max(variance ** 0.5, 1e-8)
        return (value - self.mean) / std

# In RewardCalculator:
def __init__(self, config):
    self.patrol_normalizer = RewardNormalizer()
    self.event_normalizer = RewardNormalizer()
    self.efficiency_normalizer = RewardNormalizer()

def calculate(self, ...):
    r_patrol_raw = self._calculate_patrol_reward(...)
    r_event_raw = self._calculate_event_reward(...)

    r_patrol = self.patrol_normalizer.normalize(r_patrol_raw)
    r_event = self.event_normalizer.normalize(r_event_raw)

    total_reward = r_patrol + r_event + ...
```

#### 2.3. Redesign Event Rewards Based on SLA
**File:** `src/rl_dispatch/rewards/reward_calculator.py`

**Current Issue:** Event rewards are arbitrary numbers, not reflecting real-world value.

**Solution:** Base rewards on realistic Service Level Agreement costs.

**Example:**
```python
# SLA-based reward design
EVENT_SUCCESS_VALUE = 1000.0  # $1000 saved per successful event
EVENT_FAILURE_COST = -2000.0  # $2000 penalty per missed event
SLA_URGENCY_MULTIPLIER = 2.0  # 2x penalty if event is urgent

def _calculate_event_reward(self, event_outcome: str, urgency: float) -> float:
    if event_outcome == "success":
        base_reward = EVENT_SUCCESS_VALUE
    elif event_outcome == "failure":
        base_reward = EVENT_FAILURE_COST * (1.0 + urgency * SLA_URGENCY_MULTIPLIER)
    else:
        base_reward = 0.0

    return base_reward
```

### Validation Methods

**Reward Component Logging:**
```python
# In patrol_env.py step()
info['reward_components'] = {
    'patrol_raw': r_patrol_raw,
    'patrol_normalized': r_patrol,
    'event_raw': r_event_raw,
    'event_normalized': r_event,
}
```

**Check in TensorBoard:**
- Reward components should have similar magnitudes (-1 to +1 after normalization)
- Coverage trajectory should show upward trend
- Event success rate should remain stable or improve

---

## Phase 3: Curriculum Learning

**Goal:** Train progressively from simple to complex maps for better generalization.

### Success Criteria
- [ ] Phase 1 (simple maps): Value loss < 100k, coverage > 30%
- [ ] Phase 2 (medium maps): Smooth transition, no performance collapse
- [ ] Phase 3 (complex maps): Coverage > 15%, nav_success > 70%
- [ ] All maps: Stable Return with low variance

### Curriculum Design

#### Stage 1: Simple Maps (0 - 2M steps)
**Maps:** corridor, l_shaped, large_square
**Goal:** Build stable critic baseline

**Success Criteria:**
- Avg coverage > 30%
- Nav success rate > 90%
- Value loss < 100k
- Return std < 5k

**Training:**
```bash
python3 scripts/train_multi_map.py \
  --maps corridor l_shaped large_square \
  --total_timesteps 2000000 \
  --checkpoint_dir checkpoints/curriculum_stage1
```

#### Stage 2: Medium Maps (2M - 5M steps)
**Maps:** corridor, l_shaped, large_square, warehouse
**Goal:** Transfer to larger map

**Starting Point:** Load checkpoint from Stage 1
**Success Criteria:**
- Warehouse coverage > 20%
- No catastrophic forgetting on simple maps
- Return variance increase < 50%

**Training:**
```bash
python3 scripts/train_multi_map.py \
  --maps corridor l_shaped large_square warehouse \
  --total_timesteps 3000000 \
  --load_checkpoint checkpoints/curriculum_stage1/final.pt \
  --checkpoint_dir checkpoints/curriculum_stage2
```

#### Stage 3: Complex Maps (5M - 10M steps)
**Maps:** All 6 maps including campus, office_building
**Goal:** Full generalization

**Starting Point:** Load checkpoint from Stage 2
**Success Criteria:**
- Campus coverage > 15%
- Office coverage > 15%
- Nav success > 70% on all maps
- Stable learning for 5M steps

**Training:**
```bash
python3 scripts/train_multi_map.py \
  --maps corridor l_shaped large_square warehouse office_building campus \
  --total_timesteps 5000000 \
  --load_checkpoint checkpoints/curriculum_stage2/final.pt \
  --checkpoint_dir checkpoints/curriculum_stage3
```

### Implementation

**File:** `scripts/train_curriculum.py` (new file)

**Features:**
- Automatic stage transition based on KPI thresholds
- Checkpoint loading between stages
- Stage-specific hyperparameters (entropy schedule, LR)

---

## Phase 4: State Space Enhancement

**Goal:** Provide policy with critical decision-making information.

### Success Criteria
- [ ] Policy can distinguish urgent vs non-urgent events
- [ ] Policy can identify patrol crisis zones
- [ ] Value function accuracy improves (lower TD error)

### Implementation Tasks

#### 4.1. Add Event Urgency Information
**File:** `src/rl_dispatch/env/observation_processor.py`

**New Features (2D):**
- `selected_event_urgency`: Urgency score of current event target (0-1)
- `time_until_event_timeout`: Seconds until selected event times out

#### 4.2. Add Patrol Crisis Information
**File:** `src/rl_dispatch/env/observation_processor.py`

**New Features (2D):**
- `max_patrol_gap`: Maximum time since any patrol point was visited
- `patrol_crisis_level`: Normalized crisis level (max_gap / threshold)

#### 4.3. Add Candidate Feasibility Information
**File:** `src/rl_dispatch/env/observation_processor.py`

**New Features (num_candidates × 2):**
- `candidate_feasible`: Binary flag per candidate
- `candidate_eta`: Normalized ETA per candidate

**Total State Dimension:** 77D → 83D (6 new features)

### Validation

**Check:**
- Policy network accepts new state dimension
- TensorBoard shows value prediction accuracy improving
- Policy learns to prioritize urgent events (check action distribution)

---

## Verification Checkpoints

After each phase, verify these metrics before proceeding:

### Phase 1 Completion Checklist
- [ ] Campus `nav_time < 1.0s` rate < 10%
- [ ] Office `nav_time < 1.0s` rate < 15%
- [ ] Campus Return std < 40k
- [ ] Overall nav_success > 80%

### Phase 2 Completion Checklist
- [ ] Reward components have similar magnitudes
- [ ] Campus coverage > 15%
- [ ] Event success rate stable or improved
- [ ] Return continues improving

### Phase 3 Completion Checklist
- [ ] All stages completed without collapse
- [ ] Final coverage: simple maps > 30%, complex maps > 15%
- [ ] Return variance stable across all maps
- [ ] Nav success > 70% on all maps

### Phase 4 Completion Checklist
- [ ] State dimension updated successfully
- [ ] Value prediction error decreased
- [ ] Policy shows intelligent prioritization
- [ ] No performance regression

---

## Testing & Monitoring

### Quick Tests (After Each Code Change)
```bash
# Test nav feasibility
python3 scripts/test_nav_feasibility.py --map campus

# Test reward calculation
python3 scripts/test_reward_calculator.py

# Test observation processing
python3 scripts/test_observation.py
```

### Training Runs (After Each Phase)
```bash
# Short training run (1M steps)
./train_fixed.sh --total_timesteps 1000000 --log_interval 5000

# Monitor in TensorBoard
tensorboard --logdir runs/

# Check CSV logs
python3 scripts/analyze_training_logs.py runs/latest/steps.csv
```

### Key Metrics to Monitor

**Nav Metrics:**
- `nav_time` distribution (should shift right)
- `nav_success_rate` (target: >80%)
- `nav_immediate_fail_rate` (target: <10%)

**Reward Metrics:**
- `reward_patrol_raw` vs `reward_patrol_normalized`
- `reward_event_raw` vs `reward_event_normalized`
- `total_reward` trend (should increase)

**Performance Metrics:**
- `patrol_coverage` (target: >15% complex, >30% simple)
- `event_success_rate` (target: maintain >80%)
- `mean_return` (should increase with low variance)
- `std_return` (should decrease)

**Learning Metrics:**
- `approx_kl` (should stay 0.004-0.01)
- `clip_fraction` (should stay 0.05-0.15)
- `entropy` (should follow schedule)
- `value_loss` (should decrease after nav fix)

---

## Timeline & Priority

**Priority Order:**
1. **Phase 1 (Critical):** Nav stabilization - Without this, nothing else matters
2. **Phase 2 (High):** Reward redesign - Enables meaningful learning signals
3. **Phase 3 (Medium):** Curriculum - Improves generalization
4. **Phase 4 (Low):** State enhancement - Final optimization

**Estimated Implementation:**
- Phase 1: 1-2 days (critical path)
- Phase 2: 1 day (reward engineering)
- Phase 3: 3-5 days (training time dominant)
- Phase 4: 0.5 day (small changes)

**Note:** Do not skip Phase 1. All other improvements will fail if navigation is broken.

---

## Appendix: Debug Data Sources

**Training Logs Analyzed:**
- `steps.csv`: Step-level nav_time, nav_success, reward components
- `episodes.csv`: Episode-level returns, success rates, coverage
- `tensorboard_logs/`: PPO learning curves, KL divergence, entropy

**Key Finding Quote from Debug Guide v2.1:**
> "Campus 맵에서 스텝의 95%가 1초 안에 끝난다는 것은, 정책이 목표를 정해도 SimulatedNav2가 경로 계획 자체를 시작하지 못하고 즉시 실패(abort)를 반환하고 있음을 의미합니다."

This confirms nav immediate failure is the root cause, not reward design issues.
