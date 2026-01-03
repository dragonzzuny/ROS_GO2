# Phase 1: Feasible Goal Generation - Implementation Patches

**Created:** 2025-12-30
**Reviewer:** 박용준
**Goal:** Reduce nav immediate failure rate from 95% to <10% by integrating A* pathfinding into candidate generation

---

## Quick Summary

✅ **What's Done:**
- Created `test_phase1_feasible_goals.py` - comprehensive test suite
- Created `IMPROVEMENT_PLAN.md` - full 4-phase plan
- Organized markdown files into `readme/completed_fixes/`

⚠️ **What Needs Manual Fix:**
- `src/rl_dispatch/planning/candidate_generator.py` - full rewrite needed
- `src/rl_dispatch/env/patrol_env.py` - one line addition

---

## Patch 1: patrol_env.py

**File:** `src/rl_dispatch/env/patrol_env.py`
**Location:** Line 191 (after nav_interface initialization)

### Change

Add this single line after `self.nav_interface = SimulatedNav2(...)`:

```python
# Reviewer 박용준: Connect nav_interface to candidate factory for A* pathfinding
self.candidate_factory.set_nav_interface(self.nav_interface)
```

### Full Context (Lines 184-195)

```python
self.np_random = np.random.default_rng(seed)

# Initialize navigation interface with seeded random state (Reviewer 박용준: Added occupancy grid)
self.nav_interface = SimulatedNav2(
    occupancy_grid=self.occupancy_grid,
    grid_resolution=self.env_config.grid_resolution,
    max_velocity=self.env_config.robot_max_velocity,
    nav_failure_rate=0.05,
    collision_rate=0.01,
    np_random=self.np_random,
)

# Reviewer 박용준: Connect nav_interface to candidate factory for A* pathfinding
self.candidate_factory.set_nav_interface(self.nav_interface)  # ← ADD THIS LINE
```

---

## Patch 2: candidate_generator.py

**File:** `src/rl_dispatch/planning/candidate_generator.py`

**This file needs extensive changes. Two options:**

### Option A: Use Git Patch (Recommended)

1. Save the complete new file (provided separately as `candidate_generator_phase1.py`)
2. Replace the old file:
   ```bash
   cp src/rl_dispatch/planning/candidate_generator.py src/rl_dispatch/planning/candidate_generator.py.backup
   cp candidate_generator_phase1.py src/rl_dispatch/planning/candidate_generator.py
   ```

### Option B: Manual Edits

Apply these changes to `src/rl_dispatch/planning/candidate_generator.py`:

#### 1. Import changes (Line 22)
```python
# Add Optional to imports
from typing import List, Tuple, Optional
```

#### 2. CandidateGenerator class (Lines 44-74)

**After `self.strategy_id = strategy_id`, add:**

```python
self.nav_interface = None  # Reviewer 박용준: Will be set by CandidateFactory

def set_nav_interface(self, nav_interface):
    """
    Set navigation interface for realistic path planning.

    Reviewer 박용준: Enables A* pathfinding for feasibility checks and ETA.

    Args:
        nav_interface: NavigationInterface instance (SimulatedNav2 or RealNav2)
    """
    self.nav_interface = nav_interface
```

#### 3. Add helper methods (after `__init__` and before `@abstractmethod`)

```python
def _get_distance_between(
    self,
    pos1: Tuple[float, float],
    pos2: Tuple[float, float],
) -> float:
    """
    Get distance between two positions.

    Reviewer 박용준: Uses A* path distance if available, else Euclidean.
    """
    if self.nav_interface is not None:
        distance = self.nav_interface.pathfinder.get_distance(pos1, pos2) if hasattr(self.nav_interface, 'pathfinder') and self.nav_interface.pathfinder else np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        return distance
    else:
        return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

def _get_eta(
    self,
    start: Tuple[float, float],
    goal: Tuple[float, float],
) -> float:
    """
    Get estimated time of arrival from start to goal.

    Reviewer 박용준: Uses NavigationInterface.get_eta() if available.
    """
    if self.nav_interface is not None:
        return self.nav_interface.get_eta(start, goal)
    else:
        distance = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        return distance / 1.0
```

#### 4. Replace `_estimate_route_distance` (Lines 75-108)

**Old:**
```python
def _estimate_route_distance(self, robot_pos, patrol_points, visit_order) -> float:
    # ... old Euclidean-based code
    total_distance += np.sqrt(dx**2 + dy**2)
```

**New:**
```python
def _estimate_route_distance(
    self,
    robot_pos: Tuple[float, float],
    patrol_points: Tuple[PatrolPoint, ...],
    visit_order: List[int],
) -> float:
    """
    Estimate total distance for a patrol route.

    Reviewer 박용준: Now uses A* path distance instead of Euclidean.
    """
    if len(visit_order) == 0:
        return 0.0

    total_distance = 0.0
    current_pos = robot_pos

    for idx in visit_order:
        point = patrol_points[idx]
        next_pos = (point.x, point.y)

        # Reviewer 박용준: Use A* distance
        distance = self._get_distance_between(current_pos, next_pos)

        if distance == np.inf:
            return np.inf  # Infeasible route

        total_distance += distance
        current_pos = next_pos

    return total_distance
```

#### 5. Replace `_calculate_max_coverage_gap` (Lines 110-156)

**Change the distance calculation to:**
```python
# Reviewer 박용준: Use realistic ETA
eta = self._get_eta(current_pos, next_pos)

if eta == np.inf:
    return np.inf  # Infeasible

estimated_time += eta
```

#### 6. Update NearestFirstGenerator.generate (Lines 225-256)

**Change the nearest point selection to:**
```python
while remaining:
    # Reviewer 박용준: Find nearest using A* distance
    nearest_idx = None
    nearest_dist = np.inf

    for idx in remaining:
        point = patrol_points[idx]
        dist = self._get_distance_between(current_pos, point.position)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_idx = idx

    if nearest_idx is None:
        nearest_idx = min(remaining)

    visit_order.append(nearest_idx)
    remaining.remove(nearest_idx)
    current_pos = patrol_points[nearest_idx].position
```

#### 7. Update OverdueETABalanceGenerator (Lines 350-385)

**Change distance calculation and add fallback:**
```python
# Reviewer 박용준: Calculate A* distance
distance = self._get_distance_between(current_pos, point.position)

if distance == np.inf:
    continue  # Skip infeasible
```

**And update time estimation:**
```python
eta = self._get_eta(current_pos, point.position)
estimated_time += eta if eta != np.inf else 10.0  # Fallback
```

#### 8. Update other generators similarly

Apply same pattern to:
- `BalancedCoverageGenerator`
- `OverdueThresholdFirstGenerator`
- `MinimalDeviationInsertGenerator`
- `ShortestETAFirstGenerator`

#### 9. Update CandidateFactory (Lines 852-866)

**Add nav_interface support:**

```python
def __init__(self):
    """Initialize factory with all 10 generators."""
    self.generators = [
        # ... existing generators
    ]
    self.nav_interface = None  # Reviewer 박용준: Will be set by environment

def set_nav_interface(self, nav_interface):
    """
    Set navigation interface for all generators.

    Reviewer 박용준: CRITICAL - Must be called before generate_all()!
    """
    self.nav_interface = nav_interface
    for generator in self.generators:
        generator.set_nav_interface(nav_interface)
```

---

## Testing

After applying patches:

```bash
# Test Phase 1 implementation
python test_phase1_feasible_goals.py

# Expected output:
# ✅ Test 1 PASS: All candidates are feasible
# ✅ Test 2 PASS: A* distance >= Euclidean
# ✅ Test 3 PASS: Immediate failure rate < 10%
```

---

## Verification Checklist

After implementing:

- [ ] `patrol_env.py` line added
- [ ] `candidate_generator.py` updated (or replaced)
- [ ] Test suite runs without errors
- [ ] All 3 tests pass
- [ ] Run short training to verify metrics:
  - [ ] Nav immediate failure rate < 10% (campus map)
  - [ ] Return std decreases
  - [ ] No `np.inf` distances in logs

---

## Next Steps (After Phase 1 Complete)

1. ✅ Phase 1 complete → Nav stabilized
2. 🔄 Phase 2: Reward redesign (delta coverage + normalization)
3. ⏳ Phase 3: Curriculum learning
4. ⏳ Phase 4: State space enhancement

---

**Questions or issues? Check:**
- `IMPROVEMENT_PLAN.md` for full context
- `readme/debug_guide/` for problem analysis
- Test output for specific failure reasons
