# TODO: Pending Improvements and Issues

**Date:** 2026-01-03
**Status:** Active Development
**Priority Legend:** P0 (Critical) > P1 (High) > P2 (Medium) > P3 (Low)

---

## Critical Issues (P0)

### 1. Git Status Cleanup Required

**Problem:** 많은 파일들이 unstaged 또는 untracked 상태입니다.

**Files to Stage:**
```bash
# Modified files needing commit
git add src/rl_dispatch/algorithms/ppo.py
git add src/rl_dispatch/core/config.py
git add src/rl_dispatch/core/types.py
git add src/rl_dispatch/env/patrol_env.py
git add src/rl_dispatch/planning/candidate_generator.py
git add src/rl_dispatch/rewards/reward_calculator.py
git add src/rl_dispatch/utils/observation.py
git add configs/map_*.yaml
git add IMPLEMENTATION_GUIDE.md
```

**Files to Add (Untracked):**
```bash
# New implementation files
git add IMPROVEMENT_PLAN.md
git add PHASE1_PATCHES.md
git add PHASE2_SUMMARY.md
git add PHASE3_SUMMARY.md
git add PHASE4_SUMMARY.md
git add PBT_DESIGN.md

# New scripts
git add scripts/train_curriculum.py
git add scripts/train_pbt.py
git add scripts/run_curriculum.sh
git add scripts/validate_phase2.sh

# New PBT module
git add src/rl_dispatch/pbt/

# Test files
git add test_phase1_feasible_goals.py
git add test_phase2_reward_redesign.py
git add test_phase3_curriculum.py
git add test_phase4_state_enhancements.py
git add test_pbt_implementation.py
git add test_patrol_env_fix.py

# Documentation
git add readme/completed_fixes/
git add readme/debug_guide/
```

**Files to Delete (from git tracking):**
```bash
# Files moved to readme/completed_fixes/
git rm ACTION_MASKING_FIX.md
git rm BUGFIX_UNPACKING.md
git rm BUG_FIXES_REPORT.md
git rm IMPLEMENTATION_SUMMARY.md
git rm LOGGING_SYSTEM.md
git rm POLICY_COLLAPSE_FIX.md
git rm QUICK_START_LOGGING.md
git rm readme/debug_guide.md
```

**Action Required:** 커밋 전에 위 파일들을 정리해야 합니다.

---

### 2. LF/CRLF Line Ending Inconsistency

**Problem:** Git 경고 메시지:
```
warning: in the working copy of 'src/rl_dispatch/core/config.py',
LF will be replaced by CRLF the next time Git touches it
```

**Solution:** `.gitattributes` 파일 추가:
```
# .gitattributes
* text=auto eol=lf
*.py text eol=lf
*.yaml text eol=lf
*.md text eol=lf
*.sh text eol=lf
```

**Action Required:** `.gitattributes` 파일을 생성하고 line ending을 일관되게 만들어야 합니다.

---

## High Priority Issues (P1)

### 3. Test Files in Root Directory

**Problem:** 테스트 파일들이 루트 디렉토리에 산발적으로 위치해 있습니다.

**Current State:**
```
/test_phase1_feasible_goals.py
/test_phase2_reward_redesign.py
/test_phase3_curriculum.py
/test_phase4_state_enhancements.py
/test_pbt_implementation.py
/test_patrol_env_fix.py
/quick_phase2_validation.py
/patch_patrol_env.py
```

**Recommended:**
```
/tests/
  test_phase1_feasible_goals.py
  test_phase2_reward_redesign.py
  test_phase3_curriculum.py
  test_phase4_state_enhancements.py
  test_pbt_implementation.py
  test_patrol_env_fix.py
  test_core_types.py  (already exists)
  test_candidate_generation.py  (already exists)
  test_env.py  (already exists)
```

**Action Required:** 테스트 파일들을 `tests/` 폴더로 이동하고 import 경로를 수정해야 합니다.

---

### 4. Shell Scripts Need Cleanup

**Problem:** 루트에 여러 .sh 파일들이 있습니다.

**Files:**
```
/train_fixed.sh
/run_test_phase3.sh
/apply_phase1.sh
```

**Recommended:** `scripts/` 폴더로 이동 또는 필요 없는 파일 삭제

---

### 5. Validation Training Not Complete

**Problem:** Phase 2~4의 실제 훈련 검증이 아직 완료되지 않았습니다.

**Current Status (from PHASE2_SUMMARY.md):**
```
- [x] Component normalization tests: PASS
- [x] Delta coverage tests: PASS
- [x] SLA scaling tests: PASS
- [ ] Training validation: Return std < 40k (in progress)
- [ ] Learning stability: Value loss < 100k (in progress)
```

**Action Required:**
```bash
# Run validation training
bash scripts/run_curriculum.sh

# Monitor TensorBoard
tensorboard --logdir runs/
```

---

## Medium Priority Issues (P2)

### 6. Documentation Inconsistencies

**Problem:** 일부 문서가 최신 코드와 불일치합니다.

**Examples:**
- `FINAL_STATUS.md`: "77D observation" (now 88D)
- `IMPLEMENTATION_STATUS.md`: Old line counts
- Some docstrings reference old parameter names

**Action Required:**
- Update `readme/FINAL_STATUS.md` with Phase 4 changes
- Update line counts in status documents
- Review and update docstrings

---

### 7. Hardcoded Values in Code

**Problem:** 일부 값들이 config가 아닌 코드에 하드코딩되어 있습니다.

**Examples:**
```python
# observation.py
max_num_patrol_points: int = 30  # Should be configurable

# reward_calculator.py
crisis_threshold = self.max_coverage_gap * 0.5  # Magic number
```

**Recommendation:** 중요한 값들을 config로 이동

---

### 8. PBT Module Not Fully Integrated

**Problem:** `src/rl_dispatch/pbt/` 모듈이 생성되었지만 main training loop와 통합되지 않았습니다.

**Current State:**
- `pbt_manager.py` - Created
- `population_member.py` - Created
- `hyperparameter_space.py` - Created
- Integration with `train_pbt.py` - Needs verification

**Action Required:** PBT 훈련 파이프라인 테스트 및 검증

---

## Low Priority Issues (P3)

### 9. Missing Type Annotations

**Problem:** 일부 새 함수들에 타입 힌트가 누락되어 있습니다.

**Examples:**
```python
# Should have return type
def _extract_event_risk(self, state: State):  # -> float
```

**Recommendation:** mypy 또는 pyright로 검사 후 수정

---

### 10. Test Coverage Gaps

**Problem:** 일부 새 기능들의 테스트 커버리지가 부족합니다.

**Missing Tests:**
- `ComponentNormalizer` edge cases (zero variance, large values)
- `_extract_candidate_feasibility` with all-infeasible candidates
- PBT module integration tests

**Recommendation:** pytest-cov로 커버리지 측정 후 보완

---

### 11. Deprecated Parameters

**Problem:** Phase 2에서 deprecated된 파라미터들이 아직 config에 남아 있습니다.

**Deprecated:**
```yaml
# config.py
patrol_gap_penalty_rate: float = 0.1  # DEPRECATED
patrol_visit_bonus: float = 2.0  # DEPRECATED
```

**Recommendation:** v2.0에서 완전히 제거하거나 warning 추가

---

### 12. Large Binary File

**Problem:** `rl_dispatch_mvp.zip`이 매우 커졌습니다 (278KB → 7.9MB).

**Current Size:** 7,957,866 bytes

**Recommendation:**
- `.gitignore`에 `*.zip` 추가
- 또는 Git LFS 사용
- 또는 별도 release artifacts로 관리

---

## Code Quality Improvements

### 13. Consistent Logging

**Problem:** 로깅 스타일이 일관되지 않습니다.

**Current:**
```python
print(f"✅ Test passed")  # Some use print with emoji
self.logger.info("...")   # Some use logger
logging.debug("...")      # Some use logging directly
```

**Recommendation:** 모든 모듈에서 일관된 logger 사용

---

### 14. Magic Strings

**Problem:** Action mode 등이 문자열로 비교됩니다.

**Example:**
```python
if action.mode == "DISPATCH":  # Magic string
```

**Recommendation:** Enum 또는 상수 사용

---

## Feature Improvements

### 15. Multi-Agent Support

**Status:** Not implemented

**Description:** 현재 단일 로봇만 지원합니다. 다중 로봇 협력 순찰을 위한 확장이 필요합니다.

**Estimated Effort:** Large (3-5 weeks)

---

### 16. ROS2 Integration

**Status:** Simulated only

**Description:** 실제 Nav2 ROS2 노드와의 통합이 필요합니다.

**Files to Modify:**
- `src/rl_dispatch/navigation/nav2_interface.py`
- Add `src/rl_dispatch/ros2/` module

**Estimated Effort:** Medium (1-2 weeks)

---

### 17. Visualization Dashboard

**Status:** Basic only

**Description:** 실시간 모니터링 대시보드가 필요합니다.

**Current:**
- TensorBoard logging
- matplotlib plots

**Needed:**
- Real-time web dashboard
- Episode replay visualization

**Estimated Effort:** Medium (1-2 weeks)

---

## Quick Fix Commands

```bash
# 1. Fix line endings
git config core.autocrlf input

# 2. Stage all modified source files
git add src/rl_dispatch/**/*.py

# 3. Stage new files
git add IMPROVEMENT_PLAN.md PHASE*.md PBT_DESIGN.md
git add scripts/train_curriculum.py scripts/train_pbt.py

# 4. Remove deleted files from tracking
git rm --cached ACTION_MASKING_FIX.md BUGFIX_UNPACKING.md BUG_FIXES_REPORT.md

# 5. Create commit
git commit -m "feat: Add 4-phase learning improvement system

- Phase 1: A* pathfinding for feasible goal generation
- Phase 2: Per-component reward normalization + Delta Coverage
- Phase 3: 3-stage curriculum learning
- Phase 4: State space expansion (77D → 88D)

Breaking changes:
- Observation dimension changed from 77D to 88D
- New config parameters required
- Existing trained models incompatible

Generated with Claude Code"

# 6. Run tests before push
python -m pytest tests/ test_phase*.py -v
```

---

## Progress Tracking

| Task | Priority | Status | Assignee |
|------|----------|--------|----------|
| Git status cleanup | P0 | TODO | - |
| Line ending fix | P0 | TODO | - |
| Move test files | P1 | TODO | - |
| Validation training | P1 | In Progress | - |
| Documentation update | P2 | TODO | - |
| PBT integration | P2 | TODO | - |
| Type annotations | P3 | TODO | - |
| Test coverage | P3 | TODO | - |

---

**Last Updated:** 2026-01-03
**Next Review:** After validation training complete
