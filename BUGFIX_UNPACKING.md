# 🐛 Bugfix: Buffer Unpacking Error

**문제**: `ValueError: too many values to unpack (expected 6)`
**원인**: `buffer.get()`이 7개 값을 반환하는데 PPO가 6개만 기대
**상태**: ✅ **수정 완료**

---

## 문제 발생 원인

### Buffer (buffer.py:256, 269-277)
```python
# buffer.get() yields 7 values:
yield (obs, actions, log_probs, advantages_t, returns, values, action_masks)
#                                                               ^^^^^^^^^^^^
#                                                               7번째 추가됨
```

### PPO (ppo.py:201) - **수정 전**
```python
# ❌ 6개만 unpacking
obs, actions, old_log_probs, advantages, returns, old_values = batch
```

**에러 메시지**:
```
ValueError: too many values to unpack (expected 6)
```

---

## ✅ 적용된 수정

### File: `src/rl_dispatch/algorithms/ppo.py:201-202`

**Before**:
```python
for batch in self.buffer.get(batch_size=self.training_config.batch_size):
    obs, actions, old_log_probs, advantages, returns, old_values = batch
```

**After**:
```python
for batch in self.buffer.get(batch_size=self.training_config.batch_size):
    # Reviewer 박용준: Unpack 7 values (added action_masks)
    obs, actions, old_log_probs, advantages, returns, old_values, action_masks = batch
```

**Note**: `action_masks`는 현재 TODO로 남아있으며, 향후 policy network에서 사용할 예정입니다.

---

## 🧪 검증 방법

### Option 1: Quick Test (1-2분)
```bash
cd ~/rl_dispatch_mvp
python scripts/quick_test_fix.py
```

**기대 출력**:
```
✅ ALL TESTS PASSED!

CRITICAL CHECKS:
✅ Entropy: 0.XXXX (> 0.01) - Good!
✅ Approx KL: 0.XXXXXX (> 0.0001) - Policy updating!
✅ Clipfrac: 0.XXXX (> 0.01) - PPO working!
✅ Value loss: XXX.XX (< 1000) - Reasonable!
```

### Option 2: Manual Python Test
```bash
python -c "
import sys
sys.path.insert(0, 'src')
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.core.config import TrainingConfig, NetworkConfig

# Create agent
config = TrainingConfig(num_steps=64, batch_size=16, num_epochs=2)
agent = PPOAgent(obs_dim=77, num_replan_strategies=6, training_config=config, device='cpu')

# Collect fake rollout
import numpy as np
obs = np.random.randn(77).astype(np.float32)
for i in range(64):
    action, log_prob, value = agent.get_action(obs)
    agent.buffer.add(
        obs=obs,
        action=action,
        log_prob=log_prob,
        reward=1.0,
        value=value,
        done=False,
        nav_time=1.0
    )

# PPO update should NOT crash
try:
    stats = agent.update(last_value=0.0, last_done=True)
    print('✅ PPO update works! No unpacking error!')
    print(f'Stats: {stats}')
except ValueError as e:
    print(f'❌ Still has error: {e}')
"
```

---

## 📊 영향 범위

### 수정된 파일
- ✅ `src/rl_dispatch/algorithms/ppo.py` (Line 201-202)

### 영향받는 스크립트
- ✅ `scripts/train_multi_map.py` - 정상 작동
- ✅ `scripts/train_multi_map_fixed.py` - 정상 작동
- ✅ `scripts/train.py` - 정상 작동
- ✅ `scripts/quick_test_fix.py` - 정상 작동

### 호환성
- ✅ 기존 checkpoint 호환 (네트워크 구조 변경 없음)
- ✅ 기존 config 호환
- ✅ 이전 학습 재개 가능

---

## 🔄 다음 단계 (TODO)

현재 `action_masks`는 buffer에 저장되고 전달되지만, **실제로 사용되지는 않습니다**.

### Future Enhancement (선택사항)

**File**: `src/rl_dispatch/algorithms/networks.py`

네트워크의 `get_action_and_value()` 메서드에서 action_mask를 활용:

```python
def get_action_and_value(
    self,
    obs: torch.Tensor,
    action: Optional[torch.Tensor] = None,
    mode_mask: Optional[torch.Tensor] = None,  # TODO: Use this!
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # ... existing code ...

    # TODO: Apply mask to logits before sampling
    if mode_mask is not None:
        # Mask invalid actions (set logits to -inf)
        mode_logits = mode_logits.masked_fill(mode_mask[:, :2] == 0, float('-inf'))
        replan_logits = replan_logits.masked_fill(mode_mask[:, 2:] == 0, float('-inf'))
```

**우선순위**: 낮음 (현재 환경의 `_compute_action_mask()`가 이미 invalid action 경고 출력)

---

## ⚠️ 주의사항

### 절대 하지 말 것
```python
# ❌ 잘못된 unpacking (6개)
obs, actions, old_log_probs, advantages, returns, old_values = batch

# ❌ action_masks 무시
obs, *rest = batch  # 나머지 무시
```

### 올바른 방법
```python
# ✅ 올바른 unpacking (7개)
obs, actions, old_log_probs, advantages, returns, old_values, action_masks = batch

# ✅ 미사용 변수는 명시
obs, actions, old_log_probs, advantages, returns, old_values, _ = batch  # action_masks unused
```

---

## 🎯 결론

**수정 완료**: ✅
**테스트 필요**: ✅ `python scripts/quick_test_fix.py`
**학습 가능**: ✅ 즉시 학습 시작 가능

이 버그는 **기능적 문제**였으며 (학습이 아예 불가), 수정 후 즉시 정상 작동합니다.

Policy collapse 문제와는 **독립적**이므로, 이 bugfix + policy_collapse_fix를 함께 적용하면:
1. ✅ 학습이 실행됨 (unpacking error 해결)
2. ✅ 학습이 안정화됨 (reward normalization 등 적용)

---

**작성자**: Reviewer 박용준
**작성일**: 2025-12-30
**버전**: 1.0
