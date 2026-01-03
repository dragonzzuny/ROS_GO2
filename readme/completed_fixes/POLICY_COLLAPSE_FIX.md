# Policy Collapse 문제 진단 및 해결 가이드

**작성자**: Reviewer 박용준
**작성일**: 2025-12-30
**문제**: Global step 245k→491k 동안 성능 악화 및 정책 붕괴

---

## 🔥 관측된 증상

### 1. 성능 지표 악화
- **Success Rate**: 55-59% → 29-36% (지속 하락)
- **Return**: 점점 더 음수로 악화
- **Coverage**: 10-25% 정체 (낮음)

### 2. 학습 지표 이상
- **Entropy**: 0.0011 → 0.0000 (탐색 붕괴)
- **Approx KL**: ≈0 (정책 업데이트 멈춤)
- **Clipfrac**: 0 (PPO clipping 작동 안 함)
- **Policy Loss**: ≈0 (업데이트 효과 없음)

### 3. Critic 붕괴
- **Value Loss**: 2.5M~3.1M (매우 높음!)
- **Explained Variance**: 0.005~0.01 (target을 전혀 설명 못함)

---

## 🔍 근본 원인 분석

### ✅ 발견된 버그 (Critical)

#### 1. **Reward Normalization 미구현** ⭐⭐⭐
**위치**: `src/rl_dispatch/core/config.py:355`, `scripts/train_multi_map.py`

**문제**:
```python
# config.py에 설정만 있음
normalize_rewards: bool = True
clip_rewards: float = 10.0

# 실제 train_multi_map.py에서 사용 안 함!
agent.buffer.add(
    reward=reward,  # Raw reward를 그대로 사용
    ...
)
```

**영향**:
- Raw reward scale이 매우 큼 (-200 ~ +100 범위)
- Value function이 huge target 학습 시도 → value_loss 폭주 (2.5M+)
- Advantage normalization도 불안정
- Critic이 폭주하면 policy gradient도 엉망됨

**해결**:
```python
# Welford's online algorithm으로 running mean/std 계산
reward_normalizer = RunningMeanStd()
reward_normalizer.update([reward])
normalized_reward = (reward - mean) / sqrt(var + eps)
normalized_reward = np.clip(normalized_reward, -5.0, 5.0)
```

---

#### 2. **Entropy Coefficient 너무 낮음** ⭐⭐
**위치**: `src/rl_dispatch/core/config.py:348`

**문제**:
```python
entropy_coef: float = 0.01  # Too low!
```

**영향**:
- 초반부터 탐색이 부족
- 정책이 빠르게 deterministic으로 수렴 (entropy → 0)
- Local optima에 갇힘 (계속 patrol만 or dispatch만)
- Approx KL = 0 → 정책 업데이트가 멈춤

**해결**:
```python
# 초반: 0.1 (탐색 강화)
# 후반: 0.01 (exploitation)
# Annealing 적용
progress = update / num_updates
entropy_coef = 0.1 * (1 - progress) + 0.01 * progress
```

---

#### 3. **Learning Rate 너무 높음** ⭐
**위치**: `scripts/train_multi_map.py:230`

**문제**:
```python
default=3e-4  # 초기값이 높음
```

**영향**:
- 불안정한 학습 (value loss 진동)
- Policy collapse 가속화
- Entropy가 급격히 0으로 수렴

**해결**:
```python
default=1e-4  # 3e-4 → 1e-4 (3배 감소)
# 또는 LR annealing 강화
```

---

#### 4. **진단 로깅 부족** ⭐
**위치**: `scripts/train_multi_map.py` (전반)

**문제**:
- Valid action count 로그 없음 → action mask 문제 감지 불가
- Advantage/return stats 없음 → 보상 스케일 문제 감지 불가
- Reward normalization stats 없음 → normalize 작동 확인 불가
- Action mode distribution 없음 → "출동만" or "순찰만" 감지 불가

**해결**:
```python
# 필수 로그 추가:
- diagnostics/valid_action_count_mean
- diagnostics/valid_action_count_min
- diagnostics/advantage_mean, advantage_std
- diagnostics/reward_mean_raw, reward_std_raw
- diagnostics/reward_mean_normalized
- diagnostics/action_patrol_ratio, action_dispatch_ratio
- diagnostics/value_return_gap
```

---

### ⚠️ 의심되는 문제 (검증 필요)

#### 5. **Action Masking Shape 불일치 가능**
**위치**: `src/rl_dispatch/env/patrol_env.py:539`

**의심**:
```python
# patrol_env.py
def _compute_action_mask(self) -> np.ndarray:
    mask = np.ones((2, num_candidates), dtype=np.float32)
    # ... masking logic ...
    return mask  # Shape: (2, num_candidates)

# buffer.py
self.action_masks = np.ones((buffer_size, 20), dtype=np.float32)
```

**문제**:
- Env가 (2, K) shape 반환
- Buffer는 (20,) 고정 크기 기대
- Flatten/reshape 로직 불일치 가능

**검증 방법**:
```python
# train loop에서 로그 추가
action_mask = info.get("action_mask")
print(f"Action mask shape: {action_mask.shape}")
print(f"Valid actions: {np.sum(action_mask > 0.5)}")
```

**임시 해결** (검증 전):
```python
# patrol_env.py에서 flatten
mask = mask.flatten()  # (2*num_candidates,)
return mask
```

---

#### 6. **Value Loss Coefficient 너무 낮음**
**위치**: `src/rl_dispatch/core/config.py:349`

**현재**:
```python
value_loss_coef: float = 0.5
```

**문제**:
- Reward scale이 크면 critic이 제대로 학습 못함
- Value loss가 policy loss에 비해 너무 약함

**해결**:
```python
value_loss_coef: float = 1.0  # 0.5 → 1.0
```

---

## 🛠️ 적용된 수정 사항

### `train_multi_map_fixed.py`에 구현된 개선사항

#### 1. **Reward Normalization 완전 구현** ✅
```python
class RunningMeanStd:
    """Welford's online algorithm"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        # Online mean/var update
        ...

# Train loop에서 사용
reward_normalizer = RunningMeanStd()

for step in range(num_steps):
    next_obs, reward, done, trunc, info = env.step(action)

    # Normalize reward
    reward_normalizer.update(np.array([reward]))
    normalized_reward = (reward - reward_normalizer.mean) / np.sqrt(reward_normalizer.var + 1e-8)
    normalized_reward = np.clip(normalized_reward, -5.0, 5.0)

    # Store normalized reward
    agent.buffer.add(reward=normalized_reward, ...)
```

#### 2. **Entropy Annealing** ✅
```python
# 초반: 탐색 강화 (0.1)
# 후반: exploitation (0.01)
for update in range(num_updates):
    progress = update / num_updates
    current_entropy_coef = 0.1 * (1.0 - progress) + 0.01 * progress
    agent.training_config.entropy_coef = current_entropy_coef
```

#### 3. **개선된 하이퍼파라미터** ✅
```python
parser.add_argument("--learning-rate", type=float, default=1e-4)  # 3e-4 → 1e-4
parser.add_argument("--entropy-coef", type=float, default=0.1)    # 0.01 → 0.1 (annealed)
parser.add_argument("--value-loss-coef", type=float, default=1.0) # 0.5 → 1.0
parser.add_argument("--save-interval", type=int, default=50)      # 100 → 50
```

#### 4. **포괄적인 진단 로깅** ✅
```python
# Reward statistics
writer.add_scalar("diagnostics/reward_mean_raw", np.mean(raw_rewards), global_step)
writer.add_scalar("diagnostics/reward_std_raw", np.std(raw_rewards), global_step)
writer.add_scalar("diagnostics/reward_mean_normalized", reward_normalizer.mean, global_step)

# Action distribution
writer.add_scalar("diagnostics/action_patrol_ratio", patrol_ratio, global_step)
writer.add_scalar("diagnostics/action_dispatch_ratio", dispatch_ratio, global_step)

# Valid actions
writer.add_scalar("diagnostics/valid_action_count_mean", np.mean(valid_actions), global_step)
writer.add_scalar("diagnostics/valid_action_count_min", np.min(valid_actions), global_step)

# Advantage/value/return
writer.add_scalar("diagnostics/advantage_mean", np.mean(advantages), global_step)
writer.add_scalar("diagnostics/advantage_std", np.std(advantages), global_step)
writer.add_scalar("diagnostics/value_mean", np.mean(values), global_step)
writer.add_scalar("diagnostics/return_mean", np.mean(returns), global_step)
writer.add_scalar("diagnostics/value_return_gap", np.mean(returns - values), global_step)

# Entropy coefficient (annealing)
writer.add_scalar("train/entropy_coef", current_entropy_coef, global_step)
```

---

## 🚀 실행 방법

### 1. 빠른 테스트 (10K steps, ~1분)
```bash
python scripts/train_multi_map_fixed.py \
    --total-timesteps 10000 \
    --seed 42 \
    --log-interval 1

# 확인 사항:
# - 에러 없이 실행됨
# - diagnostics/* 로그 생성됨
# - entropy가 0으로 붕괴하지 않음
# - value_loss가 초기 값보다 감소함
```

### 2. 짧은 학습 (100K steps, ~5-10분)
```bash
python scripts/train_multi_map_fixed.py \
    --total-timesteps 100000 \
    --seed 42 \
    --log-interval 5 \
    --save-interval 20

# TensorBoard로 모니터링
tensorboard --logdir runs/multi_map_ppo_fixed
```

### 3. 전체 학습 (5M steps, ~1-3시간)
```bash
python scripts/train_multi_map_fixed.py \
    --total-timesteps 5000000 \
    --seed 42 \
    --cuda \
    --log-interval 10 \
    --save-interval 50
```

---

## 📊 기대 결과

### 초기 (0-100K steps)
- ✅ Entropy: **0.08-0.10** 유지 (0으로 붕괴 안 함)
- ✅ Approx KL: **0.005-0.02** (0이 아님, 정책 업데이트 작동)
- ✅ Clipfrac: **0.05-0.15** (PPO clipping 작동)
- ✅ Value Loss: **<1M** (2.5M+에서 감소)
- ✅ Explained Variance: **>0.1** (0.005에서 증가)

### 중기 (100K-500K steps)
- ✅ Success Rate: **상승 또는 정체** (하락 멈춤)
- ✅ Return: **최소 정체** (계속 악화 멈춤)
- ✅ Coverage: **점진적 증가** (10% → 30%+)
- ✅ Entropy: **0.06-0.08** (annealing으로 천천히 감소)

### 후기 (500K-5M steps)
- ✅ Success Rate: **60-75%+**
- ✅ Return: **양수 또는 -1000 이상**
- ✅ Coverage: **50-80%+**
- ✅ Entropy: **0.02-0.04** (exploitation으로 수렴, 0은 아님)
- ✅ Explained Variance: **>0.5**

---

## 🔍 TensorBoard 모니터링 가이드

### 필수 확인 그래프

#### 1. **train/entropy** (가장 중요!)
```
정상: 0.08~0.10 → 0.06~0.08 → 0.02~0.04 (서서히 감소)
비정상: 0.01 → 0.001 → 0.0000 (급격히 0으로 붕괴)
```

#### 2. **train/approx_kl**
```
정상: 0.005~0.02 (적당한 변동)
비정상: 0.0000... (정책 업데이트 멈춤)
```

#### 3. **train/clipfrac**
```
정상: 0.05~0.20 (PPO clipping 작동)
비정상: 0.000 (clipping 안 일어남 = 업데이트 없음)
```

#### 4. **train/value_loss**
```
정상: 초기 높음 → 점진적 감소 → 안정화
비정상: 2.5M+ 유지 또는 계속 증가
```

#### 5. **train/explained_variance**
```
정상: 초기 낮음 → 0.5+ 증가
비정상: 0.005~0.01 정체 (critic이 target 설명 못함)
```

#### 6. **diagnostics/valid_action_count_mean**
```
정상: 10-20 (충분한 선택지)
비정상: 1-2 (action mask가 너무 제한적)
```

#### 7. **diagnostics/action_patrol_ratio vs action_dispatch_ratio**
```
정상: 균형잡힌 분포 (예: 70% patrol, 30% dispatch)
비정상: 한쪽으로 치우침 (예: 99% patrol, 1% dispatch)
```

#### 8. **diagnostics/reward_mean_normalized**
```
정상: 0 근처에서 안정화 (normalization 작동)
비정상: 큰 값 유지 (normalization 안 됨)
```

---

## 🐛 추가 디버깅 팁

### 1. **Entropy가 여전히 0으로 붕괴한다면**
```python
# entropy_coef를 더 높이거나 annealing 속도 줄임
entropy_coef = 0.15 * (1 - progress**0.5) + 0.02 * progress**0.5
# 또는 고정값
entropy_coef = 0.05  # No annealing
```

### 2. **Value loss가 여전히 크다면**
```python
# Reward scale 문제일 가능성
# 1. clip_rewards를 더 낮춤
clip_rewards = 3.0  # 5.0 → 3.0

# 2. 또는 reward config 자체를 조정
reward_config = RewardConfig(
    event_response_bonus=25.0,  # 50.0 → 25.0
    collision_penalty=-50.0,     # -100.0 → -50.0
    # ...
)
```

### 3. **Success rate가 계속 낮다면**
```python
# Event reward weight 증가
reward_config.w_event = 2.0  # 1.0 → 2.0
reward_config.w_patrol = 0.3  # 0.5 → 0.3
```

### 4. **Coverage가 낮다면**
```python
# Patrol reward weight 증가
reward_config.w_patrol = 1.0  # 0.5 → 1.0
reward_config.w_event = 0.8   # 1.0 → 0.8
```

---

## 📌 체크리스트

학습 시작 전:
- [ ] `train_multi_map_fixed.py` 사용 (기존 스크립트 아님!)
- [ ] `--log-interval 1` 또는 `10` 설정 (자주 로깅)
- [ ] TensorBoard 실행: `tensorboard --logdir runs`

학습 중 (매 100K steps마다):
- [ ] `train/entropy` > 0.02 확인
- [ ] `train/approx_kl` > 0.001 확인
- [ ] `train/clipfrac` > 0.02 확인
- [ ] `train/value_loss` < 1M 확인
- [ ] `train/explained_variance` > 0.1 확인
- [ ] `diagnostics/valid_action_count_mean` > 5 확인
- [ ] `episode_per_map/*/return` 하락 멈춤 확인

---

## 🎯 성공 기준

### Minimum Acceptable Performance (MAP)
- Entropy: **> 0.015** (전 구간)
- Approx KL: **> 0.001** (업데이트 작동)
- Value Loss: **< 1M** (안정화)
- Explained Variance: **> 0.2**
- Success Rate: **하락 멈춤 + 정체 or 상승**

### Target Performance (TP)
- Entropy: **0.03-0.08** (annealing curve)
- Success Rate: **60%+**
- Coverage: **50%+**
- Return: **-3000 이상**

---

## 📞 문제 발생 시

1. **TensorBoard 스크린샷 캡처** (특히 entropy, approx_kl, value_loss)
2. **Console 출력 복사** (Diagnostics 섹션)
3. **실행 명령어 기록**
4. GitHub Issue 또는 팀에 공유

---

**작성자**: Reviewer 박용준
**최종 수정**: 2025-12-30
**버전**: 1.0
