# 📊 포괄적 로깅 시스템

**목적**: 학습 과정의 모든 데이터를 빠짐없이 기록하여 문제 진단 및 성능 분석 가능

**작성자**: Reviewer 박용준
**작성일**: 2025-12-30

---

## 🎯 로깅 철학

> **"측정할 수 없으면 개선할 수 없다"**

모든 학습 데이터를 다음 3가지 형식으로 저장:
1. **TensorBoard** - 실시간 모니터링
2. **CSV** - 상세 분석 및 플롯
3. **JSON** - 설정 및 최종 결과

---

## 📁 디렉토리 구조

```
runs/multi_map_logged/20251230-123456/
├── tensorboard/              # TensorBoard 로그
│   └── events.out.tfevents.*
├── csv/                      # CSV 데이터
│   ├── steps.csv            # Step-level 데이터
│   ├── episodes.csv         # Episode-level 데이터
│   ├── updates.csv          # Update-level 데이터
│   ├── map_large_square.csv # 맵별 데이터
│   ├── map_corridor.csv
│   └── ...
├── checkpoints/              # 모델 체크포인트
│   ├── update_50.pth
│   ├── update_100.pth
│   └── final.pth
├── coverage/                 # 커버리지 히트맵
│   ├── update_50/
│   └── update_100/
├── analysis/                 # 분석 결과
│   ├── learning_curves.png
│   ├── training_diagnostics.png
│   ├── map_comparison.png
│   └── training_report.txt
├── training_config.yaml      # 학습 설정
└── results.json              # 최종 결과
```

---

## 📊 로깅 레벨

### 1. **Step-Level** (매 step)

**파일**: `csv/steps.csv`

**기록 데이터**:
```
global_step, update, episode, map_name,
action_mode, action_replan_idx,
reward_total, reward_event, reward_patrol, reward_safety, reward_efficiency,
has_event, nav_time, battery_level,
valid_action_count, patrol_valid, dispatch_valid,
collision, nav_success, event_resolved
```

**용도**:
- 세밀한 행동 분석
- Reward component 분석
- Action masking 검증
- 배터리 사용 패턴

**크기**: ~10MB per 100K steps

---

### 2. **Episode-Level** (에피소드 종료 시)

**파일**: `csv/episodes.csv`

**기록 데이터**:
```
episode, global_step, map_name,
return, length, duration,
event_success_rate, patrol_coverage, safety_violations,
avg_reward_event, avg_reward_patrol, avg_reward_safety, avg_reward_efficiency,
patrol_ratio, dispatch_ratio,
avg_nav_time, final_battery,
events_detected, events_responded, events_successful
```

**용도**:
- 학습 곡선 플롯
- 성능 추세 분석
- 에피소드 통계

**크기**: ~1MB per 10K episodes

---

### 3. **Update-Level** (PPO 업데이트마다)

**파일**: `csv/updates.csv`

**기록 데이터**:
```
update, global_step,
policy_loss, value_loss, entropy, approx_kl, clipfrac, explained_variance,
entropy_coef, learning_rate,
advantage_mean, advantage_std, advantage_min, advantage_max,
value_mean, return_mean, value_return_gap,
reward_raw_mean, reward_raw_std, reward_normalized_mean, reward_normalized_std,
grad_norm, fps
```

**용도**:
- PPO 알고리즘 진단
- Policy collapse 감지
- Critic 성능 평가
- Hyperparameter 영향 분석

**크기**: ~100KB per 2K updates

---

### 4. **Map-Level** (맵별 에피소드)

**파일**: `csv/map_{name}.csv`

**기록 데이터**:
```
episode, global_step,
return, length,
event_success_rate, patrol_coverage,
patrol_ratio, dispatch_ratio
```

**용도**:
- 맵별 성능 비교
- 일반화 능력 평가
- 맵 난이도 분석

**크기**: ~100KB per map

---

## 🚀 사용법

### 1단계: 학습 시작
```bash
python scripts/train_with_logging.py \
    --total-timesteps 5000000 \
    --seed 42 \
    --cuda \
    --experiment-name my_experiment
```

**실시간 모니터링**:
```bash
# 별도 터미널
tensorboard --logdir runs/my_experiment
# http://localhost:6006
```

---

### 2단계: 학습 중 확인

**TensorBoard에서 실시간 확인**:
- `episode/return` - 리턴 추세
- `episode/event_success_rate` - 성공률
- `train/entropy` - 탐색 상태
- `train/value_loss` - Critic 학습
- `diagnostics/*` - 상세 진단

**Console 출력**:
```
Update 100/2441 (Step 204,800):
  FPS: 1234.5
  Entropy coef: 0.0959

  Per-Map Performance:
    map_large_square: Return=523.4±45.2, Success=87.3%, Cov=94.2%
    map_office_building: Return=-234.1±67.8, Success=72.1%, Cov=88.6%
    ...

  Training:
    policy_loss: 0.0012
    value_loss: 12345.67
    entropy: 0.0456
    approx_kl: 0.0123
    explained_variance: 0.7890
```

---

### 3단계: 학습 후 분석

```bash
# 로그 분석 및 플롯 생성
python scripts/analyze_logs.py runs/my_experiment/20251230-123456

# 출력:
# ✅ Loaded 15234 episodes
# ✅ Loaded 2441 updates
# ✅ Loaded 5000000 steps
# ✅ Loaded map: map_large_square (2534 episodes)
# ...
# ✅ Saved: runs/.../analysis/learning_curves.png
# ✅ Saved: runs/.../analysis/training_diagnostics.png
# ✅ Saved: runs/.../analysis/map_comparison.png
# ✅ Saved: runs/.../analysis/training_report.txt
```

---

## 📈 생성되는 플롯

### 1. **learning_curves.png**

6개 서브플롯:
1. **Episode Return** - 학습 진행 상황
2. **Event Success Rate** - 이벤트 대응 성능
3. **Patrol Coverage** - 순찰 커버리지
4. **Action Distribution** - Patrol/Dispatch 비율
5. **Episode Length** - 에피소드 길이
6. **Safety Violations** - 안전 위반 추세

**해석**:
- Return이 상승 → 학습 성공
- Success rate > 75% → 목표 달성
- Coverage > 80% → 순찰 효율적
- Action distribution 균형 → 정상 학습

---

### 2. **training_diagnostics.png**

6개 PPO 진단 플롯:
1. **Entropy** - 탐색 vs exploitation
2. **Approx KL** - 정책 변화 크기
3. **Value Loss** - Critic 학습 상태
4. **Explained Variance** - Critic 성능
5. **Policy Loss** - Actor 학습 상태
6. **Clipfrac** - PPO clipping 작동

**경고 신호**:
- ❌ Entropy < 0.02 → Policy collapse
- ❌ Approx KL ≈ 0 → 정책 업데이트 멈춤
- ❌ Value Loss > 100K → Critic 폭주
- ❌ Explained Var < 0.1 → Critic 부정확
- ❌ Clipfrac < 0.05 → PPO 작동 안 함

---

### 3. **map_comparison.png**

4개 맵별 비교 플롯:
1. **Average Return** - 맵별 평균 리턴
2. **Success Rate** - 맵별 성공률
3. **Coverage** - 맵별 커버리지
4. **Learning Curves** - 맵별 학습 곡선

**해석**:
- 모든 맵에서 양수 리턴 → 일반화 성공
- 맵 간 성능 차이 작음 → 균형 학습
- 어려운 맵 성능 낮음 → 정상 (expected)

---

### 4. **training_report.txt**

텍스트 요약:
```
================================================================================
학습 결과 리포트
================================================================================

학습 시간: 2.45 hours
총 Steps: 5,000,000
총 Updates: 2,441

================================================================================
전체 성능 (최근 100 에피소드)
================================================================================
  평균 Return: 1234.56 ± 456.78
  Event Success Rate: 78.9%
  Patrol Coverage: 85.3%
  Safety Violations: 0.12 per episode
  Patrol Ratio: 67.8%
  Dispatch Ratio: 32.2%

================================================================================
맵별 성능 (최근 100 에피소드)
================================================================================

map_large_square:
  Episodes: 2534
  Return: 1523.4 ± 345.2
  Success: 87.3%
  Coverage: 94.2%

...

================================================================================
학습 상태 진단 (최근 100 updates)
================================================================================
  Entropy: 0.034567 (healthy: > 0.02)
  Approx KL: 0.012345 (healthy: > 0.001)
  Value Loss: 45678.90 (healthy: < 100K)
  Explained Variance: 0.6789 (good: > 0.5)
  Clipfrac: 0.1234 (healthy: > 0.05)

⚠️  경고 사항:
  없음 - 학습 정상 진행 중 ✅
```

---

## 🔍 문제 진단 가이드

### Case 1: Policy Collapse

**증상**:
- Entropy < 0.02
- Approx KL ≈ 0
- Success rate 하락

**확인**:
```bash
# updates.csv 확인
python -c "
import pandas as pd
df = pd.read_csv('runs/.../csv/updates.csv')
print('Recent entropy:', df['entropy'].tail(100).mean())
print('Recent approx_kl:', df['approx_kl'].tail(100).mean())
"
```

**해결**:
- Entropy coefficient 증가 (0.1 → 0.15)
- Learning rate 감소
- `POLICY_COLLAPSE_FIX.md` 참고

---

### Case 2: Value Loss Explosion

**증상**:
- Value loss > 100K
- Explained variance < 0.1
- 학습 불안정

**확인**:
```bash
# updates.csv 확인
python -c "
import pandas as pd
df = pd.read_csv('runs/.../csv/updates.csv')
print('Recent value_loss:', df['value_loss'].tail(100).mean())
print('Recent explained_var:', df['explained_variance'].tail(100).mean())
"
```

**해결**:
- Reward normalization 확인
- Clip rewards 감소 (5.0 → 3.0)
- Value loss coefficient 증가 (1.0 → 2.0)

---

### Case 3: Invalid Action 많음

**증상**:
- Console warning 폭발
- Patrol ratio 100%
- Dispatch ratio 0%

**확인**:
```bash
# steps.csv 확인
python -c "
import pandas as pd
df = pd.read_csv('runs/.../csv/steps.csv')
print('Avg valid actions:', df['valid_action_count'].mean())
print('Dispatch valid rate:', df['dispatch_valid'].mean())
"
```

**해결**:
- Action masking 적용 확인
- `ACTION_MASKING_FIX.md` 참고

---

## 💾 데이터 크기 예상

| 데이터 | 5M steps | 10M steps |
|--------|----------|-----------|
| steps.csv | ~500MB | ~1GB |
| episodes.csv | ~5MB | ~10MB |
| updates.csv | ~250KB | ~500KB |
| map CSVs (6개) | ~3MB | ~6MB |
| **Total CSV** | **~508MB** | **~1.02GB** |
| TensorBoard | ~100MB | ~200MB |
| **Total** | **~608MB** | **~1.22GB** |

**디스크 공간**: 최소 2GB 여유 권장

---

## 🎯 Best Practices

### 1. **항상 TensorBoard 실행**
```bash
# 학습 전에 미리 실행
tensorboard --logdir runs --port 6006
```

### 2. **정기적으로 analyze_logs 실행**
```bash
# 100K steps마다
python scripts/analyze_logs.py runs/my_experiment/20251230-123456
```

### 3. **Checkpoint 주기적 백업**
```bash
# 중요한 checkpoint 백업
cp runs/.../checkpoints/update_500.pth backups/
```

### 4. **CSV 파일 압축 보관**
```bash
# 학습 완료 후
cd runs/my_experiment/20251230-123456
tar -czf logs.tar.gz csv/
```

---

## 📊 TensorBoard 주요 Metrics

### 실시간 모니터링 (필수)

| Metric | 위치 | 정상 범위 | 경고 |
|--------|------|----------|------|
| **Entropy** | `train/entropy` | 0.03-0.10 | < 0.02 |
| **Approx KL** | `train/approx_kl` | 0.001-0.02 | < 0.001 |
| **Value Loss** | `train/value_loss` | < 100K | > 100K |
| **Success Rate** | `episode/event_success_rate` | 증가 추세 | 하락 |
| **Return** | `episode/return` | 증가 추세 | 하락 |

### 진단용 (주기적 확인)

| Metric | 위치 | 의미 |
|--------|------|------|
| **Explained Variance** | `train/explained_variance` | Critic 성능 |
| **Clipfrac** | `train/clipfrac` | PPO 작동 |
| **Patrol Ratio** | `episode/patrol_ratio` | 행동 분포 |
| **Valid Actions** | `diagnostics/valid_action_count_mean` | Masking 작동 |
| **Reward Components** | `step/{map}/reward` | Reward 분해 |

---

## 🔧 커스터마이징

### 추가 Metric 로깅

`train_with_logging.py` 수정:

```python
# Step-level에 새 metric 추가
step_data['my_custom_metric'] = calculate_custom_metric()

# TensorBoard에 로깅
self.writer.add_scalar("custom/my_metric", value, global_step)

# CSV에 추가 (header 먼저 수정)
self.step_writer.writerow([..., step_data['my_custom_metric']])
```

---

## ✅ 체크리스트

학습 시작 전:
- [ ] `tensorboard --logdir runs` 실행
- [ ] 디스크 공간 2GB+ 확보
- [ ] Experiment name 설정

학습 중 (매 100K steps):
- [ ] TensorBoard에서 entropy 확인 (> 0.02)
- [ ] Value loss 확인 (< 100K)
- [ ] Success rate 추세 확인
- [ ] Console warning 확인 (없어야 함)

학습 후:
- [ ] `analyze_logs.py` 실행
- [ ] `training_report.txt` 읽기
- [ ] 플롯 확인 (learning_curves.png 등)
- [ ] 중요 checkpoint 백업

---

## 📞 문제 해결

### CSV 파일이 너무 큼
```bash
# Step CSV 샘플링 (매 N번째 step만)
python -c "
import pandas as pd
df = pd.read_csv('steps.csv')
df_sampled = df.iloc[::100]  # 100 step마다
df_sampled.to_csv('steps_sampled.csv', index=False)
"
```

### TensorBoard가 느림
```bash
# 특정 run만 로드
tensorboard --logdir runs/my_experiment/20251230-123456
```

### 분석 스크립트 에러
```bash
# Matplotlib 설치
pip install matplotlib pandas

# 다시 실행
python scripts/analyze_logs.py runs/.../
```

---

**작성자**: Reviewer 박용준
**최종 수정**: 2025-12-30
**버전**: 1.0
