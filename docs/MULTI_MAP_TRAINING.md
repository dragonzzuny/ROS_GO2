# Multi-Map Training for Policy Generalization

**목표: 하나의 맵에서만 최적화된 것이 아닌, 어떤 맵이 주어져도 좋은 성능을 내는 일반화된 정책 학습**

---

## 🎯 왜 멀티맵 학습인가?

### 문제: 단일 맵 학습의 한계

기존 방식 (단일 맵 학습):
```bash
python scripts/train.py --config configs/default.yaml
```

**문제점:**
- ❌ 학습한 맵에서만 잘 작동
- ❌ 맵 구조를 "암기"하여 과적합
- ❌ 새로운 환경에서 성능 급격히 하락
- ❌ 실제 배포 시 맵이 바뀌면 재학습 필요

### 해결: 멀티맵 학습

```bash
python scripts/train_multi_map.py  # 6개 다양한 맵에서 학습
```

**장점:**
- ✅ 다양한 맵에서 좋은 성능
- ✅ 맵 구조 대신 "전략"을 학습
- ✅ 새로운 환경에서도 일반화
- ✅ 실전 배포 즉시 가능

---

## 📊 구현된 6가지 다양한 맵

| 맵 | 크기 | 평수 | 포인트 | 특징 | 난이도 |
|----|------|------|--------|------|--------|
| **large_square** | 100×100m | 3,025평 | 12 | 넓은 개방 공간 | ⭐⭐⭐ |
| **corridor** | 120×30m | 1,090평 | 10 | 긴 직선 복도 | ⭐⭐ |
| **l_shaped** | 80×80m | 1,450평 | 10 | L자 비정형 구조 | ⭐⭐⭐ |
| **office_building** | 90×70m | 1,900평 | 14 | 복잡한 실내 | ⭐⭐⭐⭐ |
| **warehouse** | 140×100m | 4,235평 | 12 | 산업 시설 | ⭐⭐⭐⭐ |
| **campus** | 150×120m | 5,445평 | 16 | 대규모 야외 | ⭐⭐⭐⭐⭐ |

**다양성:**
- 크기: 3,600㎡ ~ 18,000㎡ (1,090평 ~ 5,445평)
- 형태: 정사각형, 직선, L자형, 복합 구조
- 환경: 실내 vs 야외
- 복잡도: 4개 포인트 ~ 16개 포인트

---

## 🚀 사용 방법

### 1. 기본 멀티맵 학습

```bash
python scripts/train_multi_map.py \
    --total-timesteps 5000000 \
    --seed 42 \
    --cuda
```

**동작:**
- 에피소드마다 6개 맵 중 하나를 **랜덤 선택**
- 모든 맵에서 균등하게 학습
- 맵별 성능 자동 추적

**결과 (TensorBoard):**
- `episode_per_map/map_large_square/return` - 맵별 리턴
- `episode_per_map/map_office_building/event_success_rate` - 맵별 성공률
- `episode/return` - 전체 평균 리턴

---

### 2. 특정 맵들만 선택

```bash
# 실내 환경만
python scripts/train_multi_map.py \
    --maps \
        configs/map_office_building.yaml \
        configs/map_corridor.yaml \
        configs/map_l_shaped.yaml

# 야외 + 창고
python scripts/train_multi_map.py \
    --maps \
        configs/map_campus.yaml \
        configs/map_warehouse.yaml
```

---

### 3. 커리큘럼 학습 (쉬운 맵 → 어려운 맵)

```bash
python scripts/train_multi_map.py \
    --map-mode curriculum \
    --total-timesteps 10000000 \
    --seed 42 \
    --cuda
```

**동작:**
- 초반 100 에피소드: `corridor` (가장 쉬움)
- 100-200 에피소드: `corridor`, `l_shaped`
- 200-300 에피소드: `corridor`, `l_shaped`, `large_square`
- ...
- 600+ 에피소드: 모든 맵

**장점:**
- 초기 안정적인 학습
- 점진적 난이도 증가
- 더 빠른 수렴

---

### 4. 순차 학습

```bash
python scripts/train_multi_map.py \
    --map-mode sequential
```

**동작:**
- 에피소드 1: map 1
- 에피소드 2: map 2
- ...
- 에피소드 7: map 1 (반복)

**용도:**
- 각 맵에서 균등한 에피소드 수 보장
- 디버깅 및 분석

---

## 📈 일반화 성능 평가

### 1. 여러 맵에서 평가

```bash
python scripts/evaluate_generalization.py \
    --model checkpoints/multi_map_ppo/20250101-120000/checkpoints/final.pth \
    --episodes 50 \
    --save-json
```

**출력:**

```
================================================================================
Generalization Evaluation Results
================================================================================
Map                       Return               Event Success        Patrol Coverage
------------------------------------------------------------------------------------
map_large_square          523.4 ± 45.2        87.3% ± 8.1%        94.2% ± 3.5%
map_corridor              -112.3 ± 23.5       78.4% ± 6.2%        96.1% ± 2.1%
map_l_shaped              312.8 ± 56.7        82.1% ± 9.3%        89.5% ± 4.8%
map_office_building       -234.1 ± 67.8       72.1% ± 11.2%       88.6% ± 5.2%
map_warehouse             201.5 ± 78.3        80.5% ± 10.1%       92.3% ± 3.9%
map_campus                412.5 ± 89.3        85.4% ± 7.7%        91.3% ± 4.8%
------------------------------------------------------------------------------------

Overall Statistics
  Average Return: 183.8 (std across maps: 262.1)
  Average Event Success: 80.9% (std: 5.2%)
  Average Patrol Coverage: 92.0% (std: 2.8%)

Generalization Metrics
  Return Variance: 68723.5 (lower is better)
  Worst Map Return: -234.1
  Best Map Return: 523.4
  Return Range: 757.5
================================================================================
```

**생성 파일:**
- `outputs/generalization/generalization_results.json`
- `outputs/generalization/generalization_comparison.png`
- `outputs/generalization/return_distribution.png`

---

### 2. 커버리지 히트맵 시각화

```bash
python scripts/visualize_coverage.py \
    --model checkpoints/multi_map_ppo/final.pth \
    --episodes-per-map 20 \
    --output outputs/coverage_heatmaps.png
```

**생성 결과:**

히트맵 이미지 (각 맵별):
- 로봇이 방문한 영역 (빨간색 = 자주 방문)
- 순찰 포인트 표시
- 커버리지 통계:
  - Coverage: 92.3%
  - Cells Visited: 1,235/1,338
  - Total Visits: 8,457

**용도:**
- 사각지대 파악
- 순찰 경로 분석
- 정책 행동 이해

---

## 🔬 학습 과정 분석

### TensorBoard 활용

```bash
tensorboard --logdir runs/multi_map_ppo
```

**확인 가능한 메트릭:**

1. **전체 성능**
   - `episode/return` - 평균 리턴
   - `episode/length` - 에피소드 길이

2. **맵별 성능**
   - `episode_per_map/map_large_square/return`
   - `episode_per_map/map_office_building/event_success_rate`
   - `episode_per_map/map_campus/patrol_coverage`

3. **학습 메트릭**
   - `train/policy_loss`
   - `train/value_loss`
   - `train/learning_rate`

4. **맵 선택 통계**
   - 각 맵이 선택된 횟수
   - 맵별 학습 진행도

---

## 🎓 예상 결과 (학습 곡선)

### 단일 맵 vs 멀티맵

```
Return
  ^
  |     ┌─────── Single Map (default.yaml)
  |    ╱
  |   ╱
  |  ╱                ┌────── Multi-Map (6 maps)
  | ╱                ╱
  |╱                ╱
  |               ╱
  +────────────────────────────────> Episodes
  0      200K    400K    600K    800K   1M

Generalization Score (Avg across all maps)
  ^
  |                        ┌────── Multi-Map
  |                      ╱
  |                    ╱
  |                  ╱
  |    ┌──────────── Single Map (overfits)
  |   ╱
  |  ╱
  +────────────────────────────────> Episodes
```

**예상:**
- **단일 맵 학습**:
  - 학습 맵: 높은 성능 (Return = 600+)
  - 다른 맵: 낮은 성능 (Return = -300 ~ 200)
  - 일반화 점수: 낮음

- **멀티맵 학습**:
  - 모든 맵: 중간~높은 성능 (Return = 150 ~ 500)
  - 분산: 낮음 (일관성 있음)
  - 일반화 점수: 높음

---

## 💡 Best Practices

### 1. 학습 시간 배분

```bash
# 초기 탐색: 1M steps
python scripts/train_multi_map.py --total-timesteps 1000000

# 표준 학습: 5M steps
python scripts/train_multi_map.py --total-timesteps 5000000

# 고성능: 10M+ steps
python scripts/train_multi_map.py --total-timesteps 10000000
```

### 2. 맵 선택 전략

| 상황 | 추천 모드 | 이유 |
|------|----------|------|
| **처음 시작** | `curriculum` | 안정적 학습 |
| **균등 학습** | `sequential` | 모든 맵 균등 |
| **최종 성능** | `random` | 실전 시나리오 |

### 3. 하이퍼파라미터 튜닝

```bash
# 큰 배치 (안정성)
python scripts/train_multi_map.py \
    --num-steps 4096 \
    --batch-size 512

# 작은 학습률 (세밀한 조정)
python scripts/train_multi_map.py \
    --learning-rate 0.0001

# 많은 에폭 (강한 업데이트)
python scripts/train_multi_map.py \
    --num-epochs 15
```

### 4. 정기 평가

```bash
# 학습 중간에 체크포인트 평가
python scripts/evaluate_generalization.py \
    --model runs/multi_map_ppo/20250101-120000/checkpoints/update_500.pth \
    --episodes 20

python scripts/evaluate_generalization.py \
    --model runs/multi_map_ppo/20250101-120000/checkpoints/update_1000.pth \
    --episodes 20
```

---

## 📚 실전 배포 시나리오

### 시나리오 1: 새로운 건물에 배포

**문제:**
- 새 건물 평면도: 복잡한 구조
- 기존 학습 맵과 다름

**해결:**

1. **커스텀 맵 설정 생성**
   ```yaml
   # configs/new_building.yaml
   env:
     map_width: 75.0
     map_height: 55.0
     patrol_points:
       - [실제 좌표들...]
   ```

2. **멀티맵 모델 로드 후 평가**
   ```bash
   python scripts/evaluate_generalization.py \
       --model checkpoints/multi_map_ppo/final.pth \
       --maps configs/new_building.yaml \
       --episodes 50
   ```

3. **성능 확인**
   - 즉시 사용 가능 수준: Return > 200
   - Fine-tuning 필요: Return < 200

4. **필요 시 Fine-tuning**
   ```bash
   python scripts/train.py \
       --config configs/new_building.yaml \
       --load-model checkpoints/multi_map_ppo/final.pth \
       --total-timesteps 500000
   ```

---

### 시나리오 2: 여러 건물 동시 운영

**상황:**
- A동, B동, C동 3개 건물
- 각각 다른 구조

**해결:**

1. **각 건물 맵 설정**
   ```bash
   configs/building_a.yaml
   configs/building_b.yaml
   configs/building_c.yaml
   ```

2. **통합 학습**
   ```bash
   python scripts/train_multi_map.py \
       --maps \
           configs/building_a.yaml \
           configs/building_b.yaml \
           configs/building_c.yaml \
       --total-timesteps 5000000
   ```

3. **단일 모델로 모든 건물 커버**
   - 하나의 정책으로 3개 건물 운영
   - 건물별 재학습 불필요
   - 유지보수 비용 절감

---

## 🎉 성공 기준

### 좋은 일반화 성능

✅ **모든 맵에서 양수 리턴**
- Worst map return > 0

✅ **낮은 분산**
- Return std across maps < 150

✅ **높은 평균 성공률**
- Average event success > 75%
- Average patrol coverage > 85%

✅ **새로운 맵에서도 작동**
- Unseen map return > 0.8 * avg seen map return

---

## 🔧 문제 해결

### Q: 특정 맵에서만 성능이 낮음

**원인:** 그 맵이 너무 어렵거나 자주 선택되지 않음

**해결:**
```bash
# 1. Sequential 모드로 균등 학습
python scripts/train_multi_map.py --map-mode sequential

# 2. 어려운 맵 우선순위 증가 (커스텀 코드 필요)
# 3. 더 긴 학습
```

### Q: 수렴이 느림

**원인:** 맵이 너무 다양해서 학습 불안정

**해결:**
```bash
# 1. Curriculum 모드 사용
python scripts/train_multi_map.py --map-mode curriculum

# 2. 학습률 감소
python scripts/train_multi_map.py --learning-rate 0.0001

# 3. 더 큰 배치
python scripts/train_multi_map.py --num-steps 4096
```

### Q: 메모리 부족

**원인:** 여러 맵 설정 로드

**해결:**
- 맵 개수 줄이기 (3-4개)
- 배치 크기 줄이기
- CPU 모드 사용 (`--cuda` 제거)

---

## 📖 추가 자료

- **맵 설정 가이드**: `configs/README.md`
- **맵 구성 설명**: `docs/MAP_CONFIGURATION.md`
- **환경 코드**: `src/rl_dispatch/env/multi_map_env.py`
- **학습 스크립트**: `scripts/train_multi_map.py`
- **평가 스크립트**: `scripts/evaluate_generalization.py`

---

**작성일**: 2025-12-29
**작성자**: 박용준 (YJP)
**버전**: 1.0.0

🚀 **멀티맵 학습으로 실전에서 바로 사용 가능한 일반화된 정책을 만드세요!**
