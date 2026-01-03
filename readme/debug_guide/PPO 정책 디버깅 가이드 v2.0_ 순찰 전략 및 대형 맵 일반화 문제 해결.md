# PPO 정책 디버깅 가이드 v2.0: 순찰 전략 및 대형 맵 일반화 문제 해결

**문서 버전:** 2.0  
**검토 일자:** 2025년 12월 30일  
**작성자:** Manus AI (AI 수석 연구원)

---

## 1. 개요

본 문서는 `rl_dispatch_mvp` 프로젝트의 학습 로그(`runs/multi_map_logged/`)와 사용자 분석(`pasted_content.txt`)을 종합하여, 현재 PPO 정책이 직면한 핵심 문제인 **순찰 전략 부재**와 **대형 맵 일반화 실패**를 해결하기 위한 구체적인 디버깅 및 개선 가이드라인을 제공합니다. 

현재 정책은 소형 맵에서 '이벤트 발생 시 출동'이라는 단기 목표는 성공적으로 학습했으나, 장기적인 순찰 커버리지를 유지하는 능력이 부재하며, 이는 대형/복잡 맵에서 총 보상(Return)의 급격한 하락과 분산 폭증으로 이어지고 있습니다. 분석 결과, 이는 PPO 알고리즘 자체의 문제가 아닌, **보상 함수 설계의 불균형**과 **단순한 학습 커리큘럼**에서 기인한 구조적 문제입니다.

본 가이드는 `pasted_content.txt`에서 제안된 해결책을 기반으로, 즉시 코드에 적용 가능한 형태로 구체화한 3단계 솔루션을 제시합니다.

## 2. 문제 현상 및 데이터 기반 진단

### 2.1. 정량적 데이터 분석 (`results.json`)

| 맵 유형 | 맵 이름 | 이벤트 성공률 | 순찰 커버리지 | 평균 Return | Return 표준편차 |
|:---|:---|:---:|:---:|:---:|:---:|
| **소형/단순** | `large_square` | 85.7% | 9.9% | -5,205 | 1,787 |
| | `corridor` | 83.4% | 15.4% | -3,628 | 1,366 |
| | `l_shaped` | 84.6% | 15.7% | -4,146 | 1,605 |
| **대형/복잡** | `office_building` | 80.5% | 5.6% | -28,388 | 33,756 |
| | `warehouse` | 72.0% | 9.9% | -16,765 | 9,866 |
| | `campus` | **51.5%** | **5.7%** | **-77,202** | **83,321** |

- **출동 편향성:** 모든 맵에서 이벤트 성공률은 비교적 높게 유지되나, 순찰 커버리지는 5~15% 수준으로 매우 낮습니다. 이는 정책이 순찰의 가치를 거의 학습하지 못하고 출동에만 집중하고 있음을 시사합니다.
- **일반화 실패:** `campus`와 같이 크고 복잡한 맵에서 Return 값이 기하급수적으로 감소하고 표준편차가 폭발적으로 증가합니다. 이는 정책이 비효율적인 이동을 반복하며 장기적인 손실을 크게 입고 있음을 의미합니다.

### 2.2. 근본 원인 분석

1.  **보상 함수 불균형:** 현재 `reward/default.yaml`에 설정된 가중치(`w_event: 1.0`, `w_patrol: 0.5`)와 보상 구조는 이벤트 성공 시 즉각적인 큰 보너스(`event_response_bonus: 50.0`)를 제공하는 반면, 순찰 공백에 대한 페널티(`patrol_gap_penalty_rate: 0.1`)는 선형적이고 미미합니다. 이는 정책이 "순찰을 포기하더라도 출동만 성공하면 이득"이라는 근시안적 전략에 빠지게 만듭니다.

2.  **커리큘럼 부재:** 모든 맵(쉬운 맵과 어려운 맵)을 동일한 확률로 학습시키는 방식은 정책이 상태-행동 분포가 판이하게 다른 `campus` 같은 고난이도 맵에 적응하는 것을 방해합니다. 쉬운 맵에서 학습된 단기 전략이 어려운 맵에서는 통하지 않아 성능 저하가 발생합니다.

## 3. 해결을 위한 3단계 디버깅 가이드

`pasted_content.txt`의 제안에 따라, 아래 3가지 핵심 수정안을 코드 레벨에서 구체화하여 제안합니다.

### 3.1. 1단계: 보상 함수 재설계 (Reward Shaping)

**목표:** '출동만 하는 정책'을 깨고, 순찰의 장기적 가치를 학습하도록 유도합니다.

#### A. 순찰 커버리지 보상: 비선형 누적 페널티 도입

- **문제점:** 현재의 선형적인 `patrol_gap_penalty`는 장시간 방치된 구역에 대한 강력한 페널티를 주지 못합니다.
- **수정 제안:** `reward_calculator.py`의 `_calculate_patrol_reward` 함수를 수정하여, 미방문 시간(`gap`)이 특정 임계값(`overdue_threshold`)을 넘어서면 페널티가 비선형적으로 급증하는 `softplus` 함수 기반의 페널티를 도입합니다.

```python
# src/rl_dispatch/rewards/reward_calculator.py

def _calculate_patrol_reward(self, patrol_points, current_time, ...):
    # ... 기존 visit_bonus 로직 ...

    # 수정: 비선형 Gap 페널티 계산
    total_gap_penalty = 0.0
    overdue_threshold = self.config.patrol_overdue_threshold # 예: 300.0
    gap_scale = self.config.patrol_gap_scale # 예: 0.1 (softplus의 기울기 조절)

    for point in patrol_points:
        gap = current_time - point.last_visit_time
        # softplus(x) = log(1 + exp(x))
        # gap이 threshold를 넘으면 페널티가 급격히 증가
        overdue_value = np.log(1 + np.exp(gap_scale * (gap - overdue_threshold)))
        total_gap_penalty += overdue_value * point.priority

    # config에서 읽어온 가중치 적용
    reward -= self.config.patrol_gap_penalty_rate * total_gap_penalty 
    return reward
```

#### B. 이벤트 보상 재균형: 성공 보너스 축소 및 지연 페널티 구조화

- **문제점:** 과도한 성공 보너스가 순찰 포기를 부추깁니다.
- **수정 제안:** `_calculate_event_reward` 함수를 수정하여 성공 보너스를 낮추고, 지연 페널티를 로그 함수 형태로 변경하여 초반 지연에는 관대하되 시간이 지날수록 페널티가 커지도록 합니다.

```python
# src/rl_dispatch/rewards/reward_calculator.py

def _calculate_event_reward(self, event, current_time, event_resolved, ...):
    # ...
    if event_resolved:
        # w_succ: 1.0 -> 0.8 (또는 그 이하)로 조정 제안
        return self.config.event_success_bonus 

    delay = current_time - event.detection_time
    if delay > self.config.event_max_delay:
        # w_fail: 2.0 (유지 또는 소폭 조정)
        return -self.config.event_failure_penalty

    # w_late: 0.6 (신규 도입)
    # T0: 페널티 스케일 조절 파라미터 (예: 60.0)
    latency_penalty = self.config.event_latency_penalty * np.log(1 + delay / self.config.event_latency_scale)
    return -latency_penalty
```

#### C. 탐색 유도 보상 (Shaping) 추가

- **문제점:** 이벤트가 없을 때 정책이 무의미한 행동을 하며 커버리지를 회복하지 못합니다.
- **수정 제안:** 이벤트가 없을 때, 가장 오래 방치된(overdue) 지점으로 향하도록 작은 보상을 주는 `shaping` 항을 추가합니다.

```python
# src/rl_dispatch/rewards/reward_calculator.py -> calculate 함수 내

# ... r_event, r_patrol 등 계산 후 ...

r_shaping = 0.0
if not event:
    # 가장 오래된 포인트 찾기
    most_overdue_point = max(patrol_points, key=lambda p: current_time - p.last_visit_time)
    target_pos = most_overdue_point.position
    
    # 이전 스텝과 현재 스텝에서 타겟까지의 거리 계산
    dist_prev = np.linalg.norm(np.array(prev_robot_pos) - np.array(target_pos))
    dist_now = np.linalg.norm(np.array(robot.position) - np.array(target_pos))
    
    # 타겟에 가까워졌으면 보상
    # w_toward: 0.05 ~ 0.15
    r_shaping = self.config.toward_overdue_bonus * (dist_prev - dist_now)

# 최종 보상에 추가
rewards = RewardComponents(..., shaping=r_shaping)
```

### 3.2. 2단계: 맵 난이도 기반 커리큘럼 학습 도입

**목표:** 쉬운 맵에서 기본 전략을 학습한 후, 점진적으로 어려운 맵으로 확장하여 일반화 성능을 높입니다.

- **문제점:** 모든 맵을 한 번에 학습하여 어려운 맵에 대한 적응에 실패합니다.
- **수정 제안:** `train_multi_map.py` 스크립트를 수정하여, 전체 학습 과정을 3단계로 나누고, 각 단계별로 학습 맵과 보상 가중치를 다르게 설정합니다.

```python
# scripts/train_multi_map_curriculum.py (신규 또는 수정)

PHASES = [
    {
        "name": "Phase 0: Warming Up",
        "steps": 300_000,
        "maps": ["corridor", "l_shaped"],
        "reward_weights": "configs/rewards/phase0.yaml"
    },
    {
        "name": "Phase 1: Intermediate",
        "steps": 700_000, # 누적 1M
        "maps": ["corridor", "l_shaped", "large_square", "warehouse"],
        "reward_weights": "configs/rewards/phase1.yaml"
    },
    {
        "name": "Phase 2: Generalization",
        "steps": 2_000_000, # 누적 3M
        "maps": ["corridor", "l_shaped", "large_square", "warehouse", "office_building", "campus"],
        "map_weights": [0.1, 0.1, 0.1, 0.1, 0.3, 0.3], # 어려운 맵에 가중치 부여
        "reward_weights": "configs/rewards/phase2.yaml"
    }
]

# 학습 루프에서 각 phase를 순차적으로 실행
# 각 phase 시작 시, 해당하는 맵 리스트와 보상 가중치로 환경과 reward_calculator를 재설정
```

- **보상 가중치 커리큘럼:** `pasted_content.txt`의 제안에 따라, `w_gap` (순찰 공백 페널티 가중치)를 단계적으로 높여 순찰의 중요성을 점차 강조합니다.

| 가중치 | Phase 0 | Phase 1 | Phase 2 |
|:---|:---:|:---:|:---:|
| `w_succ` | 1.0 | 1.0 | **0.8** |
| `w_late` | 0.6 | 0.6 | 0.6 |
| `w_fail` | 2.0 | 2.0 | 2.0 |
| `w_gap` | **0.2** | **0.5** | **1.0** |
| `w_tow` | 0.1 | 0.1 | 0.15 |

### 3.3. 3단계: 상태 공간에 '맵 스케일 힌트' 추가

**목표:** 단일 정책이 다양한 크기의 맵에 더 잘 일반화되도록, 맵의 특성을 직접적인 정보로 제공합니다.

- **문제점:** 정책이 현재 관측만으로는 맵의 전체 크기나 구조를 추론하기 어려워, 단기적인 판단에 갇히기 쉽습니다.
- **수정 제안:** `PatrolEnv`의 관측 공간(`Observation`)에 맵의 특성을 나타내는 3가지 정보를 추가하고, `ObservationProcessor`와 `ActorCriticNetwork`의 입력 차원을 그에 맞게 수정합니다. (77D → 80D)

1.  **맵 전체 크기 (1D):** `sqrt(map_width * map_height)`
2.  **평균 후보 간 거리 (1D):** 현재 생성된 후보 경로들의 평균 ETA
3.  **현재 위치의 지역 밀도 (1D):** 로봇 반경 N미터 내에 있는 순찰 지점의 수

```python
# src/rl_dispatch/utils/observation.py

class ObservationProcessor:
    def process(self, state: State, ...) -> Observation:
        # ... 기존 관측 벡터 생성 ...

        # 맵 스케일 힌트 추가
        map_scale = np.sqrt(state.map_width * state.map_height)
        avg_candidate_eta = np.mean([c.eta for c in state.candidates if c.feasible]) if state.candidates else 0
        
        robot_pos = np.array(state.robot.position)
        local_density = 0
        for p in state.patrol_points:
            if np.linalg.norm(robot_pos - np.array(p.position)) < 10.0: # 10m 반경
                local_density += 1

        # 정규화 후 벡터에 추가
        # ...
        obs_vector = np.concatenate([..., [norm_map_scale, norm_avg_eta, norm_density]])
        return Observation(vector=obs_vector, ...)
```

## 4. 검증 체크포인트

위 수정안들을 적용한 후, 다음 5가지 지표를 중점적으로 모니터링하여 개선 여부를 판단합니다.

1.  **순찰 커버리지 상승:** `campus`, `office_building` 맵에서 순찰 커버리지가 5%대에서 **10% 이상으로 상승**하는가?
2.  **Return 분산 감소:** `campus` 맵의 `std_return`이 83k에서 **40k 이하로 유의미하게 감소**하는가?
3.  **이벤트 성공률 유지:** 커버리지가 상승하는 동안, 이벤트 성공률이 급격히 하락하지 않고 **안정적으로 유지**되는가? (만약 하락 시, `w_gap`을 너무 빨리 올렸을 가능성)
4.  **응답 시간 변화:** 평균 `response_time`이 과도하게 증가하지 않는가? (만약 급증 시, `w_late` 가중치가 너무 클 수 있음)
5.  **탐색 행동 변화:** 이벤트가 없을 때, 로봇이 특정 지점으로 이동하려는 경향(shaping 보상 효과)을 보이는가?

---

*본 문서는 `rl_dispatch_mvp` 코드베이스의 학습 로그 및 사용자 분석을 기반으로 작성되었습니다.*
