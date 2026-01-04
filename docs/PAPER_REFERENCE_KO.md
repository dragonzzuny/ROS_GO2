# RL Dispatch MVP - 논문 참고용 기술 문서

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [문제 정의](#2-문제-정의)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [수학적 배경](#4-수학적-배경)
5. [알고리즘 상세](#5-알고리즘-상세)
6. [보상 함수 설계](#6-보상-함수-설계)
7. [관측 공간 설계](#7-관측-공간-설계)
8. [행동 공간 설계](#8-행동-공간-설계)
9. [후보 경로 생성 전략](#9-후보-경로-생성-전략)
10. [신경망 아키텍처](#10-신경망-아키텍처)
11. [핵심 변수 정의](#11-핵심-변수-정의)
12. [파일별 상세 설명](#12-파일별-상세-설명)
13. [실험 설정](#13-실험-설정)
14. [참고문헌](#14-참고문헌)

---

## 1. 프로젝트 개요

### 1.1 연구 목적

본 프로젝트는 자율 보안 순찰 로봇의 **디스패치 및 경로 재계획 문제**를 강화학습으로 해결합니다. Unitree Go2 사족보행 로봇이 CCTV 연동 보안 시스템에서 다음 두 가지 상충되는 목표를 최적화합니다:

1. **이벤트 대응**: CCTV에서 감지된 이상 상황에 신속히 대응
2. **순찰 커버리지**: 지정된 순찰 지점들을 정기적으로 방문하여 커버리지 유지

### 1.2 핵심 기여

- **Semi-MDP 환경 설계**: 가변 시간 간격의 내비게이션 기반 의사결정
- **다목적 보상 함수**: 이벤트/순찰/안전/효율성 4개 컴포넌트 균형
- **후보 기반 행동 공간**: 10개 휴리스틱 전략으로 조합 폭발 문제 해결
- **Phase 2 보상 정규화**: 컴포넌트별 러닝 정규화로 학습 안정성 향상
- **Phase 4 관측 공간 확장**: 88차원 관측 벡터로 의사결정 품질 개선

### 1.3 대상 플랫폼

- **로봇**: Unitree Go2 (사족보행)
- **내비게이션**: ROS2 Humble + Nav2 스택
- **시뮬레이션**: Gazebo, NVIDIA Isaac Sim
- **학습 프레임워크**: PyTorch + Gymnasium

---

## 2. 문제 정의

### 2.1 Semi-Markov Decision Process (SMDP)

표준 MDP와 달리, 본 문제는 **Semi-MDP**로 모델링됩니다. 각 스텝이 고정 시간이 아닌 가변 시간(내비게이션 완료까지)을 갖습니다.

**SMDP 튜플**: $(S, A, P, R, \gamma, \tau)$

| 기호 | 설명 |
|------|------|
| $S$ | 상태 공간 (로봇 상태, 순찰 지점, 이벤트, LiDAR) |
| $A$ | 행동 공간 (모드 × 재계획 전략) |
| $P$ | 상태 전이 확률 |
| $R$ | 보상 함수 |
| $\gamma$ | 할인율 (시간 기반: $\gamma^{\tau}$) |
| $\tau$ | 소요 시간 (스텝마다 가변) |

### 2.2 SMDP 할인율

일반 MDP에서는 $\gamma^t$ (스텝 수 기반)을 사용하지만, SMDP에서는 실제 소요 시간을 반영합니다:

$$
\gamma_{SMDP} = \gamma^{\tau_{nav}}
$$

여기서 $\tau_{nav}$는 내비게이션 소요 시간(초)입니다.

### 2.3 목표 함수

에피소드 누적 보상 최대화:

$$
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^{\sum_{i=0}^{t-1} \tau_i} R_t \right]
$$

---

## 3. 시스템 아키텍처

### 3.1 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                     RL Dispatch MVP                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │  Environment │────▶│   Policy     │────▶│   Action     │ │
│  │  (PatrolEnv) │     │ (PPOAgent)   │     │  Execution   │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   Reward     │     │   Network    │     │  Navigation  │ │
│  │  Calculator  │     │ (Actor-Critic)│    │  Interface   │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │  Candidate   │     │   Buffer     │     │  A* Path     │ │
│  │  Generator   │     │ (RolloutBuffer)│   │  Planner     │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 모듈 구성

| 모듈 | 위치 | 역할 |
|------|------|------|
| `core/` | `src/rl_dispatch/core/` | 타입 정의, 설정 관리 |
| `env/` | `src/rl_dispatch/env/` | Gymnasium 환경 구현 |
| `algorithms/` | `src/rl_dispatch/algorithms/` | PPO 알고리즘, 신경망, 버퍼 |
| `planning/` | `src/rl_dispatch/planning/` | 후보 경로 생성 전략 |
| `rewards/` | `src/rl_dispatch/rewards/` | 보상 함수 계산 |
| `navigation/` | `src/rl_dispatch/navigation/` | Nav2 인터페이스, A* 경로계획 |
| `deployment/` | `src/rl_dispatch/deployment/` | 실 로봇 배포 인터페이스 |
| `pbt/` | `src/rl_dispatch/pbt/` | Population-Based Training |

---

## 4. 수학적 배경

### 4.1 Proximal Policy Optimization (PPO)

PPO는 정책 경사(Policy Gradient) 기반 알고리즘으로, 클리핑을 통해 안정적인 정책 업데이트를 보장합니다.

#### 4.1.1 정책 경사 정리

정책 $\pi_\theta$의 기대 보상:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

정책 경사:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) A^\pi(s_t, a_t) \right]
$$

#### 4.1.2 PPO-Clip 목적 함수

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

여기서:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ (확률 비율)
- $\epsilon$ = 클리핑 파라미터 (기본값: 0.2)
- $A_t$ = 어드밴티지 추정값

#### 4.1.3 가치 손실 (Value Loss)

$$
L^{VF}(\theta) = \mathbb{E}_t \left[ (V_\theta(s_t) - V_t^{target})^2 \right]
$$

#### 4.1.4 엔트로피 보너스

탐색(exploration)을 촉진하기 위한 엔트로피 항:

$$
H(\pi_\theta) = -\mathbb{E}_{a \sim \pi_\theta} \left[ \log \pi_\theta(a|s) \right]
$$

#### 4.1.5 전체 손실 함수

$$
L(\theta) = -L^{CLIP}(\theta) + c_1 L^{VF}(\theta) - c_2 H(\pi_\theta)
$$

여기서:
- $c_1$ = 가치 손실 계수 (기본값: 0.5)
- $c_2$ = 엔트로피 계수 (기본값: 0.01)

### 4.2 Generalized Advantage Estimation (GAE)

GAE는 편향-분산 트레이드오프를 조절하여 어드밴티지를 추정합니다.

#### 4.2.1 TD 잔차 (Temporal Difference Residual)

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

#### 4.2.2 GAE 공식

$$
A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

실제 구현 (역순 계산):

$$
A_t = \delta_t + \gamma \lambda A_{t+1}
$$

**SMDP 확장** (본 프로젝트):

$$
\delta_t = r_t + \gamma^{\tau_t} V(s_{t+1}) - V(s_t)
$$

$$
A_t = \delta_t + (\gamma \lambda)^{\tau_t} A_{t+1}
$$

여기서 $\tau_t$는 스텝 $t$의 내비게이션 소요 시간입니다.

### 4.3 A* 경로 탐색

#### 4.3.1 비용 함수

$$
f(n) = g(n) + h(n)
$$

- $g(n)$: 시작 노드에서 현재 노드까지의 실제 비용
- $h(n)$: 현재 노드에서 목표까지의 휴리스틱 추정

#### 4.3.2 Octile Distance 휴리스틱

8방향 이동을 지원하는 환경에서의 휴리스틱:

$$
h(n) = \max(|dx|, |dy|) + (\sqrt{2} - 1) \min(|dx|, |dy|)
$$

여기서 $dx = |x_{goal} - x_n|$, $dy = |y_{goal} - y_n|$

---

## 5. 알고리즘 상세

### 5.1 PPO 학습 절차

```
알고리즘: PPO for Patrol Dispatch

입력: 환경 E, 정책 π_θ, 가치함수 V_φ
초기화: θ, φ (신경망 파라미터)

for iteration = 1, 2, ... do
    # 롤아웃 수집
    for step = 1 to T do
        a_t, logπ_t = π_θ(s_t)
        s_{t+1}, r_t, τ_t = E.step(a_t)
        Buffer.add(s_t, a_t, r_t, logπ_t, τ_t)
    end for

    # GAE 계산 (SMDP 할인)
    for t = T-1 down to 0 do
        δ_t = r_t + γ^{τ_t} V_φ(s_{t+1}) - V_φ(s_t)
        A_t = δ_t + (γλ)^{τ_t} A_{t+1}
    end for

    # PPO 업데이트
    for epoch = 1 to K do
        for minibatch in Buffer do
            r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
            L_clip = min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)
            L_vf = (V_φ(s_t) - V_t^{target})^2
            L = -L_clip + c_1 L_vf - c_2 H(π_θ)
            θ, φ ← θ, φ - α∇L
        end for
    end for
end for
```

### 5.2 행동 선택 절차

```
함수: get_action(observation, action_mask)

입력:
  - observation: 88차원 관측 벡터
  - action_mask: 유효 행동 마스크 (optional)

1. 관측을 신경망에 입력
   mode_logits, replan_logits, value = network(observation)

2. 행동 마스킹 적용
   if action_mask is not None:
       mode_logits[~mode_mask] = -∞
       replan_logits[~replan_mask] = -∞

3. 확률 분포 생성
   mode_probs = softmax(mode_logits)
   replan_probs = softmax(replan_logits)

4. 행동 샘플링
   mode = Categorical(mode_probs).sample()
   replan = Categorical(replan_probs).sample()

5. 로그 확률 계산
   log_prob = log(mode_probs[mode]) + log(replan_probs[replan])

반환: (mode, replan), log_prob, value
```

---

## 6. 보상 함수 설계

### 6.1 다목적 보상 구조

총 보상은 4개 컴포넌트의 가중합으로 구성됩니다:

$$
R_{total} = w_{evt} R^{evt} + w_{pat} R^{pat} + w_{safe} R^{safe} + w_{eff} R^{eff}
$$

| 컴포넌트 | 기호 | 기본 가중치 | 목적 |
|----------|------|-------------|------|
| 이벤트 대응 | $R^{evt}$ | 1.0 | 이벤트 신속 대응 유도 |
| 순찰 커버리지 | $R^{pat}$ | 0.5 | 순찰 지점 정기 방문 유도 |
| 안전성 | $R^{safe}$ | 2.0 | 충돌 회피, 안전 운행 |
| 효율성 | $R^{eff}$ | 0.1 | 불필요한 이동 억제 |

### 6.2 이벤트 대응 보상 ($R^{evt}$)

**Phase 2 SLA 기반 설계:**

#### 6.2.1 이벤트 해결 성공 시

$$
R^{evt}_{success} = V_{SLA} \times M_{risk} \times (0.5 + 0.5 \times Q_{SLA})
$$

여기서:
- $V_{SLA}$ = 100 (SLA 성공 기본값)
- $M_{risk} = 0.5 + \frac{risk\_level}{9} \times 1.5$ (위험도 승수)
- $Q_{SLA} = \max(0, 1 - \frac{delay}{T_{max}})$ (SLA 품질 점수)

#### 6.2.2 이벤트 미해결 시 (지연 페널티)

$$
R^{evt}_{delay} = -P_{SLA} \times \frac{delay}{T_{max}} \times \frac{risk\_level}{9}
$$

여기서:
- $P_{SLA}$ = 10 (SLA 저하 페널티율)
- $T_{max}$ = 120초 (최대 허용 지연)

#### 6.2.3 SLA 실패 시

$$
R^{evt}_{failure} = -C_{SLA} \times M_{risk}
$$

여기서 $C_{SLA}$ = 200 (SLA 실패 비용)

### 6.3 순찰 커버리지 보상 ($R^{pat}$)

**Phase 2 Delta Coverage 설계:**

#### 6.3.1 순찰 지점 방문 시 (양의 보상)

$$
R^{pat}_{visit} = \Delta_{gap} \times p_i \times \alpha_{visit}
$$

여기서:
- $\Delta_{gap} = t_{current} - t_{last\_visit}$ (해소된 커버리지 갭)
- $p_i$ = 해당 순찰 지점의 우선순위
- $\alpha_{visit}$ = 0.5 (방문 보상률)

#### 6.3.2 기저 페널티 (누적 갭)

$$
R^{pat}_{baseline} = -\beta_{baseline} \times \frac{\sum_{i} (t_{current} - t_i) \times p_i}{N}
$$

여기서:
- $\beta_{baseline}$ = 0.01 (기저 페널티율)
- $N$ = 순찰 지점 개수

### 6.4 안전성 보상 ($R^{safe}$)

$$
R^{safe} = \begin{cases}
-10.0 & \text{if 충돌 발생} \\
-2.0 & \text{if Nav2 실패} \\
0.0 & \text{otherwise}
\end{cases}
$$

**참고**: 안전성 보상은 희소(sparse)하고 중요한 신호이므로 정규화하지 않습니다.

### 6.5 효율성 보상 ($R^{eff}$)

$$
R^{eff} = -\kappa \times d_{traveled}
$$

여기서:
- $\kappa$ = 0.01 (거리 페널티율)
- $d_{traveled}$ = 이동 거리 (미터)

### 6.6 컴포넌트별 정규화 (Phase 2)

각 보상 컴포넌트를 개별적으로 정규화하여 스케일을 통일합니다:

$$
R^{norm}_c = \frac{R^{raw}_c - \mu_c}{\sigma_c}
$$

**Welford 온라인 알고리즘**으로 러닝 통계 계산:

```
count ← count + 1
delta ← value - mean
mean ← mean + delta / count
delta2 ← value - mean
M2 ← M2 + delta × delta2
variance ← M2 / (count - 1)
```

---

## 7. 관측 공간 설계

### 7.1 관측 벡터 구성 (88차원)

| 인덱스 | 차원 | 설명 | 정규화 방법 |
|--------|------|------|-------------|
| 0-1 | 2 | 목표 상대 위치 $(dx, dy)$ | $\div$ map_size |
| 2-3 | 2 | 로봇 방향 $(\sin\theta, \cos\theta)$ | 자연적으로 [-1, 1] |
| 4-5 | 2 | 속도/각속도 $(v, \omega)$ | $\div$ max_vel |
| 6 | 1 | 배터리 레벨 | 자연적으로 [0, 1] |
| 7-70 | 64 | LiDAR 범위 | $\div$ max_range |
| 71-74 | 4 | 이벤트 특징 | 각각 [0, 1] |
| 75-76 | 2 | 순찰 특징 | 정규화 |
| 77 | 1 | 이벤트 위험도 (Phase 4) | $\div 9$ |
| 78-80 | 3 | 순찰 위기 상태 (Phase 4) | 정규화 |
| 81-86 | 6 | 후보 실행가능성 (Phase 4) | [0, 1] |
| 87 | 1 | 긴급도-위험도 결합 (Phase 4) | [0, 1] |

### 7.2 이벤트 특징 상세 (4차원)

```python
event_features = [
    1.0 if has_event else 0.0,      # 이벤트 존재 여부
    event.urgency if has_event else 0.0,  # 긴급도 [0, 1]
    event.confidence if has_event else 0.0,  # 탐지 신뢰도 [0, 1]
    elapsed_time / max_delay,  # 경과 시간 (정규화)
]
```

### 7.3 순찰 특징 상세 (2차원)

```python
patrol_features = [
    distance_to_next / max_distance,  # 다음 지점까지 거리
    coverage_gap_ratio,  # 커버리지 갭 비율
]
```

### 7.4 Phase 4 확장 특징 (11차원)

| 특징 | 차원 | 계산 방법 |
|------|------|-----------|
| 이벤트 위험도 | 1 | risk_level / 9 |
| 최대 갭 | 1 | max_gap / max_time |
| 위기 지점 수 | 1 | count(gap > threshold) / N |
| 위기 점수 | 1 | 가중 위기 지표 |
| 후보 실행가능성 | 6 | 각 후보의 A* 실행가능 여부 |
| 결합 우선순위 | 1 | urgency × risk_level / 9 |

---

## 8. 행동 공간 설계

### 8.1 복합 행동 공간

$$
A = \{0, 1\} \times \{0, 1, ..., K-1\}
$$

- **모드 (Mode)**: 2개 선택지
  - 0 = PATROL (순찰 계속)
  - 1 = DISPATCH (이벤트 대응)

- **재계획 전략 (Replan)**: K개 선택지 (기본 K=6, 확장 K=10)

### 8.2 행동 마스킹

유효하지 않은 행동을 마스킹하여 정책 안정성을 확보합니다:

$$
\pi_{masked}(a|s) = \frac{\mathbb{1}_{valid}(a) \cdot \exp(logits(a))}{\sum_{a'} \mathbb{1}_{valid}(a') \cdot \exp(logits(a'))}
$$

**마스킹 조건**:

| 조건 | 마스킹 대상 |
|------|-------------|
| 이벤트 없음 | DISPATCH 모드 |
| 배터리 < 20% | DISPATCH 모드 |
| A* 경로 없음 | 해당 재계획 전략 |

### 8.3 행동 해석

```python
def interpret_action(action):
    mode = ActionMode(action[0])  # 0=PATROL, 1=DISPATCH
    replan_idx = action[1]        # 0 ~ K-1

    if mode == DISPATCH:
        goal = event_position
    else:
        candidate = candidates[replan_idx]
        patrol_route = candidate.patrol_order
        goal = patrol_points[patrol_route[0]]

    return goal
```

---

## 9. 후보 경로 생성 전략

### 9.1 전략 개요

10개의 휴리스틱 전략으로 순찰 경로 후보를 생성합니다:

| ID | 전략명 | 설명 | 최적화 목표 |
|----|--------|------|-------------|
| 0 | Keep-Order | 현재 순서 유지 | 안정성 |
| 1 | Nearest-First | 최근접 우선 | 즉각 효율 |
| 2 | Most-Overdue-First | 최장 미방문 우선 | 커버리지 회복 |
| 3 | Overdue-ETA-Balance | 미방문+이동시간 균형 | 하이브리드 |
| 4 | Risk-Weighted | 고위험 지역 우선 | 위험 관리 |
| 5 | Balanced-Coverage | 최대 갭 최소화 | 균등 분배 |
| 6 | Overdue-Threshold-First | 임계값 초과 우선 | 위기 대응 |
| 7 | Windowed-Replan | 앞 H개만 재계획 | 계산 효율 |
| 8 | Minimal-Deviation-Insert | 최소 우회 삽입 | 안정성 |
| 9 | Shortest-ETA-First | 도착시간 순 | Nav2 인식 |

### 9.2 Nearest-First 알고리즘

```
알고리즘: Nearest-First

입력: robot_position, patrol_points, current_time
출력: visit_order

1. remaining ← {0, 1, ..., N-1}
2. current_pos ← robot_position
3. visit_order ← []

4. while remaining ≠ ∅:
5.     nearest ← argmin_{i ∈ remaining} A*(current_pos, point_i)
6.     visit_order.append(nearest)
7.     remaining.remove(nearest)
8.     current_pos ← point_{nearest}.position

9. return visit_order
```

### 9.3 Overdue-ETA-Balance 스코어 함수

$$
score(i) = \alpha \cdot (t_{current} - t_{last\_visit,i}) - \beta \cdot d_{A*}(pos, point_i)
$$

여기서:
- $\alpha$ = 1.0 (긴급도 가중치)
- $\beta$ = 0.1 (효율성 가중치)
- $d_{A*}$ = A* 경로 거리

### 9.4 Balanced-Coverage 알고리즘

```
알고리즘: Balanced-Coverage (Minimax)

입력: robot_position, patrol_points, current_time
출력: visit_order (최대 갭 최소화)

1. remaining ← {0, 1, ..., N-1}
2. projected_visit_times ← {i: last_visit_time[i] for i in remaining}
3. current_pos ← robot_position
4. estimated_time ← current_time
5. visit_order ← []

6. while remaining ≠ ∅:
7.     best_idx ← None
8.     best_max_gap ← ∞
9.
10.    for i in remaining:
11.        eta ← A*(current_pos, point_i) / velocity
12.        arrival ← estimated_time + eta
13.        temp_times ← projected_visit_times.copy()
14.        temp_times[i] ← arrival
15.        max_gap ← max_{j ∈ remaining}(estimated_time + eta - temp_times[j])
16.
17.        if max_gap < best_max_gap:
18.            best_max_gap ← max_gap
19.            best_idx ← i
20.
21.    visit_order.append(best_idx)
22.    remaining.remove(best_idx)
23.    projected_visit_times[best_idx] ← estimated_time + eta
24.    current_pos ← point_{best_idx}.position
25.    estimated_time ← projected_visit_times[best_idx]

26. return visit_order
```

---

## 10. 신경망 아키텍처

### 10.1 Actor-Critic 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Actor-Critic Network                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    입력: Observation (88D)                                   │
│           │                                                  │
│           ▼                                                  │
│    ┌──────────────────────────────────────────┐             │
│    │         Shared Encoder                    │             │
│    │   Linear(88, 256) → ReLU                 │             │
│    │   Linear(256, 256) → ReLU                │             │
│    └──────────────────────────────────────────┘             │
│           │                                                  │
│     ┌─────┴─────┐                                           │
│     ▼           ▼                                           │
│  ┌──────┐   ┌──────┐                                        │
│  │ Actor │   │Critic│                                        │
│  │ Head  │   │ Head │                                        │
│  └──┬────┘   └──────┘                                        │
│     │            │                                           │
│  ┌──┴────┐       │                                           │
│  ▼       ▼       ▼                                           │
│ Mode   Replan  Value                                         │
│ (2D)    (6D)    (1D)                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 레이어 구성

| 레이어 | 입력 | 출력 | 활성화 |
|--------|------|------|--------|
| encoder_1 | 88 | 256 | ReLU |
| encoder_2 | 256 | 256 | ReLU |
| mode_head | 256 | 2 | None (logits) |
| replan_head | 256 | K | None (logits) |
| critic_head | 256 | 1 | None |

### 10.3 초기화 전략

**Orthogonal Initialization** 사용:

$$
W \leftarrow \text{orthogonal}(\sqrt{2})
$$

마지막 레이어는 작은 스케일로 초기화:

```python
layer_init(critic_head, std=1.0)
layer_init(mode_head, std=0.01)
layer_init(replan_head, std=0.01)
```

### 10.4 순전파

```python
def forward(self, obs, action_mask=None):
    # 공유 인코더
    features = self.encoder(obs)

    # 액터 헤드
    mode_logits = self.mode_head(features)
    replan_logits = self.replan_head(features)

    # 행동 마스킹
    if action_mask is not None:
        mode_mask = action_mask[:, :2]
        replan_mask = action_mask[:, 2:]
        mode_logits = mode_logits.masked_fill(~mode_mask.bool(), -1e9)
        replan_logits = replan_logits.masked_fill(~replan_mask.bool(), -1e9)

    # 크리틱 헤드
    value = self.critic_head(features)

    return mode_logits, replan_logits, value.squeeze(-1)
```

---

## 11. 핵심 변수 정의

### 11.1 상태 변수

| 변수명 | 타입 | 단위 | 설명 |
|--------|------|------|------|
| `robot.x` | float | m | 로봇 X 좌표 |
| `robot.y` | float | m | 로봇 Y 좌표 |
| `robot.heading` | float | rad | 로봇 방향 (0=동, 반시계) |
| `robot.velocity` | float | m/s | 선속도 |
| `robot.angular_velocity` | float | rad/s | 각속도 |
| `robot.battery_level` | float | [0,1] | 배터리 잔량 비율 |
| `patrol_point.last_visit_time` | float | s | 마지막 방문 시각 |
| `patrol_point.priority` | float | [0,1] | 순찰 지점 중요도 |
| `event.urgency` | float | [0,1] | 이벤트 긴급도 |
| `event.risk_level` | int | 1-9 | 산업안전 위험 등급 |
| `event.detection_time` | float | s | 이벤트 감지 시각 |
| `current_time` | float | s | 에피소드 경과 시간 |

### 11.2 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `map_width` | 50.0 | 맵 너비 (m) |
| `map_height` | 50.0 | 맵 높이 (m) |
| `num_patrol_points` | 4 | 순찰 지점 개수 |
| `num_candidates` | 6 | 재계획 후보 개수 |
| `max_episode_steps` | 200 | 최대 에피소드 스텝 |
| `max_episode_time` | 600.0 | 최대 에피소드 시간 (s) |
| `robot_max_velocity` | 1.5 | 최대 선속도 (m/s) |
| `lidar_num_channels` | 64 | LiDAR 채널 수 |
| `lidar_max_range` | 10.0 | LiDAR 최대 범위 (m) |
| `event_generation_rate` | 5.0 | 에피소드당 이벤트 발생률 |

### 11.3 학습 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `learning_rate` | 3e-4 | 학습률 |
| `gamma` | 0.99 | 할인율 |
| `gae_lambda` | 0.95 | GAE 람다 |
| `clip_epsilon` | 0.2 | PPO 클리핑 |
| `value_loss_coef` | 0.5 | 가치 손실 계수 |
| `entropy_coef` | 0.01 | 엔트로피 계수 |
| `max_grad_norm` | 0.5 | 그래디언트 클리핑 |
| `num_steps` | 2048 | 롤아웃 스텝 수 |
| `num_epochs` | 10 | PPO 에폭 수 |
| `batch_size` | 256 | 미니배치 크기 |

---

## 12. 파일별 상세 설명

### 12.1 핵심 알고리즘 (`src/rl_dispatch/algorithms/`)

#### 12.1.1 `ppo.py` - PPOAgent 클래스

**역할**: PPO 알고리즘 구현 및 학습 관리

**주요 메서드**:

| 메서드 | 설명 |
|--------|------|
| `__init__()` | 네트워크, 옵티마이저, 버퍼 초기화 |
| `get_action()` | 관측에서 행동 샘플링 (행동 마스킹 지원) |
| `update()` | PPO 정책 업데이트 수행 |
| `save()` / `load()` | 체크포인트 저장/로드 |

**핵심 변수**:
- `self.network`: ActorCriticNetwork 인스턴스
- `self.buffer`: RolloutBuffer 인스턴스
- `self.optimizer`: Adam 옵티마이저
- `self.global_step`: 전역 학습 스텝 카운터

#### 12.1.2 `networks.py` - ActorCriticNetwork 클래스

**역할**: 신경망 아키텍처 정의

**구성 요소**:
- `encoder`: 공유 특징 추출기 (MLP)
- `mode_head`: 모드 선택 액터 헤드
- `replan_head`: 재계획 전략 액터 헤드
- `critic_head`: 가치 함수 헤드

**핵심 함수**:
- `layer_init()`: Orthogonal 초기화 적용
- `forward()`: 순전파 및 행동 마스킹

#### 12.1.3 `buffer.py` - RolloutBuffer 클래스

**역할**: 경험 저장 및 GAE 계산

**주요 메서드**:
- `add()`: 트랜지션 저장 (SMDP nav_time 포함)
- `compute_returns_and_advantages()`: GAE 계산 (시간 기반 할인)
- `sample_minibatches()`: 미니배치 생성기
- `reset()`: 버퍼 초기화

**SMDP 확장**:
```python
# 시간 기반 할인율 계산
effective_gamma = gamma ** nav_time
effective_gae = (gamma * gae_lambda) ** nav_time
```

### 12.2 환경 (`src/rl_dispatch/env/`)

#### 12.2.1 `patrol_env.py` - PatrolEnv 클래스

**역할**: Gymnasium 환경 구현

**주요 메서드**:

| 메서드 | 설명 |
|--------|------|
| `reset()` | 환경 초기화, 초기 관측 반환 |
| `step()` | SMDP 스텝 실행 |
| `_compute_action_mask()` | 유효 행동 마스크 계산 |
| `_determine_navigation_goal()` | 행동에서 목표 위치 결정 |
| `_maybe_generate_event()` | 포아송 프로세스로 이벤트 생성 |
| `_simulate_lidar()` | Bresenham 레이캐스팅으로 LiDAR 시뮬레이션 |
| `_raycast()` | 단일 레이 캐스팅 |

**핵심 로직**:
1. 행동 파싱 (모드, 재계획)
2. 행동 마스킹 적용
3. Nav2 내비게이션 실행
4. 로봇 상태 업데이트
5. 이벤트 해결 처리
6. 새 이벤트 생성 (포아송)
7. 보상 계산
8. 종료 조건 체크

### 12.3 보상 (`src/rl_dispatch/rewards/`)

#### 12.3.1 `reward_calculator.py`

**클래스**:

| 클래스 | 역할 |
|--------|------|
| `RewardCalculator` | 4-컴포넌트 보상 계산 |
| `RewardNormalizer` | 전역 보상 정규화 (미사용) |
| `ComponentNormalizer` | 컴포넌트별 정규화 (Phase 2) |

**주요 메서드**:
- `calculate()`: 전체 보상 계산
- `_calculate_event_reward()`: SLA 기반 이벤트 보상
- `_calculate_patrol_reward()`: Delta Coverage 순찰 보상
- `_calculate_safety_reward()`: 충돌/실패 페널티
- `_calculate_efficiency_reward()`: 거리 페널티

### 12.4 계획 (`src/rl_dispatch/planning/`)

#### 12.4.1 `candidate_generator.py`

**클래스 계층**:
```
CandidateGenerator (ABC)
├── KeepOrderGenerator
├── NearestFirstGenerator
├── MostOverdueFirstGenerator
├── OverdueETABalanceGenerator
├── RiskWeightedGenerator
├── BalancedCoverageGenerator
├── OverdueThresholdFirstGenerator
├── WindowedReplanGenerator
├── MinimalDeviationInsertGenerator
└── ShortestETAFirstGenerator
```

**CandidateFactory**:
- 모든 생성기 인스턴스화 및 관리
- `generate_all()`: 전체 후보 생성
- `set_nav_interface()`: A* 경로탐색기 연결

### 12.5 내비게이션 (`src/rl_dispatch/navigation/`)

#### 12.5.1 `nav2_interface.py`

**클래스**:
- `NavigationInterface` (ABC): 내비게이션 추상 인터페이스
- `SimulatedNav2`: 시뮬레이션용 Nav2 구현

**주요 메서드**:
- `navigate_to_goal()`: 목표까지 이동 시뮬레이션
- `get_eta()`: 예상 도착 시간 계산

#### 12.5.2 `pathfinding.py`

**클래스**:
- `AStarPathfinder`: A* 경로 탐색기

**주요 메서드**:
- `find_path()`: A* 경로 탐색
- `get_distance()`: 경로 거리 반환
- `path_exists()`: 경로 존재 여부 확인

**좌표 변환**:
```python
def world_to_grid(x, y):
    return (int(x / resolution), int(y / resolution))

def grid_to_world(gx, gy):
    return ((gx + 0.5) * resolution, (gy + 0.5) * resolution)
```

### 12.6 타입 정의 (`src/rl_dispatch/core/`)

#### 12.6.1 `types.py`

**데이터 클래스**:

| 클래스 | 용도 |
|--------|------|
| `ActionMode` | 행동 모드 열거형 (PATROL=0, DISPATCH=1) |
| `PatrolPoint` | 순찰 지점 정보 |
| `Event` | 이벤트 정보 |
| `RobotState` | 로봇 상태 |
| `Candidate` | 후보 경로 정보 |
| `Action` | 복합 행동 |
| `State` | 전체 환경 상태 |
| `Observation` | 88D 관측 벡터 |
| `RewardComponents` | 보상 컴포넌트 분해 |
| `EpisodeMetrics` | 에피소드 메트릭 |

#### 12.6.2 `config.py`

**설정 클래스**:

| 클래스 | 용도 |
|--------|------|
| `EnvConfig` | 환경 설정 (맵, 순찰지점, 이벤트) |
| `RewardConfig` | 보상 가중치 및 파라미터 |
| `TrainingConfig` | PPO 학습 하이퍼파라미터 |
| `NetworkConfig` | 신경망 아키텍처 설정 |

#### 12.6.3 `event_types.py`

**한국 산업안전보건 기준 (KOSHA)** 적용:

| 카테고리 | 위험도 범위 | 예시 |
|----------|-------------|------|
| 화재/폭발 | 7-9 | 화재발생, 가스폭발, 전기화재 |
| 침입/보안 | 4-8 | 무단침입, 절도시도, 보안위반 |
| 낙하/추락 | 6-9 | 고소추락, 물체낙하, 구조물붕괴 |
| 누수/누출 | 4-8 | 가스누출, 위험물질유출, 수도누수 |
| 설비고장 | 3-7 | 기계고장, 전력차단, 설비이상 |

---

## 13. 실험 설정

### 13.1 맵 구성

| 맵 | 크기 | 순찰지점 | 특징 |
|----|------|----------|------|
| default | 50×50m | 4 | 기본 사각형 |
| large_square | 100×100m | 4 | 대형 오픈 |
| corridor | 80×20m | 6 | 긴 복도 |
| l_shaped | 60×60m | 6 | L자 형태 |
| office_building | 40×30m | 40+ | 사무실 내부 |
| campus | 150×120m | 16 | 대형 캠퍼스 |
| warehouse | 80×60m | 20+ | 창고 레이아웃 |

### 13.2 평가 메트릭

| 메트릭 | 설명 | 계산 |
|--------|------|------|
| Episode Return | 에피소드 누적 보상 | $\sum_t R_t$ |
| Event Response Rate | 이벤트 대응률 | responded / detected |
| Event Success Rate | 이벤트 성공률 | successful / detected |
| Avg Event Delay | 평균 대응 지연 | mean(response_time - detection_time) |
| Patrol Coverage | 순찰 커버리지 | visited_on_time / total |
| Safety Violations | 안전 위반 횟수 | collisions + nav_failures |
| Total Distance | 총 이동 거리 | $\sum_t d_t$ |

### 13.3 베이스라인 정책

| ID | 정책명 | 설명 |
|----|--------|------|
| B0 | Always Patrol | 항상 순찰 모드 |
| B1 | Always Dispatch | 항상 디스패치 모드 |
| B2 | Threshold | urgency > 0.7 이면 디스패치 |
| B3 | Urgency-Based | 긴급도 기반 확률적 디스패치 |
| B4 | Heuristic | 휴리스틱 결정 트리 |

---

## 14. 참고문헌

### 14.1 강화학습 알고리즘

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347
2. Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv:1506.02438
3. Mnih, V., et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." ICML 2016

### 14.2 로봇 순찰 및 경로 계획

4. Portugal, D., & Rocha, R. (2011). "A Survey on Multi-robot Patrolling Algorithms." DARS
5. Machado, A., et al. (2002). "Multi-Agent Patrolling: An Empirical Analysis of Alternative Architectures." MABS
6. Hart, P. E., et al. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." IEEE SSC

### 14.3 Semi-MDP 및 옵션 프레임워크

7. Sutton, R. S., et al. (1999). "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning." Artificial Intelligence
8. Puterman, M. L. (1994). "Markov Decision Processes: Discrete Stochastic Dynamic Programming." Wiley

### 14.4 Nav2 및 ROS2

9. Macenski, S., et al. (2020). "The Marathon 2: A Navigation System." arXiv:2003.00368
10. ROS 2 Documentation. https://docs.ros.org/

### 14.5 Unitree Go2

11. Unitree Robotics. "Go2 Technical Documentation." https://www.unitree.com/

---

## 부록 A: 수식 요약

### A.1 PPO 손실 함수

$$
L(\theta) = -\mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right] + c_1 \mathbb{E}_t \left[ (V_\theta(s_t) - V_t^{target})^2 \right] - c_2 H(\pi_\theta)
$$

### A.2 SMDP GAE

$$
A_t = \sum_{l=0}^{\infty} \left( \prod_{i=t}^{t+l-1} \gamma^{\tau_i} \lambda^{\tau_i} \right) \delta_{t+l}
$$

### A.3 총 보상 함수

$$
R_{total} = w_{evt} \cdot \frac{R^{evt}_{raw} - \mu_{evt}}{\sigma_{evt}} + w_{pat} \cdot \frac{R^{pat}_{raw} - \mu_{pat}}{\sigma_{pat}} + w_{safe} \cdot R^{safe} + w_{eff} \cdot \frac{R^{eff}_{raw} - \mu_{eff}}{\sigma_{eff}}
$$

### A.4 Octile Distance

$$
h(n) = \max(|dx|, |dy|) + (\sqrt{2} - 1) \cdot \min(|dx|, |dy|)
$$

---

## 부록 B: 코드 예제

### B.1 환경 사용 예제

```python
from rl_dispatch.env import PatrolEnv
from rl_dispatch.core.config import EnvConfig, RewardConfig

# 환경 생성
env_config = EnvConfig(
    map_width=50.0,
    map_height=50.0,
    num_patrol_points=4,
    event_generation_rate=5.0
)
reward_config = RewardConfig(
    w_event=1.0,
    w_patrol=0.5,
    w_safety=2.0,
    w_efficiency=0.1
)

env = PatrolEnv(env_config, reward_config)

# 에피소드 실행
obs, info = env.reset(seed=42)
total_reward = 0

for step in range(200):
    action = env.action_space.sample()  # 또는 정책에서 선택
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print(f"Total reward: {total_reward:.2f}")
```

### B.2 PPO 학습 예제

```python
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.core.config import TrainingConfig, NetworkConfig

# 에이전트 생성
agent = PPOAgent(
    obs_dim=88,
    num_replan_strategies=6,
    training_config=TrainingConfig(
        learning_rate=3e-4,
        gamma=0.99,
        num_steps=2048,
        num_epochs=10
    ),
    network_config=NetworkConfig(
        encoder_hidden_dims=[256, 256]
    )
)

# 학습 루프
obs, info = env.reset()
for global_step in range(1_000_000):
    # 행동 선택
    action, log_prob, value = agent.get_action(
        obs, action_mask=info.get('action_mask')
    )

    # 환경 스텝
    next_obs, reward, terminated, truncated, info = env.step(action)

    # 버퍼에 저장
    agent.buffer.add(
        obs, action, reward, value, log_prob,
        terminated, info.get('nav_time', 1.0)
    )

    obs = next_obs

    # 버퍼가 가득 차면 업데이트
    if agent.buffer.is_full():
        stats = agent.update()
        agent.buffer.reset()

    if terminated or truncated:
        obs, info = env.reset()
```

---

*문서 버전: 1.0*
*마지막 업데이트: 2026-01-04*
*작성자: Claude Code*
