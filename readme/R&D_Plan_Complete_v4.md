# CCTV 이벤트 기반 순찰로봇의 출동·재스케줄링 통합 운영정책 강화학습 및 Nav2 기반 Sim2Real 검증

## 연구 및 개발 계획서 (완전판)

---

**문서 버전:** v3.0  
**작성일:** 2024년 12월  
**목적:** 개발팀이 즉시 구현에 착수할 수 있는 완전한 기술 명세 제공

---

## 목차

1. [연구 개요](#1-연구-개요)
2. [문제 정의 및 수학적 모델링](#2-문제-정의-및-수학적-모델링)
3. [MDP/SMDP 정형화](#3-mdpsmdp-정형화)
4. [상태 공간 상세 정의](#4-상태-공간-상세-정의)
5. [행동 공간 상세 정의](#5-행동-공간-상세-정의)
6. [보상 함수 설계](#6-보상-함수-설계)
7. [재스케줄링 후보 생성 시스템](#7-재스케줄링-후보-생성-시스템)
8. [Nav2 통합 인터페이스](#8-nav2-통합-인터페이스)
9. [시뮬레이션 환경 구축](#9-시뮬레이션-환경-구축)
10. [강화학습 알고리즘 설계](#10-강화학습-알고리즘-설계)
11. [실험 설계 및 평가 지표](#11-실험-설계-및-평가-지표)
12. [개발 로드맵 및 WBS](#12-개발-로드맵-및-wbs)
13. [프로젝트 구조 및 코드 명세](#13-프로젝트-구조-및-코드-명세)
14. [참고 문헌](#14-참고-문헌)

---

# 1. 연구 개요

## 1.1 연구 배경 및 필요성

### 1.1.1 문제 상황

산업현장에서 CCTV 기반 이벤트 감지 시스템은 **오탐(False Positive)**과 **미탐(False Negative)**이 존재하며, 특히 "회색 영역(gray-zone)" 이벤트는 현장 확인이 필수적이다. 기존의 인력 기반 대응은 다음과 같은 한계를 가진다:

- **비용 과다:** 24시간 상시 인력 배치의 경제적 부담
- **지속 가능성 저하:** 인력 피로도 및 집중력 저하로 인한 대응 품질 하락
- **확장성 제한:** 대규모 시설에서의 인력 확보 어려움

### 1.1.2 기존 연구의 한계

| 연구 분야 | 주요 초점 | 한계점 |
|-----------|-----------|--------|
| 로봇 내비게이션 | 주행 안정성, 경로 계획 | 운영 의사결정(출동/재스케줄링) 미고려 |
| 순찰 경로 최적화 | 정적 TSP/VRP 기반 | 동적 이벤트 대응 불가 |
| 이벤트 대응 시스템 | 단순 출동 규칙 | 장기 커버리지 비용 미반영 |
| 강화학습 로봇 제어 | Low-level 제어 | 운영 레벨 의사결정 미적용 |

### 1.1.3 본 연구의 차별성

본 연구는 다음과 같은 핵심 차별점을 가진다:

1. **운영정책 통합 학습:** 출동(Dispatch)과 재스케줄링(Rescheduling)을 단일 강화학습 정책으로 통합
2. **SMDP 기반 모델링:** Nav2 goal 수행 시간을 반영한 Semi-MDP 구조
3. **후보 기반 재스케줄링:** 휴리스틱 생성 + RL 선택의 하이브리드 구조로 조합 폭발 억제
4. **운영 Sim2Real:** Nav2 기반으로 실제 내비게이션 실패/지연을 포함한 검증

## 1.2 연구 목표

### 1.2.1 최종 목표 (12개월)

> **단일 로봇 순찰 환경에서 출동·재스케줄링 통합 운영정책을 강화학습으로 학습하고, Nav2 기반 Isaac Sim→Go2 실기기로 정책을 검증한다.**

### 1.2.2 정량적 목표

| 지표 | 기호 | 목표값 | 비교 기준 |
|------|------|--------|-----------|
| 평균 커버리지 공백비용 | $\bar{C}_{gap}$ | 20% 이상 감소 | 베이스라인 대비 |
| 이벤트 응답시간 | $T_{resp}$ | 유지 또는 개선 | 베이스라인 대비 |
| Nav2 실패율 | $F_{nav}$ | 동등 이하 | 베이스라인 대비 |
| 이벤트 성공률 | $S_{evt}$ | 95% 이상 | 절대 기준 |

### 1.2.3 연구 산출물

- **논문:** 중간 논문 3편 + 최종 SCI 저널 1편
- **코드:** 오픈소스 가능한 시뮬레이션 환경 및 학습 코드
- **실증:** Go2 로봇 기반 실환경 검증 데이터

## 1.3 연구 범위 및 가정

### 1.3.1 연구 범위

| 항목 | 포함 | 제외 |
|------|------|------|
| 이동 제어 | Nav2 활용 (고정) | Low-level 제어 학습 |
| 의사결정 | 출동/재스케줄링 | 배터리 충전 스케줄링 (4개월 이후) |
| 로봇 수 | 단일 로봇 (0-6개월) | 멀티로봇 (7개월 이후 확장) |
| 맵 | 정적 맵 (초기) | 동적 SLAM |
| 이벤트 | 단일 이벤트 (0-3개월) | 멀티 이벤트 (4개월 이후) |

### 1.3.2 핵심 가정

1. **Nav2 신뢰성:** Nav2가 이동/회피/제어를 안정적으로 수행
2. **로컬라이제이션:** Ground-truth 또는 안정적 AMCL 사용
3. **이벤트 입력:** CCTV 인식 모델의 출력을 입력으로 사용 (모델 개선은 범위 외)
4. **통신 안정성:** 로봇-서버 간 통신 지연 무시 가능

---

# 2. 문제 정의 및 수학적 모델링

## 2.1 문제의 수학적 정의

### 2.1.1 핵심 문제 (한 문장)

> **동적 이벤트가 발생하면 "출동"과 "출동 후 순찰 재스케줄링"을 단일 정책 $\pi_\theta$가 통합적으로 결정하여, 장기적으로 (이벤트 대응 지연 + 순찰 공백 + 안전위험 + 에너지 비용)을 최소화한다.**

### 2.1.2 최적화 목표

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$$

여기서:
- $\pi$: 정책 함수
- $\gamma \in (0, 1)$: 할인율 (예: 0.99)
- $r_t$: 시점 $t$에서의 보상
- $T$: 에피소드 종료 시점

## 2.2 변수 정의표

### 2.2.1 세트 및 인덱스

| 기호 | 의미 | 비고 |
|------|------|------|
| $\mathcal{W}$ | 맵(워크스페이스) | 2D 평면, 연속 좌표계 |
| $\mathcal{O}$ | 장애물 집합 | Axis-aligned box 또는 polygon |
| $\mathcal{P} = \{p_1, \ldots, p_M\}$ | 순찰 포인트 집합 | $M$: 포인트 수 (예: 12-40개) |
| $e_t$ | 현재 이벤트 | 없으면 $\emptyset$ |
| $\mathcal{C}_k = \{C_k^{(0)}, \ldots, C_k^{(N)}\}$ | 재스케줄 후보 집합 | $N+1$개 후보 |

### 2.2.2 로봇 상태 변수

| 기호 | 의미 | 단위/범위 | 설명 |
|------|------|-----------|------|
| $(x_t, y_t)$ | 로봇 위치 | m | 맵 프레임 기준 |
| $\psi_t$ | 로봇 yaw | rad | $[-\pi, \pi]$ |
| $(v_t, \omega_t)$ | 선속/각속 | m/s, rad/s | 현재 속도 |
| $b_t$ | 배터리 SoC | $[0, 1]$ | 잔량 비율 |
| $L_t \in \mathbb{R}^K$ | LiDAR 거리 샘플 | m | $K$: 샘플 수 (예: 64) |

### 2.2.3 이벤트 상태 변수

| 기호 | 의미 | 범위 | 설명 |
|------|------|------|------|
| $z_t \in \{0, 1\}$ | 이벤트 존재 여부 | binary | 1이면 출동 의사결정 필요 |
| $(x_t^e, y_t^e)$ | 이벤트 위치 | m | CCTV/맵 좌표 |
| $u_t$ | 이벤트 긴급도 | $[0, 1]$ 또는 $\{1,2,3\}$ | 높을수록 긴급 |
| $\tau_t$ | 이벤트 경과시간 | s | 발생 후 경과 |
| $c_t$ | 이벤트 신뢰도 | $[0, 1]$ | CCTV 모델 confidence |

### 2.2.4 순찰 스케줄 변수

| 기호 | 의미 | 정의 |
|------|------|------|
| $\pi_t^{pat}$ | 현재 순찰 포인트 순서 | 길이 $M$의 순열 |
| $i_t$ | 현재 목표 포인트 index | $\in \{1, \ldots, M\}$ |
| $g_i$ | 포인트 $i$의 공백 시간 | $g_i = T - T_i^{last}$ |
| $T_i^{last}$ | 포인트 $i$의 마지막 방문 시간 | s |
| $G_{th}$ | 공백 임계값 | s (예: 300s) |

### 2.2.5 후보 관련 변수

| 기호 | 의미 | 정의 |
|------|------|------|
| $\Pi_H^{(j)}$ | 후보 $j$의 방문 시퀀스 | $[q_1, \ldots, q_H]$, $H$: horizon |
| $\ell_L^{(j)}$ | 후보 $j$의 경로 길이 | m |
| $\ell_T^{(j)}$ | 후보 $j$의 ETA 추정 | s |
| $\Theta^{(j)}$ | 후보 $j$의 누적 회전량 | rad |
| $w_i$ | 포인트 $i$의 중요도 | $\geq 0$, 기본값 1 |

---

# 3. MDP/SMDP 정형화

## 3.1 기본 MDP 정의

### 3.1.1 MDP 튜플

$$M = \langle \mathcal{S}, \mathcal{A}, P, r, \gamma \rangle$$

| 요소 | 정의 | 설명 |
|------|------|------|
| $\mathcal{S}$ | 상태 공간 | 로봇 + 이벤트 + 순찰 상태 |
| $\mathcal{A}$ | 행동 공간 | 출동 여부 + 재스케줄 후보 선택 |
| $P(s_{t+1} \mid s_t, a_t)$ | 전이 확률 | Isaac Sim 물리엔진 + 이벤트 생성 규칙 |
| $r(s_t, a_t, s_{t+1})$ | 보상 함수 | 다중 목표 가중합 |
| $\gamma$ | 할인율 | 0.99 (권장) |

### 3.1.2 시간 모델 선택

**Semi-MDP (SMDP) 채택 이유:**

Nav2 goal 수행은 가변 시간 $\Delta t_k$를 동반하므로, 고정 timestep MDP보다 SMDP가 적합하다.

$$\text{SMDP: } s_k \xrightarrow{a_k, \Delta t_k} s_{k+1}$$

| 모델 | 의사결정 시점 | 장점 | 단점 |
|------|--------------|------|------|
| 고정 MDP | 매 0.1s | 단순 | 불필요한 결정 과다 |
| **SMDP** | Goal 완료 시점 | 효율적, 안정적 | 구현 복잡도 증가 |

## 3.2 SMDP 상세 정의

### 3.2.1 의사결정 시점 (Decision Epoch)

RL은 다음 시점에서만 행동을 선택한다:

1. **NAV_DONE:** NavigateToPose 결과 수신 (SUCCEEDED/ABORTED/TIMEOUT)
2. **EVENT_ARRIVAL:** 이벤트 큐에 신규 이벤트 도착 (멀티 이벤트 시)
3. **SAFETY:** 충돌 위험/금지구역 침범/통신 장애 등 안전 스위치

### 3.2.2 매크로 스텝 정의

매크로 스텝 $k$에서:

```
시점 T_k에서 행동 a_k 선택
  → Nav2가 goal 수행 (시간 Δt_k 소요)
  → 시점 T_{k+1} = T_k + Δt_k에서 다음 상태 s_{k+1} 관측
```

### 3.2.3 할인율 조정

SMDP에서 가변 시간을 반영한 할인:

$$\gamma_k = \gamma^{\Delta t_k / \Delta t_{base}}$$

여기서 $\Delta t_{base}$는 기준 시간 단위 (예: 1초)

## 3.3 SMDP 상태 전이 다이어그램

![SMDP 상태 전이 다이어그램](./diagrams/rendered/12_smdp_transition.png)




---

# 4. 상태 공간 상세 정의

## 4.1 관측 벡터 구성

![상태 공간 구조](./diagrams/rendered/02_state_space.png)


### 4.1.1 전체 관측 벡터

정책 입력은 완전 상태 대신 관측 $o_t$로 정의한다:

$$o_t = \left[\underbrace{\Delta x_g, \Delta y_g}_{\text{목표 상대벡터}}, \underbrace{\cos\psi_t, \sin\psi_t}_{\text{각도 표현}}, \underbrace{v_t, \omega_t}_{\text{동역학}}, \underbrace{b_t}_{\text{배터리}}, \underbrace{L_t^{(1)}, \ldots, L_t^{(K)}}_{\text{LiDAR}}, \underbrace{z_t, u_t, c_t, \tilde{\tau}_t}_{\text{이벤트}}, \underbrace{\tilde{d}_t^{pat}, \tilde{\kappa}_t^{pat}}_{\text{순찰 요약}}\right]$$

### 4.1.2 차원별 상세 정의

| 구성 요소 | 차원 | 범위 | 정규화 방법 |
|-----------|------|------|-------------|
| 목표 상대벡터 $(\Delta x_g, \Delta y_g)$ | 2 | m | $\div d_{max}$ (예: 50m) |
| 각도 표현 $(\cos\psi, \sin\psi)$ | 2 | $[-1, 1]$ | 이미 정규화 |
| 동역학 $(v, \omega)$ | 2 | m/s, rad/s | $\div v_{max}, \div \omega_{max}$ |
| 배터리 $b$ | 1 | $[0, 1]$ | 이미 정규화 |
| LiDAR $L^{(1:K)}$ | 64 | m | $\div L_{max}$ (예: 10m) |
| 이벤트 존재 $z$ | 1 | $\{0, 1\}$ | binary |
| 이벤트 긴급도 $u$ | 1 | $[0, 1]$ | 이미 정규화 |
| 이벤트 신뢰도 $c$ | 1 | $[0, 1]$ | 이미 정규화 |
| 이벤트 경과시간 $\tilde{\tau}$ | 1 | $[0, 1]$ | $\div \tau_{max}$ (예: 300s) |
| 순찰 요약 $(\tilde{d}^{pat}, \tilde{\kappa}^{pat})$ | 2 | $[0, 1]$ | 정규화 |
| **총 차원** | **77** | | |

### 4.1.3 순찰 요약 변수 정의

$$\tilde{d}_t^{pat} = \frac{d(x_t, p_{i_t})}{d_{max}}$$

$$\tilde{\kappa}_t^{pat} = \frac{\bar{g}}{G_{th}}$$

여기서:
- $d(x_t, p_{i_t})$: 현재 위치에서 다음 순찰 포인트까지 거리
- $\bar{g} = \frac{1}{M}\sum_{i=1}^{M} g_i$: 평균 공백 시간

## 4.2 후보 피처 벡터

### 4.2.1 후보별 피처

각 재스케줄 후보 $C^{(j)}$에 대해:

$$f^{(j)} = \left[\tilde{\ell}_L^{(j)}, \tilde{\ell}_T^{(j)}, \tilde{\Theta}^{(j)}, \text{feasible}^{(j)}\right]$$

| 피처 | 정의 | 정규화 |
|------|------|--------|
| $\tilde{\ell}_L^{(j)}$ | 경로 길이 | $\div L_{max}$ |
| $\tilde{\ell}_T^{(j)}$ | ETA 추정 | $\div T_{max}$ |
| $\tilde{\Theta}^{(j)}$ | 누적 회전량 | $\div \pi$ |
| $\text{feasible}^{(j)}$ | 실행 가능 여부 | $\{0, 1\}$ |

### 4.2.2 후보 피처 행렬

$$F_k = \begin{bmatrix} f^{(0)} \\ f^{(1)} \\ \vdots \\ f^{(N)} \end{bmatrix} \in \mathbb{R}^{(N+1) \times 4}$$

### 4.2.3 확장 관측 벡터 (후보 포함)

```python
# 관측 벡터 구성 (Python 의사코드)
obs = np.concatenate([
    # 기본 관측 (77차원)
    goal_relative,      # 2
    heading_sincos,     # 2
    dynamics,           # 2
    battery,            # 1
    lidar,              # 64
    event_features,     # 4
    patrol_summary,     # 2
    
    # 후보 피처 (K_obs * 4차원)
    candidate_features.flatten(),  # K_obs * 4
    
    # 커버리지 요약 (3차원)
    gap_mean,           # 1
    gap_max,            # 1
    gap_std,            # 1
])
```

## 4.3 커버리지 상태 계산

### 4.3.1 공백 시간 계산

각 포인트 $i$의 공백 시간:

$$g_i(T) = T - T_i^{last}$$

### 4.3.2 공백 비용 함수

$$C_{gap}(T) = \frac{1}{M}\sum_{i=1}^{M} \max(0, g_i(T) - G_{th})$$

### 4.3.3 커버리지 요약 통계

| 통계량 | 정의 | 용도 |
|--------|------|------|
| $\bar{g}$ | $\frac{1}{M}\sum_i g_i$ | 평균 공백 |
| $g_{max}$ | $\max_i g_i$ | 최대 공백 |
| $\sigma_g$ | $\sqrt{\frac{1}{M}\sum_i (g_i - \bar{g})^2}$ | 공백 분산 |

```python
def coverage_metrics(gaps_s: List[float]) -> Tuple[float, float, float]:
    """커버리지 요약 통계 계산"""
    arr = np.array(gaps_s)
    gap_mean = float(np.mean(arr))
    gap_max = float(np.max(arr))
    gap_std = float(np.std(arr))
    return gap_mean, gap_max, gap_std
```



---

# 5. 행동 공간 상세 정의

## 5.1 행동 공간 구조

### 5.1 복합 행동 공간

![행동 공간 및 의사결정 흐름](./diagrams/rendered/03_action_space.png)
정의

행동 $a_k$는 두 개의 이산 구성요소로 정의된다:

$$a_k = (a_k^{mode}, a_k^{replan})$$

| 구성요소 | 타입 | 범위 | 의미 |
|----------|------|------|------|
| $a_k^{mode}$ | Bernoulli | $\{0, 1\}$ | 출동 여부 |
| $a_k^{replan}$ | Categorical | $\{0, 1, \ldots, N\}$ | 재스케줄 후보 선택 |

### 5.1.2 행동 해석

**$a_k^{mode}$ (출동 결정):**

| 값 | 의미 | 다음 Nav2 goal |
|----|------|----------------|
| 0 | 순찰 계속 | 다음 순찰 포인트 |
| 1 | 이벤트 출동 | 이벤트 위치 |

**$a_k^{replan}$ (재스케줄 후보 선택):**

| 값 | 의미 |
|----|------|
| 0 | 현재 순찰 일정 유지 (No-Replan) |
| 1 | Nearest-First 후보 |
| 2 | Most-Overdue-First 후보 |
| 3 | Overdue + ETA Balance 후보 |
| 4 | Risk-Weighted 후보 |
| 5 | Balanced-Coverage 후보 |

### 5.1.3 행동 조합 로직

```python
def interpret_action(a_mode: int, a_replan: int, 
                     event_active: bool, event_xy: Tuple[float, float],
                     candidates: List[Candidate]) -> Tuple[str, Tuple[float, float]]:
    """
    행동을 해석하여 다음 Nav2 goal을 결정
    
    Returns:
        goal_type: "EVENT" | "PATROL"
        goal_xy: (x, y) 좌표
    """
    # 재스케줄 적용 (항상)
    if a_replan != 0 and a_replan <= len(candidates):
        current_schedule = candidates[a_replan].seq
    
    # 출동 결정
    if a_mode == 1 and event_active:
        return "EVENT", event_xy
    else:
        # 순찰 계속: 현재 스케줄의 첫 번째 포인트
        next_point_idx = current_schedule[0] if current_schedule else 0
        return "PATROL", points_xy[next_point_idx]
```

## 5.2 행동 마스킹

### 5.2.1 마스킹 필요성

- **Infeasible 후보:** Nav2 planner가 경로를 찾지 못한 후보
- **이벤트 없음:** $z_t = 0$일 때 $a^{mode} = 1$ 무의미
- **안전 제약:** Keep-out zone으로 향하는 후보

### 5.2.2 마스킹 구현

```python
def compute_action_mask(event_active: bool, 
                        candidate_feasible: List[bool]) -> Dict[str, np.ndarray]:
    """
    행동 마스킹 계산
    
    Returns:
        mode_mask: [2] - 출동 가능 여부
        replan_mask: [N+1] - 각 후보 선택 가능 여부
    """
    # 출동 마스크: 이벤트가 있을 때만 출동 가능
    mode_mask = np.array([1.0, 1.0 if event_active else 0.0])
    
    # 재스케줄 마스크: feasible한 후보만 선택 가능
    replan_mask = np.array([1.0] + [1.0 if f else 0.0 for f in candidate_feasible])
    
    return {"mode_mask": mode_mask, "replan_mask": replan_mask}
```

### 5.2.3 마스킹 적용 (PPO)

```python
def masked_categorical_sample(logits: torch.Tensor, mask: torch.Tensor) -> int:
    """마스킹된 Categorical 분포에서 샘플링"""
    # 마스킹된 위치에 큰 음수 값 적용
    masked_logits = logits + (mask.log() + 1e-8)
    probs = F.softmax(masked_logits, dim=-1)
    return torch.multinomial(probs, 1).item()
```

## 5.3 의사결정 빈도

### 5.3.1 결정 시점 정의

Nav2 goal은 시간이 걸리므로, RL은 goal 단위로만 결정한다:

| 시점 | 결정 내용 | 빈도 |
|------|-----------|------|
| 이벤트 발생 | 출동 여부 + 재스케줄 | 이벤트당 1회 |
| 이벤트 처리 직후 | 재스케줄 | 처리당 1회 |
| 순찰 포인트 도착 | 재스케줄 (선택적) | 도착당 1회 |
| Nav2 실패 | 재스케줄 | 실패당 1회 |

### 5.3.2 결정 빈도 제한의 이점

- **학습 안정성:** 불필요한 고빈도 결정 제거
- **Nav2 호환성:** Goal 수행 중 간섭 방지
- **해석 가능성:** 각 결정의 의미 명확화

---

# 6. 보상 함수 설계

## 6.1 전체 보상 함수

![보상 함수 구조](./diagrams/rendered/04_reward_structure.png)


### 6.1.1 총 보상 정의

$$R_k = R_k^{evt} + R_k^{pat} + R_k^{safe} + R_k^{eff}$$

| 항목 | 기호 | 목적 | 가중치 |
|------|------|------|--------|
| 이벤트 대응 | $R_k^{evt}$ | 신속한 이벤트 처리 | $w_{evt}$ |
| 순찰 커버리지 | $R_k^{pat}$ | 공백 최소화 | $w_{pat}$ |
| 안전 | $R_k^{safe}$ | 충돌/실패 방지 | $w_{safe}$ |
| 효율 | $R_k^{eff}$ | 불필요한 이동 억제 | $w_{eff}$ |

### 6.1.2 가중치 초기값 (조정 대상)

```yaml
# configs/reward_weights.yaml
reward:
  # 이벤트 관련
  w_evt_delay: 1.0        # 이벤트 지연 페널티 가중치
  R_hit: 50.0             # 이벤트 처리 성공 보너스
  
  # 순찰 관련
  w_pat: 1.0              # 커버리지 공백 가중치 (통합학습 핵심)
  G_th: 300.0             # 공백 임계값 (초)
  
  # 안전 관련
  R_col: 100.0            # 충돌 페널티
  R_abort: 30.0           # Nav2 abort 페널티
  R_timeout: 20.0         # Nav2 timeout 페널티
  
  # 효율 관련
  lambda_L: 0.1           # 경로 길이 페널티 계수
```

## 6.2 이벤트 대응 보상 ($R_k^{evt}$)

### 6.2.1 구성 요소

$$R_k^{evt} = R_k^{evt,delay} + R_k^{evt,hit}$$

### 6.2.2 지연 페널티

이벤트가 미처리 상태로 남아있는 동안 누적 페널티:

$$R_k^{evt,delay} = -\alpha \cdot \Delta t_k \cdot \mathbf{1}[\text{event not handled}]$$

여기서:
- $\alpha$: 지연 페널티 계수 (예: 1.0)
- $\Delta t_k$: 매크로 스텝 $k$의 소요 시간
- $\mathbf{1}[\cdot]$: 지시 함수

### 6.2.3 처리 성공 보너스

$$R_k^{evt,hit} = R_{hit} \cdot \mathbf{1}[\text{event handled in macro-step } k]$$

### 6.2.4 코드 구현

```python
def compute_event_reward(event_active: bool, event_handled: bool,
                         delta_t: float, alpha: float = 1.0, 
                         R_hit: float = 50.0) -> Tuple[float, float]:
    """
    이벤트 관련 보상 계산
    
    Returns:
        r_delay: 지연 페널티
        r_hit: 처리 성공 보너스
    """
    r_delay = 0.0
    r_hit = 0.0
    
    if event_active and not event_handled:
        r_delay = -alpha * delta_t
    
    if event_handled:
        r_hit = R_hit
    
    return r_delay, r_hit
```

## 6.3 순찰 커버리지 보상 ($R_k^{pat}$)

### 6.3.1 공백 비용 정의

$$C_{gap}(T) = \frac{1}{M}\sum_{i=1}^{M} \max(0, g_i(T) - G_{th})$$

### 6.3.2 매크로 스텝 보상

공백 비용의 증가량(또는 적분)을 벌점으로 부여:

$$R_k^{pat} = -\int_{T_k}^{T_{k+1}} C_{gap}(t) \, dt \approx -\Delta t_k \cdot C_{gap}(T_k)$$

### 6.3.3 핵심 의의

> **이 항이 있어야 "출동을 하면 커버리지가 무너지는 비용"이 학습에 반영되어, 출동과 재스케줄링을 분해하지 않고 통합 학습하는 의미가 생긴다.**

### 6.3.4 코드 구현

```python
def compute_coverage_reward(gaps_s: List[float], delta_t: float,
                            G_th: float = 300.0, w_pat: float = 1.0) -> float:
    """
    순찰 커버리지 보상 계산
    
    Args:
        gaps_s: 각 포인트의 공백 시간 리스트
        delta_t: 매크로 스텝 소요 시간
        G_th: 공백 임계값
        w_pat: 가중치
    
    Returns:
        r_pat: 커버리지 보상 (음수)
    """
    M = len(gaps_s)
    C_gap = sum(max(0, g - G_th) for g in gaps_s) / M
    r_pat = -w_pat * delta_t * C_gap
    return r_pat
```

## 6.4 안전 보상 ($R_k^{safe}$)

### 6.4.1 충돌 페널티

$$R_k^{safe,col} = -R_{col} \cdot \mathbf{1}[\text{collision}]$$

### 6.4.2 Nav2 실패 페널티

$$R_k^{safe,nav} = -R_{abort} \cdot \mathbf{1}[\text{nav2 aborted}] - R_{timeout} \cdot \mathbf{1}[\text{nav2 timeout}]$$

### 6.4.3 총 안전 보상

$$R_k^{safe} = R_k^{safe,col} + R_k^{safe,nav}$$

### 6.4.4 코드 구현

```python
def compute_safety_reward(nav_status: str, collision: bool,
                          R_col: float = 100.0, R_abort: float = 30.0,
                          R_timeout: float = 20.0) -> float:
    """
    안전 관련 보상 계산
    
    Args:
        nav_status: Nav2 결과 ("SUCCEEDED", "ABORTED", "TIMEOUT")
        collision: 충돌 발생 여부
    
    Returns:
        r_safe: 안전 보상 (음수 또는 0)
    """
    r_safe = 0.0
    
    if collision:
        r_safe -= R_col
    
    if nav_status == "ABORTED":
        r_safe -= R_abort
    elif nav_status == "TIMEOUT":
        r_safe -= R_timeout
    
    return r_safe
```

## 6.5 효율 보상 ($R_k^{eff}$)

### 6.5.1 경로 길이 기반 비용

$$R_k^{eff} = -\lambda_L \cdot L_k$$

여기서 $L_k$는 Nav2가 제공하는 경로 길이

### 6.5.2 회전량 기반 비용 (선택적)

$$R_k^{eff,turn} = -\lambda_\Theta \cdot \Theta_k$$

### 6.5.3 코드 구현

```python
def compute_efficiency_reward(path_length: float, heading_change: float,
                              lambda_L: float = 0.1, 
                              lambda_Theta: float = 0.05) -> float:
    """
    효율 관련 보상 계산
    
    Args:
        path_length: 경로 길이 (m)
        heading_change: 누적 회전량 (rad)
    
    Returns:
        r_eff: 효율 보상 (음수)
    """
    r_eff = -lambda_L * path_length - lambda_Theta * heading_change
    return r_eff
```

## 6.6 보상 계산기 클래스

### 6.6.1 전체 구현

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import yaml

@dataclass
class RewardWeights:
    """보상 가중치 설정"""
    w_evt_delay: float = 1.0
    R_hit: float = 50.0
    w_pat: float = 1.0
    G_th: float = 300.0
    R_col: float = 100.0
    R_abort: float = 30.0
    R_timeout: float = 20.0
    lambda_L: float = 0.1
    lambda_Theta: float = 0.05
    
    @classmethod
    def from_yaml(cls, path: str) -> "RewardWeights":
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)['reward']
        return cls(**cfg)

class RewardCalculator:
    """보상 함수 계산기"""
    
    def __init__(self, weights: RewardWeights):
        self.w = weights
    
    def compute(self, 
                event_active: bool,
                event_handled: bool,
                gaps_s: List[float],
                delta_t: float,
                nav_status: str,
                collision: bool,
                path_length: float,
                heading_change: float) -> Dict[str, float]:
        """
        전체 보상 계산 (분해된 형태로 반환)
        
        Returns:
            Dict with keys: r_evt_delay, r_evt_hit, r_pat, r_safe, r_eff, r_total
        """
        # 이벤트 보상
        r_evt_delay = 0.0
        r_evt_hit = 0.0
        if event_active and not event_handled:
            r_evt_delay = -self.w.w_evt_delay * delta_t
        if event_handled:
            r_evt_hit = self.w.R_hit
        
        # 커버리지 보상
        M = len(gaps_s)
        C_gap = sum(max(0, g - self.w.G_th) for g in gaps_s) / max(M, 1)
        r_pat = -self.w.w_pat * delta_t * C_gap
        
        # 안전 보상
        r_safe = 0.0
        if collision:
            r_safe -= self.w.R_col
        if nav_status == "ABORTED":
            r_safe -= self.w.R_abort
        elif nav_status == "TIMEOUT":
            r_safe -= self.w.R_timeout
        
        # 효율 보상
        r_eff = -self.w.lambda_L * path_length - self.w.lambda_Theta * heading_change
        
        # 총 보상
        r_total = r_evt_delay + r_evt_hit + r_pat + r_safe + r_eff
        
        return {
            "r_evt_delay": r_evt_delay,
            "r_evt_hit": r_evt_hit,
            "r_pat": r_pat,
            "r_safe": r_safe,
            "r_eff": r_eff,
            "r_total": r_total
        }
```

## 6.7 보상 항 Ablation 설계

### 6.7.1 Ablation 조합

| 조합 | 이벤트 | 커버리지 | 안전 | 효율 | 목적 |
|------|--------|----------|------|------|------|
| A | ✓ | ✗ | ✓ | ✗ | 이벤트만 학습 |
| B | ✓ | ✓ | ✓ | ✗ | 통합학습 핵심 |
| C | ✓ | ✓ | ✓ | ✓ | 완전 모델 |

### 6.7.2 Ablation 스위치

```python
class RewardCalculatorAblation(RewardCalculator):
    """Ablation 실험용 보상 계산기"""
    
    def __init__(self, weights: RewardWeights, 
                 use_event: bool = True,
                 use_coverage: bool = True,
                 use_safety: bool = True,
                 use_efficiency: bool = True):
        super().__init__(weights)
        self.use_event = use_event
        self.use_coverage = use_coverage
        self.use_safety = use_safety
        self.use_efficiency = use_efficiency
    
    def compute(self, **kwargs) -> Dict[str, float]:
        result = super().compute(**kwargs)
        
        # Ablation 적용
        if not self.use_event:
            result["r_evt_delay"] = 0.0
            result["r_evt_hit"] = 0.0
        if not self.use_coverage:
            result["r_pat"] = 0.0
        if not self.use_safety:
            result["r_safe"] = 0.0
        if not self.use_efficiency:
            result["r_eff"] = 0.0
        
        # 총 보상 재계산
        result["r_total"] = (result["r_evt_delay"] + result["r_evt_hit"] + 
                            result["r_pat"] + result["r_safe"] + result["r_eff"])
        
        return result
```



---

# 7. 재스케줄링 후보 생성 시스템

![후보 생성 및 평가 시스템](./diagrams/rendered/05_candidate_system.png)


## 7.1 재스케줄 후보의 정확한 정의

### 7.1.1 개념 정의

> **재스케줄링 후보(Candidate)는 이벤트로 인해 교란된 순찰 운영을 복구하기 위해 생성되는 대체 순찰 계획(Plan)이다.**

### 7.1.2 수학적 정의

각 후보 $C_k^{(n)}$는 다음 구조를 가진다:

$$C_k^{(n)} := (\Pi_H^{(n)}, \text{resume\_rule}^{(n)})$$

여기서:
- $\Pi_H^{(n)} = [q_1, \ldots, q_H]$: 앞으로 방문할 포인트 $H$개의 순서
- $\text{resume\_rule}^{(n)}$: $H$개를 다 방문한 뒤 나머지를 이어갈 규칙

### 7.1.3 Rolling Horizon 정의

완전한 길이 $M$ 순열 대신, 앞으로 $H$개만 정하는 계획:

$$\Pi_H = [q_1, \ldots, q_H], \quad H \ll M$$

**권장값:** $H = 3$ 또는 $H = 5$

### 7.1.4 후보에 포함하지 않는 것 (경계 정의)

| 항목 | 포함 여부 | 담당 |
|------|-----------|------|
| 다음 방문 순서 | ✓ 포함 | RL 선택 |
| 로컬 충돌 회피 | ✗ 제외 | Nav2 |
| 속도 제어 | ✗ 제외 | Nav2 |
| 지도 업데이트/SLAM | ✗ 제외 | 범위 외 |
| 배터리 충전 스케줄 | ✗ 제외 | 4개월 이후 |

## 7.2 후보 6종 정의

### 7.2.1 공통 입력

```python
@dataclass
class CandidateInput:
    """후보 생성 공통 입력"""
    points_xy: List[Tuple[float, float]]  # 순찰 포인트 좌표
    gaps_s: List[float]                    # 각 포인트 공백 시간
    weights: List[float]                   # 포인트 중요도 (기본값 1)
    queue_current: List[int]               # 현재 순찰 큐
    H: int                                 # Rolling horizon
    start_xy: Tuple[float, float]          # 현재 로봇 위치
    event_xy: Optional[Tuple[float, float]] = None  # 이벤트 위치 (있을 경우)
```

### 7.2.2 후보 0: Keep-Order (No-Replan)

**정의:**

$$\Pi_H^{(0)} := \Pi_H^{current}$$

**의미:** 현재 순찰 큐에서 다음 $H$개 그대로 유지

**알고리즘:**
```python
def build_keep_order(queue_current: List[int], H: int) -> List[int]:
    """후보 0: 현재 순서 유지"""
    return list(queue_current[:H])
```

### 7.2.3 후보 1: Nearest-First (효율 우선)

**스코어 함수:**

$$S_1(i) = -\ell(\text{start} \to p_i)$$

**의미:** 가장 가까운 곳부터 방문 (이동 효율 최대화)

**알고리즘:**
```python
def build_nearest_first(points_xy: List[Tuple[float, float]], 
                        H: int, 
                        start_xy: Tuple[float, float],
                        remaining: Optional[Set[int]] = None) -> List[int]:
    """후보 1: 거리 최소 우선"""
    return greedy_select(
        points_xy=points_xy,
        gaps_s=[0.0] * len(points_xy),
        weights=[1.0] * len(points_xy),
        H=H,
        start_xy=start_xy,
        remaining=remaining,
        score_fn=lambda i, cur_xy: -euclid(cur_xy, points_xy[i])
    )
```

### 7.2.4 후보 2: Most-Overdue-First (공백 우선)

**스코어 함수:**

$$S_2(i) = g_i$$

**의미:** 가장 오래 비어있는 곳부터 방문 (커버리지 복구)

**알고리즘:**
```python
def build_overdue_first(points_xy: List[Tuple[float, float]],
                        gaps_s: List[float],
                        H: int,
                        start_xy: Tuple[float, float],
                        remaining: Optional[Set[int]] = None) -> List[int]:
    """후보 2: 공백 최대 우선"""
    return greedy_select(
        points_xy=points_xy,
        gaps_s=gaps_s,
        weights=[1.0] * len(points_xy),
        H=H,
        start_xy=start_xy,
        remaining=remaining,
        score_fn=lambda i, cur_xy: gaps_s[i]
    )
```

### 7.2.5 후보 3: Overdue + ETA Balance (균형형)

**스코어 함수:**

$$S_3(i) = g_i - \lambda \cdot \ell(\text{start} \to p_i)$$

**의미:** 공백도 줄이되 너무 멀면 미룸 (균형 추구)

**파라미터:** $\lambda > 0$ (초기값: 0.5~1.0)

**알고리즘:**
```python
def build_overdue_eta_balance(points_xy: List[Tuple[float, float]],
                              gaps_s: List[float],
                              H: int,
                              start_xy: Tuple[float, float],
                              lam: float = 0.7,
                              remaining: Optional[Set[int]] = None) -> List[int]:
    """후보 3: 공백-거리 균형"""
    return greedy_select(
        points_xy=points_xy,
        gaps_s=gaps_s,
        weights=[1.0] * len(points_xy),
        H=H,
        start_xy=start_xy,
        remaining=remaining,
        score_fn=lambda i, cur_xy: gaps_s[i] - lam * euclid(cur_xy, points_xy[i])
    )
```

### 7.2.6 후보 4: Risk-Weighted Overdue (중요구역 우선)

**스코어 함수:**

$$S_4(i) = w_i \cdot g_i - \lambda \cdot \ell(\text{start} \to p_i)$$

**의미:** 중요도가 높은 구역의 공백을 더 빨리 메움

**알고리즘:**
```python
def build_risk_weighted(points_xy: List[Tuple[float, float]],
                        gaps_s: List[float],
                        weights: List[float],
                        H: int,
                        start_xy: Tuple[float, float],
                        lam: float = 0.7,
                        remaining: Optional[Set[int]] = None) -> List[int]:
    """후보 4: 중요도 가중 공백 우선"""
    return greedy_select(
        points_xy=points_xy,
        gaps_s=gaps_s,
        weights=weights,
        H=H,
        start_xy=start_xy,
        remaining=remaining,
        score_fn=lambda i, cur_xy: (weights[i] * gaps_s[i]) - lam * euclid(cur_xy, points_xy[i])
    )
```

### 7.2.7 후보 5: Balanced-Coverage (최대 공백 억제형)

**목적:** 전체를 고르게 방문하여 최대 공백 폭발 방지

**스코어 함수:**

$$S_5(i) = -\max_k(g_k') - \lambda \cdot \ell(\text{start} \to p_i)$$

여기서 $g_k' = 0$ if $k = i$ (방문 시 리셋), else $g_k' = g_k$

**알고리즘:**
```python
def build_balanced_coverage(points_xy: List[Tuple[float, float]],
                            gaps_s: List[float],
                            H: int,
                            start_xy: Tuple[float, float],
                            lam: float = 0.5,
                            remaining: Optional[Set[int]] = None) -> List[int]:
    """후보 5: 최대 공백 억제"""
    if remaining is None:
        remaining = set(range(len(points_xy)))
    
    seq = []
    cur_xy = start_xy
    cur_gaps = list(gaps_s)
    
    for _ in range(H):
        if not remaining:
            break
        
        best_i, best_score = None, -float("inf")
        for i in remaining:
            # i를 방문했을 때의 새로운 최대 공백
            max_gap_after = max((0.0 if k == i else cur_gaps[k]) 
                               for k in range(len(cur_gaps)))
            travel = euclid(cur_xy, points_xy[i])
            score = -max_gap_after - lam * travel
            
            if score > best_score:
                best_i, best_score = i, score
        
        seq.append(best_i)
        remaining.remove(best_i)
        cur_xy = points_xy[best_i]
        cur_gaps[best_i] = 0.0  # 방문 시 공백 리셋
    
    return seq
```

## 7.3 공통 Greedy 선택 함수

### 7.3.1 유틸리티 함수

```python
import math
from typing import Callable, Set, Optional, List, Tuple

def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """유클리드 거리 계산"""
    return math.hypot(b[0] - a[0], b[1] - a[1])

def greedy_select(points_xy: List[Tuple[float, float]],
                  gaps_s: List[float],
                  weights: List[float],
                  H: int,
                  start_xy: Tuple[float, float],
                  remaining: Optional[Set[int]],
                  score_fn: Callable[[int, Tuple[float, float]], float]) -> List[int]:
    """
    Generic greedy builder for a sequence of length H.
    
    Args:
        points_xy: 포인트 좌표 리스트
        gaps_s: 공백 시간 리스트
        weights: 중요도 리스트
        H: 선택할 포인트 수
        start_xy: 시작 위치
        remaining: 선택 가능한 포인트 인덱스 집합
        score_fn: 스코어 함수 (i, cur_xy) -> float (높을수록 좋음)
    
    Returns:
        선택된 포인트 인덱스 시퀀스
    """
    if remaining is None:
        remaining = set(range(len(points_xy)))
    else:
        remaining = set(remaining)  # 복사
    
    seq = []
    cur_xy = start_xy
    
    for _ in range(H):
        if not remaining:
            break
        
        best_i, best_s = None, -float("inf")
        for i in remaining:
            s = score_fn(i, cur_xy)
            if s > best_s:
                best_i, best_s = i, s
        
        if best_i is not None:
            seq.append(best_i)
            remaining.remove(best_i)
            cur_xy = points_xy[best_i]
    
    return seq
```

## 7.4 후보 생성기 클래스

### 7.4.1 Candidate 데이터 구조

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Candidate:
    """재스케줄 후보"""
    rule_name: str                          # 후보 타입 이름
    seq: List[int]                          # 방문 시퀀스 [q1, q2, ..., qH]
    
    # Stage-1: 근사 메트릭 (유클리드 기반)
    approx_len_m: float = float("inf")
    approx_heading_rad: float = float("inf")
    approx_eta_s: float = float("inf")
    
    # Stage-2: 정밀 메트릭 (Nav2 planner 기반)
    length_m: float = float("inf")
    heading_change_rad: float = float("inf")
    eta_s: float = float("inf")
    feasible: bool = False
    error_code: int = 0
    error_msg: str = ""
    exact_evaluated: bool = False
```

### 7.4.2 근사 메트릭 계산

```python
def approx_path_metrics(start_xy: Tuple[float, float],
                        points_xy: List[Tuple[float, float]],
                        seq: List[int],
                        v_eff: float = 0.35,
                        k_theta: float = 0.20) -> Tuple[float, float, float]:
    """
    근사 경로 메트릭 계산 (유클리드 기반)
    
    Returns:
        (approx_len, approx_heading_change, approx_eta)
    """
    if not seq:
        return float("inf"), float("inf"), float("inf")
    
    pts = [start_xy] + [points_xy[i] for i in seq]
    L = 0.0
    Theta = 0.0
    
    def angle(p: Tuple[float, float], q: Tuple[float, float]) -> float:
        return math.atan2(q[1] - p[1], q[0] - p[0])
    
    prev_ang = None
    for i in range(len(pts) - 1):
        L += euclid(pts[i], pts[i + 1])
        ang = angle(pts[i], pts[i + 1])
        if prev_ang is not None:
            d = ang - prev_ang
            # Wrap to [-π, π]
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            Theta += abs(d)
        prev_ang = ang
    
    eta = (L / max(v_eff, 1e-6)) + k_theta * Theta
    return L, Theta, eta
```

### 7.4.3 후보 생성기 클래스

```python
class CandidateGenerator:
    """후보 생성기"""
    
    def __init__(self, H: int = 3, lambda_values: List[float] = [0.5, 0.7, 1.0]):
        self.H = H
        self.lambda_values = lambda_values
    
    def generate_all(self, 
                     points_xy: List[Tuple[float, float]],
                     gaps_s: List[float],
                     weights: List[float],
                     queue_current: List[int],
                     start_xy: Tuple[float, float],
                     event_xy: Optional[Tuple[float, float]] = None) -> List[Candidate]:
        """
        모든 후보 생성
        
        Returns:
            6개 기본 후보 리스트
        """
        candidates = []
        
        # 후보 0: Keep-Order
        seq0 = build_keep_order(queue_current, self.H)
        candidates.append(self._make_candidate("keep_order", seq0, start_xy, points_xy))
        
        # 후보 1: Nearest-First
        seq1 = build_nearest_first(points_xy, self.H, start_xy)
        candidates.append(self._make_candidate("nearest_first", seq1, start_xy, points_xy))
        
        # 후보 2: Overdue-First
        seq2 = build_overdue_first(points_xy, gaps_s, self.H, start_xy)
        candidates.append(self._make_candidate("overdue_first", seq2, start_xy, points_xy))
        
        # 후보 3: Overdue + ETA Balance
        seq3 = build_overdue_eta_balance(points_xy, gaps_s, self.H, start_xy, lam=0.7)
        candidates.append(self._make_candidate("overdue_balance", seq3, start_xy, points_xy))
        
        # 후보 4: Risk-Weighted
        seq4 = build_risk_weighted(points_xy, gaps_s, weights, self.H, start_xy, lam=0.7)
        candidates.append(self._make_candidate("risk_weighted", seq4, start_xy, points_xy))
        
        # 후보 5: Balanced-Coverage
        seq5 = build_balanced_coverage(points_xy, gaps_s, self.H, start_xy, lam=0.5)
        candidates.append(self._make_candidate("balanced_coverage", seq5, start_xy, points_xy))
        
        return candidates
    
    def _make_candidate(self, rule_name: str, seq: List[int],
                        start_xy: Tuple[float, float],
                        points_xy: List[Tuple[float, float]]) -> Candidate:
        """후보 객체 생성 및 근사 메트릭 계산"""
        approx_len, approx_heading, approx_eta = approx_path_metrics(
            start_xy, points_xy, seq
        )
        return Candidate(
            rule_name=rule_name,
            seq=seq,
            approx_len_m=approx_len,
            approx_heading_rad=approx_heading,
            approx_eta_s=approx_eta
        )
```

## 7.5 확장 후보 (Zone 기반)

### 7.5.1 Zone 정의

```python
def build_zones_grid(points_xy: List[Tuple[float, float]], 
                     grid_w: int = 3, 
                     grid_h: int = 3) -> Tuple[List[int], int]:
    """
    맵을 그리드로 분할하여 각 포인트에 zone ID 할당
    
    Returns:
        zones: 각 포인트의 zone ID 리스트
        K: 총 zone 수
    """
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    
    def zid(x: float, y: float) -> int:
        gx = min(grid_w - 1, max(0, int((x - minx) / max(1e-6, (maxx - minx)) * grid_w)))
        gy = min(grid_h - 1, max(0, int((y - miny) / max(1e-6, (maxy - miny)) * grid_h)))
        return gy * grid_w + gx
    
    zones = [zid(x, y) for x, y in points_xy]
    K = grid_w * grid_h
    return zones, K
```

### 7.5.2 Zone 기반 후보: zone_max_overdue

```python
def build_zone_max_overdue(points_xy: List[Tuple[float, float]],
                           gaps_s: List[float],
                           zones: List[int],
                           K: int,
                           H: int,
                           start_xy: Tuple[float, float]) -> List[int]:
    """
    가장 공백이 큰 zone을 먼저 방문
    """
    # Zone별 최대 공백 계산
    zone_max = [0.0] * K
    zone_pts = [[] for _ in range(K)]
    for i, z in enumerate(zones):
        zone_pts[z].append(i)
        zone_max[z] = max(zone_max[z], gaps_s[i])
    
    # 타겟 zone 선택
    target_zone = max(range(K), key=lambda z: zone_max[z])
    
    # 타겟 zone 내부에서 overdue 우선
    remaining = set(zone_pts[target_zone])
    seq = greedy_select(
        points_xy, gaps_s, [1.0] * len(points_xy),
        min(H, len(remaining)), start_xy, remaining,
        score_fn=lambda i, cur_xy: gaps_s[i]
    )
    
    # 부족하면 전체에서 채움
    if len(seq) < H:
        left = H - len(seq)
        already = set(seq)
        rem_all = set(range(len(points_xy))) - already
        last_xy = points_xy[seq[-1]] if seq else start_xy
        seq += greedy_select(
            points_xy, gaps_s, [1.0] * len(points_xy),
            left, last_xy, rem_all,
            score_fn=lambda i, cur_xy: gaps_s[i] - 0.3 * euclid(cur_xy, points_xy[i])
        )
    
    return seq
```

## 7.6 2단계 평가 시스템

### 7.6.1 Stage-1: 근사 평가 (빠름)

- 유클리드 거리 기반
- 모든 후보에 적용
- 정렬 및 필터링용

### 7.6.2 Stage-2: 정밀 평가 (Nav2 planner)

- ComputePathThroughPoses 액션 호출
- 상위 K개 후보에만 적용
- 실제 경로 길이/ETA/feasibility 확인

```python
def evaluate_candidates_two_stage(candidates: List[Candidate],
                                  planner_client,
                                  start_pose,
                                  points_xy: List[Tuple[float, float]],
                                  K_exact: int = 3) -> List[Candidate]:
    """
    2단계 후보 평가
    
    Args:
        candidates: 후보 리스트
        planner_client: Nav2 planner 클라이언트
        start_pose: 시작 pose
        points_xy: 포인트 좌표
        K_exact: 정밀 평가할 후보 수
    
    Returns:
        평가 완료된 후보 리스트 (approx_eta 기준 정렬)
    """
    # Stage-1 정렬
    sorted_cands = sorted(candidates, key=lambda c: c.approx_eta_s)
    
    # Stage-2: 상위 K개만 정밀 평가
    for i, cand in enumerate(sorted_cands[:K_exact]):
        if not cand.seq:
            continue
        
        # Nav2 planner로 경로 계산
        goals = [points_xy[idx] for idx in cand.seq]
        result = planner_client.compute_path_through_poses(start_pose, goals)
        
        if result.success:
            cand.length_m = result.path_length
            cand.heading_change_rad = result.heading_change
            cand.eta_s = result.eta
            cand.feasible = True
        else:
            cand.feasible = False
            cand.error_code = result.error_code
            cand.error_msg = result.error_msg
        
        cand.exact_evaluated = True
    
    return sorted_cands
```



---

# 8. Nav2 통합 인터페이스

![Nav2 통합 인터페이스](./diagrams/rendered/09_nav2_interface.png)


## 8.1 Nav2 아키텍처 개요

### 8.1.1 역할 분담

| 구성요소 | 역할 | 담당 |
|----------|------|------|
| **RL 정책** | 출동/재스케줄 결정 | 본 연구 |
| **Nav2** | 경로 계획 + 이동 제어 + 충돌 회피 | 기존 스택 |
| **Isaac Sim** | 물리 시뮬레이션 + 센서 시뮬레이션 | 시뮬레이터 |

### 8.1.2 필수 토픽/서비스

| 토픽/서비스 | 타입 | 방향 | 용도 |
|-------------|------|------|------|
| `/clock` | rosgraph_msgs/Clock | Sim → Nav2 | 시뮬레이션 시간 |
| `/tf`, `/tf_static` | tf2_msgs/TFMessage | Sim → Nav2 | 좌표 변환 |
| `/odom` | nav_msgs/Odometry | Sim → Nav2 | 오도메트리 |
| `/scan` | sensor_msgs/LaserScan | Sim → Nav2 | LiDAR 데이터 |
| `/map` | nav_msgs/OccupancyGrid | Env → Nav2 | 맵 정보 |
| `/navigate_to_pose` | nav2_msgs/action/NavigateToPose | Env → Nav2 | 목표 전송 |
| `/compute_path_through_poses` | nav2_msgs/action/ComputePathThroughPoses | Env → Nav2 | 경로 계산 |

## 8.2 Nav2 Planner 클라이언트

### 8.2.1 경로 길이 계산 공식

Nav2 planner 결과의 `nav_msgs/Path`에서 경로 길이 계산:

$$L(\text{path}) = \sum_{i=1}^{n-1} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}$$

### 8.2.2 누적 회전량 계산

$$\Theta(\text{path}) = \sum_{i=1}^{n-1} |\text{wrap}(\psi_{i+1} - \psi_i)|$$

여기서 $\text{wrap}(\cdot)$은 $[-\pi, \pi]$로 정규화

### 8.2.3 ETA 추정 공식

$$\ell_T^{(j)} = \frac{\ell_L^{(j)}}{v_{eff}} + \kappa_\theta \cdot \Theta^{(j)}$$

여기서:
- $v_{eff}$: 유효 평균 속도 (캘리브레이션 필요)
- $\kappa_\theta$: 회전 시간 페널티 계수

### 8.2.4 Planner 클라이언트 구현

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import ComputePathThroughPoses
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class PlannedMetrics:
    """Planner 결과 메트릭"""
    success: bool
    error_code: int
    error_msg: str
    path: Optional[Path]
    path_length: float
    heading_change: float
    eta: float

def path_length_xy(path: Path) -> float:
    """경로 길이 계산 (2D)"""
    poses = path.poses
    if len(poses) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(poses) - 1):
        p1 = poses[i].pose.position
        p2 = poses[i + 1].pose.position
        total += math.hypot(p2.x - p1.x, p2.y - p1.y)
    return total

def path_heading_change(path: Path) -> float:
    """누적 회전량 계산"""
    poses = path.poses
    if len(poses) < 2:
        return 0.0
    
    def yaw_from_quat(q):
        # Quaternion to yaw (z-axis rotation)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    total = 0.0
    prev_yaw = yaw_from_quat(poses[0].pose.orientation)
    
    for i in range(1, len(poses)):
        yaw = yaw_from_quat(poses[i].pose.orientation)
        diff = yaw - prev_yaw
        # Wrap to [-π, π]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        total += abs(diff)
        prev_yaw = yaw
    
    return total

class Nav2PlannerClient(Node):
    """Nav2 Planner 클라이언트"""
    
    def __init__(self, 
                 v_eff: float = 0.35,
                 k_theta: float = 0.20,
                 timeout: float = 5.0,
                 cache_ttl: float = 2.0):
        super().__init__('nav2_planner_client')
        
        self._v_eff = v_eff
        self._k_theta = k_theta
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._cache = {}  # {cache_key: (timestamp, result)}
        
        # Action client
        self._action_client = ActionClient(
            self, 
            ComputePathThroughPoses, 
            '/compute_path_through_poses'
        )
    
    def compute_path_through_poses(self,
                                   start_pose: PoseStamped,
                                   goal_xys: List[Tuple[float, float]],
                                   planner_id: str = "GridBased") -> PlannedMetrics:
        """
        다중 경유점 경로 계산
        
        Args:
            start_pose: 시작 pose
            goal_xys: 경유점 좌표 리스트 [(x1, y1), (x2, y2), ...]
            planner_id: 플래너 ID
        
        Returns:
            PlannedMetrics: 계산 결과
        """
        # 캐시 키 생성
        cache_key = self._make_cache_key(start_pose, goal_xys)
        now = self.get_clock().now().nanoseconds / 1e9
        
        # 캐시 확인
        if cache_key in self._cache:
            ts, res = self._cache[cache_key]
            if (now - ts) < self._cache_ttl:
                return res
        
        # Action 서버 대기
        if not self._action_client.wait_for_server(timeout_sec=self._timeout):
            res = PlannedMetrics(False, 9999, "Action server not available", 
                                None, float("inf"), float("inf"), float("inf"))
            self._cache[cache_key] = (now, res)
            return res
        
        # Goal 메시지 생성
        goal_msg = ComputePathThroughPoses.Goal()
        goal_msg.start = start_pose
        goal_msg.goals = [
            make_pose_stamped(x, y, 0.0, start_pose.header.frame_id) 
            for x, y in goal_xys
        ]
        goal_msg.planner_id = planner_id
        goal_msg.use_start = True
        
        # Action 호출
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=self._timeout)
        
        if not send_goal_future.done():
            res = PlannedMetrics(False, 9998, "send_goal timeout",
                                None, float("inf"), float("inf"), float("inf"))
            self._cache[cache_key] = (now, res)
            return res
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            res = PlannedMetrics(False, 9996, "Goal rejected",
                                None, float("inf"), float("inf"), float("inf"))
            self._cache[cache_key] = (now, res)
            return res
        
        # 결과 대기
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self._timeout)
        
        if not result_future.done() or result_future.result() is None:
            res = PlannedMetrics(False, 9997, "get_result timeout",
                                None, float("inf"), float("inf"), float("inf"))
            self._cache[cache_key] = (now, res)
            return res
        
        result = result_future.result().result
        err = int(result.error_code)
        msg = str(getattr(result, 'error_msg', ''))
        
        if err != 0:  # NONE=0 in nav2_msgs
            res = PlannedMetrics(False, err, msg,
                                None, float("inf"), float("inf"), float("inf"))
            self._cache[cache_key] = (now, res)
            return res
        
        # 메트릭 계산
        path = result.path
        L = path_length_xy(path)
        Theta = path_heading_change(path)
        eta = (L / max(self._v_eff, 1e-6)) + self._k_theta * Theta
        
        res = PlannedMetrics(True, err, msg, path, L, Theta, eta)
        self._cache[cache_key] = (now, res)
        return res
    
    def _make_cache_key(self, start_pose: PoseStamped, 
                        goal_xys: List[Tuple[float, float]]) -> str:
        """캐시 키 생성"""
        sp = start_pose.pose.position
        start_key = f"{sp.x:.2f},{sp.y:.2f}"
        goals_key = ";".join(f"{x:.2f},{y:.2f}" for x, y in goal_xys)
        return f"{start_key}|{goals_key}"

def make_pose_stamped(x: float, y: float, yaw: float, 
                      frame_id: str = "map") -> PoseStamped:
    """PoseStamped 생성 유틸리티"""
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = 0.0
    
    # Yaw to quaternion
    half = yaw * 0.5
    ps.pose.orientation.x = 0.0
    ps.pose.orientation.y = 0.0
    ps.pose.orientation.z = math.sin(half)
    ps.pose.orientation.w = math.cos(half)
    
    return ps
```

## 8.3 Nav2 Navigation 클라이언트

### 8.3.1 NavigateToPose 액션 클라이언트

```python
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus

class Nav2NavigationClient(Node):
    """Nav2 Navigation 클라이언트"""
    
    def __init__(self, timeout: float = 120.0):
        super().__init__('nav2_navigation_client')
        
        self._timeout = timeout
        self._action_client = ActionClient(
            self, 
            NavigateToPose, 
            '/navigate_to_pose'
        )
        self._goal_handle = None
        self._status = "IDLE"
    
    def send_goal(self, goal_pose: PoseStamped, timeout_s: float = 120.0) -> bool:
        """
        목표 전송
        
        Returns:
            True if goal accepted, False otherwise
        """
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self._status = "SERVER_UNAVAILABLE"
            return False
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        send_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        
        if not send_future.done():
            self._status = "SEND_TIMEOUT"
            return False
        
        self._goal_handle = send_future.result()
        if not self._goal_handle.accepted:
            self._status = "REJECTED"
            return False
        
        self._status = "RUNNING"
        self._timeout = timeout_s
        return True
    
    def tick(self) -> str:
        """
        현재 상태 확인 및 업데이트
        
        Returns:
            "RUNNING" | "SUCCEEDED" | "ABORTED" | "TIMEOUT" | "CANCELED"
        """
        if self._goal_handle is None:
            return "IDLE"
        
        # 결과 확인 (non-blocking)
        if self._goal_handle.status == GoalStatus.STATUS_SUCCEEDED:
            self._status = "SUCCEEDED"
        elif self._goal_handle.status == GoalStatus.STATUS_ABORTED:
            self._status = "ABORTED"
        elif self._goal_handle.status == GoalStatus.STATUS_CANCELED:
            self._status = "CANCELED"
        
        return self._status
    
    def cancel(self) -> bool:
        """현재 목표 취소"""
        if self._goal_handle is None:
            return False
        
        cancel_future = self._goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
        
        self._status = "CANCELED"
        return True
    
    def is_running(self) -> bool:
        """실행 중 여부"""
        return self._status == "RUNNING"
    
    def go_to(self, goal_pose: PoseStamped) -> str:
        """
        목표로 이동 (blocking)
        
        Returns:
            최종 상태 문자열
        """
        if not self.send_goal(goal_pose):
            return self._status
        
        # 결과 대기
        result_future = self._goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self._timeout)
        
        if not result_future.done():
            self._status = "TIMEOUT"
            self.cancel()
            return self._status
        
        return self.tick()
    
    def _feedback_callback(self, feedback_msg):
        """피드백 콜백 (로깅용)"""
        pass  # 필요 시 구현
```

## 8.4 ETA 캘리브레이션

### 8.4.1 캘리브레이션 절차

ETA를 단순히 $L / v_{max}$로 하면 너무 낙관적이다. 실제 주행에는 회피, 감속, 회전이 포함되므로 캘리브레이션이 필요하다.

**절차:**

1. 랜덤 start-goal 쌍을 K개 샘플링 (예: 200개)
2. Nav2 NavigateToPose로 실제 주행
3. 실제 주행시간 $t_{real}$과 planner path 길이 $L$ 기록
4. $v_{eff} = \text{median}(L / t_{real})$로 설정

### 8.4.2 캘리브레이션 코드

```python
import numpy as np
from typing import List, Tuple

def calibrate_v_eff(nav_client: Nav2NavigationClient,
                    planner_client: Nav2PlannerClient,
                    sample_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                    frame_id: str = "map") -> float:
    """
    유효 평균 속도 캘리브레이션
    
    Args:
        nav_client: Navigation 클라이언트
        planner_client: Planner 클라이언트
        sample_pairs: [(start_xy, goal_xy), ...] 샘플 쌍
        frame_id: 좌표 프레임
    
    Returns:
        v_eff: 캘리브레이션된 유효 속도
    """
    ratios = []
    
    for start_xy, goal_xy in sample_pairs:
        # Planner로 경로 길이 계산
        start_pose = make_pose_stamped(start_xy[0], start_xy[1], 0.0, frame_id)
        result = planner_client.compute_path_through_poses(start_pose, [goal_xy])
        
        if not result.success:
            continue
        
        L = result.path_length
        
        # 실제 주행
        goal_pose = make_pose_stamped(goal_xy[0], goal_xy[1], 0.0, frame_id)
        t_start = time.time()
        status = nav_client.go_to(goal_pose)
        t_end = time.time()
        
        if status != "SUCCEEDED":
            continue
        
        t_real = t_end - t_start
        if t_real > 0.1:  # 최소 시간 필터
            ratios.append(L / t_real)
    
    if not ratios:
        return 0.35  # 기본값
    
    return float(np.median(ratios))
```

## 8.5 Nav2 파라미터 설정

### 8.5.1 nav2_params.yaml 템플릿

```yaml
# configs/nav2_params.yaml
# ROS2 Nav2 common template for Isaac Sim + GO2

amcl:
  ros__parameters:
    use_sim_time: true
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    odom_frame_id: "odom"
    global_frame_id: "map"
    tf_broadcast: true
    scan_topic: "/scan"
    min_particles: 500
    max_particles: 2000

bt_navigator:
  ros__parameters:
    use_sim_time: true
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 20

controller_server:
  ros__parameters:
    use_sim_time: true
    controller_frequency: 10.0
    min_x_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugins: ["progress_checker"]
    goal_checker_plugins: ["goal_checker"]
    controller_plugins: ["FollowPath"]

    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.35
      yaw_goal_tolerance: 0.35
      stateful: true

    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      min_vel_x: 0.0
      max_vel_x: 0.6
      max_vel_theta: 1.2
      acc_lim_x: 1.0
      acc_lim_theta: 2.0
      vx_samples: 20
      vtheta_samples: 20
      sim_time: 2.0
      xy_goal_tolerance: 0.35

planner_server:
  ros__parameters:
    use_sim_time: true
    expected_planner_frequency: 5.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: false

global_costmap:
  global_costmap:
    ros__parameters:
      use_sim_time: true
      global_frame: "map"
      robot_base_frame: "base_link"
      update_frequency: 2.0
      publish_frequency: 1.0
      resolution: 0.05
      robot_radius: 0.35
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]

      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_topic: "/map"

      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        observation_sources: "scan"
        scan:
          topic: "/scan"
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        inflation_radius: 0.6
        cost_scaling_factor: 3.0

local_costmap:
  local_costmap:
    ros__parameters:
      use_sim_time: true
      global_frame: "odom"
      robot_base_frame: "base_link"
      update_frequency: 5.0
      publish_frequency: 2.0
      rolling_window: true
      width: 10
      height: 10
      resolution: 0.05
      robot_radius: 0.35
      plugins: ["obstacle_layer", "inflation_layer"]

      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        observation_sources: "scan"
        scan:
          topic: "/scan"
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        inflation_radius: 0.6
        cost_scaling_factor: 3.0
```

### 8.5.2 필수 TF 프레임 구조

```
map
 └── odom
      └── base_link
           └── base_footprint (선택)
           └── lidar_link
           └── camera_link (선택)
```

### 8.5.3 학습 안정성을 위한 체크리스트

| 항목 | 확인 방법 | 중요도 |
|------|-----------|--------|
| TF 프레임 연결 | `ros2 run tf2_tools view_frames` | 필수 |
| /scan 토픽 발행 | `ros2 topic echo /scan` | 필수 |
| /odom 토픽 발행 | `ros2 topic echo /odom` | 필수 |
| /map 토픽 수신 | `ros2 topic echo /map` | 필수 |
| Costmap 업데이트 | `ros2 topic echo /global_costmap/costmap` | 권장 |



---

# 9. 시뮬레이션 환경 구축

## 9.1 랜덤 맵 생성

### 9.1.1 맵 생성 파라미터

```python
from dataclasses import dataclass

@dataclass
class MapSpec:
    """맵 생성 파라미터"""
    seed: int                           # 랜덤 시드
    width_m: float = 20.0               # 맵 너비 (m)
    height_m: float = 20.0              # 맵 높이 (m)
    resolution: float = 0.05            # 해상도 (m/cell)
    obstacle_density: float = 0.08      # 장애물 밀도
    min_corridor_cells: int = 8         # 최소 통로폭 (cells)
    min_free_ratio: float = 0.35        # 최소 자유공간 비율
    n_corridor_carves: int = 12         # 통로 생성 횟수
```

### 9.1.2 맵 생성 함수

```python
import numpy as np
from typing import Tuple

FREE = 0
OCC = 100
UNK = -1

def make_random_map(spec: MapSpec) -> Tuple[np.ndarray, int, int, float]:
    """
    랜덤 맵 생성
    
    Returns:
        grid: 점유 그리드 (H x W)
        W: 너비 (cells)
        H: 높이 (cells)
        resolution: 해상도
    """
    rng = np.random.default_rng(spec.seed)
    W = int(spec.width_m / spec.resolution)
    H = int(spec.height_m / spec.resolution)
    
    # 초기화: 모두 FREE
    grid = np.zeros((H, W), dtype=np.int16)
    grid[:, :] = FREE
    
    # 경계: OCC
    grid[0, :] = OCC
    grid[-1, :] = OCC
    grid[:, 0] = OCC
    grid[:, -1] = OCC
    
    # 장애물: 랜덤 사각형
    n_rect = max(8, int(spec.obstacle_density * (W * H) / (28 * 28)))
    for _ in range(n_rect):
        rw = int(rng.integers(8, 26))
        rh = int(rng.integers(8, 26))
        x = int(rng.integers(1, max(2, W - rw - 1)))
        y = int(rng.integers(1, max(2, H - rh - 1)))
        grid[y:y+rh, x:x+rw] = OCC
    
    # 통로 carve: 연결성 확보
    for _ in range(spec.n_corridor_carves):
        if rng.random() < 0.5:
            y = int(rng.integers(2, H - 2))
            grid[y:y+2, 2:W-2] = FREE
        else:
            x = int(rng.integers(2, W - 2))
            grid[2:H-2, x:x+2] = FREE
    
    return grid.astype(np.int8), W, H, spec.resolution
```

### 9.1.3 맵 유효성 검사

```python
from collections import deque
from typing import Optional

def _neighbors4(y: int, x: int, H: int, W: int):
    """4방향 이웃"""
    if y > 0: yield (y-1, x)
    if y < H-1: yield (y+1, x)
    if x > 0: yield (y, x-1)
    if x < W-1: yield (y, x+1)

def _bfs_component(grid: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    """BFS로 연결 요소 탐색"""
    H, W = grid.shape
    vis = np.zeros((H, W), dtype=np.bool_)
    sy, sx = start
    
    if grid[sy, sx] != FREE:
        return vis
    
    q = deque([(sy, sx)])
    vis[sy, sx] = True
    
    while q:
        y, x = q.popleft()
        for ny, nx in _neighbors4(y, x, H, W):
            if not vis[ny, nx] and grid[ny, nx] == FREE:
                vis[ny, nx] = True
                q.append((ny, nx))
    
    return vis

def validate_map(grid: np.ndarray, 
                 min_free_ratio: float = 0.35,
                 min_component_ratio: float = 0.30) -> Tuple[bool, dict]:
    """
    맵 유효성 검사
    
    Args:
        grid: 점유 그리드
        min_free_ratio: 최소 자유공간 비율
        min_component_ratio: 최대 연결 요소의 최소 비율
    
    Returns:
        valid: 유효 여부
        info: 검사 정보 딕셔너리
    """
    H, W = grid.shape
    total_cells = H * W
    free_cells = np.sum(grid == FREE)
    free_ratio = free_cells / total_cells
    
    # 자유공간 비율 검사
    if free_ratio < min_free_ratio:
        return False, {"reason": "free_ratio too low", "free_ratio": free_ratio}
    
    # 최대 연결 요소 찾기
    visited_global = np.zeros((H, W), dtype=np.bool_)
    largest_component = None
    largest_size = 0
    start_cell = None
    
    for y in range(H):
        for x in range(W):
            if grid[y, x] == FREE and not visited_global[y, x]:
                comp = _bfs_component(grid, (y, x))
                comp_size = np.sum(comp)
                visited_global |= comp
                
                if comp_size > largest_size:
                    largest_size = comp_size
                    largest_component = comp
                    # 시작점: 컴포넌트 내부 중앙 근처
                    ys, xs = np.where(comp)
                    mid_idx = len(ys) // 2
                    start_cell = (int(ys[mid_idx]), int(xs[mid_idx]))
    
    if largest_component is None:
        return False, {"reason": "no free space"}
    
    component_ratio = largest_size / free_cells
    if component_ratio < min_component_ratio:
        return False, {"reason": "largest component too small", 
                      "component_ratio": component_ratio}
    
    return True, {
        "free_ratio": free_ratio,
        "component_ratio": component_ratio,
        "component_mask": largest_component,
        "start_cell": start_cell
    }
```

### 9.1.4 OccupancyGrid 메시지 변환

```python
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose

def to_occupancy_msg(grid: np.ndarray, W: int, H: int, 
                     resolution: float, frame_id: str = "map") -> OccupancyGrid:
    """
    numpy 배열을 ROS OccupancyGrid 메시지로 변환
    """
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.info = MapMetaData()
    msg.info.resolution = resolution
    msg.info.width = W
    msg.info.height = H
    
    origin = Pose()
    origin.position.x = 0.0
    origin.position.y = 0.0
    msg.info.origin = origin
    
    # ROS expects row-major flattened list
    msg.data = grid.flatten(order="C").tolist()
    return msg
```

## 9.2 순찰 포인트 생성

### 9.2.1 Poisson-Disk 샘플링

```python
def poisson_disk_points_on_grid(occ: np.ndarray,
                                resolution: float,
                                seed: int,
                                r_m: float = 1.8,
                                k: int = 30,
                                n_points_target: int = 40,
                                margin_cells: int = 6) -> List[Tuple[float, float]]:
    """
    Poisson-Disk 샘플링으로 순찰 포인트 생성
    
    Args:
        occ: 점유 그리드
        resolution: 해상도 (m/cell)
        seed: 랜덤 시드
        r_m: 최소 거리 (m)
        k: 시도 횟수
        n_points_target: 목표 포인트 수
        margin_cells: 경계 마진 (cells)
    
    Returns:
        포인트 좌표 리스트 [(x, y), ...]
    """
    rng = np.random.default_rng(seed)
    H, W = occ.shape
    r_cells = r_m / resolution
    
    # FREE 셀 목록
    free_ys, free_xs = np.where(occ == FREE)
    if len(free_ys) == 0:
        return []
    
    # 마진 적용
    valid_mask = (
        (free_ys >= margin_cells) & (free_ys < H - margin_cells) &
        (free_xs >= margin_cells) & (free_xs < W - margin_cells)
    )
    free_ys = free_ys[valid_mask]
    free_xs = free_xs[valid_mask]
    
    if len(free_ys) == 0:
        return []
    
    points = []
    active = []
    
    # 첫 포인트
    idx = rng.integers(0, len(free_ys))
    first_y, first_x = int(free_ys[idx]), int(free_xs[idx])
    points.append((first_x * resolution, first_y * resolution))
    active.append(0)
    
    while active and len(points) < n_points_target:
        # 활성 포인트 선택
        active_idx = rng.integers(0, len(active))
        px, py = points[active[active_idx]]
        
        found = False
        for _ in range(k):
            # 랜덤 방향, r ~ 2r 거리
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(r_m, 2 * r_m)
            nx = px + dist * np.cos(angle)
            ny = py + dist * np.sin(angle)
            
            # 그리드 좌표
            cx = int(nx / resolution)
            cy = int(ny / resolution)
            
            # 범위 및 FREE 검사
            if not (margin_cells <= cy < H - margin_cells and 
                    margin_cells <= cx < W - margin_cells):
                continue
            if occ[cy, cx] != FREE:
                continue
            
            # 기존 포인트와 거리 검사
            too_close = False
            for (ex, ey) in points:
                if math.hypot(nx - ex, ny - ey) < r_m:
                    too_close = True
                    break
            
            if not too_close:
                points.append((nx, ny))
                active.append(len(points) - 1)
                found = True
                break
        
        if not found:
            active.pop(active_idx)
    
    return points
```

## 9.3 이벤트 생성

### 9.3.1 이벤트 데이터 구조

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Event:
    """이벤트 데이터"""
    id: int
    x: float
    y: float
    t_arr: float                    # 도착 시간
    urg: float                      # 긴급도 [0, 1]
    conf: float                     # 신뢰도 [0, 1]
    status: str = "PENDING"         # PENDING/VERIFIED/CLEARED/FALSE
    t_handled: Optional[float] = None
```

### 9.3.2 이벤트 생성기

```python
import random

class EventGenerator:
    """확률적 이벤트 생성기"""
    
    def __init__(self, 
                 seed: int = 0,
                 lambda_per_min: float = 0.6,
                 burst_every_min: float = 10.0,
                 burst_factor: float = 4.0):
        """
        Args:
            seed: 랜덤 시드
            lambda_per_min: 분당 평균 이벤트 수
            burst_every_min: 폭증 주기 (분)
            burst_factor: 폭증 시 배율
        """
        self.rng = random.Random(seed)
        self.lambda_per_sec = lambda_per_min / 60.0
        self.burst_every = burst_every_min * 60.0
        self.burst_factor = burst_factor
        self.last_burst_t = None
        self.next_id = 1
    
    def sample_next_events(self, 
                           now_t: float, 
                           dt: float,
                           free_sampler) -> List[Event]:
        """
        다음 시간 구간의 이벤트 샘플링
        
        Args:
            now_t: 현재 시간
            dt: 시간 구간
            free_sampler: 자유공간 위치 샘플러 함수
        
        Returns:
            생성된 이벤트 리스트
        """
        # 폭증 스케줄
        lam = self.lambda_per_sec
        if self.last_burst_t is None:
            self.last_burst_t = now_t
        
        if (now_t - self.last_burst_t) >= self.burst_every:
            lam *= self.burst_factor
            if (now_t - self.last_burst_t) >= (self.burst_every + 60.0):
                self.last_burst_t = now_t
        
        # Poisson 도착
        expected = lam * dt
        k = 0
        if self.rng.random() < expected:
            k = 1
            if self.rng.random() < (expected - 1.0):
                k = 2
        
        events = []
        for _ in range(k):
            x, y = free_sampler()
            urg = min(1.0, max(0.0, self.rng.betavariate(2.0, 2.0)))
            conf = min(1.0, max(0.0, self.rng.betavariate(3.0, 1.5)))
            
            events.append(Event(
                id=self.next_id,
                x=x, y=y,
                t_arr=now_t,
                urg=urg,
                conf=conf
            ))
            self.next_id += 1
        
        return events
```

### 9.3.3 이벤트 큐 관리

```python
from typing import List, Tuple

class EventQueue:
    """이벤트 큐 관리"""
    
    def __init__(self, Emax: int = 5):
        """
        Args:
            Emax: 상태에 노출할 최대 이벤트 수
        """
        self.Emax = Emax
        self.events: List[Event] = []
    
    def add(self, event: Event):
        """이벤트 추가"""
        self.events.append(event)
    
    def mark_done(self, eid: int, status: str, t_handled: float):
        """이벤트 처리 완료 마킹"""
        for e in self.events:
            if e.id == eid:
                e.status = status
                e.t_handled = t_handled
                break
    
    def pending_events(self) -> List[Event]:
        """미처리 이벤트 목록"""
        return [e for e in self.events if e.status == "PENDING"]
    
    def top_for_state(self, now_t: float) -> List[Tuple[float, float, float, float, float, int]]:
        """
        상태에 노출할 상위 이벤트 (긴급도 기준 정렬)
        
        Returns:
            [(elapsed, urg, conf, x, y, eid), ...] 최대 Emax개
        """
        pending = self.pending_events()
        # 긴급도 내림차순 정렬
        pending.sort(key=lambda e: -e.urg)
        
        result = []
        for e in pending[:self.Emax]:
            elapsed = now_t - e.t_arr
            result.append((elapsed, e.urg, e.conf, e.x, e.y, e.id))
        
        return result
    
    def summary_stats(self, now_t: float) -> dict:
        """이벤트 요약 통계"""
        pending = self.pending_events()
        if not pending:
            return {"count": 0, "avg_elapsed": 0.0, "max_urg": 0.0}
        
        elapsed_list = [now_t - e.t_arr for e in pending]
        return {
            "count": len(pending),
            "avg_elapsed": np.mean(elapsed_list),
            "max_urg": max(e.urg for e in pending)
        }
```

## 9.4 Safety Shield

### 9.4.1 Safety Shield 클래스

```python
class SafetyShield:
    """안전 보호 모듈"""
    
    def __init__(self,
                 keep_out_zones: List[Tuple[float, float, float]] = None,
                 abort_streak_threshold: int = 3,
                 fallback_distance: float = 2.0):
        """
        Args:
            keep_out_zones: [(x, y, radius), ...] 금지 구역
            abort_streak_threshold: 연속 실패 임계값
            fallback_distance: 폴백 이동 거리
        """
        self.keep_out_zones = keep_out_zones or []
        self.abort_streak_threshold = abort_streak_threshold
        self.fallback_distance = fallback_distance
        
        self.abort_streak = 0
        self.recent_nav_results = []
    
    def reset(self):
        """상태 초기화"""
        self.abort_streak = 0
        self.recent_nav_results = []
    
    def is_in_keep_out(self, x: float, y: float) -> bool:
        """금지 구역 내 여부"""
        for kx, ky, kr in self.keep_out_zones:
            if math.hypot(x - kx, y - ky) < kr:
                return True
        return False
    
    def filter_candidate_sequences(self, 
                                   candidates: List[Candidate],
                                   points_xy: List[Tuple[float, float]]):
        """후보 시퀀스에서 금지 구역 포인트 필터링"""
        for cand in candidates:
            for idx in cand.seq:
                px, py = points_xy[idx]
                if self.is_in_keep_out(px, py):
                    cand.feasible = False
                    cand.error_msg = "keep_out_zone"
                    break
    
    def update_nav_result(self, status: str):
        """Nav2 결과 업데이트"""
        self.recent_nav_results.append(status)
        if len(self.recent_nav_results) > 10:
            self.recent_nav_results.pop(0)
        
        if status in ("ABORTED", "TIMEOUT"):
            self.abort_streak += 1
        else:
            self.abort_streak = 0
    
    def need_fallback(self) -> bool:
        """폴백 필요 여부"""
        return self.abort_streak >= self.abort_streak_threshold
    
    def pick_safe_waypoint(self, robot_xy: Tuple[float, float]) -> Tuple[float, float]:
        """안전한 폴백 위치 선택 (단순: 뒤로 이동)"""
        # 실제 구현에서는 costmap 기반으로 안전한 위치 선택
        return (robot_xy[0] - self.fallback_distance, robot_xy[1])
```

## 9.5 에피소드 빌더

### 9.5.1 에피소드 번들 구조

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class EpisodeBuildConfig:
    """에피소드 빌드 설정"""
    map_width_m: float = 20.0
    map_height_m: float = 20.0
    resolution: float = 0.05
    obstacle_density: float = 0.08
    min_corridor_cells: int = 8
    min_free_ratio: float = 0.35
    n_points_target: int = 40
    poisson_r_m: float = 1.8
    poisson_k: int = 30
    margin_cells: int = 6
    n_zone_x: int = 4
    n_zone_y: int = 4

@dataclass
class EpisodeBundle:
    """에피소드 데이터 번들"""
    grid: np.ndarray
    W: int
    H: int
    resolution: float
    seed_map: int
    start_xy: Tuple[float, float]
    component_mask: np.ndarray
    points_xy: List[Tuple[float, float]]
    point_zones: List[int]
    zone_count: int
    sample_free_xy: Callable[[], Tuple[float, float]]
```

### 9.5.2 에피소드 빌더 함수

```python
def build_episode_bundle(base_seed: int, 
                         cfg: EpisodeBuildConfig) -> EpisodeBundle:
    """
    에피소드 데이터 번들 생성
    
    Args:
        base_seed: 기본 시드
        cfg: 빌드 설정
    
    Returns:
        EpisodeBundle: 에피소드 데이터
    """
    rng = np.random.default_rng(base_seed)
    
    # 1) 유효한 맵 생성 (최대 300회 시도)
    grid = None
    map_seed = None
    comp_mask = None
    start_cell = None
    
    for attempt in range(300):
        s = base_seed + attempt
        spec = MapSpec(
            seed=s,
            width_m=cfg.map_width_m,
            height_m=cfg.map_height_m,
            resolution=cfg.resolution,
            obstacle_density=cfg.obstacle_density,
            min_corridor_cells=cfg.min_corridor_cells,
            min_free_ratio=cfg.min_free_ratio
        )
        g, W, H, res = make_random_map(spec)
        
        ok, info = validate_map(
            g,
            min_free_ratio=cfg.min_free_ratio,
            min_component_ratio=0.30
        )
        
        if ok:
            map_seed = s
            grid = g
            comp_mask = info["component_mask"]
            start_cell = info["start_cell"]
            break
    
    if grid is None:
        raise RuntimeError("Failed to generate valid map in 300 tries")
    
    # 2) 시작 위치
    sy, sx = start_cell
    start_xy = (sx * res, sy * res)
    
    # 3) Poisson-disk 포인트 생성
    points_xy = poisson_disk_points_on_grid(
        occ=grid,
        resolution=res,
        seed=map_seed,
        r_m=cfg.poisson_r_m,
        k=cfg.poisson_k,
        n_points_target=cfg.n_points_target,
        margin_cells=cfg.margin_cells
    )
    
    # 3-1) 컴포넌트 내부 포인트만 필터링
    filtered = []
    for (x, y) in points_xy:
        cy = int(round(y / res))
        cx = int(round(x / res))
        if 0 <= cy < H and 0 <= cx < W and comp_mask[cy, cx]:
            filtered.append((x, y))
    points_xy = filtered
    
    # 3-2) 포인트 부족 시 추가 샘플링
    if len(points_xy) < max(10, int(cfg.n_points_target * 0.6)):
        ys, xs = np.where(comp_mask)
        for _ in range(5000):
            if len(points_xy) >= cfg.n_points_target:
                break
            i = int(rng.integers(0, len(ys)))
            cy, cx = int(ys[i]), int(xs[i])
            if grid[cy, cx] != FREE:
                continue
            x, y = cx * res, cy * res
            points_xy.append((x, y))
    
    # 4) Zone 매핑
    def zone_id(x: float, y: float) -> int:
        gx = min(cfg.n_zone_x - 1, max(0, int(x / cfg.map_width_m * cfg.n_zone_x)))
        gy = min(cfg.n_zone_y - 1, max(0, int(y / cfg.map_height_m * cfg.n_zone_y)))
        return gy * cfg.n_zone_x + gx
    
    zone_ids = [zone_id(x, y) for (x, y) in points_xy]
    zone_count = cfg.n_zone_x * cfg.n_zone_y
    
    # 5) 이벤트 위치 샘플러
    rng_ev = np.random.default_rng(base_seed + 99991)
    
    def sample_free_xy() -> Tuple[float, float]:
        ys, xs = np.where(comp_mask)
        i = int(rng_ev.integers(0, len(ys)))
        cy, cx = int(ys[i]), int(xs[i])
        return (cx * res, cy * res)
    
    return EpisodeBundle(
        grid=grid, W=W, H=H, resolution=res,
        seed_map=map_seed,
        start_xy=start_xy,
        component_mask=comp_mask,
        points_xy=points_xy,
        point_zones=zone_ids,
        zone_count=zone_count,
        sample_free_xy=sample_free_xy
    )
```

## 9.6 환경 클래스 (PatrolEnv)

![에피소드 생성 및 환경 초기화 흐름](./diagrams/rendered/08_episode_flow.png)


### 9.6.1 환경 초기화

```python
import rclpy
from rclpy.node import Node
import numpy as np
import time
from typing import Dict, Any

class PatrolEnv(Node):
    """순찰 로봇 강화학습 환경"""
    
    def __init__(self,
                 Kobs: int = 6,
                 Emax: int = 5,
                 H: int = 3,
                 max_steps: int = 100,
                 visit_radius: float = 0.5,
                 decision_tick_s: float = 0.1,
                 decision_max_s: float = 60.0):
        """
        Args:
            Kobs: 관측에 포함할 후보 수
            Emax: 관측에 포함할 이벤트 수
            H: Rolling horizon
            max_steps: 에피소드 최대 스텝
            visit_radius: 방문 판정 반경 (m)
            decision_tick_s: 의사결정 틱 간격 (s)
            decision_max_s: 최대 의사결정 대기 시간 (s)
        """
        super().__init__('patrol_env')
        
        self.Kobs = Kobs
        self.Emax = Emax
        self.H = H
        self.max_steps = max_steps
        self.visit_radius = visit_radius
        self.decision_tick_s = decision_tick_s
        self.decision_max_s = decision_max_s
        
        # 컴포넌트 초기화
        self.planner = Nav2PlannerClient()
        self.nav2 = Nav2NavigationClient()
        self.cand_gen = CandidateGenerator(H=H)
        self.shield = SafetyShield()
        self.rew_calc = RewardCalculator(RewardWeights())
        self.events = EventQueue(Emax=Emax)
        self.event_gen = EventGenerator()
        
        # 상태 변수
        self.points_xy = []
        self.weights = []
        self.gaps_s = []
        self.last_visit_t = []
        self.failure_heat = []
        self.queue_current = []
        self._step_idx = 0
        self._episode_seed = 0
    
    def now(self) -> float:
        """현재 시뮬레이션 시간"""
        return self.get_clock().now().nanoseconds / 1e9
```

### 9.6.2 Reset 함수

```python
    def reset(self, seed: int = 0) -> np.ndarray:
        """
        환경 리셋
        
        Args:
            seed: 에피소드 시드
        
        Returns:
            초기 관측 벡터
        """
        np.random.seed(seed)
        self._episode_seed = seed
        self._step_idx = 0
        
        # 에피소드 번들 생성
        cfg = EpisodeBuildConfig()
        bundle = build_episode_bundle(seed, cfg)
        
        # /map 퍼블리시
        if not hasattr(self, 'map_pub'):
            from nav_msgs.msg import OccupancyGrid
            self.map_pub = self.create_publisher(OccupancyGrid, "/map", 1)
        
        map_msg = to_occupancy_msg(bundle.grid, bundle.W, bundle.H, 
                                   bundle.resolution, frame_id="map")
        map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(map_msg)
        
        # 상태 초기화
        self.points_xy = bundle.points_xy
        self.weights = [1.0] * len(self.points_xy)
        now_t = self.now()
        self.last_visit_t = [now_t] * len(self.points_xy)
        self.gaps_s = [0.0] * len(self.points_xy)
        self.failure_heat = [0.0] * len(self.points_xy)
        self.queue_current = list(range(len(self.points_xy)))
        
        # 이벤트 초기화
        self.events = EventQueue(Emax=self.Emax)
        self.event_gen = EventGenerator(seed=seed)
        self.sample_free_xy = bundle.sample_free_xy
        
        # Nav 상태 초기화
        self.pending_nav = None
        self.shield.reset()
        
        return self._build_obs()
```

### 9.6.3 Step 함수

```python
    def step(self, a_mode: int, a_cand: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        환경 스텝 실행
        
        Args:
            a_mode: 출동 여부 (0: 순찰, 1: 출동)
            a_cand: 재스케줄 후보 인덱스
        
        Returns:
            obs: 관측 벡터
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        self._step_idx += 1
        t0 = self.now()
        
        # 현재 위치
        pose = self.get_robot_pose()
        rx, ry = pose.pose.position.x, pose.pose.position.y
        
        # 이벤트 생성
        new_events = self.event_gen.sample_next_events(
            t0, self.decision_tick_s, self.sample_free_xy
        )
        for e in new_events:
            self.events.add(e)
        
        # 공백 업데이트
        self.update_gaps(t0)
        
        # 후보 생성 및 평가
        candidates = self.cand_gen.generate_all(
            self.points_xy, self.gaps_s, self.weights,
            self.queue_current, (rx, ry)
        )
        candidates = evaluate_candidates_two_stage(
            candidates, self.planner, pose, self.points_xy, K_exact=3
        )
        
        # Safety shield 적용
        self.shield.filter_candidate_sequences(candidates, self.points_xy)
        
        # 행동 해석
        pending_events = self.events.pending_events()
        event_active = len(pending_events) > 0
        event_xy = (pending_events[0].x, pending_events[0].y) if event_active else None
        
        goal_type, goal_xy = self._interpret_action(
            a_mode, a_cand, event_active, event_xy, candidates
        )
        
        # Nav2 목표 전송 및 실행
        goal_pose = make_pose_stamped(goal_xy[0], goal_xy[1], 0.0, "map")
        nav_status = self.nav2.go_to(goal_pose)
        
        # 결과 처리
        t1 = self.now()
        delta_t = t1 - t0
        
        pose2 = self.get_robot_pose()
        rx2, ry2 = pose2.pose.position.x, pose2.pose.position.y
        
        self.update_gaps(t1)
        self.mark_visited_if_close((rx2, ry2), t1)
        
        # 이벤트 처리 확인
        event_handled = False
        if goal_type == "EVENT" and event_active:
            if math.hypot(rx2 - event_xy[0], ry2 - event_xy[1]) <= self.visit_radius:
                self.events.mark_done(pending_events[0].id, "VERIFIED", t1)
                event_handled = True
        
        # 실패 히트 업데이트
        if nav_status in ("ABORTED", "TIMEOUT"):
            if a_cand < len(candidates) and candidates[a_cand].seq:
                self.failure_heat[candidates[a_cand].seq[0]] += 1.0
        
        self.shield.update_nav_result(nav_status)
        
        # 보상 계산
        path_len = candidates[a_cand].length_m if a_cand < len(candidates) else 0.0
        heading_change = candidates[a_cand].heading_change_rad if a_cand < len(candidates) else 0.0
        
        reward_dict = self.rew_calc.compute(
            event_active=event_active,
            event_handled=event_handled,
            gaps_s=self.gaps_s,
            delta_t=delta_t,
            nav_status=nav_status,
            collision=False,  # 충돌 감지 별도 구현 필요
            path_length=path_len,
            heading_change=heading_change
        )
        
        reward = reward_dict["r_total"]
        
        # 종료 조건
        done = False
        if self._step_idx >= self.max_steps:
            done = True
        if event_handled:
            done = True  # 0-3개월: 이벤트 1개면 처리 후 종료
        
        # 관측 생성
        obs = self._build_obs()
        
        info = {
            "nav_status": nav_status,
            "goal_type": goal_type,
            "event_handled": event_handled,
            "reward_breakdown": reward_dict
        }
        
        return obs, reward, done, info
```

### 9.6.4 유틸리티 메서드

```python
    def update_gaps(self, now_t: float):
        """공백 시간 업데이트"""
        for i in range(len(self.points_xy)):
            self.gaps_s[i] = now_t - self.last_visit_t[i]
    
    def mark_visited_if_close(self, robot_xy: Tuple[float, float], now_t: float):
        """근접 포인트 방문 처리"""
        for i, (px, py) in enumerate(self.points_xy):
            if math.hypot(robot_xy[0] - px, robot_xy[1] - py) <= self.visit_radius:
                self.last_visit_t[i] = now_t
                self.gaps_s[i] = 0.0
    
    def coverage_metrics(self) -> Tuple[float, float, float]:
        """커버리지 요약 통계"""
        arr = np.array(self.gaps_s)
        return float(np.mean(arr)), float(np.max(arr)), float(np.std(arr))
    
    def get_robot_pose(self) -> PoseStamped:
        """현재 로봇 pose 획득 (TF2 기반)"""
        # 실제 구현에서는 tf2_ros 사용
        # 여기서는 placeholder
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.pose.position.x = 0.0
        ps.pose.position.y = 0.0
        return ps
    
    def _interpret_action(self, a_mode, a_cand, event_active, event_xy, candidates):
        """행동 해석"""
        # 재스케줄 적용
        if a_cand > 0 and a_cand <= len(candidates):
            self.queue_current = candidates[a_cand - 1].seq + self.queue_current
        
        # 출동 결정
        if a_mode == 1 and event_active:
            return "EVENT", event_xy
        else:
            if self.queue_current:
                next_idx = self.queue_current[0]
                return "PATROL", self.points_xy[next_idx]
            else:
                return "PATROL", self.points_xy[0]
    
    def _build_obs(self) -> np.ndarray:
        """관측 벡터 생성"""
        # 실제 구현에서는 전체 관측 벡터 구성
        # 여기서는 placeholder
        return np.zeros(77 + self.Kobs * 4 + 3, dtype=np.float32)
```



---

# 10. 강화학습 알고리즘 설계

## 10.1 알고리즘 선택: PPO

### 10.1.1 PPO 선택 이유

| 특성 | PPO 적합성 | 비고 |
|------|------------|------|
| 샘플 효율성 | 중간 | Off-policy보다 낮지만 안정적 |
| 구현 복잡도 | 낮음 | 단일 네트워크, 간단한 업데이트 |
| 하이퍼파라미터 민감도 | 낮음 | 튜닝 용이 |
| 연속/이산 행동 | 모두 지원 | 본 연구의 복합 행동 공간에 적합 |
| 병렬화 | 용이 | 다중 환경 동시 수집 |

### 10.1.2 PPO 목적 함수

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

여기서:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$: 확률 비율
- $\hat{A}_t$: 추정 어드밴티지
- $\epsilon$: 클리핑 파라미터 (0.2 권장)

### 10.1.3 GAE (Generalized Advantage Estimation)

$$\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

여기서 $\lambda$는 GAE 파라미터 (0.95 권장)

## 10.2 네트워크 아키텍처

![Actor-Critic 네트워크 구조](./diagrams/rendered/11_network_architecture.png)


### 10.2.1 Actor-Critic 구조

```
                    ┌─────────────────────────────────────────┐
                    │              관측 벡터 o_t              │
                    │  [77 + K_obs*4 + 3 = ~107 차원]        │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │           공유 인코더 (MLP)              │
                    │   Linear(107, 256) → ReLU               │
                    │   Linear(256, 256) → ReLU               │
                    └─────────────────────────────────────────┘
                                        │
                        ┌───────────────┼───────────────┐
                        │               │               │
                        ▼               ▼               ▼
              ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
              │  Mode Head  │  │ Replan Head │  │ Value Head  │
              │ Linear(256, │  │ Linear(256, │  │ Linear(256, │
              │     2)      │  │    N+1)     │  │     1)      │
              └─────────────┘  └─────────────┘  └─────────────┘
                    │               │               │
                    ▼               ▼               ▼
              ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
              │ Bernoulli   │  │ Categorical │  │   V(s_t)    │
              │ a_mode      │  │ a_replan    │  │             │
              └─────────────┘  └─────────────┘  └─────────────┘
```

### 10.2.2 PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

class ActorCritic(nn.Module):
    """PPO Actor-Critic 네트워크"""
    
    def __init__(self, 
                 obs_dim: int = 107,
                 hidden_dim: int = 256,
                 n_replan_candidates: int = 6):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_replan = n_replan_candidates + 1  # +1 for no-replan
        
        # 공유 인코더
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor heads
        self.mode_head = nn.Linear(hidden_dim, 2)      # 출동 여부
        self.replan_head = nn.Linear(hidden_dim, self.n_replan)  # 후보 선택
        
        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # Policy heads: 작은 초기 가중치
        nn.init.orthogonal_(self.mode_head.weight, gain=0.01)
        nn.init.orthogonal_(self.replan_head.weight, gain=0.01)
        
        # Value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor, 
                mode_mask: torch.Tensor = None,
                replan_mask: torch.Tensor = None):
        """
        순전파
        
        Args:
            obs: 관측 벡터 [B, obs_dim]
            mode_mask: 출동 마스크 [B, 2]
            replan_mask: 재스케줄 마스크 [B, n_replan]
        
        Returns:
            mode_logits, replan_logits, value
        """
        h = self.encoder(obs)
        
        mode_logits = self.mode_head(h)
        replan_logits = self.replan_head(h)
        value = self.value_head(h).squeeze(-1)
        
        # 마스킹 적용
        if mode_mask is not None:
            mode_logits = mode_logits + (mode_mask.log() + 1e-8)
        if replan_mask is not None:
            replan_logits = replan_logits + (replan_mask.log() + 1e-8)
        
        return mode_logits, replan_logits, value
    
    def get_action(self, obs: torch.Tensor,
                   mode_mask: torch.Tensor = None,
                   replan_mask: torch.Tensor = None,
                   deterministic: bool = False):
        """
        행동 샘플링
        
        Returns:
            a_mode, a_replan, log_prob_mode, log_prob_replan, value
        """
        mode_logits, replan_logits, value = self.forward(obs, mode_mask, replan_mask)
        
        mode_probs = F.softmax(mode_logits, dim=-1)
        replan_probs = F.softmax(replan_logits, dim=-1)
        
        if deterministic:
            a_mode = mode_probs.argmax(dim=-1)
            a_replan = replan_probs.argmax(dim=-1)
        else:
            mode_dist = Categorical(mode_probs)
            replan_dist = Categorical(replan_probs)
            a_mode = mode_dist.sample()
            a_replan = replan_dist.sample()
        
        log_prob_mode = Categorical(mode_probs).log_prob(a_mode)
        log_prob_replan = Categorical(replan_probs).log_prob(a_replan)
        
        return a_mode, a_replan, log_prob_mode, log_prob_replan, value
    
    def evaluate_actions(self, obs: torch.Tensor,
                         a_mode: torch.Tensor,
                         a_replan: torch.Tensor,
                         mode_mask: torch.Tensor = None,
                         replan_mask: torch.Tensor = None):
        """
        행동 평가 (PPO 업데이트용)
        
        Returns:
            log_prob_mode, log_prob_replan, entropy_mode, entropy_replan, value
        """
        mode_logits, replan_logits, value = self.forward(obs, mode_mask, replan_mask)
        
        mode_dist = Categorical(logits=mode_logits)
        replan_dist = Categorical(logits=replan_logits)
        
        log_prob_mode = mode_dist.log_prob(a_mode)
        log_prob_replan = replan_dist.log_prob(a_replan)
        
        entropy_mode = mode_dist.entropy()
        entropy_replan = replan_dist.entropy()
        
        return log_prob_mode, log_prob_replan, entropy_mode, entropy_replan, value
```

## 10.3 PPO 학습 루프

![PPO 학습 파이프라인](./diagrams/rendered/06_ppo_pipeline.png)


### 10.3.1 하이퍼파라미터

```yaml
# configs/ppo_config.yaml
ppo:
  # 환경
  n_envs: 8                    # 병렬 환경 수
  n_steps: 256                 # 환경당 수집 스텝
  batch_size: 64               # 미니배치 크기
  n_epochs: 4                  # 에폭 수
  
  # 학습
  lr: 3.0e-4                   # 학습률
  gamma: 0.99                  # 할인율
  gae_lambda: 0.95             # GAE 파라미터
  clip_range: 0.2              # PPO 클리핑 범위
  clip_range_vf: null          # Value 클리핑 (null=비활성)
  
  # 정규화
  ent_coef: 0.01               # 엔트로피 계수
  vf_coef: 0.5                 # Value 손실 계수
  max_grad_norm: 0.5           # 그래디언트 클리핑
  
  # 스케줄
  total_timesteps: 2_000_000   # 총 학습 스텝
  lr_schedule: "linear"        # 학습률 스케줄
  
  # 로깅
  log_interval: 10             # 로깅 간격
  save_interval: 50000         # 체크포인트 간격
```

### 10.3.2 롤아웃 버퍼

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class RolloutBuffer:
    """PPO 롤아웃 버퍼"""
    
    def __init__(self, 
                 n_steps: int,
                 n_envs: int,
                 obs_dim: int,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 버퍼 초기화
        self.observations = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions_mode = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.actions_replan = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs_mode = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs_replan = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.mode_masks = np.ones((n_steps, n_envs, 2), dtype=np.float32)
        self.replan_masks = np.ones((n_steps, n_envs, 7), dtype=np.float32)
        
        # GAE 계산 결과
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        
        self.ptr = 0
        self.full = False
    
    def add(self, obs, a_mode, a_replan, reward, done, value, 
            log_prob_mode, log_prob_replan, mode_mask, replan_mask):
        """데이터 추가"""
        self.observations[self.ptr] = obs
        self.actions_mode[self.ptr] = a_mode
        self.actions_replan[self.ptr] = a_replan
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs_mode[self.ptr] = log_prob_mode
        self.log_probs_replan[self.ptr] = log_prob_replan
        self.mode_masks[self.ptr] = mode_mask
        self.replan_masks[self.ptr] = replan_mask
        
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True
    
    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray):
        """GAE 계산"""
        last_gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            
            delta = (self.rewards[t] + 
                    self.gamma * next_values * next_non_terminal - 
                    self.values[t])
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        self.returns = self.advantages + self.values
    
    def get_samples(self, batch_size: int):
        """미니배치 생성 (제너레이터)"""
        indices = np.random.permutation(self.n_steps * self.n_envs)
        
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            # 2D 인덱스로 변환
            step_indices = batch_indices // self.n_envs
            env_indices = batch_indices % self.n_envs
            
            yield {
                "obs": self.observations[step_indices, env_indices],
                "a_mode": self.actions_mode[step_indices, env_indices],
                "a_replan": self.actions_replan[step_indices, env_indices],
                "old_log_prob_mode": self.log_probs_mode[step_indices, env_indices],
                "old_log_prob_replan": self.log_probs_replan[step_indices, env_indices],
                "advantages": self.advantages[step_indices, env_indices],
                "returns": self.returns[step_indices, env_indices],
                "mode_mask": self.mode_masks[step_indices, env_indices],
                "replan_mask": self.replan_masks[step_indices, env_indices]
            }
    
    def reset(self):
        """버퍼 리셋"""
        self.ptr = 0
        self.full = False
```

### 10.3.3 PPO 학습기

```python
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class PPOTrainer:
    """PPO 학습기"""
    
    def __init__(self,
                 policy: ActorCritic,
                 envs,  # VecEnv
                 config: dict,
                 device: str = "cuda"):
        
        self.policy = policy.to(device)
        self.envs = envs
        self.config = config
        self.device = device
        
        # 옵티마이저
        self.optimizer = optim.Adam(
            policy.parameters(), 
            lr=config["lr"],
            eps=1e-5
        )
        
        # 버퍼
        self.buffer = RolloutBuffer(
            n_steps=config["n_steps"],
            n_envs=config["n_envs"],
            obs_dim=policy.obs_dim,
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"]
        )
        
        # 로깅
        self.writer = SummaryWriter()
        self.global_step = 0
    
    def collect_rollouts(self):
        """롤아웃 수집"""
        self.buffer.reset()
        obs = self.envs.reset()
        
        for _ in range(self.config["n_steps"]):
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).to(self.device)
                mode_mask = torch.ones(self.config["n_envs"], 2).to(self.device)
                replan_mask = torch.ones(self.config["n_envs"], 7).to(self.device)
                
                a_mode, a_replan, log_p_mode, log_p_replan, value = \
                    self.policy.get_action(obs_t, mode_mask, replan_mask)
            
            # 환경 스텝
            actions = list(zip(a_mode.cpu().numpy(), a_replan.cpu().numpy()))
            next_obs, rewards, dones, infos = self.envs.step(actions)
            
            # 버퍼에 저장
            self.buffer.add(
                obs=obs,
                a_mode=a_mode.cpu().numpy(),
                a_replan=a_replan.cpu().numpy(),
                reward=rewards,
                done=dones,
                value=value.cpu().numpy(),
                log_prob_mode=log_p_mode.cpu().numpy(),
                log_prob_replan=log_p_replan.cpu().numpy(),
                mode_mask=mode_mask.cpu().numpy(),
                replan_mask=replan_mask.cpu().numpy()
            )
            
            obs = next_obs
            self.global_step += self.config["n_envs"]
        
        # 마지막 value 계산
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).to(self.device)
            _, _, last_value = self.policy.forward(obs_t)
            last_value = last_value.cpu().numpy()
        
        # GAE 계산
        self.buffer.compute_gae(last_value, dones)
        
        return obs
    
    def update(self):
        """PPO 업데이트"""
        clip_range = self.config["clip_range"]
        ent_coef = self.config["ent_coef"]
        vf_coef = self.config["vf_coef"]
        max_grad_norm = self.config["max_grad_norm"]
        
        # 어드밴티지 정규화
        advantages = self.buffer.advantages.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.reshape(self.buffer.n_steps, self.buffer.n_envs)
        
        # 에폭 반복
        for epoch in range(self.config["n_epochs"]):
            for batch in self.buffer.get_samples(self.config["batch_size"]):
                # 텐서 변환
                obs = torch.FloatTensor(batch["obs"]).to(self.device)
                a_mode = torch.LongTensor(batch["a_mode"]).to(self.device)
                a_replan = torch.LongTensor(batch["a_replan"]).to(self.device)
                old_log_p_mode = torch.FloatTensor(batch["old_log_prob_mode"]).to(self.device)
                old_log_p_replan = torch.FloatTensor(batch["old_log_prob_replan"]).to(self.device)
                advantages = torch.FloatTensor(batch["advantages"]).to(self.device)
                returns = torch.FloatTensor(batch["returns"]).to(self.device)
                mode_mask = torch.FloatTensor(batch["mode_mask"]).to(self.device)
                replan_mask = torch.FloatTensor(batch["replan_mask"]).to(self.device)
                
                # 현재 정책 평가
                log_p_mode, log_p_replan, ent_mode, ent_replan, values = \
                    self.policy.evaluate_actions(obs, a_mode, a_replan, mode_mask, replan_mask)
                
                # 확률 비율
                ratio_mode = torch.exp(log_p_mode - old_log_p_mode)
                ratio_replan = torch.exp(log_p_replan - old_log_p_replan)
                ratio = ratio_mode * ratio_replan  # 결합 비율
                
                # PPO 손실
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value 손실
                value_loss = F.mse_loss(values, returns)
                
                # 엔트로피 보너스
                entropy_loss = -(ent_mode.mean() + ent_replan.mean())
                
                # 총 손실
                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                
                # 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()
        
        # 로깅
        self.writer.add_scalar("loss/policy", policy_loss.item(), self.global_step)
        self.writer.add_scalar("loss/value", value_loss.item(), self.global_step)
        self.writer.add_scalar("loss/entropy", -entropy_loss.item(), self.global_step)
    
    def train(self, total_timesteps: int):
        """학습 루프"""
        obs = self.envs.reset()
        
        while self.global_step < total_timesteps:
            # 롤아웃 수집
            obs = self.collect_rollouts()
            
            # PPO 업데이트
            self.update()
            
            # 로깅
            if self.global_step % self.config["log_interval"] == 0:
                print(f"Step: {self.global_step}")
            
            # 체크포인트
            if self.global_step % self.config["save_interval"] == 0:
                torch.save(self.policy.state_dict(), 
                          f"checkpoints/policy_{self.global_step}.pt")
```

## 10.4 커리큘럼 학습

### 10.4.1 커리큘럼 단계

| 단계 | 맵 크기 | 포인트 수 | 이벤트 빈도 | 목표 |
|------|---------|-----------|-------------|------|
| 1 | 10×10m | 8-12 | 0.3/min | 기본 순찰 |
| 2 | 15×15m | 15-20 | 0.5/min | 출동 학습 |
| 3 | 20×20m | 25-35 | 0.8/min | 재스케줄 학습 |
| 4 | 25×25m | 35-50 | 1.0/min | 통합 정책 |

### 10.4.2 커리큘럼 스케줄러

```python
class CurriculumScheduler:
    """커리큘럼 학습 스케줄러"""
    
    def __init__(self, stages: List[dict], success_threshold: float = 0.7):
        """
        Args:
            stages: 단계별 설정 리스트
            success_threshold: 다음 단계 진입 기준
        """
        self.stages = stages
        self.success_threshold = success_threshold
        self.current_stage = 0
        self.stage_metrics = []
    
    def get_current_config(self) -> dict:
        """현재 단계 설정"""
        return self.stages[self.current_stage]
    
    def update(self, success_rate: float) -> bool:
        """
        성공률 기반 단계 업데이트
        
        Returns:
            True if stage advanced
        """
        self.stage_metrics.append(success_rate)
        
        # 최근 10 에피소드 평균
        if len(self.stage_metrics) >= 10:
            recent_avg = np.mean(self.stage_metrics[-10:])
            
            if recent_avg >= self.success_threshold:
                if self.current_stage < len(self.stages) - 1:
                    self.current_stage += 1
                    self.stage_metrics = []
                    return True
        
        return False
```

## 10.5 학습 안정화 기법

### 10.5.1 관측 정규화

```python
class RunningMeanStd:
    """온라인 평균/표준편차 계산"""
    
    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
```

### 10.5.2 보상 스케일링

```python
class RewardScaler:
    """보상 스케일링"""
    
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = None
    
    def scale(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """보상 스케일링"""
        if self.returns is None:
            self.returns = np.zeros_like(rewards)
        
        self.returns = self.returns * self.gamma * (1 - dones) + rewards
        self.ret_rms.update(self.returns.flatten())
        
        return rewards / np.sqrt(self.ret_rms.var + self.epsilon)
```



---

# 11. 실험 설계 및 평가 지표

## 11.1 베이스라인 정의

### 11.1.1 베이스라인 목록

| ID | 이름 | 설명 | 출동 규칙 | 재스케줄 규칙 |
|----|------|------|-----------|---------------|
| B0 | **Static-TSP** | 고정 순찰 경로, 이벤트 무시 | 무출동 | 없음 |
| B1 | **Greedy-Dispatch** | 이벤트 즉시 출동, 복귀 후 순찰 재개 | 항상 출동 | 없음 (원래 순서) |
| B2 | **Nearest-Replan** | 출동 후 가장 가까운 포인트부터 재스케줄 | 항상 출동 | Nearest-First |
| B3 | **Overdue-Replan** | 출동 후 공백 최대 포인트부터 재스케줄 | 항상 출동 | Overdue-First |
| B4 | **Threshold-Dispatch** | 긴급도 임계값 이상만 출동 | 조건부 출동 | Overdue-First |

### 11.1.2 베이스라인 구현

```python
class BaselinePolicy:
    """베이스라인 정책 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
    
    def select_action(self, obs: dict) -> Tuple[int, int]:
        """
        행동 선택
        
        Returns:
            (a_mode, a_replan)
        """
        raise NotImplementedError

class StaticTSPPolicy(BaselinePolicy):
    """B0: 고정 순찰, 이벤트 무시"""
    
    def __init__(self):
        super().__init__("Static-TSP")
    
    def select_action(self, obs: dict) -> Tuple[int, int]:
        return (0, 0)  # 순찰 계속, 재스케줄 없음

class GreedyDispatchPolicy(BaselinePolicy):
    """B1: 항상 출동, 재스케줄 없음"""
    
    def __init__(self):
        super().__init__("Greedy-Dispatch")
    
    def select_action(self, obs: dict) -> Tuple[int, int]:
        if obs["event_active"]:
            return (1, 0)  # 출동, 재스케줄 없음
        return (0, 0)

class NearestReplanPolicy(BaselinePolicy):
    """B2: 출동 후 Nearest-First 재스케줄"""
    
    def __init__(self):
        super().__init__("Nearest-Replan")
    
    def select_action(self, obs: dict) -> Tuple[int, int]:
        if obs["event_active"]:
            return (1, 1)  # 출동, Nearest-First
        return (0, 0)

class OverdueReplanPolicy(BaselinePolicy):
    """B3: 출동 후 Overdue-First 재스케줄"""
    
    def __init__(self):
        super().__init__("Overdue-Replan")
    
    def select_action(self, obs: dict) -> Tuple[int, int]:
        if obs["event_active"]:
            return (1, 2)  # 출동, Overdue-First
        return (0, 0)

class ThresholdDispatchPolicy(BaselinePolicy):
    """B4: 긴급도 임계값 기반 출동"""
    
    def __init__(self, urg_threshold: float = 0.5):
        super().__init__("Threshold-Dispatch")
        self.urg_threshold = urg_threshold
    
    def select_action(self, obs: dict) -> Tuple[int, int]:
        if obs["event_active"] and obs["event_urg"] >= self.urg_threshold:
            return (1, 2)  # 출동, Overdue-First
        return (0, 0)
```

## 11.2 평가 지표 정의

### 11.2.1 주요 지표 (Primary Metrics)

| 지표 | 기호 | 정의 | 단위 | 목표 |
|------|------|------|------|------|
| **평균 커버리지 공백비용** | $\bar{C}_{gap}$ | $\frac{1}{T}\int_0^T C_{gap}(t)dt$ | s | 최소화 |
| **이벤트 응답시간** | $T_{resp}$ | 이벤트 발생~도착 평균 시간 | s | 최소화 |
| **이벤트 성공률** | $S_{evt}$ | 처리 이벤트 / 총 이벤트 | % | 최대화 |
| **Nav2 실패율** | $F_{nav}$ | (ABORTED + TIMEOUT) / 총 goal | % | 최소화 |

### 11.2.2 보조 지표 (Secondary Metrics)

| 지표 | 기호 | 정의 | 단위 |
|------|------|------|------|
| 최대 공백 시간 | $g_{max}$ | $\max_i g_i$ | s |
| 공백 표준편차 | $\sigma_g$ | $\sqrt{\frac{1}{M}\sum_i(g_i - \bar{g})^2}$ | s |
| 총 이동 거리 | $L_{total}$ | 에피소드 총 경로 길이 | m |
| 에너지 소비 | $E_{total}$ | 배터리 소모량 | % |
| 평균 에피소드 보상 | $\bar{R}$ | 에피소드 누적 보상 평균 | - |

### 11.2.3 지표 계산 코드

```python
from dataclasses import dataclass
from typing import List

@dataclass
class EpisodeMetrics:
    """에피소드 평가 지표"""
    # Primary
    avg_gap_cost: float
    event_response_time: float
    event_success_rate: float
    nav_failure_rate: float
    
    # Secondary
    max_gap: float
    gap_std: float
    total_distance: float
    energy_consumed: float
    episode_reward: float
    
    # Metadata
    n_events: int
    n_events_handled: int
    n_nav_goals: int
    n_nav_failures: int
    episode_length: float

class MetricsCalculator:
    """지표 계산기"""
    
    def __init__(self, G_th: float = 300.0):
        self.G_th = G_th
        self.reset()
    
    def reset(self):
        """리셋"""
        self.gap_costs = []
        self.event_response_times = []
        self.events_total = 0
        self.events_handled = 0
        self.nav_goals = 0
        self.nav_failures = 0
        self.distances = []
        self.rewards = []
        self.gaps_history = []
    
    def record_gap_cost(self, gaps_s: List[float], dt: float):
        """공백 비용 기록"""
        M = len(gaps_s)
        C_gap = sum(max(0, g - self.G_th) for g in gaps_s) / max(M, 1)
        self.gap_costs.append(C_gap * dt)
        self.gaps_history.append(list(gaps_s))
    
    def record_event(self, handled: bool, response_time: float = None):
        """이벤트 기록"""
        self.events_total += 1
        if handled:
            self.events_handled += 1
            if response_time is not None:
                self.event_response_times.append(response_time)
    
    def record_nav_result(self, status: str, distance: float):
        """Nav2 결과 기록"""
        self.nav_goals += 1
        if status in ("ABORTED", "TIMEOUT"):
            self.nav_failures += 1
        self.distances.append(distance)
    
    def record_reward(self, reward: float):
        """보상 기록"""
        self.rewards.append(reward)
    
    def compute(self, episode_length: float) -> EpisodeMetrics:
        """최종 지표 계산"""
        # Primary
        avg_gap_cost = sum(self.gap_costs) / max(episode_length, 1.0)
        
        event_response_time = (
            np.mean(self.event_response_times) 
            if self.event_response_times else 0.0
        )
        
        event_success_rate = (
            self.events_handled / max(self.events_total, 1) * 100
        )
        
        nav_failure_rate = (
            self.nav_failures / max(self.nav_goals, 1) * 100
        )
        
        # Secondary
        if self.gaps_history:
            all_gaps = [g for gaps in self.gaps_history for g in gaps]
            max_gap = max(all_gaps) if all_gaps else 0.0
            gap_std = np.std(all_gaps) if all_gaps else 0.0
        else:
            max_gap = 0.0
            gap_std = 0.0
        
        total_distance = sum(self.distances)
        energy_consumed = total_distance * 0.01  # 간단한 모델
        episode_reward = sum(self.rewards)
        
        return EpisodeMetrics(
            avg_gap_cost=avg_gap_cost,
            event_response_time=event_response_time,
            event_success_rate=event_success_rate,
            nav_failure_rate=nav_failure_rate,
            max_gap=max_gap,
            gap_std=gap_std,
            total_distance=total_distance,
            energy_consumed=energy_consumed,
            episode_reward=episode_reward,
            n_events=self.events_total,
            n_events_handled=self.events_handled,
            n_nav_goals=self.nav_goals,
            n_nav_failures=self.nav_failures,
            episode_length=episode_length
        )
```

## 11.3 실험 시나리오

### 11.3.1 시나리오 정의

| 시나리오 | 맵 | 포인트 | 이벤트 빈도 | 이벤트 분포 | 목적 |
|----------|-----|--------|-------------|-------------|------|
| S1-Sparse | 20×20m | 20 | 0.3/min | 균등 | 기본 성능 |
| S2-Dense | 20×20m | 40 | 0.8/min | 균등 | 고밀도 순찰 |
| S3-Burst | 20×20m | 30 | 0.5/min + burst | 균등 | 폭증 대응 |
| S4-Hotspot | 20×20m | 30 | 0.6/min | 특정 구역 집중 | 핫스팟 대응 |
| S5-Large | 30×30m | 50 | 0.6/min | 균등 | 확장성 |

### 11.3.2 시나리오 설정 코드

```python
@dataclass
class ScenarioConfig:
    """시나리오 설정"""
    name: str
    map_width_m: float
    map_height_m: float
    n_points_target: int
    event_lambda_per_min: float
    event_burst_every_min: float = 0.0
    event_burst_factor: float = 1.0
    event_hotspot_zones: List[int] = None
    episode_duration_s: float = 600.0

SCENARIOS = {
    "S1-Sparse": ScenarioConfig(
        name="S1-Sparse",
        map_width_m=20.0, map_height_m=20.0,
        n_points_target=20,
        event_lambda_per_min=0.3
    ),
    "S2-Dense": ScenarioConfig(
        name="S2-Dense",
        map_width_m=20.0, map_height_m=20.0,
        n_points_target=40,
        event_lambda_per_min=0.8
    ),
    "S3-Burst": ScenarioConfig(
        name="S3-Burst",
        map_width_m=20.0, map_height_m=20.0,
        n_points_target=30,
        event_lambda_per_min=0.5,
        event_burst_every_min=5.0,
        event_burst_factor=4.0
    ),
    "S4-Hotspot": ScenarioConfig(
        name="S4-Hotspot",
        map_width_m=20.0, map_height_m=20.0,
        n_points_target=30,
        event_lambda_per_min=0.6,
        event_hotspot_zones=[0, 1, 4, 5]  # 좌상단 4개 zone
    ),
    "S5-Large": ScenarioConfig(
        name="S5-Large",
        map_width_m=30.0, map_height_m=30.0,
        n_points_target=50,
        event_lambda_per_min=0.6
    )
}
```

## 11.4 평가 프로토콜

### 11.4.1 평가 절차

```python
def evaluate_policy(policy, env, scenario: ScenarioConfig, 
                    n_episodes: int = 50, seeds: List[int] = None) -> dict:
    """
    정책 평가
    
    Args:
        policy: 평가할 정책
        env: 환경
        scenario: 시나리오 설정
        n_episodes: 평가 에피소드 수
        seeds: 시드 리스트 (재현성)
    
    Returns:
        평가 결과 딕셔너리
    """
    if seeds is None:
        seeds = list(range(n_episodes))
    
    all_metrics = []
    
    for i, seed in enumerate(seeds):
        # 환경 리셋
        obs = env.reset(seed=seed)
        metrics_calc = MetricsCalculator()
        
        done = False
        t_start = env.now()
        
        while not done:
            # 행동 선택
            a_mode, a_replan = policy.select_action(obs)
            
            # 환경 스텝
            obs, reward, done, info = env.step(a_mode, a_replan)
            
            # 지표 기록
            metrics_calc.record_gap_cost(env.gaps_s, info.get("delta_t", 1.0))
            metrics_calc.record_reward(reward)
            
            if info.get("event_handled"):
                metrics_calc.record_event(True, info.get("response_time"))
            
            metrics_calc.record_nav_result(
                info.get("nav_status", "SUCCEEDED"),
                info.get("path_length", 0.0)
            )
        
        # 에피소드 지표 계산
        episode_length = env.now() - t_start
        metrics = metrics_calc.compute(episode_length)
        all_metrics.append(metrics)
        
        print(f"Episode {i+1}/{n_episodes}: "
              f"Gap={metrics.avg_gap_cost:.2f}, "
              f"Resp={metrics.event_response_time:.1f}s, "
              f"Success={metrics.event_success_rate:.1f}%")
    
    # 집계
    return aggregate_metrics(all_metrics)

def aggregate_metrics(metrics_list: List[EpisodeMetrics]) -> dict:
    """지표 집계"""
    result = {}
    
    for field in ["avg_gap_cost", "event_response_time", "event_success_rate",
                  "nav_failure_rate", "max_gap", "gap_std", "total_distance",
                  "energy_consumed", "episode_reward"]:
        values = [getattr(m, field) for m in metrics_list]
        result[field] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
    
    return result
```

### 11.4.2 통계적 유의성 검정

```python
from scipy import stats

def compare_policies(metrics_a: List[EpisodeMetrics], 
                     metrics_b: List[EpisodeMetrics],
                     metric_name: str = "avg_gap_cost") -> dict:
    """
    두 정책 비교 (Welch's t-test)
    
    Returns:
        비교 결과 딕셔너리
    """
    values_a = [getattr(m, metric_name) for m in metrics_a]
    values_b = [getattr(m, metric_name) for m in metrics_b]
    
    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
    
    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    improvement = (mean_b - mean_a) / mean_b * 100 if mean_b != 0 else 0
    
    return {
        "metric": metric_name,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "improvement_pct": improvement,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
```

## 11.5 Ablation Study 설계

### 11.5.1 Ablation 조합

| ID | 이벤트 보상 | 커버리지 보상 | 안전 보상 | 효율 보상 | 목적 |
|----|------------|--------------|----------|----------|------|
| A0 | ✓ | ✓ | ✓ | ✓ | Full model |
| A1 | ✓ | ✗ | ✓ | ✓ | 커버리지 효과 |
| A2 | ✗ | ✓ | ✓ | ✓ | 이벤트 효과 |
| A3 | ✓ | ✓ | ✗ | ✓ | 안전 효과 |
| A4 | ✓ | ✓ | ✓ | ✗ | 효율 효과 |

### 11.5.2 후보 수 Ablation

| ID | 후보 수 | 후보 종류 | 목적 |
|----|---------|-----------|------|
| C0 | 1 | Keep-Order만 | 재스케줄 없음 |
| C1 | 3 | Keep, Nearest, Overdue | 최소 후보 |
| C2 | 6 | 전체 | Full model |
| C3 | 10 | 전체 + Zone 기반 | 확장 후보 |

### 11.5.3 Ablation 실험 코드

```python
def run_ablation_study(env, base_config: dict, 
                       ablation_configs: List[dict],
                       n_episodes: int = 30) -> dict:
    """
    Ablation study 실행
    
    Args:
        env: 환경
        base_config: 기본 설정
        ablation_configs: ablation 설정 리스트
        n_episodes: 에피소드 수
    
    Returns:
        결과 딕셔너리
    """
    results = {}
    
    for abl_cfg in ablation_configs:
        name = abl_cfg["name"]
        print(f"\n=== Running ablation: {name} ===")
        
        # 설정 병합
        config = {**base_config, **abl_cfg}
        
        # 정책 학습
        policy = train_policy(env, config)
        
        # 평가
        metrics = evaluate_policy(policy, env, n_episodes=n_episodes)
        results[name] = metrics
    
    return results
```

## 11.6 Sim2Real 검증 계획

![Sim2Real 전이 파이프라인](./diagrams/rendered/07_sim2real_pipeline.png)


### 11.6.1 3단계 검증 파이프라인

| 단계 | 환경 | 목적 | 성공 기준 |
|------|------|------|-----------|
| Stage 1 | Isaac Sim (학습) | 정책 학습 | 수렴 |
| Stage 2 | Gazebo (검증) | 전이 검증 | 성능 유지 |
| Stage 3 | Go2 실기기 | 최종 검증 | 실환경 동작 |

### 11.6.2 Domain Randomization

```python
@dataclass
class DomainRandomization:
    """도메인 랜덤화 설정"""
    # 물리 파라미터
    friction_range: Tuple[float, float] = (0.5, 1.5)
    mass_scale_range: Tuple[float, float] = (0.8, 1.2)
    
    # 센서 노이즈
    lidar_noise_std: float = 0.02
    odom_drift_rate: float = 0.01
    
    # 지연
    action_delay_range: Tuple[float, float] = (0.0, 0.1)
    observation_delay_range: Tuple[float, float] = (0.0, 0.05)
    
    # Nav2 변동
    nav_success_rate: float = 0.95
    nav_time_scale_range: Tuple[float, float] = (0.9, 1.3)

def apply_domain_randomization(env, dr: DomainRandomization, rng):
    """도메인 랜덤화 적용"""
    # 물리 파라미터
    env.set_friction(rng.uniform(*dr.friction_range))
    env.set_mass_scale(rng.uniform(*dr.mass_scale_range))
    
    # 센서 노이즈
    env.set_lidar_noise(dr.lidar_noise_std)
    env.set_odom_drift(dr.odom_drift_rate)
    
    # 지연
    env.set_action_delay(rng.uniform(*dr.action_delay_range))
    env.set_observation_delay(rng.uniform(*dr.observation_delay_range))
    
    # Nav2 변동
    env.set_nav_success_rate(dr.nav_success_rate)
    env.set_nav_time_scale(rng.uniform(*dr.nav_time_scale_range))
```

### 11.6.3 실기기 테스트 체크리스트

| 항목 | 확인 내용 | 통과 기준 |
|------|-----------|-----------|
| 로컬라이제이션 | AMCL 수렴 | 위치 오차 < 0.3m |
| Nav2 연결 | Action 서버 응답 | 5초 내 응답 |
| 센서 데이터 | LiDAR 토픽 수신 | 10Hz 이상 |
| 충돌 회피 | 장애물 회피 동작 | 충돌 없음 |
| 정책 추론 | 추론 시간 | < 50ms |
| 배터리 | 테스트 시간 | > 30분 |



---

# 12. 개발 로드맵 및 WBS

## 12.1 전체 로드맵 (12개월)

![프로젝트 로드맵 간트 차트](./diagrams/rendered/10_roadmap_gantt.png)


### 12.1.1 단계별 개요

| 단계 | 기간 | 목표 | 주요 산출물 |
|------|------|------|-------------|
| **Phase 1** | 0-3개월 | 기반 구축 | 환경, MDP, 베이스라인 |
| **Phase 2** | 4-6개월 | 통합 학습 | PPO 정책, Ablation |
| **Phase 3** | 7-9개월 | Sim2Real | Gazebo 검증, DR |
| **Phase 4** | 10-12개월 | 실증 | Go2 실기기 검증 |

### 12.1.2 상세 WBS

```
1. Phase 1: 기반 구축 (0-3개월)
   ├── 1.1 환경 구축 (0-1개월)
   │   ├── 1.1.1 Isaac Sim 설치 및 설정
   │   ├── 1.1.2 ROS2 Humble 환경 구성
   │   ├── 1.1.3 Nav2 스택 통합
   │   └── 1.1.4 Go2 URDF/USD 모델 준비
   │
   ├── 1.2 MDP 구현 (1-2개월)
   │   ├── 1.2.1 상태 공간 구현
   │   ├── 1.2.2 행동 공간 구현
   │   ├── 1.2.3 보상 함수 구현
   │   ├── 1.2.4 환경 클래스 (PatrolEnv) 구현
   │   └── 1.2.5 단위 테스트
   │
   ├── 1.3 후보 생성 시스템 (2-3개월)
   │   ├── 1.3.1 6종 후보 알고리즘 구현
   │   ├── 1.3.2 2단계 평가 시스템 구현
   │   ├── 1.3.3 Nav2 Planner 클라이언트 구현
   │   └── 1.3.4 통합 테스트
   │
   └── 1.4 베이스라인 (2-3개월)
       ├── 1.4.1 5종 베이스라인 구현
       ├── 1.4.2 베이스라인 평가
       └── 1.4.3 결과 분석 및 문서화

2. Phase 2: 통합 학습 (4-6개월)
   ├── 2.1 PPO 구현 (4개월)
   │   ├── 2.1.1 Actor-Critic 네트워크 구현
   │   ├── 2.1.2 롤아웃 버퍼 구현
   │   ├── 2.1.3 PPO 학습기 구현
   │   └── 2.1.4 하이퍼파라미터 튜닝
   │
   ├── 2.2 커리큘럼 학습 (5개월)
   │   ├── 2.2.1 커리큘럼 스케줄러 구현
   │   ├── 2.2.2 단계별 학습 실행
   │   └── 2.2.3 학습 곡선 분석
   │
   ├── 2.3 Ablation Study (5-6개월)
   │   ├── 2.3.1 보상 항 Ablation
   │   ├── 2.3.2 후보 수 Ablation
   │   └── 2.3.3 결과 분석
   │
   └── 2.4 중간 논문 (6개월)
       ├── 2.4.1 실험 결과 정리
       ├── 2.4.2 논문 작성
       └── 2.4.3 학회 투고

3. Phase 3: Sim2Real (7-9개월)
   ├── 3.1 Gazebo 환경 (7개월)
   │   ├── 3.1.1 Gazebo 환경 구축
   │   ├── 3.1.2 Isaac Sim 정책 이식
   │   └── 3.1.3 성능 비교
   │
   ├── 3.2 Domain Randomization (8개월)
   │   ├── 3.2.1 DR 파라미터 정의
   │   ├── 3.2.2 DR 적용 학습
   │   └── 3.2.3 전이 성능 평가
   │
   └── 3.3 실기기 준비 (9개월)
       ├── 3.3.1 Go2 하드웨어 설정
       ├── 3.3.2 센서 캘리브레이션
       └── 3.3.3 통신 테스트

4. Phase 4: 실증 (10-12개월)
   ├── 4.1 실기기 테스트 (10-11개월)
   │   ├── 4.1.1 실내 환경 테스트
   │   ├── 4.1.2 이벤트 대응 테스트
   │   └── 4.1.3 장시간 운영 테스트
   │
   ├── 4.2 성능 분석 (11개월)
   │   ├── 4.2.1 Sim vs Real 비교
   │   ├── 4.2.2 실패 사례 분석
   │   └── 4.2.3 개선점 도출
   │
   └── 4.3 최종 논문 (12개월)
       ├── 4.3.1 전체 결과 정리
       ├── 4.3.2 SCI 저널 논문 작성
       └── 4.3.3 투고 및 리뷰 대응
```

## 12.2 마일스톤

| 마일스톤 | 시점 | 산출물 | 검증 기준 |
|----------|------|--------|-----------|
| M1 | 3개월 | 환경 + 베이스라인 | 베이스라인 평가 완료 |
| M2 | 6개월 | PPO 정책 + 중간 논문 | 베이스라인 대비 20% 개선 |
| M3 | 9개월 | Sim2Real 검증 | Gazebo 성능 유지 |
| M4 | 12개월 | 실기기 검증 + 최종 논문 | 실환경 동작 확인 |

## 12.3 리스크 관리

| 리스크 | 확률 | 영향 | 대응 방안 |
|--------|------|------|-----------|
| Nav2 불안정 | 중 | 고 | 파라미터 튜닝, 대안 플래너 |
| 학습 불수렴 | 중 | 고 | 커리큘럼 조정, 보상 재설계 |
| Sim2Real Gap | 고 | 중 | DR 강화, 점진적 전이 |
| 하드웨어 문제 | 저 | 고 | 예비 장비 확보 |
| 일정 지연 | 중 | 중 | 버퍼 기간 확보, 범위 조정 |

---

# 13. 프로젝트 구조 및 코드 명세

![시스템 아키텍처](./diagrams/rendered/01_system_architecture.png)


## 13.1 디렉토리 구조

```
patrol_rl/
├── README.md
├── pyproject.toml
├── setup.py
│
├── configs/                          # 설정 파일
│   ├── env_config.yaml               # 환경 설정
│   ├── ppo_config.yaml               # PPO 하이퍼파라미터
│   ├── reward_weights.yaml           # 보상 가중치
│   ├── nav2_params.yaml              # Nav2 파라미터
│   └── scenarios/                    # 시나리오 설정
│       ├── S1_sparse.yaml
│       ├── S2_dense.yaml
│       └── ...
│
├── patrol_rl/                        # 메인 패키지
│   ├── __init__.py
│   │
│   ├── env/                          # 환경 모듈
│   │   ├── __init__.py
│   │   ├── patrol_env.py             # PatrolEnv 클래스
│   │   ├── map_generator.py          # 맵 생성
│   │   ├── point_generator.py        # 포인트 생성
│   │   ├── event_generator.py        # 이벤트 생성
│   │   └── safety_shield.py          # Safety Shield
│   │
│   ├── candidate/                    # 후보 생성 모듈
│   │   ├── __init__.py
│   │   ├── candidate.py              # Candidate 데이터 구조
│   │   ├── generator.py              # CandidateGenerator
│   │   ├── heuristics.py             # 6종 휴리스틱
│   │   └── evaluator.py              # 2단계 평가기
│   │
│   ├── nav2/                         # Nav2 인터페이스
│   │   ├── __init__.py
│   │   ├── planner_client.py         # Planner 클라이언트
│   │   ├── navigation_client.py      # Navigation 클라이언트
│   │   └── utils.py                  # 유틸리티
│   │
│   ├── reward/                       # 보상 모듈
│   │   ├── __init__.py
│   │   ├── calculator.py             # RewardCalculator
│   │   └── ablation.py               # Ablation 버전
│   │
│   ├── policy/                       # 정책 모듈
│   │   ├── __init__.py
│   │   ├── actor_critic.py           # ActorCritic 네트워크
│   │   ├── baselines.py              # 베이스라인 정책
│   │   └── ppo.py                    # PPO 학습기
│   │
│   ├── training/                     # 학습 모듈
│   │   ├── __init__.py
│   │   ├── buffer.py                 # RolloutBuffer
│   │   ├── curriculum.py             # 커리큘럼 스케줄러
│   │   ├── trainer.py                # PPOTrainer
│   │   └── callbacks.py              # 학습 콜백
│   │
│   ├── evaluation/                   # 평가 모듈
│   │   ├── __init__.py
│   │   ├── metrics.py                # MetricsCalculator
│   │   ├── evaluator.py              # 정책 평가기
│   │   └── comparison.py             # 정책 비교
│   │
│   └── utils/                        # 유틸리티
│       ├── __init__.py
│       ├── config.py                 # 설정 로더
│       ├── logging.py                # 로깅
│       └── visualization.py          # 시각화
│
├── scripts/                          # 실행 스크립트
│   ├── train.py                      # 학습 스크립트
│   ├── evaluate.py                   # 평가 스크립트
│   ├── ablation.py                   # Ablation 스크립트
│   └── sim2real.py                   # Sim2Real 스크립트
│
├── launch/                           # ROS2 런치 파일
│   ├── isaac_sim.launch.py
│   ├── gazebo.launch.py
│   └── real_robot.launch.py
│
├── tests/                            # 테스트
│   ├── test_env.py
│   ├── test_candidate.py
│   ├── test_reward.py
│   └── test_policy.py
│
├── notebooks/                        # 분석 노트북
│   ├── 01_baseline_analysis.ipynb
│   ├── 02_training_curves.ipynb
│   └── 03_ablation_results.ipynb
│
└── docs/                             # 문서
    ├── api.md
    ├── installation.md
    └── experiments.md
```

## 13.2 핵심 인터페이스 정의

### 13.2.1 환경 인터페이스

```python
# patrol_rl/env/patrol_env.py

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np

class PatrolEnvInterface(ABC):
    """순찰 환경 인터페이스"""
    
    @abstractmethod
    def reset(self, seed: int = 0) -> np.ndarray:
        """
        환경 리셋
        
        Args:
            seed: 에피소드 시드
        
        Returns:
            초기 관측 벡터
        """
        pass
    
    @abstractmethod
    def step(self, a_mode: int, a_cand: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        환경 스텝
        
        Args:
            a_mode: 출동 여부 (0: 순찰, 1: 출동)
            a_cand: 재스케줄 후보 인덱스
        
        Returns:
            obs: 관측 벡터
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        pass
    
    @abstractmethod
    def get_action_mask(self) -> Dict[str, np.ndarray]:
        """
        행동 마스크 반환
        
        Returns:
            {"mode_mask": [2], "replan_mask": [N+1]}
        """
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, Any]:
        """관측 공간 정의"""
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Dict[str, Any]:
        """행동 공간 정의"""
        pass
```

### 13.2.2 정책 인터페이스

```python
# patrol_rl/policy/base.py

from abc import ABC, abstractmethod
from typing import Tuple, Dict
import numpy as np

class PolicyInterface(ABC):
    """정책 인터페이스"""
    
    @abstractmethod
    def select_action(self, obs: np.ndarray, 
                      mask: Dict[str, np.ndarray] = None,
                      deterministic: bool = False) -> Tuple[int, int]:
        """
        행동 선택
        
        Args:
            obs: 관측 벡터
            mask: 행동 마스크
            deterministic: 결정적 선택 여부
        
        Returns:
            (a_mode, a_replan)
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """정책 저장"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """정책 로드"""
        pass
```

### 13.2.3 후보 생성기 인터페이스

```python
# patrol_rl/candidate/generator.py

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from .candidate import Candidate

class CandidateGeneratorInterface(ABC):
    """후보 생성기 인터페이스"""
    
    @abstractmethod
    def generate(self,
                 points_xy: List[Tuple[float, float]],
                 gaps_s: List[float],
                 weights: List[float],
                 queue_current: List[int],
                 start_xy: Tuple[float, float],
                 event_xy: Optional[Tuple[float, float]] = None) -> List[Candidate]:
        """
        후보 생성
        
        Returns:
            후보 리스트
        """
        pass
```

## 13.3 설정 파일 템플릿

### 13.3.1 환경 설정 (env_config.yaml)

```yaml
# configs/env_config.yaml

environment:
  # 관측 설정
  Kobs: 6                    # 후보 피처 수
  Emax: 5                    # 이벤트 피처 수
  lidar_samples: 64          # LiDAR 샘플 수
  
  # 행동 설정
  H: 3                       # Rolling horizon
  n_candidates: 6            # 후보 수
  
  # 에피소드 설정
  max_steps: 100             # 최대 스텝
  episode_duration_s: 600.0  # 에피소드 길이
  
  # 물리 설정
  visit_radius: 0.5          # 방문 판정 반경
  robot_radius: 0.35         # 로봇 반경
  
  # 시간 설정
  decision_tick_s: 0.1       # 의사결정 틱
  decision_max_s: 60.0       # 최대 대기 시간

map:
  width_m: 20.0
  height_m: 20.0
  resolution: 0.05
  obstacle_density: 0.08
  min_corridor_cells: 8
  min_free_ratio: 0.35

points:
  n_target: 40
  poisson_r_m: 1.8
  poisson_k: 30
  margin_cells: 6
  n_zone_x: 4
  n_zone_y: 4

events:
  lambda_per_min: 0.6
  burst_every_min: 10.0
  burst_factor: 4.0
  urg_dist: "beta(2,2)"
  conf_dist: "beta(3,1.5)"
```

### 13.3.2 PPO 설정 (ppo_config.yaml)

```yaml
# configs/ppo_config.yaml

ppo:
  # 환경
  n_envs: 8
  n_steps: 256
  batch_size: 64
  n_epochs: 4
  
  # 학습
  lr: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  clip_range_vf: null
  
  # 정규화
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  
  # 스케줄
  total_timesteps: 2000000
  lr_schedule: "linear"
  
  # 로깅
  log_interval: 10
  save_interval: 50000
  eval_interval: 10000
  eval_episodes: 10

network:
  hidden_dim: 256
  n_layers: 2
  activation: "relu"
  orthogonal_init: true

normalization:
  normalize_obs: true
  normalize_reward: true
  clip_obs: 10.0
  clip_reward: 10.0
```

### 13.3.3 보상 가중치 (reward_weights.yaml)

```yaml
# configs/reward_weights.yaml

reward:
  # 이벤트 관련
  w_evt_delay: 1.0
  R_hit: 50.0
  
  # 순찰 관련
  w_pat: 1.0
  G_th: 300.0
  
  # 안전 관련
  R_col: 100.0
  R_abort: 30.0
  R_timeout: 20.0
  
  # 효율 관련
  lambda_L: 0.1
  lambda_Theta: 0.05

# Ablation 설정
ablation:
  A0:  # Full model
    use_event: true
    use_coverage: true
    use_safety: true
    use_efficiency: true
  A1:  # No coverage
    use_event: true
    use_coverage: false
    use_safety: true
    use_efficiency: true
  A2:  # No event
    use_event: false
    use_coverage: true
    use_safety: true
    use_efficiency: true
```

---

# 14. 참고 문헌

## 14.1 강화학습 기초

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). **Proximal Policy Optimization Algorithms**. arXiv preprint arXiv:1707.06347.

2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). **High-Dimensional Continuous Control Using Generalized Advantage Estimation**. arXiv preprint arXiv:1506.02438.

## 14.2 로봇 내비게이션

3. Macenski, S., Martín, F., White, R., & Clavero, J. G. (2020). **The Marathon 2: A Navigation System**. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

4. Mavrogiannis, C., Baldini, F., Wang, A., Zhao, D., Trautman, P., Steinfeld, A., & Oh, J. (2023). **Core Challenges of Social Robot Navigation: A Survey**. ACM Transactions on Human-Robot Interaction.

## 14.3 Vehicle Routing & Dispatch

5. Kool, W., van Hoof, H., & Welling, M. (2019). **Attention, Learn to Solve Routing Problems!**. International Conference on Learning Representations (ICLR).

6. Hottung, A., & Tierney, K. (2020). **Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem**. European Conference on Artificial Intelligence (ECAI).

7. Joe, W., & Lau, H. C. (2020). **Deep Reinforcement Learning Approach to Solve Dynamic Vehicle Routing Problem with Stochastic Customers**. International Conference on Automated Planning and Scheduling (ICAPS).

8. Nazari, M., Oroojlooy, A., Snyder, L. V., & Takáč, M. (2018). **Reinforcement Learning for Solving the Vehicle Routing Problem**. Advances in Neural Information Processing Systems (NeurIPS).

## 14.4 Multi-Robot Task Allocation

9. Korsah, G. A., Stentz, A., & Dias, M. B. (2013). **A Comprehensive Taxonomy for Multi-Robot Task Allocation**. The International Journal of Robotics Research.

10. Rizk, Y., Awad, M., & Tunstel, E. W. (2019). **Cooperative Heterogeneous Multi-Robot Systems: A Survey**. ACM Computing Surveys.

11. Otte, M. W., Kuhlman, M. J., & Sofge, D. (2020). **Multi-Robot Task Allocation Using Graph Neural Networks**. IEEE International Conference on Robotics and Automation (ICRA).

## 14.5 Sim2Real Transfer

12. Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World**. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

13. Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018). **Sim-to-Real Transfer of Robotic Control with Dynamics Randomization**. IEEE International Conference on Robotics and Automation (ICRA).

14. Muratore, F., Ramos, F., Turk, G., Yu, W., Gienger, M., & Peters, J. (2022). **Robot Learning from Randomized Simulations: A Review**. Frontiers in Robotics and AI.

## 14.6 로봇 보행 제어

15. Margolis, G., Yang, G., Paigwar, K., Chen, T., & Agrawal, P. (2022). **Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior**. Conference on Robot Learning (CoRL).

16. Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., & Hutter, M. (2019). **Learning Agile and Dynamic Motor Skills for Legged Robots**. Science Robotics.

## 14.7 실환경 로봇 정책

17. 김민수, 이정훈, 박철수. (2024). **실환경 통합점수 기반 로봇 정책 선택 기법**. 한국로봇학회 논문지.

---

# 부록

## A. 수식 기호 총정리

| 기호 | 의미 | 단위/범위 |
|------|------|-----------|
| $\mathcal{S}$ | 상태 공간 | - |
| $\mathcal{A}$ | 행동 공간 | - |
| $P$ | 전이 확률 | - |
| $r$ | 보상 함수 | - |
| $\gamma$ | 할인율 | $(0, 1)$ |
| $\pi_\theta$ | 정책 | - |
| $(x_t, y_t)$ | 로봇 위치 | m |
| $\psi_t$ | 로봇 yaw | rad |
| $(v_t, \omega_t)$ | 선속/각속 | m/s, rad/s |
| $b_t$ | 배터리 SoC | $[0, 1]$ |
| $L_t$ | LiDAR 거리 | m |
| $z_t$ | 이벤트 존재 | $\{0, 1\}$ |
| $(x_t^e, y_t^e)$ | 이벤트 위치 | m |
| $u_t$ | 이벤트 긴급도 | $[0, 1]$ |
| $c_t$ | 이벤트 신뢰도 | $[0, 1]$ |
| $\tau_t$ | 이벤트 경과시간 | s |
| $g_i$ | 포인트 공백 시간 | s |
| $G_{th}$ | 공백 임계값 | s |
| $C_{gap}$ | 공백 비용 | s |
| $H$ | Rolling horizon | - |
| $\Pi_H$ | 방문 시퀀스 | - |
| $\ell_L$ | 경로 길이 | m |
| $\ell_T$ | ETA 추정 | s |
| $\Theta$ | 누적 회전량 | rad |
| $w_i$ | 포인트 중요도 | $\geq 0$ |

## B. 코드 실행 예시

### B.1 학습 실행

```bash
# 환경 설정
export CUDA_VISIBLE_DEVICES=0
export ROS_DOMAIN_ID=42

# Isaac Sim 실행 (별도 터미널)
./isaac-sim.sh --headless

# 학습 실행
python scripts/train.py \
    --config configs/ppo_config.yaml \
    --env-config configs/env_config.yaml \
    --reward-config configs/reward_weights.yaml \
    --output-dir outputs/exp001 \
    --seed 42
```

### B.2 평가 실행

```bash
python scripts/evaluate.py \
    --policy outputs/exp001/policy_final.pt \
    --scenario S2-Dense \
    --n-episodes 50 \
    --output-dir outputs/exp001/eval
```

### B.3 Ablation 실행

```bash
python scripts/ablation.py \
    --base-config configs/ppo_config.yaml \
    --ablation-config configs/reward_weights.yaml \
    --output-dir outputs/ablation
```

---

**문서 끝**



---

# 15. 부록 A: Unitree Go2 인터페이스 명세

## 15.1 개요

본 연구는 Unitree Go2 로봇을 최종 실증 플랫폼으로 사용한다. 따라서 시뮬레이션 환경과 실제 로봇 간의 인터페이스 일관성을 확보하는 것이 매우 중요하다. Unitree Go2는 `unitree_ros2` 패키지를 통해 ROS2 인터페이스를 제공하며, 주요 서비스 및 토픽은 다음과 같다.

## 15.2 주요 ROS2 토픽 및 서비스

### 15.2.1 내비게이션 및 SLAM

Go2는 내장된 SLAM 및 내비게이션 기능을 ROS2 Action 인터페이스로 제공한다. 이는 Nav2 스택과 유사한 형태로, `NavigateToPose` 액션을 통해 목표 지점 이동을 요청할 수 있다.

- **Action:** `/navigate_to_pose`
- **Type:** `nav2_msgs/action/NavigateToPose`
- **Goal:** `pose` (목표 지점), `behavior_tree` (동작 트리, 비워두면 기본값 사용)
- **Result:** `result` (결과 코드)
- **Feedback:** `distance_remaining`, `estimated_time_remaining`

**본 연구와의 통합:**
- 환경의 `navigation_client`는 이 Action 클라이언트를 사용하여 Go2의 이동을 제어한다.
- ETA 추정 캘리브레이션 시 피드백의 `estimated_time_remaining`을 활용할 수 있다.

### 15.2.2 스포츠 모드 (저수준 제어)

`unitree_legged_sdk`를 통해 더 저수준의 제어가 가능하다. `SportMode` 서비스를 통해 로봇의 이동, 자세 등을 직접 제어할 수 있다.

- **Service:** `/sport_mode`
- **Type:** `unitree_msgs/srv/SportMode`
- **Request:** `mode` (0: idle, 1: force_stand, 2: walk, ...), `gait_type`, `speed_level`, `foot_raise_height`, `body_height`, `position`, `euler`, `velocity` 등

**본 연구와의 통합:**
- 본 연구는 Nav2 기반의 고수준 제어를 기본으로 하므로, `SportMode`는 주로 초기화, 안전 정지, 특정 자세 제어 등 보조적인 역할로 사용된다.
- 예를 들어, 이벤트 확인을 위해 특정 방향을 주시해야 할 때 `euler` 각도를 직접 제어하는 데 사용할 수 있다.

### 15.2.3 센서 데이터

| 토픽 | 메시지 타입 | 설명 |
|---|---|---|
| `/imu/data` | `sensor_msgs/Imu` | IMU 데이터 |
| `/odom` | `nav_msgs/Odometry` | Odometry 정보 |
| `/scan` | `sensor_msgs/LaserScan` | 2D LiDAR 데이터 |
| `/camera/color/image_raw` | `sensor_msgs/Image` | 전방 카메라 이미지 |

**본 연구와의 통합:**
- 시뮬레이션에서 생성하는 센서 데이터는 이 메시지 타입들과 동일한 형식을 따라야 한다.
- 특히 LiDAR 데이터는 정책의 입력으로 직접 사용되므로, 시뮬레이션과 실제 로봇 간의 스펙(각도 범위, 샘플 수, 노이즈 특성)을 일치시키는 것이 중요하다.

## 15.3 Sim2Real 전이 시 고려사항

1. **좌표계 일치:** 시뮬레이션의 `map` 프레임과 실제 로봇의 `map` 프레임을 일치시킨다.
2. **Action/Service 지연:** 실제 로봇과의 통신 및 실행 지연을 시뮬레이션에 반영해야 한다. (Domain Randomization의 `action_delay`)
3. **물리 파라미터:** Go2의 실제 무게, 마찰 계수, 최대 속도/가속도 등을 시뮬레이션에 최대한 가깝게 설정한다.
4. **Nav2 파라미터:** 시뮬레이션에서 사용한 Nav2 파라미터(`nav2_params.yaml`)를 실제 로봇에도 동일하게 적용하고, 필요한 경우 미세 조정한다.


---

# 16. 부록 B: 중간 산출물 논문 정의

## 16.1 개요

본 연구는 12개월의 로드맵 동안 총 3편의 중간 논문과 1편의 최종 SCI 저널 논문 제출을 목표로 한다. 각 중간 논문은 특정 기술적 기여를 중심으로 작성되며, 후속 연구의 기반이 된다.

## 16.2 논문 1: 통합 순찰-출동 환경 및 베이스라인 연구

- **예상 제목:** "A Simulation Framework for Integrated Patrol and Dispatch Operations of Mobile Robots using Semi-Markov Decision Processes"
- **투고 목표:** 국내 학술대회 (KRoC, KACC 등) 또는 국제 워크샵 (ICRA/IROS 워크샵)
- **시점:** 3개월차 (Phase 1 완료 후)
- **주요 내용:**
    1. **문제 정의:** 동적 이벤트 대응과 장기 순찰 커버리지 간의 트레이드오프 문제 공식화
    2. **SMDP 환경:** Nav2 기반의 가변 시간 액션을 모델링하는 `PatrolEnv` 시뮬레이션 환경 설계 및 구현
    3. **베이스라인:** 5가지 휴리스틱 기반 베이스라인 정책(Greedy, Overdue-first 등)의 성능 분석
    4. **평가 지표:** 커버리지, 응답시간, 안전성 등 다중 목표 평가 지표 제안

## 16.3 논문 2: 후보 기반 재스케줄링을 위한 강화학습 정책 연구

- **예상 제목:** "Learning to Dispatch and Reschedule: A Hybrid Approach with Heuristic Candidate Generation and Deep Reinforcement Learning"
- **투고 목표:** 국제 학술대회 (ICRA, IROS, CoRL 등)
- **시점:** 6개월차 (Phase 2 완료 후)
- **주요 내용:**
    1. **하이브리드 행동 공간:** 6가지 휴리스틱으로 재스케줄링 후보를 생성하고, RL 에이전트가 이를 선택하는 구조 제안
    2. **PPO 기반 학습:** 복합 행동 공간(출동+재스케줄링)을 학습하기 위한 PPO 알고리즘 적용
    3. **실험 결과:** 제안하는 RL 정책이 모든 베이스라인 정책 대비 주요 성능 지표(커버리지, 이벤트 대응)에서 우월함을 입증
    4. **Ablation Study:** 보상 함수 각 항의 기여도 및 후보 수에 따른 성능 변화 분석

## 16.4 논문 3: 운영 정책의 Sim2Real 전이 및 Domain Randomization 연구

- **예상 제목:** "Sim2Real for Operations: Transferring High-Level Robotic Policies from Simulation to Reality with Domain Randomization"
- **투고 목표:** 국제 학술대회 (ICRA, IROS, CoRL 등) 또는 로봇 분야 SCI(E) 저널 (T-RO, RA-L 등)
- **시점:** 9개월차 (Phase 3 완료 후)
- **주요 내용:**
    1. **운영 수준 Sim2Real:** 저수준 제어가 아닌, 고수준 운영 정책의 Sim2Real 문제 정의
    2. **Domain Randomization:** Nav2의 실패/지연, 센서 노이즈, 물리 파라미터 변동 등을 포함한 DR 기법 적용
    3. **2단계 전이 검증:** Isaac Sim(학습) → Gazebo(검증) 파이프라인을 통한 전이 성능 정량 분석
    4. **결과:** DR을 적용한 정책이 적용하지 않은 정책에 비해 현실과의 성능 격차(Reality Gap)를 크게 줄임을 입증
