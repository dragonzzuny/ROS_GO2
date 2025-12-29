# AI 개발 지침 프롬프트
## 통합 순찰-출동 정책 강화학습 프로젝트

---

## 1. 프로젝트 개요

**목표:** 단일 로봇 순찰 환경에서 출동(Dispatch)과 재스케줄링(Rescheduling)을 통합적으로 결정하는 강화학습 정책을 개발하고, Nav2 기반으로 Isaac Sim에서 Unitree Go2 실기기까지 검증한다.

**기간:** 12개월 (M1-M12)

**핵심 차별점:** 출동과 재스케줄링을 단일 RL 정책으로 학습하며, SMDP 모델링을 통해 Nav2 Goal 수행의 가변 시간을 정확히 반영한다.

---

## 2. 핵심 원칙 (Golden Rules)

### 2.1. 통합 학습 (Unified Learning)

출동(Dispatch)과 재스케줄링(Rescheduling)은 분리된 문제가 아니다. 항상 단일 RL 정책으로 두 가지를 동시에 결정해야 한다. 이를 위해 보상 함수의 순찰 커버리지 항($R^{pat}$)이 핵심 역할을 한다. 이 항이 없으면 "출동을 하면 커버리지가 무너지는 비용"이 학습에 반영되지 않아 통합 학습의 의미가 사라진다.

### 2.2. SMDP 모델링 (Semi-MDP)

Nav2 Goal 수행 시간($\Delta t_k$)은 경로, 장애물, 로봇 상태에 따라 가변적이다. 모든 보상 계산과 할인율($\gamma_k = \gamma^{\Delta t_k / \Delta t_{base}}$)은 이 가변 시간을 정확히 반영해야 한다. 고정 시간 스텝 MDP를 가정하면 안 된다.

### 2.3. 후보 기반 행동 공간 (Candidate-based Action Space)

순찰 포인트 순열(Permutation)을 직접 다루는 것은 조합 폭발을 일으킨다. 행동 공간은 사전에 생성된 K개의 후보 중 하나를 선택하는 Categorical 분포로 제한한다. 이를 통해 계산 복잡도를 선형으로 유지한다.

### 2.4. 구현 우선주의 (Implementation First)

모든 아이디어는 R&D 계획서 v4에 명시된 Python 의사코드를 기반으로 구체적인 코드로 먼저 작성되어야 한다. 추상적인 논의나 이론적 검증만으로는 진행하지 않는다.

### 2.5. 엄격한 버전 관리 (Strict Version Control)

모든 코드, 설정 파일, 학습된 모델, 실험 결과는 Git으로 관리한다. 커밋 메시지는 명확하고 추적 가능해야 한다. 예: `feat(reward): add patrol coverage term` 또는 `fix(env): correct SMDP discount calculation`.

---

## 3. 기술 명세 (Technical Specifications)

### 3.1. 상태 공간 ($o_t$)

**총 77차원 벡터**를 사용한다. (R&D 계획서 v4, 4.1.2 참조)

| 구성 요소 | 차원 | 정규화 방법 |
|-----------|------|-----------|
| 목표 상대벡터 ($\Delta x_g, \Delta y_g$) | 2 | $\div d_{max}$ (예: 50m) |
| 각도 표현 ($\cos\psi, \sin\psi$) | 2 | 이미 정규화 |
| 동역학 $(v, \omega)$ | 2 | $\div v_{max}, \div \omega_{max}$ |
| 배터리 $b$ | 1 | 이미 정규화 |
| LiDAR $L^{(1:K)}$ | 64 | $\div L_{max}$ (예: 10m) |
| 이벤트 피처 $(z, u, c, \tilde{\tau})$ | 4 | 이미 정규화 |
| 순찰 요약 $(\tilde{d}^{pat}, \tilde{\kappa}^{pat})$ | 2 | 정규화 |
| **총** | **77** | |

모든 입력은 명시된 방법에 따라 **반드시 정규화**되어야 한다. 정규화되지 않은 입력은 학습 불안정성을 초래한다.

### 3.2. 행동 공간 ($a_k$)

**복합 이산 행동 공간(Composite Discrete Action Space):** $a_k = (a_k^{mode}, a_k^{replan})$

-   $a_k^{mode} \in \{0, 1\}$: 출동(1) / 순찰(0) 결정. Bernoulli 분포로 모델링.
-   $a_k^{replan} \in \{0, ..., N\}$: 재스케줄링 후보 선택. Categorical 분포로 모델링.

**행동 마스킹(Action Masking)**은 필수적이다. 이벤트가 없으면 $a_k^{mode}=1$은 마스킹되어야 한다. 이를 통해 정책이 불가능한 행동을 선택하지 않도록 한다.

### 3.3. 보상 함수 ($r_t$)

**4가지 주요 항목**의 가중합으로 구성된다. (R&D 계획서 v4, 6장 참조)

$$r_t = R^{evt} + R^{pat} + R^{safe} + R^{eff}$$

| 항목 | 정의 | 의미 |
|------|------|------|
| $R^{evt}$ | 이벤트 대응 | 지연 페널티 + 성공 보너스 |
| $R^{pat}$ | 순찰 커버리지 | 공백 비용 (통합학습 핵심) |
| $R^{safe}$ | 안전 | 충돌/Nav2 실패 페널티 |
| $R^{eff}$ | 효율 | 경로 길이 페널티 |

`RewardCalculator` 클래스를 구현하여 모든 보상 계산을 중앙에서 관리한다. 가중치는 `configs/reward_weights.yaml` 파일로 관리하며, 실험 중 쉽게 조정할 수 있어야 한다.

### 3.4. 재스케줄링 후보 생성

**6가지 휴리스틱**을 사용하여 후보 경로를 생성한다. (R&D 계획서 v4, 7.2 참조)

1. Keep-Order: 원래 순서 유지
2. Nearest-First: 현재 위치에서 가장 가까운 포인트부터
3. Overdue-First: 공백 시간이 가장 긴 포인트부터
4. Overdue-Balance: 공백 시간과 거리의 균형
5. Risk-Weighted: 중요도 가중치 반영
6. Balanced-Coverage: 전체 커버리지 균형

**2단계 평가 시스템**을 따른다.

1. **1단계 (Approximate):** Euclidean 거리로 모든 휴리스틱 평가 후 상위 K개 선택.
2. **2단계 (Precise):** Nav2 Planner(`ComputePathThroughPoses`)로 상위 K개만 정밀 평가.

이를 통해 계산 효율성과 정확성의 균형을 맞춘다.

### 3.5. 강화학습 알고리즘

**PPO (Proximal Policy Optimization)**를 사용한다. (R&D 계획서 v4, 10장 참조)

-   **Actor-Critic** 구조를 따르며, 네트워크는 다음 구조를 준수한다:
    -   Shared Encoder: FC(256, ReLU) → FC(256, ReLU) → LayerNorm
    -   Actor Head: FC(128, ReLU) → Mode Output (Softmax, dim=2) + Candidate Output (Softmax, dim=K)
    -   Critic Head: FC(128, ReLU) → Value Output (Linear, dim=1)

-   **GAE (Generalized Advantage Estimation)**를 사용하여 Advantage를 계산한다.

-   **관측 정규화(RunningMeanStd)**와 **보상 스케일링(RewardScaler)**을 적용하여 학습을 안정화한다.

-   **클립 범위(Clip Range):** 0.2 (권장)

-   **학습률(Learning Rate):** 3e-4 (초기값, 커리큘럼에 따라 조정)

---

## 4. 개발 단계별 지침

### 단계 1: 환경 구축 (M1-M2)

**목표:** 완전히 작동하는 시뮬레이션 환경을 구축하고 모든 베이스라인이 정상 작동하는지 검증한다.

**요구사항:**

-   `PatrolEnv` Gym 환경 클래스 구현 (R&D 계획서 v4, 9.6 참조).
-   `EpisodeBundle` 데이터클래스 및 `build_episode_bundle` 함수 구현 (R&D 계획서 v4, 9.5 참조).
-   `Nav2PlannerClient` 및 `Nav2NavigationClient` 구현 및 테스트 (R&D 계획서 v4, 8.2-8.3 참조).
-   5가지 베이스라인 정책 구현 (B0-B4, R&D 계획서 v4, 11.1 참조).

**검증 기준:**

-   랜덤 에이전트가 환경에서 100스텝 이상 오류 없이 실행되는가?
-   모든 베이스라인 정책(B0-B4)이 환경에서 정상 작동하는가?
-   관측 벡터가 정확히 77차원인가? (후보 포함 시 77 + K*4)
-   보상 계산이 모든 항목을 포함하고 있는가?

**산출물:**

-   `patrol_env.py`: PatrolEnv 클래스
-   `nav2_clients.py`: Nav2 클라이언트
-   `baselines.py`: 5가지 베이스라인 정책
-   `episode_builder.py`: 에피소드 생성 로직
-   `reward_calculator.py`: 보상 계산기
-   `test_env.py`: 환경 테스트 스크립트

### 단계 2: PPO 통합 및 학습 (M3-M6)

**목표:** PPO 알고리즘을 구현하고 단순한 시나리오에서 학습이 진행되는지 확인한다.

**요구사항:**

-   PPO 알고리즘 및 Actor-Critic 네트워크 구현 (R&D 계획서 v4, 10.2-10.3 참조).
-   `RewardCalculator` 및 `CandidateGenerator` 구현 (R&D 계획서 v4, 6-7 참조).
-   학습 루프 및 로깅(Tensorboard) 구현.
-   커리큘럼 학습 스케줄러 구현 (R&D 계획서 v4, 10.4 참조).

**검증 기준:**

-   가장 단순한 시나리오(10x10m, 10 points, no event)에서 커버리지 비용이 감소하는가?
-   이벤트 발생 시나리오에서 B1(Greedy)보다 높은 총 보상을 달성하는가?
-   Tensorboard 로그에서 정책 엔트로피가 안정적으로 유지되는가?
-   학습 곡선이 단조증가하지 않고 적절한 변동성을 보이는가? (과적합 방지)

**산출물:**

-   `ppo.py`: PPO 알고리즘 구현
-   `models.py`: Actor-Critic 네트워크
-   `train.py`: 학습 루프
-   `curriculum.py`: 커리큘럼 스케줄러
-   학습 로그 및 모델 가중치 (Tensorboard, checkpoint)

### 단계 3: Sim2Real 및 검증 (M7-M9)

**목표:** Domain Randomization을 적용하고 Gazebo 환경에서 정책을 검증한다.

**요구사항:**

-   Domain Randomization 기법 적용 (Nav2 지연/실패 주입, 맵 변동성 증대).
-   Gazebo 환경에서 학습된 정책 테스트.
-   Ablation Study 설계 및 실행 (보상 항 제거 실험).

**검증 기준:**

-   Gazebo 환경에서 Sim 환경 대비 성능 저하가 30% 이내인가?
-   Ablation Study 결과가 예상과 일치하는가? (예: $R^{pat}$ 제거 시 커버리지 성능 저하)
-   실패 시나리오(Nav2 abort/timeout)에서 정책이 안정적으로 폴백하는가?

**산출물:**

-   `sim2real.py`: Domain Randomization 구현
-   `ablation.py`: Ablation Study 스크립트
-   Gazebo 테스트 리포트 (성능 지표, 영상)
-   Ablation Study 결과 (표, 그래프)

### 단계 4: 실기기 적용 (M10-M12)

**목표:** Unitree Go2 로봇에 정책을 적용하고 실환경에서 검증한다.

**요구사항:**

-   Unitree Go2 ROS2 인터페이스 연동 (R&D 계획서 v4, 부록 A 참조).
-   실제 사무실 환경에서 정책 테스트.
-   최종 성능 평가 및 논문 작성.

**검증 기준:**

-   실환경에서 1시간 이상 안정적으로 순찰 및 출동 임무를 수행하는가?
-   정량적 목표(평균 커버리지 비용 20% 감소 등)를 달성하는가?
-   안전 사고(충돌, 넘어짐)가 없는가?

**산출물:**

-   `go2_interface.py`: Go2 연동 코드
-   실증 영상 및 데이터
-   최종 성능 평가 보고서
-   논문 초안

---

## 5. 제약 및 금지사항

### 5.1. 절대 금지 사항

-   **출동과 재스케줄링 로직을 분리하여 개발하지 말 것.** 두 가지는 단일 정책으로 통합되어야 한다.
-   **고정 시간 스텝(Fixed Timestep) MDP를 가정하지 말 것.** SMDP 모델링을 정확히 따를 것.
-   **Nav2 파라미터를 튜닝하지 말 것.** 초기 설정값을 유지하고 RL 정책이 이에 적응하도록 할 것. Nav2 튜닝은 이 프로젝트의 범위가 아님.

### 5.2. 주의 사항

-   모든 코드에는 타입 힌트(Type Hint)와 상세한 Docstring을 작성할 것.
-   모든 설정은 YAML 파일로 관리하고, 하드코딩을 지양할 것.
-   모든 실험은 재현 가능해야 한다. 시드(Seed)를 명시하고 로그를 남길 것.
-   모든 모델 가중치와 실험 결과는 버전 관리 대상이 아니다. `.gitignore`에 추가할 것.

---

## 6. 코드 구조 및 명명 규칙

### 6.1. 디렉토리 구조

```
patrol_rl/
├── configs/                    # 설정 파일
│   ├── reward_weights.yaml
│   ├── env_config.yaml
│   └── train_config.yaml
├── patrol_env/                 # 환경 모듈
│   ├── __init__.py
│   ├── env.py                 # PatrolEnv 클래스
│   ├── episode_builder.py      # 에피소드 생성
│   ├── nav2_clients.py         # Nav2 클라이언트
│   ├── reward_calculator.py    # 보상 계산
│   ├── candidate_generator.py  # 후보 생성
│   └── safety_shield.py        # 안전 방패
├── rl/                         # RL 모듈
│   ├── __init__.py
│   ├── ppo.py                 # PPO 알고리즘
│   ├── models.py              # Actor-Critic 네트워크
│   ├── buffer.py              # 경험 버퍼
│   └── curriculum.py          # 커리큘럼 스케줄러
├── baselines/                  # 베이스라인 정책
│   ├── __init__.py
│   └── policies.py            # B0-B4 정책
├── scripts/                    # 실행 스크립트
│   ├── train.py               # 학습 스크립트
│   ├── eval.py                # 평가 스크립트
│   ├── test_env.py            # 환경 테스트
│   └── ablation.py            # Ablation Study
├── tests/                      # 단위 테스트
│   ├── test_env.py
│   ├── test_reward.py
│   └── test_candidates.py
├── logs/                       # 학습 로그 (Git 제외)
├── models/                     # 학습된 모델 (Git 제외)
├── results/                    # 실험 결과 (Git 제외)
├── requirements.txt            # 의존성
├── README.md                   # 프로젝트 설명
└── .gitignore                  # Git 무시 파일
```

### 6.2. 명명 규칙

-   **클래스:** PascalCase (예: `PatrolEnv`, `RewardCalculator`)
-   **함수/메서드:** snake_case (예: `compute_reward`, `build_episode_bundle`)
-   **상수:** UPPER_SNAKE_CASE (예: `MAX_STEPS`, `DEFAULT_GAMMA`)
-   **변수:** snake_case (예: `obs_dim`, `action_space`)

---

## 7. 참고 자료

-   **R&D 계획서 v4:** 모든 기술 명세의 원본 문서
-   **Unitree Go2 개발 문서:** https://support.unitree.com/home/en/developer/Basic_services
-   **Nav2 공식 문서:** https://nav2.org/
-   **PPO 논문:** Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
-   **관련 논문들:** R&D 계획서 v4의 참고 문헌 참조

---

## 8. 체크리스트 (각 단계 완료 시)

### 단계 1 완료 체크리스트

-   [ ] PatrolEnv가 Gym 인터페이스를 완벽히 구현했는가?
-   [ ] 모든 베이스라인이 100+ 스텝 오류 없이 실행되는가?
-   [ ] 관측 벡터가 정확히 77차원(+후보 피처)인가?
-   [ ] 보상 계산이 모든 항목을 포함하는가?
-   [ ] 코드가 Git에 커밋되었는가?

### 단계 2 완료 체크리스트

-   [ ] PPO 알고리즘이 정상 작동하는가?
-   [ ] 단순 시나리오에서 커버리지 비용이 감소하는가?
-   [ ] Tensorboard 로그가 생성되는가?
-   [ ] 모든 코드에 타입 힌트와 Docstring이 있는가?
-   [ ] 코드가 Git에 커밋되었는가?

### 단계 3 완료 체크리스트

-   [ ] Domain Randomization이 적용되었는가?
-   [ ] Gazebo 환경에서 테스트했는가?
-   [ ] Ablation Study 결과가 예상과 일치하는가?
-   [ ] 테스트 리포트가 작성되었는가?
-   [ ] 코드가 Git에 커밋되었는가?

### 단계 4 완료 체크리스트

-   [ ] Go2 인터페이스가 구현되었는가?
-   [ ] 실환경에서 1시간 이상 테스트했는가?
-   [ ] 정량적 목표를 달성했는가?
-   [ ] 최종 보고서가 작성되었는가?
-   [ ] 논문 초안이 완성되었는가?

---

**이 프롬프트는 프로젝트 진행 중 언제든지 참조할 수 있습니다. 명확하지 않은 부분이 있으면 R&D 계획서 v4의 해당 장을 참고하세요.**
