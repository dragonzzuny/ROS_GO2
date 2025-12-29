
# 통합 순찰-출동 RL 정책 코드 검토 및 수정 가이드라인

**문서 버전:** 1.0  
**검토 일자:** 2025년 12월 30일  
**작성자:** Manus AI (AI 수석 연구원)

---

## 1. 개요

본 문서는 제출된 `rl_dispatch_mvp` 코드베이스를 `R&D 계획서 v4` 및 `AI 개발 지침 프롬프트`에 명시된 설계 원칙과 기술 명세에 따라 심층 검토한 결과를 담고 있습니다. 검토 결과, 프로젝트의 핵심 골격은 잘 구현되어 있으나, 현실적인 시뮬레이션과 안정적인 학습을 위해 반드시 수정되어야 할 몇 가지 중대한 불일치 사항들이 발견되었습니다.

각 항목은 **[문제점 분석]**, **[설계 근거]**, **[수정 제안]**의 세 부분으로 구성되어 있으며, 즉시 개발에 착수할 수 있도록 구체적인 코드 레벨의 가이드라인을 제공합니다. 본 가이드라인을 따름으로써 불필요한 실험을 최소화하고, 계획된 연구개발 목표를 효과적으로 달성할 수 있을 것으로 기대합니다.


## 2. 핵심 설계 불일치 및 수정 가이드라인

### 2.1. (심각도: 최상) SMDP 가변 할인율(Variable Discount Factor) 미적용

**[문제점 분석]**

현재 구현된 PPO 알고리즘의 GAE(Generalized Advantage Estimation) 계산 로직(`src/rl_dispatch/algorithms/buffer.py`)은 모든 시간 스텝(SMDP 매크로 스텝)에 대해 고정된 할인율(`self.gamma`)을 사용하고 있습니다. 이는 Nav2 Goal 수행에 가변 시간(`Δt_k`)이 소요된다는 프로젝트의 핵심 전제인 **SMDP(Semi-MDP) 모델링을 정면으로 위배**하는 중대한 설계 오류입니다. 고정 할인율은 시간의 길고 짧음에 관계없이 미래 보상을 동일하게 취급하므로, 정책이 효율적인 경로(짧은 `Δt_k`)를 선호하도록 학습하는 것을 방해합니다.

**[설계 근거]**

`R&D 계획서 v4`의 3.2.3절 「할인율 조정」 파트에서는 SMDP의 가변 시간을 보상 평가에 정확히 반영하기 위해 다음과 같은 시간 종속적인 할인율 `γ_k`를 명시하고 있습니다:

> $$
> \gamma_k = \gamma^{\Delta t_k / \Delta t_{base}}
> $$
> 여기서 `Δt_base`는 기준 시간 단위 (예: 1초)입니다.

이 원칙은 `AI 개발 지침 프롬프트`의 2.2절 「SMDP 모델링」에서도 다시 한번 강조됩니다.

**[수정 제안]**

`RolloutBuffer`가 각 스텝의 `Δt_k`를 저장하고, 이를 GAE 계산 시 반영하도록 수정해야 합니다.

1.  **`RolloutBuffer.add` 메서드 시그니처 변경:**
    `step()` 함수로부터 `nav_time` (즉, `Δt_k`)을 추가로 전달받아 저장합니다.

    ```python
    # src/rl_dispatch/algorithms/buffer.py
    
    # 1. dones 옆에 nav_times 배열 추가
    self.nav_times = np.zeros(buffer_size, dtype=np.float32)
    
    # 2. add 메서드에 nav_time 파라미터 추가
    def add(self, obs, action, log_prob, reward, value, done, nav_time):
        # ... 기존 저장 로직 ...
        self.nav_times[self.pos] = nav_time
        self.pos += 1
    ```

2.  **`compute_returns_and_advantages` 메서드 수정:**
    GAE 계산 루프 내에서 고정 `self.gamma` 대신, 각 스텝의 `nav_time`에 따라 계산된 `gamma_k`를 사용합니다.

    ```python
    # src/rl_dispatch/algorithms/buffer.py
    
    def compute_returns_and_advantages(self, last_value: float, last_done: bool = False):
        # ...
        for step in reversed(range(self.buffer_size)):
            # ... next_non_terminal, next_value 계산 ...

            # 3. 가변 할인율 계산 (delta_t_base = 1.0 가정)
            nav_time = self.nav_times[step]
            gamma_k = self.gamma ** nav_time 

            # 4. TD error 계산 시 gamma_k 적용
            delta = (
                self.rewards[step] +
                gamma_k * next_value * next_non_terminal -
                self.values[step]
            )

            # 5. GAE 계산 시 gamma_k 적용
            last_gae_lam = (
                delta +
                gamma_k * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values
    ```

3.  **`PPOAgent` 및 `PatrolEnv` 연동 수정:**
    `patrol_env.py`의 `step` 함수에서 반환하는 `info` 딕셔너리에 `nav_time`을 포함하고, 학습 루프(`train.py`)에서 이를 `buffer.add`에 전달하도록 수정해야 합니다.


### 2.2. (심각도: 상) 불완전한 행동 마스킹 (Action Masking) 로직

**[문제점 분석]**

현재 `PatrolEnv`의 `step` 함수 내 `_is_action_valid` 메서드는 잘못된 행동에 대해 경고 메시지만 출력할 뿐, 이 정보를 정책 네트워크에 전달하여 학습에 활용하지 못하고 있습니다. (`patrol_env.py:289`) 정책 네트워크(`ActorCriticNetwork`)는 `mode_mask`를 인자로 받을 수 있도록 설계되어 있으나, 실제 학습 루프에서는 이 마스크가 생성되거나 전달되지 않습니다. 이로 인해 에이전트는 **'이벤트가 없을 때 출동을 시도하는' 등의 불가능한 행동을 학습 과정에서 배제하지 못하며**, 이는 학습 비효율성의 핵심 원인이 됩니다.

**[설계 근거]**

`AI 개발 지침 프롬프트` 3.2절 「행동 공간」과 `R&D 계획서 v4` 5.2절 「행동 마스킹」에서는 행동 마스킹의 중요성을 명확히 기술하고 있습니다.

> **행동 마스킹(Action Masking)은 필수적이다.** 이벤트가 없으면 $a_k^{mode}=1$은 마스킹되어야 한다. 이를 통해 정책이 불가능한 행동을 선택하지 않도록 한다. (AI 개발 지침 프롬프트)

> Infeasible 후보, 이벤트 없음($z_t=0$)일 때 $a^{mode}=1$ 무의미, Keep-out zone으로 향하는 후보는 마스킹해야 한다. (R&D 계획서 v4)

**[수정 제안]**

환경(`PatrolEnv`)이 매 스텝마다 유효한 행동 마스크를 계산하여 `info` 딕셔너리를 통해 반환하고, 학습 루프가 이를 정책 네트워크에 전달하는 구조로 변경해야 합니다.

1.  **`PatrolEnv._is_action_valid`를 `_compute_action_mask`로 변경:**
    단순 유효성 검사를 넘어, 정책에 직접 전달할 마스크 배열을 생성합니다.

    ```python
    # src/rl_dispatch/env/patrol_env.py

    def _compute_action_mask(self) -> np.ndarray:
        """현재 상태에서 유효한 행동 마스크를 계산합니다."""
        # mode_mask: [patrol_가능여부, dispatch_가능여부]
        mode_mask = np.ones(2, dtype=np.float32)
        
        # 이벤트가 없으면 출동(dispatch) 불가
        if not self.current_state.has_event:
            mode_mask[1] = 0.0
        
        # 배터리가 부족하면 출동 불가 (R&D 계획서 확장 제안)
        if self.current_state.robot.battery_level < 0.2:
             mode_mask[1] = 0.0

        # (추가) 재스케줄링 후보 마스크도 여기서 생성 가능
        # replan_mask = np.array([c.feasible for c in self.current_state.candidates], dtype=np.float32)

        return mode_mask
    ```

2.  **`PatrolEnv.step` 및 `reset`에서 마스크 반환:**
    `info` 딕셔너리에 `action_mask` 키를 추가하여 마스크를 반환합니다.

    ```python
    # src/rl_dispatch/env/patrol_env.py

    def reset(...):
        # ...
        obs, info = self._get_obs_and_info()
        return obs, info

    def step(...):
        # ... (상태 업데이트 후)
        obs, info = self._get_obs_and_info()
        return obs, rewards.total, terminated, truncated, info

    def _get_obs_and_info(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """관측과 정보 딕셔너리를 생성합니다."""
        obs = self.obs_processor.process(self.current_state, update_stats=False)
        info = {
            # ... 기존 info 내용 ...
            "action_mask": self._compute_action_mask()
        }
        return obs.vector, info
    ```

3.  **학습 루프(`train.py`)에서 마스크 전달:**
    `RolloutBuffer`에 `action_masks`를 추가하고, `agent.get_action_and_value` 호출 시 마스크를 전달합니다.

    ```python
    # 예시: train.py 내 학습 루프

    # buffer에 action_masks 저장 공간 추가 필요
    # ...
    next_obs, reward, done, info = env.step(action)
    buffer.add(obs, action, log_prob, reward, value, done, info["nav_time"], info["action_mask"])
    # ...

    # PPOAgent.update 내부
    # ...
    for batch in self.buffer.get(...):
        obs, actions, old_log_probs, advantages, returns, old_values, masks = batch
        _, new_log_probs, entropy, values = self.network.get_action_and_value(
            obs, action=actions, mode_mask=masks
        )
        # ... 손실 계산 ...
    ```

이러한 수정을 통해 정책은 처음부터 불가능한 행동을 탐색에서 제외하게 되어 학습이 훨씬 안정적이고 빠르게 수렴될 것입니다.

### 2.3. (심각도: 상) 비현실적인 LiDAR 및 Nav2 시뮬레이션

**[문제점 분석]**

현재 시뮬레이션 환경은 Sim2Real 격차를 유발할 수 있는 두 가지 주요 비현실적 요소를 포함하고 있습니다.

1.  **LiDAR 데이터:** `_simulate_lidar` 함수(`patrol_env.py:653`)는 단순히 최대 거리에 정규분포 노이즈를 추가하는 방식으로, 실제 장애물이나 맵 구조를 전혀 반영하지 못합니다. 이러한 가짜 LiDAR 데이터로 학습된 정책은 실제 환경에서 장애물을 회피하거나 맵의 구조를 이해하는 능력을 갖출 수 없습니다.
2.  **Nav2 시뮬레이션:** `SimulatedNav2` 클래스(`nav2_interface.py:155`)는 경로 길이를 직선 거리(Euclidean distance)로 계산하고, 여기에 단순한 노이즈만 추가합니다. 이는 벽이나 장애물을 통과하는 비현실적인 경로를 가정하며, 실제 Nav2의 복잡한 경로 계획(A* 등) 및 동역학적 제약과 큰 차이가 있습니다.

**[설계 근거]**

`R&D 계획서 v4`의 8.1절 「시뮬레이션 환경 요구사항」에서는 "실제와 유사한 센서 데이터(LiDAR)"와 "Nav2의 동작(경로 계획, 시간 소요)을 정확히 모사"하는 것을 명시하고 있습니다. `AI 개발 지침 프롬프트` 3.1절의 상태 공간 정의에서 LiDAR가 64차원의 중요한 입력으로 포함된 이유도 실제 환경의 기하학적 정보를 정책이 활용하도록 하기 위함입니다.

**[수정 제안]**

단기적으로는 실제 맵을 표현하는 2D 그리드를 도입하고, 이를 기반으로 LiDAR와 Nav2 시뮬레이션을 대폭 개선해야 합니다.

1.  **맵 표현을 위한 2D Occupancy Grid 도입:**
    `PatrolEnv`가 초기화될 때, `configs/`에 정의된 맵에 해당하는 2D `np.ndarray` (0: free, 1: occupied)를 로드하거나 생성하도록 합니다. `multi_map_env.py`의 `CoverageHeatmap`과 유사한 그리드 개념을 활용할 수 있습니다.

    ```python
    # src/rl_dispatch/env/patrol_env.py

    class PatrolEnv(gym.Env):
        def __init__(self, ...):
            # ...
            self.occupancy_grid = self._load_occupancy_grid(self.env_config.map_name)
            # ...

        def _load_occupancy_grid(self, map_name: str) -> np.ndarray:
            # 실제로는 png, pgm 파일 등에서 로드해야 함
            # 여기서는 간단한 예시로 생성
            grid = np.zeros((100, 100), dtype=np.uint8)
            # 벽 추가 (예시)
            grid[0, :] = 1
            grid[-1, :] = 1
            grid[:, 0] = 1
            grid[:, -1] = 1
            grid[20:40, 30:35] = 1 # 장애물
            return grid
    ```

2.  **LiDAR 시뮬레이션 개선 (Ray-casting):**
    `_simulate_lidar` 함수를 `occupancy_grid`에 대해 Ray-casting을 수행하는 로직으로 교체합니다. `Bresenham's line algorithm` 같은 효율적인 알고리즘을 사용하여 각 LiDAR 각도에 대해 광선을 쏘고, 그리드 상의 장애물과 처음 만나는 지점까지의 거리를 계산합니다.

3.  **Nav2 시뮬레이션 개선 (A* Pathfinding):**
    `SimulatedNav2` 클래스가 `occupancy_grid`를 인자로 받도록 수정합니다. `get_eta`와 `navigate_to_goal` 메서드에서 직선 거리 대신, 그리드 상에서 **A* 알고리즘**을 사용하여 실제 경로를 찾고, 그 경로 길이를 기반으로 이동 시간을 계산하도록 변경합니다. 이는 벽을 통과하지 않고 장애물을 돌아가는 현실적인 경로와 시간을 제공합니다.

    ```python
    # src/rl_dispatch/navigation/nav2_interface.py

    class SimulatedNav2(NavigationInterface):
        def __init__(self, occupancy_grid: np.ndarray, ...):
            self.grid = occupancy_grid
            # ...

        def get_eta(self, start: Tuple[float, float], goal: Tuple[float, float]) -> float:
            # 1. 좌표를 그리드 셀 인덱스로 변환
            # 2. A* 알고리즘으로 경로 탐색 (예: scikit-image, networkx 라이브러리 활용)
            # 3. 경로 길이(셀 수) * 해상도 = 실제 거리 (m)
            # 4. 거리 / 평균 속도 = ETA
            # 경로 탐색 실패 시 매우 큰 값 반환
            pass
    ```

이러한 개선은 Sim2Real 격차를 크게 줄여, 시뮬레이션에서 학습된 정책이 실제 로봇에서도 유의미한 성능을 발휘할 수 있는 기반이 됩니다.

## 3. 기타 권장 개선사항

### 3.1. 후보 생성 휴리스틱 미구현

-   **문제점:** `candidate_generator.py`가 존재하지만, 실제 휴리스틱 로직(Nearest-First, Overdue-First 등)이 구현되어 있지 않고 플레이스홀더로 남아있습니다.
-   **수정 제안:** `R&D 계획서 v4` 7.2절에 명시된 6가지 휴리스틱을 모두 구현해야 합니다. 각 휴리스틱은 `(patrol_order, metrics)` 튜플을 반환하는 함수로 작성하고, `CandidateGenerator`가 이를 호출하여 최종 후보 리스트를 생성하도록 합니다.

### 3.2. 보상 함수 설계 불일치

-   **문제점:** `reward_calculator.py`의 `calculate_reward` 함수가 `R^{pat}`(순찰 커버리지) 항을 계산할 때, 단순히 현재 목표 지점 방문에 대한 보너스(`patrol_visit_bonus`)만 고려하고 있습니다. 설계상 핵심인 **'공백 비용(coverage gap penalty)'**이 누락되었습니다.
-   **수정 제안:** `R^{pat}` 계산 시, 모든 순찰 지점의 `time_since_visit`을 확인하여, 특정 임계값을 초과하는 지점들에 대해 패널티를 부과하는 로직을 추가해야 합니다. 이는 `AI 개발 지침 프롬프트` 2.1절 「통합 학습」의 핵심 요구사항입니다.

### 3.3. 설정 파일 관리

-   **문제점:** `default.yaml`에 모든 설정이 포함되어 있어 편리하지만, `map_warehouse.yaml`과 같이 맵 특정 설정이 분리되어 있어 혼동의 여지가 있습니다. 또한, `reward_weights.yaml`이 별도로 존재하지 않습니다.
-   **수정 제안:** `AI 개발 지침 프롬프트` 3.3절의 권장사항에 따라, 보상 가중치는 `configs/reward_weights.yaml`로 분리하고, 기본 학습 파라미터는 `configs/training/ppo_default.yaml` 등으로, 환경 관련 설정은 `configs/envs/office.yaml` 등으로 구조화하는 것을 권장합니다. 이를 통해 실험 관리가 용이해집니다.

## 4. 결론

본 검토에서 제시된 수정 가이드라인, 특히 **SMDP 가변 할인율 적용, 완전한 행동 마스킹 구현, 현실적인 센서/액추에이터 시뮬레이션**은 프로젝트의 성공을 위해 필수적인 선결 과제입니다. 이 문제들을 해결하지 않고 학습을 진행할 경우, 학습이 수렴하지 않거나, 수렴하더라도 실제 환경에서는 전혀 작동하지 않는 정책이 될 가능성이 매우 높습니다.

제시된 가이드라인에 따라 코드를 수정한 후, `단계 1: 환경 구축`의 검증 기준을 다시 한번 철저히 확인하고 `단계 2: PPO 통합 및 학습`으로 진행할 것을 강력히 권고합니다.


---

## 부록 A: 수정 항목 요약 테이블

| 항목 번호 | 심각도 | 문제 요약 | 관련 파일 | 예상 작업 시간 |
|:---------:|:------:|:----------|:----------|:--------------:|
| 2.1 | **최상** | SMDP 가변 할인율 미적용 | `buffer.py`, `ppo.py`, `patrol_env.py` | 4~6 시간 |
| 2.2 | **상** | 행동 마스킹 미연동 | `patrol_env.py`, `ppo.py`, `train.py` | 3~4 시간 |
| 2.3 | **상** | 비현실적 LiDAR/Nav2 시뮬레이션 | `patrol_env.py`, `nav2_interface.py` | 8~12 시간 |
| 3.1 | 중 | 후보 생성 휴리스틱 미구현 | `candidate_generator.py` | 4~6 시간 |
| 3.2 | 중 | 순찰 커버리지 패널티 누락 | `reward_calculator.py` | 2~3 시간 |
| 3.3 | 하 | 설정 파일 구조화 | `configs/` | 1~2 시간 |

---

## 부록 B: 권장 수정 순서

1.  **SMDP 가변 할인율 적용 (2.1):** 이 문제는 학습의 근본적인 정확성에 영향을 미치므로 가장 먼저 해결해야 합니다.
2.  **행동 마스킹 구현 (2.2):** 학습 효율성을 크게 높여주므로 두 번째로 수정합니다.
3.  **순찰 커버리지 패널티 추가 (3.2):** 보상 함수의 핵심 항목이므로, 학습 전에 반드시 추가해야 합니다.
4.  **후보 생성 휴리스틱 구현 (3.1):** 행동 공간의 품질을 결정하므로 학습 전에 완료해야 합니다.
5.  **LiDAR/Nav2 시뮬레이션 개선 (2.3):** 작업량이 많으므로, 위 항목들이 완료된 후 진행합니다. 이 항목은 Sim2Real 단계 전까지 완료되면 됩니다.
6.  **설정 파일 구조화 (3.3):** 실험 관리의 편의성을 위한 것이므로, 본격적인 실험 시작 전에 정리합니다.

---

*본 문서는 `rl_dispatch_mvp` 코드베이스의 `R&D 계획서 v4` 및 `AI 개발 지침 프롬프트` 준수 여부를 검토한 결과입니다.*
