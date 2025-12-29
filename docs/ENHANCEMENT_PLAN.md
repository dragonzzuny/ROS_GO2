# 시스템 개선 계획 (System Enhancement Plan)

**작성일**: 2025-12-29
**목표**: Nav2 통합, 산업안전 표준 적용, 순찰 루트 최적화

---

## 🎯 개선 요구사항

### 1. **Nav2 통합** ✅ 설계 중
- **현재**: 간소화된 시뮬레이션 (`_simulate_navigation`)
- **목표**: 실제 Nav2 인터페이스 사용
- **방안**: 추상화 레이어 설계 (시뮬/실제 전환 가능)

### 2. **이벤트 구조 확장** ✅ 완료
- **현재**: urgency (0-1), confidence (0-1)
- **추가**:
  - ✅ **위험도** (risk_level: 1-9, 산업안전평가 기준)
  - ✅ **이벤트명** (event_name: "화재", "침입", "낙하물" 등)
  - ✅ 34개 산업안전 이벤트 타입 정의
  - ✅ PatrolEnv 통합 완료 (이벤트 생성 로직 업데이트)
- **호환성**: urgency를 risk_level로부터 계산 (backward compatible)

### 3. **충전 스테이션** ✅ 완료
- **추가**: 각 맵마다 고정된 충전 스테이션 위치
- **구현**:
  - ✅ EnvConfig에 charging_station_position 추가
  - ✅ 6개 맵 모두 충전 스테이션 위치 설정
  - ✅ 맵 특성에 맞는 전략적 위치 선정 (입구/경비실 근처)

### 4. **순찰 루트 전략 확장** ⏸️ 대기
- **현재**: 6개 전략
- **목표**: 10개 전략 (heuristic_method.md 권장)
- **추가 필요**:
  - Shortest-ETA First (Nav2 ETA 기반)
  - Overdue-Threshold First
  - Minimal-Deviation Insert
  - Windowed Replan

---

## 📊 구현 현황

| 항목 | 현재 | 목표 | 상태 |
|------|------|------|------|
| Event 구조 | urgency + confidence | + risk_level + event_name | ✅ 완료 |
| 산업안전 이벤트 | - | 34개 이벤트 타입 | ✅ 완료 |
| PatrolEnv 통합 | - | 이벤트 생성 로직 업데이트 | ✅ 완료 |
| 충전 스테이션 | - | 6개 맵 고정 위치 | ✅ 완료 |
| Nav2 인터페이스 | 간소화 시뮬 | 추상화 레이어 | 🔄 설계 중 |
| 순찰 전략 | 6개 | 10개 | ⏸️ 6/10 |
| 맵별 초기 루트 | 랜덤 | 맵 특성 기반 | ⏸️ 대기 |

---

## 🔧 구현 상세

### 1. 산업안전 이벤트 시스템 ✅

**파일**: `src/rl_dispatch/core/event_types.py`

#### 34개 이벤트 타입 (위험도별)

**고위험 (7-9)**: 즉시 대응 필수
```
🔴 [9] 화재감지        - 화재/폭발
🔴 [9] 가스누출        - 위험물질
🔴 [9] 추락위험        - 낙하/추락
🔴 [9] 화학물질누출    - 위험물질
🔴 [8] 연기감지        - 화재/폭발
🔴 [8] 무단침입        - 침입/보안
🔴 [8] 낙하물감지      - 낙하/추락
🔴 [8] 유류누출        - 위험물질
🔴 [8] 쓰러짐감지      - 이상행동
🔴 [7] 과열감지        - 화재/폭발
🔴 [7] 비인가구역접근  - 침입/보안
🔴 [7] 구조물손상      - 설비고장
🔴 [7] 배관파열        - 누수/누출
🔴 [7] 비상구차단      - 통로차단
🔴 [7] 폭력의심        - 이상행동
```

**중위험 (4-6)**: 조사 및 대응 필요
```
🟡 [6] 도난의심        - 침입/보안
🟡 [6] 바닥파손        - 설비고장
🟡 [6] 전력이상        - 설비고장
🟡 [6] 싸움감지        - 이상행동
🟡 [5] 누수감지        - 누수/누출
🟡 [5] 설비이상음      - 설비고장
🟡 [5] 환기시스템고장  - 환경이상
🟡 [5] 통로차단        - 통로차단
🟡 [5] 이상행동        - 이상행동
🟡 [4] 배회            - 이상행동
🟡 [4] 미끄러움위험    - 환경이상
🟡 [4] 온도이상        - 환경이상
```

**저위험 (1-3)**: 모니터링, 일상 점검
```
🟢 [3] 조명고장        - 설비고장
🟢 [3] 소음이상        - 환경이상
🟢 [3] 점검필요        - 설비고장
🟢 [2] 청결이상        - 환경이상
🟢 [1] 정상순찰        - 환경이상
```

#### Event 생성 예시

```python
from rl_dispatch.core.event_types import get_random_event_name, get_event_risk_level
from rl_dispatch.core.types_extended import Event

# 랜덤 이벤트 생성 (위험도 역가중)
event_name = get_random_event_name(np_random)
risk_level = get_event_risk_level(event_name)

event = Event(
    x=25.0,
    y=30.0,
    risk_level=risk_level,  # 1-9
    event_name=event_name,   # "화재감지", "무단침입" 등
    confidence=0.92,
    detection_time=120.0,
    event_id=1
)

# Backward compatibility
print(f"Urgency: {event.urgency:.2f}")  # risk_level/9.0 → 0.0-1.0
```

---

### 2. 충전 스테이션 시스템 ✅

**목적**: 각 맵에 고정된 충전 스테이션 위치 설정

#### EnvConfig 확장

```python
# src/rl_dispatch/core/config.py

@dataclass
class EnvConfig:
    # Map configuration
    map_width: float = 50.0
    map_height: float = 50.0
    patrol_points: List[Tuple[float, float]] = ...
    patrol_point_priorities: List[float] = ...
    charging_station_position: Tuple[float, float] = (5.0, 5.0)  # NEW
    ...
```

#### 맵별 충전 스테이션 위치

| 맵 이름 | 크기 (m) | 충전 스테이션 위치 | 설명 |
|---------|----------|-------------------|------|
| map_large_square | 100×100 | (5.0, 5.0) | 좌하단 입구 근처 |
| map_corridor | 120×30 | (5.0, 15.0) | 복도 시작 지점 |
| map_l_shaped | 80×80 | (5.0, 5.0) | L자 시작점 근처 |
| map_office_building | 90×70 | (10.0, 10.0) | 정문 입구 근처 |
| map_campus | 150×120 | (30.0, 20.0) | 경비실 근처 |
| map_warehouse | 140×100 | (10.0, 10.0) | 하역장 입구 근처 |

#### 맵 설정 예시

```yaml
# configs/map_large_square.yaml

env:
  map_width: 100.0
  map_height: 100.0

  patrol_points:
    - [15.0, 15.0]
    - [50.0, 15.0]
    ...

  # Charging station - 충전 스테이션 (좌하단 입구 근처)
  charging_station_position: [5.0, 5.0]

  max_episode_steps: 300
  ...
```

---

### 3. Nav2 인터페이스 추상화 🔄

**설계 원칙**:
- Simulation과 Real Nav2를 동일한 인터페이스로 사용
- 학습은 Simulation, 배포는 Real Nav2
- 간단한 flag로 전환 가능

#### 인터페이스 설계

```python
# src/rl_dispatch/navigation/nav2_interface.py

from abc import ABC, abstractmethod
from typing import Tuple, Optional

class NavigationInterface(ABC):
    """Abstract interface for navigation systems."""

    @abstractmethod
    def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> Optional[list]:
        """Plan path from start to goal."""
        pass

    @abstractmethod
    def get_eta(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> float:
        """Get estimated time of arrival."""
        pass

    @abstractmethod
    def navigate_to_goal(
        self,
        goal: Tuple[float, float]
    ) -> Tuple[float, bool, bool]:
        """
        Navigate to goal.

        Returns:
            (nav_time, success, collision)
        """
        pass


class SimulatedNav2(NavigationInterface):
    """Simplified navigation for training."""

    def navigate_to_goal(self, goal):
        # Current implementation (_simulate_navigation)
        ...


class RealNav2(NavigationInterface):
    """Actual Nav2 integration for deployment."""

    def __init__(self, ros_node):
        self.ros_node = ros_node
        ...

    def navigate_to_goal(self, goal):
        # Real Nav2 action client
        ...
```

#### 환경 설정에서 선택

```python
# PatrolEnv 초기화
env = PatrolEnv(
    env_config=config,
    nav_mode="simulated"  # or "real_nav2"
)
```

---

### 3. 순찰 루트 전략 확장 🔄

**현재 구현된 전략 (6개)**:
1. ✅ Keep-Order (baseline)
2. ✅ Nearest-First (greedy)
3. ✅ Most-Overdue-First (gap-based)
4. ✅ Overdue-ETA-Balance (hybrid)
5. ✅ Risk-Weighted (priority-based)
6. ✅ Balanced-Coverage (variance minimization)

**추가 필요 전략 (4개)** - heuristic_method.md 권장:
7. ⏸️ **Shortest-ETA First** (Nav2 ETA 기반)
8. ⏸️ **Overdue-Threshold First** (gap > threshold)
9. ⏸️ **Minimal-Deviation Insert** (현재 루트에 삽입)
10. ⏸️ **Windowed Replan** (앞 H개만 재정렬)

#### 추가 전략 구현 예시

```python
# src/rl_dispatch/planning/candidate_generator.py

class ShortestETAFirstGenerator(CandidateGenerator):
    """
    Sort patrol points by Nav2 ETA (shortest first).

    Uses actual Nav2 path planning ETA instead of Euclidean distance.
    """

    def generate(self, robot, patrol_points, current_time):
        # Get ETA from Nav2 for each point
        etas = [
            nav2.get_eta(robot.position, pt.position)
            for pt in patrol_points
        ]
        # Sort by ETA
        sorted_indices = np.argsort(etas)
        return Candidate(
            patrol_order=tuple(sorted_indices),
            strategy_name="shortest_eta_first"
        )
```

---

### 4. 맵별 초기 순찰 루트 ⏸️

**개념**: 각 맵의 구조적 특성에 맞는 초기 순찰 순서 정의

#### 예시: map_office_building.yaml

```yaml
env:
  map_width: 90.0
  map_height: 70.0

  patrol_points:
    - [15.0, 15.0]   # P0: 정문
    - [30.0, 15.0]   # P1: 로비
    - [45.0, 15.0]   # P2: 안내데스크
    ...

  # 초기 순찰 루트 (맵 구조 기반)
  initial_patrol_route:
    - description: "외곽 순환 루트"
      route: [0, 1, 2, 3, 4, 5, 11, 10, 9, 8, 7, 6, 12, 13, 0]  # 순환
    - description: "중요구역 우선 루트"
      route: [0, 1, 12, 13, 11, 6, 7, 8, 9, 10, 5, 4, 3, 2, 0]
    - description: "최단거리 루트"
      route: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0]

  # 기본 사용 루트
  default_route: 0  # 외곽 순환
```

---

## 📈 통합 테스트 계획

### Phase 1: 이벤트 시스템 & 충전 스테이션 검증 ✅
```bash
python test_industrial_events.py
```

**검증 항목**:
- ✅ 34개 산업안전 이벤트 타입 정의
- ✅ 위험도별 이벤트 분포 (1-9)
- ✅ 위험도 역가중 샘플링 (고위험 = 낮은 빈도)
- ✅ Extended Event dataclass (backward compatible)
- ✅ 6개 맵 충전 스테이션 위치 검증
- ✅ PatrolEnv 통합 테스트

### Phase 2: Nav2 시뮬레이션 검증 🔄
```bash
python test_nav2_simulation.py
```

### Phase 3: 확장 전략 검증 ⏸️
```bash
python test_extended_heuristics.py
```

### Phase 4: 통합 학습 테스트 ⏸️
```bash
python scripts/train_multi_map.py \
    --total-timesteps 100000 \
    --use-industrial-events \
    --nav-mode simulated
```

---

## 🎯 마일스톤

| 마일스톤 | 완료 기준 | 상태 | 예상일 |
|----------|-----------|------|--------|
| M1: 이벤트 시스템 | 34개 이벤트 타입 + Event 확장 | ✅ 완료 | 2025-12-29 |
| M2: Nav2 추상화 | 인터페이스 설계 + 시뮬 구현 | 🔄 진행중 | 2025-12-30 |
| M3: 전략 확장 | 10개 전략 구현 | ⏸️ 대기 | 2025-12-31 |
| M4: 맵별 루트 | 6개 맵 초기 루트 설정 | ⏸️ 대기 | 2026-01-02 |
| M5: 통합 테스트 | 모든 기능 검증 | ⏸️ 대기 | 2026-01-05 |

---

## 📝 다음 단계

### 완료 (2025-12-29)
1. ✅ 산업안전 이벤트 타입 정의 (34개 이벤트)
2. ✅ Event dataclass 확장 (risk_level, event_name)
3. ✅ 이벤트 생성 로직 업데이트 (PatrolEnv 통합)
4. ✅ 충전 스테이션 위치 설정 (6개 맵)
5. ✅ 통합 테스트 스크립트 작성

### 다음 우선순위
1. ⏸️ Nav2 인터페이스 추상화 설계
2. ⏸️ 추가 휴리스틱 전략 구현 (4개)
3. ⏸️ 맵별 초기 순찰 루트 설정

### 중기 (다음 주)
7. ⏸️ Nav2 실제 통합 (ROS2)
8. ⏸️ 통합 테스트 및 검증
9. ⏸️ 성능 비교 (기존 vs 개선)

---

**작성자**: 박용준 (YJP)
**버전**: 1.0.0
**상태**: 진행 중 (Phase 1 완료)
