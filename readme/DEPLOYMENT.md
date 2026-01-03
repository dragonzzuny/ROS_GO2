# RL Dispatch - Deployment Guide

이 문서는 훈련된 정책을 시뮬레이션 및 실제 Unitree Go2 로봇에 배포하는 방법을 설명합니다.

---

## 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [설치](#설치)
3. [시뮬레이션 배포](#시뮬레이션-배포)
4. [실제 Go2 배포](#실제-go2-배포)
5. [배포 구성](#배포-구성)
6. [API 레퍼런스](#api-레퍼런스)
7. [문제 해결](#문제-해결)

---

## 시스템 요구사항

### 공통

- Python 3.10+
- PyTorch 2.0+
- numpy, yaml

### 시뮬레이션 전용

- 추가 요구사항 없음 (순수 Python)

### Gazebo 시뮬레이션

- ROS2 Humble
- Gazebo Fortress
- Nav2

### 실제 Go2 로봇

- ROS2 Humble
- Nav2 stack
- unitree_ros2 패키지
- go2_description 패키지
- 네트워크 연결 (Go2 기본 IP: 192.168.123.161)

---

## 설치

### 기본 설치

```bash
# 프로젝트 클론
git clone https://github.com/your-repo/rl_dispatch_mvp.git
cd rl_dispatch_mvp

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows

# 의존성 설치
pip install -e .
```

### ROS2 배포용 설치 (Go2/Gazebo)

```bash
# ROS2 workspace에서
cd ~/ros2_ws/src
git clone https://github.com/your-repo/rl_dispatch_mvp.git

# ROS2 의존성 설치
pip install -e rl_dispatch_mvp[ros2]

# 빌드
cd ~/ros2_ws
colcon build --packages-select rl_dispatch_mvp
source install/setup.bash
```

---

## 시뮬레이션 배포

### 빠른 시작

```python
from rl_dispatch.deployment import PolicyRunner

# 정책 로드 및 시뮬레이션 연결
runner = PolicyRunner(
    model_path="checkpoints/best_model.pth",
    mode="simulation"
)
runner.connect()

# 순찰 포인트 설정
runner.set_patrol_points([
    (5.0, 5.0),
    (10.0, 5.0),
    (10.0, 10.0),
    (5.0, 10.0),
])

# 에피소드 실행
metrics = runner.run_episode(max_steps=400)
print(f"Episode result: {metrics}")

# 연결 해제
runner.disconnect()
```

### 커맨드라인 실행

```bash
# 단일 에피소드
python -m rl_dispatch.deployment.inference \
    --model checkpoints/best_model.pth \
    --mode simulation \
    --episodes 10
```

---

## 실제 Go2 배포

### 사전 준비

1. **ROS2 환경 설정**
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/ros2_ws/install/setup.bash
   ```

2. **Go2 연결 확인**
   ```bash
   # Go2 IP 핑 테스트
   ping 192.168.123.161

   # ROS2 토픽 확인
   ros2 topic list
   ```

3. **Nav2 시작**
   ```bash
   ros2 launch nav2_bringup bringup_launch.py \
       use_sim_time:=False \
       map:=/path/to/your/map.yaml
   ```

### 정책 배포

```python
from rl_dispatch.deployment import PolicyRunner, RealRobotConfig

# Go2 설정
config = RealRobotConfig(
    robot_ip="192.168.123.161",
    safety_enabled=True,
    max_linear_velocity=1.5,
    max_angular_velocity=1.0,
)

# 정책 로드 및 Go2 연결
runner = PolicyRunner(
    model_path="checkpoints/best_model.pth",
    config=config,
    mode="real"
)

if not runner.connect():
    print("Go2 연결 실패!")
    exit(1)

# 순찰 시작
try:
    runner.run_continuous(max_episodes=100)
except KeyboardInterrupt:
    print("사용자에 의해 중지됨")
finally:
    runner.disconnect()
```

### ROS2 노드로 실행

```bash
# ROS2 런치 파일 사용
ros2 launch rl_dispatch patrol_launch.py \
    model_path:=checkpoints/best_model.pth \
    map_config:=configs/map_campus.yaml
```

---

## 배포 구성

### 시뮬레이션 설정 (SimulationConfig)

```yaml
# configs/deploy_simulation.yaml
mode: simulation
model_path: checkpoints/best_model.pth
device: cpu
deterministic: true

sim_time_step: 0.1
render: false
max_episode_steps: 400
event_generation_rate: 20.0
```

### 실제 로봇 설정 (RealRobotConfig)

```yaml
# configs/deploy_real.yaml
mode: real
model_path: checkpoints/best_model.pth
device: cpu
deterministic: true

# Go2 연결
robot_ip: "192.168.123.161"
robot_port: 8080

# Nav2 설정
nav2_namespace: ""
navigate_to_pose_action: /navigate_to_pose

# 안전 설정
safety_enabled: true
max_linear_velocity: 1.5
max_angular_velocity: 1.0
min_obstacle_distance: 0.5
emergency_stop_distance: 0.3

# 배터리 관리
low_battery_threshold: 0.20
critical_battery_threshold: 0.10
auto_return_on_low_battery: true
```

---

## API 레퍼런스

### RobotInterface

```python
class RobotInterface:
    """통합 로봇 인터페이스"""

    def connect() -> bool:
        """로봇 연결"""

    def disconnect() -> None:
        """연결 해제"""

    def get_state() -> RobotState:
        """현재 상태 조회 (위치, 배터리, 센서)"""

    def navigate_to(goal: Tuple[float, float]) -> NavigationFeedback:
        """목표 지점으로 이동"""

    def emergency_stop() -> bool:
        """비상 정지"""

    def get_eta(goal: Tuple[float, float]) -> float:
        """예상 도착 시간"""
```

### PolicyRunner

```python
class PolicyRunner:
    """정책 실행기"""

    def __init__(model_path, config, mode):
        """모델 로드 및 초기화"""

    def connect() -> bool:
        """로봇 연결"""

    def set_patrol_points(points: List[Tuple[float, float]]):
        """순찰 포인트 설정"""

    def run_episode(max_steps) -> Dict:
        """단일 에피소드 실행"""

    def run_continuous(max_episodes):
        """연속 순찰 실행"""
```

---

## 문제 해결

### 연결 문제

**증상**: `Failed to connect to robot`

**해결**:
1. Go2 전원 및 네트워크 확인
2. IP 주소 확인 (`ping 192.168.123.161`)
3. ROS2 도메인 ID 확인 (`export ROS_DOMAIN_ID=0`)

### Nav2 문제

**증상**: `Nav2 action server not available`

**해결**:
1. Nav2 실행 확인 (`ros2 node list | grep nav2`)
2. 맵 로드 확인 (`ros2 topic echo /map`)
3. TF 확인 (`ros2 run tf2_tools view_frames`)

### 배터리 문제

**증상**: `Critical battery - navigation disabled`

**해결**:
1. 충전소로 수동 이동
2. `auto_return_on_low_battery: true` 설정 확인
3. 배터리 임계값 조정

### 안전 정지

**증상**: `EMERGENCY STOP activated`

**해결**:
1. 장애물 제거
2. `reset_emergency_stop()` 호출
3. `safety_enabled: false`로 임시 비활성화 (테스트 시에만)

---

## 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                   PolicyRunner                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Policy (PPO)│  │ Obs Processor │  │ Candidates │ │
│  └──────┬──────┘  └───────┬──────┘  └─────┬──────┘ │
│         │                 │               │         │
│         └────────┬────────┴───────┬──────┘         │
│                  ▼                ▼                 │
│         ┌────────────────────────────────┐         │
│         │       RobotInterface           │         │
│         │  (Abstract - Sim2Real)         │         │
│         └───────────────┬────────────────┘         │
└─────────────────────────┼───────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Simulation   │ │   Gazebo     │ │   Go2 Real   │
│ (Pure Python)│ │   (ROS2)     │ │   (ROS2)     │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## 체크리스트

### 배포 전 확인

- [ ] 모델 체크포인트 존재 확인
- [ ] 맵 설정 파일 확인
- [ ] 순찰 포인트 좌표 확인
- [ ] 충전소 위치 확인

### Go2 배포 전 확인

- [ ] ROS2 환경 소싱
- [ ] Nav2 스택 실행
- [ ] Go2 네트워크 연결
- [ ] 맵 로드 확인
- [ ] 안전 구역 확보

### 운영 중 모니터링

- [ ] 배터리 레벨
- [ ] 네비게이션 성공률
- [ ] 이벤트 대응 시간
- [ ] 커버리지 비율

---

**마지막 업데이트**: 2026-01-03
