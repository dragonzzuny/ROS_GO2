# Reviewer: 박용준 - 핵심 수정사항 구현 가이드

**작성일**: 2025-12-30
**목적**: debug_guide.md 및 추가 요구사항에 따른 최소 침습 구현 가이드

---

## 완료된 작업

### 1. ✅ A* Pathfinding 모듈 구현
- **파일**: `src/rl_dispatch/navigation/pathfinding.py`
- **내용**:
  - `AStarPathfinder` 클래스: 8방향 이동 A* 구현
  - `create_occupancy_grid_from_walls()`: 벽 폴리곤 → occupancy grid 변환
  - `world_to_grid()`, `grid_to_world()`: 좌표 변환
  - `find_path()`, `get_distance()`, `path_exists()`: 경로 탐색 API

### 2. ✅ EnvConfig 확장
- **파일**: `src/rl_dispatch/core/config.py`
- **추가 필드**:
  ```python
  grid_resolution: float = 0.5  # 그리드 해상도
  walls: List[List[Tuple[float, float]]] = []  # 벽 폴리곤들
  num_pedestrians: int = 0  # 동적 장애물 (사람)
  num_vehicles: int = 0  # 동적 장애물 (차량/지게차)
  pedestrian_speed: float = 1.0
  vehicle_speed: float = 0.8
  dynamic_obstacle_radius: float = 0.5
  ```

### 3. ✅ 맵 YAML 스키마 업데이트
- **파일**: `configs/map_large_square.yaml`
- **추가 내용**:
  ```yaml
  env:
    grid_resolution: 0.5
    walls:
      - [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]]  # 장애물
      - [[20.0, 20.0], [25.0, 20.0], [25.0, 75.0], [20.0, 75.0]]  # L자 벽
  ```
  - **TODO**: 나머지 5개 맵에도 같은 스키마 적용

---

## 🔥 우선순위 높은 미완료 작업

### 1. SimulatedNav2에 A* 통합 (심각도: 최상)

**파일**: `src/rl_dispatch/navigation/nav2_interface.py`

**수정 내용**:
```python
# Reviewer: 박용준
from rl_dispatch.navigation.pathfinding import AStarPathfinder

class SimulatedNav2(NavigationInterface):
    def __init__(
        self,
        occupancy_grid: np.ndarray,  # 추가
        grid_resolution: float = 0.5,  # 추가
        max_velocity: float = 1.5,
        nav_failure_rate: float = 0.05,
        collision_rate: float = 0.01,
        np_random: Optional[np.random.RandomState] = None,
    ):
        self.pathfinder = AStarPathfinder(occupancy_grid, grid_resolution)
        self.max_velocity = max_velocity
        # ...

    def get_eta(self, start: Tuple[float, float], goal: Tuple[float, float]) -> float:
        """A* 기반 ETA 계산"""
        distance = self.pathfinder.get_distance(start, goal)
        if distance == np.inf:
            return np.inf  # 경로 없음
        avg_velocity = self.max_velocity * 0.7
        return distance / avg_velocity

    def navigate_to_goal(self, start, goal) -> NavigationResult:
        """A* 경로 기반 내비게이션"""
        result = self.pathfinder.find_path(start, goal)
        if result is None:
            return NavigationResult(time=0, success=False, collision=False)

        path, distance = result
        avg_velocity = self.max_velocity * 0.7
        nav_time = distance / avg_velocity * self.np_random.normal(1.0, 0.1)

        # 실패 확률
        success = self.np_random.random() > self.nav_failure_rate
        collision = self.np_random.random() < self.collision_rate if success else False

        return NavigationResult(
            time=nav_time,
            success=success and not collision,
            collision=collision,
            path=path if success else None
        )

    def plan_path(self, start, goal) -> Optional[List[Tuple[float, float]]]:
        """A* 경로 계획"""
        result = self.pathfinder.find_path(start, goal)
        return result[0] if result else None
```

**PatrolEnv 수정** (`src/rl_dispatch/env/patrol_env.py`):
```python
from rl_dispatch.navigation.pathfinding import create_occupancy_grid_from_walls

class PatrolEnv:
    def __init__(self, env_config, reward_config):
        # Occupancy grid 생성
        self.occupancy_grid = create_occupancy_grid_from_walls(
            env_config.map_width,
            env_config.map_height,
            env_config.walls,
            env_config.grid_resolution
        )

        # Nav2 interface에 grid 전달
        self.nav_interface = SimulatedNav2(
            occupancy_grid=self.occupancy_grid,
            grid_resolution=env_config.grid_resolution,
            max_velocity=env_config.robot_max_velocity,
            np_random=self.np_random
        )
```

---

### 2. SMDP 가변 할인율 적용 (심각도: 최상)

**파일**: `src/rl_dispatch/algorithms/buffer.py`

**수정 내용**:
```python
# Reviewer: 박용준 - SMDP 가변 할인율
class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, gamma, gae_lambda, device):
        # ...
        self.nav_times = np.zeros(buffer_size, dtype=np.float32)  # 추가

    def add(self, obs, action, log_prob, reward, value, done, nav_time):  # nav_time 추가
        # ...
        self.nav_times[self.pos] = nav_time
        self.pos += 1

    def compute_returns_and_advantages(self, last_value, last_done=False):
        """GAE 계산 with 가변 할인율"""
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            # ...
            # 가변 할인율 계산 (dt_base = 1.0)
            nav_time = self.nav_times[step]
            gamma_k = self.gamma ** nav_time

            # TD error with gamma_k
            delta = (
                self.rewards[step] +
                gamma_k * next_value * next_non_terminal -
                self.values[step]
            )

            # GAE with gamma_k
            last_gae_lam = (
                delta +
                gamma_k * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values
```

**학습 루프 수정** (`scripts/train_multi_map.py`):
```python
# Reviewer: 박용준
# step 후 nav_time 저장
next_obs, reward, terminated, truncated, info = env_wrapper.step(action)
nav_time = info.get("nav_time", 1.0)  # PatrolEnv에서 제공

agent.buffer.add(
    obs=obs,
    action=action,
    reward=reward,
    value=value,
    log_prob=log_prob,
    done=terminated,
    nav_time=nav_time  # 추가
)
```

**PatrolEnv 수정**:
```python
def step(self, action):
    # ...
    nav_result = self.nav_interface.navigate_to_goal(start, goal)
    nav_time = nav_result.time

    # ...
    info = {
        # ...
        "nav_time": nav_time  # 추가
    }
    return obs, reward, terminated, truncated, info
```

---

### 3. 완전한 행동 마스킹 (심각도: 상)

**파일**: `src/rl_dispatch/env/patrol_env.py`

**수정 내용**:
```python
# Reviewer: 박용준 - 행동 마스킹
def _compute_action_mask(self) -> np.ndarray:
    """현재 상태에서 유효한 행동 마스크"""
    # mode_mask: [patrol 가능, dispatch 가능]
    mode_mask = np.ones(2, dtype=np.float32)

    # 1. 이벤트 없으면 dispatch 불가
    if not self.current_state.has_event:
        mode_mask[1] = 0.0

    # 2. 배터리 부족하면 dispatch 불가
    if self.current_state.robot.battery_level < 0.2:
        mode_mask[1] = 0.0

    # 3. (선택) 후보별 마스크 (경로 없음, keep-out zone 등)
    # replan_mask = np.ones(10, dtype=np.float32)
    # for i, candidate in enumerate(self.current_state.candidates):
    #     if not self.pathfinder.path_exists(robot_pos, candidate.next_goal):
    #         replan_mask[i] = 0.0

    return mode_mask

def _get_obs_and_info(self) -> Tuple[np.ndarray, Dict]:
    obs = self.obs_processor.process(self.current_state, update_stats=False)
    info = {
        "action_mask": self._compute_action_mask()  # 추가
    }
    return obs.vector, info

def reset(self, ...):
    # ...
    obs, info = self._get_obs_and_info()
    return obs, info

def step(self, action):
    # ...
    obs, info = self._get_obs_and_info()
    return obs, reward, terminated, truncated, info
```

**PPOAgent 수정** (`src/rl_dispatch/algorithms/ppo.py`):
```python
# Reviewer: 박용준
def update(self, last_value, last_done):
    # ...
    for batch in self.buffer.get(...):
        obs, actions, old_log_probs, advantages, returns, old_values, masks = batch

        # Forward with mask
        _, new_log_probs, entropy, values = self.network.get_action_and_value(
            obs, action=actions, mode_mask=masks
        )
        # ...
```

**RolloutBuffer 수정**:
```python
def __init__(self, ...):
    # ...
    self.action_masks = np.zeros((buffer_size, 2), dtype=np.float32)  # 추가

def add(self, obs, action, log_prob, reward, value, done, nav_time, action_mask):
    # ...
    self.action_masks[self.pos] = action_mask  # 추가

def get(self, batch_size):
    # ...
    masks = torch.from_numpy(self.action_masks).to(self.device)
    yield (obs, actions, log_probs, advantages_t, returns, values, masks)  # masks 추가
```

---

### 4. 배터리/충전 로직 (심각도: 상)

**파일**: `src/rl_dispatch/env/patrol_env.py`

**수정 내용**:
```python
# Reviewer: 박용준 - 배터리 관리
class PatrolEnv:
    def step(self, action):
        # 배터리 체크
        if self.current_state.robot.battery_level < 0.15:
            # 강제로 충전소로 이동
            charging_pos = self.env_config.charging_station_position
            nav_result = self.nav_interface.navigate_to_goal(
                (self.current_state.robot.x, self.current_state.robot.y),
                charging_pos
            )

            # 충전소 도착 → 충전
            if nav_result.success:
                distance_to_charging = np.sqrt(
                    (self.current_state.robot.x - charging_pos[0])**2 +
                    (self.current_state.robot.y - charging_pos[1])**2
                )
                if distance_to_charging < 2.0:  # 2m 반경
                    # 충전 (50초에 100% 충전)
                    charging_time = 50.0
                    self.current_state.robot.battery_level = 1.0
                    self.current_time += charging_time

                    info["charging"] = True
                    info["charging_time"] = charging_time

        # 배터리 소모 (이동 중)
        battery_consumed = nav_result.time * self.env_config.robot_battery_drain_rate / 3600.0
        self.current_state.robot.battery_level = max(
            0.0,
            self.current_state.robot.battery_level - battery_consumed / self.env_config.robot_battery_capacity
        )

        # ...
```

---

### 5. 이벤트 샘플링을 Free-Space로 제한

**파일**: `src/rl_dispatch/env/patrol_env.py`

**수정 내용**:
```python
# Reviewer: 박용준 - Free-space 이벤트 생성
def _maybe_generate_event(self, current_time, step_duration):
    # ... (기존 Poisson 샘플링)

    # Free-space에서 위치 샘플링 (최대 10회 시도)
    from rl_dispatch.navigation.pathfinding import AStarPathfinder

    for attempt in range(10):
        event_x = self.np_random.uniform(0, self.env_config.map_width)
        event_y = self.np_random.uniform(0, self.env_config.map_height)

        # Occupancy grid로 free 체크
        grid_y, grid_x = self.pathfinder.world_to_grid(event_x, event_y)
        if self.occupancy_grid[grid_y, grid_x] == 0:  # Free
            # 로봇으로부터 경로 존재 확인
            robot_pos = (self.current_state.robot.x, self.current_state.robot.y)
            if self.pathfinder.path_exists(robot_pos, (event_x, event_y)):
                # 이벤트 생성
                event = ExtendedEvent(...)
                return event

    # 10회 시도 실패 → 이벤트 생성 안 함
    return None
```

---

### 6. LiDAR Ray-casting 구현

**파일**: `src/rl_dispatch/env/patrol_env.py`

**수정 내용**:
```python
# Reviewer: 박용준 - LiDAR ray-casting
def _simulate_lidar(self) -> np.ndarray:
    """Occupancy grid 기반 ray-casting"""
    robot_pos = (self.current_state.robot.x, self.current_state.robot.y)
    robot_heading = self.current_state.robot.heading

    lidar_ranges = np.full(self.lidar_num_channels, self.lidar_max_range, dtype=np.float32)

    for i in range(self.lidar_num_channels):
        angle = robot_heading + (2 * np.pi * i / self.lidar_num_channels)

        # Ray-casting (Bresenham)
        for r in np.arange(self.lidar_min_range, self.lidar_max_range, self.env_config.grid_resolution):
            x = robot_pos[0] + r * np.cos(angle)
            y = robot_pos[1] + r * np.sin(angle)

            grid_y, grid_x = self.pathfinder.world_to_grid(x, y)

            # 범위 체크
            if not (0 <= grid_y < self.occupancy_grid.shape[0] and
                    0 <= grid_x < self.occupancy_grid.shape[1]):
                lidar_ranges[i] = r
                break

            # 장애물 충돌
            if self.occupancy_grid[grid_y, grid_x] == 1:
                lidar_ranges[i] = r + self.np_random.normal(0, 0.02)  # 노이즈
                break

    return lidar_ranges
```

---

### 7. 저위험 이벤트 처리

**파일**: `src/rl_dispatch/env/patrol_env.py`

**수정 내용**:
```python
# Reviewer: 박용준 - 저위험 이벤트는 순찰 중 근접 해결
def step(self, action):
    # ...

    # 이벤트가 있고, risk_level이 낮으면 (1-3) 순찰 중 근접 확인
    if self.current_state.current_event and self.current_state.current_event.risk_level <= 3:
        event_pos = (self.current_state.current_event.x, self.current_state.current_event.y)
        robot_pos = (self.current_state.robot.x, self.current_state.robot.y)

        distance = np.sqrt(
            (event_pos[0] - robot_pos[0])**2 +
            (event_pos[1] - robot_pos[1])**2
        )

        # 반경 5m 내 진입 → 자동 해결
        if distance < 5.0:
            self.current_state.current_event = None
            reward += self.reward_config.event_response_bonus * 0.5  # 절반 보상
            info["low_risk_event_resolved"] = True

    # 고위험 이벤트 (risk >= 7)는 즉시 dispatch 필요
    # 중위험 (4-6)은 정책이 판단
```

---

### 8. 순찰 커버리지 패널티 추가

**파일**: `src/rl_dispatch/rewards/reward_calculator.py`

**수정 내용**:
```python
# Reviewer: 박용준 - 순찰 커버리지 패널티
def calculate_patrol_reward(self, state, next_state, action, config):
    # 기존 visit bonus
    patrol_reward = 0.0
    if self._reached_patrol_point(...):
        patrol_reward += config.patrol_visit_bonus

    # ✅ 추가: 공백 비용 (coverage gap penalty)
    gap_penalty = 0.0
    gap_threshold = 60.0  # 60초 이상 방문 안 한 포인트

    for point in next_state.patrol_points:
        time_gap = next_state.current_time - point.last_visit_time
        if time_gap > gap_threshold:
            gap_penalty += config.patrol_gap_penalty_rate * (time_gap - gap_threshold)

    patrol_reward -= gap_penalty
    return patrol_reward
```

---

### 9. 동적 장애물 시뮬레이션 (사람/장비)

**새 파일**: `src/rl_dispatch/env/dynamic_obstacles.py`

**내용**:
```python
# Reviewer: 박용준 - 동적 장애물
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class DynamicObstacle:
    x: float
    y: float
    vx: float  # 속도 (x방향)
    vy: float  # 속도 (y방향)
    radius: float  # 안전 반경
    obstacle_type: str  # "pedestrian" or "vehicle"
    waypoints: List[Tuple[float, float]] = None  # 목표 지점들

class DynamicObstacleManager:
    def __init__(self, num_pedestrians, num_vehicles, map_width, map_height, np_random):
        self.obstacles = []
        self.map_width = map_width
        self.map_height = map_height
        self.np_random = np_random

        # 사람 초기화 (랜덤 워크)
        for _ in range(num_pedestrians):
            x = np_random.uniform(5, map_width - 5)
            y = np_random.uniform(5, map_height - 5)
            self.obstacles.append(DynamicObstacle(
                x=x, y=y, vx=0, vy=0,
                radius=0.5, obstacle_type="pedestrian"
            ))

        # 차량 초기화 (waypoint 왕복)
        for _ in range(num_vehicles):
            x = np_random.uniform(10, map_width - 10)
            y = np_random.uniform(10, map_height - 10)
            waypoints = [(x, y), (map_width - x, map_height - y)]  # 왕복
            self.obstacles.append(DynamicObstacle(
                x=x, y=y, vx=0, vy=0,
                radius=1.0, obstacle_type="vehicle",
                waypoints=waypoints
            ))

    def update(self, dt: float, occupancy_grid: np.ndarray):
        """매 스텝 장애물 위치 업데이트"""
        for obs in self.obstacles:
            if obs.obstacle_type == "pedestrian":
                # 랜덤 워크
                if self.np_random.random() < 0.3:  # 30% 확률로 방향 전환
                    angle = self.np_random.uniform(0, 2 * np.pi)
                    speed = 1.0  # m/s
                    obs.vx = speed * np.cos(angle)
                    obs.vy = speed * np.sin(angle)

                obs.x += obs.vx * dt
                obs.y += obs.vy * dt

                # 맵 경계 반사
                if obs.x < 2 or obs.x > self.map_width - 2:
                    obs.vx = -obs.vx
                if obs.y < 2 or obs.y > self.map_height - 2:
                    obs.vy = -obs.vy

            elif obs.obstacle_type == "vehicle":
                # Waypoint 기반 이동 (여기서는 간단히 직선 이동)
                # TODO: A* 경로 따라 이동하도록 개선
                pass

    def get_dynamic_occupancy(self, grid_resolution: float) -> np.ndarray:
        """동적 장애물의 occupancy layer 생성"""
        grid_height = int(self.map_height / grid_resolution) + 1
        grid_width = int(self.map_width / grid_resolution) + 1
        dynamic_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        for obs in self.obstacles:
            grid_x = int(obs.x / grid_resolution)
            grid_y = int(obs.y / grid_resolution)

            # 반경 내 셀 점유
            r_cells = int(obs.radius / grid_resolution) + 1
            for dy in range(-r_cells, r_cells + 1):
                for dx in range(-r_cells, r_cells + 1):
                    gy, gx = grid_y + dy, grid_x + dx
                    if 0 <= gy < grid_height and 0 <= gx < grid_width:
                        dynamic_grid[gy, gx] = 1

        return dynamic_grid
```

**PatrolEnv 통합**:
```python
from rl_dispatch.env.dynamic_obstacles import DynamicObstacleManager

class PatrolEnv:
    def __init__(self, ...):
        # ...
        self.dynamic_manager = DynamicObstacleManager(
            self.env_config.num_pedestrians,
            self.env_config.num_vehicles,
            self.env_config.map_width,
            self.env_config.map_height,
            self.np_random
        )

    def step(self, action):
        # 1. 동적 장애물 업데이트
        self.dynamic_manager.update(dt=nav_result.time, occupancy_grid=self.occupancy_grid)

        # 2. 정적 + 동적 occupancy 병합
        dynamic_layer = self.dynamic_manager.get_dynamic_occupancy(self.env_config.grid_resolution)
        combined_grid = np.maximum(self.occupancy_grid, dynamic_layer)

        # 3. Nav2는 combined_grid 사용
        self.nav_interface.pathfinder.grid = combined_grid  # 업데이트

        # 4. 충돌 체크 (로봇과 동적 장애물)
        for obs in self.dynamic_manager.obstacles:
            distance = np.sqrt(
                (self.current_state.robot.x - obs.x)**2 +
                (self.current_state.robot.y - obs.y)**2
            )
            if distance < obs.radius + 0.5:  # 로봇 반경 0.5m
                reward += self.reward_config.collision_penalty
                info["dynamic_collision"] = True
```

---

### 10. 시각화 with 벽 오버레이

**새 파일**: `scripts/visualize_training_results.py`

**내용**:
```python
#!/usr/bin/env python3
# Reviewer: 박용준 - 학습 결과 시각화 (벽 오버레이)
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_dispatch.core.config import EnvConfig
from rl_dispatch.env import create_multi_map_env

def visualize_coverage_with_walls(log_dir: Path, update_num: int = 400):
    """
    Coverage heatmap 위에 벽/장애물 오버레이

    Args:
        log_dir: runs/multi_map_ppo/TIMESTAMP
        update_num: Update 번호 (예: 400)
    """
    coverage_dir = log_dir / "coverage" / f"update_{update_num}"

    if not coverage_dir.exists():
        print(f"Coverage 디렉토리가 없습니다: {coverage_dir}")
        return

    # 6개 맵 로드
    map_configs = [
        "configs/map_large_square.yaml",
        "configs/map_corridor.yaml",
        "configs/map_l_shaped.yaml",
        "configs/map_office_building.yaml",
        "configs/map_campus.yaml",
        "configs/map_warehouse.yaml",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, config_path in enumerate(map_configs):
        map_name = Path(config_path).stem
        heatmap_path = coverage_dir / f"{map_name}_heatmap.npy"

        if not heatmap_path.exists():
            print(f"Heatmap 없음: {heatmap_path}")
            continue

        # Heatmap 로드
        heatmap = np.load(heatmap_path)

        # Config 로드 (벽 정보)
        config = EnvConfig.load_yaml(config_path)

        ax = axes[idx]

        # Heatmap 표시
        im = ax.imshow(
            heatmap,
            cmap='hot',
            interpolation='bilinear',
            origin='lower',
            extent=[0, config.map_width, 0, config.map_height],
            alpha=0.7
        )

        # 벽 오버레이 (선/컨투어로 표시)
        for wall in config.walls:
            if len(wall) < 2:
                continue

            # 폴리곤 그리기
            wall_array = np.array(wall)
            polygon = patches.Polygon(
                wall_array,
                closed=True,
                edgecolor='cyan',
                facecolor='none',
                linewidth=2,
                linestyle='-'
            )
            ax.add_patch(polygon)

        # 순찰 포인트 표시
        for i, (px, py) in enumerate(config.patrol_points):
            ax.plot(px, py, 'go', markersize=8, markeredgecolor='white', markeredgewidth=1)
            ax.text(px, py, f'P{i}', color='white', fontsize=8, ha='center', va='center')

        # 충전 스테이션
        cx, cy = config.charging_station_position
        ax.plot(cx, cy, 'b^', markersize=12, markeredgecolor='white', markeredgewidth=1)
        ax.text(cx, cy + 3, 'Charging', color='white', fontsize=10, ha='center')

        ax.set_title(f"{map_name}\n(Update {update_num})", fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)

        # Colorbar
        plt.colorbar(im, ax=ax, label='Visit Count')

    plt.tight_layout()

    # 저장
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"coverage_update_{update_num}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True, help="runs/multi_map_ppo/TIMESTAMP")
    parser.add_argument("--update", type=int, default=400, help="Update number")
    args = parser.parse_args()

    visualize_coverage_with_walls(Path(args.log_dir), args.update)
```

**실행 명령어**:
```bash
python scripts/visualize_training_results.py \
    --log-dir runs/multi_map_ppo/20251230-120000 \
    --update 400
```

---

## ✅ 체크리스트

완료 후 다음을 확인하세요:

- [ ] **벽 관통 없음**: 로봇이 벽/장애물을 통과하지 않고 우회함
- [ ] **이벤트 도달 가능**: 생성된 이벤트가 모두 free-space이며 A* 경로가 존재함
- [ ] **배터리 충전 동작**: 배터리 low일 때 충전소로 이동하고 충전함
- [ ] **저위험 이벤트**: risk_level 낮은 이벤트는 즉시 dispatch하지 않음 (순찰 중 근접 해결)
- [ ] **행동 마스킹**: 이벤트 없음/배터리 부족 시 dispatch가 마스킹됨
- [ ] **SMDP 할인율**: nav_time에 따라 gamma^(nav_time)로 할인율 적용
- [ ] **동적 장애물**: 사람/차량이 움직이며, 로봇이 충돌하지 않고 우회/대기함
- [ ] **시각화**: Coverage heatmap에 벽이 선/컨투어로 오버레이됨

---

## 🚀 실행 명령어

### 1. 테스트
```bash
# 환경 테스트 (A*, 이벤트 샘플링, 배터리)
python test_industrial_events.py
python test_nav2_and_heuristics.py

# Quick training (수정 후 테스트)
python test_quick_training.py
```

### 2. 학습
```bash
# 100K steps (테스트용)
python scripts/train_multi_map.py --total-timesteps 100000 --seed 42

# Full training (5M steps)
python scripts/train_multi_map.py --total-timesteps 5000000 --seed 42 --log-interval 10
```

### 3. 시각화
```bash
# Coverage heatmap with walls
python scripts/visualize_training_results.py \
    --log-dir runs/multi_map_ppo/<TIMESTAMP> \
    --update 400

# TensorBoard
tensorboard --logdir runs
```

---

## 📌 참고사항

1. **나머지 5개 맵**: `configs/map_*.yaml` 파일들에도 같은 방식으로 `walls` 추가 필요
2. **동적 장애물 수**: 초기에는 `num_pedestrians=0, num_vehicles=0`으로 시작, 점진적으로 증가
3. **SMDP 할인율**: dt_base=1.0 (1초) 기준, 필요시 조정 가능
4. **테스트 우선**: 각 기능을 추가한 후 반드시 quick_test로 검증

---

**작성자**: Reviewer 박용준
**최종 수정**: 2025-12-30
