"""
PatrolEnv: Gymnasium environment for patrol robot dispatch and rescheduling.

This environment implements a Semi-Markov Decision Process (SMDP) where:
- State: Robot state, patrol points, events, LiDAR, candidates
- Actions: Composite (dispatch_mode, replan_strategy)
- Rewards: Multi-component (event, patrol, safety, efficiency)
- Time: Variable-duration steps (SMDP, not fixed-timestep MDP)

The environment models a patrol robot that must balance:
1. Responding to CCTV-detected events requiring investigation
2. Maintaining regular patrol coverage of designated waypoints

Key Features:
- SMDP semantics: Each step() corresponds to reaching a Nav2 goal (variable time)
- Action masking: Invalid actions are masked based on state
- Dense rewards: Non-zero rewards at most steps for learning efficiency
- Realistic simulation: Nav2-style navigation with failure modes
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import replace

from rl_dispatch.core.types import (
    State,
    RobotState,
    PatrolPoint,
    Event,
    Action,
    ActionMode,
    Candidate,
    Observation,
    RewardComponents,
    EpisodeMetrics,
)
from rl_dispatch.core.config import EnvConfig, RewardConfig
from rl_dispatch.planning import CandidateFactory
from rl_dispatch.navigation import NavigationInterface, SimulatedNav2
from rl_dispatch.navigation.pathfinding import create_occupancy_grid_from_walls  # Reviewer 박용준
from rl_dispatch.rewards import RewardCalculator
from rl_dispatch.utils import ObservationProcessor


class PatrolEnv(gym.Env):
    """
    Gymnasium environment for patrol robot dispatch and rescheduling.

    This environment models the complete patrol dispatch problem including:
    - Dynamic event generation (simulating CCTV detections)
    - Nav2-style navigation with variable execution times
    - Multi-objective reward function
    - Action masking for feasibility constraints

    Observation Space:
        Box(77,) - Normalized 77D observation vector:
        - Goal relative position (2D)
        - Robot heading sin/cos (2D)
        - Velocity/angular velocity (2D)
        - Battery level (1D)
        - LiDAR ranges (64D)
        - Event features (4D)
        - Patrol features (2D)

    Action Space:
        MultiDiscrete([2, K]) where K is number of candidate strategies:
        - action[0]: Mode (0=patrol, 1=dispatch)
        - action[1]: Replan strategy index (0 to K-1)

    Rewards:
        Float scalar - weighted sum of:
        - Event response reward
        - Patrol coverage reward
        - Safety reward
        - Efficiency reward

    Episode Termination:
        - Max steps reached
        - Max time reached
        - Battery depleted
        - Terminal collision

    Example:
        >>> env_config = EnvConfig()
        >>> reward_config = RewardConfig()
        >>> env = PatrolEnv(env_config, reward_config)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        """
        Initialize PatrolEnv.

        Args:
            env_config: Environment configuration (uses defaults if None)
            reward_config: Reward configuration (uses defaults if None)
        """
        super().__init__()

        # Configuration
        self.env_config = env_config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()

        # Initialize subsystems
        self.candidate_factory = CandidateFactory()
        self.reward_calculator = RewardCalculator(self.reward_config)
        self.obs_processor = ObservationProcessor(
            use_normalization=False,  # Normalization handled externally during training
            max_goal_distance=max(self.env_config.map_width, self.env_config.map_height),
            max_velocity=self.env_config.robot_max_velocity,
            max_angular_velocity=self.env_config.robot_max_angular_velocity,
            max_lidar_range=self.env_config.lidar_max_range,
        )

        # Reviewer 박용준: Build occupancy grid from walls for realistic navigation
        if self.env_config.walls:
            self.occupancy_grid = create_occupancy_grid_from_walls(
                width=self.env_config.map_width,
                height=self.env_config.map_height,
                walls=self.env_config.walls,
                resolution=self.env_config.grid_resolution,
            )
        else:
            # No walls defined - create empty grid with boundary only
            self.occupancy_grid = None

        # Initialize navigation interface (SimulatedNav2 for training)
        # Will be initialized with proper random state in reset()
        self.nav_interface: Optional[NavigationInterface] = None

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([
            2,  # Mode: 0=patrol, 1=dispatch
            self.env_config.num_candidates,  # Replan strategy index
        ])

        # Reviewer 박용준: Phase 4 - Observation space 77D → 88D
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(88,),
            dtype=np.float32,
        )

        # Episode state (initialized in reset())
        self.current_state: Optional[State] = None
        self.episode_step: int = 0
        self.episode_metrics: EpisodeMetrics = EpisodeMetrics()
        self.current_patrol_route: list = []
        self.pending_event: Optional[Event] = None
        self.event_counter: int = 0
        self.last_position: Tuple[float, float] = (0.0, 0.0)

        # Random number generator (seeded in reset())
        self.np_random: np.random.Generator = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional reset parameters (unused)

        Returns:
            observation: Initial 88D observation (Phase 4 enhanced)
            info: Dictionary with episode info
        """
        # Seed RNG
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Initialize navigation interface with seeded random state (Reviewer 박용준: Added occupancy grid)
        self.nav_interface = SimulatedNav2(
            occupancy_grid=self.occupancy_grid,
            grid_resolution=self.env_config.grid_resolution,
            max_velocity=self.env_config.robot_max_velocity,
            nav_failure_rate=0.05,
            collision_rate=0.01,
            np_random=np.random.RandomState(seed),
        )

        # Reviewer 박용준: Connect nav_interface to candidate factory for A* pathfinding
        self.candidate_factory.set_nav_interface(self.nav_interface)

        # Initialize robot at random patrol point
        initial_point_idx = self.np_random.integers(0, self.env_config.num_patrol_points)
        initial_point = self.env_config.patrol_points[initial_point_idx]

        robot = RobotState(
            x=initial_point[0],
            y=initial_point[1],
            heading=self.np_random.uniform(0, 2 * np.pi),
            velocity=0.0,
            angular_velocity=0.0,
            battery_level=1.0,
            current_goal_idx=0,  # Start going to first patrol point
        )

        # Initialize patrol points with visit times
        patrol_points = tuple(
            PatrolPoint(
                x=pos[0],
                y=pos[1],
                last_visit_time=0.0 if i == initial_point_idx else -100.0,  # Start overdue
                priority=self.env_config.patrol_point_priorities[i],
                point_id=i,
            )
            for i, pos in enumerate(self.env_config.patrol_points)
        )

        # Initialize patrol route (randomized for better exploration during training)
        self.current_patrol_route = list(range(self.env_config.num_patrol_points))
        self.np_random.shuffle(self.current_patrol_route)

        # Generate initial candidates
        candidates = self.candidate_factory.generate_all(
            robot, patrol_points, current_time=0.0
        )

        # Initialize LiDAR (all clear initially)
        lidar_ranges = np.full(
            self.env_config.lidar_num_channels,
            self.env_config.lidar_max_range,
            dtype=np.float32,
        )

        # Create initial state
        self.current_state = State(
            robot=robot,
            patrol_points=patrol_points,
            current_event=None,
            current_time=0.0,
            candidates=candidates,
            lidar_ranges=lidar_ranges,
        )

        # Reset episode tracking
        self.episode_step = 0
        self.episode_metrics = EpisodeMetrics()
        self.pending_event = None
        self.event_counter = 0
        self.last_position = (robot.x, robot.y)

        # Generate observation
        obs = self.obs_processor.process(self.current_state, update_stats=False)

        # Compute action mask (Reviewer 박용준)
        action_mask = self._compute_action_mask()

        info = {
            "episode_step": 0,
            "current_time": 0.0,
            "has_event": False,
            "action_mask": action_mask,  # Reviewer 박용준: For action masking
        }

        return obs.vector, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one SMDP step (navigate to goal, then decide next action).

        This is a Semi-Markov step, meaning it takes variable time depending on
        the navigation goal distance and complexity. The step sequence is:

        1. Parse action (mode, replan_idx)
        2. Apply action masking (check feasibility)
        3. Execute navigation to goal (variable time)
        4. Update robot state and patrol visits
        5. Handle event resolution
        6. Generate/update events
        7. Calculate reward
        8. Check termination conditions
        9. Generate next candidates and observation

        Args:
            action: [mode, replan_idx] - dispatch decision and replan strategy

        Returns:
            observation: Next 77D observation
            reward: Scalar reward for this transition
            terminated: Whether episode ended (goal reached or failure)
            truncated: Whether episode was cut off (time/step limit)
            info: Dictionary with transition details
        """
        assert self.current_state is not None, "Must call reset() before step()"

        # Parse action
        mode = ActionMode(int(action[0]))
        replan_idx = int(action[1])
        # Reviewer 박용준: Clamp replan_idx to valid range to prevent IndexError
        replan_idx = max(0, min(replan_idx, self.env_config.num_candidates - 1))
        action_obj = Action(mode=mode, replan_idx=replan_idx)

        # Apply action masking (silently adjust invalid actions)
        if not self._is_action_valid(action_obj):
            # Silently adjust to PATROL mode instead of printing warning
            # print(f"Warning: Invalid action {action_obj} - adjusting to PATROL mode")
            # Reviewer 박용준: Also ensure replan_idx is valid
            replan_idx = max(0, min(replan_idx, self.env_config.num_candidates - 1))
            action_obj = Action(mode=ActionMode.PATROL, replan_idx=replan_idx)

        # Reviewer 박용준: Check if forced charging is needed (battery < 15%)
        robot = self.current_state.robot
        forced_charging = robot.battery_level < 0.15
        is_at_charging_station = False

        if forced_charging:
            # Override goal to charging station
            goal_pos = self.env_config.charging_station_position
            goal_idx = -2  # Special marker for charging station

            # Check if already at charging station (within 1m)
            dist_to_station = np.sqrt(
                (robot.x - goal_pos[0])**2 + (robot.y - goal_pos[1])**2
            )
            is_at_charging_station = dist_to_station < 1.0
        else:
            # Determine navigation goal based on action
            goal_pos, goal_idx = self._determine_navigation_goal(action_obj)

        # Navigate to goal using Nav2 interface (SMDP: variable time)
        start_pos = (robot.x, robot.y)

        # Reviewer 박용준: If already at charging station, skip navigation and charge
        if is_at_charging_station:
            # Simulate charging (50s for full charge per debug_guide.md)
            charging_time = 50.0
            charge_rate = 1.0 / charging_time  # Full charge in 50s
            charge_amount = min(charge_rate * charging_time, 1.0 - robot.battery_level)
            new_battery = robot.battery_level + charge_amount

            # Use charging time as nav_time for SMDP
            nav_time = charging_time
            nav_success = True
            collision = False
        else:
            # Normal navigation
            nav_result = self.nav_interface.navigate_to_goal(start_pos, goal_pos)
            nav_time, nav_success, collision = nav_result.time, nav_result.success, nav_result.collision
            new_battery = None  # Will be computed by battery drain

        # Update robot state (Reviewer 박용준: pass charging battery if charging)
        new_robot = self._update_robot_state(
            goal_pos=goal_pos,
            goal_idx=goal_idx,
            nav_time=nav_time,
            nav_success=nav_success,
            override_battery=new_battery if is_at_charging_station else None,
        )

        # Update current time
        new_time = self.current_state.current_time + nav_time

        # Check for patrol point visit
        patrol_point_visited = None
        if mode == ActionMode.PATROL and nav_success:
            patrol_point_visited = goal_idx
            # Remove visited point from current route
            if (len(self.current_patrol_route) > 0 and
                self.current_patrol_route[0] == goal_idx):
                self.current_patrol_route.pop(0)

        # Update patrol points with new visit time
        new_patrol_points = self._update_patrol_points(
            patrol_point_visited=patrol_point_visited,
            current_time=new_time,
        )

        # Update patrol route if replan was selected
        if replan_idx > 0:  # 0 is keep-order
            candidate = self.current_state.candidates[replan_idx]
            self.current_patrol_route = list(candidate.patrol_order)

        # Handle event resolution (Reviewer 박용준: Added proximity-based resolution for low-risk events)
        event_resolved = False
        proximity_resolution = False  # Partial credit for low-risk events

        if self.pending_event:
            # Calculate distance to event
            dist_to_event = np.sqrt(
                (new_robot.x - self.pending_event.x)**2 +
                (new_robot.y - self.pending_event.y)**2
            )

            # Check for low-risk event proximity resolution (risk_level ≤ 3, within 5m)
            is_low_risk = hasattr(self.pending_event, 'risk_level') and self.pending_event.risk_level <= 3
            within_proximity = dist_to_event < 5.0

            if is_low_risk and within_proximity:
                # Low-risk event resolved via proximity (partial credit)
                event_resolved = True
                proximity_resolution = True
                self.episode_metrics.events_responded += 1
                self.episode_metrics.events_successful += 1
                delay = new_time - self.pending_event.detection_time
                n = self.episode_metrics.events_responded
                self.episode_metrics.avg_event_delay = (
                    (self.episode_metrics.avg_event_delay * (n - 1) + delay) / n
                )
                self.pending_event = None
            elif mode == ActionMode.DISPATCH and nav_success:
                # High-risk event or direct dispatch resolution (full credit)
                event_resolved = True
                self.episode_metrics.events_responded += 1
                self.episode_metrics.events_successful += 1
                delay = new_time - self.pending_event.detection_time
                n = self.episode_metrics.events_responded
                self.episode_metrics.avg_event_delay = (
                    (self.episode_metrics.avg_event_delay * (n - 1) + delay) / n
                )
                self.pending_event = None

        # Generate new events (Poisson process)
        new_event = self._maybe_generate_event(new_time, nav_time)
        if new_event and self.pending_event is None:
            self.pending_event = new_event
            self.episode_metrics.events_detected += 1

        # Update LiDAR (simulate obstacles)
        new_lidar = self._simulate_lidar(new_robot)

        # Generate new candidates for next decision
        new_candidates = self.candidate_factory.generate_all(
            new_robot, new_patrol_points, new_time
        )

        # Create new state
        new_state = State(
            robot=new_robot,
            patrol_points=new_patrol_points,
            current_event=self.pending_event,
            current_time=new_time,
            candidates=new_candidates,
            lidar_ranges=new_lidar,
        )

        # Calculate distance traveled
        distance_traveled = np.sqrt(
            (new_robot.x - self.last_position[0]) ** 2 +
            (new_robot.y - self.last_position[1]) ** 2
        )
        self.last_position = (new_robot.x, new_robot.y)

        # Calculate reward (Reviewer 박용준: Pass proximity_resolution for partial credit)
        rewards = self.reward_calculator.calculate(
            robot=new_robot,
            action=action_obj,
            event=self.pending_event,
            patrol_points=new_patrol_points,
            current_time=new_time,
            distance_traveled=distance_traveled,
            collision=collision,
            nav_failure=not nav_success,
            event_resolved=event_resolved,
            patrol_point_visited=patrol_point_visited,
            proximity_resolution=proximity_resolution,  # Reviewer 박용준
        )

        # Update episode metrics
        self.episode_step += 1
        self.episode_metrics.episode_length = self.episode_step
        self.episode_metrics.episode_return += rewards.total
        self.episode_metrics.total_distance += distance_traveled
        self.episode_metrics.final_battery = new_robot.battery_level
        if collision or not nav_success:
            self.episode_metrics.safety_violations += 1

        # Check termination conditions
        terminated = collision or new_robot.battery_level <= 0.0
        truncated = (
            self.episode_step >= self.env_config.max_episode_steps or
            new_time >= self.env_config.max_episode_time
        )

        # Calculate patrol coverage ratio
        if truncated or terminated:
            # Reviewer 박용준: Fix ZeroDivisionError when new_time is 0
            if new_time > 0:
                visit_interval = new_time / self.env_config.num_patrol_points
                expected_visits = self.env_config.num_patrol_points
                actual_visits = sum(
                    1 for p in new_patrol_points
                    if new_time - p.last_visit_time < visit_interval
                )
                self.episode_metrics.patrol_coverage_ratio = min(1.0, actual_visits / max(1, expected_visits))
            else:
                # Episode ended immediately (time = 0)
                self.episode_metrics.patrol_coverage_ratio = 0.0

        # Update state
        self.current_state = new_state

        # Generate observation
        obs = self.obs_processor.process(new_state, update_stats=False)

        # Compute action mask for next state (Reviewer 박용준)
        action_mask = self._compute_action_mask()

        # Build info dict
        info = {
            "episode_step": self.episode_step,
            "current_time": new_time,
            "has_event": self.pending_event is not None,
            "action_mode": mode.name,
            "replan_strategy": new_state.candidates[replan_idx].strategy_name,
            "reward_components": {
                "event": rewards.event,
                "patrol": rewards.patrol,
                "safety": rewards.safety,
                "efficiency": rewards.efficiency,
            },
            "nav_success": nav_success,
            "collision": collision,
            "event_resolved": event_resolved,
            "nav_time": nav_time,  # Reviewer 박용준: For SMDP variable discount
            "action_mask": action_mask,  # Reviewer 박용준: For action masking
        }

        # Add episode metrics if done
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_metrics.episode_return,
                "l": self.episode_metrics.episode_length,
                "events_detected": self.episode_metrics.events_detected,
                "events_responded": self.episode_metrics.events_responded,
                "events_successful": self.episode_metrics.events_successful,
                "avg_event_delay": self.episode_metrics.avg_event_delay,
                "patrol_coverage": self.episode_metrics.patrol_coverage_ratio,
                "safety_violations": self.episode_metrics.safety_violations,
                "total_distance": self.episode_metrics.total_distance,
            }

        return obs.vector, rewards.total, terminated, truncated, info

    def _compute_action_mask(self) -> np.ndarray:
        """
        Compute action mask for current state (Reviewer 박용준).

        Returns binary mask where 1=valid, 0=invalid for all (mode, replan) combinations.
        Shape: (2 * num_candidates,) flattened from (2, num_candidates)

        Invalid actions:
        - DISPATCH mode when no event present
        - DISPATCH mode when battery < 20%
        - Any action with invalid replan_idx

        Returns:
            action_mask: Boolean array of shape (2 * num_candidates,)
        """
        num_candidates = self.env_config.num_candidates
        mask = np.ones((2, num_candidates), dtype=np.float32)

        # Check if DISPATCH mode is valid
        can_dispatch = True
        if not self.current_state.has_event:
            can_dispatch = False  # No event to dispatch to
        if self.current_state.robot.battery_level < 0.2:
            can_dispatch = False  # Battery too low

        # Mask DISPATCH actions (mode=1) if invalid
        if not can_dispatch:
            mask[1, :] = 0.0

        # PATROL actions (mode=0) are always valid (with sufficient battery)
        # Note: Battery check for patrol can be added here if needed

        # Flatten to 1D for compatibility with policy output
        return mask.flatten()

    def _is_action_valid(self, action: Action) -> bool:
        """
        Check if action is valid given current state.

        Args:
            action: Action to validate

        Returns:
            True if action is valid
        """
        # Cannot dispatch if no event
        if action.mode == ActionMode.DISPATCH and not self.current_state.has_event:
            return False

        # Cannot dispatch if battery too low (reserve 20%)
        if action.mode == ActionMode.DISPATCH and self.current_state.robot.battery_level < 0.2:
            return False

        # Replan index must be valid (Reviewer 박용준: Added negative check)
        if action.replan_idx < 0 or action.replan_idx >= self.env_config.num_candidates:
            return False

        return True

    def _determine_navigation_goal(
        self,
        action: Action,
    ) -> Tuple[Tuple[float, float], int]:
        """
        Determine navigation goal position based on action.

        Args:
            action: Selected action

        Returns:
            (goal_position, goal_idx): Goal (x, y) and index
        """
        if action.mode == ActionMode.DISPATCH and self.pending_event:
            # Navigate to event location
            return (self.pending_event.x, self.pending_event.y), -1
        else:
            # Navigate to next patrol point in route
            if len(self.current_patrol_route) > 0:
                next_idx = self.current_patrol_route[0]
                point = self.current_state.patrol_points[next_idx]
                return (point.x, point.y), next_idx
            else:
                # Fallback: stay at current position
                return (self.current_state.robot.x, self.current_state.robot.y), -1


    def _update_robot_state(
        self,
        goal_pos: Tuple[float, float],
        goal_idx: int,
        nav_time: float,
        nav_success: bool,
        override_battery: Optional[float] = None,  # Reviewer 박용준: For charging
    ) -> RobotState:
        """
        Update robot state after navigation.

        Args:
            goal_pos: Target position
            goal_idx: Target patrol point index (-1 if event)
            nav_time: Time taken for navigation
            nav_success: Whether navigation succeeded

        Returns:
            Updated RobotState
        """
        robot = self.current_state.robot

        if nav_success:
            # Robot reached goal
            new_x, new_y = goal_pos
            # Update heading towards goal
            dx = goal_pos[0] - robot.x
            dy = goal_pos[1] - robot.y
            new_heading = np.arctan2(dy, dx)
        else:
            # Navigation failed - robot stays roughly in same position
            new_x = robot.x + self.np_random.normal(0, 0.1)
            new_y = robot.y + self.np_random.normal(0, 0.1)
            new_heading = robot.heading

        # Update battery (Reviewer 박용준: use override if charging, else drain)
        distance = np.sqrt((new_x - robot.x) ** 2 + (new_y - robot.y) ** 2)

        if override_battery is not None:
            # Charging at station - use provided value
            new_battery = override_battery
        else:
            # Normal operation - drain based on time and distance
            energy_used = (nav_time / 3600.0) * self.env_config.robot_battery_drain_rate
            battery_drain = energy_used / self.env_config.robot_battery_capacity
            new_battery = max(0.0, robot.battery_level - battery_drain)

        # Update velocity (moving at average speed during navigation)
        new_velocity = distance / nav_time if nav_time > 0 else 0.0

        return RobotState(
            x=new_x,
            y=new_y,
            heading=new_heading,
            velocity=min(new_velocity, self.env_config.robot_max_velocity),
            angular_velocity=0.0,
            battery_level=new_battery,
            current_goal_idx=goal_idx,
        )

    def _update_patrol_points(
        self,
        patrol_point_visited: Optional[int],
        current_time: float,
    ) -> Tuple[PatrolPoint, ...]:
        """
        Update patrol points with new visit time.

        Args:
            patrol_point_visited: Index of visited point (None if none)
            current_time: Current episode time

        Returns:
            Updated tuple of PatrolPoints
        """
        updated_points = []
        for i, point in enumerate(self.current_state.patrol_points):
            if i == patrol_point_visited:
                # Update visit time
                updated_point = replace(point, last_visit_time=current_time)
            else:
                updated_point = point
            updated_points.append(updated_point)

        return tuple(updated_points)

    def _maybe_generate_event(self, current_time: float, step_duration: float) -> Optional[Event]:
        """
        Possibly generate a new event (Poisson process).

        Uses industrial safety event types with risk levels (1-9) following
        Korean KOSHA/MOEL safety standards.

        Args:
            current_time: Current episode time
            step_duration: Duration of the SMDP step (nav_time)

        Returns:
            New Event if generated, None otherwise
        """
        # Poisson rate: lambda = rate / episode_time
        rate_per_second = self.env_config.event_generation_rate / self.env_config.max_episode_time
        prob_event_this_step = rate_per_second * step_duration  # Use actual step duration

        if self.np_random.random() < prob_event_this_step:
            # Import industrial safety event utilities
            from rl_dispatch.core.event_types import get_random_event_name, get_event_risk_level

            # Reviewer 박용준: Sample event location in free-space with A* validation
            event_location = self._sample_free_space_event_location()

            # If sampling failed (no reachable free space found), skip event generation
            if event_location is None:
                return None

            event_x, event_y = event_location

            # Select event type (risk-weighted: high-risk events are less frequent)
            event_name = get_random_event_name(self.np_random)
            risk_level = get_event_risk_level(event_name)

            # Generate detection confidence
            confidence = self.np_random.uniform(
                self.env_config.event_min_confidence,
                1.0
            )

            self.event_counter += 1

            # Import extended Event class
            from rl_dispatch.core.types_extended import Event as ExtendedEvent

            return ExtendedEvent(
                x=event_x,
                y=event_y,
                risk_level=risk_level,
                event_name=event_name,
                confidence=confidence,
                detection_time=current_time,
                event_id=self.event_counter,
                is_active=True,
            )

        return None

    def _sample_free_space_event_location(
        self, max_retries: int = 10
    ) -> Optional[Tuple[float, float]]:
        """
        Sample event location in free-space with A* reachability validation.

        Reviewer 박용준: Ensures events only spawn in accessible locations.

        Strategy:
        1. Sample random free-space cell from occupancy grid
        2. Validate with A* that robot can reach it
        3. Retry up to max_retries times
        4. Return None if all retries fail

        Args:
            max_retries: Maximum sampling attempts

        Returns:
            (x, y) tuple in meters, or None if sampling failed
        """
        # Fallback: if no occupancy grid, sample uniformly (old behavior)
        if self.occupancy_grid is None:
            return (
                self.np_random.uniform(0, self.env_config.map_width),
                self.np_random.uniform(0, self.env_config.map_height)
            )

        # Get free-space cells (where grid == 0)
        free_cells = np.argwhere(self.occupancy_grid == 0)

        if len(free_cells) == 0:
            # No free space (should never happen with valid maps)
            return None

        resolution = self.env_config.grid_resolution
        robot_pos = (self.current_state.robot.x, self.current_state.robot.y)

        # Retry sampling until we find a reachable location
        for attempt in range(max_retries):
            # Sample random free cell
            idx = self.np_random.integers(0, len(free_cells))
            grid_y, grid_x = free_cells[idx]

            # Convert to world coordinates (center of cell)
            event_x = (grid_x + 0.5) * resolution
            event_y = (grid_y + 0.5) * resolution

            # Validate with A* that event is reachable from robot
            if self.nav_interface and hasattr(self.nav_interface, 'pathfinder'):
                pathfinder = self.nav_interface.pathfinder
                if pathfinder and pathfinder.path_exists(robot_pos, (event_x, event_y)):
                    return (event_x, event_y)
            else:
                # If no pathfinder available, accept the free-space sample
                return (event_x, event_y)

        # All retries failed - no reachable location found
        return None

    def _simulate_lidar(self, robot: RobotState) -> np.ndarray:
        """
        Simulate LiDAR range measurements using ray-casting on occupancy grid.

        Reviewer 박용준: Replaced simplified simulation with realistic Bresenham ray-casting.
        Each ray is cast from robot position through the occupancy grid until hitting
        an obstacle or reaching max range.

        In real deployment, this would come from actual sensor data.

        Args:
            robot: Current robot state

        Returns:
            64D array of range measurements (in meters)
        """
        num_channels = self.env_config.lidar_num_channels
        max_range = self.env_config.lidar_max_range
        min_range = self.env_config.lidar_min_range

        # If no occupancy grid, return max range (fallback to old behavior)
        if self.occupancy_grid is None:
            ranges = np.full(num_channels, max_range, dtype=np.float32)
            ranges += self.np_random.normal(0, 0.5, size=ranges.shape)
            return np.clip(ranges, min_range, max_range)

        # Reviewer 박용준: Ray-cast on occupancy grid using Bresenham
        ranges = np.zeros(num_channels, dtype=np.float32)
        resolution = self.env_config.grid_resolution

        for i in range(num_channels):
            # Compute ray angle (evenly distributed 360°)
            angle = robot.heading + (2 * np.pi * i / num_channels)

            # Cast ray from robot position
            hit_distance = self._raycast(
                robot.x, robot.y, angle, max_range, resolution
            )

            ranges[i] = hit_distance

        # Add small Gaussian noise to simulate sensor noise
        ranges += self.np_random.normal(0, 0.02, size=ranges.shape)
        ranges = np.clip(ranges, min_range, max_range)

        return ranges

    def _raycast(
        self,
        x_start: float,
        y_start: float,
        angle: float,
        max_range: float,
        resolution: float,
    ) -> float:
        """
        Cast a single ray from (x_start, y_start) at given angle.

        Reviewer 박용준: Bresenham's line algorithm for efficient ray-casting.

        Args:
            x_start: Start x position in meters
            y_start: Start y position in meters
            angle: Ray angle in radians
            max_range: Maximum ray range in meters
            resolution: Grid resolution (meters per cell)

        Returns:
            Distance to first obstacle in meters (or max_range if no hit)
        """
        # Compute ray endpoint at max range
        x_end = x_start + max_range * np.cos(angle)
        y_end = y_start + max_range * np.sin(angle)

        # Convert to grid coordinates
        x0 = int(x_start / resolution)
        y0 = int(y_start / resolution)
        x1 = int(x_end / resolution)
        y1 = int(y_end / resolution)

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        grid_h, grid_w = self.occupancy_grid.shape
        x, y = x0, y0

        while True:
            # Check if out of bounds (hit map boundary)
            if x < 0 or x >= grid_w or y < 0 or y >= grid_h:
                return max_range

            # Check if hit obstacle
            if self.occupancy_grid[y, x] == 1:
                # Calculate distance from start to hit point
                hit_x = (x + 0.5) * resolution
                hit_y = (y + 0.5) * resolution
                distance = np.sqrt((hit_x - x_start)**2 + (hit_y - y_start)**2)
                return min(distance, max_range)

            # Check if reached endpoint
            if x == x1 and y == y1:
                return max_range

            # Bresenham step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def render(self):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Close the environment."""
        pass
