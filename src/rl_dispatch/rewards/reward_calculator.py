"""
Multi-component reward calculator for the RL Dispatch system.

This module implements the reward function that guides policy learning.
The reward function is carefully designed to balance multiple competing objectives:

- Event response: Reward quick, successful event handling
- Patrol coverage: Penalize gaps in routine patrol coverage
- Safety: Heavily penalize collisions and navigation failures
- Efficiency: Mildly penalize unnecessary travel distance

Each component is computed separately for interpretability and ablation studies.
"""

from dataclasses import replace
import numpy as np
from typing import Optional, Tuple

from rl_dispatch.core.types import (
    RewardComponents,
    Event,
    PatrolPoint,
    RobotState,
    Action,
)
from rl_dispatch.core.config import RewardConfig


class RewardCalculator:
    """
    Calculates multi-component rewards for state transitions.

    The reward function is:
        R_total = w_evt * R^evt + w_pat * R^pat + w_safe * R^safe + w_eff * R^eff

    Each component is calculated based on the state transition and action taken.
    Rewards are designed to be dense (non-zero at most timesteps) to facilitate learning.

    Attributes:
        config: RewardConfig with component weights and parameters

    Example:
        >>> config = RewardConfig()
        >>> calculator = RewardCalculator(config)
        >>> rewards = calculator.calculate(
        ...     robot=robot_state,
        ...     action=action,
        ...     event=event,
        ...     patrol_points=patrol_points,
        ...     current_time=150.0,
        ...     distance_traveled=5.0,
        ...     collision=False,
        ...     nav_failure=False,
        ...     event_resolved=False
        ... )
        >>> print(f"Total reward: {rewards.total:.2f}")
        >>> print(f"Event component: {rewards.event:.2f}")
    """

    def __init__(self, config: RewardConfig):
        """
        Initialize reward calculator.

        Args:
            config: RewardConfig with weights and parameters
        """
        self.config = config

        # Reviewer 박용준: Per-component normalizers for Phase 2
        self.event_normalizer = ComponentNormalizer("event")
        self.patrol_normalizer = ComponentNormalizer("patrol")
        self.efficiency_normalizer = ComponentNormalizer("efficiency")
        # Safety is NOT normalized (sparse, critical signal)

    def calculate(
        self,
        robot: RobotState,
        action: Action,
        event: Optional[Event],
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
        distance_traveled: float,
        collision: bool,
        nav_failure: bool,
        event_resolved: bool,
        patrol_point_visited: Optional[int] = None,
        proximity_resolution: bool = False,  # Reviewer 박용준: For low-risk event partial credit
    ) -> RewardComponents:
        """
        Calculate complete reward for a state transition.

        Args:
            robot: Current robot state
            action: Action taken
            event: Current active event (None if no event)
            patrol_points: All patrol points with visit times
            current_time: Current episode time
            distance_traveled: Distance traveled in this step (meters)
            collision: Whether collision occurred
            nav_failure: Whether Nav2 planning failed
            event_resolved: Whether event was successfully resolved
            patrol_point_visited: Index of patrol point visited (None if none)
            proximity_resolution: Whether low-risk event resolved via proximity (Reviewer 박용준)

        Returns:
            RewardComponents with all components and total
        """
        # Calculate each component (RAW values)
        r_event_raw = self._calculate_event_reward(
            event=event,
            current_time=current_time,
            event_resolved=event_resolved,
            proximity_resolution=proximity_resolution,
        )

        r_patrol_raw = self._calculate_patrol_reward(
            patrol_points=patrol_points,
            current_time=current_time,
            patrol_point_visited=patrol_point_visited,
        )

        r_safety = self._calculate_safety_reward(
            collision=collision,
            nav_failure=nav_failure,
        )

        r_efficiency_raw = self._calculate_efficiency_reward(
            distance_traveled=distance_traveled,
        )

        # Reviewer 박용준: Phase 2 - Per-component normalization
        # Normalize each component separately (except safety - critical sparse signal)
        r_event_norm = self.event_normalizer.normalize(r_event_raw)
        r_patrol_norm = self.patrol_normalizer.normalize(r_patrol_raw)
        r_efficiency_norm = self.efficiency_normalizer.normalize(r_efficiency_raw)

        # Apply weights AFTER normalization (scale is now unified)
        r_event = self.config.w_event * r_event_norm
        r_patrol = self.config.w_patrol * r_patrol_norm
        r_efficiency = self.config.w_efficiency * r_efficiency_norm
        # Safety keeps original weight (not normalized)
        r_safety_weighted = self.config.w_safety * r_safety

        # Create reward components structure (store normalized values)
        rewards = RewardComponents(
            event=r_event,
            patrol=r_patrol,
            safety=r_safety_weighted,
            efficiency=r_efficiency,
        )

        # Compute total (components already weighted)
        rewards.total = r_event + r_patrol + r_safety_weighted + r_efficiency

        return rewards

    def _calculate_event_reward(
        self,
        event: Optional[Event],
        current_time: float,
        event_resolved: bool,
        proximity_resolution: bool = False,
    ) -> float:
        """
        Calculate event response reward (R^evt) - Phase 2 SLA-based.

        Reviewer 박용준: Changed to realistic SLA-based rewards.

        Reward structure (Phase 2):
        - SLA_SUCCESS_VALUE for successful resolution (based on real-world cost savings)
        - Risk/urgency multiplier: higher risk events worth more
        - SLA_FAILURE_COST for missed events (based on real penalties)
        - Continuous SLA-based delay penalty
        - 50% credit for proximity resolution of low-risk events

        Old problem: Arbitrary values (event_response_bonus = 70.0) with no real-world basis
        New solution: Based on realistic SLA contracts and cost structures

        Args:
            event: Active event (None if no event)
            current_time: Current time
            event_resolved: Whether event was resolved this step
            proximity_resolution: Whether resolved via proximity (partial credit)

        Returns:
            Event reward component (will be normalized in calculate())
        """
        if event is None:
            # No event exists - neutral reward
            return 0.0

        # Get risk level from extended event if available
        risk_level = getattr(event, 'risk_level', 5)  # Default to mid-level risk
        urgency = getattr(event, 'urgency', 0.5)

        if event_resolved:
            # Successfully resolved event - SLA success value
            # Base value: $1000 per successful event (typical security SLA)
            base_success = self.config.sla_event_success_value

            # Risk multiplier: higher risk events are worth more
            # Risk 1-3: 0.5x, Risk 4-6: 1.0x, Risk 7-9: 2.0x
            risk_multiplier = 0.5 + (risk_level / 9.0) * 1.5

            # Delay factor: encourage quick response (SLA quality score)
            delay = current_time - event.detection_time
            sla_quality = max(0.0, 1.0 - delay / self.config.event_max_delay)

            # Total success reward
            success_reward = base_success * risk_multiplier * (0.5 + 0.5 * sla_quality)

            # Proximity resolution gets 50% credit (lower quality response)
            if proximity_resolution:
                return success_reward * 0.5
            else:
                return success_reward

        # Event exists but not resolved
        delay = current_time - event.detection_time

        if delay > self.config.event_max_delay:
            # SLA failure - large penalty (typical contract penalty: 2x success value)
            base_failure = -self.config.sla_event_failure_cost
            risk_multiplier = 0.5 + (risk_level / 9.0) * 1.5
            return base_failure * risk_multiplier

        # Continuous delay penalty based on SLA degradation
        # Linear degradation from 0 to failure cost
        delay_ratio = delay / self.config.event_max_delay
        sla_degradation_penalty = (
            -self.config.sla_delay_penalty_rate *
            delay_ratio *
            (risk_level / 9.0)  # Higher risk = faster SLA degradation
        )
        return sla_degradation_penalty

    def _calculate_patrol_reward(
        self,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
        patrol_point_visited: Optional[int],
    ) -> float:
        """
        Calculate patrol coverage reward (R^pat) - Phase 2 Delta Coverage.

        Reviewer 박용준: Changed from absolute gap penalty to delta coverage reward.

        Reward structure (Phase 2):
        - POSITIVE reward for closing coverage gap when visiting a point
        - Small baseline penalty for total accumulated gaps (normalized by num_points)
        - No longer punishes every step heavily - focuses on improvement

        Old problem: Campus map had -332/step penalty, completely dominated event reward
        New solution: Reward improvement, not absolute coverage state

        Args:
            patrol_points: All patrol points
            current_time: Current time
            patrol_point_visited: Index of point visited (None if none)

        Returns:
            Patrol reward component (will be normalized in calculate())
        """
        reward = 0.0
        num_points = len(patrol_points)

        # POSITIVE reward for closing coverage gap (Delta Coverage)
        if patrol_point_visited is not None:
            point = patrol_points[patrol_point_visited]
            gap_closed = current_time - point.last_visit_time

            # Larger gap closed = larger reward (incentivizes visiting overdue points)
            # Priority weight: more important points give more reward
            visit_reward = gap_closed * point.priority * self.config.patrol_visit_reward_rate
            reward += visit_reward

        # Small baseline penalty for accumulated gaps (prevents policy from ignoring patrol)
        # Normalized by num_points to make it map-independent
        total_gap = sum(
            (current_time - p.last_visit_time) * p.priority
            for p in patrol_points
        )
        normalized_gap = total_gap / max(num_points, 1)
        baseline_penalty = -self.config.patrol_baseline_penalty_rate * normalized_gap
        reward += baseline_penalty

        return reward

    def _calculate_safety_reward(
        self,
        collision: bool,
        nav_failure: bool,
    ) -> float:
        """
        Calculate safety reward (R^safe).

        Reward structure:
        - Large penalty for collision (terminal failure)
        - Moderate penalty for Nav2 planning failure (recoverable)
        - Zero reward for safe operation (safety is expected)

        Safety violations are heavily penalized to ensure the policy learns
        safe behaviors.

        Args:
            collision: Whether collision occurred
            nav_failure: Whether Nav2 planning failed

        Returns:
            Safety reward component
        """
        if collision:
            # Collision is terminal and heavily penalized
            return self.config.collision_penalty

        if nav_failure:
            # Nav2 failure is penalized but recoverable
            return self.config.nav_failure_penalty

        # No safety violations - neutral reward
        return 0.0

    def _calculate_efficiency_reward(
        self,
        distance_traveled: float,
    ) -> float:
        """
        Calculate efficiency reward (R^eff).

        Reward structure:
        - Small penalty proportional to distance traveled
        - Encourages shorter paths without dominating other objectives

        This component has the smallest weight to avoid over-optimization
        at the expense of event response or coverage.

        Args:
            distance_traveled: Distance in meters

        Returns:
            Efficiency reward component
        """
        # Linear penalty for distance
        penalty = -self.config.distance_penalty_rate * distance_traveled
        return penalty

    def calculate_cumulative_patrol_gap(
        self,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> float:
        """
        Calculate total cumulative coverage gap across all patrol points.

        This metric is useful for logging and evaluation (not used in reward).

        Args:
            patrol_points: All patrol points
            current_time: Current time

        Returns:
            Sum of (current_time - last_visit_time) for all points
        """
        total_gap = sum(
            (current_time - point.last_visit_time) * point.priority
            for point in patrol_points
        )
        return total_gap

    def calculate_max_patrol_gap(
        self,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
    ) -> float:
        """
        Calculate maximum coverage gap across all patrol points.

        This metric is useful for evaluation (not used in reward).

        Args:
            patrol_points: All patrol points
            current_time: Current time

        Returns:
            Maximum of (current_time - last_visit_time) across all points
        """
        if len(patrol_points) == 0:
            return 0.0

        max_gap = max(
            current_time - point.last_visit_time
            for point in patrol_points
        )
        return max_gap

    def evaluate_event_response_quality(
        self,
        event: Event,
        detection_time: float,
        response_time: float,
    ) -> Tuple[bool, float]:
        """
        Evaluate whether event response was successful and its quality.

        Args:
            event: The event
            detection_time: When event was detected
            response_time: When robot responded

        Returns:
            (success: bool, quality_score: float 0-1)
            success: True if within max_delay
            quality_score: 1.0 for instant response, 0.0 at max_delay
        """
        delay = response_time - detection_time

        if delay > self.config.event_max_delay:
            return False, 0.0

        # Quality degrades linearly with delay
        quality_score = max(0.0, 1.0 - delay / self.config.event_max_delay)
        return True, quality_score


class RewardNormalizer:
    """
    Running normalization for rewards to stabilize training.

    Maintains running statistics (mean, std) of rewards and normalizes
    them to approximately unit variance. This improves PPO training stability.

    Attributes:
        mean: Running mean estimate
        var: Running variance estimate
        count: Number of samples seen
        epsilon: Small constant for numerical stability

    Example:
        >>> normalizer = RewardNormalizer()
        >>> for reward in rewards:
        ...     normalized = normalizer.normalize(reward)
        ...     # Train on normalized reward
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize reward normalizer.

        Args:
            epsilon: Small constant for numerical stability
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon

    def normalize(self, reward: float, update_stats: bool = True) -> float:
        """
        Normalize reward using running statistics.

        Args:
            reward: Raw reward value
            update_stats: Whether to update running statistics

        Returns:
            Normalized reward (approximately mean=0, std=1)
        """
        if update_stats:
            self.update(reward)

        # Normalize: (x - mean) / sqrt(var + eps)
        normalized = (reward - self.mean) / np.sqrt(self.var + self.epsilon)
        return normalized

    def update(self, reward: float) -> None:
        """
        Update running statistics with new reward.

        Uses Welford's online algorithm for numerical stability.

        Args:
            reward: New reward value
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def reset(self) -> None:
        """Reset statistics to initial state."""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def get_stats(self) -> Tuple[float, float, int]:
        """
        Get current statistics.

        Returns:
            (mean, variance, count)
        """
        return self.mean, self.var, self.count


class ComponentNormalizer:
    """
    Per-component reward normalizer for Phase 2.

    Reviewer 박용준: Normalizes each reward component separately to ensure
    they have similar magnitudes, preventing one component from dominating.

    Uses Welford's online algorithm for numerical stability.

    Attributes:
        name: Component name (for logging)
        mean: Running mean
        M2: Running sum of squared differences (for variance)
        count: Number of samples
        epsilon: Small constant for numerical stability
    """

    def __init__(self, name: str, epsilon: float = 1e-8):
        """
        Initialize component normalizer.

        Args:
            name: Component name (e.g., "event", "patrol")
            epsilon: Small constant for numerical stability
        """
        self.name = name
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0
        self.epsilon = epsilon

    def normalize(self, value: float, update_stats: bool = True) -> float:
        """
        Normalize value using running statistics.

        Args:
            value: Raw component value
            update_stats: Whether to update running statistics

        Returns:
            Normalized value (approximately mean=0, std=1)
        """
        if update_stats:
            self._update(value)

        # Need at least 2 samples to compute std
        if self.count < 2:
            return value

        # Compute std from M2
        variance = self.M2 / (self.count - 1)
        std = max(np.sqrt(variance), self.epsilon)

        # Normalize: (x - mean) / std
        normalized = (value - self.mean) / std
        return normalized

    def _update(self, value: float) -> None:
        """
        Update running statistics using Welford's online algorithm.

        Args:
            value: New value to incorporate
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def get_stats(self) -> Tuple[float, float, float, int]:
        """
        Get current statistics.

        Returns:
            (mean, std, variance, count)
        """
        if self.count < 2:
            return self.mean, 0.0, 0.0, self.count

        variance = self.M2 / (self.count - 1)
        std = np.sqrt(variance)
        return self.mean, std, variance, self.count

    def reset(self) -> None:
        """Reset statistics to initial state."""
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0
