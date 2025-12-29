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
        # Calculate each component (Reviewer 박용준: Pass proximity_resolution)
        r_event = self._calculate_event_reward(
            event=event,
            current_time=current_time,
            event_resolved=event_resolved,
            proximity_resolution=proximity_resolution,
        )

        r_patrol = self._calculate_patrol_reward(
            patrol_points=patrol_points,
            current_time=current_time,
            patrol_point_visited=patrol_point_visited,
        )

        r_safety = self._calculate_safety_reward(
            collision=collision,
            nav_failure=nav_failure,
        )

        r_efficiency = self._calculate_efficiency_reward(
            distance_traveled=distance_traveled,
        )

        # Create reward components structure
        rewards = RewardComponents(
            event=r_event,
            patrol=r_patrol,
            safety=r_safety,
            efficiency=r_efficiency,
        )

        # Compute weighted total
        rewards.compute_total(self.config)

        return rewards

    def _calculate_event_reward(
        self,
        event: Optional[Event],
        current_time: float,
        event_resolved: bool,
        proximity_resolution: bool = False,  # Reviewer 박용준
    ) -> float:
        """
        Calculate event response reward (R^evt).

        Reward structure:
        - Large bonus for successfully resolving event (full credit for direct dispatch)
        - 50% bonus for low-risk event resolved via proximity (Reviewer 박용준)
        - Continuous penalty for delay (encourages quick response)
        - Large penalty if event exceeds max delay (failure)

        Args:
            event: Active event (None if no event)
            current_time: Current time
            event_resolved: Whether event was resolved this step
            proximity_resolution: Whether resolved via proximity (partial credit) (Reviewer 박용준)

        Returns:
            Event reward component
        """
        if event is None:
            # No event exists - neutral reward
            return 0.0

        if event_resolved:
            # Successfully resolved event
            # Bonus decreases slightly with delay to encourage speed
            delay = current_time - event.detection_time
            delay_factor = max(0.0, 1.0 - delay / self.config.event_max_delay)
            base_bonus = self.config.event_response_bonus * (0.5 + 0.5 * delay_factor)

            # Reviewer 박용준: Apply 50% multiplier for proximity resolution
            if proximity_resolution:
                return base_bonus * 0.5  # Partial credit for low-risk event
            else:
                return base_bonus  # Full credit for direct dispatch

        # Event exists but not resolved - apply delay penalty
        delay = current_time - event.detection_time

        if delay > self.config.event_max_delay:
            # Event failed due to excessive delay - large penalty
            return -self.config.event_response_bonus

        # Continuous delay penalty (linear with time)
        # Weighted by urgency (higher urgency = larger penalty)
        penalty = -self.config.event_delay_penalty_rate * delay * event.urgency
        return penalty

    def _calculate_patrol_reward(
        self,
        patrol_points: Tuple[PatrolPoint, ...],
        current_time: float,
        patrol_point_visited: Optional[int],
    ) -> float:
        """
        Calculate patrol coverage reward (R^pat).

        Reward structure:
        - Small bonus for visiting a patrol point
        - Continuous penalty for coverage gaps (sum of all overdue times)
        - Penalty weighted by patrol point priority

        This component is CRITICAL for unified learning. Without it, the policy
        could learn to dispatch frequently without understanding the cost to coverage.

        Args:
            patrol_points: All patrol points
            current_time: Current time
            patrol_point_visited: Index of point visited (None if none)

        Returns:
            Patrol reward component
        """
        reward = 0.0

        # Bonus for visiting a patrol point
        if patrol_point_visited is not None:
            reward += self.config.patrol_visit_bonus

        # Penalty for coverage gaps (cumulative overdue time)
        total_gap_penalty = 0.0
        for point in patrol_points:
            time_since_visit = current_time - point.last_visit_time

            # Apply penalty if point is overdue
            # Priority weight: higher priority points incur larger penalties
            gap_penalty = (
                self.config.patrol_gap_penalty_rate *
                time_since_visit *
                point.priority
            )
            total_gap_penalty += gap_penalty

        reward -= total_gap_penalty

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
