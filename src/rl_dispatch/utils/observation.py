"""
Observation processing and normalization utilities.

This module handles conversion from full State objects to normalized
Observation vectors suitable for neural network input.

Key responsibilities:
- Extract features from State
- Normalize to stable ranges
- Maintain running statistics for online normalization
- Handle edge cases (no event, no goal, etc.)
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple

from rl_dispatch.core.types import State, Observation, PatrolPoint, Event
from rl_dispatch.utils.math import compute_relative_vector, normalize_angle


class RunningMeanStd:
    """
    Tracks running mean and standard deviation using Welford's algorithm.

    Used for online normalization of observations during training.
    Maintains numerical stability even with many updates.

    Attributes:
        mean: Running mean estimate (vector)
        var: Running variance estimate (vector)
        count: Number of samples seen
        epsilon: Small constant for numerical stability

    Example:
        >>> normalizer = RunningMeanStd(shape=(77,))
        >>> obs = np.random.randn(77)
        >>> normalized = normalizer.normalize(obs, update=True)
    """

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        """
        Initialize running statistics.

        Args:
            shape: Shape of the data to track
            epsilon: Small constant for numerical stability
        """
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: npt.NDArray[np.float32]) -> None:
        """
        Update running statistics with new data.

        Uses Welford's online algorithm for numerical stability.

        Args:
            x: New observation (shape must match self.mean.shape)
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: npt.NDArray[np.float32],
        batch_var: npt.NDArray[np.float32],
        batch_count: int,
    ) -> None:
        """
        Update from batch statistics.

        Args:
            batch_mean: Mean of batch
            batch_var: Variance of batch
            batch_count: Number of samples in batch
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(
        self,
        x: npt.NDArray[np.float32],
        clip: float = 10.0,
    ) -> npt.NDArray[np.float32]:
        """
        Normalize data using running statistics.

        Args:
            x: Data to normalize
            clip: Maximum absolute value for clipping

        Returns:
            Normalized data
        """
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        if clip > 0:
            normalized = np.clip(normalized, -clip, clip)
        return normalized

    def reset(self) -> None:
        """Reset statistics to initial values."""
        self.mean = np.zeros_like(self.mean)
        self.var = np.ones_like(self.var)
        self.count = self.epsilon


class ObservationProcessor:
    """
    Processes State objects into normalized Observation vectors.

    Handles feature extraction, normalization, and edge cases.
    The output is a 88-dimensional vector suitable for neural network input.

    Observation Structure (88D) - Reviewer 박용준: Phase 4 Enhanced:
        - goal_relative_vec (2D): Normalized (dx, dy) to current goal
        - heading_sin_cos (2D): (sin(theta), cos(theta))
        - velocity_angular (2D): Normalized (v, omega)
        - battery (1D): Battery level [0, 1]
        - lidar_ranges (64D): Normalized LiDAR ranges
        - event_features (4D): [exists, urgency, confidence, elapsed_time_norm]
        - patrol_features (2D): [distance_to_next_norm, coverage_gap_ratio]
        - event_risk_level (1D): Risk level normalized [0, 1] (Phase 4)
        - patrol_crisis (3D): [max_gap_norm, critical_count_norm, crisis_score] (Phase 4)
        - candidate_feasibility (6D): Feasibility score per candidate [0, 1] (Phase 4)
        - urgency_risk_product (1D): urgency × risk combined signal (Phase 4)

    Attributes:
        use_normalization: Whether to apply running normalization
        normalizer: RunningMeanStd for observations
        max_goal_distance: Maximum expected distance to goal (for normalization)
        max_velocity: Maximum robot velocity (for normalization)
        max_angular_velocity: Maximum angular velocity (for normalization)
        max_lidar_range: Maximum LiDAR range (for normalization)
        max_event_delay: Maximum event delay before failure (for normalization)
        max_coverage_gap: Maximum expected coverage gap (for normalization)
        max_num_patrol_points: Maximum number of patrol points (for normalization)

    Example:
        >>> processor = ObservationProcessor(use_normalization=True)
        >>> obs = processor.process(state, update_stats=True)
        >>> assert obs.vector.shape[0] == 88
    """

    def __init__(
        self,
        use_normalization: bool = True,
        max_goal_distance: float = 50.0,
        max_velocity: float = 1.5,
        max_angular_velocity: float = 1.0,
        max_lidar_range: float = 10.0,
        max_event_delay: float = 120.0,
        max_coverage_gap: float = 300.0,
        max_num_patrol_points: int = 30,  # Reviewer 박용준: Phase 4
    ):
        """
        Initialize observation processor.

        Args:
            use_normalization: Whether to use running normalization
            max_goal_distance: Maximum expected goal distance (meters)
            max_velocity: Maximum robot velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
            max_lidar_range: Maximum LiDAR range (meters)
            max_event_delay: Maximum event delay (seconds)
            max_coverage_gap: Maximum coverage gap (seconds)
            max_num_patrol_points: Maximum number of patrol points (for crisis normalization)
        """
        self.use_normalization = use_normalization
        self.normalizer = RunningMeanStd(shape=(88,))  # Reviewer 박용준: Phase 4 - 77 → 88

        self.max_goal_distance = max_goal_distance
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.max_lidar_range = max_lidar_range
        self.max_event_delay = max_event_delay
        self.max_coverage_gap = max_coverage_gap
        self.max_num_patrol_points = max_num_patrol_points  # Reviewer 박용준: Phase 4

    def process(
        self,
        state: State,
        update_stats: bool = False,
    ) -> Observation:
        """
        Process State into normalized Observation vector.

        Args:
            state: Full environment state
            update_stats: Whether to update running statistics

        Returns:
            88-dimensional Observation (Phase 4 enhanced)
        """
        # Initialize observation vector (Reviewer 박용준: Phase 4 - 77 → 88)
        obs_vector = np.zeros(88, dtype=np.float32)

        # 1. Goal relative vector (2D) - indices 0:2
        goal_dx, goal_dy = self._extract_goal_relative(state)
        obs_vector[0] = goal_dx / self.max_goal_distance
        obs_vector[1] = goal_dy / self.max_goal_distance

        # 2. Heading sin/cos (2D) - indices 2:4
        obs_vector[2] = np.sin(state.robot.heading)
        obs_vector[3] = np.cos(state.robot.heading)

        # 3. Velocity and angular velocity (2D) - indices 4:6
        obs_vector[4] = state.robot.velocity / self.max_velocity
        obs_vector[5] = state.robot.angular_velocity / self.max_angular_velocity

        # 4. Battery (1D) - index 6
        obs_vector[6] = state.robot.battery_level

        # 5. LiDAR ranges (64D) - indices 7:71
        lidar_normalized = np.clip(
            state.lidar_ranges / self.max_lidar_range,
            0.0,
            1.0
        )
        obs_vector[7:71] = lidar_normalized

        # 6. Event features (4D) - indices 71:75
        event_features = self._extract_event_features(state)
        obs_vector[71:75] = event_features

        # 7. Patrol features (2D) - indices 75:77
        patrol_features = self._extract_patrol_features(state)
        obs_vector[75:77] = patrol_features

        # 8. Event risk level (1D) - index 77 (Reviewer 박용준: Phase 4)
        obs_vector[77] = self._extract_event_risk(state)

        # 9. Patrol crisis indicators (3D) - indices 78:81 (Reviewer 박용준: Phase 4)
        patrol_crisis = self._extract_patrol_crisis(state)
        obs_vector[78:81] = patrol_crisis

        # 10. Candidate feasibility (6D) - indices 81:87 (Reviewer 박용준: Phase 4)
        candidate_feasibility = self._extract_candidate_feasibility(state)
        obs_vector[81:87] = candidate_feasibility

        # 11. Urgency-risk combined signal (1D) - index 87 (Reviewer 박용준: Phase 4)
        obs_vector[87] = self._extract_urgency_risk_combined(state)

        # Apply running normalization if enabled
        if self.use_normalization:
            if update_stats:
                self.normalizer.update(obs_vector.reshape(1, -1))
            obs_vector = self.normalizer.normalize(obs_vector, clip=10.0)

        return Observation(vector=obs_vector)

    def _extract_goal_relative(self, state: State) -> Tuple[float, float]:
        """
        Extract relative vector to current goal.

        If robot has a current goal, compute relative position in robot frame.
        Otherwise, return (0, 0).

        Args:
            state: Environment state

        Returns:
            (dx_local, dy_local) relative to current goal
        """
        if state.robot.current_goal_idx < 0:
            # No current goal
            return (0.0, 0.0)

        if state.robot.current_goal_idx >= len(state.patrol_points):
            # Invalid goal index
            return (0.0, 0.0)

        goal = state.patrol_points[state.robot.current_goal_idx]
        dx_local, dy_local = compute_relative_vector(
            state.robot.x,
            state.robot.y,
            state.robot.heading,
            goal.x,
            goal.y,
        )

        return (dx_local, dy_local)

    def _extract_event_features(self, state: State) -> npt.NDArray[np.float32]:
        """
        Extract event-related features.

        Features (4D):
            [0] event_exists: 1.0 if event exists, 0.0 otherwise
            [1] event_urgency: urgency level (0-1), 0 if no event
            [2] event_confidence: confidence level (0-1), 0 if no event
            [3] event_elapsed_time: normalized elapsed time, 0 if no event

        Args:
            state: Environment state

        Returns:
            4D event feature vector
        """
        features = np.zeros(4, dtype=np.float32)

        if state.has_event:
            event = state.current_event
            features[0] = 1.0  # Event exists
            features[1] = event.urgency
            features[2] = event.confidence
            elapsed = event.time_elapsed(state.current_time)
            features[3] = np.clip(elapsed / self.max_event_delay, 0.0, 1.0)

        return features

    def _extract_patrol_features(self, state: State) -> npt.NDArray[np.float32]:
        """
        Extract patrol coverage features.

        Features (2D):
            [0] distance_to_next: Normalized distance to next patrol point
            [1] coverage_gap_ratio: Ratio of max gap to expected gap

        Args:
            state: Environment state

        Returns:
            2D patrol feature vector
        """
        features = np.zeros(2, dtype=np.float32)

        if len(state.patrol_points) == 0:
            return features

        # Distance to next patrol point (if goal exists)
        if state.robot.current_goal_idx >= 0:
            goal_idx = state.robot.current_goal_idx
            if goal_idx < len(state.patrol_points):
                goal = state.patrol_points[goal_idx]
                distance = np.sqrt(
                    (goal.x - state.robot.x)**2 +
                    (goal.y - state.robot.y)**2
                )
                features[0] = np.clip(
                    distance / self.max_goal_distance,
                    0.0,
                    1.0
                )

        # Coverage gap ratio
        if len(state.patrol_points) > 0:
            max_gap = max(
                state.current_time - point.last_visit_time
                for point in state.patrol_points
            )
            features[1] = np.clip(
                max_gap / self.max_coverage_gap,
                0.0,
                1.0
            )

        return features

    def _extract_event_risk(self, state: State) -> float:
        """
        Extract event risk level.

        Reviewer 박용준: Phase 4 - Event risk information

        Args:
            state: Environment state

        Returns:
            Normalized risk level [0, 1], 0 if no event
        """
        if not state.has_event:
            return 0.0

        event = state.current_event
        # Risk level is 1-9, normalize to [0, 1]
        risk_level = getattr(event, 'risk_level', 5)  # Default to medium risk
        return risk_level / 9.0

    def _extract_patrol_crisis(self, state: State) -> npt.NDArray[np.float32]:
        """
        Extract patrol crisis indicators.

        Reviewer 박용준: Phase 4 - Patrol crisis awareness

        Features (3D):
            [0] max_gap_normalized: Max coverage gap / threshold
            [1] critical_count_normalized: Count of critical points / total points
            [2] crisis_score: Overall crisis level [0, 1+]

        Args:
            state: Environment state

        Returns:
            3D patrol crisis vector
        """
        features = np.zeros(3, dtype=np.float32)

        if len(state.patrol_points) == 0:
            return features

        # Calculate gaps for all points
        gaps = [
            state.current_time - point.last_visit_time
            for point in state.patrol_points
        ]

        # 1. Max gap normalized
        max_gap = max(gaps)
        features[0] = np.clip(max_gap / self.max_coverage_gap, 0.0, 2.0)  # Allow >1 for crisis

        # 2. Critical point count (gap > threshold)
        crisis_threshold = self.max_coverage_gap * 0.5  # 50% of max is "critical"
        critical_count = sum(1 for gap in gaps if gap > crisis_threshold)
        features[1] = critical_count / max(len(state.patrol_points), 1)

        # 3. Overall crisis score (weighted by priority)
        weighted_gap_sum = sum(
            (state.current_time - p.last_visit_time) * p.priority
            for p in state.patrol_points
        )
        total_priority = sum(p.priority for p in state.patrol_points)
        avg_weighted_gap = weighted_gap_sum / max(total_priority, 1.0)
        features[2] = np.clip(avg_weighted_gap / self.max_coverage_gap, 0.0, 2.0)

        return features

    def _extract_candidate_feasibility(self, state: State) -> npt.NDArray[np.float32]:
        """
        Extract candidate feasibility hints.

        Reviewer 박용준: Phase 4 - Candidate quality information

        For each of the 6 candidates, provide a feasibility score:
            - 1.0: Highly feasible (all goals reachable, low risk)
            - 0.5: Moderately feasible (some uncertainty)
            - 0.0: Infeasible (contains unreachable goals)

        Args:
            state: Environment state

        Returns:
            6D feasibility vector (one per candidate)
        """
        features = np.zeros(6, dtype=np.float32)

        if len(state.candidates) == 0:
            return features

        # For each candidate, estimate feasibility
        for i, candidate in enumerate(state.candidates[:6]):  # Ensure max 6
            if i >= 6:
                break

            # Check if candidate route is feasible
            # Heuristic: If total distance is reasonable and no inf distances
            total_distance = candidate.estimated_total_distance

            if total_distance == np.inf or total_distance < 0:
                # Infeasible route
                features[i] = 0.0
            elif total_distance > self.max_goal_distance * 10:
                # Very long route - low feasibility
                features[i] = 0.3
            else:
                # Feasible route - score based on route length
                # Shorter routes = higher feasibility
                normalized_dist = total_distance / (self.max_goal_distance * 5)
                features[i] = np.clip(1.0 - normalized_dist * 0.5, 0.5, 1.0)

        return features

    def _extract_urgency_risk_combined(self, state: State) -> float:
        """
        Extract combined urgency-risk signal.

        Reviewer 박용준: Phase 4 - Combined event priority signal

        Combines event urgency and risk level into single priority signal.
        Helps agent quickly identify high-priority events.

        Args:
            state: Environment state

        Returns:
            Combined urgency × risk signal [0, 1], 0 if no event
        """
        if not state.has_event:
            return 0.0

        event = state.current_event
        urgency = event.urgency
        risk_level = getattr(event, 'risk_level', 5)
        risk_normalized = risk_level / 9.0

        # Geometric mean for combined signal (prevents one dimension from dominating)
        combined = np.sqrt(urgency * risk_normalized)

        return float(np.clip(combined, 0.0, 1.0))

    def reset_stats(self) -> None:
        """Reset running statistics."""
        self.normalizer.reset()
