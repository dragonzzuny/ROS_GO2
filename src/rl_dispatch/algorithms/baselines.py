"""
Baseline policies for comparison with learned PPO policy.

This module implements hand-crafted baseline policies that provide
performance benchmarks. These policies use fixed rules rather than learning.

Baselines:
- B0: Always Patrol (never dispatch, maintain patrol route)
- B1: Always Dispatch (always respond to events, keep order replan)
- B2: Threshold Dispatch (dispatch if urgency > threshold, nearest-first replan)
- B3: Urgency-Based (dispatch based on urgency, most-overdue-first replan)
- B4: Heuristic Policy (rule-based decision tree)

These baselines help answer:
- How much does learning improve over simple rules?
- Which components of the problem are learnable?
- What is the value of different heuristic strategies?
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from rl_dispatch.core.types import State, Action, ActionMode


class BaselinePolicy(ABC):
    """
    Abstract base class for baseline policies.

    All baseline policies must implement the select_action() method
    which takes a State and returns an Action.
    """

    def __init__(self, name: str):
        """
        Initialize baseline policy.

        Args:
            name: Human-readable policy name
        """
        self.name = name

    @abstractmethod
    def select_action(self, state: State) -> Action:
        """
        Select action for given state.

        Args:
            state: Current environment state

        Returns:
            Action to take
        """
        pass

    def reset(self) -> None:
        """Reset policy state (if stateful). Override if needed."""
        pass


class B0_AlwaysPatrol(BaselinePolicy):
    """
    Baseline B0: Always patrol, never dispatch to events.

    This policy completely ignores events and focuses solely on
    maintaining patrol coverage. Uses keep-order replan strategy.

    Expected performance:
    - High patrol coverage
    - Zero event response
    - Minimal distance traveled
    - Baseline for patrol-only behavior
    """

    def __init__(self):
        super().__init__(name="B0_AlwaysPatrol")

    def select_action(self, state: State) -> Action:
        """Always select patrol mode with keep-order replan."""
        return Action(
            mode=ActionMode.PATROL,
            replan_idx=0,  # Keep-Order strategy
        )


class B1_AlwaysDispatch(BaselinePolicy):
    """
    Baseline B1: Always dispatch if event exists, else patrol.

    This policy always responds to events when they exist.
    Uses keep-order replan to maintain simple patrol sequence.

    Expected performance:
    - High event response rate
    - Poor patrol coverage (interruptions)
    - High distance traveled
    - Baseline for event-focused behavior
    """

    def __init__(self):
        super().__init__(name="B1_AlwaysDispatch")

    def select_action(self, state: State) -> Action:
        """Dispatch if event exists, else patrol."""
        if state.has_event:
            mode = ActionMode.DISPATCH
        else:
            mode = ActionMode.PATROL

        return Action(
            mode=mode,
            replan_idx=0,  # Keep-Order strategy
        )


class B2_ThresholdDispatch(BaselinePolicy):
    """
    Baseline B2: Dispatch if event urgency exceeds threshold.

    Uses a simple threshold rule: dispatch if urgency > threshold.
    Uses nearest-first replan for efficiency.

    This is a common heuristic in practice: only respond to
    "high urgency" events, ignore low urgency ones.

    Attributes:
        urgency_threshold: Threshold for dispatch decision (default: 0.7)
        confidence_threshold: Minimum confidence to dispatch (default: 0.8)

    Expected performance:
    - Balanced event response (high urgency only)
    - Moderate patrol coverage
    - Moderate efficiency
    """

    def __init__(
        self,
        urgency_threshold: float = 0.7,
        confidence_threshold: float = 0.8,
    ):
        super().__init__(name="B2_ThresholdDispatch")
        self.urgency_threshold = urgency_threshold
        self.confidence_threshold = confidence_threshold

    def select_action(self, state: State) -> Action:
        """Dispatch if event is urgent and confident, else patrol."""
        if state.has_event:
            event = state.current_event
            if (event.urgency >= self.urgency_threshold and
                event.confidence >= self.confidence_threshold):
                mode = ActionMode.DISPATCH
            else:
                mode = ActionMode.PATROL
        else:
            mode = ActionMode.PATROL

        return Action(
            mode=mode,
            replan_idx=1,  # Nearest-First strategy
        )


class B3_UrgencyBased(BaselinePolicy):
    """
    Baseline B3: Probabilistic dispatch based on event urgency.

    Instead of hard threshold, dispatches with probability = urgency.
    This provides softer decision boundary and more exploration.

    Uses most-overdue-first replan to recover patrol coverage.

    Expected performance:
    - Graduated event response (more urgent = more likely)
    - Better patrol coverage recovery
    - Stochastic behavior
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="B3_UrgencyBased")
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: State) -> Action:
        """Dispatch with probability proportional to urgency."""
        if state.has_event:
            event = state.current_event
            # Dispatch probability = urgency * confidence
            dispatch_prob = event.urgency * event.confidence

            if self.rng.random() < dispatch_prob:
                mode = ActionMode.DISPATCH
            else:
                mode = ActionMode.PATROL
        else:
            mode = ActionMode.PATROL

        return Action(
            mode=mode,
            replan_idx=2,  # Most-Overdue-First strategy
        )


class B4_HeuristicPolicy(BaselinePolicy):
    """
    Baseline B4: Hand-crafted heuristic policy with multiple rules.

    This policy uses a decision tree with multiple heuristics:
    1. Check battery level (low battery → patrol only)
    2. Check event age (old event → must dispatch)
    3. Check patrol coverage gaps (large gaps → patrol)
    4. Check event urgency (high urgency → dispatch)
    5. Default: patrol

    Replan strategy adapts based on situation:
    - If dispatching: use balanced-coverage to maintain patrol
    - If patrolling: use overdue-eta-balance for efficiency

    This represents a "reasonably smart" hand-crafted policy
    that a domain expert might design.

    Expected performance:
    - Good balance between objectives
    - Context-aware decisions
    - Still suboptimal compared to learning (hopefully!)
    """

    def __init__(
        self,
        battery_threshold: float = 0.3,
        event_age_threshold: float = 60.0,  # seconds
        coverage_gap_threshold: float = 150.0,  # seconds
        urgency_threshold: float = 0.75,
    ):
        super().__init__(name="B4_HeuristicPolicy")
        self.battery_threshold = battery_threshold
        self.event_age_threshold = event_age_threshold
        self.coverage_gap_threshold = coverage_gap_threshold
        self.urgency_threshold = urgency_threshold

    def select_action(self, state: State) -> Action:
        """Select action using decision tree."""
        # Rule 1: Low battery → always patrol
        if state.robot.battery_level < self.battery_threshold:
            return Action(mode=ActionMode.PATROL, replan_idx=1)  # Nearest-first

        # Rule 2: Old high-urgency event → must dispatch
        if state.has_event:
            event = state.current_event
            event_age = event.time_elapsed(state.current_time)

            if (event_age > self.event_age_threshold and
                event.urgency > self.urgency_threshold):
                return Action(mode=ActionMode.DISPATCH, replan_idx=5)  # Balanced-coverage

        # Rule 3: Large coverage gaps → prioritize patrol
        max_gap = max(
            state.current_time - point.last_visit_time
            for point in state.patrol_points
        )
        if max_gap > self.coverage_gap_threshold:
            return Action(mode=ActionMode.PATROL, replan_idx=2)  # Most-overdue-first

        # Rule 4: High urgency event → dispatch
        if state.has_event:
            event = state.current_event
            if event.urgency > self.urgency_threshold and event.confidence > 0.8:
                return Action(mode=ActionMode.DISPATCH, replan_idx=5)  # Balanced-coverage

        # Rule 5: Default → patrol with balanced strategy
        return Action(mode=ActionMode.PATROL, replan_idx=3)  # Overdue-ETA-balance


class BaselineEvaluator:
    """
    Utility class for evaluating baseline policies.

    Provides methods to run multiple baselines on an environment
    and compare their performance.

    Example:
        >>> from rl_dispatch.env import PatrolEnv
        >>> env = PatrolEnv()
        >>> evaluator = BaselineEvaluator(env)
        >>> results = evaluator.evaluate_all(episodes=100)
        >>> evaluator.print_comparison(results)
    """

    def __init__(self, env):
        """
        Initialize evaluator.

        Args:
            env: PatrolEnv instance
        """
        self.env = env
        self.baselines = [
            B0_AlwaysPatrol(),
            B1_AlwaysDispatch(),
            B2_ThresholdDispatch(),
            B3_UrgencyBased(),
            B4_HeuristicPolicy(),
        ]

    def evaluate_policy(
        self,
        policy: BaselinePolicy,
        num_episodes: int = 100,
    ) -> dict:
        """
        Evaluate a single baseline policy.

        Args:
            policy: Baseline policy to evaluate
            num_episodes: Number of episodes

        Returns:
            Dictionary of performance metrics
        """
        episode_returns = []
        episode_lengths = []
        events_detected = []
        events_responded = []
        events_successful = []
        patrol_coverage = []

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            # Note: baseline policies work on State, not obs
            # In practice, you'd need to track state or modify env
            # For now, this is a simplified interface
            episode_return = 0
            done = False

            while not done:
                # Get state from environment (would need env modification)
                state = self.env.current_state

                # Select action
                action = policy.select_action(state)

                # Convert to array format
                action_array = np.array([action.mode, action.replan_idx])

                # Step
                obs, reward, terminated, truncated, info = self.env.step(action_array)
                episode_return += reward
                done = terminated or truncated

            # Collect metrics
            if "episode" in info:
                episode_returns.append(info["episode"]["r"])
                episode_lengths.append(info["episode"]["l"])
                events_detected.append(info["episode"]["events_detected"])
                events_responded.append(info["episode"]["events_responded"])
                events_successful.append(info["episode"]["events_successful"])
                patrol_coverage.append(info["episode"]["patrol_coverage"])

        return {
            "policy_name": policy.name,
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_length": np.mean(episode_lengths),
            "event_success_rate": np.mean(events_successful) / max(np.mean(events_detected), 1),
            "patrol_coverage": np.mean(patrol_coverage),
        }

    def evaluate_all(self, num_episodes: int = 100) -> list:
        """
        Evaluate all baseline policies.

        Args:
            num_episodes: Number of episodes per policy

        Returns:
            List of result dictionaries
        """
        results = []
        for policy in self.baselines:
            print(f"Evaluating {policy.name}...")
            result = self.evaluate_policy(policy, num_episodes)
            results.append(result)
        return results

    def print_comparison(self, results: list) -> None:
        """
        Print comparison table of baseline results.

        Args:
            results: List of result dictionaries
        """
        print("=" * 100)
        print("BASELINE POLICY COMPARISON")
        print("=" * 100)
        print(f"{'Policy':<25} {'Return':<15} {'Success Rate':<15} {'Coverage':<15}")
        print("-" * 100)

        for result in results:
            print(
                f"{result['policy_name']:<25} "
                f"{result['mean_return']:>8.2f} ± {result['std_return']:<4.2f} "
                f"{result['event_success_rate']:>13.1%} "
                f"{result['patrol_coverage']:>13.1%}"
            )

        print("=" * 100)
