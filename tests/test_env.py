"""
Unit tests for PatrolEnv Gymnasium environment.

Tests environment interface, episode flow, and SMDP semantics.
"""

import pytest
import numpy as np
import gymnasium as gym

from rl_dispatch.env import PatrolEnv
from rl_dispatch.core.config import EnvConfig, RewardConfig


class TestPatrolEnv:
    """Test PatrolEnv environment."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return PatrolEnv()

    def test_creation(self, env):
        """Test environment creation."""
        assert isinstance(env, gym.Env)
        assert env.observation_space.shape == (77,)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset(seed=42)

        assert obs.shape == (77,)
        assert isinstance(info, dict)
        assert "episode_step" in info
        assert info["episode_step"] == 0

    def test_step(self, env):
        """Test environment step."""
        env.reset(seed=42)

        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (77,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_length(self, env):
        """Test that episode terminates."""
        env.reset(seed=42)

        done = False
        steps = 0
        max_steps = 1000

        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        # Should terminate before max_steps
        assert done
        assert steps < max_steps

    def test_action_space(self, env):
        """Test action space structure."""
        assert env.action_space.nvec.tolist() == [2, 6]

    def test_seeding(self, env):
        """Test that seeding produces reproducible results."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_info_dict(self, env):
        """Test that info dict contains required keys."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)

        assert "episode_step" in info
        assert "current_time" in info
        assert "has_event" in info
        assert "action_mode" in info
        assert "reward_components" in info

    def test_episode_metrics(self, env):
        """Test that episode metrics are logged."""
        env.reset(seed=42)

        done = False
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Final step should have episode metrics
        if done:
            assert "episode" in info
            assert "r" in info["episode"]
            assert "l" in info["episode"]
            assert "events_detected" in info["episode"]


class TestEnvConfiguration:
    """Test environment configuration."""

    def test_custom_config(self):
        """Test environment with custom configuration."""
        config = EnvConfig(
            map_width=100.0,
            map_height=100.0,
            max_episode_steps=50,
        )
        env = PatrolEnv(env_config=config)
        assert env.env_config.map_width == 100.0

    def test_reward_config(self):
        """Test environment with custom reward configuration."""
        reward_config = RewardConfig(
            w_event=2.0,
            w_patrol=1.0,
        )
        env = PatrolEnv(reward_config=reward_config)
        assert env.reward_config.w_event == 2.0


class TestEnvSMDP:
    """Test SMDP semantics of environment."""

    def test_variable_time_steps(self):
        """Test that steps take variable time."""
        env = PatrolEnv()
        env.reset(seed=42)

        times = []
        for _ in range(10):
            _, _, _, _, info = env.step(env.action_space.sample())
            times.append(info["current_time"])

        # Time deltas should vary (not fixed timestep)
        deltas = np.diff(times)
        assert np.std(deltas) > 0  # Not all the same

    def test_time_monotonic(self):
        """Test that time increases monotonically."""
        env = PatrolEnv()
        env.reset(seed=42)

        times = []
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            times.append(info["current_time"])
            done = terminated or truncated

        # Times should be strictly increasing
        assert all(times[i] < times[i+1] for i in range(len(times)-1))
