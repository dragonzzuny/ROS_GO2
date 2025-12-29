# Reviewer: 박용준
"""
Rollout buffer for storing trajectories and computing GAE.

This module implements the experience buffer used in PPO to store
trajectories from environment interaction and compute advantages
using Generalized Advantage Estimation (GAE).

Changes by Reviewer 박용준:
- Added SMDP (Semi-Markov Decision Process) support with variable discount factors
- Gamma is now applied as gamma^(nav_time) to account for variable-duration actions
- This fixes the fundamental learning accuracy issue for time-extended navigation
"""

from typing import Optional, Generator, Tuple
import numpy as np
import torch


class RolloutBuffer:
    """
    Buffer for storing rollout trajectories and computing GAE.

    PPO is an on-policy algorithm that collects a batch of experience,
    computes advantages, and then performs multiple epochs of updates.
    This buffer stores one batch of trajectories.

    The buffer stores:
    - Observations
    - Actions
    - Log probabilities
    - Rewards
    - Values (critic estimates)
    - Dones (episode termination flags)

    After collection, it computes:
    - Advantages using GAE (Generalized Advantage Estimation)
    - Returns (advantages + values)

    Attributes:
        buffer_size: Number of steps to collect before update
        obs_dim: Observation dimensionality
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        device: PyTorch device (CPU or CUDA)

    Example:
        >>> buffer = RolloutBuffer(buffer_size=2048, obs_dim=77)
        >>> # Collect experience
        >>> for step in range(2048):
        ...     buffer.add(obs, action, log_prob, reward, value, done)
        >>> # Compute advantages
        >>> buffer.compute_returns_and_advantages(last_value)
        >>> # Sample minibatches
        >>> for batch in buffer.get(batch_size=256):
        ...     obs_batch, action_batch, ... = batch
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Number of timesteps to store
            obs_dim: Observation dimensionality
            gamma: Discount factor for returns
            gae_lambda: Lambda for GAE computation
            device: PyTorch device ("cpu" or "cuda")
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        # Storage arrays (CPU - faster for collection)
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 2), dtype=np.int64)  # [mode, replan]
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.nav_times = np.zeros(buffer_size, dtype=np.float32)  # Reviewer 박용준: SMDP nav times

        # Reviewer 박용준: Action masks for preventing invalid actions
        # Assume max 20 candidates (2 modes × 10 strategies = 20 action space size)
        self.action_masks = np.ones((buffer_size, 20), dtype=np.float32)

        # Computed during finalize
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        # Current position
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        nav_time: float = 1.0,  # Reviewer 박용준: SMDP navigation time
        action_mask: Optional[np.ndarray] = None,  # Reviewer 박용준: Action masking
    ) -> None:
        """
        Add one timestep of experience to buffer.

        Args:
            obs: Observation, shape (obs_dim,)
            action: Action taken, shape (2,) - [mode, replan]
            log_prob: Log probability of action
            reward: Reward received
            value: Value estimate from critic
            done: Episode termination flag
            nav_time: Navigation time in seconds (for SMDP discount) - Reviewer 박용준
            action_mask: Action mask for next state (1=valid, 0=invalid) - Reviewer 박용준
        """
        if self.pos >= self.buffer_size:
            raise RuntimeError("Buffer overflow - call reset() after compute_returns_and_advantages()")

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = done
        self.nav_times[self.pos] = nav_time  # Reviewer 박용준: Store nav time

        # Reviewer 박용준: Store action mask (pad/truncate to fixed size)
        if action_mask is not None:
            mask_size = min(len(action_mask), self.action_masks.shape[1])
            self.action_masks[self.pos, :mask_size] = action_mask[:mask_size]
        else:
            # Default: all actions valid
            self.action_masks[self.pos, :] = 1.0

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool = False,
    ) -> None:
        """
        Compute advantages using GAE with SMDP variable discount factors.

        Reviewer 박용준: Modified to support SMDP (Semi-Markov Decision Process).
        For navigation actions with duration t, we use gamma^t instead of gamma.

        GAE (Generalized Advantage Estimation) computes advantages as:
            A_t = δ_t + (γ^t_k λ)δ_{t+1} + (γ^t_k λ)²δ_{t+2} + ...
        where:
            δ_t = r_t + γ^t_k V(s_{t+1}) - V(s_t)
            γ^t_k = gamma^(nav_time_k)  [SMDP variable discount]

        This correctly accounts for variable-duration navigation actions:
        - Short navigation (e.g., 5s): γ^5 ≈ 0.95 (less discounting)
        - Long navigation (e.g., 60s): γ^60 ≈ 0.05 (strong discounting)

        Args:
            last_value: Value estimate for the last state (bootstrap)
            last_done: Whether last state was terminal
        """
        if not self.full:
            raise RuntimeError("Buffer not full - cannot compute returns")

        # Start from the end and work backwards
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                # Last step - bootstrap from last_value
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                # Regular step - use next stored value
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]

            # Reviewer 박용준: SMDP variable discount factor
            # gamma_k = gamma^(nav_time) to account for variable action durations
            gamma_k = self.gamma ** self.nav_times[step]

            # TD error: δ_t = r_t + γ^t V(s_{t+1}) - V(s_t)
            delta = (
                self.rewards[step] +
                gamma_k * next_value * next_non_terminal -
                self.values[step]
            )

            # GAE: A_t = δ_t + (γ^t λ)A_{t+1}
            last_gae_lam = (
                delta +
                gamma_k * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam

        # Returns = Advantages + Values
        # This is equivalent to the discounted sum of rewards
        self.returns = self.advantages + self.values

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate random minibatches for training.

        Yields minibatches of experience for PPO updates. Each epoch
        should iterate through all data exactly once (without replacement).

        Args:
            batch_size: Minibatch size. If None, returns full buffer.

        Yields:
            Tuple of (obs, actions, log_probs, advantages, returns, values, action_masks)
            All tensors have batch dimension = batch_size
            Reviewer 박용준: Added action_masks to return tuple

        Example:
            >>> for epoch in range(num_epochs):
            ...     for batch in buffer.get(batch_size=256):
            ...         obs, actions, old_log_probs, advantages, returns, values, masks = batch
            ...         # Perform PPO update with action masking
        """
        if not self.full:
            raise RuntimeError("Buffer not full - cannot sample")

        # Normalize advantages (improves training stability)
        # Important: normalize across the entire buffer, not per minibatch
        advantages = self.advantages.copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors (Reviewer 박용준: Added action_masks)
        obs = torch.from_numpy(self.observations).to(self.device)
        actions = torch.from_numpy(self.actions).to(self.device)
        log_probs = torch.from_numpy(self.log_probs).to(self.device)
        advantages_t = torch.from_numpy(advantages).to(self.device)
        returns = torch.from_numpy(self.returns).to(self.device)
        values = torch.from_numpy(self.values).to(self.device)
        action_masks = torch.from_numpy(self.action_masks).to(self.device)

        if batch_size is None:
            # Return entire buffer as one batch
            yield (obs, actions, log_probs, advantages_t, returns, values, action_masks)
        else:
            # Generate random minibatches
            indices = np.arange(self.buffer_size)
            np.random.shuffle(indices)

            # Split into minibatches
            num_batches = self.buffer_size // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                batch_indices = indices[start:end]

                yield (
                    obs[batch_indices],
                    actions[batch_indices],
                    log_probs[batch_indices],
                    advantages_t[batch_indices],
                    returns[batch_indices],
                    values[batch_indices],
                    action_masks[batch_indices],  # Reviewer 박용준: Action masks
                )

    def reset(self) -> None:
        """Reset buffer to empty state."""
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        """Return number of timesteps currently stored."""
        return self.pos if not self.full else self.buffer_size

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.full
