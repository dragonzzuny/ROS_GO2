"""
Proximal Policy Optimization (PPO) algorithm implementation.

This module implements the complete PPO algorithm including:
- Policy and value network updates
- Clipped surrogate objective
- Value function loss
- Entropy bonus
- Learning rate scheduling
- Gradient clipping
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_dispatch.algorithms.networks import ActorCriticNetwork
from rl_dispatch.algorithms.buffer import RolloutBuffer
from rl_dispatch.core.config import TrainingConfig, NetworkConfig


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    PPO is a policy gradient algorithm that uses a clipped surrogate objective
    to prevent large policy updates. This implementation includes:

    - Clipped policy loss (PPO-CLIP)
    - Value function loss (MSE)
    - Entropy bonus (exploration)
    - Advantage normalization
    - Gradient clipping
    - Learning rate annealing

    The agent maintains:
    - Actor-critic network
    - Optimizer
    - Rollout buffer
    - Training statistics

    Attributes:
        network: ActorCriticNetwork
        optimizer: Adam optimizer
        buffer: RolloutBuffer for experience
        config: TrainingConfig with hyperparameters

    Example:
        >>> config = TrainingConfig()
        >>> agent = PPOAgent(obs_dim=77, num_replan_strategies=6, config=config)
        >>> # Collect experience
        >>> for step in range(config.num_steps):
        ...     action, log_prob, value = agent.get_action(obs)
        ...     next_obs, reward, done, info = env.step(action)
        ...     agent.buffer.add(obs, action, log_prob, reward, value, done)
        ...     obs = next_obs
        >>> # Update policy
        >>> stats = agent.update()
    """

    def __init__(
        self,
        obs_dim: int = 77,
        num_replan_strategies: int = 6,
        training_config: Optional[TrainingConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize PPO agent.

        Args:
            obs_dim: Observation dimensionality
            num_replan_strategies: Number of replan candidate strategies
            training_config: TrainingConfig (uses defaults if None)
            network_config: NetworkConfig (uses defaults if None)
            device: Device string ("cpu", "cuda", or None for auto-detect)
        """
        self.training_config = training_config or TrainingConfig()
        self.network_config = network_config or NetworkConfig()

        # Determine device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.training_config.cuda else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Initialize network
        self.network = ActorCriticNetwork(
            obs_dim=obs_dim,
            num_replan_strategies=num_replan_strategies,
            config=self.network_config,
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.training_config.learning_rate,
            eps=1e-5,
        )

        # Initialize rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.training_config.num_steps,
            obs_dim=obs_dim,
            gamma=self.training_config.gamma,
            gae_lambda=self.training_config.gae_lambda,
            device=str(self.device),
        )

        # Training state
        self.global_step = 0
        self.update_count = 0

    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action for given observation.

        Args:
            obs: Observation array, shape (obs_dim,)
            deterministic: If True, select argmax action (for evaluation)
            action_mask: Optional action mask, shape (2*K,) flattened.
                        First K elements are patrol masks, next K are dispatch masks.

        Returns:
            action: Action array, shape (2,) - [mode, replan]
            log_prob: Log probability of action
            value: Value estimate
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

            # Reviewer 박용준: Convert action_mask to mode_mask
            mode_mask = None
            if action_mask is not None:
                # action_mask: (2*K,) -> extract mode availability
                num_candidates = action_mask.shape[0] // 2
                patrol_valid = action_mask[:num_candidates].max() > 0.5
                dispatch_valid = action_mask[num_candidates:].max() > 0.5
                mode_mask = torch.tensor(
                    [[patrol_valid, dispatch_valid]],
                    dtype=torch.bool,
                    device=self.device
                )

            if deterministic:
                # Deterministic policy (argmax)
                mode_logits, replan_logits, value = self.network(obs_t)

                # Apply masking before argmax
                if mode_mask is not None:
                    mode_logits = torch.where(
                        mode_mask,
                        mode_logits,
                        torch.tensor(-1e8, device=mode_logits.device),
                    )

                mode_action = mode_logits.argmax(dim=-1)
                replan_action = replan_logits.argmax(dim=-1)
                action = torch.stack([mode_action, replan_action], dim=1)
                log_prob = torch.tensor(0.0)  # Not used in eval
            else:
                # Stochastic policy (sample) with masking
                action, log_prob, _, value = self.network.get_action_and_value(
                    obs_t, mode_mask=mode_mask
                )

            return (
                action.cpu().numpy()[0],
                log_prob.cpu().item(),
                value.cpu().item(),
            )

    def update(
        self,
        last_value: float = 0.0,
        last_done: bool = True,
    ) -> Dict[str, float]:
        """
        Perform PPO update using collected experience.

        This executes the full PPO update procedure:
        1. Compute returns and advantages (GAE)
        2. For each epoch:
            a. Sample minibatches
            b. Compute policy loss (clipped)
            c. Compute value loss (MSE)
            d. Compute entropy bonus
            e. Backprop and update
        3. Update learning rate (if annealing)

        Args:
            last_value: Value estimate for last state (for bootstrapping)
            last_done: Whether last state was terminal

        Returns:
            Dictionary of training statistics:
            - policy_loss: Mean policy loss
            - value_loss: Mean value loss
            - entropy: Mean entropy
            - approx_kl: Approximate KL divergence
            - clipfrac: Fraction of updates clipped
            - explained_variance: How well value function fits returns
        """
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(last_value, last_done)

        # Training statistics
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clipfracs = []

        # Multiple epochs of updates
        for epoch in range(self.training_config.num_epochs):
            # Iterate through minibatches
            for batch in self.buffer.get(batch_size=self.training_config.batch_size):
                # Reviewer 박용준: Unpack 7 values (added action_masks)
                obs, actions, old_log_probs, advantages, returns, old_values, action_masks = batch

                # Reviewer 박용준: Convert action_masks to mode_masks
                # action_masks: (batch, 2*K) -> mode_mask: (batch, 2)
                num_candidates = action_masks.shape[1] // 2
                patrol_valid = action_masks[:, :num_candidates].max(dim=1)[0] > 0.5
                dispatch_valid = action_masks[:, num_candidates:].max(dim=1)[0] > 0.5
                mode_mask = torch.stack([patrol_valid, dispatch_valid], dim=1)

                # Forward pass through network with action masking
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    obs, action=actions, mode_mask=mode_mask
                )

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                policy_loss_1 = -advantages * ratio
                policy_loss_2 = -advantages * torch.clamp(
                    ratio,
                    1.0 - self.training_config.clip_epsilon,
                    1.0 + self.training_config.clip_epsilon,
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                # Value loss (MSE with optional clipping)
                values = values.squeeze()
                if self.training_config.clip_epsilon > 0:
                    # Clipped value loss (reduces large value updates)
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.training_config.clip_epsilon,
                        self.training_config.clip_epsilon,
                    )
                    value_loss_1 = (values - returns) ** 2
                    value_loss_2 = (values_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    # Unclipped value loss
                    value_loss = 0.5 * ((values - returns) ** 2).mean()

                # Entropy bonus (encourages exploration)
                entropy_loss = entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.training_config.value_loss_coef * value_loss -
                    self.training_config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (prevents exploding gradients)
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.training_config.max_grad_norm,
                )

                self.optimizer.step()

                # Logging statistics
                with torch.no_grad():
                    # Approximate KL divergence
                    approx_kl = ((ratio - 1) - ratio.log()).mean()
                    # Fraction of ratios that were clipped
                    clipfrac = ((ratio - 1.0).abs() > self.training_config.clip_epsilon).float().mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())
                clipfracs.append(clipfrac.item())

        # Update learning rate (linear annealing)
        if self.training_config.anneal_lr:
            frac = 1.0 - self.update_count / self.training_config.total_updates
            new_lr = frac * self.training_config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

        # Compute explained variance
        with torch.no_grad():
            y_pred = self.buffer.values
            y_true = self.buffer.returns
            var_y = np.var(y_true)
            explained_var = (
                1 - np.var(y_true - y_pred) / (var_y + 1e-8)
                if var_y > 0 else 0.0
            )

        # Reset buffer for next rollout
        self.buffer.reset()

        # Increment counters
        self.update_count += 1
        self.global_step += self.training_config.num_steps

        # Return statistics
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
            "approx_kl": np.mean(approx_kls),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        }

    def save(self, path: str) -> None:
        """
        Save agent state to file.

        Args:
            path: File path to save to (e.g., "models/checkpoint.pth")
        """
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "training_config": self.training_config,
            "network_config": self.network_config,
        }, path)

    def load(self, path: str) -> None:
        """
        Load agent state from file.

        Args:
            path: File path to load from
        """
        # PyTorch 2.6+: Need weights_only=False to load custom classes (TrainingConfig, etc.)
        # This is safe since we trust our own checkpoints
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.update_count = checkpoint["update_count"]

    def eval_mode(self) -> None:
        """Set network to evaluation mode."""
        self.network.eval()

    def train_mode(self) -> None:
        """Set network to training mode."""
        self.network.train()
