"""
Neural network architectures for PPO algorithm.

This module implements the actor-critic network with:
- Shared encoder for feature extraction
- Separate actor heads for mode and replan actions
- Critic head for value estimation
- Support for action masking
- Orthogonal initialization for stable training
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from rl_dispatch.core.config import NetworkConfig


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
    orthogonal: bool = True,
) -> nn.Module:
    """
    Initialize layer weights using orthogonal initialization.

    This initialization strategy improves training stability for deep RL.

    Args:
        layer: Neural network layer to initialize
        std: Standard deviation for orthogonal init
        bias_const: Constant value for bias initialization
        orthogonal: Whether to use orthogonal init (else normal init)

    Returns:
        Initialized layer
    """
    if orthogonal:
        torch.nn.init.orthogonal_(layer.weight, std)
    else:
        torch.nn.init.normal_(layer.weight, std=std)

    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO with composite action space.

    Architecture:
        Input (77D) → Shared Encoder → Actor/Critic Heads

        Shared Encoder: MLP with configurable hidden layers (default 256-256)

        Actor Head: Splits into two sub-heads:
          - Mode Head: 2 outputs (patrol/dispatch) with softmax
          - Replan Head: K outputs (strategy selection) with softmax

        Critic Head: 1 output (state value estimate)

    The network supports action masking for the mode head to enforce
    feasibility constraints (e.g., can't dispatch without event).

    Attributes:
        obs_dim: Observation dimension (77)
        num_replan_strategies: Number of replan strategies (K, typically 6)
        config: NetworkConfig with architecture parameters

    Example:
        >>> config = NetworkConfig()
        >>> network = ActorCriticNetwork(obs_dim=77, num_replan_strategies=6, config=config)
        >>> obs = torch.randn(32, 77)  # Batch of 32 observations
        >>> mode_logits, replan_logits, value = network(obs)
        >>> mode_probs = F.softmax(mode_logits, dim=-1)
    """

    def __init__(
        self,
        obs_dim: int = 77,
        num_replan_strategies: int = 6,
        config: Optional[NetworkConfig] = None,
    ):
        """
        Initialize actor-critic network.

        Args:
            obs_dim: Observation dimensionality
            num_replan_strategies: Number of replan candidate strategies
            config: NetworkConfig (uses defaults if None)
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.num_replan_strategies = num_replan_strategies
        self.config = config or NetworkConfig()

        # Build shared encoder
        self.encoder = self._build_encoder()

        # Actor heads
        # Mode head: 2 outputs (patrol=0, dispatch=1)
        self.mode_head = layer_init(
            nn.Linear(self.config.encoder_hidden_dims[-1], 2),
            std=0.01,  # Small init for policy head
            orthogonal=self.config.orthogonal_init,
        )

        # Replan head: K outputs (one per strategy)
        self.replan_head = layer_init(
            nn.Linear(self.config.encoder_hidden_dims[-1], num_replan_strategies),
            std=0.01,  # Small init for policy head
            orthogonal=self.config.orthogonal_init,
        )

        # Critic head: 1 output (value estimate)
        self.critic_head = layer_init(
            nn.Linear(self.config.encoder_hidden_dims[-1], 1),
            std=1.0,  # Normal init for critic
            orthogonal=self.config.orthogonal_init,
        )

    def _build_encoder(self) -> nn.Sequential:
        """
        Build shared encoder MLP.

        Returns:
            Sequential encoder network
        """
        layers = []
        input_dim = self.obs_dim

        for hidden_dim in self.config.encoder_hidden_dims:
            layers.append(
                layer_init(
                    nn.Linear(input_dim, hidden_dim),
                    std=self.config.init_scale,
                    orthogonal=self.config.orthogonal_init,
                )
            )

            # Activation
            if self.config.activation == "relu":
                layers.append(nn.ReLU())
            elif self.config.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.config.activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {self.config.activation}")

            # Optional layer normalization
            if self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            input_dim = hidden_dim

        return nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            obs: Observation tensor, shape (batch, 77)

        Returns:
            mode_logits: Mode action logits, shape (batch, 2)
            replan_logits: Replan action logits, shape (batch, K)
            value: State value estimates, shape (batch, 1)
        """
        # Shared encoding
        features = self.encoder(obs)

        # Actor heads
        mode_logits = self.mode_head(features)
        replan_logits = self.replan_head(features)

        # Critic head
        value = self.critic_head(features)

        return mode_logits, replan_logits, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for observations.

        Args:
            obs: Observation tensor, shape (batch, 77)

        Returns:
            value: Value estimates, shape (batch, 1)
        """
        features = self.encoder(obs)
        value = self.critic_head(features)
        return value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        mode_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action (sampling or given) and compute log probs and value.

        This is the main method used during training and inference.

        Args:
            obs: Observation tensor, shape (batch, 77)
            action: Optional action tensor, shape (batch, 2). If None, samples new action.
            mode_mask: Optional boolean mask for mode actions, shape (batch, 2).
                       True indicates valid actions. If None, all actions valid.

        Returns:
            action: Selected actions, shape (batch, 2) - [mode, replan]
            log_prob: Log probability of actions, shape (batch,)
            entropy: Entropy of action distribution, shape (batch,)
            value: State value estimates, shape (batch, 1)

        Example:
            >>> network = ActorCriticNetwork()
            >>> obs = torch.randn(32, 77)
            >>> # Sample new actions
            >>> action, log_prob, entropy, value = network.get_action_and_value(obs)
            >>> # Evaluate given actions
            >>> given_action = torch.tensor([[0, 2], [1, 1]])
            >>> _, log_prob, entropy, value = network.get_action_and_value(obs, action=given_action)
        """
        # Forward pass
        mode_logits, replan_logits, value = self.forward(obs)

        # Apply mode action masking if provided
        if mode_mask is not None:
            # Mask out invalid actions with large negative logits
            mode_logits = torch.where(
                mode_mask,
                mode_logits,
                torch.tensor(-1e8, device=mode_logits.device),
            )

        # Create categorical distributions
        mode_dist = Categorical(logits=mode_logits)
        replan_dist = Categorical(logits=replan_logits)

        # Sample or use given actions
        if action is None:
            mode_action = mode_dist.sample()
            replan_action = replan_dist.sample()
            action = torch.stack([mode_action, replan_action], dim=1)
        else:
            mode_action = action[:, 0]
            replan_action = action[:, 1]

        # Compute log probabilities
        mode_log_prob = mode_dist.log_prob(mode_action)
        replan_log_prob = replan_dist.log_prob(replan_action)
        log_prob = mode_log_prob + replan_log_prob  # Independent actions

        # Compute entropy
        mode_entropy = mode_dist.entropy()
        replan_entropy = replan_dist.entropy()
        entropy = mode_entropy + replan_entropy

        return action, log_prob, entropy, value


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder for feature extraction.

    Can be used standalone or as part of larger architectures.

    Example:
        >>> encoder = MLPEncoder(input_dim=77, hidden_dims=[256, 256])
        >>> features = encoder(torch.randn(32, 77))
        >>> assert features.shape == (32, 256)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ):
        """
        Initialize MLP encoder.

        Args:
            input_dim: Input dimensionality
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("relu", "tanh", "elu")
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(layer_init(nn.Linear(current_dim, hidden_dim)))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            current_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
