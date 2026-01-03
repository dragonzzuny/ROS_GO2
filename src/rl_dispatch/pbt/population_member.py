"""
Population Member for Population-based Training.

Each member represents an agent with its own hyperparameters,
network weights, and performance history.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from pathlib import Path

from rl_dispatch.algorithms import PPOAgent


@dataclass
class PopulationMember:
    """
    A single member of the PBT population.

    Attributes:
        member_id: Unique identifier for this member
        agent: PPO agent with network weights
        hyperparams: Current hyperparameters
        recent_returns: Sliding window of episode returns
        total_timesteps: Total timesteps trained
        exploit_count: Number of times this member exploited others
        exploited_count: Number of times this member was exploited
    """
    member_id: int
    agent: PPOAgent
    hyperparams: Dict[str, Any]
    recent_returns: List[float] = field(default_factory=list)
    total_timesteps: int = 0
    exploit_count: int = 0
    exploited_count: int = 0

    # Window size for performance calculation
    PERFORMANCE_WINDOW: int = 20

    @property
    def performance(self) -> float:
        """
        Calculate member performance as mean of recent returns.

        Returns:
            Mean return over last PERFORMANCE_WINDOW episodes,
            or -inf if insufficient data
        """
        if len(self.recent_returns) < 5:
            return -np.inf

        window = self.recent_returns[-self.PERFORMANCE_WINDOW:]
        return float(np.mean(window))

    @property
    def performance_std(self) -> float:
        """Standard deviation of recent returns."""
        if len(self.recent_returns) < 5:
            return np.inf

        window = self.recent_returns[-self.PERFORMANCE_WINDOW:]
        return float(np.std(window))

    def add_return(self, episode_return: float) -> None:
        """
        Add an episode return to history.

        Args:
            episode_return: Return from completed episode
        """
        self.recent_returns.append(episode_return)

        # Keep only last 100 returns to save memory
        if len(self.recent_returns) > 100:
            self.recent_returns = self.recent_returns[-100:]

    def save(self, save_dir: Path) -> None:
        """
        Save member state to directory.

        Args:
            save_dir: Directory to save to (will be created if needed)
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save agent checkpoint
        checkpoint_path = save_dir / f"member_{self.member_id}_agent.pth"
        self.agent.save(str(checkpoint_path))

        # Save member metadata
        metadata = {
            "member_id": self.member_id,
            "hyperparams": self.hyperparams,
            "recent_returns": self.recent_returns,
            "total_timesteps": self.total_timesteps,
            "exploit_count": self.exploit_count,
            "exploited_count": self.exploited_count,
            "performance": self.performance,
        }

        metadata_path = save_dir / f"member_{self.member_id}_metadata.pth"
        torch.save(metadata, str(metadata_path))

    @classmethod
    def load(
        cls,
        member_id: int,
        load_dir: Path,
        agent: PPOAgent,
    ) -> 'PopulationMember':
        """
        Load member from saved state.

        Args:
            member_id: Member ID to load
            load_dir: Directory to load from
            agent: Pre-initialized PPO agent (will load weights)

        Returns:
            Loaded PopulationMember
        """
        # Load agent weights
        checkpoint_path = load_dir / f"member_{member_id}_agent.pth"
        agent.load(str(checkpoint_path))

        # Load metadata
        metadata_path = load_dir / f"member_{member_id}_metadata.pth"
        metadata = torch.load(str(metadata_path), weights_only=False)

        return cls(
            member_id=member_id,
            agent=agent,
            hyperparams=metadata["hyperparams"],
            recent_returns=metadata["recent_returns"],
            total_timesteps=metadata["total_timesteps"],
            exploit_count=metadata["exploit_count"],
            exploited_count=metadata["exploited_count"],
        )

    def copy_weights_from(self, other: 'PopulationMember') -> None:
        """
        Copy network weights from another member.

        Args:
            other: Source member to copy from
        """
        self.agent.network.load_state_dict(
            other.agent.network.state_dict()
        )
        self.agent.optimizer.load_state_dict(
            other.agent.optimizer.state_dict()
        )

    def apply_hyperparams(self) -> None:
        """
        Apply current hyperparameters to the agent.

        Updates:
        - Learning rate (optimizer)
        - Batch size via num_minibatches (training_config)
        - PPO epochs (training_config)
        - Clip epsilon (training_config)
        - Entropy coefficient (training_config)
        """
        # Update learning rate
        for param_group in self.agent.optimizer.param_groups:
            param_group['lr'] = self.hyperparams['learning_rate']

        # Update training config
        # batch_size = num_steps // num_minibatches
        num_minibatches = self.agent.training_config.num_steps // self.hyperparams['batch_size']
        self.agent.training_config.num_minibatches = num_minibatches
        self.agent.training_config.batch_size = self.hyperparams['batch_size']
        self.agent.training_config.num_epochs = self.hyperparams['num_epochs']
        self.agent.training_config.clip_epsilon = self.hyperparams['clip_range']
        self.agent.training_config.entropy_coef = self.hyperparams['entropy_coef']

    def __repr__(self) -> str:
        return (
            f"Member({self.member_id}, "
            f"perf={self.performance:.1f}, "
            f"lr={self.hyperparams['learning_rate']:.2e}, "
            f"batch={self.hyperparams['batch_size']}, "
            f"exploits={self.exploit_count}/{self.exploited_count})"
        )
