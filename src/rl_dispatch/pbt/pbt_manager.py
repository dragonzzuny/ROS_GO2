"""
Population-based Training Manager.

Manages a population of agents, orchestrates exploit/explore cycles,
and tracks population-level metrics.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter

from rl_dispatch.pbt.population_member import PopulationMember
from rl_dispatch.pbt.hyperparameter_space import HyperparameterSpace


class PBTManager:
    """
    Manages population-based training process.

    Responsibilities:
    - Maintain population of agents
    - Schedule evaluation and exploit/explore cycles
    - Track population-level metrics
    - Save/load population state
    """

    def __init__(
        self,
        population_size: int = 8,
        eval_interval: int = 10_000,
        exploit_threshold: float = 0.25,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize PBT manager.

        Args:
            population_size: Number of agents in population
            eval_interval: Timesteps between exploit/explore cycles
            exploit_threshold: Fraction of top/bottom performers (0.25 = top/bottom 25%)
            log_dir: Directory for logging (TensorBoard)
        """
        self.population_size = population_size
        self.eval_interval = eval_interval
        self.exploit_threshold = exploit_threshold
        self.log_dir = log_dir

        self.population: List[PopulationMember] = []
        self.total_timesteps = 0
        self.exploit_cycle_count = 0

        # TensorBoard writer
        self.writer = None
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=str(log_dir / "pbt_tensorboard"))

    def add_member(self, member: PopulationMember) -> None:
        """
        Add a member to the population.

        Args:
            member: PopulationMember to add
        """
        if len(self.population) >= self.population_size:
            raise ValueError(
                f"Population full ({self.population_size} members). "
                f"Cannot add more."
            )

        self.population.append(member)

    def get_member(self, member_id: int) -> PopulationMember:
        """
        Get member by ID.

        Args:
            member_id: Member ID to retrieve

        Returns:
            PopulationMember with matching ID

        Raises:
            ValueError: If member not found
        """
        for member in self.population:
            if member.member_id == member_id:
                return member

        raise ValueError(f"Member {member_id} not found in population")

    def get_best_member(self) -> PopulationMember:
        """
        Get member with highest performance.

        Returns:
            Best performing member
        """
        if not self.population:
            raise ValueError("Population is empty")

        return max(self.population, key=lambda m: m.performance)

    def get_worst_member(self) -> PopulationMember:
        """
        Get member with lowest performance.

        Returns:
            Worst performing member
        """
        if not self.population:
            raise ValueError("Population is empty")

        return min(self.population, key=lambda m: m.performance)

    def rank_population(self) -> List[PopulationMember]:
        """
        Rank population by performance (best to worst).

        Returns:
            List of members sorted by descending performance
        """
        return sorted(self.population, key=lambda m: m.performance, reverse=True)

    def should_exploit(self, timesteps_since_last: int) -> bool:
        """
        Check if it's time for exploit/explore cycle.

        Args:
            timesteps_since_last: Timesteps since last exploit cycle

        Returns:
            True if eval_interval has passed
        """
        return timesteps_since_last >= self.eval_interval

    def exploit_and_explore(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Perform exploit and explore cycle.

        Bottom performers adopt weights and hyperparameters from
        top performers, with perturbations.

        Returns:
            Dictionary with:
            - "exploits": List of (bottom_id, top_id) pairs
            - "hyperparams_before": Hyperparams before perturbation
            - "hyperparams_after": Hyperparams after perturbation
        """
        # Rank population
        ranked = self.rank_population()

        # Calculate cutoff index (e.g., top/bottom 25%)
        cutoff = max(1, int(len(ranked) * self.exploit_threshold))

        top_performers = ranked[:cutoff]
        bottom_performers = ranked[-cutoff:]

        print(f"\n{'=' * 80}")
        print(f"EXPLOIT/EXPLORE CYCLE #{self.exploit_cycle_count + 1}")
        print(f"{'=' * 80}")
        print(f"Top {cutoff} performers:")
        for member in top_performers:
            print(f"  {member}")
        print(f"\nBottom {cutoff} performers (will exploit):")
        for member in bottom_performers:
            print(f"  {member}")

        # Track exploits
        exploits = []
        hyperparams_before = []
        hyperparams_after = []

        # Bottom performers exploit top performers
        for bottom in bottom_performers:
            # Randomly select a top performer
            top = np.random.choice(top_performers)

            print(f"\n  Member {bottom.member_id} exploiting Member {top.member_id}:")
            print(f"    Performance: {bottom.performance:.1f} → {top.performance:.1f}")

            # Record old hyperparams
            old_hyperparams = bottom.hyperparams.copy()
            hyperparams_before.append((bottom.member_id, old_hyperparams))

            # Copy weights
            bottom.copy_weights_from(top)

            # Copy and perturb hyperparameters
            bottom.hyperparams = HyperparameterSpace.perturb(
                top.hyperparams.copy()
            )

            # Apply new hyperparameters
            bottom.apply_hyperparams()

            # Record new hyperparams
            hyperparams_after.append((bottom.member_id, bottom.hyperparams.copy()))

            # Update counters
            bottom.exploited_count += 1
            top.exploit_count += 1

            # Record exploit pair
            exploits.append((bottom.member_id, top.member_id))

            print(f"    Old hyperparams: lr={old_hyperparams['learning_rate']:.2e}, "
                  f"batch={old_hyperparams['batch_size']}, "
                  f"epochs={old_hyperparams['num_epochs']}")
            print(f"    New hyperparams: lr={bottom.hyperparams['learning_rate']:.2e}, "
                  f"batch={bottom.hyperparams['batch_size']}, "
                  f"epochs={bottom.hyperparams['num_epochs']}")

        print(f"{'=' * 80}\n")

        # Log to TensorBoard
        if self.writer is not None:
            self._log_exploit_cycle(exploits)

        self.exploit_cycle_count += 1

        return {
            "exploits": exploits,
            "hyperparams_before": hyperparams_before,
            "hyperparams_after": hyperparams_after,
        }

    def log_population_metrics(self, global_step: int) -> None:
        """
        Log population-level metrics to TensorBoard.

        Args:
            global_step: Current global training step
        """
        if self.writer is None:
            return

        performances = [m.performance for m in self.population]
        learning_rates = [m.hyperparams['learning_rate'] for m in self.population]
        batch_sizes = [m.hyperparams['batch_size'] for m in self.population]

        # Population statistics
        self.writer.add_scalar(
            "pbt/population/best_performance",
            max(performances),
            global_step
        )
        self.writer.add_scalar(
            "pbt/population/mean_performance",
            np.mean(performances),
            global_step
        )
        self.writer.add_scalar(
            "pbt/population/worst_performance",
            min(performances),
            global_step
        )
        self.writer.add_scalar(
            "pbt/population/performance_std",
            np.std(performances),
            global_step
        )

        # Hyperparameter diversity
        self.writer.add_scalar(
            "pbt/population/lr_mean",
            np.mean(learning_rates),
            global_step
        )
        self.writer.add_scalar(
            "pbt/population/lr_std",
            np.std(learning_rates),
            global_step
        )
        self.writer.add_scalar(
            "pbt/population/batch_size_mean",
            np.mean(batch_sizes),
            global_step
        )

        # Individual member metrics
        for member in self.population:
            prefix = f"pbt/member_{member.member_id}"
            self.writer.add_scalar(
                f"{prefix}/performance",
                member.performance,
                global_step
            )
            self.writer.add_scalar(
                f"{prefix}/learning_rate",
                member.hyperparams['learning_rate'],
                global_step
            )
            self.writer.add_scalar(
                f"{prefix}/batch_size",
                member.hyperparams['batch_size'],
                global_step
            )
            self.writer.add_scalar(
                f"{prefix}/exploit_count",
                member.exploit_count,
                global_step
            )
            self.writer.add_scalar(
                f"{prefix}/exploited_count",
                member.exploited_count,
                global_step
            )

    def _log_exploit_cycle(self, exploits: List[Tuple[int, int]]) -> None:
        """
        Log exploit cycle information.

        Args:
            exploits: List of (bottom_id, top_id) exploit pairs
        """
        if self.writer is None:
            return

        step = self.exploit_cycle_count

        # Log number of exploits
        self.writer.add_scalar(
            "pbt/exploit/num_exploits",
            len(exploits),
            step
        )

        # Log exploit pairs as text
        exploit_text = "\n".join(
            f"Member {bottom} ← Member {top}"
            for bottom, top in exploits
        )
        self.writer.add_text(
            "pbt/exploit/pairs",
            exploit_text,
            step
        )

    def save_population(self, save_dir: Path) -> None:
        """
        Save entire population to directory.

        Args:
            save_dir: Directory to save to
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save each member
        for member in self.population:
            member.save(save_dir)

        # Save manager state
        manager_state = {
            "population_size": self.population_size,
            "eval_interval": self.eval_interval,
            "exploit_threshold": self.exploit_threshold,
            "total_timesteps": self.total_timesteps,
            "exploit_cycle_count": self.exploit_cycle_count,
            "member_ids": [m.member_id for m in self.population],
        }

        torch.save(
            manager_state,
            str(save_dir / "pbt_manager_state.pth")
        )

        print(f"✅ Saved population to {save_dir}")

    def print_status(self) -> None:
        """Print current population status."""
        print("\n" + "=" * 80)
        print("POPULATION STATUS")
        print("=" * 80)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Exploit cycles: {self.exploit_cycle_count}")
        print(f"\nPopulation ({len(self.population)} members):")

        ranked = self.rank_population()
        for i, member in enumerate(ranked, 1):
            print(f"  #{i:2d}  {member}")

        if self.population:
            best = self.get_best_member()
            print(f"\n🏆 Best member: {best.member_id}")
            print(f"   Performance: {best.performance:.1f}")
            print(f"   Hyperparams: lr={best.hyperparams['learning_rate']:.2e}, "
                  f"batch={best.hyperparams['batch_size']}, "
                  f"epochs={best.hyperparams['num_epochs']}, "
                  f"clip={best.hyperparams['clip_range']:.2f}, "
                  f"entropy={best.hyperparams['entropy_coef']:.3f}")

        print("=" * 80 + "\n")

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
