"""
Visualization utilities for RL Dispatch system.

Provides plotting and visualization tools for:
- Training curves
- Episode trajectories
- Reward component breakdowns
- Patrol coverage heatmaps
- Event response analysis
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from pathlib import Path


class TrainingVisualizer:
    """
    Visualizer for training metrics and performance.

    Provides methods to create publication-quality plots of:
    - Learning curves
    - Reward components over time
    - Episode statistics
    - Comparison with baselines

    Example:
        >>> visualizer = TrainingVisualizer()
        >>> fig = visualizer.plot_learning_curve(returns, steps)
        >>> fig.savefig("learning_curve.png")
    """

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        self.style = style
        sns.set_palette("husl")

    def plot_learning_curve(
        self,
        returns: List[float],
        steps: Optional[List[int]] = None,
        window: int = 100,
        title: str = "Learning Curve",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot learning curve with smoothing.

        Args:
            returns: List of episode returns
            steps: List of step counts (if None, uses episode index)
            window: Rolling average window size
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Use episode index if steps not provided
            x = steps if steps is not None else np.arange(len(returns))

            # Plot raw returns (semi-transparent)
            ax.plot(x, returns, alpha=0.3, color="blue", label="Raw")

            # Plot smoothed returns
            if len(returns) >= window:
                smoothed = self._moving_average(returns, window)
                ax.plot(x[:len(smoothed)], smoothed, color="blue", linewidth=2, label=f"Smoothed ({window})")

            ax.set_xlabel("Training Steps" if steps is not None else "Episode")
            ax.set_ylabel("Episode Return")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

            return fig

    def plot_reward_components(
        self,
        components: Dict[str, List[float]],
        steps: Optional[List[int]] = None,
        title: str = "Reward Components Over Time",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot reward component breakdown over time.

        Args:
            components: Dict mapping component names to values
            steps: List of step counts
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(12, 6))

            for name, values in components.items():
                x = steps if steps is not None else np.arange(len(values))
                smoothed = self._moving_average(values, window=100)
                ax.plot(x[:len(smoothed)], smoothed, label=name, linewidth=2)

            ax.set_xlabel("Training Steps" if steps is not None else "Episode")
            ax.set_ylabel("Reward Component Value")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

            return fig

    def plot_baseline_comparison(
        self,
        baseline_results: Dict[str, Dict[str, float]],
        metrics: List[str] = ["mean_return", "event_success_rate", "patrol_coverage"],
        title: str = "Baseline Policy Comparison",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Create bar chart comparing baseline policies.

        Args:
            baseline_results: Dict mapping policy names to result dicts
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        with plt.style.context(self.style):
            n_metrics = len(metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))

            if n_metrics == 1:
                axes = [axes]

            policy_names = list(baseline_results.keys())

            for i, metric in enumerate(metrics):
                values = [baseline_results[name].get(metric, 0) for name in policy_names]

                axes[i].bar(range(len(policy_names)), values, color=sns.color_palette("husl", len(policy_names)))
                axes[i].set_xticks(range(len(policy_names)))
                axes[i].set_xticklabels(policy_names, rotation=45, ha="right")
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for j, v in enumerate(values):
                    axes[i].text(j, v, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

            fig.suptitle(title, fontsize=14, fontweight="bold")
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

            return fig

    def plot_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Metrics",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot multiple training metrics in subplots.

        Args:
            metrics: Dict mapping metric names to value lists
            title: Overall title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        with plt.style.context(self.style):
            n_metrics = len(metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            axes = np.array(axes).flatten()

            for i, (name, values) in enumerate(metrics.items()):
                ax = axes[i]
                x = np.arange(len(values))
                ax.plot(x, values, linewidth=1.5)
                ax.set_title(name.replace("_", " ").title())
                ax.set_xlabel("Update")
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for i in range(n_metrics, len(axes)):
                axes[i].axis('off')

            fig.suptitle(title, fontsize=16, fontweight="bold")
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

            return fig

    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """Calculate moving average."""
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window)/window, mode='valid')


class TrajectoryVisualizer:
    """
    Visualizer for episode trajectories and spatial data.

    Provides methods to visualize:
    - Robot paths
    - Patrol point coverage
    - Event locations
    - Heatmaps

    Example:
        >>> visualizer = TrajectoryVisualizer(map_width=50, map_height=50)
        >>> fig = visualizer.plot_trajectory(positions, patrol_points, events)
    """

    def __init__(self, map_width: float = 50.0, map_height: float = 50.0):
        """
        Initialize trajectory visualizer.

        Args:
            map_width: Map width in meters
            map_height: Map height in meters
        """
        self.map_width = map_width
        self.map_height = map_height

    def plot_trajectory(
        self,
        robot_positions: List[Tuple[float, float]],
        patrol_points: List[Tuple[float, float]],
        events: Optional[List[Tuple[float, float]]] = None,
        title: str = "Episode Trajectory",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot robot trajectory with patrol points and events.

        Args:
            robot_positions: List of (x, y) positions
            patrol_points: List of patrol point (x, y) positions
            events: Optional list of event (x, y) positions
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot robot trajectory
        if len(robot_positions) > 0:
            positions = np.array(robot_positions)
            ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, linewidth=1, label="Robot Path")
            ax.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='o',
                      edgecolors='black', linewidth=2, label="Start", zorder=5)
            ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, marker='X',
                      edgecolors='black', linewidth=2, label="End", zorder=5)

        # Plot patrol points
        if len(patrol_points) > 0:
            patrol_arr = np.array(patrol_points)
            ax.scatter(patrol_arr[:, 0], patrol_arr[:, 1], c='blue', s=300, marker='s',
                      alpha=0.6, edgecolors='black', linewidth=2, label="Patrol Points", zorder=4)

        # Plot events
        if events and len(events) > 0:
            events_arr = np.array(events)
            ax.scatter(events_arr[:, 0], events_arr[:, 1], c='orange', s=250, marker='^',
                      alpha=0.8, edgecolors='black', linewidth=2, label="Events", zorder=4)

        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.set_xlabel("X Position (m)", fontsize=12)
        ax.set_ylabel("Y Position (m)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_coverage_heatmap(
        self,
        visit_counts: np.ndarray,
        title: str = "Patrol Coverage Heatmap",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot heatmap of patrol coverage.

        Args:
            visit_counts: 2D array of visit counts
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(visit_counts.T, origin='lower', cmap='hot', interpolation='bilinear',
                      extent=[0, self.map_width, 0, self.map_height])

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Visit Count", rotation=270, labelpad=20)

        ax.set_xlabel("X Position (m)", fontsize=12)
        ax.set_ylabel("Y Position (m)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


def create_summary_report(
    training_results: Dict,
    baseline_results: Optional[Dict] = None,
    save_dir: str = "results",
) -> None:
    """
    Create comprehensive summary report with multiple plots.

    Args:
        training_results: Dictionary with training metrics
        baseline_results: Optional baseline comparison results
        save_dir: Directory to save plots
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Training visualizer
    train_viz = TrainingVisualizer()

    # Learning curve
    if "episode_returns" in training_results:
        fig = train_viz.plot_learning_curve(
            training_results["episode_returns"],
            steps=training_results.get("steps"),
            title="Learning Curve",
            save_path=str(save_path / "learning_curve.png")
        )
        plt.close(fig)

    # Reward components
    if "reward_components" in training_results:
        fig = train_viz.plot_reward_components(
            training_results["reward_components"],
            title="Reward Components",
            save_path=str(save_path / "reward_components.png")
        )
        plt.close(fig)

    # Training metrics
    if "training_metrics" in training_results:
        fig = train_viz.plot_training_metrics(
            training_results["training_metrics"],
            title="Training Metrics",
            save_path=str(save_path / "training_metrics.png")
        )
        plt.close(fig)

    # Baseline comparison
    if baseline_results:
        fig = train_viz.plot_baseline_comparison(
            baseline_results,
            title="Baseline Comparison",
            save_path=str(save_path / "baseline_comparison.png")
        )
        plt.close(fig)

    print(f"Summary report saved to: {save_path}")
