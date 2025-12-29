#!/usr/bin/env python3
"""
Coverage Heatmap Visualization Script

이 스크립트는 순찰 커버리지를 시각적으로 확인할 수 있는 히트맵을 생성합니다.
여러 맵에서의 커버리지를 비교하여 정책의 일반화 성능을 평가할 수 있습니다.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_dispatch.env import MultiMapPatrolEnv, create_multi_map_env
from rl_dispatch.algorithms.ppo import PPOPolicy
from rl_dispatch.core.config import EnvConfig


def visualize_coverage_heatmap(
    heatmap: np.ndarray,
    config: EnvConfig,
    map_name: str,
    ax: plt.Axes,
    heatmap_resolution: float = 2.0,
):
    """
    Visualize coverage heatmap for a single map.

    Args:
        heatmap: 2D array of visit counts
        config: Environment configuration
        map_name: Name of the map
        ax: Matplotlib axes
        heatmap_resolution: Grid cell size in meters
    """
    # Display heatmap
    extent = [0, config.map_width, 0, config.map_height]

    # Use log scale for better visualization (many visits in some cells, few in others)
    heatmap_log = np.log1p(heatmap)  # log(1 + x) to handle zeros

    im = ax.imshow(
        heatmap_log,
        origin="lower",
        extent=extent,
        cmap="YlOrRd",
        aspect="equal",
        interpolation="bilinear",
        alpha=0.8,
    )

    # Draw patrol points
    patrol_x = [pt[0] for pt in config.patrol_points]
    patrol_y = [pt[1] for pt in config.patrol_points]

    ax.scatter(
        patrol_x, patrol_y,
        s=200, c="blue", marker="o",
        edgecolors="white", linewidth=2,
        label="Patrol Points", zorder=5, alpha=0.9
    )

    # Label patrol points
    for i, (x, y) in enumerate(config.patrol_points):
        ax.text(
            x, y, f"P{i}",
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            color="white", zorder=6
        )

    # Map boundary
    ax.add_patch(
        patches.Rectangle(
            (0, 0), config.map_width, config.map_height,
            fill=False, edgecolor="black", linewidth=2
        )
    )

    # Styling
    ax.set_xlim(-2, config.map_width + 2)
    ax.set_ylim(-2, config.map_height + 2)
    ax.set_xlabel("X (meters)", fontsize=10)
    ax.set_ylabel("Y (meters)", fontsize=10)
    ax.set_title(f"{map_name}\nCoverage Heatmap", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log(1 + visits)", fontsize=9)

    # Statistics
    total_visits = np.sum(heatmap)
    cells_visited = np.sum(heatmap > 0)
    total_cells = heatmap.size
    coverage_pct = 100.0 * cells_visited / total_cells

    stats_text = f"""Coverage: {coverage_pct:.1f}%
Cells Visited: {cells_visited}/{total_cells}
Total Visits: {int(total_visits)}"""

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        family="monospace"
    )


def run_coverage_test(
    env: MultiMapPatrolEnv,
    policy: PPOPolicy,
    episodes_per_map: int = 10,
) -> None:
    """
    Run episodes on each map to collect coverage data.

    Args:
        env: Multi-map environment
        policy: Trained policy (or None for random)
        episodes_per_map: Episodes to run on each map
    """
    print("\n" + "=" * 80)
    print("Running Coverage Test")
    print("=" * 80)

    for map_name in env.map_names:
        print(f"\nTesting on {map_name}...")

        for ep in range(episodes_per_map):
            obs, info = env.reset(options={"map_name": map_name})
            done = False
            step_count = 0

            while not done:
                if policy is not None:
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    # Random policy
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1

            print(f"  Episode {ep+1}/{episodes_per_map}: {step_count} steps")

    print("\n✅ Coverage test complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize patrol coverage heatmaps across multiple maps"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained PPO model (uses random policy if not provided)",
    )
    parser.add_argument(
        "--episodes-per-map",
        type=int,
        default=10,
        help="Episodes to run on each map",
    )
    parser.add_argument(
        "--heatmap-resolution",
        type=float,
        default=2.0,
        help="Grid cell size for heatmap (meters)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/coverage_heatmaps.png",
        help="Output path for visualization",
    )
    parser.add_argument(
        "--maps",
        type=str,
        nargs="+",
        default=None,
        help="Specific map config files to use (uses all if not specified)",
    )

    args = parser.parse_args()

    # Create environment
    print("Creating multi-map environment...")
    env = create_multi_map_env(
        map_configs=args.maps,
        track_coverage=True,
        heatmap_resolution=args.heatmap_resolution,
    )

    print(f"Loaded {len(env.map_names)} maps:")
    for name in env.map_names:
        config = env.env_configs[name]
        print(f"  - {name}: {config.map_width}×{config.map_height}m, "
              f"{len(config.patrol_points)} points")

    # Load policy (if provided)
    policy = None
    if args.model:
        print(f"\nLoading policy from {args.model}...")
        policy = PPOPolicy.load(args.model)
        print("✅ Policy loaded!")
    else:
        print("\nNo model provided, using random policy")

    # Run coverage test
    run_coverage_test(env, policy, args.episodes_per_map)

    # Print statistics
    env.print_statistics()

    # Visualize all heatmaps
    print("\n" + "=" * 80)
    print("Generating Coverage Visualizations")
    print("=" * 80)

    n_maps = len(env.map_names)
    fig, axes = plt.subplots(
        nrows=(n_maps + 2) // 3,  # 3 maps per row
        ncols=3,
        figsize=(18, 6 * ((n_maps + 2) // 3)),
        squeeze=False
    )

    for idx, map_name in enumerate(env.map_names):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        heatmap = env.get_coverage_heatmap(map_name)
        config = env.env_configs[map_name]

        visualize_coverage_heatmap(
            heatmap, config, map_name, ax, args.heatmap_resolution
        )

    # Hide empty subplots
    for idx in range(n_maps, len(axes.flat)):
        axes.flat[idx].axis("off")

    plt.suptitle(
        f"Patrol Coverage Analysis - {args.episodes_per_map} episodes per map",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    plt.tight_layout()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved coverage visualization to: {output_path}")

    plt.show()

    env.close()


if __name__ == "__main__":
    main()
