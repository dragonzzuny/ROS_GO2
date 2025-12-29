#!/usr/bin/env python3
"""Simple map visualization script to show patrol point layout."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_dispatch.core.config import EnvConfig


def visualize_map(config: EnvConfig, save_path: str = None):
    """Visualize the patrol map layout.

    Args:
        config: Environment configuration
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw map boundaries
    ax.add_patch(plt.Rectangle(
        (0, 0), config.map_width, config.map_height,
        fill=False, edgecolor='black', linewidth=2
    ))

    # Draw patrol points
    patrol_x = [pt[0] for pt in config.patrol_points]
    patrol_y = [pt[1] for pt in config.patrol_points]

    ax.scatter(patrol_x, patrol_y,
               s=300, c='blue', marker='o',
               alpha=0.6, edgecolors='darkblue', linewidth=2,
               label='Patrol Points', zorder=5)

    # Label patrol points
    for i, (x, y) in enumerate(config.patrol_points):
        ax.text(x, y + 2, f'P{i}\n({x:.1f}, {y:.1f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Draw patrol route (connecting lines)
    route_x = patrol_x + [patrol_x[0]]  # Close the loop
    route_y = patrol_y + [patrol_y[0]]
    ax.plot(route_x, route_y, 'b--', alpha=0.3, linewidth=1.5, label='Patrol Route')

    # Calculate and display distances
    distances = []
    for i in range(len(config.patrol_points)):
        p1 = np.array(config.patrol_points[i])
        p2 = np.array(config.patrol_points[(i + 1) % len(config.patrol_points)])
        dist = np.linalg.norm(p2 - p1)
        distances.append(dist)

        # Display distance on edge
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        ax.text(mid_x, mid_y, f'{dist:.1f}m',
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-5, config.map_width + 5)
    ax.set_ylim(-5, config.map_height + 5)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Patrol Map Configuration', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')

    # Add statistics text box
    total_route_length = sum(distances)
    avg_time = total_route_length / config.robot_max_velocity

    stats_text = f"""Map Statistics:
━━━━━━━━━━━━━━━━━━━━
Size: {config.map_width}m × {config.map_height}m
Patrol Points: {len(config.patrol_points)}
Total Route: {total_route_length:.1f}m
Robot Speed: {config.robot_max_velocity:.1f} m/s
Full Patrol Time: ~{avg_time:.1f}s ({avg_time/60:.1f} min)
Event Rate: {config.event_generation_rate:.1f} per episode
Episode Duration: {config.max_episode_time:.0f}s ({config.max_episode_time/60:.0f} min)
"""

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Map visualization saved to: {save_path}")

    plt.show()


def main():
    """Main function."""
    # Load default configuration
    config = EnvConfig()

    print("=" * 60)
    print("🗺️  Patrol Map Configuration")
    print("=" * 60)
    print(f"\n📐 Map Size: {config.map_width}m × {config.map_height}m")
    print(f"\n📍 Patrol Points ({len(config.patrol_points)} points):")
    for i, pt in enumerate(config.patrol_points):
        print(f"   P{i}: ({pt[0]:.1f}, {pt[1]:.1f})")

    print(f"\n🤖 Robot Configuration:")
    print(f"   Max Velocity: {config.robot_max_velocity:.1f} m/s")
    print(f"   Max Angular Velocity: {config.robot_max_angular_velocity:.1f} rad/s")
    print(f"   Battery Capacity: {config.robot_battery_capacity:.1f} Wh")

    print(f"\n⚡ Event Generation:")
    print(f"   Average Rate: {config.event_generation_rate:.1f} events/episode")
    print(f"   Urgency Range: [{config.event_min_urgency:.1f}, {config.event_max_urgency:.1f}]")

    print(f"\n⏱️  Episode Configuration:")
    print(f"   Max Steps: {config.max_episode_steps}")
    print(f"   Max Time: {config.max_episode_time:.0f}s ({config.max_episode_time/60:.0f} min)")

    print("\n" + "=" * 60)
    print("ℹ️  Map Design Rationale:")
    print("=" * 60)
    print("""
이 맵 구성은 다음 기준으로 설계되었습니다:

1. **테스트 및 데모 목적**
   - 간단한 정사각형 패턴으로 이해하기 쉬움
   - 시각화 및 분석이 용이함
   - 빠른 프로토타이핑 가능

2. **균형잡힌 순찰 커버리지**
   - 4개 코너 포인트가 맵을 균등하게 커버
   - 각 구간 거리가 일정함 (~30m, ~42.4m 대각선)
   - 중앙 영역도 순찰 경로에 포함

3. **현실적인 스케일**
   - 50m × 50m: 중형 건물 층 또는 야외 구역 크기
   - 로봇 속도 1.5m/s: 실제 Go2 로봇 기준
   - 전체 순찰 1회: ~2분 소요

4. **확장 가능성**
   - configs/custom.yaml 생성으로 쉽게 변경 가능
   - 복도형, L자형, 복잡한 경로 등 자유롭게 설정 가능
   - 순찰 포인트 수 제한 없음 (2개 이상)

💡 **커스터마이징 방법:**
   configs/default.yaml 파일을 편집하거나
   새로운 YAML 설정 파일을 생성하세요.

   예: python scripts/train.py --config configs/building_a.yaml
    """)

    print("\n📊 Generating visualization...")

    # Visualize
    save_path = Path(__file__).parent.parent / "outputs" / "map_layout.png"
    save_path.parent.mkdir(exist_ok=True)

    visualize_map(config, str(save_path))


if __name__ == "__main__":
    main()
