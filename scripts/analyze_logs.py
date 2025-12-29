#!/usr/bin/env python3
"""
학습 로그 분석 도구

CSV 파일에서 학습 데이터를 읽어서 분석 및 시각화합니다.

기능:
- 학습 곡선 플롯 (return, success rate, etc.)
- 맵별 성능 비교
- 문제 진단 (entropy collapse, value loss explosion, etc.)
- 요약 리포트 생성

사용법:
    python scripts/analyze_logs.py runs/multi_map_logged/20251230-123456

작성자: Reviewer 박용준
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


def load_logs(log_dir: Path):
    """Load all CSV logs."""
    csv_dir = log_dir / "csv"

    logs = {}

    # Load episode data
    episode_csv = csv_dir / "episodes.csv"
    if episode_csv.exists():
        logs['episodes'] = pd.read_csv(episode_csv)
        print(f"✅ Loaded {len(logs['episodes'])} episodes")

    # Load update data
    update_csv = csv_dir / "updates.csv"
    if update_csv.exists():
        logs['updates'] = pd.read_csv(update_csv)
        print(f"✅ Loaded {len(logs['updates'])} updates")

    # Load step data (may be large)
    step_csv = csv_dir / "steps.csv"
    if step_csv.exists():
        # Sample every 100 steps for efficiency
        logs['steps'] = pd.read_csv(step_csv)
        print(f"✅ Loaded {len(logs['steps'])} steps")

    # Load map-specific data
    logs['maps'] = {}
    for map_csv in csv_dir.glob("map_*.csv"):
        map_name = map_csv.stem.replace("map_", "")
        logs['maps'][map_name] = pd.read_csv(map_csv)
        print(f"✅ Loaded map: {map_name} ({len(logs['maps'][map_name])} episodes)")

    return logs


def plot_learning_curves(logs, output_dir: Path):
    """Plot main learning curves."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'episodes' not in logs:
        print("⚠️ No episode data found")
        return

    df = logs['episodes']

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Learning Curves', fontsize=16)

    # 1. Episode Return
    axes[0, 0].plot(df['global_step'], df['return'], alpha=0.3)
    axes[0, 0].plot(df['global_step'], df['return'].rolling(window=50).mean(), linewidth=2)
    axes[0, 0].set_title('Episode Return')
    axes[0, 0].set_xlabel('Global Step')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Event Success Rate
    axes[0, 1].plot(df['global_step'], df['event_success_rate'] * 100, alpha=0.3)
    axes[0, 1].plot(df['global_step'], (df['event_success_rate'] * 100).rolling(window=50).mean(), linewidth=2)
    axes[0, 1].set_title('Event Success Rate')
    axes[0, 1].set_xlabel('Global Step')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=75, color='g', linestyle='--', alpha=0.5, label='Target 75%')
    axes[0, 1].legend()

    # 3. Patrol Coverage
    axes[1, 0].plot(df['global_step'], df['patrol_coverage'] * 100, alpha=0.3)
    axes[1, 0].plot(df['global_step'], (df['patrol_coverage'] * 100).rolling(window=50).mean(), linewidth=2)
    axes[1, 0].set_title('Patrol Coverage')
    axes[1, 0].set_xlabel('Global Step')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Target 80%')
    axes[1, 0].legend()

    # 4. Action Distribution
    axes[1, 1].plot(df['global_step'], df['patrol_ratio'] * 100, label='Patrol', alpha=0.7)
    axes[1, 1].plot(df['global_step'], df['dispatch_ratio'] * 100, label='Dispatch', alpha=0.7)
    axes[1, 1].set_title('Action Distribution')
    axes[1, 1].set_xlabel('Global Step')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Episode Length
    axes[2, 0].plot(df['global_step'], df['length'], alpha=0.3)
    axes[2, 0].plot(df['global_step'], df['length'].rolling(window=50).mean(), linewidth=2)
    axes[2, 0].set_title('Episode Length')
    axes[2, 0].set_xlabel('Global Step')
    axes[2, 0].set_ylabel('Steps')
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Safety Violations
    axes[2, 1].plot(df['global_step'], df['safety_violations'].rolling(window=50).mean(), linewidth=2)
    axes[2, 1].set_title('Safety Violations (50-ep avg)')
    axes[2, 1].set_xlabel('Global Step')
    axes[2, 1].set_ylabel('Violations per Episode')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'learning_curves.png'}")


def plot_training_diagnostics(logs, output_dir: Path):
    """Plot training diagnostics (PPO-specific metrics)."""
    if 'updates' not in logs:
        print("⚠️ No update data found")
        return

    df = logs['updates']

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Training Diagnostics', fontsize=16)

    # 1. Entropy
    axes[0, 0].plot(df['global_step'], df['entropy'], linewidth=2)
    axes[0, 0].set_title('Entropy (탐색 지표)')
    axes[0, 0].set_xlabel('Global Step')
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].axhline(y=0.02, color='r', linestyle='--', alpha=0.5, label='Warning < 0.02')
    axes[0, 0].axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='Healthy > 0.05')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Approx KL
    axes[0, 1].plot(df['global_step'], df['approx_kl'], linewidth=2)
    axes[0, 1].set_title('Approx KL (정책 변화)')
    axes[0, 1].set_xlabel('Global Step')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].axhline(y=0.001, color='r', linestyle='--', alpha=0.5, label='Warning < 0.001')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Value Loss
    axes[1, 0].plot(df['global_step'], df['value_loss'], linewidth=2)
    axes[1, 0].set_title('Value Loss')
    axes[1, 0].set_xlabel('Global Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].axhline(y=100000, color='r', linestyle='--', alpha=0.5, label='Warning > 100K')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Explained Variance
    axes[1, 1].plot(df['global_step'], df['explained_variance'], linewidth=2)
    axes[1, 1].set_title('Explained Variance (Critic 성능)')
    axes[1, 1].set_xlabel('Global Step')
    axes[1, 1].set_ylabel('Explained Variance')
    axes[1, 1].axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Good > 0.5')
    axes[1, 1].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Warning < 0.1')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Policy Loss
    axes[2, 0].plot(df['global_step'], df['policy_loss'], linewidth=2)
    axes[2, 0].set_title('Policy Loss')
    axes[2, 0].set_xlabel('Global Step')
    axes[2, 0].set_ylabel('Loss')
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Clipfrac
    axes[2, 1].plot(df['global_step'], df['clipfrac'], linewidth=2)
    axes[2, 1].set_title('Clip Fraction (PPO clipping)')
    axes[2, 1].set_xlabel('Global Step')
    axes[2, 1].set_ylabel('Clip Fraction')
    axes[2, 1].axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Warning < 0.05')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'training_diagnostics.png'}")


def plot_map_comparison(logs, output_dir: Path):
    """Plot performance comparison across maps."""
    if 'maps' not in logs or len(logs['maps']) == 0:
        print("⚠️ No map data found")
        return

    map_names = list(logs['maps'].keys())
    n_maps = len(map_names)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Per-Map Performance Comparison', fontsize=16)

    # Prepare data
    map_returns = {name: logs['maps'][name]['return'].iloc[-100:].mean() if len(logs['maps'][name]) >= 100 else logs['maps'][name]['return'].mean() for name in map_names}
    map_success = {name: logs['maps'][name]['event_success_rate'].iloc[-100:].mean() * 100 if len(logs['maps'][name]) >= 100 else logs['maps'][name]['event_success_rate'].mean() * 100 for name in map_names}
    map_coverage = {name: logs['maps'][name]['patrol_coverage'].iloc[-100:].mean() * 100 if len(logs['maps'][name]) >= 100 else logs['maps'][name]['patrol_coverage'].mean() * 100 for name in map_names}

    # 1. Return comparison
    axes[0, 0].bar(range(n_maps), [map_returns[name] for name in map_names])
    axes[0, 0].set_title('Average Return (last 100 episodes)')
    axes[0, 0].set_xticks(range(n_maps))
    axes[0, 0].set_xticklabels(map_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Success rate comparison
    axes[0, 1].bar(range(n_maps), [map_success[name] for name in map_names])
    axes[0, 1].set_title('Event Success Rate (last 100 episodes)')
    axes[0, 1].set_xticks(range(n_maps))
    axes[0, 1].set_xticklabels(map_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].axhline(y=75, color='g', linestyle='--', alpha=0.5, label='Target 75%')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Coverage comparison
    axes[1, 0].bar(range(n_maps), [map_coverage[name] for name in map_names])
    axes[1, 0].set_title('Patrol Coverage (last 100 episodes)')
    axes[1, 0].set_xticks(range(n_maps))
    axes[1, 0].set_xticklabels(map_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Target 80%')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Learning curves by map
    for name in map_names:
        df = logs['maps'][name]
        axes[1, 1].plot(df['global_step'], df['return'].rolling(window=20).mean(), label=name, alpha=0.7)
    axes[1, 1].set_title('Return by Map (20-ep avg)')
    axes[1, 1].set_xlabel('Global Step')
    axes[1, 1].set_ylabel('Return')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'map_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'map_comparison.png'}")


def generate_report(logs, output_dir: Path):
    """Generate text report with summary statistics."""
    report_path = output_dir / "training_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("학습 결과 리포트\n")
        f.write("=" * 80 + "\n\n")

        # Load results JSON if available
        results_json = output_dir.parent / "results.json"
        if results_json.exists():
            with open(results_json, 'r') as jf:
                results = json.load(jf)
                f.write(f"학습 시간: {results.get('training_time', 0)/3600:.2f} hours\n")
                f.write(f"총 Steps: {results.get('total_steps', 0):,}\n")
                f.write(f"총 Updates: {results.get('total_updates', 0):,}\n\n")

        # Episode statistics (last 100 episodes)
        if 'episodes' in logs and len(logs['episodes']) > 0:
            df = logs['episodes'].iloc[-100:]

            f.write("=" * 80 + "\n")
            f.write("전체 성능 (최근 100 에피소드)\n")
            f.write("=" * 80 + "\n")
            f.write(f"  평균 Return: {df['return'].mean():.2f} ± {df['return'].std():.2f}\n")
            f.write(f"  Event Success Rate: {df['event_success_rate'].mean()*100:.1f}%\n")
            f.write(f"  Patrol Coverage: {df['patrol_coverage'].mean()*100:.1f}%\n")
            f.write(f"  Safety Violations: {df['safety_violations'].mean():.2f} per episode\n")
            f.write(f"  Patrol Ratio: {df['patrol_ratio'].mean()*100:.1f}%\n")
            f.write(f"  Dispatch Ratio: {df['dispatch_ratio'].mean()*100:.1f}%\n\n")

        # Map-specific statistics
        if 'maps' in logs and len(logs['maps']) > 0:
            f.write("=" * 80 + "\n")
            f.write("맵별 성능 (최근 100 에피소드)\n")
            f.write("=" * 80 + "\n")

            for map_name, df in logs['maps'].items():
                if len(df) > 0:
                    df_recent = df.iloc[-min(100, len(df)):]
                    f.write(f"\n{map_name}:\n")
                    f.write(f"  Episodes: {len(df)}\n")
                    f.write(f"  Return: {df_recent['return'].mean():.2f} ± {df_recent['return'].std():.2f}\n")
                    f.write(f"  Success: {df_recent['event_success_rate'].mean()*100:.1f}%\n")
                    f.write(f"  Coverage: {df_recent['patrol_coverage'].mean()*100:.1f}%\n")

        # Training diagnostics (last 100 updates)
        if 'updates' in logs and len(logs['updates']) > 0:
            df = logs['updates'].iloc[-100:]

            f.write("\n" + "=" * 80 + "\n")
            f.write("학습 상태 진단 (최근 100 updates)\n")
            f.write("=" * 80 + "\n")
            f.write(f"  Entropy: {df['entropy'].mean():.6f} (healthy: > 0.02)\n")
            f.write(f"  Approx KL: {df['approx_kl'].mean():.6f} (healthy: > 0.001)\n")
            f.write(f"  Value Loss: {df['value_loss'].mean():.2f} (healthy: < 100K)\n")
            f.write(f"  Explained Variance: {df['explained_variance'].mean():.4f} (good: > 0.5)\n")
            f.write(f"  Clipfrac: {df['clipfrac'].mean():.4f} (healthy: > 0.05)\n\n")

            # Warnings
            f.write("⚠️  경고 사항:\n")
            warnings = []

            if df['entropy'].mean() < 0.02:
                warnings.append(f"  - Entropy 너무 낮음 ({df['entropy'].mean():.6f} < 0.02) - 탐색 부족")

            if df['approx_kl'].mean() < 0.001:
                warnings.append(f"  - Approx KL 너무 낮음 ({df['approx_kl'].mean():.6f} < 0.001) - 정책 업데이트 약함")

            if df['value_loss'].mean() > 100000:
                warnings.append(f"  - Value Loss 너무 높음 ({df['value_loss'].mean():.0f} > 100K) - Critic 학습 실패")

            if df['explained_variance'].mean() < 0.1:
                warnings.append(f"  - Explained Variance 너무 낮음 ({df['explained_variance'].mean():.4f} < 0.1) - Critic 부정확")

            if df['clipfrac'].mean() < 0.05:
                warnings.append(f"  - Clipfrac 너무 낮음 ({df['clipfrac'].mean():.4f} < 0.05) - PPO 작동 약함")

            if warnings:
                for w in warnings:
                    f.write(w + "\n")
            else:
                f.write("  없음 - 학습 정상 진행 중 ✅\n")

    print(f"✅ Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("log_dir", type=str, help="Path to log directory (e.g., runs/multi_map_logged/20251230-123456)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots (default: log_dir/analysis)")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        return

    output_dir = Path(args.output) if args.output else log_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Analyzing logs from: {log_dir}")
    print(f"📁 Output directory: {output_dir}\n")

    # Load logs
    logs = load_logs(log_dir)

    if not logs:
        print("❌ No logs found!")
        return

    # Generate plots
    print("\n📈 Generating plots...")
    plot_learning_curves(logs, output_dir)
    plot_training_diagnostics(logs, output_dir)
    plot_map_comparison(logs, output_dir)

    # Generate report
    print("\n📝 Generating report...")
    generate_report(logs, output_dir)

    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"\n주요 파일:")
    print(f"  - learning_curves.png - 학습 곡선")
    print(f"  - training_diagnostics.png - PPO 진단")
    print(f"  - map_comparison.png - 맵별 성능 비교")
    print(f"  - training_report.txt - 텍스트 리포트")


if __name__ == "__main__":
    main()
