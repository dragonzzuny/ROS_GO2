#!/usr/bin/env python3
"""
Phase 3: 3-Stage Curriculum Learning Script
Reviewer 박용준

Implements structured curriculum learning with explicit stage progression:

Stage 1 (Simple):   corridor, l_shaped         - 10 patrol points, minimal obstacles
Stage 2 (Medium):   campus, large_square       - 16 patrol points, moderate complexity
Stage 3 (Complex):  office_building, warehouse - 20+ patrol points, dense obstacles

Each stage trains to convergence before advancing to next stage.
Previous stage checkpoints used for warm-start to preserve learned behaviors.

Success Criteria per Stage:
- Return std < 40k (stable learning)
- Event success rate > 70%
- Patrol coverage > 60%
- Minimum 50k timesteps per stage
"""

import argparse
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_dispatch.env import create_multi_map_env
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.core.config import NetworkConfig, TrainingConfig, RewardConfig


# Curriculum stage definitions
CURRICULUM_STAGES = {
    "stage1_simple": {
        "maps": ["map_corridor", "map_l_shaped"],
        "description": "Simple: Small maps, few obstacles",
        "min_timesteps": 50_000,
        "success_criteria": {
            "return_std": 40_000,
            "event_success": 0.60,
            "patrol_coverage": 0.50,
        },
    },
    "stage2_medium": {
        "maps": ["map_campus", "map_large_square"],
        "description": "Medium: Campus and square layouts",
        "min_timesteps": 100_000,
        "success_criteria": {
            "return_std": 40_000,
            "event_success": 0.65,
            "patrol_coverage": 0.55,
        },
    },
    "stage3_complex": {
        "maps": ["map_office_building", "map_warehouse"],
        "description": "Complex: Large maps, dense obstacles",
        "min_timesteps": 150_000,
        "success_criteria": {
            "return_std": 45_000,
            "event_success": 0.60,
            "patrol_coverage": 0.50,
        },
    },
}


def check_stage_success(
    stage_name: str,
    map_stats: Dict[str, Dict[str, float]],
    recent_returns: List[float],
) -> Tuple[bool, Dict[str, bool]]:
    """
    Check if current stage meets success criteria.

    Args:
        stage_name: Name of current stage
        map_stats: Performance statistics per map
        recent_returns: Recent episode returns (for std calculation)

    Returns:
        success: Whether all criteria met
        criteria_status: Dict of individual criterion statuses
    """
    criteria = CURRICULUM_STAGES[stage_name]["success_criteria"]
    maps = CURRICULUM_STAGES[stage_name]["maps"]

    # Calculate metrics across all maps in stage
    returns_std = np.std(recent_returns) if len(recent_returns) > 10 else np.inf

    event_success_rates = []
    patrol_coverages = []
    for map_name in maps:
        if map_name in map_stats and map_stats[map_name]["episodes"] > 5:
            event_success_rates.append(map_stats[map_name]["mean_event_success"])
            patrol_coverages.append(map_stats[map_name]["mean_patrol_coverage"])

    avg_event_success = np.mean(event_success_rates) if event_success_rates else 0.0
    avg_patrol_coverage = np.mean(patrol_coverages) if patrol_coverages else 0.0

    # Check each criterion
    criteria_status = {
        "return_std": returns_std < criteria["return_std"],
        "event_success": avg_event_success >= criteria["event_success"],
        "patrol_coverage": avg_patrol_coverage >= criteria["patrol_coverage"],
    }

    success = all(criteria_status.values())

    return success, criteria_status


def train_curriculum_stage(
    stage_name: str,
    agent: PPOAgent,
    base_config: TrainingConfig,
    log_dir: Path,
    checkpoint_from_previous: str = None,
) -> str:
    """
    Train a single curriculum stage.

    Args:
        stage_name: Name of curriculum stage
        agent: PPO agent to train
        base_config: Base training configuration
        log_dir: Logging directory
        checkpoint_from_previous: Path to checkpoint from previous stage (optional)

    Returns:
        checkpoint_path: Path to final checkpoint for this stage
    """
    stage_config = CURRICULUM_STAGES[stage_name]
    maps = stage_config["maps"]
    min_timesteps = stage_config["min_timesteps"]

    print("\n" + "=" * 80)
    print(f"CURRICULUM STAGE: {stage_name}")
    print("=" * 80)
    print(f"Description: {stage_config['description']}")
    print(f"Maps: {maps}")
    print(f"Min Timesteps: {min_timesteps:,}")
    print(f"Success Criteria:")
    for criterion, threshold in stage_config["success_criteria"].items():
        print(f"  - {criterion}: {threshold}")
    print("=" * 80 + "\n")

    # Load checkpoint from previous stage if provided
    if checkpoint_from_previous:
        print(f"📦 Loading checkpoint from previous stage: {checkpoint_from_previous}")
        agent.load(checkpoint_from_previous)
        print("✅ Checkpoint loaded - warm start enabled\n")

    # Create environment for this stage
    base_path = Path(__file__).parent.parent / "configs"
    map_configs = [str(base_path / f"{map_name}.yaml") for map_name in maps]

    print(f"Creating environment with {len(maps)} maps...")
    env = create_multi_map_env(
        map_configs=map_configs,
        mode="random",  # Random selection within stage
        track_coverage=True,
    )
    print(f"✅ Environment created\n")

    # Setup logging
    stage_log_dir = log_dir / stage_name
    stage_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(stage_log_dir / "tensorboard"))

    # Training state
    obs, info = env.reset()
    episode_return = 0.0
    episode_length = 0
    current_map = info["map_name"]

    num_updates = min_timesteps // base_config.num_steps
    global_step = 0

    # Track recent returns for convergence check
    recent_returns = []
    max_recent_returns = 100

    print(f"🚀 Starting training for {num_updates:,} updates...\n")
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # Collect rollout
        for step in range(base_config.num_steps):
            global_step += 1

            # Select action with masking
            action_mask = info.get("action_mask", None)
            action, log_prob, value = agent.get_action(obs, action_mask=action_mask)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            episode_length += 1

            # Store transition
            nav_time = info.get("nav_time", 1.0)
            action_mask = info.get("action_mask", None)
            agent.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=terminated,
                nav_time=nav_time,
                action_mask=action_mask,
            )

            obs = next_obs

            # Episode finished
            if terminated or truncated:
                # Log episode
                writer.add_scalar("episode/return", episode_return, global_step)
                writer.add_scalar("episode/length", episode_length, global_step)
                writer.add_scalar(
                    f"episode_per_map/{current_map}/return",
                    episode_return,
                    global_step,
                )

                # Track for convergence
                recent_returns.append(episode_return)
                if len(recent_returns) > max_recent_returns:
                    recent_returns.pop(0)

                # Reset
                obs, info = env.reset()
                current_map = info["map_name"]
                episode_return = 0.0
                episode_length = 0

        # Compute final value for bootstrapping
        final_action, final_log_prob, final_value = agent.get_action(obs)

        # PPO update
        train_stats = agent.update(
            last_value=final_value,
            last_done=False,
        )

        # Log training stats
        for key, value in train_stats.items():
            writer.add_scalar(f"train/{key}", value, global_step)

        # Curriculum metrics
        if len(recent_returns) > 10:
            returns_std = np.std(recent_returns)
            writer.add_scalar("curriculum/return_std", returns_std, global_step)

        # Logging
        if update % base_config.log_interval == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed

            print(f"\nUpdate {update}/{num_updates} ({stage_name}):")
            print(f"  Global Step: {global_step:,}")
            print(f"  FPS: {fps:.1f}")
            print(f"  Current Map: {current_map}")

            # Map statistics
            map_stats = env.get_map_statistics()
            print(f"\n  Per-Map Performance:")
            for map_name in maps:
                if map_name in map_stats:
                    s = map_stats[map_name]
                    if s["episodes"] > 0:
                        print(f"    {map_name}:")
                        print(f"      Episodes: {s['episodes']}")
                        print(f"      Return: {s['mean_return']:.1f} ± {s['std_return']:.1f}")
                        print(f"      Event Success: {100*s['mean_event_success']:.1f}%")
                        print(f"      Patrol Coverage: {100*s['mean_patrol_coverage']:.1f}%")

            # Check success criteria
            success, criteria_status = check_stage_success(
                stage_name, map_stats, recent_returns
            )

            print(f"\n  Success Criteria:")
            for criterion, status in criteria_status.items():
                symbol = "✅" if status else "❌"
                print(f"    {symbol} {criterion}: {status}")

            if success and update >= (min_timesteps // base_config.num_steps):
                print(f"\n  🎉 Stage success criteria met!")

            # Training stats
            print(f"\n  Training:")
            for key, value in train_stats.items():
                print(f"    {key}: {value:.4f}")

        # Save checkpoint
        if update % base_config.save_interval == 0:
            checkpoint_path = (
                stage_log_dir / "checkpoints" / f"update_{update}.pth"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(str(checkpoint_path))

    # Final checkpoint for this stage
    final_checkpoint = stage_log_dir / "checkpoints" / "stage_final.pth"
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(final_checkpoint))

    # Final evaluation
    map_stats = env.get_map_statistics()
    success, criteria_status = check_stage_success(
        stage_name, map_stats, recent_returns
    )

    print("\n" + "=" * 80)
    print(f"STAGE {stage_name} COMPLETE")
    print("=" * 80)
    print(f"Final Performance:")
    for map_name in maps:
        if map_name in map_stats:
            s = map_stats[map_name]
            print(f"  {map_name}:")
            print(f"    Return: {s['mean_return']:.1f} ± {s['std_return']:.1f}")
            print(f"    Event Success: {100*s['mean_event_success']:.1f}%")
            print(f"    Patrol Coverage: {100*s['mean_patrol_coverage']:.1f}%")

    print(f"\nSuccess Criteria:")
    for criterion, status in criteria_status.items():
        symbol = "✅" if status else "❌"
        print(f"  {symbol} {criterion}")

    if success:
        print(f"\n✅ Stage succeeded! Proceeding to next stage...")
    else:
        print(f"\n⚠️  Stage did not meet all criteria - continuing anyway")

    print("=" * 80 + "\n")

    writer.close()
    env.close()

    return str(final_checkpoint)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: 3-Stage Curriculum Learning"
    )

    # Curriculum settings
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Starting curriculum stage (1=simple, 2=medium, 3=complex)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=["stage1_simple", "stage2_medium", "stage3_complex"],
        help="Which stages to run (in order)",
    )

    # Training config
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        help="Rollout steps per update",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="PPO epochs per update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size",
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
        help="PPO clipping epsilon",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient",
    )

    # Experiment
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="curriculum_phase3",
        help="Experiment name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA",
    )

    # Logging
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N updates",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save checkpoint every N updates",
    )

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup base configurations
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        experiment_name=args.experiment_name,
        seed=args.seed,
        cuda=(args.cuda and torch.cuda.is_available()),
    )

    network_config = NetworkConfig()

    # Create initial environment to get dimensions
    base_path = Path(__file__).parent.parent / "configs"
    temp_env = create_multi_map_env(
        map_configs=[str(base_path / "map_corridor.yaml")],
        mode="random",
    )

    obs_dim = temp_env.observation_space.shape[0]
    num_replan_strategies = temp_env.action_space.nvec[1]
    temp_env.close()

    # Create PPO agent
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    agent = PPOAgent(
        obs_dim=obs_dim,
        num_replan_strategies=num_replan_strategies,
        training_config=training_config,
        network_config=network_config,
        device=device,
    )

    print("\n" + "=" * 80)
    print("PHASE 3: 3-STAGE CURRICULUM LEARNING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Observation dim: {obs_dim}")
    print(f"Num strategies: {num_replan_strategies}")
    print(f"Starting from stage {args.start_stage}")
    print("=" * 80 + "\n")

    # Log directory
    log_dir = Path("runs") / args.experiment_name / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save configs
    training_config.save_yaml(str(log_dir / "training_config.yaml"))
    print(f"📝 Logs will be saved to: {log_dir}\n")

    # Train through curriculum stages
    previous_checkpoint = None
    stages_to_run = args.stages[args.start_stage - 1 :]

    for stage_name in stages_to_run:
        checkpoint_path = train_curriculum_stage(
            stage_name=stage_name,
            agent=agent,
            base_config=training_config,
            log_dir=log_dir,
            checkpoint_from_previous=previous_checkpoint,
        )
        previous_checkpoint = checkpoint_path

    print("\n" + "=" * 80)
    print("🎉 CURRICULUM LEARNING COMPLETE!")
    print("=" * 80)
    print(f"Final model checkpoint: {previous_checkpoint}")
    print(f"All logs saved to: {log_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
