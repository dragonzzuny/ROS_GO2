#!/usr/bin/env python3
"""
Population-based Training (PBT) for RL Dispatch

Trains multiple agents in parallel with different hyperparameters.
Periodically, poor performers adopt weights and hyperparameters from
top performers with perturbations, enabling online hyperparameter optimization.

Usage:
    python scripts/train_pbt.py --population-size 8 --total-timesteps 500000
"""

import argparse
import time
from pathlib import Path
import sys
import numpy as np
import torch
from typing import List, Dict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_dispatch.env import create_multi_map_env
from rl_dispatch.algorithms import PPOAgent
from rl_dispatch.core.config import NetworkConfig, TrainingConfig, RewardConfig
from rl_dispatch.pbt import PopulationMember, HyperparameterSpace, PBTManager


def create_agent_with_hyperparams(
    obs_dim: int,
    num_replan_strategies: int,
    hyperparams: Dict,
    device: str,
) -> PPOAgent:
    """
    Create PPO agent with specific hyperparameters.

    Args:
        obs_dim: Observation dimension
        num_replan_strategies: Number of replan strategies
        hyperparams: Hyperparameters dict
        device: Device to use

    Returns:
        Configured PPO agent
    """
    # Create training config with hyperparameters
    # Calculate num_minibatches from batch_size
    # batch_size = num_steps // num_minibatches
    num_steps = 2048
    num_minibatches = num_steps // hyperparams['batch_size']

    training_config = TrainingConfig(
        learning_rate=hyperparams['learning_rate'],
        num_steps=num_steps,
        num_epochs=hyperparams['num_epochs'],
        num_minibatches=num_minibatches,
        clip_epsilon=hyperparams['clip_range'],  # Note: PBT calls it clip_range, config calls it clip_epsilon
        entropy_coef=hyperparams['entropy_coef'],
        gamma=0.99,  # Fixed
        gae_lambda=0.95,  # Fixed
        log_interval=10,
        save_interval=100,
        experiment_name="pbt",
        seed=42,
        cuda=False,  # Will be set by device
    )

    network_config = NetworkConfig()

    agent = PPOAgent(
        obs_dim=obs_dim,
        num_replan_strategies=num_replan_strategies,
        training_config=training_config,
        network_config=network_config,
        device=device,
    )

    return agent


def train_single_member(
    member: PopulationMember,
    env_configs: List[str],
    map_mode: str,
    num_steps: int,
    log_prefix: str = "",
    show_progress: bool = True,
) -> Dict:
    """
    Train a single population member for specified steps.

    Args:
        member: PopulationMember to train
        env_configs: List of map config paths
        map_mode: Map selection mode ("random", "sequential")
        num_steps: Number of environment steps to train
        log_prefix: Prefix for logging
        show_progress: Show progress bar

    Returns:
        Training statistics dict
    """
    # Create environment
    env = create_multi_map_env(
        map_configs=env_configs,
        mode=map_mode,
    )

    obs, info = env.reset()
    episode_return = 0.0
    episode_length = 0
    episode_returns = []

    steps_trained = 0
    updates_made = 0

    # Progress bar
    pbar = tqdm(
        total=num_steps,
        desc=f"Member {member.member_id}",
        unit="steps",
        disable=not show_progress,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Return: {postfix}',
    )

    # Track last done flag for update
    last_done = False

    while steps_trained < num_steps:
        # Collect rollout
        member.agent.buffer.reset()  # FIX 1: reset() returns None

        for step in range(member.agent.training_config.num_steps):
            # Select action
            action, log_prob, value = member.agent.get_action(obs)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            member.agent.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                nav_time=info.get('nav_time', 1.0),  # FIX 4: Add SMDP nav_time
                action_mask=info.get('action_mask', None),  # FIX 4: Add action_mask
            )

            obs = next_obs
            episode_return += reward
            episode_length += 1
            steps_trained += 1
            last_done = done  # Track last done flag

            # Update progress bar
            pbar.update(1)
            if episode_returns:
                pbar.set_postfix_str(f"{np.mean(episode_returns[-10:]):.1f}")

            if done:
                # Episode finished
                episode_returns.append(episode_return)
                member.add_return(episode_return)

                obs, info = env.reset()
                episode_return = 0.0
                episode_length = 0

            if steps_trained >= num_steps:
                break

        # Update policy (when buffer is full or at end of training)
        if member.agent.buffer.is_full or len(member.agent.buffer) >= member.agent.buffer.buffer_size:  # FIX 2: .full → .is_full
            # Compute next value for GAE
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(member.agent.device)
                next_value = member.agent.network.get_value(obs_t).cpu().item()

            # Train
            train_stats = member.agent.update(last_value=next_value, last_done=last_done)  # FIX 3: train_step() → update()
            updates_made += 1

    pbar.close()

    # Update member timesteps
    member.total_timesteps += steps_trained

    env.close()

    return {
        "steps_trained": steps_trained,
        "updates_made": updates_made,
        "episodes_completed": len(episode_returns),
        "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
    }


def initialize_population(
    population_size: int,
    obs_dim: int,
    num_replan_strategies: int,
    device: str,
) -> List[PopulationMember]:
    """
    Initialize population with random hyperparameters.

    Args:
        population_size: Number of agents
        obs_dim: Observation dimension
        num_replan_strategies: Number of replan strategies
        device: Device to use

    Returns:
        List of PopulationMembers
    """
    print("\n" + "=" * 80)
    print("INITIALIZING POPULATION")
    print("=" * 80)

    population = []

    for member_id in range(population_size):
        # Sample random hyperparameters
        hyperparams = HyperparameterSpace.sample_random()

        print(f"\nMember {member_id}:")
        print(f"  learning_rate:  {hyperparams['learning_rate']:.2e}")
        print(f"  batch_size:     {hyperparams['batch_size']}")
        print(f"  num_epochs:     {hyperparams['num_epochs']}")
        print(f"  clip_range:     {hyperparams['clip_range']:.3f}")
        print(f"  entropy_coef:   {hyperparams['entropy_coef']:.4f}")

        # Create agent
        agent = create_agent_with_hyperparams(
            obs_dim=obs_dim,
            num_replan_strategies=num_replan_strategies,
            hyperparams=hyperparams,
            device=device,
        )

        # Create member
        member = PopulationMember(
            member_id=member_id,
            agent=agent,
            hyperparams=hyperparams,
        )

        # Apply hyperparameters
        member.apply_hyperparams()

        population.append(member)

    print("\n" + "=" * 80)
    print(f"✅ Initialized {population_size} members")
    print("=" * 80 + "\n")

    return population


def train_pbt(
    population_size: int,
    total_timesteps: int,
    eval_interval: int,
    exploit_threshold: float,
    env_configs: List[str],
    map_mode: str,
    log_dir: Path,
    device: str,
    obs_dim: int,
    num_replan_strategies: int,
) -> PBTManager:
    """
    Main PBT training loop.

    Args:
        population_size: Number of agents in population
        total_timesteps: Total timesteps to train
        eval_interval: Timesteps between exploit/explore
        exploit_threshold: Fraction for top/bottom performers
        env_configs: List of map config paths
        map_mode: Map selection mode
        log_dir: Logging directory
        device: Device to use
        obs_dim: Observation dimension
        num_replan_strategies: Number of replan strategies

    Returns:
        PBTManager with trained population
    """
    # Initialize PBT manager
    pbt_manager = PBTManager(
        population_size=population_size,
        eval_interval=eval_interval,
        exploit_threshold=exploit_threshold,
        log_dir=log_dir,
    )

    # Initialize population
    population = initialize_population(
        population_size=population_size,
        obs_dim=obs_dim,
        num_replan_strategies=num_replan_strategies,
        device=device,
    )

    for member in population:
        pbt_manager.add_member(member)

    # Print hyperparameter space
    HyperparameterSpace.print_space()

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING PBT TRAINING")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Eval interval: {eval_interval:,}")
    print(f"Population size: {population_size}")
    print("=" * 80 + "\n")

    timesteps_since_exploit = 0
    global_timesteps = 0
    start_time = time.time()

    # Overall progress bar
    overall_pbar = tqdm(
        total=total_timesteps,
        desc="Overall Progress",
        unit="steps",
        position=0,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
    )

    while global_timesteps < total_timesteps:
        # Train each member
        print(f"\n{'─' * 80}")
        print(f"🎯 Global: {global_timesteps:,}/{total_timesteps:,} ({100*global_timesteps/total_timesteps:.1f}%) | "
              f"⏱️  Elapsed: {time.time()-start_time:.0f}s | "
              f"⚡ Next exploit: {eval_interval-timesteps_since_exploit:,} steps")
        print(f"{'─' * 80}\n")

        for member in pbt_manager.population:
            # Determine steps to train (up to eval_interval)
            steps_to_train = min(
                eval_interval - timesteps_since_exploit,
                total_timesteps - global_timesteps,
            )

            if steps_to_train <= 0:
                break

            # Train member
            stats = train_single_member(
                member=member,
                env_configs=env_configs,
                map_mode=map_mode,
                num_steps=steps_to_train,
                log_prefix=f"member_{member.member_id}",
                show_progress=True,
            )

            # Update overall progress
            overall_pbar.update(steps_to_train)

            # Print summary
            print(f"  ✅ Member {member.member_id}: {stats['episodes_completed']} eps, "
                  f"mean return: {stats['mean_return']:.1f}, "
                  f"performance: {member.performance:.1f}\n")

        # Update counters
        global_timesteps += steps_to_train
        timesteps_since_exploit += steps_to_train
        pbt_manager.total_timesteps = global_timesteps

        # Log population metrics
        pbt_manager.log_population_metrics(global_timesteps)

        # Print status
        pbt_manager.print_status()

        # Check if time for exploit/explore
        if pbt_manager.should_exploit(timesteps_since_exploit):
            print(f"\n⚡ Triggering exploit/explore cycle...")

            # Exploit and explore
            exploit_info = pbt_manager.exploit_and_explore()

            # Reset counter
            timesteps_since_exploit = 0

            # Save population checkpoint
            checkpoint_dir = log_dir / "pbt_checkpoints" / f"cycle_{pbt_manager.exploit_cycle_count}"
            pbt_manager.save_population(checkpoint_dir)

    # Close progress bar
    overall_pbar.close()

    # Final status
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Timesteps/sec: {total_timesteps/total_time:.1f}")
    print("=" * 80)
    pbt_manager.print_status()

    # Save final population
    final_dir = log_dir / "pbt_checkpoints" / "final"
    pbt_manager.save_population(final_dir)

    return pbt_manager


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Population-based Training for RL Dispatch"
    )

    # PBT config
    parser.add_argument(
        "--population-size",
        type=int,
        default=8,
        help="Number of agents in population",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10_000,
        help="Timesteps between exploit/explore cycles",
    )
    parser.add_argument(
        "--exploit-threshold",
        type=float,
        default=0.25,
        help="Fraction of top/bottom performers (0.25 = top/bottom 25%%)",
    )

    # Training config
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total timesteps to train",
    )
    parser.add_argument(
        "--maps",
        type=str,
        nargs="+",
        default=["map_corridor", "map_l_shaped"],
        help="Maps to train on",
    )
    parser.add_argument(
        "--map-mode",
        type=str,
        default="random",
        choices=["random", "sequential"],
        help="Map selection mode",
    )

    # Experiment
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="pbt_phase4",
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

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get map configs
    base_path = Path(__file__).parent.parent / "configs"
    env_configs = [str(base_path / f"{map_name}.yaml") for map_name in args.maps]

    # Create temp environment to get dimensions
    temp_env = create_multi_map_env(
        map_configs=env_configs,
        mode=args.map_mode,
    )
    obs_dim = temp_env.observation_space.shape[0]
    num_replan_strategies = temp_env.action_space.nvec[1]
    temp_env.close()

    print("\n" + "=" * 80)
    print("POPULATION-BASED TRAINING (PBT)")
    print("=" * 80)
    print(f"Observation dim: {obs_dim}")
    print(f"Num strategies: {num_replan_strategies}")
    print(f"Maps: {args.maps}")
    print(f"Population size: {args.population_size}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Eval interval: {args.eval_interval:,}")
    print("=" * 80 + "\n")

    # Log directory
    log_dir = Path("runs") / args.experiment_name / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"📝 Logs will be saved to: {log_dir}\n")

    # Device
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}\n")

    # Train
    pbt_manager = train_pbt(
        population_size=args.population_size,
        total_timesteps=args.total_timesteps,
        eval_interval=args.eval_interval,
        exploit_threshold=args.exploit_threshold,
        env_configs=env_configs,
        map_mode=args.map_mode,
        log_dir=log_dir,
        device=device,
        obs_dim=obs_dim,
        num_replan_strategies=num_replan_strategies,
    )

    # Get best member
    best = pbt_manager.get_best_member()
    print(f"\n{'=' * 80}")
    print(f"🏆 BEST MEMBER: {best.member_id}")
    print(f"{'=' * 80}")
    print(f"Performance: {best.performance:.1f}")
    print(f"Hyperparameters:")
    print(f"  learning_rate:  {best.hyperparams['learning_rate']:.2e}")
    print(f"  batch_size:     {best.hyperparams['batch_size']}")
    print(f"  num_epochs:     {best.hyperparams['num_epochs']}")
    print(f"  clip_range:     {best.hyperparams['clip_range']:.3f}")
    print(f"  entropy_coef:   {best.hyperparams['entropy_coef']:.4f}")
    print(f"{'=' * 80}\n")

    # Close
    pbt_manager.close()


if __name__ == "__main__":
    main()
