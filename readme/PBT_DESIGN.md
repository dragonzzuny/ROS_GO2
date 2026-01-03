# Population-based Training (PBT) Design

## Overview

Population-based Training (PBT) combines parallel search with online hyperparameter optimization. Multiple agents train simultaneously with different hyperparameters, and periodically poor performers adopt the weights and hyperparameters of top performers with small perturbations.

**Reference**: [Population Based Training of Neural Networks (DeepMind, 2017)](https://arxiv.org/abs/1711.09846)

## Architecture

### 1. PopulationMember

Each member of the population is an agent with its own:
- PPO network weights
- Hyperparameters (learning rate, batch size, etc.)
- Training environment
- Performance metrics

```python
@dataclass
class PopulationMember:
    member_id: int
    agent: PPOAgent
    hyperparams: Dict[str, Any]
    recent_returns: List[float]
    total_timesteps: int

    @property
    def performance(self) -> float:
        """Average return over recent episodes."""
        return np.mean(self.recent_returns[-20:]) if self.recent_returns else -np.inf
```

### 2. Hyperparameter Search Space

Hyperparameters to optimize:

| Parameter | Type | Range | Initial Distribution |
|-----------|------|-------|---------------------|
| learning_rate | log-uniform | [1e-5, 1e-3] | Random |
| batch_size | discrete | [64, 128, 256, 512] | Random |
| num_epochs | discrete | [5, 10, 15, 20] | Random |
| clip_range | uniform | [0.05, 0.3] | Random |
| entropy_coef | log-uniform | [1e-4, 0.05] | Random |

**Fixed parameters:**
- gamma = 0.99 (discount factor)
- gae_lambda = 0.95
- num_steps = 2048 (rollout length)

### 3. PBT Manager

Manages the population and orchestrates exploit/explore:

```python
class PBTManager:
    def __init__(
        self,
        population_size: int = 8,
        eval_interval: int = 10_000,  # Timesteps between evaluations
        exploit_threshold: float = 0.25,  # Bottom 25% exploit top 25%
        perturb_factor: float = 0.2,  # ±20% perturbation
    ):
        self.population: List[PopulationMember] = []
        self.eval_interval = eval_interval
        self.exploit_threshold = exploit_threshold
        self.perturb_factor = perturb_factor

    def exploit_and_explore(self) -> None:
        """Replace poor performers with perturbed top performers."""
        # Sort by performance
        ranked = sorted(self.population, key=lambda m: m.performance, reverse=True)

        # Top 25% and bottom 25%
        cutoff = int(len(ranked) * self.exploit_threshold)
        top_performers = ranked[:cutoff]
        bottom_performers = ranked[-cutoff:]

        # Bottom performers adopt top performers
        for bottom in bottom_performers:
            top = np.random.choice(top_performers)

            # Copy weights
            bottom.agent.network.load_state_dict(
                top.agent.network.state_dict()
            )

            # Copy and perturb hyperparameters
            bottom.hyperparams = self._perturb(top.hyperparams.copy())

            # Apply new hyperparameters
            self._apply_hyperparams(bottom)

    def _perturb(self, hyperparams: Dict) -> Dict:
        """Perturb hyperparameters with ±20% noise."""
        perturbed = hyperparams.copy()

        # Learning rate: multiply by random factor in [0.8, 1.2]
        if np.random.rand() < 0.5:
            factor = 0.8 if np.random.rand() < 0.5 else 1.2
            perturbed['learning_rate'] *= factor
            perturbed['learning_rate'] = np.clip(
                perturbed['learning_rate'], 1e-5, 1e-3
            )

        # Batch size: shift up or down
        if np.random.rand() < 0.5:
            sizes = [64, 128, 256, 512]
            current_idx = sizes.index(perturbed['batch_size'])
            shift = np.random.choice([-1, 1])
            new_idx = np.clip(current_idx + shift, 0, len(sizes) - 1)
            perturbed['batch_size'] = sizes[new_idx]

        # Similar for other hyperparams...

        return perturbed
```

### 4. Training Loop

```python
def train_pbt(
    population_size: int = 8,
    total_timesteps: int = 500_000,
    stage: str = "stage1_simple",
):
    # Initialize population with random hyperparameters
    pbt_manager = PBTManager(population_size=population_size)

    for member_id in range(population_size):
        hyperparams = sample_hyperparams()
        agent = create_agent(hyperparams)
        member = PopulationMember(member_id, agent, hyperparams)
        pbt_manager.add_member(member)

    # Training loop
    while pbt_manager.total_timesteps < total_timesteps:
        # Each member trains for eval_interval steps
        for member in pbt_manager.population:
            train_single_member(
                member,
                steps=pbt_manager.eval_interval
            )

        # Exploit and explore
        pbt_manager.exploit_and_explore()

        # Log best hyperparameters
        best = pbt_manager.get_best_member()
        log_hyperparams(best.hyperparams, best.performance)

    # Return best member
    return pbt_manager.get_best_member()
```

## Implementation Plan

### Phase 1: Core Components
1. `PopulationMember` dataclass
2. `PBTManager` class with exploit/explore logic
3. Hyperparameter sampling and perturbation

### Phase 2: Training Script
4. `scripts/train_pbt.py` main script
5. Single-member training loop
6. Population-level coordination

### Phase 3: Integration
7. TensorBoard logging for all members
8. Checkpoint saving (best member + full population)
9. Hyperparameter trajectory visualization

### Phase 4: Curriculum Integration
10. Apply PBT to 3-stage curriculum
11. Reset hyperparameters between stages vs. continuous optimization

## Expected Benefits

1. **Automatic Hyperparameter Tuning**: No manual grid search needed
2. **Online Adaptation**: Hyperparameters evolve during training
3. **Efficient Exploration**: Exploit good configurations while exploring variations
4. **Better Final Performance**: Expected 10-20% improvement over fixed hyperparams

## Configuration

### Population Size
- **Small (4 agents)**: Faster, less diversity
- **Medium (8 agents)**: Balanced (recommended)
- **Large (16 agents)**: More diversity, slower

### Evaluation Interval
- **Short (5k steps)**: Frequent adaptation, noisy signals
- **Medium (10k steps)**: Balanced (recommended)
- **Long (20k steps)**: Stable signals, slower adaptation

### Exploit Threshold
- **Conservative (15%)**: Preserve diversity
- **Balanced (25%)**: Recommended
- **Aggressive (40%)**: Fast convergence to good hyperparams

## Logging and Analysis

For each member, log:
- `pbt/member_{id}/performance`: Recent return average
- `pbt/member_{id}/learning_rate`: Current LR
- `pbt/member_{id}/batch_size`: Current batch size
- `pbt/member_{id}/exploit_events`: Number of times exploited

For population:
- `pbt/population/best_performance`: Best member performance
- `pbt/population/mean_performance`: Population average
- `pbt/population/hyperparam_diversity`: Diversity metric

## Files to Create

1. `src/rl_dispatch/pbt/population_member.py`
2. `src/rl_dispatch/pbt/pbt_manager.py`
3. `src/rl_dispatch/pbt/hyperparameter_space.py`
4. `scripts/train_pbt.py`
5. `scripts/analyze_pbt_results.py`

## Testing

Test script: `test_pbt_implementation.py`
- Test hyperparameter perturbation
- Test exploit logic (top → bottom copy)
- Test population training for 100 steps
- Verify logging works

## Estimated Timeline

- Implementation: 2-3 hours
- Testing: 30 minutes
- Initial PBT run (Stage 1): 4-6 hours
- Full curriculum (3 stages): 12-18 hours

## Alternative: Simplified PBT

If full PBT is too complex, we can implement a simpler version:
- Fixed population (no dynamic creation)
- Simpler perturbation (only LR and batch size)
- Less frequent evaluation (every 20k steps)

This would be 50% less implementation effort with 80% of the benefits.
