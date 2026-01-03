"""
Hyperparameter space definition and sampling for PBT.

Defines the search space for 5 hyperparameters:
- learning_rate: [1e-5, 1e-3] (log-uniform)
- batch_size: [64, 128, 256, 512] (discrete)
- num_epochs: [5, 10, 15, 20] (discrete)
- clip_range: [0.05, 0.3] (uniform)
- entropy_coef: [1e-4, 0.05] (log-uniform)
"""

from typing import Dict, Any
import numpy as np


class HyperparameterSpace:
    """
    Hyperparameter search space for PBT.

    Provides methods to:
    - Sample random hyperparameters (initialization)
    - Perturb hyperparameters (exploration)
    - Validate hyperparameter values
    """

    # Hyperparameter bounds
    LR_MIN = 1e-5
    LR_MAX = 1e-3

    BATCH_SIZES = [64, 128, 256, 512]

    NUM_EPOCHS_OPTIONS = [5, 10, 15, 20]

    CLIP_MIN = 0.05
    CLIP_MAX = 0.3

    ENTROPY_MIN = 1e-4
    ENTROPY_MAX = 0.05

    # Perturbation factors
    PERTURB_FACTOR = 0.2  # ±20%
    PERTURB_PROB = 0.5  # 50% chance to perturb each hyperparam

    @classmethod
    def sample_random(cls) -> Dict[str, Any]:
        """
        Sample random hyperparameters from the search space.

        Returns:
            Dictionary with 5 hyperparameters
        """
        return {
            'learning_rate': cls._sample_log_uniform(cls.LR_MIN, cls.LR_MAX),
            'batch_size': np.random.choice(cls.BATCH_SIZES),
            'num_epochs': np.random.choice(cls.NUM_EPOCHS_OPTIONS),
            'clip_range': cls._sample_uniform(cls.CLIP_MIN, cls.CLIP_MAX),
            'entropy_coef': cls._sample_log_uniform(cls.ENTROPY_MIN, cls.ENTROPY_MAX),
        }

    @classmethod
    def perturb(cls, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perturb hyperparameters with random noise.

        For each hyperparameter:
        - With 50% probability, apply perturbation
        - Continuous params: multiply by factor in [0.8, 1.2] or [1/1.2, 1.2]
        - Discrete params: shift up or down by 1 position

        Args:
            hyperparams: Original hyperparameters

        Returns:
            Perturbed hyperparameters (new dict)
        """
        perturbed = hyperparams.copy()

        # 1. Learning rate: multiply by [0.8, 1.2]
        if np.random.rand() < cls.PERTURB_PROB:
            perturbed['learning_rate'] = cls._perturb_continuous(
                value=perturbed['learning_rate'],
                min_val=cls.LR_MIN,
                max_val=cls.LR_MAX,
                factor=cls.PERTURB_FACTOR,
            )

        # 2. Batch size: shift in discrete list
        if np.random.rand() < cls.PERTURB_PROB:
            perturbed['batch_size'] = cls._perturb_discrete(
                value=perturbed['batch_size'],
                options=cls.BATCH_SIZES,
            )

        # 3. Num epochs: shift in discrete list
        if np.random.rand() < cls.PERTURB_PROB:
            perturbed['num_epochs'] = cls._perturb_discrete(
                value=perturbed['num_epochs'],
                options=cls.NUM_EPOCHS_OPTIONS,
            )

        # 4. Clip range: multiply by [0.8, 1.2]
        if np.random.rand() < cls.PERTURB_PROB:
            perturbed['clip_range'] = cls._perturb_continuous(
                value=perturbed['clip_range'],
                min_val=cls.CLIP_MIN,
                max_val=cls.CLIP_MAX,
                factor=cls.PERTURB_FACTOR,
            )

        # 5. Entropy coefficient: multiply by [0.8, 1.2]
        if np.random.rand() < cls.PERTURB_PROB:
            perturbed['entropy_coef'] = cls._perturb_continuous(
                value=perturbed['entropy_coef'],
                min_val=cls.ENTROPY_MIN,
                max_val=cls.ENTROPY_MAX,
                factor=cls.PERTURB_FACTOR,
            )

        return perturbed

    @classmethod
    def validate(cls, hyperparams: Dict[str, Any]) -> bool:
        """
        Validate that hyperparameters are within bounds.

        Args:
            hyperparams: Hyperparameters to validate

        Returns:
            True if all hyperparameters are valid
        """
        # Check learning rate
        if not (cls.LR_MIN <= hyperparams['learning_rate'] <= cls.LR_MAX):
            return False

        # Check batch size
        if hyperparams['batch_size'] not in cls.BATCH_SIZES:
            return False

        # Check num epochs
        if hyperparams['num_epochs'] not in cls.NUM_EPOCHS_OPTIONS:
            return False

        # Check clip range
        if not (cls.CLIP_MIN <= hyperparams['clip_range'] <= cls.CLIP_MAX):
            return False

        # Check entropy coef
        if not (cls.ENTROPY_MIN <= hyperparams['entropy_coef'] <= cls.ENTROPY_MAX):
            return False

        return True

    @staticmethod
    def _sample_uniform(min_val: float, max_val: float) -> float:
        """Sample uniformly from [min_val, max_val]."""
        return float(np.random.uniform(min_val, max_val))

    @staticmethod
    def _sample_log_uniform(min_val: float, max_val: float) -> float:
        """Sample log-uniformly from [min_val, max_val]."""
        log_min = np.log(min_val)
        log_max = np.log(max_val)
        log_sample = np.random.uniform(log_min, log_max)
        return float(np.exp(log_sample))

    @staticmethod
    def _perturb_continuous(
        value: float,
        min_val: float,
        max_val: float,
        factor: float,
    ) -> float:
        """
        Perturb continuous hyperparameter.

        Multiplies by a factor sampled from [1-factor, 1+factor],
        then clips to [min_val, max_val].

        Args:
            value: Current value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            factor: Perturbation factor (e.g., 0.2 for ±20%)

        Returns:
            Perturbed value
        """
        # Sample perturbation factor from [1-factor, 1+factor]
        perturbation = np.random.uniform(1 - factor, 1 + factor)
        new_value = value * perturbation

        # Clip to bounds
        return float(np.clip(new_value, min_val, max_val))

    @staticmethod
    def _perturb_discrete(value: Any, options: list) -> Any:
        """
        Perturb discrete hyperparameter.

        Shifts the value up or down by 1 position in the options list.

        Args:
            value: Current value
            options: List of valid options

        Returns:
            Perturbed value (stays in options list)
        """
        current_idx = options.index(value)

        # Randomly shift up or down
        shift = np.random.choice([-1, 1])
        new_idx = current_idx + shift

        # Clip to valid indices
        new_idx = np.clip(new_idx, 0, len(options) - 1)

        return options[new_idx]

    @classmethod
    def get_default(cls) -> Dict[str, Any]:
        """
        Get default hyperparameters (middle of search space).

        Useful for baseline comparisons.

        Returns:
            Default hyperparameters
        """
        return {
            'learning_rate': 3e-4,
            'batch_size': 256,
            'num_epochs': 10,
            'clip_range': 0.2,
            'entropy_coef': 0.01,
        }

    @classmethod
    def print_space(cls) -> None:
        """Print the hyperparameter search space."""
        print("=" * 60)
        print("HYPERPARAMETER SEARCH SPACE")
        print("=" * 60)
        print(f"learning_rate:  [{cls.LR_MIN:.0e}, {cls.LR_MAX:.0e}] (log-uniform)")
        print(f"batch_size:     {cls.BATCH_SIZES} (discrete)")
        print(f"num_epochs:     {cls.NUM_EPOCHS_OPTIONS} (discrete)")
        print(f"clip_range:     [{cls.CLIP_MIN:.2f}, {cls.CLIP_MAX:.2f}] (uniform)")
        print(f"entropy_coef:   [{cls.ENTROPY_MIN:.0e}, {cls.ENTROPY_MAX:.2f}] (log-uniform)")
        print("=" * 60)
        print(f"Perturbation factor: ±{cls.PERTURB_FACTOR * 100:.0f}%")
        print(f"Perturbation probability: {cls.PERTURB_PROB * 100:.0f}%")
        print("=" * 60)
