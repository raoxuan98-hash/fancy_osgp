"""Configuration dataclasses for SubspaceLoRA CLIP learner."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


@dataclass
class Timing:
    """Track the elapsed time of key training stages for a task."""
    train: float = 0.0
    drift: float = 0.0
    total: float = 0.0


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for optimisation and scheduling."""
    optimizer_type: str
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    iterations: int
    eta_min: float = 1e-5


@dataclass(frozen=True)
class TrainingLoopConfig:
    batch_size: int
    log_interval: int
    ema_alpha: float


@dataclass(frozen=True)
class RegularizationConfig:
    """Weights and toggles for auxiliary regularisation terms."""
    gamma_kd: float
    gamma_norm: float
    gamma_prior: float
    l2_enabled: bool
    l2_lambda: float


@dataclass(frozen=True)
class ReferenceConfig:
    """Reference dataset configuration used for knowledge distillation."""
    enabled: bool
    dataset_type: str
    dataset_path: Optional[str]
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class TrainingStepMetrics:
    """Metrics captured after completing a single optimisation step."""
    loss: float
    correct: int
    kd_value: float
    prior_value: float
    batch_size: int


@dataclass
class ReferenceBatch:
    """Container for an optional reference batch used during KD."""
    images: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
