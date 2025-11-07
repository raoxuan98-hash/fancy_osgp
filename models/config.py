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
    bidirectional_kd: bool = False  # 控制是否使用双向KL散度进行知识蒸馏
    # Layer-wise特征蒸馏配置
    layerwise_kd_enabled: bool = False  # 是否启用layer-wise特征蒸馏
    layerwise_kd_weight: float = 1.0  # layer-wise蒸馏的权重
    layerwise_kd_layers: Optional[List[int]] = None  # 指定要蒸馏的层，None表示所有层
    layerwise_kd_pooling: str = "mean"  # 池化方式："mean", "cls", "max"
    layerwise_kd_loss_type: str = "mse"  # 损失类型："mse", "cosine", "mse_cosine"
    layerwise_kd_weight_strategy: str = "uniform"  # 层权重策略："uniform", "linear", "exponential"


@dataclass(frozen=True)
class ReferenceConfig:
    """Reference dataset configuration used for knowledge distillation."""
    enabled: bool
    dataset_type: str
    dataset_path: Optional[str]
    batch_size: int
    num_workers: int
    pin_memory: bool
    # 新增配置选项
    auto_detect: bool = False  # 是否启用自动数据集类型检测
    type_hint: Optional[str] = None  # 数据集类型提示，用于辅助自动检测
    num_samples: Optional[int] = None  # 限制数据集样本数量，None表示使用全部样本
    split: str = "val"  # 数据集分割，对于ImageNet等数据集有效，可选"train"或"val"


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
