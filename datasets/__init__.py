"""Unified dataset registry used across ViT and CLIP training."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .base import BaseDataset
from .vision import DATASET_REGISTRY, get_dataset


def create_dataset(name: str, data_path: str, *, classnames: Optional[Sequence[str]] = None, **kwargs: Any) -> BaseDataset:
    """Instantiate a dataset from the registry.

    Parameters
    ----------
    name:
        Canonical dataset identifier, e.g. ``"cifar100"``.
    data_path:
        Filesystem path where the dataset is stored.
    classnames:
        Optional user provided class names that override defaults.
    **kwargs:
        Dataset specific configuration such as ``init_cls`` or ``task_name``.
    """

    dataset_cls = get_dataset(name)
    return dataset_cls(data_path, classnames=classnames, **kwargs)


__all__ = [
    "BaseDataset",
    "DATASET_REGISTRY",
    "create_dataset",
    "get_dataset",
]
