"""Shared dataset abstractions for incremental learning tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


PromptTemplate = Callable[[str], str]


@dataclass
class DatasetContents:
    """Container that bundles the canonical tensors required by trainers.

    Attributes
    ----------
    train_data:
        Either a numpy array of images or a sequence of image paths.
    train_targets:
        Integer encoded labels for ``train_data``.
    test_data:
        Either a numpy array of images or a sequence of image paths.
    test_targets:
        Integer encoded labels for ``test_data``.
    """

    train_data: np.ndarray
    train_targets: np.ndarray
    test_data: np.ndarray
    test_targets: np.ndarray


class BaseDataset(ABC):
    """Interface implemented by every dataset used in the repository.

    The class encapsulates all dataset specific logic (downloading, reading
    from disk, computing class names, etc.) and exposes a uniform API that
    ``DataManager`` can consume when creating incremental tasks.
    """

    #: Transforms that are applied only for the training split.
    train_trsf: Sequence
    #: Transforms that are applied only for the evaluation split.
    test_trsf: Sequence
    #: Transforms that are shared across train/test splits.
    common_trsf: Sequence
    #: Optional manual class ordering used when shuffling is disabled.
    class_order: Optional[Sequence[int]]
    #: Optional CLIP style templates for building textual classifiers.
    templates: Optional[Sequence[PromptTemplate]]

    def __init__(self, data_path: str, *, classnames: Optional[Sequence[str]] = None, **kwargs):
        self.data_path = data_path
        self.classnames: Optional[List[str]] = list(classnames) if classnames else None
        self.use_path: bool = getattr(self, "use_path", False)
        self.train_trsf = getattr(self, "train_trsf", [])
        self.test_trsf = getattr(self, "test_trsf", [])
        self.common_trsf = getattr(self, "common_trsf", [])
        self.class_order = getattr(self, "class_order", None)
        self.templates = getattr(self, "templates", None)
        self._extra_kwargs = kwargs

        self.train_data: Optional[np.ndarray] = None
        self.train_targets: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None
        self.test_targets: Optional[np.ndarray] = None

    @abstractmethod
    def load_data(self) -> DatasetContents:
        """Load the dataset from disk and return numpy arrays.

        Sub-classes must populate the four fields of :class:`DatasetContents`.
        ``DataManager`` expects the returned arrays to be contiguous and aligned
        (i.e. ``len(train_data) == len(train_targets)``).
        """

    def ensure_loaded(self) -> DatasetContents:
        if self.train_data is None or self.test_data is None:
            contents = self.load_data()
            self.train_data = contents.train_data
            self.train_targets = contents.train_targets
            self.test_data = contents.test_data
            self.test_targets = contents.test_targets
        else:
            contents = DatasetContents(
                train_data=self.train_data,
                train_targets=self.train_targets,
                test_data=self.test_data,
                test_targets=self.test_targets,
            )
        return contents

    def infer_class_order(self) -> Sequence[int]:
        """Return the deterministic class order used for incremental splits."""
        if self.class_order is not None:
            return list(self.class_order)
        if self.train_targets is None:
            raise RuntimeError("Dataset must be loaded before class order can be inferred.")
        return sorted(np.unique(self.train_targets).tolist())

    # Convenience helpers -------------------------------------------------
    @property
    def metadata(self) -> Tuple[Optional[Sequence[str]], Optional[Sequence[PromptTemplate]]]:
        return self.classnames, self.templates
