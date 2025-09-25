import logging
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data import (
    iCARS196_224,
    iCIFAR100_224,
    iCUB200_224,
    iDomainNet,
    iImageNetR,
    iResisc45_224,
)


class DataManager:
    """Lightweight data manager tailored to the datasets used in this project."""

    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.dataset_name = dataset_name
        self.args = args or {}
        self.init_cls = init_cls
        self.increment = increment
        self.seed = seed

        self._setup_data(shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = self._build_task_schedule()

    @property
    def nb_tasks(self) -> int:
        return len(self._increments)

    def get_task_size(self, task: int) -> int:
        return self._increments[task]

    def get_dataset(self, class_indices: Iterable[int], source: str, mode: str):
        data, targets = self._select(source, class_indices)
        transform = self._build_transform(mode)
        return DummyDataset(data, targets, transform, self.use_path)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _setup_data(self, shuffle: bool, seed: int) -> None:
        idata_args = dict(self.args)
        idata_args.setdefault("init_cls", self.init_cls)
        idata_args.setdefault("increment", self.increment)
        idata_args.setdefault("seed", seed)
        logging.info("[DataManager] args_for_idata = %s", idata_args)

        idata = _get_idata(self.dataset_name, idata_args)
        idata.download_data()

        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        unique_targets = np.unique(self._train_targets)
        if shuffle:
            rng = np.random.RandomState(seed)
            class_order = rng.permutation(unique_targets).tolist()
        else:
            class_order = (
                list(idata.class_order)
                if idata.class_order is not None
                else unique_targets.tolist()
            )
        self._class_order = class_order
        logging.info("Class order: %s", self._class_order)

        mapping = {orig: idx for idx, orig in enumerate(self._class_order)}
        self._train_targets = np.array([mapping[int(y)] for y in self._train_targets])
        self._test_targets = np.array([mapping[int(y)] for y in self._test_targets])

    def _build_task_schedule(self) -> List[int]:
        total_classes = len(self._class_order)
        increments = [self.init_cls]
        remaining = total_classes - self.init_cls
        while remaining > 0:
            step = min(self.increment, remaining)
            increments.append(step)
            remaining -= step
        return increments

    def _build_transform(self, mode: str) -> transforms.Compose:
        if mode == "train":
            ops = [*self._train_trsf, *self._common_trsf]
        elif mode == "test":
            ops = [*self._test_trsf, *self._common_trsf]
        else:
            raise ValueError(f"Unknown mode {mode}.")
        return transforms.Compose(ops)

    def _select(self, source: str, class_indices: Sequence[int]):
        if source == "train":
            data, targets = self._train_data, self._train_targets
        elif source == "test":
            data, targets = self._test_data, self._test_targets
        else:
            raise ValueError(f"Unknown data source {source}.")

        mask = np.isin(targets, class_indices)
        return data[mask], targets[mask]


class DummyDataset(Dataset):
    def __init__(self, images, labels, transform, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.transform = transform
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = pil_loader(self.images[idx])
        else:
            image = Image.fromarray(self.images[idx])
        image = self.transform(image)
        return idx, image, self.labels[idx]


def _get_idata(dataset_name: str, args=None):
    name = dataset_name.lower()
    if name == "cifar100_224":
        return iCIFAR100_224()
    if name == "imagenet-r":
        return iImageNetR()
    if name == "cub200_224":
        return iCUB200_224()
    if name == "cars196_224":
        return iCARS196_224()
    if name == "resisc45_224":
        return iResisc45_224()
    if name == "domainnet":
        return iDomainNet(args or {})
    raise NotImplementedError(f"Unknown dataset {dataset_name}.")


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
