"""Concrete dataset implementations backed by :mod:`torchvision`."""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from torchvision import datasets, transforms

from utils.toolkit import split_images_labels

from .base import BaseDataset, DatasetContents
from .prompts import build_cifar_template, build_default_template


class CIFAR10Dataset(BaseDataset):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ]
    class_order = list(range(10))
    templates = build_cifar_template()

    def __init__(self, data_path: str, *, classnames: Optional[Sequence[str]] = None, **kwargs):
        super().__init__(data_path, classnames=classnames, **kwargs)
        if self.classnames is None:
            self.classnames = list(datasets.CIFAR10.classes)

    def load_data(self) -> DatasetContents:
        train_dataset = datasets.CIFAR10(self.data_path, train=True, download=True)
        test_dataset = datasets.CIFAR10(self.data_path, train=False, download=True)
        return DatasetContents(
            train_data=train_dataset.data,
            train_targets=np.array(train_dataset.targets),
            test_data=test_dataset.data,
            test_targets=np.array(test_dataset.targets),
        )


class CIFAR100Dataset(BaseDataset):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        ),
    ]
    class_order = list(range(100))
    templates = build_cifar_template()

    def __init__(self, data_path: str, *, classnames: Optional[Sequence[str]] = None, **kwargs):
        super().__init__(data_path, classnames=classnames, **kwargs)
        if self.classnames is None:
            self.classnames = list(datasets.CIFAR100.fine_label_names)

    def load_data(self) -> DatasetContents:
        train_dataset = datasets.CIFAR100(self.data_path, train=True, download=True)
        test_dataset = datasets.CIFAR100(self.data_path, train=False, download=True)
        return DatasetContents(
            train_data=train_dataset.data,
            train_targets=np.array(train_dataset.targets),
            test_data=test_dataset.data,
            test_targets=np.array(test_dataset.targets),
        )


class CIFAR100Dataset224(CIFAR100Dataset):
    train_trsf = [
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]


class ImageNet1kDataset(BaseDataset):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    class_order = list(range(1000))
    templates = build_default_template()

    def _resolve_split(self, split: str) -> str:
        path = os.path.join(self.data_path, split)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"ImageNet split '{split}' expected at '{path}'. Please point 'data_path' to the dataset root."
            )
        return path

    def load_data(self) -> DatasetContents:
        train_dir = self._resolve_split("train")
        val_dir = self._resolve_split("val")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(val_dir)
        self.classnames = train_dset.classes
        train_data, train_targets = split_images_labels(train_dset.imgs)
        test_data, test_targets = split_images_labels(test_dset.imgs)
        return DatasetContents(
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
        )


class ImageNet100Dataset(ImageNet1kDataset):
    class_order = list(range(100))

    def load_data(self) -> DatasetContents:
        train_dir = self._resolve_split("train")
        val_dir = self._resolve_split("val")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(val_dir)
        self.classnames = train_dset.classes
        train_data, train_targets = split_images_labels(train_dset.imgs)
        test_data, test_targets = split_images_labels(test_dset.imgs)
        return DatasetContents(
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
        )


class ImageNetRDataset(ImageNet1kDataset):
    class_order = list(range(1000))


class CUB200Dataset(BaseDataset):
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    templates = build_default_template()

    def load_data(self) -> DatasetContents:
        train_dir = os.path.join(self.data_path, "train")
        val_dir = os.path.join(self.data_path, "val")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(val_dir)
        self.classnames = train_dset.classes
        train_data, train_targets = split_images_labels(train_dset.imgs)
        test_data, test_targets = split_images_labels(test_dset.imgs)
        return DatasetContents(
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
        )


class Cars196Dataset(CUB200Dataset):
    pass


class Resisc45Dataset(CUB200Dataset):
    pass


class SketchDataset(CUB200Dataset):
    pass


class DomainNetDataset(BaseDataset):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self.domain_names: Sequence[str] = kwargs.get(
            "task_name",
            ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
        )
        self.nb_sessions: int = int(kwargs.get("total_sessions", len(self.domain_names)))
        self.cl_n_inc: int = int(kwargs.get("increment", kwargs.get("init_cls", 0)))
        self.class_incremental: bool = bool(kwargs.get("class_incremental", False))
        self.templates = build_default_template()

    def _read_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        def _read(paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
            imgs: List[Tuple[str, int]] = []
            for task_id, list_path in enumerate(paths):
                if task_id >= self.nb_sessions:
                    break
                with open(list_path) as f:
                    image_list = f.readlines()
                for entry in image_list:
                    rel_path, raw_label = entry.split()[0], int(entry.split()[1])
                    if self.class_incremental:
                        if raw_label < task_id * self.cl_n_inc or raw_label >= (task_id + 1) * self.cl_n_inc:
                            continue
                        label = raw_label
                    else:
                        label = raw_label + task_id * self.cl_n_inc
                    imgs.append((os.path.join(self.data_path, rel_path), label))
            if not imgs:
                raise RuntimeError(
                    "DomainNet text files produced no samples. Check 'data_path' and domain txt files."
                )
            data = np.array([path for path, _ in imgs])
            labels = np.array([label for _, label in imgs])
            return data, labels

        list_paths = [os.path.join(self.data_path, f"{d}_{split}.txt") for d in self.domain_names]
        return _read(list_paths)

    def load_data(self) -> DatasetContents:
        train_data, train_targets = self._read_split("train")
        test_data, test_targets = self._read_split("test")
        total_classes = int(np.max(train_targets)) + 1
        self.classnames = [f"class_{i}" for i in range(total_classes)]
        return DatasetContents(
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
        )


DATASET_REGISTRY = {
    "cifar10": CIFAR10Dataset,
    "cifar100": CIFAR100Dataset,
    "cifar100_224": CIFAR100Dataset224,
    "imagenet1000": ImageNet1kDataset,
    "imagenet100": ImageNet100Dataset,
    "imagenet-r": ImageNetRDataset,
    "cub200_224": CUB200Dataset,
    "cars196_224": Cars196Dataset,
    "resisc45": Resisc45Dataset,
    "sketch345_224": SketchDataset,
    "domainnet": DomainNetDataset,
}


def get_dataset(name: str) -> type[BaseDataset]:
    try:
        return DATASET_REGISTRY[name.lower()]
    except KeyError as exc:
        raise NotImplementedError(f"Unknown dataset '{name}'.") from exc
