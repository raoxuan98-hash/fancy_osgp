import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from utils.toolkit import (
    split_domain_txt2txt,
    split_images_labels,
    write_domain_img_file2txt,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class iData:
    """Minimal dataset descriptor used by :class:`utils.data_manager.DataManager`."""

    def __init__(self) -> None:
        self.use_path: bool = False
        self.train_trsf: List[transforms.Compose] = []
        self.test_trsf: List[transforms.Compose] = []
        self.common_trsf: List[transforms.Compose] = []
        self.class_order: Optional[Sequence[int]] = None
        self.train_data: Optional[np.ndarray] = None
        self.train_targets: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None
        self.test_targets: Optional[np.ndarray] = None

    def download_data(self) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


def _load_image_folder(train_dir: str, val_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dset = datasets.ImageFolder(train_dir)
    val_dset = datasets.ImageFolder(val_dir)
    train_data, train_targets = split_images_labels(train_dset.imgs)
    val_data, val_targets = split_images_labels(val_dset.imgs)
    return train_data, train_targets, val_data, val_targets


class iCIFAR100_224(iData):
    def __init__(self) -> None:
        super().__init__()
        self.train_trsf = [
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ]
        self.class_order = list(range(100))

    def download_data(self) -> None:
        train_dataset = datasets.cifar.CIFAR100("../data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("../data", train=False, download=True)
        self.train_data = train_dataset.data
        self.train_targets = np.array(train_dataset.targets)
        self.test_data = test_dataset.data
        self.test_targets = np.array(test_dataset.targets)


class iImageNetR(iData):
    def __init__(self) -> None:
        super().__init__()
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

    def download_data(self) -> None:
        train_dir = "data/imagenet-r/train/"
        val_dir = "data/imagenet-r/val/"
        (
            self.train_data,
            self.train_targets,
            self.test_data,
            self.test_targets,
        ) = _load_image_folder(train_dir, val_dir)
        self.class_order = np.unique(self.train_targets).tolist()


class iCUB200_224(iData):
    def __init__(self) -> None:
        super().__init__()
        self.use_path = True
        self.train_trsf = [
            transforms.Resize((300, 300), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

    def download_data(self) -> None:
        train_dir = "data/cub_200/train/"
        val_dir = "data/cub_200/val/"
        (
            self.train_data,
            self.train_targets,
            self.test_data,
            self.test_targets,
        ) = _load_image_folder(train_dir, val_dir)
        self.class_order = list(range(int(self.train_targets.max()) + 1))


class iCARS196_224(iData):
    def __init__(self) -> None:
        super().__init__()
        self.use_path = True
        self.train_trsf = [
            transforms.Resize((300, 300), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

    def download_data(self) -> None:
        train_dir = "data/cars196/train/"
        val_dir = "data/cars196/val/"
        (
            self.train_data,
            self.train_targets,
            self.test_data,
            self.test_targets,
        ) = _load_image_folder(train_dir, val_dir)
        self.class_order = list(range(int(self.train_targets.max()) + 1))


class iResisc45_224(iData):
    def __init__(self) -> None:
        super().__init__()
        self.use_path = True
        self.train_trsf = [
            transforms.Resize((300, 300), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

    def download_data(self) -> None:
        train_dir = "data/resisc45/train/"
        val_dir = "data/resisc45/val/"
        (
            self.train_data,
            self.train_targets,
            self.test_data,
            self.test_targets,
        ) = _load_image_folder(train_dir, val_dir)
        self.class_order = list(range(int(self.train_targets.max()) + 1))


class iDomainNet(iData):
    def __init__(self, args: dict) -> None:
        super().__init__()
        if args is None:
            raise ValueError("DomainNet requires argument dictionary with data configuration.")

        self.args = args
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        self.nb_sessions = int(args.get("total_sessions", 6))
        init_cls = int(args.get("init_cls", 0))
        self.cl_n_inc = int(args.get("increment", init_cls if init_cls else 1))
        self.class_incremental = bool(args.get("class_incremental", False))
        self.domain_names = args.get(
            "task_name",
            ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
        )
        logging.info("Learning sequence of domains: %s", self.domain_names)

        total_classes = init_cls + max(0, self.nb_sessions - 1) * self.cl_n_inc
        if total_classes <= 0:
            total_classes = self.cl_n_inc * self.nb_sessions
        self.class_order = list(range(total_classes))

    def _read_split(self, image_list_paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        samples: List[Tuple[str, int]] = []
        for task_id, list_path in enumerate(image_list_paths):
            if task_id >= self.nb_sessions:
                break
            with open(list_path, "r", encoding="utf-8") as f:
                for line in f:
                    rel_path, label_str = line.strip().split()
                    label = int(label_str)
                    if self.class_incremental:
                        lower = task_id * self.cl_n_inc
                        upper = (task_id + 1) * self.cl_n_inc
                        if label < lower or label >= upper:
                            continue
                    else:
                        if label >= self.cl_n_inc:
                            raise ValueError("class_incremental is False, but img_label > cl_n_inc")
                        label += task_id * self.cl_n_inc
                    samples.append((rel_path, label))

        if not samples:
            return np.array([]), np.array([])

        img_x, img_y = zip(*samples)
        root = self.args["data_path"]
        return np.array([os.path.join(root, p) for p in img_x]), np.array(img_y)

    def download_data(self) -> None:
        data_root = self.args.get("data_path")
        if not data_root:
            raise RuntimeError("DomainNet requires args['data_path'] pointing to the dataset root.")

        for domain in self.domain_names:
            write_domain_img_file2txt(data_root, domain)
            split_domain_txt2txt(
                data_root,
                domain,
                train_ratio=self.args.get("train_ratio", 0.7),
                seed=int(self.args.get("seed", 1993)),
            )

        train_lists = [os.path.join(data_root, f"{d}_train.txt") for d in self.domain_names]
        test_lists = [os.path.join(data_root, f"{d}_test.txt") for d in self.domain_names]
        self.train_data, self.train_targets = self._read_split(train_lists)
        self.test_data, self.test_targets = self._read_split(test_lists)
        if self.train_targets.size:
            self.class_order = list(range(int(self.train_targets.max()) + 1))
