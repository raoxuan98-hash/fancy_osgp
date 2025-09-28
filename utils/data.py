import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from torchvision import datasets, transforms
<<<<<<< HEAD
from utils.toolkit import split_images_labels, write_domain_img_file2txt, split_domain_txt2txt
import os, logging

# ✅ 设置全局数据根目录（你可以根据需要修改这个路径）
BASE_DATA_DIR = "D:/projects/datasets/elevater"  # 例如："/home/user/datasets" 或 "D:/datasets"

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    
    train_data = None
    train_targets = None
    test_data = None
    test_targets = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        data_root = os.path.join(BASE_DATA_DIR, "cifar-10")
        train_dataset = datasets.cifar.CIFAR10(data_root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(data_root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]
=======
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
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601

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

<<<<<<< HEAD
    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        data_root = os.path.join(BASE_DATA_DIR, "cifar-100")
        train_dataset = datasets.cifar.CIFAR100(data_root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iCIFAR100_224(iCIFAR100):
    train_trsf = [
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63/255)]
    
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224)]

    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        data_root = os.path.join(BASE_DATA_DIR)
        train_dataset = datasets.cifar.CIFAR100(data_root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224)]
    
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
=======
    def download_data(self) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


def _load_image_folder(train_dir: str, val_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dset = datasets.ImageFolder(train_dir)
    val_dset = datasets.ImageFolder(val_dir)
    train_data, train_targets = split_images_labels(train_dset.imgs)
    val_data, val_targets = split_images_labels(val_dset.imgs)
    return train_data, train_targets, val_data, val_targets
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601


<<<<<<< HEAD
    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        # ✅ 建议也使用 BASE_DATA_DIR，但保留灵活性
        train_dir = os.path.join(BASE_DATA_DIR, "imagenet-1k", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "imagenet-1k", "val")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        train_dir = os.path.join(BASE_DATA_DIR, "imagenet-100", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "imagenet-100", "val")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
=======
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
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601


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

<<<<<<< HEAD
    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        train_dir = os.path.join(BASE_DATA_DIR, "imagenet-r", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "imagenet-r", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
=======
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601


class iCUB200_224(iData):
<<<<<<< HEAD
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip()]
    
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224)]
    
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    
    class_order = np.arange(1000).tolist()

    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        train_dir = os.path.join(BASE_DATA_DIR, "cub-200", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "cub-200", "test")
=======
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
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601



class iCARS196_224(iData):
<<<<<<< HEAD
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip()]
    
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224)]
    
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    
    class_order = np.arange(1000).tolist()

    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        train_dir = os.path.join(BASE_DATA_DIR, "stanford-cars", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "stanford-cars", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
=======
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
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601


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

<<<<<<< HEAD
    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        train_dir = os.path.join(BASE_DATA_DIR, "resisc45_clip", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "resisc45_clip", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
=======
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
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601

class iFood101_224(iData): 
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        train_dir = os.path.join(BASE_DATA_DIR, "food-101", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "food-101", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iCaltech101_224(iData): 
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = os.path.join(BASE_DATA_DIR, "caltech-101", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "caltech-101", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

<<<<<<< HEAD

class iSketch345_224(iData):
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        train_dir = os.path.join(BASE_DATA_DIR, "sketch345", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "sketch345", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iOxfordPet37_224(iData):
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        # ✅ 使用 BASE_DATA_DIR
        train_dir = os.path.join(BASE_DATA_DIR, "oxford-iiit-pets", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "oxford-iiit-pets", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        

=======
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601
class iDomainNet(iData):
    def __init__(self, args: dict) -> None:
        super().__init__()
        if args is None:
            raise ValueError("DomainNet requires argument dictionary with data configuration.")

        self.args = args
<<<<<<< HEAD
        init_cls = int(self.args.get("init_cls", 0))
        total_sessions = int(self.args.get("total_sessions", 6))
        increment = int(self.args.get("increment", init_cls))
        total_classes = init_cls + max(0, total_sessions - 1) * increment
        self.class_order = np.arange(total_classes).tolist()

        self.nb_sessions = total_sessions
        self.cl_n_inc = increment
         
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        self.class_incremental = True if "class_incremental" in args and args["class_incremental"] else False
=======
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
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601

        total_classes = init_cls + max(0, self.nb_sessions - 1) * self.cl_n_inc
        if total_classes <= 0:
            total_classes = self.cl_n_inc * self.nb_sessions
        self.class_order = list(range(total_classes))

<<<<<<< HEAD
    def download_data(self):
        def _read_data(image_list_paths):
            imgs = []
            for taskid, image_list_path in enumerate(image_list_paths):
                if taskid >= self.nb_sessions:
                    break
                with open(image_list_path) as f:
                    image_list = f.readlines()
                for entry in image_list:
                    img_label = int(entry.split()[1])
=======
    def _read_split(self, image_list_paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        samples: List[Tuple[str, int]] = []
        for task_id, list_path in enumerate(image_list_paths):
            if task_id >= self.nb_sessions:
                break
            with open(list_path, "r", encoding="utf-8") as f:
                for line in f:
                    rel_path, label_str = line.strip().split()
                    label = int(label_str)
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601
                    if self.class_incremental:
                        lower = task_id * self.cl_n_inc
                        upper = (task_id + 1) * self.cl_n_inc
                        if label < lower or label >= upper:
                            continue
<<<<<<< HEAD
                    elif img_label > self.cl_n_inc:
                        raise ValueError("class_incremental is False, but img_label > cl_n_inc")
                    else:
                        img_label = img_label + taskid * self.cl_n_inc
                    imgs.append((entry.split()[0], img_label))

            img_x, img_y = [], []
            for item in imgs:
                # ✅ 使用 self.image_list_root（由 args["data_path"] 或 BASE_DATA_DIR 提供）
                img_x.append(os.path.join(self.image_list_root, item[0]))
                img_y.append(item[1])
=======
                    else:
                        if label >= self.cl_n_inc:
                            raise ValueError("class_incremental is False, but img_label > cl_n_inc")
                        label += task_id * self.cl_n_inc
                    samples.append((rel_path, label))

        if not samples:
            return np.array([]), np.array([])
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601

        img_x, img_y = zip(*samples)
        root = self.args["data_path"]
        return np.array([os.path.join(root, p) for p in img_x]), np.array(img_y)

<<<<<<< HEAD
        # ✅ 优先使用 args["data_path"]，若未提供则使用 BASE_DATA_DIR + "domainnet"
        self.image_list_root = self.args.get("data_path", os.path.join(BASE_DATA_DIR, "domainnet"))
=======
    def download_data(self) -> None:
        data_root = self.args.get("data_path")
        if not data_root:
            raise RuntimeError("DomainNet requires args['data_path'] pointing to the dataset root.")
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601

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
