import numpy as np
from torchvision import datasets, transforms
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

    class_order = np.arange(100).tolist()

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

    class_order = np.arange(1000).tolist()

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


class iImageNetR(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224, interpolation=3),
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
        train_dir = os.path.join(BASE_DATA_DIR, "imagenet-r", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "imagenet-r", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iCUB200_224(iData):
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

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iCARS196_224(iData):
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


class iResisc45_224(iData): 
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
        train_dir = os.path.join(BASE_DATA_DIR, "resisc45_clip", "train")
        test_dir = os.path.join(BASE_DATA_DIR, "resisc45_clip", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

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
        

class iDomainNet(iData):
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

    def __init__(self, args):
        self.args = args
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

        logging.info("Learning sequence of domains: {}".format(self.domain_names))

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
                    if self.class_incremental:
                        if img_label < taskid * self.cl_n_inc or img_label >= (taskid + 1) * self.cl_n_inc:
                            continue
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

            return np.array(img_x), np.array(img_y)

        # ✅ 优先使用 args["data_path"]，若未提供则使用 BASE_DATA_DIR + "domainnet"
        self.image_list_root = self.args.get("data_path", os.path.join(BASE_DATA_DIR, "domainnet"))

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        self.train_data, self.train_targets = _read_data(image_list_paths)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        self.test_data, self.test_targets = _read_data(image_list_paths)