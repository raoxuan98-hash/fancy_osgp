import torch

from .utils import CustomConcatDataset, DatasetWrapper
from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .aircraft import FGVCAircraft
from .food101 import Food101
from .flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .mnist import MNIST
from torchvision import transforms


dataset_list = {
                "aircraft": FGVCAircraft,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "eurosat": EuroSAT,
                "flowers": OxfordFlowers,
                "food101": Food101,
                "mnist": MNIST,
                "oxford_pets": OxfordPets,
                "stanford_cars": StanfordCars,
                "sun397": SUN397}


def build_cur_task_data_loader(root, dataset_name, transform_train, transform_test, num_shots, batch_size, num_workers):
    dataset = dataset_list[dataset_name](root, num_shots)
    train_set = dataset.train_x
    test_set = dataset.test
    classnames = dataset.classnames

    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(train_set, transform=transform_train),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True)

    train_loader4updating = torch.utils.data.DataLoader(
        DatasetWrapper(train_set, transform=transform_test),
        batch_size=256,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True)
    

    test_loader = torch.utils.data.DataLoader(
        DatasetWrapper(test_set, transform=transform_test),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True)

    return train_loader, train_loader4updating, test_loader, classnames

def build_TAIL_testloader(root, dataset_sequence, transform_test, batch_size, num_workers):
    TAIL_testset_list = []
    merged_classnames = []
    indices = []
    offset = 0
    for dataset_name in dataset_sequence:
        dataset = dataset_list[dataset_name](root, -1)
        TAIL_testset_list.append(dataset.test)
        merged_classnames += dataset.classnames
        indices.append(offset)
        offset += len(dataset.classnames)

    test_dataset_instances = [DatasetWrapper(dataset, transform=transform_test) for dataset in TAIL_testset_list]
    TAIL_testset = CustomConcatDataset(test_dataset_instances, indices)
    test_loader = torch.utils.data.DataLoader(
        TAIL_testset,batch_size=batch_size, num_workers=num_workers,shuffle=False,pin_memory=True)

    return test_loader, merged_classnames, indices

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
resolution = 224
root = "./x_tail_data"
dataset_sequence = ["aircraft", "caltech101", "dtd", "eurosat", "flowers", "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"]

def get_loaders(args):
    if args['dataset_name'] in ["mnist", "eurosat"]:
        transform_train = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    build_TAIL_testloader_return = build_TAIL_testloader(
        root=root, dataset_sequence=dataset_sequence, transform_test=transform_test, batch_size=64, num_workers=4)
    
    TAIL_test_loader, merged_classnames, indices = build_TAIL_testloader_return
    
    build_cur_task_data_loader_return = build_cur_task_data_loader(
        root=root, dataset_name=args['dataset_name'], transform_train=transform_train, 
        transform_test=transform_test, num_shots=256, batch_size=64,
        num_workers=4)
    
    train_loader, train_loader4updating, test_loader, classnames = build_cur_task_data_loader_return

    return train_loader, train_loader4updating, test_loader, TAIL_test_loader, classnames, merged_classnames, indices


args = {
    "dataset_name": "mnist"}