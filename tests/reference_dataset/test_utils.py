"""
测试工具模块，提供创建模拟数据集结构和测试辅助函数
"""

import os
import tempfile
import shutil
from typing import Dict, List, Optional
from PIL import Image
import numpy as np


def create_dummy_image(path: str, size: tuple = (224, 224)) -> None:
    """创建一个虚拟图像文件
    
    Args:
        path: 图像保存路径
        size: 图像尺寸
    """
    # 创建一个随机RGB图像
    img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)


def create_imagenet_structure(root_dir: str, num_classes: int = 5, images_per_class: int = 2) -> None:
    """创建模拟的ImageNet数据集目录结构
    
    Args:
        root_dir: 根目录路径
        num_classes: 类别数量
        images_per_class: 每个类别的图像数量
    """
    # 创建train和val目录
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 创建类别目录和图像
    for i in range(num_classes):
        class_name = f"class_{i:04d}"
        
        # 训练集
        train_class_dir = os.path.join(train_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        for j in range(images_per_class):
            img_path = os.path.join(train_class_dir, f"train_img_{i}_{j}.jpg")
            create_dummy_image(img_path)
        
        # 验证集
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(val_class_dir, exist_ok=True)
        for j in range(images_per_class):
            img_path = os.path.join(val_class_dir, f"val_img_{i}_{j}.jpg")
            create_dummy_image(img_path)


def create_flickr8k_structure(root_dir: str, num_images: int = 10) -> None:
    """创建模拟的Flickr8k数据集目录结构
    
    Args:
        root_dir: 根目录路径
        num_images: 图像数量
    """
    # 创建images目录
    images_dir = os.path.join(root_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 创建图像文件
    image_names = []
    for i in range(num_images):
        img_name = f"flickr_{i:06d}.jpg"
        img_path = os.path.join(images_dir, img_name)
        create_dummy_image(img_path)
        image_names.append(img_name)
    
    # 创建captions.txt文件
    captions_file = os.path.join(root_dir, "captions.txt")
    with open(captions_file, 'w', encoding='utf-8') as f:
        f.write("image,caption\n")
        for img_name in image_names:
            for j in range(5):  # 每张图片5个标题
                caption = f"This is caption {j+1} for image {img_name}"
                f.write(f"{img_name},{caption}\n")


def create_mixed_dataset_structure(root_dir: str) -> None:
    """创建混合/无效的数据集目录结构，用于测试错误处理
    
    Args:
        root_dir: 根目录路径
    """
    # 创建一些不相关的目录和文件
    os.makedirs(os.path.join(root_dir, "random_dir"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "another_dir"), exist_ok=True)
    
    # 创建一些文件
    with open(os.path.join(root_dir, "random_file.txt"), 'w') as f:
        f.write("This is not a dataset")
    
    # 创建不完整的ImageNet结构（只有train没有val）
    train_dir = os.path.join(root_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    class_dir = os.path.join(train_dir, "class_0000")
    os.makedirs(class_dir, exist_ok=True)
    create_dummy_image(os.path.join(class_dir, "img.jpg"))


class TempDatasetDir:
    """临时数据集目录上下文管理器"""
    
    def __init__(self, dataset_type: str, **kwargs):
        """
        Args:
            dataset_type: 数据集类型 ('imagenet', 'flickr8k', 'mixed')
            **kwargs: 创建数据集的额外参数
        """
        self.dataset_type = dataset_type.lower()
        self.kwargs = kwargs
        self.temp_dir = None
    
    def __enter__(self) -> str:
        """进入上下文，创建临时目录和数据集结构
        
        Returns:
            str: 临时目录路径
        """
        self.temp_dir = tempfile.mkdtemp(prefix=f"test_{self.dataset_type}_")
        
        if self.dataset_type == "imagenet":
            create_imagenet_structure(self.temp_dir, **self.kwargs)
        elif self.dataset_type == "flickr8k":
            create_flickr8k_structure(self.temp_dir, **self.kwargs)
        elif self.dataset_type == "mixed":
            create_mixed_dataset_structure(self.temp_dir)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        return self.temp_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，清理临时目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.temp_dir = None


def count_files_in_dir(directory: str, extension: Optional[str] = None) -> int:
    """计算目录中文件的数量
    
    Args:
        directory: 目录路径
        extension: 文件扩展名过滤，None表示不过滤
        
    Returns:
        int: 文件数量
    """
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if extension is None or file.lower().endswith(extension.lower()):
                count += 1
    return count


def validate_dataset_structure(root_dir: str, dataset_type: str) -> Dict[str, bool]:
    """验证数据集目录结构
    
    Args:
        root_dir: 根目录路径
        dataset_type: 数据集类型
        
    Returns:
        Dict[str, bool]: 验证结果
    """
    result = {}
    
    if dataset_type.lower() == "imagenet":
        result["train_exists"] = os.path.exists(os.path.join(root_dir, "train"))
        result["val_exists"] = os.path.exists(os.path.join(root_dir, "val"))
        result["train_has_classes"] = False
        result["val_has_classes"] = False
        
        if result["train_exists"]:
            train_dir = os.path.join(root_dir, "train")
            class_dirs = [d for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))]
            result["train_has_classes"] = len(class_dirs) > 0
        
        if result["val_exists"]:
            val_dir = os.path.join(root_dir, "val")
            class_dirs = [d for d in os.listdir(val_dir) 
                          if os.path.isdir(os.path.join(val_dir, d))]
            result["val_has_classes"] = len(class_dirs) > 0
    
    elif dataset_type.lower() == "flickr8k":
        result["images_exists"] = os.path.exists(os.path.join(root_dir, "images"))
        result["captions_exists"] = os.path.exists(os.path.join(root_dir, "captions.txt"))
        
        if result["images_exists"]:
            images_dir = os.path.join(root_dir, "images")
            img_count = count_files_in_dir(images_dir, ".jpg")
            result["images_has_files"] = img_count > 0
        
        if result["captions_exists"]:
            captions_file = os.path.join(root_dir, "captions.txt")
            with open(captions_file, 'r') as f:
                lines = f.readlines()
                result["captions_has_content"] = len(lines) > 1  # 超过标题行
    
    return result