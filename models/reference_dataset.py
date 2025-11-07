"""参考数据集模块，提供统一的数据集接口和自动类型检测功能。

该模块实现了以下组件：
1. DatasetTypeDetector - 数据集类型自动检测
2. BaseReferenceDataset - 参考数据集抽象基类
3. ImageNetRefDataset - ImageNet数据集实现
4. Flickr8kRefDataset - Flickr8k数据集实现
5. PathValidator - 路径验证工具
6. ReferenceDatasetFactory - 数据集工厂类
7. 异常类定义
"""

import os
import csv
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type, Union, Any

import torch
from PIL import Image
from torch.utils.data import Dataset

# 图像扩展名集合
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


class DatasetDetectionError(Exception):
    """数据集类型检测失败时抛出的异常"""
    pass


class DatasetPathError(Exception):
    """数据集路径验证失败时抛出的异常"""
    pass


class DatasetLoadError(Exception):
    """数据集加载失败时抛出的异常"""
    pass


class PathValidator:
    """路径验证工具类，提供数据集路径验证功能"""
    
    @staticmethod
    def validate_path_exists(path: str, path_type: str = "directory") -> None:
        """验证路径是否存在
        
        Args:
            path: 要验证的路径
            path_type: 路径类型描述，用于错误信息
            
        Raises:
            DatasetPathError: 路径不存在时抛出
        """
        if not os.path.exists(path):
            raise DatasetPathError(f"{path_type}路径不存在: {path}")
    
    @staticmethod
    def validate_directory_readable(path: str) -> None:
        """验证目录是否可读
        
        Args:
            path: 要验证的目录路径
            
        Raises:
            DatasetPathError: 目录不可读时抛出
        """
        if not os.path.isdir(path):
            raise DatasetPathError(f"路径不是目录: {path}")
        
        if not os.access(path, os.R_OK):
            raise DatasetPathError(f"目录不可读: {path}")
    
    @staticmethod
    def validate_file_readable(path: str) -> None:
        """验证文件是否可读
        
        Args:
            path: 要验证的文件路径
            
        Raises:
            DatasetPathError: 文件不可读时抛出
        """
        if not os.path.isfile(path):
            raise DatasetPathError(f"路径不是文件: {path}")
        
        if not os.access(path, os.R_OK):
            raise DatasetPathError(f"文件不可读: {path}")
    
    @staticmethod
    def validate_imagenet_structure(root_path: str) -> bool:
        """验证ImageNet数据集目录结构
        
        Args:
            root_path: ImageNet根目录路径
            
        Returns:
            bool: 如果结构符合预期返回True，否则返回False
        """
        try:
            # 检查train和val目录
            train_dir = os.path.join(root_path, "train")
            val_dir = os.path.join(root_path, "val")
            
            if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
                return False
            
            # 检查train目录下是否有子目录（类别目录）
            if not any(os.path.isdir(os.path.join(train_dir, d)) 
                      for d in os.listdir(train_dir)):
                return False
                
            # 检查val目录下是否有子目录（类别目录）
            if not any(os.path.isdir(os.path.join(val_dir, d)) 
                      for d in os.listdir(val_dir)):
                return False
                
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_flickr8k_structure(root_path: str) -> bool:
        """验证Flickr8k数据集目录结构
        
        Args:
            root_path: Flickr8k根目录路径
            
        Returns:
            bool: 如果结构符合预期返回True，否则返回False
        """
        try:
            # 检查images目录和captions.txt文件
            images_dir = os.path.join(root_path, "images")
            captions_file = os.path.join(root_path, "captions.txt")
            
            if not (os.path.exists(images_dir) and os.path.exists(captions_file)):
                return False
            
            # 检查images目录下是否有图片文件
            if not any(os.path.splitext(f)[1].lower() in IMG_EXTS 
                      for f in os.listdir(images_dir)):
                return False
                
            return True
        except Exception:
            return False


class DatasetTypeDetector:
    """数据集类型自动检测器"""
    
    # 数据集类型关键字映射
    DATASET_KEYWORDS = {
        "imagenet": ["imagenet", "ilsvrc", "image_net"],
        "flickr8k": ["flickr8k", "flickr_8k"],
        "coco": ["coco", "ms_coco"],
        "places365": ["places365", "places_365"],
    }
    
    @classmethod
    def detect_by_path_keywords(cls, path: str) -> Optional[str]:
        """根据路径关键字检测数据集类型
        
        Args:
            path: 数据集路径
            
        Returns:
            Optional[str]: 检测到的数据集类型，无法确定时返回None
        """
        path_lower = path.lower()
        
        for dataset_type, keywords in cls.DATASET_KEYWORDS.items():
            if any(keyword in path_lower for keyword in keywords):
                return dataset_type
                
        return None
    
    @classmethod
    def detect_by_directory_structure(cls, path: str) -> Optional[str]:
        """根据目录结构检测数据集类型
        
        Args:
            path: 数据集路径
            
        Returns:
            Optional[str]: 检测到的数据集类型，无法确定时返回None
        """
        try:
            PathValidator.validate_path_exists(path)
            
            # 检测ImageNet结构
            if PathValidator.validate_imagenet_structure(path):
                return "imagenet"
            
            # 检测Flickr8k结构
            if PathValidator.validate_flickr8k_structure(path):
                return "flickr8k"
                
        except DatasetPathError:
            logging.warning(f"无法访问路径进行结构检测: {path}")
            
        return None
    
    @classmethod
    def detect_dataset_type(cls, path: str, hint: Optional[str] = None) -> str:
        """自动检测数据集类型
        
        Args:
            path: 数据集路径
            hint: 可选的类型提示，优先级高于自动检测
            
        Returns:
            str: 检测到的数据集类型
            
        Raises:
            DatasetDetectionError: 无法确定数据集类型时抛出
        """
        # 如果提供了类型提示，优先使用
        if hint:
            return hint.lower()
        
        # 首先尝试根据路径关键字检测
        detected_type = cls.detect_by_path_keywords(path)
        if detected_type:
            logging.info(f"通过路径关键字检测到数据集类型: {detected_type}")
            return detected_type
        
        # 然后尝试根据目录结构检测
        detected_type = cls.detect_by_directory_structure(path)
        if detected_type:
            logging.info(f"通过目录结构检测到数据集类型: {detected_type}")
            return detected_type
        
        # 无法确定类型
        raise DatasetDetectionError(
            f"无法确定数据集类型。路径: {path}。"
            f"请确保路径包含可识别的关键字或具有标准的数据集目录结构。"
        )


class BaseReferenceDataset(Dataset, ABC):
    """参考数据集抽象基类，定义统一接口"""
    
    def __init__(self, root: str, transform=None, num_samples: Optional[int] = None):
        """初始化参考数据集
        
        Args:
            root: 数据集根目录
            transform: 图像变换
            num_samples: 限制样本数量，None表示使用全部样本
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.num_samples = num_samples
        
        # 验证路径
        self._validate_paths()
        
        # 加载数据集
        self._load_dataset()
        
        # 应用样本数量限制
        if num_samples is not None and len(self.samples) > num_samples:
            self.samples = self.samples[:num_samples]
            logging.info(f"数据集样本数量限制为 {num_samples}")
    
    @abstractmethod
    def _validate_paths(self) -> None:
        """验证数据集路径，子类必须实现"""
        pass
    
    @abstractmethod
    def _load_dataset(self) -> None:
        """加载数据集，子类必须实现"""
        pass
    
    @abstractmethod
    def return_labels_and_prompts(self) -> Tuple[List[int], List[List[str]]]:
        """返回标签和提示词列表，子类必须实现
        
        Returns:
            Tuple[List[int], List[List[str]]]: 标签列表和对应的提示词列表
        """
        pass
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本
        
        Args:
            index: 样本索引
            
        Returns:
            Tuple[torch.Tensor, int]: 图像张量和标签
        """
        path, label = self.samples[index]
        
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        except Exception as e:
            raise DatasetLoadError(f"无法加载图像 {path}: {e}")
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label


class ImageNetRefDataset(BaseReferenceDataset):
    """ImageNet参考数据集实现"""
    
    def __init__(self, root: str, split: str = "val", transform=None, num_samples: Optional[int] = None):
        """初始化ImageNet参考数据集
        
        Args:
            root: ImageNet根目录
            split: 数据集分割，可选"train"或"val"
            transform: 图像变换
            num_samples: 限制样本数量
        """
        self.split = split
        super().__init__(root, transform, num_samples)
    
    def _validate_paths(self) -> None:
        """验证ImageNet数据集路径"""
        PathValidator.validate_path_exists(self.root)
        PathValidator.validate_directory_readable(self.root)
        
        # 验证特定分割的目录
        split_dir = os.path.join(self.root, self.split)
        if not os.path.exists(split_dir):
            raise DatasetPathError(f"ImageNet {self.split} 目录不存在: {split_dir}")
        
        PathValidator.validate_directory_readable(split_dir)
    
    def _load_dataset(self) -> None:
        """加载ImageNet数据集"""
        split_dir = os.path.join(self.root, self.split)
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        
        # 获取所有类别目录
        class_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))]
        class_dirs.sort()
        
        # 创建类别到索引的映射
        for idx, class_name in enumerate(class_dirs):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        # 收集所有图像样本
        for class_name in class_dirs:
            class_dir = os.path.join(split_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                ext = os.path.splitext(img_name)[1].lower()
                
                if ext in IMG_EXTS:
                    self.samples.append((img_path, class_idx))
        
        logging.info(f"加载了 {len(self.samples)} 个ImageNet {self.split}样本，"
                    f"涵盖 {len(class_dirs)} 个类别")
    
    def return_labels_and_prompts(self) -> Tuple[List[int], List[List[str]]]:
        """返回ImageNet标签和提示词
        
        Returns:
            Tuple[List[int], List[List[str]]]: 标签列表和对应的提示词列表
        """
        labels = list(range(len(self.idx_to_class)))
        prompts = []
        
        for idx in labels:
            class_name = self.idx_to_class[idx]
            # 简单的类别名称处理，将下划线替换为空格，首字母大写
            formatted_name = class_name.replace('_', ' ').title()
            prompts.append([f"a photo of a {formatted_name}"])
        
        return labels, prompts


class Flickr8kRefDataset(BaseReferenceDataset):
    """Flickr8k参考数据集实现，改进原有实现"""
    
    def __init__(self, root: str, transform=None, num_samples: Optional[int] = None):
        """初始化Flickr8k参考数据集
        
        Args:
            root: Flickr8k根目录
            transform: 图像变换
            num_samples: 限制样本数量
        """
        super().__init__(root, transform, num_samples)
    
    def _validate_paths(self) -> None:
        """验证Flickr8k数据集路径"""
        PathValidator.validate_path_exists(self.root)
        PathValidator.validate_directory_readable(self.root)
        
        # 验证images目录
        images_dir = os.path.join(self.root, "images")
        if not os.path.exists(images_dir):
            raise DatasetPathError(f"Flickr8k images目录不存在: {images_dir}")
        
        PathValidator.validate_directory_readable(images_dir)
        
        # 验证captions.txt文件
        captions_file = os.path.join(self.root, "captions.txt")
        if not os.path.exists(captions_file):
            raise DatasetPathError(f"Flickr8k captions.txt文件不存在: {captions_file}")
        
        PathValidator.validate_file_readable(captions_file)
    
    def _load_dataset(self) -> None:
        """加载Flickr8k数据集"""
        images_dir = os.path.join(self.root, "images")
        captions_file = os.path.join(self.root, "captions.txt")
        
        # 加载标题映射
        captions_map: Dict[str, List[str]] = defaultdict(list)
        
        try:
            with open(captions_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # 跳过标题行
                if header is None:
                    raise DatasetLoadError("captions.txt文件为空或格式不正确")
                
                for row in reader:
                    if len(row) < 2:
                        continue
                    img_name, caption = row[0].strip(), row[1].strip()
                    captions_map[img_name].append(caption)
        except Exception as e:
            raise DatasetLoadError(f"加载captions.txt失败: {e}")
        
        # 收集图像样本
        img_paths: List[str] = []
        
        for dirpath, _, filenames in os.walk(images_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS and fn in captions_map:
                    img_paths.append(os.path.join(dirpath, fn))
        
        # 排序以确保确定性
        img_paths.sort()
        
        # 创建样本列表和提示词列表
        self.samples: List[Tuple[str, int]] = []
        self.prompts_list: List[List[str]] = []
        
        for idx, path in enumerate(img_paths):
            fname = os.path.basename(path)
            self.samples.append((path, idx))
            self.prompts_list.append(captions_map.get(fname, []))
        
        logging.info(f"加载了 {len(self.samples)} 个Flickr8k样本")
    
    def return_labels_and_prompts(self) -> Tuple[List[int], List[List[str]]]:
        """返回Flickr8k标签和提示词
        
        Returns:
            Tuple[List[int], List[List[str]]]: 标签列表和对应的提示词列表
        """
        labels = list(range(len(self.prompts_list)))
        return labels, self.prompts_list


class ReferenceDatasetFactory:
    """参考数据集工厂类，根据配置创建数据集实例"""
    
    # 数据集类型到实现类的映射
    DATASET_REGISTRY: Dict[str, Type[BaseReferenceDataset]] = {
        "imagenet": ImageNetRefDataset,
        "flickr8k": Flickr8kRefDataset,
    }
    
    @classmethod
    def create_dataset(
        cls,
        dataset_type: str,
        dataset_path: str,
        transform=None,
        num_samples: Optional[int] = None,
        **kwargs
    ) -> BaseReferenceDataset:
        """创建参考数据集实例
        
        Args:
            dataset_type: 数据集类型
            dataset_path: 数据集路径
            transform: 图像变换
            num_samples: 限制样本数量
            **kwargs: 其他数据集特定参数
            
        Returns:
            BaseReferenceDataset: 数据集实例
            
        Raises:
            ValueError: 不支持的数据集类型
        """
        dataset_type = dataset_type.lower()
        
        if dataset_type not in cls.DATASET_REGISTRY:
            supported_types = ", ".join(cls.DATASET_REGISTRY.keys())
            raise ValueError(
                f"不支持的数据集类型: {dataset_type}。"
                f"支持的类型: {supported_types}"
            )
        
        dataset_class = cls.DATASET_REGISTRY[dataset_type]
        
        try:
            if dataset_type == "imagenet":
                split = kwargs.get("split", "val")
                return ImageNetRefDataset(dataset_path, split=split, transform=transform, num_samples=num_samples)
            else:
                return dataset_class(dataset_path, transform=transform, num_samples=num_samples)
        except Exception as e:
            raise DatasetLoadError(f"创建{dataset_type}数据集失败: {e}")
    
    @classmethod
    def create_dataset_auto_detect(
        cls,
        dataset_path: str,
        transform=None,
        num_samples: Optional[int] = None,
        type_hint: Optional[str] = None,
        **kwargs
    ) -> BaseReferenceDataset:
        """自动检测数据集类型并创建实例
        
        Args:
            dataset_path: 数据集路径
            transform: 图像变换
            num_samples: 限制样本数量
            type_hint: 可选的类型提示
            **kwargs: 其他数据集特定参数
            
        Returns:
            BaseReferenceDataset: 数据集实例
        """
        try:
            dataset_type = DatasetTypeDetector.detect_dataset_type(dataset_path, type_hint)
            logging.info(f"自动检测到数据集类型: {dataset_type}")
            return cls.create_dataset(dataset_type, dataset_path, transform, num_samples, **kwargs)
        except DatasetDetectionError as e:
            raise DatasetLoadError(f"自动检测数据集类型失败: {e}")
    
    @classmethod
    def register_dataset(cls, dataset_type: str, dataset_class: Type[BaseReferenceDataset]) -> None:
        """注册新的数据集类型
        
        Args:
            dataset_type: 数据集类型名称
            dataset_class: 数据集实现类
        """
        cls.DATASET_REGISTRY[dataset_type.lower()] = dataset_class
        logging.info(f"已注册数据集类型: {dataset_type}")