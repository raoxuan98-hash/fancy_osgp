"""
测试数据集工厂功能
"""

import os
import unittest
import logging
from models.reference_dataset import (
    ReferenceDatasetFactory, 
    ImageNetRefDataset, 
    Flickr8kRefDataset,
    DatasetLoadError,
    DatasetDetectionError
)
from tests.reference_dataset.test_utils import TempDatasetDir


class TestReferenceDatasetFactory(unittest.TestCase):
    """测试ReferenceDatasetFactory类的功能"""
    
    def setUp(self):
        """测试前的设置"""
        logging.basicConfig(level=logging.INFO)
    
    def test_create_imagenet_dataset(self):
        """测试创建ImageNet数据集"""
        with TempDatasetDir("imagenet", num_classes=3, images_per_class=2) as temp_dir:
            # 创建数据集
            dataset = ReferenceDatasetFactory.create_dataset(
                dataset_type="imagenet",
                dataset_path=temp_dir,
                split="val",
                num_samples=5
            )
            
            # 验证数据集类型
            self.assertIsInstance(dataset, ImageNetRefDataset)
            
            # 验证数据集大小
            self.assertLessEqual(len(dataset), 5)
            self.assertGreater(len(dataset), 0)
            
            # 测试获取样本
            img, label = dataset[0]
            self.assertIsNotNone(img)
            self.assertIsInstance(label, int)
            
            # 测试获取标签和提示词
            labels, prompts = dataset.return_labels_and_prompts()
            # ImageNet数据集应该有类别映射
            if hasattr(dataset, 'class_to_idx'):
                self.assertEqual(len(labels), len(dataset.class_to_idx))
            self.assertEqual(len(prompts), len(labels))
    
    def test_create_flickr8k_dataset(self):
        """测试创建Flickr8k数据集"""
        with TempDatasetDir("flickr8k", num_images=5) as temp_dir:
            # 创建数据集
            dataset = ReferenceDatasetFactory.create_dataset(
                dataset_type="flickr8k",
                dataset_path=temp_dir,
                num_samples=3
            )
            
            # 验证数据集类型
            self.assertIsInstance(dataset, Flickr8kRefDataset)
            
            # 验证数据集大小
            self.assertLessEqual(len(dataset), 3)
            self.assertGreater(len(dataset), 0)
            
            # 测试获取样本
            img, label = dataset[0]
            self.assertIsNotNone(img)
            self.assertIsInstance(label, int)
            
            # 测试获取标签和提示词
            labels, prompts = dataset.return_labels_and_prompts()
            self.assertEqual(len(labels), len(dataset))
            self.assertEqual(len(prompts), len(labels))
    
    def test_create_dataset_invalid_type(self):
        """测试创建不支持的数据集类型"""
        with TempDatasetDir("imagenet") as temp_dir:
            with self.assertRaises(ValueError) as context:
                ReferenceDatasetFactory.create_dataset(
                    dataset_type="invalid_type",
                    dataset_path=temp_dir
                )
            
            self.assertIn("不支持的数据集类型", str(context.exception))
            self.assertIn("invalid_type", str(context.exception))
    
    def test_create_dataset_invalid_path(self):
        """测试创建数据集时路径无效"""
        invalid_path = "/path/that/does/not/exist"
        
        with self.assertRaises(DatasetLoadError):
            ReferenceDatasetFactory.create_dataset(
                dataset_type="imagenet",
                dataset_path=invalid_path
            )
    
    def test_create_dataset_auto_detect_imagenet(self):
        """测试自动检测并创建ImageNet数据集"""
        with TempDatasetDir("imagenet", num_classes=3, images_per_class=2) as temp_dir:
            # 自动检测并创建数据集
            dataset = ReferenceDatasetFactory.create_dataset_auto_detect(
                dataset_path=temp_dir,
                split="val",
                num_samples=5
            )
            
            # 验证数据集类型
            self.assertIsInstance(dataset, ImageNetRefDataset)
            
            # 验证数据集大小
            self.assertLessEqual(len(dataset), 5)
            self.assertGreater(len(dataset), 0)
    
    def test_create_dataset_auto_detect_flickr8k(self):
        """测试自动检测并创建Flickr8k数据集"""
        with TempDatasetDir("flickr8k", num_images=5) as temp_dir:
            # 自动检测并创建数据集
            dataset = ReferenceDatasetFactory.create_dataset_auto_detect(
                dataset_path=temp_dir,
                num_samples=3
            )
            
            # 验证数据集类型
            self.assertIsInstance(dataset, Flickr8kRefDataset)
            
            # 验证数据集大小
            self.assertLessEqual(len(dataset), 3)
            self.assertGreater(len(dataset), 0)
    
    def test_create_dataset_auto_detect_with_hint(self):
        """测试使用类型提示的自动检测"""
        with TempDatasetDir("mixed") as temp_dir:  # 使用混合结构，无法自动检测
            # 使用类型提示
            dataset = ReferenceDatasetFactory.create_dataset_auto_detect(
                dataset_path=temp_dir,
                type_hint="flickr8k",
                num_samples=2
            )
            
            # 应该创建Flickr8k数据集（即使结构不完整，也会尝试创建）
            # 这里可能会失败，但重要的是它尝试了正确的类型
            self.assertIsInstance(dataset, Flickr8kRefDataset)
    
    def test_create_dataset_auto_detect_failure(self):
        """测试自动检测失败的情况"""
        with TempDatasetDir("mixed") as temp_dir:
            with self.assertRaises(DatasetLoadError):
                ReferenceDatasetFactory.create_dataset_auto_detect(
                    dataset_path=temp_dir
                )
    
    def test_register_dataset(self):
        """测试注册新的数据集类型"""
        # 创建一个简单的测试数据集类
        class TestDataset(ImageNetRefDataset):
            pass
        
        # 注册新类型
        ReferenceDatasetFactory.register_dataset("test_type", TestDataset)
        
        # 验证注册成功
        self.assertIn("test_type", ReferenceDatasetFactory.DATASET_REGISTRY)
        self.assertEqual(ReferenceDatasetFactory.DATASET_REGISTRY["test_type"], TestDataset)
        
        # 尝试创建新类型的数据集
        with TempDatasetDir("imagenet") as temp_dir:
            dataset = ReferenceDatasetFactory.create_dataset(
                dataset_type="test_type",
                dataset_path=temp_dir
            )
            self.assertIsInstance(dataset, TestDataset)
    
    def test_imagenet_dataset_splits(self):
        """测试ImageNet数据集的不同分割"""
        with TempDatasetDir("imagenet", num_classes=2, images_per_class=2) as temp_dir:
            # 测试训练集
            train_dataset = ReferenceDatasetFactory.create_dataset(
                dataset_type="imagenet",
                dataset_path=temp_dir,
                split="train"
            )
            self.assertIsInstance(train_dataset, ImageNetRefDataset)
            if hasattr(train_dataset, 'split'):
                self.assertEqual(train_dataset.split, "train")
            
            # 测试验证集
            val_dataset = ReferenceDatasetFactory.create_dataset(
                dataset_type="imagenet",
                dataset_path=temp_dir,
                split="val"
            )
            self.assertIsInstance(val_dataset, ImageNetRefDataset)
            if hasattr(val_dataset, 'split'):
                self.assertEqual(val_dataset.split, "val")
    
    def test_dataset_with_transform(self):
        """测试使用变换的数据集创建"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        with TempDatasetDir("flickr8k", num_images=3) as temp_dir:
            dataset = ReferenceDatasetFactory.create_dataset(
                dataset_type="flickr8k",
                dataset_path=temp_dir,
                transform=transform
            )
            
            # 验证变换已设置
            self.assertEqual(dataset.transform, transform)
            
            # 测试获取样本（应该应用变换）
            img, label = dataset[0]
            # 变换后的图像应该是tensor
            import torch
            self.assertIsInstance(img, torch.Tensor)
    
    def test_dataset_no_num_samples_limit(self):
        """测试不限制样本数量的数据集创建"""
        with TempDatasetDir("flickr8k", num_images=5) as temp_dir:
            dataset = ReferenceDatasetFactory.create_dataset(
                dataset_type="flickr8k",
                dataset_path=temp_dir
                # 不设置num_samples
            )
            
            # 应该包含所有样本
            self.assertEqual(len(dataset), 5)


if __name__ == "__main__":
    unittest.main()