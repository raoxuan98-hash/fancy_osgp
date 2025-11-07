"""
测试数据集类型自动检测功能
"""

import os
import unittest
import logging
import tempfile
import shutil
from models.reference_dataset import (
    DatasetTypeDetector,
    PathValidator,
    DatasetDetectionError,
    DatasetPathError
)
# 使用相对导入
try:
    from .test_utils import (
        TempDatasetDir,
        validate_dataset_structure
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tests.reference_dataset.test_utils import (
        TempDatasetDir,
        validate_dataset_structure
    )


class TestDatasetTypeDetector(unittest.TestCase):
    """测试DatasetTypeDetector类的功能"""
    
    def setUp(self):
        """测试前的设置"""
        # 设置日志级别
        logging.basicConfig(level=logging.INFO)
    
    def test_detect_by_path_keywords_imagenet(self):
        """测试通过路径关键字检测ImageNet数据集"""
        # 测试各种ImageNet相关的路径
        imagenet_paths = [
            "/data/imagenet",
            "/datasets/ILSVRC2012",
            "/path/to/image_net",
            "/some/imagenet_data"
        ]
        
        for path in imagenet_paths:
            with self.subTest(path=path):
                detected_type = DatasetTypeDetector.detect_by_path_keywords(path)
                self.assertEqual(detected_type, "imagenet", 
                                f"Failed to detect ImageNet in path: {path}")
    
    def test_detect_by_path_keywords_flickr8k(self):
        """测试通过路径关键字检测Flickr8k数据集"""
        # 测试各种Flickr8k相关的路径
        flickr8k_paths = [
            "/data/flickr8k",
            "/datasets/flickr_8k",
            "/path/to/flickr8k_data"
        ]
        
        for path in flickr8k_paths:
            with self.subTest(path=path):
                detected_type = DatasetTypeDetector.detect_by_path_keywords(path)
                self.assertEqual(detected_type, "flickr8k", 
                                f"Failed to detect Flickr8k in path: {path}")
    
    def test_detect_by_path_keywords_unknown(self):
        """测试未知路径的关键字检测"""
        unknown_paths = [
            "/data/unknown_dataset",
            "/datasets/some_other_data",
            "/path/to/no_keywords"
        ]
        
        for path in unknown_paths:
            with self.subTest(path=path):
                detected_type = DatasetTypeDetector.detect_by_path_keywords(path)
                self.assertIsNone(detected_type, 
                                 f"Should not detect any type for path: {path}")
    
    def test_detect_by_directory_structure_imagenet(self):
        """测试通过目录结构检测ImageNet数据集"""
        with TempDatasetDir("imagenet", num_classes=3, images_per_class=2) as temp_dir:
            # 验证数据集结构创建成功
            structure = validate_dataset_structure(temp_dir, "imagenet")
            self.assertTrue(structure["train_exists"])
            self.assertTrue(structure["val_exists"])
            self.assertTrue(structure["train_has_classes"])
            self.assertTrue(structure["val_has_classes"])
            
            # 测试检测
            detected_type = DatasetTypeDetector.detect_by_directory_structure(temp_dir)
            self.assertEqual(detected_type, "imagenet")
    
    def test_detect_by_directory_structure_flickr8k(self):
        """测试通过目录结构检测Flickr8k数据集"""
        with TempDatasetDir("flickr8k", num_images=5) as temp_dir:
            # 验证数据集结构创建成功
            structure = validate_dataset_structure(temp_dir, "flickr8k")
            self.assertTrue(structure["images_exists"])
            self.assertTrue(structure["captions_exists"])
            self.assertTrue(structure["images_has_files"])
            self.assertTrue(structure["captions_has_content"])
            
            # 测试检测
            detected_type = DatasetTypeDetector.detect_by_directory_structure(temp_dir)
            self.assertEqual(detected_type, "flickr8k")
    
    def test_detect_by_directory_structure_invalid(self):
        """测试无效目录结构的检测"""
        with TempDatasetDir("mixed") as temp_dir:
            # 测试检测
            detected_type = DatasetTypeDetector.detect_by_directory_structure(temp_dir)
            self.assertIsNone(detected_type)
    
    def test_detect_by_directory_structure_nonexistent(self):
        """测试不存在路径的目录结构检测"""
        nonexistent_path = "/path/that/does/not/exist"
        detected_type = DatasetTypeDetector.detect_by_directory_structure(nonexistent_path)
        self.assertIsNone(detected_type)
    
    def test_detect_dataset_type_with_hint(self):
        """测试使用类型提示的数据集检测"""
        # 即使路径不包含关键字，类型提示也应该生效
        unknown_path = "/data/unknown_dataset"
        
        # 测试ImageNet提示
        detected_type = DatasetTypeDetector.detect_dataset_type(unknown_path, hint="imagenet")
        self.assertEqual(detected_type, "imagenet")
        
        # 测试Flickr8k提示
        detected_type = DatasetTypeDetector.detect_dataset_type(unknown_path, hint="flickr8k")
        self.assertEqual(detected_type, "flickr8k")
    
    def test_detect_dataset_type_auto_imagenet(self):
        """测试自动检测ImageNet数据集（关键字优先）"""
        with TempDatasetDir("imagenet") as temp_dir:
            detected_type = DatasetTypeDetector.detect_dataset_type(temp_dir)
            self.assertEqual(detected_type, "imagenet")
    
    def test_detect_dataset_type_auto_flickr8k(self):
        """测试自动检测Flickr8k数据集（关键字优先）"""
        with TempDatasetDir("flickr8k") as temp_dir:
            detected_type = DatasetTypeDetector.detect_dataset_type(temp_dir)
            self.assertEqual(detected_type, "flickr8k")
    
    def test_detect_dataset_type_structure_fallback(self):
        """测试当关键字检测失败时，回退到结构检测"""
        # 使用不包含关键字的路径，但有正确的目录结构
        with TempDatasetDir("imagenet") as temp_dir:
            # 重命名目录以移除关键字
            import tempfile
            import shutil
            temp_no_keyword = tempfile.mkdtemp(prefix="test_data_")
            try:
                shutil.move(temp_dir, temp_no_keyword)
                dataset_path = temp_no_keyword
                
                detected_type = DatasetTypeDetector.detect_dataset_type(dataset_path)
                self.assertEqual(detected_type, "imagenet")
            finally:
                if os.path.exists(temp_no_keyword):
                    shutil.rmtree(temp_no_keyword)
    
    def test_detect_dataset_type_failure(self):
        """测试数据集检测失败的情况"""
        with TempDatasetDir("mixed") as temp_dir:
            with self.assertRaises(DatasetDetectionError):
                DatasetTypeDetector.detect_dataset_type(temp_dir)


class TestPathValidator(unittest.TestCase):
    """测试PathValidator类的功能"""
    
    def test_validate_path_exists_success(self):
        """测试验证存在的路径"""
        with TempDatasetDir("imagenet") as temp_dir:
            # 应该不抛出异常
            PathValidator.validate_path_exists(temp_dir)
    
    def test_validate_path_exists_failure(self):
        """测试验证不存在的路径"""
        nonexistent_path = "/path/that/does/not/exist"
        with self.assertRaises(DatasetPathError):
            PathValidator.validate_path_exists(nonexistent_path)
    
    def test_validate_directory_readable_success(self):
        """测试验证可读的目录"""
        with TempDatasetDir("flickr8k") as temp_dir:
            # 应该不抛出异常
            PathValidator.validate_directory_readable(temp_dir)
    
    def test_validate_directory_readable_not_dir(self):
        """测试验证非目录路径"""
        with TempDatasetDir("flickr8k") as temp_dir:
            file_path = os.path.join(temp_dir, "captions.txt")
            with self.assertRaises(DatasetPathError):
                PathValidator.validate_directory_readable(file_path)
    
    def test_validate_file_readable_success(self):
        """测试验证可读的文件"""
        with TempDatasetDir("flickr8k") as temp_dir:
            file_path = os.path.join(temp_dir, "captions.txt")
            # 应该不抛出异常
            PathValidator.validate_file_readable(file_path)
    
    def test_validate_file_readable_not_file(self):
        """测试验证非文件路径"""
        with TempDatasetDir("flickr8k") as temp_dir:
            with self.assertRaises(DatasetPathError):
                PathValidator.validate_file_readable(temp_dir)
    
    def test_validate_imagenet_structure_success(self):
        """测试验证ImageNet结构成功"""
        with TempDatasetDir("imagenet") as temp_dir:
            result = PathValidator.validate_imagenet_structure(temp_dir)
            self.assertTrue(result)
    
    def test_validate_imagenet_structure_failure(self):
        """测试验证ImageNet结构失败"""
        with TempDatasetDir("mixed") as temp_dir:
            result = PathValidator.validate_imagenet_structure(temp_dir)
            self.assertFalse(result)
    
    def test_validate_flickr8k_structure_success(self):
        """测试验证Flickr8k结构成功"""
        with TempDatasetDir("flickr8k") as temp_dir:
            result = PathValidator.validate_flickr8k_structure(temp_dir)
            self.assertTrue(result)
    
    def test_validate_flickr8k_structure_failure(self):
        """测试验证Flickr8k结构失败"""
        with TempDatasetDir("imagenet") as temp_dir:
            result = PathValidator.validate_flickr8k_structure(temp_dir)
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()