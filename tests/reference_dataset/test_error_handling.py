"""
测试错误处理机制
"""

import os
import unittest
import logging
from models.reference_dataset import (
    DatasetTypeDetector, 
    ReferenceDatasetFactory,
    PathValidator,
    DatasetDetectionError,
    DatasetPathError,
    DatasetLoadError
)
from tests.reference_dataset.test_utils import TempDatasetDir


class TestErrorHandling(unittest.TestCase):
    """测试各种错误处理机制"""
    
    def setUp(self):
        """测试前的设置"""
        logging.basicConfig(level=logging.INFO)
    
    def test_dataset_detection_error(self):
        """测试数据集检测错误"""
        # 测试无法识别的路径
        with TempDatasetDir("mixed") as temp_dir:
            with self.assertRaises(DatasetDetectionError) as context:
                DatasetTypeDetector.detect_dataset_type(temp_dir)
            
            self.assertIn("无法确定数据集类型", str(context.exception))
            self.assertIn(temp_dir, str(context.exception))
    
    def test_dataset_path_error_nonexistent(self):
        """测试不存在路径的错误"""
        nonexistent_path = "/path/that/does/not/exist"
        
        # 测试路径验证
        with self.assertRaises(DatasetPathError) as context:
            PathValidator.validate_path_exists(nonexistent_path)
        
        self.assertIn("路径不存在", str(context.exception))
        self.assertIn(nonexistent_path, str(context.exception))
        
        # 测试目录验证
        with self.assertRaises(DatasetPathError) as context:
            PathValidator.validate_directory_readable(nonexistent_path)
        
        self.assertIn("路径不存在", str(context.exception))
        
        # 测试文件验证
        with self.assertRaises(DatasetPathError) as context:
            PathValidator.validate_file_readable(nonexistent_path)
        
        self.assertIn("路径不存在", str(context.exception))
    
    def test_dataset_path_error_not_directory(self):
        """测试非目录路径的错误"""
        with TempDatasetDir("flickr8k") as temp_dir:
            file_path = os.path.join(temp_dir, "captions.txt")
            
            with self.assertRaises(DatasetPathError) as context:
                PathValidator.validate_directory_readable(file_path)
            
            self.assertIn("路径不是目录", str(context.exception))
    
    def test_dataset_path_error_not_file(self):
        """测试非文件路径的错误"""
        with TempDatasetDir("flickr8k") as temp_dir:
            with self.assertRaises(DatasetPathError) as context:
                PathValidator.validate_file_readable(temp_dir)
            
            self.assertIn("路径不是文件", str(context.exception))
    
    def test_dataset_path_error_unreadable(self):
        """测试不可读路径的错误"""
        # 创建临时目录和文件
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            # 创建一个文件
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            
            # 移除读权限（仅在Unix系统上有效）
            if os.name != 'nt':  # 非Windows系统
                os.chmod(test_file, 0o000)
                
                with self.assertRaises(DatasetPathError) as context:
                    PathValidator.validate_file_readable(test_file)
                
                self.assertIn("文件不可读", str(context.exception))
                
                # 恢复权限以便清理
                os.chmod(test_file, 0o644)
            else:
                # Windows系统跳过权限测试
                self.skipTest("权限测试在Windows系统上不适用")
                
        finally:
            # 清理
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_dataset_load_error_invalid_type(self):
        """测试不支持的数据集类型加载错误"""
        with TempDatasetDir("imagenet") as temp_dir:
            with self.assertRaises(ValueError) as context:
                ReferenceDatasetFactory.create_dataset(
                    dataset_type="unsupported_type",
                    dataset_path=temp_dir
                )
            
            self.assertIn("不支持的数据集类型", str(context.exception))
            self.assertIn("unsupported_type", str(context.exception))
    
    def test_dataset_load_error_invalid_path(self):
        """测试无效路径的加载错误"""
        invalid_path = "/path/that/does/not/exist"
        
        with self.assertRaises(DatasetLoadError):
            ReferenceDatasetFactory.create_dataset(
                dataset_type="imagenet",
                dataset_path=invalid_path
            )
        
        with self.assertRaises(DatasetLoadError):
            ReferenceDatasetFactory.create_dataset_auto_detect(
                dataset_path=invalid_path
            )
    
    def test_dataset_load_error_incomplete_structure(self):
        """测试不完整结构的加载错误"""
        with TempDatasetDir("mixed") as temp_dir:
            # 尝试创建ImageNet数据集，但结构不完整
            with self.assertRaises(DatasetLoadError):
                ReferenceDatasetFactory.create_dataset(
                    dataset_type="imagenet",
                    dataset_path=temp_dir
                )
            
            # 尝试创建Flickr8k数据集，但结构不完整
            with self.assertRaises(DatasetLoadError):
                ReferenceDatasetFactory.create_dataset(
                    dataset_type="flickr8k",
                    dataset_path=temp_dir
                )
    
    def test_dataset_load_error_auto_detect_failure(self):
        """测试自动检测失败的加载错误"""
        with TempDatasetDir("mixed") as temp_dir:
            with self.assertRaises(DatasetLoadError) as context:
                ReferenceDatasetFactory.create_dataset_auto_detect(
                    dataset_path=temp_dir
                )
            
            self.assertIn("自动检测数据集类型失败", str(context.exception))
    
    def test_imagenet_dataset_missing_split(self):
        """测试ImageNet数据集缺少分割的错误"""
        # 创建只有train目录的ImageNet结构
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            # 只创建train目录
            train_dir = os.path.join(temp_dir, "train")
            os.makedirs(train_dir)
            class_dir = os.path.join(train_dir, "class_0000")
            os.makedirs(class_dir)
            
            # 创建一个图像文件
            from tests.reference_dataset.test_utils import create_dummy_image
            create_dummy_image(os.path.join(class_dir, "img.jpg"))
            
            # 尝试创建验证集数据集（应该失败）
            with self.assertRaises(DatasetLoadError):
                ReferenceDatasetFactory.create_dataset(
                    dataset_type="imagenet",
                    dataset_path=temp_dir,
                    split="val"
                )
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_flickr8k_dataset_missing_captions(self):
        """测试Flickr8k数据集缺少标题文件的错误"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            # 只创建images目录
            images_dir = os.path.join(temp_dir, "images")
            os.makedirs(images_dir)
            
            # 创建一个图像文件
            from tests.reference_dataset.test_utils import create_dummy_image
            create_dummy_image(os.path.join(images_dir, "img.jpg"))
            
            # 尝试创建数据集（应该失败，因为缺少captions.txt）
            with self.assertRaises(DatasetLoadError):
                ReferenceDatasetFactory.create_dataset(
                    dataset_type="flickr8k",
                    dataset_path=temp_dir
                )
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_flickr8k_dataset_empty_captions(self):
        """测试Flickr8k数据集空标题文件的错误"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            # 创建images目录
            images_dir = os.path.join(temp_dir, "images")
            os.makedirs(images_dir)
            
            # 创建一个图像文件
            from tests.reference_dataset.test_utils import create_dummy_image
            create_dummy_image(os.path.join(images_dir, "img.jpg"))
            
            # 创建空的captions.txt文件
            captions_file = os.path.join(temp_dir, "captions.txt")
            with open(captions_file, 'w') as f:
                f.write("image,caption\n")  # 只有标题行
            
            # 尝试创建数据集（应该失败，因为没有有效的标题）
            with self.assertRaises(DatasetLoadError):
                ReferenceDatasetFactory.create_dataset(
                    dataset_type="flickr8k",
                    dataset_path=temp_dir
                )
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_error_messages_clarity(self):
        """测试错误消息的清晰度"""
        # 测试各种错误消息是否包含有用信息
        test_cases = [
            # (函数, 参数, 预期消息片段)
            (
                PathValidator.validate_path_exists,
                ["/nonexistent/path"],
                "路径不存在"
            ),
            (
                lambda p: PathValidator.validate_directory_readable(p),
                ["/nonexistent/path"],
                "路径不存在"
            ),
            (
                lambda p: PathValidator.validate_file_readable(p),
                ["/nonexistent/path"],
                "路径不存在"
            ),
        ]
        
        for func, args, expected_msg in test_cases:
            with self.subTest(func=func.__name__ if hasattr(func, '__name__') else str(func)):
                try:
                    func(*args)
                    self.fail(f"应该抛出异常: {func.__name__ if hasattr(func, '__name__') else str(func)}")
                except DatasetPathError as e:
                    self.assertIn(expected_msg, str(e))
    
    def test_error_logging(self):
        """测试错误日志记录"""
        # 捕获日志
        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        try:
            # 触发一个会产生日志的操作
            with TempDatasetDir("imagenet") as temp_dir:
                # 重命名目录以避免关键字检测
                import tempfile
                import shutil
                temp_no_keyword = tempfile.mkdtemp(prefix="test_data_")
                try:
                    shutil.move(temp_dir, temp_no_keyword)
                    
                    # 这应该会产生日志信息
                    detected_type = DatasetTypeDetector.detect_dataset_type(temp_no_keyword)
                    
                    # 检查日志
                    log_contents = log_capture.getvalue()
                    self.assertIn("检测到数据集类型", log_contents)
                    
                finally:
                    if os.path.exists(temp_no_keyword):
                        shutil.rmtree(temp_no_keyword)
        finally:
            logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()