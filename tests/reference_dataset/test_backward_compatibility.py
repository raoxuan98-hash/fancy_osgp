"""
测试向后兼容性
"""

import os
import unittest
import sys
from unittest.mock import patch, MagicMock


class TestBackwardCompatibility(unittest.TestCase):
    """测试向后兼容性"""
    
    def test_main_clip_imports(self):
        """测试main_clip.py的导入兼容性"""
        try:
            # 测试能否正常导入main模块
            import main_clip
            self.assertTrue(hasattr(main_clip, 'main'))
            self.assertTrue(hasattr(main_clip, 'build_parser'))
        except ImportError as e:
            self.fail(f"无法导入main_clip模块: {e}")
    
    def test_trainer_clip_imports(self):
        """测试trainer_clip.py的导入兼容性"""
        try:
            # 测试能否正常导入trainer模块
            import trainer_clip
            self.assertTrue(hasattr(trainer_clip, 'train'))
        except ImportError as e:
            self.fail(f"无法导入trainer_clip模块: {e}")
    
    def test_subspace_lora_clip_alias(self):
        """测试SubspaceLoRA_CLIP别名兼容性"""
        try:
            from models.subspace_lora_clip import SubspaceLoRA_CLIP
            from models.subspace_lora_clip_learner import SubspaceLoRAClipLearner
            
            # 验证别名指向同一个类
            self.assertEqual(SubspaceLoRA_CLIP, SubspaceLoRAClipLearner)
        except ImportError as e:
            self.fail(f"无法导入SubspaceLoRA模块: {e}")
    
    def test_config_compatibility(self):
        """测试配置类的兼容性"""
        try:
            from models.config import ReferenceConfig
            
            # 测试原有参数仍然有效
            config = ReferenceConfig(
                enabled=True,
                dataset_type="imagenet",
                dataset_path="/data/imagenet",
                batch_size=32,
                num_workers=4,
                pin_memory=True
            )
            
            self.assertTrue(config.enabled)
            self.assertEqual(config.dataset_type, "imagenet")
            self.assertEqual(config.dataset_path, "/data/imagenet")
            self.assertEqual(config.batch_size, 32)
            self.assertEqual(config.num_workers, 4)
            self.assertTrue(config.pin_memory)
            
            # 测试新参数有默认值
            self.assertFalse(config.auto_detect)
            self.assertIsNone(config.type_hint)
            self.assertIsNone(config.num_samples)
            self.assertEqual(config.split, "val")
            
        except ImportError as e:
            self.fail(f"无法导入配置模块: {e}")
    
    def test_reference_dataset_factory_compatibility(self):
        """测试数据集工厂的兼容性"""
        try:
            from models.reference_dataset import ReferenceDatasetFactory
            
            # 验证原有数据集类型仍然支持
            self.assertIn("imagenet", ReferenceDatasetFactory.DATASET_REGISTRY)
            self.assertIn("flickr8k", ReferenceDatasetFactory.DATASET_REGISTRY)
            
            # 测试原有方法仍然存在
            self.assertTrue(hasattr(ReferenceDatasetFactory, 'create_dataset'))
            self.assertTrue(hasattr(ReferenceDatasetFactory, 'create_dataset_auto_detect'))
            self.assertTrue(hasattr(ReferenceDatasetFactory, 'register_dataset'))
            
        except ImportError as e:
            self.fail(f"无法导入数据集工厂模块: {e}")
    
    def test_old_flickr8k_compatibility(self):
        """测试原有Flickr8k实现的兼容性"""
        try:
            from utils.flickr8k_ref import Flickr8kRefDataset as OldFlickr8kRefDataset
            from models.reference_dataset import Flickr8kRefDataset as NewFlickr8kRefDataset
            
            # 验证两个类都存在
            self.assertIsNotNone(OldFlickr8kRefDataset)
            self.assertIsNotNone(NewFlickr8kRefDataset)
            
            # 验证它们有相似的方法
            old_methods = [method for method in dir(OldFlickr8kRefDataset) if not method.startswith('_')]
            new_methods = [method for method in dir(NewFlickr8kRefDataset) if not method.startswith('_')]
            
            # 检查关键方法是否存在
            common_methods = ['__len__', '__getitem__', 'return_labels_and_prompts']
            for method in common_methods:
                self.assertIn(method, old_methods, f"旧实现缺少方法: {method}")
                self.assertIn(method, new_methods, f"新实现缺少方法: {method}")
                
        except ImportError as e:
            self.fail(f"无法导入Flickr8k数据集模块: {e}")
    
    def test_argument_parser_compatibility(self):
        """测试参数解析器的兼容性"""
        try:
            from main_clip import build_parser
            
            parser = build_parser()
            
            # 验证原有参数仍然存在
            actions = {action.dest for action in parser._actions}
            
            required_old_args = {
                'aux_dataset_type', 'auxiliary_data_path', 'clip_use_reference_data',
                'clip_num_workers', 'clip_pin_memory', 'reference_batch_size'
            }
            
            missing_args = required_old_args - actions
            self.assertEqual(len(missing_args), 0, 
                           f"缺少原有参数: {missing_args}")
            
            # 验证新参数已添加
            new_args = {
                'aux_auto_detect', 'aux_type_hint', 'aux_num_samples', 'aux_split'
            }
            
            missing_new_args = new_args - actions
            self.assertEqual(len(missing_new_args), 0, 
                           f"缺少新参数: {missing_new_args}")
                
        except ImportError as e:
            self.fail(f"无法导入参数解析器: {e}")
    
    def test_training_manager_compatibility(self):
        """测试训练管理器的兼容性"""
        try:
            from models.training_and_reference import TrainingAndReferenceManager
            
            # 验证关键方法存在
            required_methods = [
                'initialise_reference_components',
                'next_reference_batch',
                'run_training_step',
                'run_training_loop'
            ]
            
            for method in required_methods:
                self.assertTrue(hasattr(TrainingAndReferenceManager, method),
                               f"TrainingAndReferenceManager缺少方法: {method}")
                
        except ImportError as e:
            self.fail(f"无法导入训练管理器模块: {e}")
    
    @patch('models.subspace_lora_clip_learner.CLIP_BaseNet')
    @patch('models.subspace_lora_clip_learner.ClipIncrementalDataManager')
    def test_learner_initialization_compatibility(self, mock_clip_manager, mock_clip_net):
        """测试学习器初始化的兼容性"""
        try:
            from models.subspace_lora_clip_learner import SubspaceLoRAClipLearner
            
            # 模拟必要的依赖
            mock_net_instance = MagicMock()
            mock_net_instance.valid_preprocess = MagicMock()
            mock_clip_net.return_value = mock_net_instance
            
            mock_manager_instance = MagicMock()
            mock_manager_instance.task_names = ['test_dataset']
            mock_clip_manager.return_value = mock_manager_instance
            
            # 测试最小参数集
            minimal_args = {
                'optimizer': 'adamw',
                'lrate': 5e-4,
                'weight_decay': 0.1,
                'warmup_steps': 0,
                'iterations': 100,
                'batch_size': 32,
                'log_interval': 10,
                'ema_alpha': 0.9,
                'gamma_kd': 0.0,
                'gamma_norm': 0.0,
                'kl_gamma': 0.0,
                'l2_protection': False,
                'l2_protection_lambda': 0.0,
                'clip_use_reference_data': False,
                'clip_dataset_sequence': ['test_dataset'],
                'clip_dataset_shuffle': False,
                'clip_dataset_seed': 0,
                'clip_num_workers': 4,
                'clip_pin_memory': True,
                'aux_dataset_type': 'imagenet',
                'auxiliary_data_path': '/data/test',
                'amp': False,
                'seed': 1993
            }
            
            # 这应该不会抛出异常
            learner = SubspaceLoRAClipLearner(minimal_args)
            self.assertIsNotNone(learner)
            
        except ImportError as e:
            self.fail(f"无法导入学习器模块: {e}")
        except Exception as e:
            self.fail(f"学习器初始化失败: {e}")
    
    def test_default_behavior_compatibility(self):
        """测试默认行为的兼容性"""
        try:
            # 测试默认情况下不启用新功能
            from models.config import ReferenceConfig
            
            config = ReferenceConfig(
                enabled=True,
                dataset_type="imagenet",
                dataset_path="/data/test",
                batch_size=32,
                num_workers=4,
                pin_memory=True
            )
            
            # 新功能应该默认关闭
            self.assertFalse(config.auto_detect)
            self.assertIsNone(config.type_hint)
            self.assertIsNone(config.num_samples)
            self.assertEqual(config.split, "val")  # 合理的默认值
            
        except Exception as e:
            self.fail(f"默认行为测试失败: {e}")
    
    def test_error_handling_compatibility(self):
        """测试错误处理的兼容性"""
        try:
            from models.reference_dataset import (
                DatasetDetectionError,
                DatasetPathError,
                DatasetLoadError
            )
            
            # 验证异常类仍然存在
            self.assertIsNotNone(DatasetDetectionError)
            self.assertIsNotNone(DatasetPathError)
            self.assertIsNotNone(DatasetLoadError)
            
            # 验证它们是Exception的子类
            self.assertTrue(issubclass(DatasetDetectionError, Exception))
            self.assertTrue(issubclass(DatasetPathError, Exception))
            self.assertTrue(issubclass(DatasetLoadError, Exception))
            
        except ImportError as e:
            self.fail(f"无法导入异常类: {e}")


if __name__ == "__main__":
    unittest.main()