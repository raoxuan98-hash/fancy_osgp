"""
测试配置参数解析功能
"""

import os
import unittest
from models.config import ReferenceConfig
from models.subspace_lora_clip_learner import SubspaceLoRAClipLearner


class TestConfigParsing(unittest.TestCase):
    """测试配置参数解析功能"""
    
    def test_reference_config_basic(self):
        """测试基本参考数据集配置"""
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
    
    def test_reference_config_with_new_options(self):
        """测试包含新选项的参考数据集配置"""
        config = ReferenceConfig(
            enabled=True,
            dataset_type="flickr8k",
            dataset_path="/data/flickr8k",
            batch_size=16,
            num_workers=2,
            pin_memory=False,
            auto_detect=True,
            type_hint="flickr8k",
            num_samples=100,
            split="val"
        )
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.dataset_type, "flickr8k")
        self.assertEqual(config.dataset_path, "/data/flickr8k")
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.num_workers, 2)
        self.assertFalse(config.pin_memory)
        
        # 新选项
        self.assertTrue(config.auto_detect)
        self.assertEqual(config.type_hint, "flickr8k")
        self.assertEqual(config.num_samples, 100)
        self.assertEqual(config.split, "val")
    
    def test_subspace_learner_config_parsing(self):
        """测试SubspaceLoRAClipLearner的配置解析"""
        # 测试基本配置
        args = {
            "optimizer": "adamw",
            "lrate": 5e-4,
            "weight_decay": 0.1,
            "warmup_steps": 0,
            "iterations": 800,
            "batch_size": 32,
            "log_interval": 10,
            "ema_alpha": 0.9,
            "gamma_kd": 5.0,
            "gamma_norm": 0.1,
            "kl_gamma": 1.0,
            "l2_protection": False,
            "l2_protection_lambda": 1.0,
            "clip_use_reference_data": True,
            "aux_dataset_type": "imagenet",
            "auxiliary_data_path": "/data/imagenet",
            "clip_num_workers": 4,
            "clip_pin_memory": True,
            "reference_batch_size": 32,
            "aux_auto_detect": False,
            "aux_type_hint": None,
            "aux_num_samples": None,
            "aux_split": "val"
        }
        
        try:
            optim_cfg, loop_cfg, reg_cfg, reference_cfg = SubspaceLoRAClipLearner._build_configs(args)
            
            # 验证参考数据集配置
            self.assertTrue(reference_cfg.enabled)
            self.assertEqual(reference_cfg.dataset_type, "imagenet")
            self.assertEqual(reference_cfg.dataset_path, "/data/imagenet")
            self.assertEqual(reference_cfg.batch_size, 32)
            self.assertEqual(reference_cfg.num_workers, 4)
            self.assertTrue(reference_cfg.pin_memory)
            self.assertFalse(reference_cfg.auto_detect)
            self.assertIsNone(reference_cfg.type_hint)
            self.assertIsNone(reference_cfg.num_samples)
            self.assertEqual(reference_cfg.split, "val")
            
        except Exception as e:
            self.fail(f"配置解析失败: {e}")
    
    def test_subspace_learner_config_auto_detect(self):
        """测试SubspaceLoRAClipLearner的自动检测配置"""
        args = {
            "optimizer": "adamw",
            "lrate": 5e-4,
            "weight_decay": 0.1,
            "warmup_steps": 0,
            "iterations": 800,
            "batch_size": 32,
            "log_interval": 10,
            "ema_alpha": 0.9,
            "gamma_kd": 5.0,
            "gamma_norm": 0.1,
            "kl_gamma": 1.0,
            "l2_protection": False,
            "l2_protection_lambda": 1.0,
            "clip_use_reference_data": True,
            "aux_dataset_type": "auto",  # 启用自动检测
            "auxiliary_data_path": "/data/flickr8k",
            "clip_num_workers": 4,
            "clip_pin_memory": True,
            "reference_batch_size": 32,
            "aux_auto_detect": True,  # 显式启用自动检测
            "aux_type_hint": "flickr8k",  # 提供类型提示
            "aux_num_samples": 100,  # 限制样本数量
            "aux_split": "val"
        }
        
        try:
            optim_cfg, loop_cfg, reg_cfg, reference_cfg = SubspaceLoRAClipLearner._build_configs(args)
            
            # 验证自动检测配置
            self.assertTrue(reference_cfg.enabled)
            self.assertEqual(reference_cfg.dataset_type, "auto")  # 应该是"auto"
            self.assertEqual(reference_cfg.dataset_path, "/data/flickr8k")
            self.assertTrue(reference_cfg.auto_detect)
            self.assertEqual(reference_cfg.type_hint, "flickr8k")
            self.assertEqual(reference_cfg.num_samples, 100)
            self.assertEqual(reference_cfg.split, "val")
            
        except Exception as e:
            self.fail(f"自动检测配置解析失败: {e}")
    
    def test_subspace_learner_config_disabled(self):
        """测试禁用参考数据集的配置"""
        args = {
            "optimizer": "adamw",
            "lrate": 5e-4,
            "weight_decay": 0.1,
            "warmup_steps": 0,
            "iterations": 800,
            "batch_size": 32,
            "log_interval": 10,
            "ema_alpha": 0.9,
            "gamma_kd": 5.0,
            "gamma_norm": 0.1,
            "kl_gamma": 1.0,
            "l2_protection": False,
            "l2_protection_lambda": 1.0,
            "clip_use_reference_data": False,  # 禁用参考数据集
            "aux_dataset_type": "imagenet",
            "auxiliary_data_path": "/data/imagenet",
            "clip_num_workers": 4,
            "clip_pin_memory": True,
            "reference_batch_size": 32,
            "aux_auto_detect": False,
            "aux_type_hint": None,
            "aux_num_samples": None,
            "aux_split": "val"
        }
        
        try:
            optim_cfg, loop_cfg, reg_cfg, reference_cfg = SubspaceLoRAClipLearner._build_configs(args)
            
            # 验证禁用配置
            self.assertFalse(reference_cfg.enabled)
            
        except Exception as e:
            self.fail(f"禁用配置解析失败: {e}")
    
    def test_main_clip_argument_parsing(self):
        """测试main_clip.py的参数解析"""
        # 这里我们只能测试参数解析器的构建，因为实际解析需要命令行参数
        try:
            from main_clip import build_parser
            parser = build_parser()
            
            # 验证解析器包含必要的参数
            actions = [action.dest for action in parser._actions]
            
            # 检查新增的参数
            self.assertIn("aux_auto_detect", actions)
            self.assertIn("aux_type_hint", actions)
            self.assertIn("aux_num_samples", actions)
            self.assertIn("aux_split", actions)
            
            # 检查原有参数
            self.assertIn("aux_dataset_type", actions)
            self.assertIn("auxiliary_data_path", actions)
            self.assertIn("clip_use_reference_data", actions)
            
        except ImportError:
            self.skipTest("无法导入main_clip模块")
        except Exception as e:
            self.fail(f"参数解析器构建失败: {e}")
    
    def test_config_defaults(self):
        """测试配置默认值"""
        # 测试ReferenceConfig的默认值
        config = ReferenceConfig(
            enabled=True,
            dataset_type="imagenet",
            dataset_path="/data/test",
            batch_size=32,
            num_workers=4,
            pin_memory=True
        )
        
        # 新选项应该有合理的默认值
        self.assertFalse(config.auto_detect)  # 默认不启用自动检测
        self.assertIsNone(config.type_hint)  # 默认无类型提示
        self.assertIsNone(config.num_samples)  # 默认无样本限制
        self.assertEqual(config.split, "val")  # 默认使用验证集
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试必需参数
        with self.assertRaises(TypeError):
            # 缺少必需参数
            ReferenceConfig(
                enabled=True,
                dataset_type="imagenet",
                dataset_path="/data/test",
                batch_size=32,
                num_workers=4,
                pin_memory=True
                # 这些是必需的参数，但为了测试类型验证，我们提供所有必需参数
            )
        
        # 测试类型验证
        try:
            config = ReferenceConfig(
                enabled=True,  # bool
                dataset_type="imagenet",  # str
                dataset_path="/data/test",  # str
                batch_size=32,  # int
                num_workers=4,  # int
                pin_memory=True,  # bool
                auto_detect=True,  # bool
                type_hint="imagenet",  # str or None
                num_samples=100,  # int or None
                split="val"  # str
            )
            # 如果没有异常，说明类型正确
            self.assertTrue(config.enabled)
        except Exception as e:
            self.fail(f"配置验证失败: {e}")


if __name__ == "__main__":
    unittest.main()