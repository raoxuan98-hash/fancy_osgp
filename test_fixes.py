#!/usr/bin/env python3
"""测试修复后的参考数据集功能"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_aux_num_samples():
    """测试aux_num_samples参数是否正确传递"""
    print("测试1: 验证aux_num_samples参数传递...")
    
    # 创建临时目录结构模拟flickr8k数据集
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建images目录
        images_dir = os.path.join(temp_dir, "images")
        os.makedirs(images_dir)
        
        # 创建一些虚拟图片文件
        for i in range(10):
            img_path = os.path.join(images_dir, f"img_{i:03d}.jpg")
            with open(img_path, 'w') as f:
                f.write("fake image content")
        
        # 创建captions.txt文件
        captions_file = os.path.join(temp_dir, "captions.txt")
        with open(captions_file, 'w') as f:
            f.write("image,caption\n")
            for i in range(10):
                f.write(f"img_{i:03d}.jpg,Caption for image {i}\n")
        
        # 测试数据集创建
        try:
            from models.reference_dataset import ReferenceDatasetFactory
            
            # 测试不限制样本数量
            dataset_all = ReferenceDatasetFactory.create_dataset(
                dataset_type="flickr8k",
                dataset_path=temp_dir,
                transform=None,
                num_samples=None
            )
            print(f"  不限制样本数量: {len(dataset_all)} 个样本")
            
            # 测试限制样本数量为5
            dataset_limited = ReferenceDatasetFactory.create_dataset(
                dataset_type="flickr8k",
                dataset_path=temp_dir,
                transform=None,
                num_samples=5
            )
            print(f"  限制样本数量为5: {len(dataset_limited)} 个样本")
            
            if len(dataset_limited) == 5:
                print("  ✓ aux_num_samples参数正确传递和使用")
                return True
            else:
                print("  ✗ aux_num_samples参数未正确传递或使用")
                return False
                
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            return False

def test_cosine_metrics():
    """测试余弦相似度指标计算"""
    print("\n测试2: 验证余弦相似度指标计算...")
    
    try:
        from models.clip_utils import build_metric_smoothers
        
        # 创建指标平滑器
        monitor_ema = build_metric_smoothers(alpha=0.9)
        
        # 检查是否包含我们需要的指标
        required_metrics = ["input_feature_positive_cosine", "input_feature_negative_cosine"]
        missing_metrics = [m for m in required_metrics if m not in monitor_ema]
        
        if missing_metrics:
            print(f"  ✗ 缺少指标: {missing_metrics}")
            return False
        else:
            print("  ✓ 包含所有必需的余弦相似度指标")
            
            # 测试指标更新
            import torch
            monitor_ema["input_feature_positive_cosine"].update(0.8)
            monitor_ema["input_feature_negative_cosine"].update(0.3)
            
            pos_cos = monitor_ema["input_feature_positive_cosine"].get()
            neg_cos = monitor_ema["input_feature_negative_cosine"].get()
            
            print(f"  正样本余弦相似度: {pos_cos:.4f}")
            print(f"  负样本余弦相似度: {neg_cos:.4f}")
            
            if pos_cos > 0 and neg_cos > 0:
                print("  ✓ 余弦相似度指标可以正确更新")
                return True
            else:
                print("  ✗ 余弦相似度指标更新异常")
                return False
                
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False

def test_config_parsing():
    """测试配置解析"""
    print("\n测试3: 验证配置解析...")
    
    try:
        from models.config import ReferenceConfig
        
        # 测试新配置选项
        config = ReferenceConfig(
            enabled=True,
            dataset_type="flickr8k",
            dataset_path="/fake/path",
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            auto_detect=True,
            type_hint="flickr8k",
            num_samples=100,
            split="val"
        )
        
        print(f"  auto_detect: {config.auto_detect}")
        print(f"  type_hint: {config.type_hint}")
        print(f"  num_samples: {config.num_samples}")
        print(f"  split: {config.split}")
        
        if (config.auto_detect and config.type_hint == "flickr8k" and 
            config.num_samples == 100 and config.split == "val"):
            print("  ✓ 新配置选项正确解析")
            return True
        else:
            print("  ✗ 新配置选项解析错误")
            return False
            
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始测试修复后的功能...\n")
    
    results = []
    results.append(test_aux_num_samples())
    results.append(test_cosine_metrics())
    results.append(test_config_parsing())
    
    print(f"\n测试结果: {sum(results)}/{len(results)} 通过")
    
    if all(results):
        print("✓ 所有测试通过，修复成功！")
        return 0
    else:
        print("✗ 部分测试失败，需要进一步检查")
        return 1

if __name__ == "__main__":
    sys.exit(main())