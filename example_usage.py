#!/usr/bin/env python3
"""示例：如何使用修复后的参考数据集功能"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def example_auto_detection():
    """示例：自动检测数据集类型"""
    print("示例1: 自动检测数据集类型")
    print("=" * 50)
    
    # 模拟命令行参数
    args = {
        'aux_auto_detect': True,
        'auxiliary_data_path': '/data1/open_datasets/flickr8k',  # 替换为实际路径
        'clip_use_reference_data': True,
        'batch_size': 32,
        'clip_num_workers': 4,
        'clip_pin_memory': True
    }
    
    print(f"数据集路径: {args['auxiliary_data_path']}")
    print(f"启用自动检测: {args['aux_auto_detect']}")
    print("\n使用命令:")
    print("python main_clip.py --aux_auto_detect --auxiliary_data_path /path/to/dataset")
    print()

def example_manual_type():
    """示例：手动指定数据集类型"""
    print("示例2: 手动指定数据集类型")
    print("=" * 50)
    
    # 模拟命令行参数
    args = {
        'aux_auto_detect': False,
        'aux_dataset_type': 'imagenet',
        'auxiliary_data_path': '/data1/open_datasets/imagenet',  # 替换为实际路径
        'aux_split': 'val',
        'clip_use_reference_data': True,
        'batch_size': 32,
        'clip_num_workers': 4,
        'clip_pin_memory': True
    }
    
    print(f"数据集类型: {args['aux_dataset_type']}")
    print(f"数据集路径: {args['auxiliary_data_path']}")
    print(f"数据集分割: {args['aux_split']}")
    print("\n使用命令:")
    print("python main_clip.py --aux_dataset_type imagenet --auxiliary_data_path /path/to/imagenet --aux_split val")
    print()

def example_sample_limitation():
    """示例：限制参考数据集样本数量"""
    print("示例3: 限制参考数据集样本数量")
    print("=" * 50)
    
    # 模拟命令行参数
    args = {
        'aux_auto_detect': True,
        'auxiliary_data_path': '/data1/open_datasets/flickr8k',  # 替换为实际路径
        'aux_num_samples': 1000,  # 只使用1000个样本
        'clip_use_reference_data': True,
        'batch_size': 32,
        'clip_num_workers': 4,
        'clip_pin_memory': True
    }
    
    print(f"数据集路径: {args['auxiliary_data_path']}")
    print(f"样本数量限制: {args['aux_num_samples']}")
    print("\n使用命令:")
    print("python main_clip.py --aux_auto_detect --auxiliary_data_path /path/to/dataset --aux_num_samples 1000")
    print()

def example_type_hint():
    """示例：使用类型提示辅助自动检测"""
    print("示例4: 使用类型提示辅助自动检测")
    print("=" * 50)
    
    # 模拟命令行参数
    args = {
        'aux_auto_detect': True,
        'aux_type_hint': 'imagenet',  # 提示可能是imagenet
        'auxiliary_data_path': '/data1/open_datasets/custom_dataset',  # 替换为实际路径
        'clip_use_reference_data': True,
        'batch_size': 32,
        'clip_num_workers': 4,
        'clip_pin_memory': True
    }
    
    print(f"数据集路径: {args['auxiliary_data_path']}")
    print(f"类型提示: {args['aux_type_hint']}")
    print(f"启用自动检测: {args['aux_auto_detect']}")
    print("\n使用命令:")
    print("python main_clip.py --aux_auto_detect --aux_type_hint imagenet --auxiliary_data_path /path/to/dataset")
    print()

def main():
    """运行所有示例"""
    print("参考数据集路径判别逻辑优化 - 使用示例")
    print("=" * 60)
    print()
    
    example_auto_detection()
    example_manual_type()
    example_sample_limitation()
    example_type_hint()
    
    print("注意事项:")
    print("1. 请将示例路径替换为实际的数据集路径")
    print("2. 确保数据集路径存在且具有正确的目录结构")
    print("3. 对于ImageNet，确保包含train和val目录")
    print("4. 对于Flickr8k，确保包含images目录和captions.txt文件")
    print("5. 训练时观察日志输出，确认数据集类型检测正确")
    print("6. 监控训练日志中的pos_cos和neg_cos指标，确保不为0")

if __name__ == "__main__":
    main()