#!/usr/bin/env python3
"""
简化的layer-wise特征蒸馏测试脚本
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.layerwise_distillation import (
    layerwise_feature_distillation_loss, 
    create_layer_weights
)

def test_layerwise_distillation_loss():
    """测试layer-wise特征蒸馏损失"""
    print("测试layer-wise特征蒸馏损失...")
    
    # 创建模拟特征
    num_layers = 4
    batch_size, feature_dim = 8, 768
    
    teacher_features = [torch.randn(batch_size, feature_dim) for _ in range(num_layers)]
    student_features = [torch.randn(batch_size, feature_dim) for _ in range(num_layers)]
    
    # 测试不同损失类型
    loss_types = ['mse', 'cosine', 'mse_cosine']
    
    for loss_type in loss_types:
        try:
            loss = layerwise_feature_distillation_loss(
                teacher_features, student_features, loss_type=loss_type
            )
            print(f"{loss_type}损失: {loss.item():.6f}")
        except Exception as e:
            print(f"❌ {loss_type}损失计算失败: {e}")
            return False
    
    print("✅ Layer-wise特征蒸馏损失测试通过!")
    return True

def test_create_layer_weights():
    """测试层权重创建"""
    print("\n测试层权重创建...")
    
    num_layers = 6
    strategies = ['uniform', 'linear', 'exponential']
    
    for strategy in strategies:
        try:
            weights = create_layer_weights(num_layers, strategy)
            print(f"{strategy}策略权重: {weights}")
            print(f"权重和: {sum(weights):.6f}")
        except Exception as e:
            print(f"❌ {strategy}策略权重创建失败: {e}")
            return False
    
    print("✅ 层权重创建测试通过!")
    return True

def test_integration():
    """集成测试"""
    print("\n进行集成测试...")
    
    try:
        # 创建模拟特征
        num_layers = 4
        batch_size, feature_dim = 4, 512
        
        teacher_features = [torch.randn(batch_size, feature_dim) for _ in range(num_layers)]
        student_features = [torch.randn(batch_size, feature_dim) for _ in range(num_layers)]
        
        # 创建权重
        layer_weights = create_layer_weights(len(teacher_features), 'linear')
        
        # 计算损失
        loss = layerwise_feature_distillation_loss(
            teacher_features, 
            student_features, 
            layer_weights,
            'mse_cosine'
        )
        
        print(f"集成测试损失: {loss.item():.6f}")
        print(f"教师特征层数: {len(teacher_features)}")
        print(f"学生特征层数: {len(student_features)}")
        print(f"层权重: {layer_weights}")
        
        print("✅ 集成测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def test_config_integration():
    """测试配置集成"""
    print("\n测试配置集成...")
    
    try:
        from models.config import RegularizationConfig
        
        # 测试默认配置
        config_default = RegularizationConfig(
            gamma_kd=1.0,
            gamma_norm=0.1,
            gamma_prior=0.5,
            l2_enabled=False,
            l2_lambda=0.0,
            bidirectional_kd=False
        )
        print(f"默认配置中的layerwise_kd_enabled: {config_default.layerwise_kd_enabled}")
        
        # 测试启用layer-wise蒸馏的配置
        config_layerwise = RegularizationConfig(
            gamma_kd=1.0,
            gamma_norm=0.1,
            gamma_prior=0.5,
            l2_enabled=False,
            l2_lambda=0.0,
            bidirectional_kd=False,
            layerwise_kd_enabled=True,
            layerwise_kd_weight=2.0,
            layerwise_kd_pooling="mean",
            layerwise_kd_loss_type="mse_cosine",
            layerwise_kd_weight_strategy="linear"
        )
        print(f"启用配置中的layerwise_kd_enabled: {config_layerwise.layerwise_kd_enabled}")
        print(f"layerwise_kd_weight: {config_layerwise.layerwise_kd_weight}")
        print(f"layerwise_kd_pooling: {config_layerwise.layerwise_kd_pooling}")
        print(f"layerwise_kd_loss_type: {config_layerwise.layerwise_kd_loss_type}")
        print(f"layerwise_kd_weight_strategy: {config_layerwise.layerwise_kd_weight_strategy}")
        
        print("✅ 配置集成测试通过!")
        return True
    except Exception as e:
        print(f"❌ 配置集成测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始测试layer-wise特征蒸馏实现...\n")
    
    tests = [
        test_layerwise_distillation_loss,
        test_create_layer_weights,
        test_integration,
        test_config_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 50)
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! Layer-wise特征蒸馏核心功能正常工作。")
        print("注意：特征收集器的完整测试需要实际的CLIP模型结构。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查实现。")
        return False

if __name__ == "__main__":
    main()