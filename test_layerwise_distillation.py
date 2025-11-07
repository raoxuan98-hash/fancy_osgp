#!/usr/bin/env python3
"""
测试layer-wise特征蒸馏实现的简单脚本
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.layerwise_distillation import (
    FeatureHook, 
    LayerwiseFeatureCollector, 
    layerwise_feature_distillation_loss, 
    create_layer_weights
)

class MockTransformerLayer(nn.Module):
    """模拟的Transformer层"""
    def __init__(self, feature_dim=768):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        x = self.linear(x)
        x = self.norm(x)
        return x

class MockVisionModel(nn.Module):
    """模拟的Vision Model"""
    def __init__(self, num_layers=12, feature_dim=768):
        super().__init__()
        self.layers = nn.ModuleList([
            MockTransformerLayer(feature_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        for layer in self.layers:
            x = layer(x)
        return x

def test_feature_hook():
    """测试特征钩子"""
    print("测试特征钩子...")
    
    # 创建模拟数据
    batch_size, seq_len, feature_dim = 4, 197, 768
    features = torch.randn(batch_size, seq_len, feature_dim)
    
    # 创建钩子
    hook = FeatureHook()
    
    # 模拟钩子调用
    mock_layer = MockTransformerLayer(feature_dim)
    output = mock_layer(features)
    hook.hook(mock_layer, (features,), output)
    
    # 测试池化
    mean_pooled = hook.get_pooled_features('mean')
    cls_pooled = hook.get_pooled_features('cls')
    max_pooled = hook.get_pooled_features('max')
    
    print(f"原始特征形状: {features.shape}")
    print(f"平均池化后形状: {mean_pooled.shape if mean_pooled is not None else None}")
    print(f"CLS token形状: {cls_pooled.shape if cls_pooled is not None else None}")
    print(f"最大池化后形状: {max_pooled.shape if max_pooled is not None else None}")
    
    if mean_pooled is not None and cls_pooled is not None and max_pooled is not None:
        print("✅ 特征钩子测试通过!")
        return True
    else:
        print("❌ 特征钩子测试失败!")
        return False

def test_layerwise_feature_collector():
    """测试layer-wise特征收集器"""
    print("\n测试layer-wise特征收集器...")
    
    try:
        # 创建模拟模型
        teacher_model = MockVisionModel(num_layers=6, feature_dim=768)
        student_model = MockVisionModel(num_layers=6, feature_dim=768)
        
        # 创建特征收集器
        teacher_collector = LayerwiseFeatureCollector(teacher_model, pooling_type='mean')
        student_collector = LayerwiseFeatureCollector(student_model, pooling_type='mean')
        
        # 创建模拟输入
        batch_size, seq_len, feature_dim = 2, 197, 768
        input_data = torch.randn(batch_size, seq_len, feature_dim)
        
        # 前向传播
        with torch.no_grad():
            teacher_output = teacher_model(input_data)
            student_output = student_model(input_data)
        
        # 获取特征
        teacher_features = teacher_collector.get_layer_features()
        student_features = student_collector.get_layer_features()
        
        print(f"教师模型层数: {len(teacher_features)}")
        print(f"学生模型层数: {len(student_features)}")
        
        if teacher_features and student_features:
            layer_idx = list(teacher_features.keys())[0]
            print(f"第{layer_idx}层教师特征形状: {teacher_features[layer_idx].shape}")
            print(f"第{layer_idx}层学生特征形状: {student_features[layer_idx].shape}")
            
            # 清理钩子
            teacher_collector.remove_hooks()
            student_collector.remove_hooks()
            
            print("✅ Layer-wise特征收集器测试通过!")
            return True
        else:
            print("❌ Layer-wise特征收集器测试失败!")
            return False
            
    except Exception as e:
        print(f"❌ Layer-wise特征收集器测试失败: {e}")
        return False

def test_layerwise_distillation_loss():
    """测试layer-wise特征蒸馏损失"""
    print("\n测试layer-wise特征蒸馏损失...")
    
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
        # 创建模型
        teacher_model = MockVisionModel(num_layers=4, feature_dim=512)
        student_model = MockVisionModel(num_layers=4, feature_dim=512)
        
        # 创建收集器
        teacher_collector = LayerwiseFeatureCollector(teacher_model, layers_to_hook=[1, 2, 3], pooling_type='mean')
        student_collector = LayerwiseFeatureCollector(student_model, layers_to_hook=[1, 2, 3], pooling_type='mean')
        
        # 创建输入
        batch_size, seq_len, feature_dim = 4, 49, 512
        input_data = torch.randn(batch_size, seq_len, feature_dim)
        
        # 前向传播
        teacher_output = teacher_model(input_data)
        student_output = student_model(input_data)
        
        # 获取特征
        teacher_feature_list = teacher_collector.get_layer_features_list()
        student_feature_list = student_collector.get_layer_features_list()
        
        # 创建权重
        layer_weights = create_layer_weights(len(teacher_feature_list), 'linear')
        
        # 计算损失
        loss = layerwise_feature_distillation_loss(
            teacher_feature_list, 
            student_feature_list, 
            layer_weights,
            'mse_cosine'
        )
        
        print(f"集成测试损失: {loss.item():.6f}")
        print(f"教师特征层数: {len(teacher_feature_list)}")
        print(f"学生特征层数: {len(student_feature_list)}")
        print(f"层权重: {layer_weights}")
        
        # 清理
        teacher_collector.remove_hooks()
        student_collector.remove_hooks()
        
        print("✅ 集成测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始测试layer-wise特征蒸馏实现...\n")
    
    tests = [
        test_feature_hook,
        test_layerwise_feature_collector,
        test_layerwise_distillation_loss,
        test_create_layer_weights,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 50)
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! Layer-wise特征蒸馏实现正常工作。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查实现。")
        return False

if __name__ == "__main__":
    main()