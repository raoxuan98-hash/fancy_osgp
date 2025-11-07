"""测试layerwise蒸馏修复效果的脚本"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from models.layerwise_distillation import LayerwiseFeatureCollector, layerwise_feature_distillation_loss

def create_simple_transformer():
    """创建一个简单的transformer模型用于测试"""
    
    class SimpleTransformerLayer(nn.Module):
        def __init__(self, d_model=512, nhead=8):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.linear1 = nn.Linear(d_model, d_model * 4)
            self.dropout = nn.Dropout(0.1)
            self.linear2 = nn.Linear(d_model * 4, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
        def forward(self, x):
            # Self attention
            attn_output, _ = self.self_attn(x, x, x)
            x = self.norm1(x + self.dropout(attn_output))
            
            # Feed forward
            ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
            x = self.norm2(x + ff_output)
            return x
    
    class SimpleTransformer(nn.Module):
        def __init__(self, num_layers=4, d_model=512):
            super().__init__()
            self.embedding = nn.Linear(10, d_model)  # 输入维度10 -> d_model
            self.layers = nn.ModuleList([SimpleTransformerLayer(d_model) for _ in range(num_layers)])
            self.output = nn.Linear(d_model, 5)  # 输出维度5
            
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x.mean(dim=1))  # 平均池化后输出
    
    return SimpleTransformer()

def test_layerwise_gradient_flow():
    """测试修复后的layerwise蒸馏梯度流"""
    
    print("=== 测试修复后的layerwise蒸馏梯度流 ===")
    
    # 创建教师和学生模型
    teacher = create_simple_transformer()
    student = create_simple_transformer()
    
    # 设置特征收集器
    teacher_collector = LayerwiseFeatureCollector(teacher, layers_to_hook=[1, 2, 3])
    student_collector = LayerwiseFeatureCollector(student, layers_to_hook=[1, 2, 3])
    
    # 创建测试数据
    batch_size = 4
    seq_len = 8
    input_dim = 10
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    
    # 前向传播
    print("\n1. 执行前向传播...")
    teacher_output = teacher(x)
    student_output = student(x)
    
    # 获取特征
    teacher_features = teacher_collector.get_layer_features_list()
    student_features = student_collector.get_layer_features_list()
    
    print(f"   教师特征数量: {len(teacher_features)}")
    print(f"   学生特征数量: {len(student_features)}")
    
    # 检查特征是否有梯度
    for i, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
        print(f"   层 {i+1}: 教师特征形状: {t_feat.shape}, 学生特征形状: {s_feat.shape}")
        print(f"   层 {i+1}: 学生特征requires_grad: {s_feat.requires_grad}")
    
    print("\n2. 计算layerwise蒸馏损失...")
    
    # 计算layerwise蒸馏损失
    layerwise_loss = layerwise_feature_distillation_loss(
        teacher_features,
        student_features,
        layer_weights=[1.0, 1.0, 1.0],
        loss_type='mse'
    )
    
    print(f"   Layerwise蒸馏损失: {layerwise_loss.item():.6f}")
    print(f"   损失requires_grad: {layerwise_loss.requires_grad}")
    
    print("\n3. 执行反向传播...")
    
    # 清零梯度
    student.zero_grad()
    
    # 反向传播
    layerwise_loss.backward()
    
    # 检查梯度
    print("\n4. 检查梯度流...")
    
    # 检查各层是否有梯度
    has_gradients = {}
    for name, param in student.named_parameters():
        if param.grad is not None:
            has_gradients[name] = True
            print(f"   ✅ {name}: 梯度范数 = {param.grad.norm().item():.6f}")
        else:
            has_gradients[name] = False
            print(f"   ❌ {name}: 无梯度")
    
    # 统计有梯度的参数
    total_params = len(has_gradients)
    params_with_grad = sum(has_gradients.values())
    
    print(f"\n   参数统计: {params_with_grad}/{total_params} 个参数有梯度")
    
    if params_with_grad > 0:
        print("   ✅ 修复成功！layerwise蒸馏损失可以反向传播到学生模型")
    else:
        print("   ❌ 修复失败！layerwise蒸馏损失仍无法反向传播")
    
    # 清理钩子
    teacher_collector.remove_hooks()
    student_collector.remove_hooks()
    
    return params_with_grad > 0

def test_with_main_loss():
    """测试layerwise蒸馏损失与主损失的组合"""
    
    print("\n=== 测试layerwise蒸馏损失与主损失的组合 ===")
    
    # 创建教师和学生模型
    teacher = create_simple_transformer()
    student = create_simple_transformer()
    
    # 设置特征收集器
    teacher_collector = LayerwiseFeatureCollector(teacher, layers_to_hook=[1, 2])
    student_collector = LayerwiseFeatureCollector(student, layers_to_hook=[1, 2])
    
    # 创建测试数据
    batch_size = 4
    seq_len = 8
    input_dim = 10
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    target = torch.randn(batch_size, 5)  # 主任务目标
    
    # 前向传播
    teacher_output = teacher(x)
    student_output = student(x)
    
    # 获取特征
    teacher_features = teacher_collector.get_layer_features_list()
    student_features = student_collector.get_layer_features_list()
    
    # 计算主损失
    main_loss = F.mse_loss(student_output, target)
    
    # 计算layerwise蒸馏损失
    layerwise_loss = layerwise_feature_distillation_loss(
        teacher_features,
        student_features,
        layer_weights=[1.0, 1.0],
        loss_type='mse'
    )
    
    # 组合损失
    total_loss = main_loss + 0.5 * layerwise_loss
    
    print(f"   主损失: {main_loss.item():.6f}")
    print(f"   Layerwise蒸馏损失: {layerwise_loss.item():.6f}")
    print(f"   总损失: {total_loss.item():.6f}")
    
    # 反向传播
    student.zero_grad()
    total_loss.backward()
    
    # 检查梯度
    has_gradients = {}
    for name, param in student.named_parameters():
        if param.grad is not None:
            has_gradients[name] = True
    
    total_params = len(has_gradients)
    params_with_grad = sum(has_gradients.values())
    
    print(f"\n   参数统计: {params_with_grad}/{total_params} 个参数有梯度")
    
    if params_with_grad > 0:
        print("   ✅ 组合损失可以正常反向传播")
    else:
        print("   ❌ 组合损失无法反向传播")
    
    # 清理钩子
    teacher_collector.remove_hooks()
    student_collector.remove_hooks()
    
    return params_with_grad > 0

if __name__ == "__main__":
    success1 = test_layerwise_gradient_flow()
    success2 = test_with_main_loss()
    
    print("\n=== 测试总结 ===")
    if success1 and success2:
        print("✅ 所有测试通过！layerwise蒸馏损失的梯度截断问题已修复")
    else:
        print("❌ 测试失败！仍存在梯度截断问题")