"""调试脚本：验证layerwise蒸馏损失的梯度流问题"""

import torch
import torch.nn as nn
from typing import Tuple
from models.layerwise_distillation import LayerwiseFeatureCollector, FeatureHook

def test_gradient_flow():
    """测试特征钩子是否截断了梯度"""
    
    # 创建一个简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 30)
            self.layer3 = nn.Linear(30, 5)
            
        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            x = torch.relu(x)
            x = self.layer3(x)
            return x
    
    # 创建教师和学生模型
    teacher = TestModel()
    student = TestModel()
    
    # 设置特征收集器
    teacher_collector = LayerwiseFeatureCollector(teacher, [1, 2])
    student_collector = LayerwiseFeatureCollector(student, [1, 2])
    
    # 创建测试数据
    x = torch.randn(4, 10, requires_grad=True)
    
    # 前向传播
    teacher_output = teacher(x)
    student_output = student(x)
    
    # 获取特征
    teacher_features = teacher_collector.get_layer_features_list()
    student_features = student_collector.get_layer_features_list()
    
    print("=== 梯度流测试 ===")
    print(f"教师特征数量: {len(teacher_features)}")
    print(f"学生特征数量: {len(student_features)}")
    
    # 检查特征是否有梯度
    for i, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
        print(f"\n层 {i+1}:")
        print(f"  教师特征形状: {t_feat.shape}, requires_grad: {t_feat.requires_grad}")
        print(f"  学生特征形状: {s_feat.shape}, requires_grad: {s_feat.requires_grad}")
        
        # 尝试计算损失并反向传播
        loss = torch.nn.functional.mse_loss(s_feat, t_feat.detach())
        loss.backward()
        
        # 检查梯度
        has_grad = any(param.grad is not None for param in student.parameters())
        print(f"  损失: {loss.item():.6f}")
        print(f"  学生模型是否有梯度: {has_grad}")
        
        # 清零梯度
        student.zero_grad()
    
    # 测试修复后的特征钩子
    print("\n=== 测试修复后的特征钩子 ===")
    
    class FixedFeatureHook:
        """修复后的特征钩子，不截断梯度"""
        def __init__(self):
            self.features = None
            
        def hook(self, module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
            # 不使用detach()，保持梯度流
            if isinstance(output, torch.Tensor):
                self.features = output.clone()  # 使用clone()而不是detach()
            elif isinstance(output, tuple):
                self.features = output[0].clone() if output[0] is not None else None
    
    # 手动创建修复后的钩子
    fixed_hook = FixedFeatureHook()
    handle = student.layer2.register_forward_hook(fixed_hook.hook)
    
    # 重新前向传播
    student.zero_grad()
    student_output = student(x)
    
    # 获取特征并计算损失
    fixed_feature = fixed_hook.features
    if fixed_feature is not None:
        print(f"修复后特征形状: {fixed_feature.shape}, requires_grad: {fixed_feature.requires_grad}")
        
        loss = torch.nn.functional.mse_loss(fixed_feature, teacher_features[1].detach())
        loss.backward()
        
        has_grad = any(param.grad is not None for param in student.parameters())
        print(f"修复后损失: {loss.item():.6f}")
        print(f"修复后学生模型是否有梯度: {has_grad}")
    
    # 移除钩子
    handle.remove()
    teacher_collector.remove_hooks()
    student_collector.remove_hooks()

if __name__ == "__main__":
    test_gradient_flow()