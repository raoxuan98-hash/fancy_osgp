"""简化的梯度流测试脚本"""

import torch
import torch.nn as nn
from typing import Tuple

def test_detach_vs_clone():
    """测试detach()和clone()对梯度流的影响"""
    
    print("=== 测试detach() vs clone()对梯度流的影响 ===")
    
    # 创建一个简单的模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # 创建测试数据
    x = torch.randn(4, 10, requires_grad=True)
    target = torch.randn(4, 5)
    
    print("\n1. 使用detach()的情况（当前实现）：")
    
    # 前向传播
    output = model(x)
    
    # 模拟当前FeatureHook的行为 - 使用detach()
    detached_feature = output.detach()
    
    # 计算损失
    loss = nn.MSELoss()(detached_feature, target)
    print(f"   损失值: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    model_grad_exists = any(param.grad is not None for param in model.parameters())
    input_grad_exists = x.grad is not None
    
    print(f"   模型参数是否有梯度: {model_grad_exists}")
    print(f"   输入是否有梯度: {input_grad_exists}")
    
    # 清零梯度
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    
    print("\n2. 使用clone()的情况（修复方案）：")
    
    # 前向传播
    output = model(x)
    
    # 模拟修复后的FeatureHook行为 - 使用clone()
    cloned_feature = output.clone()
    
    # 计算损失
    loss = nn.MSELoss()(cloned_feature, target)
    print(f"   损失值: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    model_grad_exists = any(param.grad is not None for param in model.parameters())
    input_grad_exists = x.grad is not None
    
    print(f"   模型参数是否有梯度: {model_grad_exists}")
    print(f"   输入是否有梯度: {input_grad_exists}")
    
    print("\n3. 测试中间层特征的梯度流：")
    
    # 创建一个可以获取中间层特征的模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 30)
            self.layer3 = nn.Linear(30, 5)
            self.intermediate_feature = None
            
        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            
            # 保存中间特征（模拟钩子）
            self.intermediate_feature = x
            
            x = torch.relu(x)
            x = self.layer3(x)
            return x
    
    model = TestModel()
    x = torch.randn(4, 10, requires_grad=True)
    target = torch.randn(4, 5)
    
    print("\n   使用detach()保存中间特征：")
    
    # 前向传播
    output = model(x)
    
    # 使用detach()保存特征（当前实现）
    if model.intermediate_feature is not None:
        detached_intermediate = model.intermediate_feature.detach()
        
        # 基于中间特征计算损失
        intermediate_loss = nn.MSELoss()(detached_intermediate, torch.zeros_like(detached_intermediate))
        total_loss = nn.MSELoss()(output, target) + intermediate_loss
        
        print(f"   总损失值: {total_loss.item():.6f}")
        
        # 反向传播
        total_loss.backward()
        
        # 检查各层梯度
        layer1_grad = model.layer1.weight.grad is not None
        layer2_grad = model.layer2.weight.grad is not None
        layer3_grad = model.layer3.weight.grad is not None
        
        print(f"   layer1是否有梯度: {layer1_grad}")
        print(f"   layer2是否有梯度: {layer2_grad}")
        print(f"   layer3是否有梯度: {layer3_grad}")
        
        # 清零梯度
        model.zero_grad()
    
    print("\n   使用clone()保存中间特征：")
    
    # 前向传播
    output = model(x)
    
    # 使用clone()保存特征（修复方案）
    if model.intermediate_feature is not None:
        cloned_intermediate = model.intermediate_feature.clone()
        
        # 基于中间特征计算损失
        intermediate_loss = nn.MSELoss()(cloned_intermediate, torch.zeros_like(cloned_intermediate))
        total_loss = nn.MSELoss()(output, target) + intermediate_loss
        
        print(f"   总损失值: {total_loss.item():.6f}")
        
        # 反向传播
        total_loss.backward()
        
        # 检查各层梯度
        layer1_grad = model.layer1.weight.grad is not None
        layer2_grad = model.layer2.weight.grad is not None
        layer3_grad = model.layer3.weight.grad is not None
        
        print(f"   layer1是否有梯度: {layer1_grad}")
        print(f"   layer2是否有梯度: {layer2_grad}")
        print(f"   layer3是否有梯度: {layer3_grad}")

if __name__ == "__main__":
    test_detach_vs_clone()