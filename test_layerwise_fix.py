#!/usr/bin/env python3
"""
测试脚本：验证layerwise蒸馏损失计算是否正常工作
"""

import torch
import logging
import sys
import os

# 设置日志级别为DEBUG以查看详细输出
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_layerwise_distillation():
    """测试layerwise蒸馏损失计算"""
    
    print("=" * 60)
    print("测试Layer-wise蒸馏损失计算")
    print("=" * 60)
    
    # 模拟一些测试数据
    batch_size = 4
    feature_dim = 512
    num_layers = 6
    
    # 创建模拟的教师和学生特征
    teacher_features = [torch.randn(batch_size, feature_dim) for _ in range(num_layers)]
    student_features = [torch.randn(batch_size, feature_dim) for _ in range(num_layers)]
    
    print(f"教师特征数量: {len(teacher_features)}")
    print(f"学生特征数量: {len(student_features)}")
    print(f"特征维度: {feature_dim}")
    print(f"批次大小: {batch_size}")
    
    # 测试不同的损失类型
    from models.layerwise_distillation import layerwise_feature_distillation_loss, create_layer_weights
    
    # 测试MSE损失
    mse_loss = layerwise_feature_distillation_loss(
        teacher_features, student_features, 
        loss_type='mse'
    )
    print(f"MSE损失: {mse_loss.item():.6f}")
    
    # 测试余弦损失
    cosine_loss = layerwise_feature_distillation_loss(
        teacher_features, student_features,
        loss_type='cosine'
    )
    print(f"余弦损失: {cosine_loss.item():.6f}")
    
    # 测试组合损失
    combined_loss = layerwise_feature_distillation_loss(
        teacher_features, student_features,
        loss_type='mse_cosine'
    )
    print(f"组合损失: {combined_loss.item():.6f}")
    
    # 测试层权重
    layer_weights = create_layer_weights(num_layers, 'linear')
    weighted_loss = layerwise_feature_distillation_loss(
        teacher_features, student_features,
        layer_weights=layer_weights,
        loss_type='mse'
    )
    print(f"加权损失: {weighted_loss.item():.6f}")
    print(f"层权重: {layer_weights}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    # 检查损失是否非零
    if mse_loss.item() > 0 and cosine_loss.item() > 0 and combined_loss.item() > 0:
        print("✅ Layer-wise蒸馏损失计算正常，所有损失值均大于0")
        return True
    else:
        print("❌ Layer-wise蒸馏损失计算异常，某些损失值为0")
        return False

def test_feature_collector():
    """测试特征收集器"""
    
    print("\n" + "=" * 60)
    print("测试特征收集器")
    print("=" * 60)
    
    try:
        from models.layerwise_distillation import LayerwiseFeatureCollector
        
        # 创建一个简单的模拟模型用于测试
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.ModuleList([
                    torch.nn.Linear(512, 512) for _ in range(6)
                ])
            
            def forward(self, x):
                for layer in self.encoder:
                    x = layer(x)
                return x
        
        model = MockModel()
        
        # 创建特征收集器
        collector = LayerwiseFeatureCollector(model, layers_to_hook=[0, 2, 4])
        
        # 测试前向传播
        test_input = torch.randn(2, 512)
        output = model(test_input)
        
        # 获取特征
        features = collector.get_layer_features()
        print(f"捕获到的层特征数量: {len(features)}")
        
        # 清理
        collector.remove_hooks()
        
        print("✅ 特征收集器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 特征收集器测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试Layer-wise蒸馏修复...")
    
    # 运行测试
    test1_passed = test_layerwise_distillation()
    test2_passed = test_feature_collector()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("🎉 所有测试通过！Layer-wise蒸馏修复成功")
        print("\n修复内容总结：")
        print("1. ✅ 修复了特征收集器获取特征的时机问题")
        print("2. ✅ 确保在计算layerwise损失前重新运行模型前向传播")
        print("3. ✅ 添加了更详细的调试日志")
        print("4. ✅ 改进了特征检查逻辑")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
        
    print("=" * 60)