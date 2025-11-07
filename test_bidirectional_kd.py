#!/usr/bin/env python3
"""
测试双向KL散度知识蒸馏实现的简单脚本
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.training_components import bidirectional_kl_loss

def test_bidirectional_kl_loss():
    """测试双向KL散度损失函数"""
    print("测试双向KL散度损失函数...")
    
    # 创建模拟的teacher和student logits
    batch_size = 8
    num_classes = 10
    temperature = 2.0
    
    teacher_logits = torch.randn(batch_size, num_classes)
    student_logits = torch.randn(batch_size, num_classes)
    
    # 计算双向KL散度损失
    bidirectional_loss = bidirectional_kl_loss(teacher_logits, student_logits, temperature)
    
    print(f"双向KL散度损失: {bidirectional_loss.item():.6f}")
    
    # 手动计算验证
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    
    # KL(p_t || p_s) - mode covering
    log_student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_teacher_to_student = F.kl_div(log_student_probs, teacher_probs, reduction="batchmean")
    
    # KL(p_s || p_t) - mode seeking
    log_teacher_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    kl_student_to_teacher = F.kl_div(log_teacher_probs, student_probs, reduction="batchmean")
    
    # 双向KL散度，平均权重
    expected_bidirectional_kl = 0.5 * kl_teacher_to_student + 0.5 * kl_student_to_teacher
    expected_loss = expected_bidirectional_kl * (temperature * temperature)
    
    print(f"手动计算的双向KL散度损失: {expected_loss.item():.6f}")
    
    # 检查两者是否接近
    diff = abs(bidirectional_loss.item() - expected_loss.item())
    print(f"差异: {diff:.8f}")
    
    if diff < 1e-6:
        print("✅ 双向KL散度损失函数测试通过!")
        return True
    else:
        print("❌ 双向KL散度损失函数测试失败!")
        return False

def test_unidirectional_vs_bidirectional():
    """比较单向和双向KL散度的差异"""
    print("\n比较单向和双向KL散度的差异...")
    
    # 创建模拟的teacher和student logits
    batch_size = 4
    num_classes = 5
    temperature = 2.0
    
    teacher_logits = torch.randn(batch_size, num_classes)
    student_logits = torch.randn(batch_size, num_classes)
    
    # 计算单向KL散度 (D_KL(p_t || p_s))
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    unidirectional_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)
    
    # 计算双向KL散度
    bidirectional_loss = bidirectional_kl_loss(teacher_logits, student_logits, temperature)
    
    print(f"单向KL散度损失 (D_KL(p_t || p_s)): {unidirectional_loss.item():.6f}")
    print(f"双向KL散度损失: {bidirectional_loss.item():.6f}")
    print(f"差异: {abs(unidirectional_loss.item() - bidirectional_loss.item()):.6f}")
    
    return True

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
            l2_lambda=0.0
        )
        print(f"默认配置中的bidirectional_kd: {config_default.bidirectional_kd}")
        
        # 测试启用双向KL的配置
        config_bidirectional = RegularizationConfig(
            gamma_kd=1.0,
            gamma_norm=0.1,
            gamma_prior=0.5,
            l2_enabled=False,
            l2_lambda=0.0,
            bidirectional_kd=True
        )
        print(f"启用配置中的bidirectional_kd: {config_bidirectional.bidirectional_kd}")
        
        print("✅ 配置集成测试通过!")
        return True
    except Exception as e:
        print(f"❌ 配置集成测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始测试双向KL散度知识蒸馏实现...\n")
    
    tests = [
        test_bidirectional_kl_loss,
        test_unidirectional_vs_bidirectional,
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
        print("🎉 所有测试通过! 双向KL散度知识蒸馏实现正常工作。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查实现。")
        return False

if __name__ == "__main__":
    main()