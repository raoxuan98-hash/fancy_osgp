#!/usr/bin/env python3
"""
测试脚本：运行训练并查看调试输出
"""

import subprocess
import sys
import os

def run_training_with_debug():
    """运行训练并捕获调试输出"""
    print("=== 运行训练并查看调试输出 ===")
    print()
    
    # 构建命令
    cmd = [
        sys.executable,
        "main_clip.py",
        "--gamma_kd", "5.0",
        "--clip_use_reference_data",
        "--auxiliary_data_path", "/data1/open_datasets/flickr8k",
        "--aux_dataset_type", "flickr8k",
        "--aux_auto_detect",
        "--aux_num_samples", "1024",
        "--iterations", "10",  # 减少迭代次数以便快速测试
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    print("预期输出:")
    print("1. [DEBUG] next_reference_batch called")
    print("2. [DEBUG] run_training_step called with use_feature_kd: True")
    print("3. [DEBUG] _compute_reference_regularisation called")
    print("4. [DEBUG] 参考损失计算结果:")
    print("5. 如果有条件不满足，会输出相应的警告信息")
    print()
    
    print("正在运行训练...")
    print("-" * 50)
    
    try:
        # 运行命令并捕获输出
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        # 输出标准输出
        if result.stdout:
            print("标准输出:")
            print(result.stdout)
        
        # 输出标准错误
        if result.stderr:
            print("标准错误:")
            print(result.stderr)
            
        # 输出返回码
        print(f"返回码: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("训练超时（5分钟）")
    except Exception as e:
        print(f"运行出错: {e}")

def analyze_debug_output():
    """分析调试输出"""
    print("\n=== 分析调试输出 ===")
    print()
    
    print("如果看到以下情况，说明存在问题：")
    print()
    
    print("1. 如果看到 '参考损失计算跳过' 的警告，说明某个条件不满足：")
    print("   - use_feature_kd = False")
    print("   - reference_images is None")
    print("   - student_ref_feats is None")
    print("   - reference_text_embeddings is None")
    print("   - reference_text_labels is None")
    print("   - reference_labels is None")
    print()
    
    print("2. 如果看到 'Returning empty reference batch'，说明参考数据加载有问题：")
    print("   - reference_cfg.enabled = False")
    print("   - reference_loader is None")
    print()
    
    print("3. 如果参考损失计算结果始终为0，可能的原因：")
    print("   - 参考数据为空")
    print("   - 参考标签超出范围")
    print("   - 特征计算有问题")
    print()
    
    print("4. 正常情况下应该看到：")
    print("   - [DEBUG] next_reference_batch called")
    print("   - [DEBUG] reference_images shape: [batch_size, channels, height, width]")
    print("   - [DEBUG] reference_labels shape: [batch_size]")
    print("   - [DEBUG] _compute_reference_regularisation called")
    print("   - [DEBUG] 参考损失计算结果: (非零值)")
    print()

if __name__ == "__main__":
    run_training_with_debug()
    print("\n" + "="*50 + "\n")
    analyze_debug_output()