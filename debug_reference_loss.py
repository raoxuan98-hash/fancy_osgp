#!/usr/bin/env python3
"""
调试脚本：检查参考损失为零的原因
"""

import logging
import sys
import os

# 设置日志级别为DEBUG以查看所有调试信息
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def check_reference_config():
    """检查参考配置"""
    print("=== 参考损失调试分析 ===")
    print()
    
    print("1. 检查可能的根本原因：")
    print("   - use_feature_kd = gamma_kd > 0.0 AND use_reference_data")
    print("   - 如果 gamma_kd = 0.0，则 use_feature_kd = False")
    print("   - 如果 use_reference_data = False，则 use_feature_kd = False")
    print()
    
    print("2. 参考损失计算的条件检查（在 _compute_reference_regularisation 中）：")
    print("   - use_feature_kd 必须为 True")
    print("   - reference_images 不能为 None")
    print("   - student_ref_feats 不能为 None")
    print("   - reference_text_embeddings 不能为 None")
    print("   - reference_text_labels 不能为 None")
    print("   - reference_labels 不能为 None")
    print()
    
    print("3. 可能的解决方案：")
    print()
    print("   方案A：确保 gamma_kd > 0.0")
    print("   在命令行参数中设置：")
    print("   --gamma_kd 5.0  # 或其他大于0的值")
    print()
    
    print("   方案B：确保参考数据集已启用")
    print("   在命令行参数中设置：")
    print("   --clip_use_reference_data True")
    print("   --auxiliary_data_path /path/to/reference/dataset")
    print()
    
    print("   方案C：检查参考数据集路径和类型")
    print("   确保数据集路径存在且可访问：")
    print("   --auxiliary_data_path /data1/open_datasets/flickr8k")
    print("   --aux_dataset_type flickr8k")
    print("   --aux_auto_detect True")
    print()
    
    print("4. 调试步骤：")
    print("   1. 运行训练并查看日志中的调试信息")
    print("   2. 检查 '参考数据配置 - enabled' 日志")
    print("   3. 检查 '知识蒸馏配置 - gamma_kd' 日志")
    print("   4. 检查 '特征知识蒸馏启用状态 - use_feature_kd' 日志")
    print("   5. 检查 '参考损失计算跳过' 相关的调试日志")
    print()

if __name__ == "__main__":
    check_reference_config()