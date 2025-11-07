#!/usr/bin/env python3
"""调试训练步骤问题"""

import logging
import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.config import ReferenceBatch
from models.training_and_reference import TrainingAndReferenceManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(filename)s] => %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_training_step():
    """测试训练步骤"""
    logging.info("开始测试训练步骤...")
    
    # 创建模拟数据
    batch_size = 4
    image_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟输入
    inputs = torch.randn(batch_size, 3, image_size, image_size)
    targets = torch.randint(0, 10, (batch_size,))
    zeroshot_weights = torch.randn(10, 512)
    
    # 模拟参考批次
    reference_batch = ReferenceBatch(
        images=torch.randn(batch_size, 3, image_size, image_size),
        labels=torch.randint(0, 100, (batch_size,))
    )
    
    logging.info(f"输入数据形状: {inputs.shape}")
    logging.info(f"目标数据形状: {targets.shape}")
    logging.info(f"零样本权重形状: {zeroshot_weights.shape}")
    logging.info(f"参考图像形状: {reference_batch.images.shape}")
    logging.info(f"参考标签形状: {reference_batch.labels.shape}")
    
    # 检查参考损失计算的条件
    use_feature_kd = True
    reference_images = reference_batch.images
    reference_labels = reference_batch.labels
    
    logging.info(f"use_feature_kd: {use_feature_kd}")
    logging.info(f"reference_images is None: {reference_images is None}")
    logging.info(f"reference_labels is None: {reference_labels is None}")
    
    # 检查数据类型
    if reference_images is not None:
        logging.info(f"reference_images dtype: {reference_images.dtype}")
    if reference_labels is not None:
        logging.info(f"reference_labels dtype: {reference_labels.dtype}")
    
    logging.info("测试完成")

if __name__ == "__main__":
    test_training_step()