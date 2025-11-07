#!/usr/bin/env python3
"""调试参考标签问题"""

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

def test_reference_labels():
    """测试参考标签索引问题"""
    logging.info("开始测试参考标签索引...")
    
    # 创建模拟数据
    batch_size = 4
    n_reference_text = 10
    
    # 模拟参考标签
    reference_text_labels = torch.arange(n_reference_text)
    reference_labels = torch.randint(0, n_reference_text, (batch_size,))
    
    logging.info(f"reference_text_labels: {reference_text_labels}")
    logging.info(f"reference_labels: {reference_labels}")
    logging.info(f"reference_labels.min(): {reference_labels.min()}")
    logging.info(f"reference_labels.max(): {reference_labels.max()}")
    logging.info(f"n_reference_text: {n_reference_text}")
    
    # 检查索引是否有效
    if (
        reference_labels.numel() == 0
        or reference_labels.min().item() < 0
        or reference_labels.max().item() >= n_reference_text
    ):
        logging.error("无效的参考标签索引!")
        return False
    
    # 尝试索引
    try:
        ref_indices = reference_text_labels[reference_labels]
        logging.info(f"ref_indices: {ref_indices}")
        logging.info("索引操作成功")
        return True
    except Exception as e:
        logging.error(f"索引操作失败: {e}")
        return False

if __name__ == "__main__":
    test_reference_labels()