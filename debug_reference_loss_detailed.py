#!/usr/bin/env python3
"""
详细调试脚本：检查参考损失为零的原因
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

def analyze_reference_loss_issue():
    """分析参考损失为零的问题"""
    print("=== 参考损失详细调试分析 ===")
    print()
    
    print("从日志中观察到的问题：")
    print("1. ref_L2=0.000000 | ref_cos=0.000000 | ref_KL=0.000000 始终为零")
    print("2. 这表明参考损失计算被跳过或者计算结果为零")
    print()
    
    print("可能的根本原因分析：")
    print()
    
    print("A. use_feature_kd 计算问题")
    print("   - 在 subspace_lora_clip_learner.py 第140行：")
    print("   - self.use_feature_kd: bool = self.gamma_kd > 0.0 and self.use_reference_data")
    print("   - 从日志看 gamma_kd=5.0 > 0.0 且 use_reference_data=True")
    print("   - 所以 use_feature_kd 应该为 True")
    print()
    
    print("B. _compute_reference_regularisation 方法中的条件检查")
    print("   - 检查以下条件是否满足：")
    print("     1) use_feature_kd 是否为 True")
    print("     2) reference_images 是否不为 None")
    print("     3) student_ref_feats 是否不为 None")
    print("     4) reference_text_embeddings 是否不为 None")
    print("     5) reference_text_labels 是否不为 None")
    print("     6) reference_labels 是否不为 None")
    print()
    
    print("C. 参考数据集问题")
    print("   - 参考数据集是否正确加载")
    print("   - 参考数据集是否包含有效数据")
    print("   - 参考文本嵌入是否正确计算")
    print("   - 参考教师嵌入是否正确缓存")
    print()
    
    print("D. 数据流问题")
    print("   - next_reference_batch() 是否返回有效数据")
    print("   - 参考批次数据格式是否正确")
    print("   - 参考标签是否在有效范围内")
    print()
    
    print("建议的调试步骤：")
    print()
    print("1. 在 _compute_reference_regularisation 方法开始处添加详细日志")
    print("2. 在每个条件检查处添加日志，输出具体哪个条件不满足")
    print("3. 在参考损失计算完成后添加日志，输出计算结果")
    print("4. 在 next_reference_batch 方法中添加日志，确认返回的数据")
    print()
    
    print("临时修复方案：")
    print()
    print("1. 确保日志级别设置为 DEBUG，可以看到所有调试信息")
    print("2. 在训练循环中添加更详细的日志输出")
    print("3. 检查参考数据集的加载和处理过程")
    print()

def create_debug_patch():
    """创建调试补丁"""
    print("=== 创建调试补丁 ===")
    print()
    
    print("建议在 training_and_reference.py 中添加以下调试代码：")
    print()
    
    print("1. 在 _compute_reference_regularisation 方法开始处添加：")
    print("```python")
    print("def _compute_reference_regularisation(self, ...):")
    print('    logging.info(f"[DEBUG] _compute_reference_regularisation called")')
    print('    logging.info(f"[DEBUG] use_feature_kd: {use_feature_kd}")')
    print('    logging.info(f"[DEBUG] reference_images is None: {reference_images is None}")')
    print('    logging.info(f"[DEBUG] student_ref_feats is None: {student_ref_feats is None}")')
    print('    logging.info(f"[DEBUG] reference_text_embeddings is None: {self.reference_text_embeddings is None}")')
    print('    logging.info(f"[DEBUG] reference_text_labels is None: {self.reference_text_labels is None}")')
    print('    logging.info(f"[DEBUG] reference_labels is None: {reference_labels is None}")')
    print("    # 原有代码...")
    print("```")
    print()
    
    print("2. 在 next_reference_batch 方法中添加：")
    print("```python")
    print("def next_reference_batch(self) -> ReferenceBatch:")
    print('    logging.info(f"[DEBUG] next_reference_batch called")')
    print('    logging.info(f"[DEBUG] reference_loader is None: {self.reference_loader is None}")')
    print('    logging.info(f"[DEBUG] reference_cfg.enabled: {self.reference_cfg.enabled}")')
    print("    # 原有代码...")
    print('    if isinstance(images, torch.Tensor):')
    print('        logging.info(f"[DEBUG] reference_images shape: {images.shape}")')
    print('    if isinstance(labels, torch.Tensor):')
    print('        logging.info(f"[DEBUG] reference_labels shape: {labels.shape}")')
    print("    # 原有代码...")
    print("```")
    print()
    
    print("3. 在 run_training_step 方法中添加：")
    print("```python")
    print("def run_training_step(self, ...):")
    print('    logging.info(f"[DEBUG] run_training_step called with use_feature_kd: {use_feature_kd}")')
    print('    logging.info(f"[DEBUG] reference_batch.images is None: {reference_batch.images is None}")')
    print("    # 原有代码...")
    print("```")
    print()

if __name__ == "__main__":
    analyze_reference_loss_issue()
    print("\n" + "="*50 + "\n")
    create_debug_patch()