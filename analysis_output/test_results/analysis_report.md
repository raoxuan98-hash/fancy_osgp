# 贝叶斯优化结果分析报告

## 基本统计信息

- 总试验数: 5
- 成功试验数: 5
- 失败试验数: 0
- 最佳准确率: 0.5355
- 平均准确率: 0.5145 ± 0.0139
- 准确率范围: [0.5001, 0.5355]

## 最佳配置

- bidirectional_kd: False
- layerwise_kd_enabled: True
- layerwise_kd_weight: 0.8037055415050339
- layerwise_kd_loss_type: cosine
- layerwise_kd_weight_strategy: linear
- lrate: 0.0005

## 双向KL损失分析

- 启用双向KL损失的平均准确率: 0.5266 ± 0.0000
- 禁用双向KL损失的平均准确率: 0.5115 ± 0.0140
- 改进: 0.0151 (2.95%)

## Layer-wise蒸馏损失分析

- 启用Layer-wise蒸馏损失的平均准确率: 0.5139 ± 0.0154
- 禁用Layer-wise蒸馏损失的平均准确率: 0.5155 ± 0.0111
- 改进: -0.0016 (-0.30%)

## Layer-wise蒸馏损失类型分析

- mse: 0.5031 ± 0.0030 (n=2)
- cosine: 0.5355 ± 0.0000 (n=1)

## Layer-wise蒸馏权重策略分析

- uniform: 0.5031 ± 0.0030 (n=2)
- linear: 0.5355 ± 0.0000 (n=1)
