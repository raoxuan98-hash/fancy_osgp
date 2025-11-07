# 双向KL损失和Layer-wise蒸馏损失的贝叶斯优化

本项目实现了基于optuna的贝叶斯优化，用于确定是否需要双向的KL损失以及Layer-wise的蒸馏损失，并优化相关参数。

## 文件说明

### 核心文件

1. **bayesian_optimize_kl_layerwise.py** - 主要的贝叶斯优化脚本
   - 优化双向KL损失和Layer-wise蒸馏损失参数
   - 支持单独或组合优化
   - 自动记录和分析结果

2. **analyze_optimization_results.py** - 结果分析脚本
   - 分析优化结果
   - 生成可视化图表
   - 比较不同参数配置的效果

3. **test_optimization.py** - 测试脚本
   - 测试优化功能的正确性
   - 使用模拟数据验证代码

4. **run_kl_layerwise_optimization.sh** - 运行脚本
   - 自动化执行不同类型的优化
   - 提供完整的优化流程

## 优化参数

### 双向KL损失参数

- `bidirectional_kd`: 是否使用双向KL损失
  - `True`: 使用双向KL散度 (mode-covering + mode-seeking)
  - `False`: 使用单向KL散度 (mode-covering)

### Layer-wise蒸馏损失参数

- `layerwise_kd_enabled`: 是否启用Layer-wise蒸馏损失
  - `True`: 启用多层特征蒸馏，并优化以下相关参数
  - `False`: 禁用多层特征蒸馏，以下参数将被设置为默认值

- `layerwise_kd_weight`: Layer-wise蒸馏损失的权重
  - 可选值: [1.0, 2.0, 5.0]
  - 控制Layer-wise蒸馏损失在总损失中的比重
  - 注意: 仅在`layerwise_kd_enabled=True`时有效

- `layerwise_kd_loss_type`: Layer-wise蒸馏损失的类型
  - `mse`: 均方误差损失
  - `cosine`: 余弦相似度损失
  - `mse_cosine`: MSE和余弦相似度的组合损失
  - 注意: 仅在`layerwise_kd_enabled=True`时有效

- `layerwise_kd_weight_strategy`: Layer-wise蒸馏权重的分配策略
  - `uniform`: 所有层使用相同权重
  - `linear`: 线性增长，深层权重更高
  - `exponential`: 指数增长，深层权重更高
  - 注意: 仅在`layerwise_kd_enabled=True`时有效

### 参数依赖关系

优化脚本会自动处理参数之间的依赖关系：

1. 当`layerwise_kd_enabled=False`时：
   - `layerwise_kd_weight`设置为默认值1.0
   - `layerwise_kd_loss_type`设置为默认值'mse'
   - `layerwise_kd_weight_strategy`设置为默认值'uniform'

2. 当`layerwise_kd_enabled=True`时：
   - 所有Layer-wise相关参数都会被优化

这种设计避免了在禁用Layer-wise蒸馏时浪费计算资源去优化无效参数。

### 基础超参数

- `lrate`: 学习率
- `weight_temp`: 权重温度
- `iterations`: 迭代次数
- `gamma_kd`: 知识蒸馏损失权重
- `sgp_soft_projection`: 是否使用软投影

## 使用方法

### 1. 快速开始

```bash
# 运行完整的优化流程
chmod +x run_kl_layerwise_optimization.sh
./run_kl_layerwise_optimization.sh
```

### 2. 单独运行优化

#### 基础超参数优化
```bash
python bayesian_optimize_kl_layerwise.py \
    --n-trials 20 \
    --study-name "base-optimization" \
    --output "base_optimization_results.json"
```

#### 双向KL损失优化
```bash
python bayesian_optimize_kl_layerwise.py \
    --n-trials 20 \
    --study-name "bidirectional-kl-optimization" \
    --output "bidirectional_kl_results.json" \
    --optimize-bidirectional-kd
```

#### Layer-wise蒸馏损失优化
```bash
python bayesian_optimize_kl_layerwise.py \
    --n-trials 20 \
    --study-name "layerwise-kd-optimization" \
    --output "layerwise_kd_results.json" \
    --optimize-layerwise-kd
```

#### 综合优化
```bash
python bayesian_optimize_kl_layerwise.py \
    --n-trials 50 \
    --study-name "comprehensive-optimization" \
    --output "comprehensive_optimization_results.json" \
    --optimize-bidirectional-kd \
    --optimize-layerwise-kd
```

### 3. 分析结果

```bash
# 分析单个结果文件
python analyze_optimization_results.py comprehensive_optimization_results.json --plot

# 分析多个结果文件
python analyze_optimization_results.py base_optimization_results.json bidirectional_kd_results.json layerwise_kd_results.json --plot
```

### 4. 测试功能

```bash
# 测试目标函数
python test_optimization.py --test objective

# 测试完整脚本
python test_optimization.py --test script

# 测试所有功能
python test_optimization.py --test all
```

## 输出说明

### 优化结果文件

优化结果以JSON格式保存，包含以下信息：

```json
{
  "index": 0,
  "parameters": {
    "lrate": 0.0005,
    "bidirectional_kd": true,
    "layerwise_kd_enabled": true,
    "layerwise_kd_weight": 1.2,
    "layerwise_kd_loss_type": "mse_cosine",
    "layerwise_kd_weight_strategy": "exponential"
  },
  "value": 0.8542,
  "duration_sec": 120.5,
  "failed": false,
  "optimized_params": {...},
  "mean_accuracies_per_seed": [0.8542],
  "avg_layerwise_kd_loss": 0.12
}
```

### 分析报告

分析脚本会生成以下文件：

1. **analysis_report.md** - 文本分析报告
2. **value_distribution.png** - 优化结果分布图
3. **optimization_progress.png** - 优化进度图
4. **bidirectional_kd_comparison.png** - 双向KL损失效果比较
5. **layerwise_kd_comparison.png** - Layer-wise蒸馏损失效果比较
6. **layerwise_loss_types_comparison.png** - 不同损失类型比较
7. **layerwise_weight_strategies_comparison.png** - 不同权重策略比较

## 实现细节

### 双向KL损失

双向KL损失结合了mode-covering和mode-seeking两种特性：

```python
def bidirectional_kl_loss(teacher_logits, student_logits, temperature=2.0):
    # 计算KL(p_t || p_s) - mode covering
    kl_teacher_to_student = F.kl_div(log_student_probs, teacher_probs, reduction="batchmean")
    
    # 计算KL(p_s || p_t) - mode seeking
    kl_student_to_teacher = F.kl_div(log_teacher_probs, student_probs, reduction="batchmean")
    
    # 双向KL散度，平均权重
    bidirectional_kl = 0.5 * kl_teacher_to_student + 0.5 * kl_student_to_teacher
    
    return bidirectional_kl * (temperature * temperature)
```

### Layer-wise蒸馏损失

Layer-wise蒸馏损失通过比较教师和学生模型的多层特征：

```python
def layerwise_feature_distillation_loss(
    teacher_features, student_features, layer_weights, loss_type
):
    total_loss = 0.0
    for t_feat, s_feat, weight in zip(teacher_features, student_features, layer_weights):
        if loss_type == 'mse':
            layer_loss = F.mse_loss(s_feat, t_feat)
        elif loss_type == 'cosine':
            layer_loss = 1.0 - F.cosine_similarity(s_feat, t_feat, dim=-1).mean()
        elif loss_type == 'mse_cosine':
            mse_loss = F.mse_loss(s_feat, t_feat)
            cosine_loss = 1.0 - F.cosine_similarity(s_feat, t_feat, dim=-1).mean()
            layer_loss = mse_loss + cosine_loss
        
        total_loss += weight * layer_loss
    
    return total_loss / len(teacher_features)
```

## 最佳实践

1. **逐步优化**: 先优化基础超参数，再优化特定功能参数
2. **充分试验**: 每种参数组合至少运行10-20次试验
3. **结果验证**: 使用测试脚本验证优化结果的正确性
4. **可视化分析**: 使用分析脚本生成图表，直观理解参数影响
5. **参数重要性**: 关注optuna提供的参数重要性分析

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减少batch_size或使用更少的试验次数
2. **训练失败**: 检查数据路径和模型配置
3. **优化不收敛**: 增加试验次数或调整搜索空间范围
4. **结果分析失败**: 确保结果文件格式正确且包含成功试验

### 调试技巧

1. 使用`--log-level DEBUG`获取详细日志
2. 运行测试脚本验证功能正确性
3. 检查优化结果文件中的`failed`字段
4. 分析错误信息并调整参数范围

## 扩展功能

可以根据需要扩展以下功能：

1. **新增参数**: 在搜索空间中添加新的优化参数
2. **自定义损失**: 实现新的损失函数类型
3. **多目标优化**: 同时优化准确率和训练速度
4. **分布式优化**: 使用多个GPU并行运行试验
5. **自动调参**: 根据中间结果自动调整搜索空间

## 参考文献

1. [Optuna: Hyperparameter Optimization Framework](https://optuna.org/)
2. [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
3. [Layer-wise Distillation for Deep Networks](https://arxiv.org/abs/1812.05871)
4. [Bidirectional KL Divergence for Knowledge Distillation](https://arxiv.org/abs/1905.08836)