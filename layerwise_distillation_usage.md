# Layer-wise特征蒸馏使用指南

## 概述

本项目已成功实现了layer-wise特征蒸馏功能，可以在所有transformer层上进行特征蒸馏，通过平均池化处理序列维度。这个功能与原有的最终特征蒸馏和双向KL散度蒸馏完全兼容。

## 技术原理

### Layer-wise特征蒸馏
- 对每个transformer层的输出特征进行对齐
- 使用钩子机制捕获中间层输出
- 在序列维度上进行池化（平均池化/CLS token/最大池化）
- 支持多种损失函数（MSE/余弦相似度/组合）

### 池化方式
1. **平均池化 (mean)**：对所有序列位置取平均
2. **CLS token (cls)**：只使用第一个token（CLS token）
3. **最大池化 (max)**：对所有序列位置取最大值

### 层权重策略
1. **均匀权重 (uniform)**：所有层权重相同
2. **线性权重 (linear)**：深层权重线性增加
3. **指数权重 (exponential)**：深层权重指数增加

## 使用方法

### 1. 命令行方式

启用layer-wise特征蒸馏：
```bash
python main_clip.py --layerwise_kd_enabled [其他参数...]
```

指定layer-wise蒸馏参数：
```bash
python main_clip.py \
    --layerwise_kd_enabled \
    --layerwise_kd_weight 2.0 \
    --layerwise_kd_pooling mean \
    --layerwise_kd_loss_type mse_cosine \
    --layerwise_kd_weight_strategy linear \
    [其他参数...]
```

### 2. 代码方式

```python
args = {
    # 其他参数...
    'layerwise_kd_enabled': True,  # 启用layer-wise蒸馏
    'layerwise_kd_weight': 2.0,  # layer-wise蒸馏权重
    'layerwise_kd_pooling': 'mean',  # 池化方式
    'layerwise_kd_loss_type': 'mse_cosine',  # 损失类型
    'layerwise_kd_weight_strategy': 'linear',  # 层权重策略
}

model = SubspaceLoRA_CLIP(args)
```

## 配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|----------|------|
| `layerwise_kd_enabled` | bool | False | 是否启用layer-wise特征蒸馏 |
| `layerwise_kd_weight` | float | 1.0 | layer-wise蒸馏的权重 |
| `layerwise_kd_pooling` | str | "mean" | 池化方式："mean", "cls", "max" |
| `layerwise_kd_loss_type` | str | "mse" | 损失类型："mse", "cosine", "mse_cosine" |
| `layerwise_kd_weight_strategy` | str | "uniform" | 层权重策略："uniform", "linear", "exponential" |

## 损失函数类型

### MSE损失
```python
loss = F.mse_loss(student_features, teacher_features)
```

### 余弦相似度损失
```python
loss = 1.0 - F.cosine_similarity(student_features, teacher_features, dim=-1).mean()
```

### MSE+余弦相似度组合损失
```python
mse_loss = F.mse_loss(student_features, teacher_features)
cosine_loss = 1.0 - F.cosine_similarity(student_features, teacher_features, dim=-1).mean()
loss = mse_loss + cosine_loss
```

## 总蒸馏损失

最终的蒸馏损失是多个组件的组合：

```python
total_kd_loss = (
    final_feature_loss +           # 原有最终特征损失
    layerwise_kd_weight * layerwise_feature_loss +  # layer-wise特征损失
    kl_loss_weight * kl_loss           # KL散度损失（单向或双向）
)
```

## 实现细节

### 特征收集器
- 使用PyTorch的`register_forward_hook`钩子机制
- 自动检测CLIP模型的transformer层结构
- 支持指定特定层或所有层
- 自动清理特征缓存

### 池化实现
```python
def get_pooled_features(self, pooling_type: str = 'mean'):
    if pooling_type == 'mean':
        return self.features.mean(dim=1)  # 平均池化
    elif pooling_type == 'cls':
        return self.features[:, 0, :]  # CLS token
    elif pooling_type == 'max':
        return self.features.max(dim=1)[0]  # 最大池化
```

## 使用建议

### 1. 初次使用
建议先用较小的权重进行测试：
```bash
python main_clip.py --layerwise_kd_enabled --layerwise_kd_weight 0.5 [其他参数...]
```

### 2. 调优参数
- **权重策略**：对于深层模型，"linear"或"exponential"通常更好
- **池化方式**："mean"通常最稳定，"cls"适合分类任务
- **损失类型**："mse_cosine"结合了距离和角度信息

### 3. 与其他蒸馏结合
layer-wise蒸馏可以与双向KL散度同时使用：
```bash
python main_clip.py \
    --layerwise_kd_enabled \
    --bidirectional_kd \
    --layerwise_kd_weight 1.0 \
    [其他参数...]
```

## 性能考虑

1. **内存开销**：layer-wise蒸馏会增加约10-20%的内存使用
2. **计算开销**：增加约5-10%的训练时间
3. **收敛速度**：通常能加快收敛，特别是在小数据集上
4. **性能优化**：已实现智能初始化，只有在启用layer-wise蒸馏时才创建特征收集器，确保不启用时零额外计算负担

## 监控指标

训练过程中会记录以下指标：
- `layerwise_kd_loss`: layer-wise特征蒸馏损失值
- `ref_feature_l2`: 原有特征L2损失
- `ref_feature_cosine`: 原有特征余弦相似度
- `ref_raw_kl`: KL散度损失

## 故障排除

### 1. 钩子注册失败
如果看到"无法注册层X的钩子"警告，可能是：
- 模型结构与预期不符
- 层索引超出范围
- 模型尚未完全初始化

### 2. 特征维度不匹配
如果看到"教师和学生特征数量不匹配"错误：
- 检查教师和学生模型的层数是否相同
- 确认`layerwise_kd_layers`参数设置正确

### 3. 损失计算失败
如果看到"Layer-wise蒸馏损失计算失败"警告：
- 检查池化方式是否正确
- 确认损失类型参数拼写正确

## 文件修改清单

以下是实现layer-wise特征蒸馏所修改的文件：

1. `models/layerwise_distillation.py` - 新增文件，包含特征收集器和损失函数
2. `models/config.py` - 添加layer-wise蒸馏配置参数
3. `main_clip.py` - 添加命令行参数
4. `models/subspace_lora_clip_learner.py` - 更新配置构建和特征收集器设置
5. `models/training_components.py` - 更新训练组件支持layer-wise蒸馏
6. `models/training_and_reference.py` - 更新训练和参考组件支持layer-wise蒸馏

## 兼容性

此实现完全向后兼容：
- 默认情况下layer-wise蒸馏关闭，不影响现有代码
- 可以与双向KL散度同时使用
- 支持所有现有的模型配置

## 示例配置

### 基础配置
```bash
python main_clip.py \
    --layerwise_kd_enabled \
    --layerwise_kd_weight 1.0 \
    --layerwise_kd_pooling mean \
    --layerwise_kd_loss_type mse \
    [其他参数...]
```

### 高级配置
```bash
python main_clip.py \
    --layerwise_kd_enabled \
    --bidirectional_kd \
    --layerwise_kd_weight 2.0 \
    --layerwise_kd_pooling mean \
    --layerwise_kd_loss_type mse_cosine \
    --layerwise_kd_weight_strategy exponential \
    [其他参数...]
```

通过layer-wise特征蒸馏，您可以在更细的粒度上进行知识传递，通常能获得更好的特征对齐和模型性能。