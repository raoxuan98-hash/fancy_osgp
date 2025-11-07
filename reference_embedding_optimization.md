# 参考文本嵌入计算优化方案

## 问题分析

从日志可以看出效率问题：
- 数据集样本数量限制为 1024
- 但计算了 8091 个参考文本嵌入
- 只缓存了 1024 个教师特征向量

## 根本原因

在 `models/training_and_reference.py` 的 `initialise_reference_components` 方法中：

1. **第210-233行**：计算**全部**数据集的文本嵌入，不管 `num_samples` 限制
2. **第235-278行**：只计算**实际使用**的样本的教师特征

文本嵌入计算没有考虑 `num_samples` 限制，导致不必要的计算浪费。

## 修复方案

### 方案1：修改文本嵌入计算逻辑

在 `initialise_reference_components` 方法中，确保文本嵌入计算也遵循 `num_samples` 限制：

```python
# 在第210行之前添加
logging.info("Precomputing reference text embeddings ...")
with torch.no_grad():
    unique_ref_labels, unique_ref_prompts = reference_dataset.return_labels_and_prompts()
    
    # 如果设置了样本数量限制，只计算前num_samples个样本的文本嵌入
    num_samples = getattr(self.reference_cfg, 'num_samples', None)
    if num_samples is not None and len(unique_ref_labels) > num_samples:
        unique_ref_labels = unique_ref_labels[:num_samples]
        unique_ref_prompts = unique_ref_prompts[:num_samples]
        logging.info(f"文本嵌入计算限制为前 {num_samples} 个样本")
    
    # 原有的文本嵌入计算逻辑...
```

### 方案2：重构为统一的样本限制处理

创建一个辅助方法来统一处理样本数量限制：

```python
def _apply_sample_limit(self, labels: List, prompts: List) -> Tuple[List, List]:
    """应用样本数量限制到标签和提示词列表"""
    num_samples = getattr(self.reference_cfg, 'num_samples', None)
    if num_samples is not None and len(labels) > num_samples:
        logging.info(f"应用样本数量限制: {len(labels)} -> {num_samples}")
        return labels[:num_samples], prompts[:num_samples]
    return labels, prompts
```

### 方案3：在数据集层面处理

修改参考数据集的 `return_labels_and_prompts` 方法，使其返回与实际使用样本数量一致的标签和提示词。

## 推荐实施顺序

1. **立即修复**：实施方案1，快速解决效率问题
2. **长期优化**：实施方案2，提高代码可维护性
3. **架构改进**：考虑方案3，从根本上统一逻辑

## 预期效果

修复后的日志应该显示：
```
数据集样本数量限制为: 1024
文本嵌入计算限制为前 1024 个样本
Precomputed 1024 reference text embeddings.
Caching reference teacher embeddings for 1024 samples...
Cached 1024 teacher feature vectors for reference data (2.22s).
```

## 性能提升

- **计算时间**：减少约87%的文本嵌入计算时间（8091 -> 1024）
- **内存使用**：减少约87%的文本嵌入内存占用
- **一致性**：文本嵌入和教师特征计算保持一致