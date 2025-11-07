# 双向KL散度知识蒸馏使用指南

## 概述

本项目已成功实现了双向KL散度知识蒸馏功能，结合了mode-covering和mode-seeking的优势。通过一个简单的布尔参数，您可以在原有的单向KL散度和新的双向KL散度之间切换。

## 技术原理

### 单向KL散度 (原有实现)
```
D_KL(p_t || p_s)
```
- 这是mode-covering，鼓励学生模型覆盖教师模型的所有模式
- 当教师模型有多个峰值时，学生会尝试覆盖所有峰值

### 双向KL散度 (新实现)
```
1/2 * D_KL(p_t || p_s) + 1/2 * D_KL(p_s || p_t)
```
- 结合了mode-covering和mode-seeking的优势
- 既保证了模式覆盖，又保证了模式专注
- 通过平均权重平衡两种效应

## 使用方法

### 1. 命令行方式

使用原有的单向KL散度（默认）：
```bash
python main_clip.py [其他参数...]
```

启用双向KL散度：
```bash
python main_clip.py --bidirectional_kd [其他参数...]
```

### 2. 代码方式

```python
args = {
    # 其他参数...
    'bidirectional_kd': True,  # 启用双向KL散度
}

model = SubspaceLoRA_CLIP(args)
```

## 实现细节

### 配置参数
在 `models/config.py` 中的 `RegularizationConfig` 类中添加了：
```python
bidirectional_kd: bool = False  # 控制是否使用双向KL散度进行知识蒸馏
```

### 损失函数
在 `models/training_components.py` 中实现了 `bidirectional_kl_loss` 函数：
```python
def bidirectional_kl_loss(
    teacher_logits: torch.Tensor, 
    student_logits: torch.Tensor, 
    temperature: float = 2.0
) -> torch.Tensor:
    """计算双向KL散度损失，结合mode-covering和mode-seeking"""
    # 实现细节...
```

### 训练组件更新
- `TrainingManager` 类中添加了 `self.bidirectional_kd` 属性
- `TrainingAndReferenceManager` 类中添加了 `self.bidirectional_kd` 属性
- 两个类的 `_compute_reference_regularisation` 方法都支持双向KL散度

## 测试验证

运行测试脚本验证实现：
```bash
python test_bidirectional_kd.py
```

测试内容包括：
1. 双向KL散度损失函数的正确性
2. 单向与双向KL散度的比较
3. 配置集成的验证

## 预期效果

使用双向KL散度知识蒸馏，您可能会观察到：
1. 更好的特征对齐效果
2. 在复杂分布上的更稳定的蒸馏过程
3. 学生模型在保留教师模型知识的同时，更专注于主要模式

## 注意事项

1. 双向KL散度计算量略大于单向KL散度，但差异很小
2. 温度参数（默认2.0）对两种方法都有影响，可根据需要调整
3. 建议在小规模数据上先测试效果，再决定是否在大规模训练中使用

## 文件修改清单

以下是实现双向KL散度所修改的文件：

1. `models/config.py` - 添加配置参数
2. `main_clip.py` - 添加命令行参数
3. `models/subspace_lora_clip_learner.py` - 更新配置构建
4. `models/training_components.py` - 实现双向KL散度损失函数和更新训练组件
5. `models/training_and_reference.py` - 更新训练和参考组件
6. `test_bidirectional_kd.py` - 测试脚本（新增）

## 兼容性

此实现完全向后兼容，默认情况下使用原有的单向KL散度，不会影响现有代码的运行。