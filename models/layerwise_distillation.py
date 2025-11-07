"""Layer-wise特征蒸馏组件，包括特征收集器和钩子机制。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class FeatureHook:
    """单个层的特征钩子，用于捕获中间层输出。"""
    
    def __init__(self):
        self.features: Optional[torch.Tensor] = None
        
    def hook(self, module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
        """
        钩子函数，捕获层的输出。
        
        Args:
            module: 被钩住的模块
            input: 输入张量
            output: 输出张量，形状为 [batch_size, seq_len, feature_dim]
        """
        # 确保输出是张量并分离计算图
        if isinstance(output, torch.Tensor):
            self.features = output.detach()
        elif isinstance(output, tuple):
            # 有些层可能返回多个输出，取第一个
            self.features = output[0].detach() if output[0] is not None else None
        
    def get_pooled_features(self, pooling_type: str = 'mean') -> Optional[torch.Tensor]:
        """
        对序列维度进行池化。
        
        Args:
            pooling_type: 池化类型，'mean', 'cls', 'max'
            
        Returns:
            池化后的特征，形状为 [batch_size, feature_dim]
        """
        if self.features is None:
            return None
            
        if pooling_type == 'mean':
            return self.features.mean(dim=1)  # 平均池化
        elif pooling_type == 'cls':
            return self.features[:, 0, :]  # CLS token (第一个token)
        elif pooling_type == 'max':
            return self.features.max(dim=1)[0]  # 最大池化
        else:
            raise ValueError(f"不支持的池化类型: {pooling_type}")
            
    def clear(self) -> None:
        """清空特征缓存。"""
        self.features = None


class LayerwiseFeatureCollector:
    """多层特征收集器，管理所有transformer层的特征钩子。"""
    
    def __init__(
        self, 
        model: nn.Module, 
        layers_to_hook: Optional[List[int]] = None,
        pooling_type: str = 'mean'
    ):
        """
        初始化特征收集器。
        
        Args:
            model: 要钩住的模型
            layers_to_hook: 要钩住的层索引列表，None表示所有层
            pooling_type: 池化类型
        """
        self.model = model
        self.layers_to_hook = layers_to_hook
        self.pooling_type = pooling_type
        self.hooks: Dict[int, FeatureHook] = {}
        self.hook_handles: Dict[int, torch.utils.hooks.RemovableHandle] = {}
        
        self._register_hooks()
        
    def _register_hooks(self) -> None:
        """注册钩子到指定的transformer层。"""
        # 获取transformer编码器层
        encoder_layers = None
        
        # 尝试不同的路径找到编码器层
        # 1. 检查CLIP_BaseNet结构: model.model.vision_model.clip_vision_model.encoder.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
            vision_model = getattr(self.model.model, 'vision_model', None)
            if vision_model is not None and hasattr(vision_model, 'clip_vision_model'):
                clip_vision_model = getattr(vision_model, 'clip_vision_model', None)
                if clip_vision_model is not None and hasattr(clip_vision_model, 'encoder'):
                    encoder = getattr(clip_vision_model, 'encoder', None)
                    if encoder is not None and hasattr(encoder, 'layers'):
                        encoder_layers = getattr(encoder, 'layers', None)
        
        # 2. 检查直接的CLIP模型结构: model.vision_model.encoder.layers
        if encoder_layers is None and hasattr(self.model, 'vision_model'):
            vision_model = getattr(self.model, 'vision_model', None)
            if vision_model is not None and hasattr(vision_model, 'encoder'):
                encoder = getattr(vision_model, 'encoder', None)
                if encoder is not None and hasattr(encoder, 'layers'):
                    encoder_layers = getattr(encoder, 'layers', None)
        
        # 3. 检查SGPLoRACLIPVisionTransformer结构: model.clip_vision_model.encoder.layers
        if encoder_layers is None and hasattr(self.model, 'clip_vision_model'):
            vision_model = getattr(self.model, 'clip_vision_model', None)
            if vision_model is not None and hasattr(vision_model, 'encoder'):
                encoder = getattr(vision_model, 'encoder', None)
                if encoder is not None and hasattr(encoder, 'layers'):
                    encoder_layers = getattr(encoder, 'layers', None)
        
        # 4. 检查嵌套结构: model.model.clip_vision_model.encoder.layers
        if encoder_layers is None and hasattr(self.model, 'model') and hasattr(self.model.model, 'clip_vision_model'):
            clip_vision_model = getattr(self.model.model, 'clip_vision_model', None)
            if clip_vision_model is not None and hasattr(clip_vision_model, 'encoder'):
                encoder = getattr(clip_vision_model, 'encoder', None)
                if encoder is not None and hasattr(encoder, 'layers'):
                    encoder_layers = getattr(encoder, 'layers', None)
        
        # 5. 检查CLIP_BaseNet结构: model.model.vision_model.encoder.layers (SGPLoRA情况)
        if encoder_layers is None and hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
            vision_model = getattr(self.model.model, 'vision_model', None)
            if vision_model is not None and hasattr(vision_model, 'encoder'):
                encoder = getattr(vision_model, 'encoder', None)
                if encoder is not None and hasattr(encoder, 'layers'):
                    encoder_layers = getattr(encoder, 'layers', None)
        
        # 6. 如果都找不到，尝试直接在model中查找
        if encoder_layers is None and hasattr(self.model, 'encoder'):
            encoder = getattr(self.model, 'encoder', None)
            if encoder is not None and hasattr(encoder, 'layers'):
                encoder_layers = getattr(encoder, 'layers', None)
        
        # 7. 最后尝试，检查是否是SGPLoRACLIPVisionTransformer本身
        if encoder_layers is None and hasattr(self.model, 'clip_vision_model'):
            clip_vision_model = getattr(self.model, 'clip_vision_model', None)
            if clip_vision_model is not None and hasattr(clip_vision_model, 'encoder'):
                encoder = getattr(clip_vision_model, 'encoder', None)
                if encoder is not None and hasattr(encoder, 'layers'):
                    encoder_layers = getattr(encoder, 'layers', None)
        
        if encoder_layers is None:
            # 打印更详细的模型结构信息用于调试
            model_info = {}
            for attr_name in dir(self.model):
                if not attr_name.startswith('_'):
                    try:
                        attr = getattr(self.model, attr_name)
                        if hasattr(attr, '__class__'):
                            model_info[attr_name] = attr.__class__.__name__
                    except:
                        pass
            
            raise ValueError(f"无法找到transformer编码器层，模型结构: {model_info}")
            
        # 确定要钩住的层
        try:
            num_layers = len(encoder_layers)  # type: ignore
        except TypeError:
            raise ValueError("encoder_layers不支持len操作")
            
        if self.layers_to_hook is None:
            layers_to_hook = list(range(num_layers))
        else:
            layers_to_hook = self.layers_to_hook
            
        # 注册钩子
        for layer_idx in layers_to_hook:
            if layer_idx < num_layers:
                hook = FeatureHook()
                try:
                    layer_module = encoder_layers[layer_idx]  # type: ignore
                    handle = layer_module.register_forward_hook(hook.hook)  # type: ignore
                    self.hooks[layer_idx] = hook
                    self.hook_handles[layer_idx] = handle
                except (IndexError, AttributeError, TypeError) as e:
                    print(f"警告：无法注册层 {layer_idx} 的钩子: {e}")
                    continue
                
    def get_layer_features(self) -> Dict[int, torch.Tensor]:
        """
        获取所有层的池化特征。
        
        Returns:
            层索引到池化特征的映射
        """
        layer_features = {}
        for layer_idx, hook in self.hooks.items():
            pooled_features = hook.get_pooled_features(self.pooling_type)
            if pooled_features is not None:
                layer_features[layer_idx] = pooled_features
        return layer_features
        
    def get_layer_features_list(self) -> List[torch.Tensor]:
        """
        获取所有层的池化特征列表，按层索引排序。
        
        Returns:
            池化特征列表，按层索引排序
        """
        layer_features = self.get_layer_features()
        sorted_indices = sorted(layer_features.keys())
        return [layer_features[idx] for idx in sorted_indices]
        
    def clear_features(self) -> None:
        """清空所有层的特征缓存。"""
        for hook in self.hooks.values():
            hook.clear()
            
    def remove_hooks(self) -> None:
        """移除所有钩子。"""
        for handle in self.hook_handles.values():
            handle.remove()
        self.hooks.clear()
        self.hook_handles.clear()
        
    def __del__(self):
        """析构函数，确保钩子被移除。"""
        self.remove_hooks()


def layerwise_feature_distillation_loss(
    teacher_features: List[torch.Tensor],
    student_features: List[torch.Tensor],
    layer_weights: Optional[List[float]] = None,
    loss_type: str = 'mse'
) -> torch.Tensor:
    """
    计算layer-wise特征蒸馏损失。
    
    Args:
        teacher_features: 教师模型各层特征列表
        student_features: 学生模型各层特征列表
        layer_weights: 各层权重，None表示等权重
        loss_type: 损失类型，'mse', 'cosine', 'mse_cosine'
        
    Returns:
        layer-wise蒸馏损失
    """
    if len(teacher_features) != len(student_features):
        raise ValueError(f"教师和学生特征数量不匹配: {len(teacher_features)} vs {len(student_features)}")
        
    if len(teacher_features) == 0:
        return torch.tensor(0.0, device=student_features[0].device if student_features else 'cpu')
        
    if layer_weights is None:
        layer_weights = [1.0] * len(teacher_features)
        
    if len(layer_weights) != len(teacher_features):
        raise ValueError(f"层权重数量与特征数量不匹配: {len(layer_weights)} vs {len(teacher_features)}")
        
    total_loss = 0.0
    device = student_features[0].device if student_features else teacher_features[0].device
    
    for t_feat, s_feat, weight in zip(teacher_features, student_features, layer_weights):
        if loss_type == 'mse':
            layer_loss = F.mse_loss(s_feat, t_feat)
        elif loss_type == 'cosine':
            # 余弦相似度损失 (1 - cosine_similarity)
            layer_loss = 1.0 - F.cosine_similarity(s_feat, t_feat, dim=-1).mean()
        elif loss_type == 'mse_cosine':
            # MSE和余弦相似度的组合
            mse_loss = F.mse_loss(s_feat, t_feat)
            cosine_loss = 1.0 - F.cosine_similarity(s_feat, t_feat, dim=-1).mean()
            layer_loss = mse_loss + cosine_loss
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
            
        total_loss += weight * layer_loss
        
    return torch.tensor(total_loss / len(teacher_features),
                      device=student_features[0].device if student_features else 'cpu')


def create_layer_weights(
    num_layers: int, 
    weight_strategy: str = 'uniform',
    **kwargs
) -> List[float]:
    """
    创建层权重。
    
    Args:
        num_layers: 层数
        weight_strategy: 权重策略，'uniform', 'linear', 'exponential'
        **kwargs: 策略特定的参数
        
    Returns:
        层权重列表
    """
    if weight_strategy == 'uniform':
        return [1.0] * num_layers
    elif weight_strategy == 'linear':
        # 线性增长，深层权重更高
        start_weight = kwargs.get('start_weight', 0.5)
        end_weight = kwargs.get('end_weight', 2.0)
        return [start_weight + (end_weight - start_weight) * i / (num_layers - 1) 
                for i in range(num_layers)]
    elif weight_strategy == 'exponential':
        # 指数增长，深层权重更高
        base = kwargs.get('base', 1.2)
        weights = [base ** i for i in range(num_layers)]
        # 归一化
        total = sum(weights)
        return [w / total * num_layers for w in weights]
    else:
        raise ValueError(f"不支持的权重策略: {weight_strategy}")