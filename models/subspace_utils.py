"""Utility functions for SubspaceLoRA CLIP learner."""

import logging
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class EMASmooth:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

    def get(self):
        return self.value if self.value is not None else 0.0

class FeatureCovarianceCalculator:
    def __init__(self, model, module_names, device='cuda'):
        self.model = model
        self.module_names = module_names
        self.device = device
        self.covariances = {name: None for name in module_names}
        self.counts = {name: 0 for name in module_names}
        
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """为指定模块注册前向钩子"""
        for name in self.module_names:
            try:
                module = self.model.lora_modules[name]
            except:
                module = self.model.lora_modules[name]
            if module is None:
                raise ValueError(f"模块 {name} 不存在于模型中")
            
            def hook_fn(module, input, output, name=name):
                self._update_covariance(name, input[0])
            
            hook = module.register_forward_hook(hook_fn)
            self.hooks.append(hook)
    
    def _update_covariance(self, name, features):
        """在线更新协方差矩阵"""
        # 特征形状: (batch_size, in_features)
        features = features.detach().to(self.device)
        B, N, D = features.size()
        features = features.view(B*N, D)
        
        # 非中心协方差: X^T X / n
        cov_batch = features.t() @ features  # (in_features, in_features)
        if self.covariances[name] is None:
            self.covariances[name] = cov_batch
        else:
            self.covariances[name] += cov_batch
        
        self.counts[name] += B*N
    
    def compute_final_covariances(self):
        """计算最终的协方差矩阵"""
        final_covs = {}
        for name in self.module_names:
            if self.counts[name] > 0:
                final_covs[name] = self.covariances[name] / self.counts[name]
            else:
                final_covs[name] = None
        return final_covs
    
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()

def compute_covariances(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Compute covariance statistics for the provided model backbone."""
    
    module_names = model.get_module_names()
    cov_calculator = FeatureCovarianceCalculator(model, module_names, device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device)
            model(images)

    covariances = cov_calculator.compute_final_covariances()
    cov_calculator.remove_hooks()
    return covariances


def norm_loss(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
    """MSE between L2-norms of teacher / student feature vectors."""
    
    t_norm = t_feat.norm(p=2, dim=1)
    s_norm = s_feat.norm(p=2, dim=1)
    return F.mse_loss(t_norm, s_norm)


def collate_clip_batch(batch: Iterable) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function aware of the CLIP-style dataset outputs."""
    
    images = []
    labels = []
    for item in batch:
        if isinstance(item, dict):
            images.append(item["images"])
            labels.append(int(item["labels"]))
            continue
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("Unexpected batch element structure for CLIP loader.")
        images.append(item[0])
        labels.append(int(item[1]))
        
    stacked_images = torch.stack(images, dim=0)
    stacked_labels = torch.tensor(labels, dtype=torch.long)
    return stacked_images, stacked_labels


def remap_targets_to_local(
    targets: torch.Tensor,
    mapping: Dict[int, int],
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map global labels to local contiguous indices for the task."""
    
    mapped_list = [mapping.get(int(t), -1) for t in targets.detach().cpu().tolist()]
    mapped = torch.tensor(mapped_list, device=targets.device, dtype=torch.long)
    valid_mask = (mapped >= 0) & (mapped < num_classes)
    return mapped, valid_mask
