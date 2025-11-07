"""Utility functions for SubspaceLoRA CLIP learner."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.subspace_utils import compute_covariances, EMASmooth

if TYPE_CHECKING:
    from utils.inc_net import CLIP_BaseNet


def norm_loss(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
    """MSE between L2-norms of teacher / student feature vectors."""
    
    t_norm = t_feat.norm(p=2, dim=1)
    s_norm = s_feat.norm(p=2, dim=1)
    return F.mse_loss(t_norm, s_norm)


def store_prev_params(
    network: torch.nn.Module,
    l2_protection: bool
) -> Optional[Dict[str, torch.Tensor]]:
    """Snapshot of trainable weights (used for L2-protection)."""
    
    if not l2_protection:
        return None

    prev_params = {
        name: param.clone().detach()
        for name, param in network.named_parameters()
        if param.requires_grad and "fc" not in name
    }
    return prev_params


def l2_protection_loss(
    network: torch.nn.Module,
    prev_params: Optional[Dict[str, torch.Tensor]],
    l2_lambda: float,
    device: torch.device
) -> torch.Tensor:
    """L2-penalty that keeps current weights close to the snapshot."""
    
    if not prev_params:
        return torch.tensor(0.0, device=device)

    loss = torch.tensor(0.0, device=device)
    for name, param in network.named_parameters():
        if not param.requires_grad or name.startswith("fc"):
            continue
        old = prev_params.get(name)
        if old is None:
            continue
        loss = loss + ((param - old.to(device)) ** 2).sum()
    return l2_lambda * loss


def update_projection_matrices(
    network: Any,
    train_loader_test_mode: DataLoader,
    covariances: Optional[Dict[str, torch.Tensor]],
    initial_weight: float = 1.0,
    incremental_weight: float = 0.9
) -> Dict[str, torch.Tensor]:
    """Update OSGP projection matrices using the current training data."""
    
    new_covs = compute_covariances(network.model.vision_model, train_loader_test_mode)

    if covariances is None:
        covariances = new_covs
        for key, item in covariances.items():
            covariances[key] = initial_weight * covariances[key]
    else:
        for key in covariances:
            covariances[key] = incremental_weight * covariances[key] + new_covs[key]

    network.model.vision_model.update_projection_matrices(covariances)
    return covariances


def save_checkpoint(
    network: Any,
    cur_task: int,
    prefix: str
) -> None:
    """Save trainable parameters after the current task."""
    
    param_dict = {
        name: param.detach().cpu()
        for name, param in network.named_parameters()
        if param.requires_grad
    }
    payload = {"task": cur_task, "model_state_dict": param_dict}
    path = f"{prefix}_after_task_{cur_task}.pth"
    torch.save(payload, path)
    logging.info("Checkpoint saved to %s", path)


def store_model_snapshot(network: Any, cur_task: int) -> Dict[str, torch.Tensor]:
    """Save a full snapshot of the current model state before training."""
    
    model_snapshot = {k: v.clone().detach() for k, v in network.state_dict().items()}
    logging.info("Model snapshot saved before task %d", cur_task + 1)
    return model_snapshot


def weight_interpolation(
    network: Any,
    model_snapshot: Optional[Dict[str, torch.Tensor]],
    weight_interpolation_alpha: float
) -> None:
    """Apply weight interpolation between current model and snapshot."""
    
    if model_snapshot is None:
        return
        
    current_state = network.state_dict()
    for name in current_state:
        if name in model_snapshot:
            current_state[name] = (
                weight_interpolation_alpha * current_state[name] +
                (1 - weight_interpolation_alpha) * model_snapshot[name]
            )
    network.load_state_dict(current_state)


def build_metric_smoothers(alpha: float) -> Dict[str, EMASmooth]:
    """Create EMA smoothers for the key metrics monitored during training."""
    
    return {
        "input_feature_positive_cosine": EMASmooth(alpha=alpha),
        "input_feature_negative_cosine": EMASmooth(alpha=alpha),
        "ref_feature_l2": EMASmooth(alpha=alpha),
        "ref_feature_cosine": EMASmooth(alpha=alpha),
        "ref_raw_kl": EMASmooth(alpha=alpha),
        "layerwise_kd_loss": EMASmooth(alpha=alpha),  # 添加layerwise蒸馏损失的监控
        "teacher_ref_probs_min": EMASmooth(alpha=alpha),
        "teacher_ref_probs_max": EMASmooth(alpha=alpha),
        "student_ref_probs_min": EMASmooth(alpha=alpha),
        "student_ref_probs_max": EMASmooth(alpha=alpha),
        "ema_acc": EMASmooth(alpha=alpha),
    }