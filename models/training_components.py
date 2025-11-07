"""Training components for SubspaceLoRA CLIP learner."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

from models.config import OptimizationConfig, ReferenceBatch, TrainingStepMetrics
from models.subspace_utils import EMASmooth

def feature_distillation_loss(
    teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
    return ((teacher_feat - student_feat) ** 2).mean()

class TrainingManager:
    """Manages the training loop and optimization for SubspaceLoRA CLIP."""
    
    def __init__(self, network, device: str, optim_cfg: OptimizationConfig, 
                 reg_cfg, use_amp: bool, amp_dtype: torch.dtype):
        self.network = network
        self.device = device
        self.optim_cfg = optim_cfg
        self.reg_cfg = reg_cfg
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self._autocast_kwargs = {"enabled": use_amp, "dtype": amp_dtype}
        self.compute_dtype = amp_dtype if use_amp else torch.float32
        self.grad_scaler = GradScaler() if use_amp and amp_dtype == torch.float16 else None
        
        # Training attributes
        self.lrate = optim_cfg.learning_rate
        self.weight_decay = optim_cfg.weight_decay
        self.optimizer_type = optim_cfg.optimizer_type
        self.warmup_steps = optim_cfg.warmup_steps
        self.iterations = optim_cfg.iterations
        self.eta_min = optim_cfg.eta_min
        
        # Regularization
        self.kd_loss_fn = feature_distillation_loss
        self.gamma_kd = reg_cfg.gamma_kd
        self.gamma_prior = reg_cfg.gamma_prior
        self.l2_protection = reg_cfg.l2_enabled
        self.l2_lambda = reg_cfg.l2_lambda
        
        # State
        self.prev_params: Optional[Dict[str, torch.Tensor]] = None
        self._last_valid_batch_size: int = 0
        
    def configure_optimizer(
        self,
        params: List[torch.nn.Parameter],
    ) -> Tuple[optim.Optimizer, _LRScheduler]:
        """Create the optimizer and scheduler pair for the task."""
        
        if self.optimizer_type == "sgd":
            optimizer = optim.SGD(params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_type == "adamw":
            optimizer = optim.AdamW(params, lr=self.lrate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(params, lr=self.lrate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
        scheduler = CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=self.lrate / 2)

        return optimizer, scheduler  # type: ignore
    
    def store_prev_params(self) -> None:
        """Snapshot of trainable weights (used for L2-protection)."""
        
        if not self.l2_protection:
            self.prev_params = None
            return

        self.prev_params = {
            name: param.clone().detach()
            for name, param in self.network.named_parameters()
            if param.requires_grad and "fc" not in name
        }
    
    def l2_protection_loss(self) -> torch.Tensor:
        """L2-penalty that keeps current weights close to the snapshot."""
        
        if not self.l2_protection or self.prev_params is None:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.network.named_parameters():
            if not param.requires_grad or name.startswith("fc"):
                continue
            old = self.prev_params.get(name)
            if old is None:
                continue
            loss = loss + ((param - old.to(self.device)) ** 2).sum()
        return self.l2_lambda * loss
    
    def run_training_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        zeroshot_weights: torch.Tensor,
        reference_batch: ReferenceBatch,
        optimizer: optim.Optimizer,
        current_global_to_local: Optional[Dict[int, int]],
        current_num_classes: int,
        use_feature_kd: bool,
        teacher_network,
        reference_text_embeddings: Optional[torch.Tensor],
        reference_text_labels: Optional[torch.Tensor],
        _n_reference_text: int,
    ) -> TrainingStepMetrics:
        """Forward/backward pass, including optional reference distillation."""
        
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        reference_images = reference_batch.images
        if isinstance(reference_images, torch.Tensor) and reference_images.device != self.device:
            reference_images = reference_images.to(self.device, non_blocking=True)

        kd_metrics = {
            "ref_feature_l2": 0.0,
            "ref_feature_cosine": 0.0,
            "ref_raw_kl": 0.0,
            "teacher_ref_probs_min": 0.0,
            "teacher_ref_probs_max": 0.0,
            "student_ref_probs_min": 0.0,
            "student_ref_probs_max": 0.0,
        }
        invalid_batch = False
        batch_size = 0
        kd_term = torch.zeros((), device=self.device)
        l2_term = torch.zeros((), device=self.device)
        prior_term = torch.zeros((), device=self.device)

        with autocast(**self._autocast_kwargs):
            combined_inputs = inputs if reference_images is None else torch.cat([inputs, reference_images], dim=0)

            combined_img_feats = self.network.encode_image(combined_inputs)
            input_img_feats = combined_img_feats[: inputs.size(0)]
            input_img_feats = input_img_feats / input_img_feats.norm(dim=-1, keepdim=True)

            reference_img_feats = None
            if reference_images is not None:
                reference_img_feats = combined_img_feats[inputs.size(0):]
                reference_img_feats = reference_img_feats / reference_img_feats.norm(dim=-1, keepdim=True)

            logits_per_image, similarity_per_image, local_targets, valid_mask = self._compute_classification_logits(
                input_img_feats,
                targets,
                zeroshot_weights,
                current_global_to_local,
                current_num_classes,
            )

            if not valid_mask.any():
                invalid_batch = True
                loss = combined_img_feats.new_zeros((), dtype=combined_img_feats.dtype, device=combined_img_feats.device)
            else:
                logits_modified = logits_per_image.clone()

                ce_loss = F.cross_entropy(logits_modified, local_targets, label_smoothing=0.1)

                kd_term, kd_metrics = self._compute_reference_regularisation(
                    reference_images,
                    reference_img_feats,
                    reference_batch.labels,
                    teacher_network,
                    reference_text_embeddings,
                    reference_text_labels,
                    _n_reference_text,
                    use_feature_kd,
                )
                l2_term = self.l2_protection_loss()
                prior_term = (
                    self.network.model.vision_model.regularization_loss()
                    if getattr(self.network, "train_mode", "lora") == "lora"
                    else torch.zeros((), device=self.device)
                )
                loss = ce_loss + self.gamma_kd * kd_term + l2_term + prior_term

        if invalid_batch:
            optimizer.zero_grad(set_to_none=True)
            self._last_valid_batch_size = 0
            return TrainingStepMetrics(loss=0.0, correct=0, kd_value=0.0, prior_value=0.0, batch_size=0)

        optimizer.zero_grad(set_to_none=True)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = logits_per_image.argmax(dim=1)
            n_correct = (pred == local_targets).sum().item()

        self._last_valid_batch_size = batch_size
        return TrainingStepMetrics(
            loss=float(loss.detach().cpu().item()),
            correct=n_correct,
            kd_value=float((kd_term + l2_term).detach().cpu().item()),
            prior_value=float(prior_term.detach().cpu().item()) if isinstance(prior_term, torch.Tensor) else float(prior_term),
            batch_size=batch_size,
        )
    
    def _compute_classification_logits(
        self,
        input_img_feats: torch.Tensor,
        targets: torch.Tensor,
        zeroshot_weights: torch.Tensor,
        current_global_to_local: Optional[Dict[int, int]],
        current_num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare logits and remapped targets for the current task."""
        
        logit_scale = self.network.model.logit_scale
        similarity_per_image = input_img_feats @ zeroshot_weights
        logits_per_image = logit_scale.exp() * similarity_per_image

        mapping = current_global_to_local if current_global_to_local is not None else {}
        if not mapping:
            try:
                unique_targets = torch.unique(targets.detach().cpu()).tolist()
                mapping = {int(t): i for i, t in enumerate(sorted(int(x) for x in unique_targets))}
            except Exception:
                pass
        num_classes = current_num_classes or logits_per_image.size(1)
        
        from models.subspace_utils import remap_targets_to_local
        local_targets, valid_mask = remap_targets_to_local(targets, mapping, num_classes)

        if not valid_mask.all():
            logits_per_image = logits_per_image[valid_mask]
            similarity_per_image = similarity_per_image[valid_mask]
            local_targets = local_targets[valid_mask]

        return logits_per_image, similarity_per_image, local_targets, valid_mask
    
    def _compute_reference_regularisation(
        self,
        reference_images: Optional[torch.Tensor],
        student_ref_feats: Optional[torch.Tensor],
        reference_labels: Optional[torch.Tensor],
        teacher_network,
        reference_text_embeddings: Optional[torch.Tensor],
        reference_text_labels: Optional[torch.Tensor],
        _n_reference_text: int,
        use_feature_kd: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        zero = torch.tensor(0.0, device=self.device)
        metrics = {
            "ref_feature_l2": 0.0,
            "ref_feature_cosine": 0.0,
            "ref_raw_kl": 0.0,
            "teacher_ref_probs_min": 0.0,
            "teacher_ref_probs_max": 0.0,
            "student_ref_probs_min": 0.0,
            "student_ref_probs_max": 0.0,
        }

        if (
            not use_feature_kd
            or reference_images is None
            or student_ref_feats is None
            or reference_text_embeddings is None
            or reference_text_labels is None
            or reference_labels is None
        ):
            return zero, metrics

        with torch.no_grad():
            with autocast(**self._autocast_kwargs):
                teacher_feats = teacher_network.encode_image(reference_images.to(self.device, non_blocking=True))
            teacher_feats = teacher_feats / teacher_feats.norm(dim=-1, keepdim=True)

        student_feats = student_ref_feats
        if student_feats is None:
            return zero, metrics
        student_feats = student_feats / student_feats.norm(dim=-1, keepdim=True)

        if isinstance(reference_labels, torch.Tensor):
            ref_labels_tensor = reference_labels.to(dtype=torch.long, device="cpu")
        elif isinstance(reference_labels, (list, tuple)):
            ref_labels_tensor = torch.tensor(reference_labels, dtype=torch.long)
        else:
            ref_labels_tensor = torch.tensor([int(reference_labels)], dtype=torch.long)

        if (
            ref_labels_tensor.numel() == 0
            or ref_labels_tensor.min().item() < 0
            or ref_labels_tensor.max().item() >= _n_reference_text
        ):
            return zero, metrics

        ref_indices = reference_text_labels[ref_labels_tensor]
        reference_text_feats = reference_text_embeddings[ref_indices].to(self.device)

        logit_scale = self.network.model.logit_scale
        ref_feature_l2_dist = F.mse_loss(student_feats, teacher_feats)
        ref_feature_cosine_sim = F.cosine_similarity(student_feats, teacher_feats).mean()

        teacher_logits_ref = logit_scale.exp() * (teacher_feats @ reference_text_feats.T)
        student_logits_ref = logit_scale.exp() * (student_feats @ reference_text_feats.T)

        prob_teacher_ref = F.softmax(teacher_logits_ref, dim=-1)
        prob_student_ref = F.softmax(student_logits_ref, dim=-1)

        temperature = 2.0
        teacher_probs = F.softmax(teacher_logits_ref / temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits_ref / temperature, dim=-1)
        ref_raw_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)

        kd_term = ref_feature_l2_dist + 2.0 * ref_raw_kl

        metrics.update(
            ref_feature_l2=float(ref_feature_l2_dist.item()),
            ref_feature_cosine=float(ref_feature_cosine_sim.item()),
            ref_raw_kl=float(ref_raw_kl.item()),
            teacher_ref_probs_min=float(prob_teacher_ref.min().item()),
            teacher_ref_probs_max=float(prob_teacher_ref.max().item()),
            student_ref_probs_min=float(prob_student_ref.min().item()),
            student_ref_probs_max=float(prob_student_ref.max().item()),
        )
        return kd_term, metrics
