"""Evaluation components for SubspaceLoRA CLIP learner."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from models.config import TrainingLoopConfig

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

class EvaluationManager:
    """Manages evaluation and metrics for SubspaceLoRA CLIP."""
    
    def __init__(self, network, device: str, loop_cfg: TrainingLoopConfig):
        self.network = network
        self.device = device
        self.ema_alpha = loop_cfg.ema_alpha
        self.monitor_ema = self._build_metric_smoothers(self.ema_alpha)
        
    @staticmethod
    def _build_metric_smoothers(alpha: float) -> Dict[str, EMASmooth]:
        """Create EMA smoothers for the key metrics monitored during training."""
        
        return {
            "input_feature_positive_cosine": EMASmooth(alpha=alpha),
            "input_feature_negative_cosine": EMASmooth(alpha=alpha),
            "ref_feature_l2": EMASmooth(alpha=alpha),
            "ref_feature_cosine": EMASmooth(alpha=alpha),
            "ref_raw_kl": EMASmooth(alpha=alpha),
            "teacher_ref_probs_min": EMASmooth(alpha=alpha),
            "teacher_ref_probs_max": EMASmooth(alpha=alpha),
            "student_ref_probs_min": EMASmooth(alpha=alpha),
            "student_ref_probs_max": EMASmooth(alpha=alpha),
            "ema_acc": EMASmooth(alpha=alpha),
        }
    
    @torch.no_grad()
    def zeroshot_classifier(
        self,
        classnames: Iterable[str],
        templates: Iterable[Any],
    ) -> torch.Tensor:
        """Build a zeroshot classifier from CLIP text embedddings."""
        
        template_fns = self._resolve_templates(templates)
        zeroshot_weights: List[torch.Tensor] = []
        for classname in classnames:
            texts = [template(classname) for template in template_fns]
            class_embeddings = self.network.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights_tensor = torch.stack(zeroshot_weights, dim=1).to(self.device)
        zeroshot_weights_tensor = zeroshot_weights_tensor / zeroshot_weights_tensor.norm(dim=0, keepdim=True)
        return zeroshot_weights_tensor
    
    def _resolve_templates(self, templates: Optional[Iterable[Any]]) -> List[Any]:
        """Normalise template inputs coming from the dataset manager."""
        
        if templates is None:
            # Fallback templates will be provided by the main class
            return []

        if isinstance(templates, (list, tuple)):
            template_list = [template for template in templates if template is not None]
        else:
            template_list = [templates]

        return template_list
    
    @torch.no_grad()
    def evaluate_zeroshot(
        self, 
        task_idx: int, 
        clip_manager,
        test_transform,
        clip_num_workers: int,
        clip_pin_memory: bool
    ) -> float:
        """Evaluate zeroshot accuracy on the specified task index."""
        
        class_names = clip_manager.get_task_class_names(task_idx, cumulative=False)
        templates = self._resolve_templates(clip_manager.get_dataset_templates(task_idx))
        zeroshot_weights = self.zeroshot_classifier(class_names, templates)
        
        label_mapping, num_classes = self._get_task_label_mapping(task_idx, clip_manager)

        test_dataset = clip_manager.get_task_dataset(
            task_idx,
            source="test",
            cumulative=False,
            transform=test_transform,
        )
        
        from models.subspace_utils import collate_clip_batch
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,  # Use a reasonable batch size for evaluation
            shuffle=False,
            num_workers=clip_num_workers,
            pin_memory=clip_pin_memory,
            drop_last=False,
            collate_fn=collate_clip_batch,
        )

        prev_mode = self.network.training
        self.network.eval()

        total = 0
        correct = 0

        for images, targets in test_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            from models.subspace_utils import remap_targets_to_local
            local_targets, valid_mask = remap_targets_to_local(targets, label_mapping, num_classes)
            
            if not valid_mask.all():
                if not hasattr(self, "_warned_eval_oob"):
                    self._warned_eval_oob = True
                    bad_indices = local_targets[~valid_mask]
                    bad_min = int(bad_indices.min().item()) if bad_indices.numel() else -1
                    bad_max = int(bad_indices.max().item()) if bad_indices.numel() else -1
                    logging.warning(
                        "Dropping %d eval samples due to invalid labels (task=%d, min=%d, max=%d, num_classes=%d)",
                        valid_mask.numel() - valid_mask.sum().item(),
                        task_idx,
                        bad_min,
                        bad_max,
                        num_classes,
                    )
            if not valid_mask.any():
                continue
            targets = local_targets[valid_mask]
            images = images[valid_mask]

            features = self.network.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            logit_scale = self.network.model.logit_scale
            logits_per_image = logit_scale.exp() * features @ zeroshot_weights

            preds = logits_per_image.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        if prev_mode:
            self.network.train()

        return 100.0 * correct / total if total else 0.0
    
    def _get_task_label_mapping(self, task_idx: int, clip_manager) -> Tuple[Dict[int, int], int]:
        """Return cached global->local mapping for the given task along with class count."""
        
        # This would be cached in the main class
        task_labels = clip_manager.get_task_labels(task_idx, cumulative=False)
        mapping = {int(global_label): idx for idx, global_label in enumerate(task_labels)}
        return mapping, len(task_labels)
    
    def update_metric_smoothers(
        self,
        similarity_per_image: torch.Tensor,
        local_targets: torch.Tensor,
        kd_metrics: Dict[str, float],
    ) -> None:
        """Update EMA trackers after a training iteration."""
        
        row_indices = torch.arange(local_targets.size(0), device=similarity_per_image.device)
        positive_cosine = similarity_per_image[row_indices, local_targets].mean().item()

        mask = torch.ones_like(similarity_per_image)
        mask[row_indices, local_targets] = 0
        negative_cosine = (similarity_per_image * mask).sum() / mask.sum()
        negative_cosine = float(negative_cosine.item())

        self.monitor_ema["input_feature_positive_cosine"].update(positive_cosine)
        self.monitor_ema["input_feature_negative_cosine"].update(negative_cosine)

        for key, value in kd_metrics.items():
            if key in self.monitor_ema:
                self.monitor_ema[key].update(value)
