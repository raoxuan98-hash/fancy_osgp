"""Data processing and evaluation components for SubspaceLoRA CLIP learner."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from models.subspace_utils import collate_clip_batch, remap_targets_to_local

if TYPE_CHECKING:
    from utils.inc_net import CLIP_BaseNet
    from utils.clip_incremental_manager import ClipIncrementalDataManager


class DataAndEvaluationManager:
    """Manages data processing and evaluation for SubspaceLoRA CLIP."""
    
    def __init__(
        self,
        network: Any,
        device: torch.device,
        clip_num_workers: int,
        clip_pin_memory: bool,
        batch_size: int,
        loop_cfg: Any
    ):
        self.network = network
        self.device = device
        self.clip_num_workers = clip_num_workers
        self.clip_pin_memory = clip_pin_memory
        self.batch_size = batch_size
        self.loop_cfg = loop_cfg
        
        # Task label mappings
        self._task_label_mappings: Dict[int, Tuple[Dict[int, int], int]] = {}
        self._current_global_to_local: Optional[Dict[int, int]] = None
        self._current_num_classes: int = 0
        
        # Fallback templates
        from utils.data1 import basic_templates as DATA1_BASIC_TEMPLATES
        self._fallback_templates = list(DATA1_BASIC_TEMPLATES)
    
    def build_dataloader(self, dataset, *, train: bool) -> DataLoader:
        """Instantiate a DataLoader with the learner defaults."""
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.clip_num_workers,
            pin_memory=self.clip_pin_memory,
            drop_last=False,
            collate_fn=collate_clip_batch,
        )
    
    def get_task_label_mapping(self, task_idx: int, clip_manager: Any) -> Tuple[Dict[int, int], int]:
        """Return cached global->local mapping for the given task along with class count."""
        
        if task_idx not in self._task_label_mappings:
            task_labels = clip_manager.get_task_labels(task_idx, cumulative=False)
            mapping = {int(global_label): idx for idx, global_label in enumerate(task_labels)}
            self._task_label_mappings[task_idx] = (mapping, len(task_labels))
        return self._task_label_mappings[task_idx]
    
    def resolve_templates(self, templates: Optional[Iterable[Any]]) -> List[Any]:
        """Normalise template inputs coming from the dataset manager."""
        
        if templates is None:
            return list(self._fallback_templates)

        if isinstance(templates, (list, tuple)):
            template_list = [template for template in templates if template is not None]
        else:
            template_list = [templates]

        if not template_list:
            return list(self._fallback_templates)
        return template_list
    
    @torch.no_grad()
    def zeroshot_classifier(
        self,
        classnames: Iterable[str],
        templates: Iterable[Any],
    ) -> torch.Tensor:
        """Build a zeroshot classifier from CLIP text embedddings."""
        
        template_fns = self.resolve_templates(templates)
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
    
    def compute_classification_logits(
        self,
        input_img_feats: torch.Tensor,
        targets: torch.Tensor,
        zeroshot_weights: torch.Tensor,
        cur_task: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare logits and remapped targets for the current task."""
        
        logit_scale = self.network.model.logit_scale
        similarity_per_image = input_img_feats @ zeroshot_weights
        logits_per_image = logit_scale.exp() * similarity_per_image

        mapping = self._current_global_to_local if self._current_global_to_local is not None else {}
        if not mapping:
            try:
                unique_targets = torch.unique(targets.detach().cpu()).tolist()
                mapping = {int(t): i for i, t in enumerate(sorted(int(x) for x in unique_targets))}
                if isinstance(cur_task, int):
                    self._task_label_mappings[cur_task] = (mapping, len(mapping))
                    self._current_global_to_local = mapping
                    self._current_num_classes = len(mapping)
                    logging.warning(
                        "Label mapping was empty; built a temporary mapping with %d classes for task %d.",
                        len(mapping), cur_task,
                    )
            except Exception:
                pass
        num_classes = self._current_num_classes or logits_per_image.size(1)
        local_targets, valid_mask = remap_targets_to_local(targets, mapping, num_classes)

        if not valid_mask.all():
            if not hasattr(self, "_warned_oob_targets"):
                self._warned_oob_targets = True
                logging.warning(
                    "Dropping %d training samples due to invalid labels (task=%d, num_classes=%d)",
                    valid_mask.numel() - valid_mask.sum().item(),
                    cur_task,
                    num_classes,
                )
                try:
                    tmin = int(targets.min().item())
                    tmax = int(targets.max().item())
                    logging.warning("Target range in batch: [%d, %d]; mapping_size=%d", tmin, tmax, len(mapping))
                except Exception:
                    pass
            logits_per_image = logits_per_image[valid_mask]
            similarity_per_image = similarity_per_image[valid_mask]
            local_targets = local_targets[valid_mask]

        return logits_per_image, similarity_per_image, local_targets, valid_mask
    
    @torch.no_grad()
    def evaluate_zeroshot(
        self,
        task_idx: int,
        clip_manager: Any,
        test_transform: Any,
    ) -> float:
        """Evaluate zeroshot accuracy on the specified task index."""
        
        class_names = clip_manager.get_task_class_names(task_idx, cumulative=False)
        templates = self.resolve_templates(clip_manager.get_dataset_templates(task_idx))
        zeroshot_weights = self.zeroshot_classifier(class_names, templates)
        label_mapping, num_classes = self.get_task_label_mapping(task_idx, clip_manager)

        test_dataset = clip_manager.get_task_dataset(
            task_idx,
            source="test",
            cumulative=False,
            transform=test_transform,
        )
        test_loader = self.build_dataloader(test_dataset, train=False)

        prev_mode = self.network.training
        self.network.eval()

        total = 0
        correct = 0

        for images, targets in test_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
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
    
    def set_current_task_mapping(self, task_idx: int, clip_manager: Any) -> None:
        """Set the current task label mapping."""
        
        self._current_global_to_local, self._current_num_classes = self.get_task_label_mapping(task_idx, clip_manager)