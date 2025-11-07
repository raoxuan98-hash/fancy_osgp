"""Reference data components for SubspaceLoRA CLIP learner."""

import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader

from models.config import ReferenceBatch, ReferenceConfig


class ReferenceDataManager:
    """Manages reference data for knowledge distillation."""
    
    def __init__(
        self,
        network,
        teacher_network,
        device: str,
        reference_cfg: ReferenceConfig,
        use_amp: bool,
        amp_dtype: torch.dtype,
        _autocast_kwargs: Dict[str, Any]
    ):
        self.network = network
        self.teacher_network = teacher_network
        self.device = device
        self.reference_cfg = reference_cfg
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self._autocast_kwargs = _autocast_kwargs
        
        # Reference data state
        self.reference_loader: Optional[DataLoader] = None
        self.reference_iter: Optional[Iterator] = None
        self.reference_text_embeddings: Optional[torch.Tensor] = None
        self.reference_text_labels: Optional[torch.Tensor] = None
        self.reference_teacher_embeddings: Optional[torch.Tensor] = None
        self._n_reference_text: int = 0
        
    def initialise_reference_components(self) -> None:
        """Prepare reference dataloaders and cached embeddings if KD is enabled."""
        
        self.reference_loader = None
        self.reference_iter = None
        self.reference_text_embeddings = None
        self.reference_text_labels = None
        self.reference_teacher_embeddings = None
        self._n_reference_text = 0

        if not self.reference_cfg.enabled:
            logging.info("Reference dataset disabled; skipping data loader.")
            return

        reference_dataset = self._build_reference_dataset()
        ref_workers = int(self.reference_cfg.num_workers)
        self.reference_loader = DataLoader(
            reference_dataset,
            batch_size=self.reference_cfg.batch_size,
            shuffle=True,
            num_workers=ref_workers,
            pin_memory=self.reference_cfg.pin_memory
        )
        self.reference_iter = iter(self.reference_loader)

        logging.info("Precomputing reference text embeddings ...")
        with torch.no_grad():
            unique_ref_labels, unique_ref_prompts = reference_dataset.return_labels_and_prompts()
            if unique_ref_prompts and isinstance(unique_ref_prompts[0], (list, tuple)):
                per_image_feats = []
                for captions in unique_ref_prompts:
                    feats = self.teacher_network.encode_text(list(captions))
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    mean_feat = feats.mean(dim=0)
                    mean_feat = mean_feat / mean_feat.norm()
                    per_image_feats.append(mean_feat)
                text_features = torch.stack(per_image_feats, dim=0)
            else:
                text_features = self.teacher_network.encode_text(unique_ref_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.reference_text_embeddings = text_features.to(dtype=torch.float32).cpu().contiguous()
            try:
                self.reference_text_labels = torch.as_tensor(unique_ref_labels, dtype=torch.long)
            except Exception:
                self.reference_text_labels = torch.tensor(list(unique_ref_labels), dtype=torch.long)
            self.reference_text_labels = self.reference_text_labels.to(dtype=torch.long, device="cpu")

            self._n_reference_text = int(self.reference_text_embeddings.size(0))
        logging.info("Precomputed %d reference text embeddings.", self._n_reference_text)

        self._precompute_reference_teacher_embeddings(reference_dataset)
    
    def _precompute_reference_teacher_embeddings(self, reference_dataset) -> None:
        """Cache teacher features for reference data to avoid redundant GPU passes."""
        
        try:
            dataset_size = len(reference_dataset)
        except Exception:
            dataset_size = 0

        if dataset_size == 0:
            self.reference_teacher_embeddings = None
            logging.info("Reference dataset empty; skipping teacher cache precomputation.")
            return

        logging.info("Caching reference teacher embeddings for %d samples...", dataset_size)
        start_time = time.time()
        loader = DataLoader(
            reference_dataset,
            batch_size=self.reference_cfg.batch_size,
            shuffle=False,
            num_workers=int(self.reference_cfg.num_workers),
            pin_memory=self.reference_cfg.pin_memory,
        )

        teacher_features: List[torch.Tensor] = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device, non_blocking=True)
                feats = self.teacher_network.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                teacher_features.append(feats.detach().cpu().to(dtype=torch.float32))

        if teacher_features:
            self.reference_teacher_embeddings = torch.cat(teacher_features, dim=0).contiguous()
            elapsed = time.time() - start_time
            logging.info(
                "Cached %d teacher feature vectors for reference data (%.2fs).",
                self.reference_teacher_embeddings.size(0),
                elapsed,
            )
        else:
            self.reference_teacher_embeddings = None
            logging.warning("Reference teacher feature cache is empty after precomputation.")
    
    def _build_reference_dataset(self):
        """Factory method returning the configured reference dataset."""
        
        aux_type = self.reference_cfg.dataset_type
        aux_path = self.reference_cfg.dataset_path

        if aux_type == "flickr8k":
            if not aux_path:
                raise ValueError("auxiliary_data_path must be provided when using Flickr8k reference data.")
            from utils.flickr8k_ref import Flickr8kRefDataset
            return Flickr8kRefDataset(aux_path, transform=self.network.valid_preprocess)

        # return ImageNet1K(self.network.valid_preprocess)
        raise ValueError(f"Unsupported reference dataset type: {aux_type}")
    
    def next_reference_batch(self) -> ReferenceBatch:
        """Retrieve the next reference batch, rewinding the iterator if necessary."""
        
        if not self.reference_cfg.enabled or self.reference_loader is None:
            return ReferenceBatch(images=None, labels=None)

        assert self.reference_iter is not None, "reference_iter must be initialised when reference data is enabled"
        try:
            batch = next(self.reference_iter)
        except StopIteration:
            self.reference_iter = iter(self.reference_loader)
            batch = next(self.reference_iter)

        if isinstance(batch, dict):
            images = batch.get("images")
            labels = batch.get("labels")
        else:
            images, labels = batch

        if isinstance(images, torch.Tensor):
            images = images.to(self.device, non_blocking=True)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(dtype=torch.long, device="cpu")
        elif isinstance(labels, (list, tuple)):
            labels = torch.as_tensor(labels, dtype=torch.long)
        return ReferenceBatch(images=images, labels=labels)
