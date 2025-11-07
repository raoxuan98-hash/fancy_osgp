"""Training loop and reference data components for SubspaceLoRA CLIP learner."""

import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader

from models.config import OptimizationConfig, ReferenceBatch, TrainingStepMetrics
from models.training_components import feature_distillation_loss

if TYPE_CHECKING:
    from utils.inc_net import CLIP_BaseNet
    from utils.clip_incremental_manager import ClipIncrementalDataManager


class TrainingAndReferenceManager:
    """Manages training loop and reference data for SubspaceLoRA CLIP."""
    
    def __init__(
        self,
        network: Any,
        teacher_network: Any,
        device: torch.device,
        optim_cfg: OptimizationConfig,
        reg_cfg: Any,
        reference_cfg: Any,
        use_amp: bool,
        amp_dtype: torch.dtype,
        _autocast_kwargs: Dict[str, Any],
        clip_num_workers: int,
        clip_pin_memory: bool
    ):
        self.network = network
        self.teacher_network = teacher_network
        self.device = device
        self.optim_cfg = optim_cfg
        self.reg_cfg = reg_cfg
        self.reference_cfg = reference_cfg
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self._autocast_kwargs = _autocast_kwargs
        self.clip_num_workers = clip_num_workers
        self.clip_pin_memory = clip_pin_memory
        
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
        self.gamma_norm = reg_cfg.gamma_norm
        self.gamma_prior = reg_cfg.gamma_prior
        self.l2_protection = reg_cfg.l2_enabled
        self.l2_lambda = reg_cfg.l2_lambda
        
        # State
        self.prev_params: Optional[Dict[str, torch.Tensor]] = None
        self._last_valid_batch_size: int = 0
        
        # Reference data state
        self.reference_loader: Optional[DataLoader] = None
        self.reference_iter: Optional[Iterator] = None
        self.reference_text_embeddings: Optional[torch.Tensor] = None
        self.reference_text_labels: Optional[torch.Tensor] = None
        self.reference_teacher_embeddings: Optional[torch.Tensor] = None
        self._n_reference_text: int = 0
        
        # Timing
        self._timings = Any  # Will be set by the main class
        
        # Compute dtype
        self.compute_dtype = amp_dtype if use_amp else torch.float32
        self.grad_scaler = GradScaler() if use_amp and amp_dtype == torch.float16 else None
    
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
            except Exception:  # pragma: no cover - fallback path for exotic types
                self.reference_text_labels = torch.tensor(list(unique_ref_labels), dtype=torch.long)
            self.reference_text_labels = self.reference_text_labels.to(dtype=torch.long, device="cpu")

            self._n_reference_text = int(self.reference_text_embeddings.size(0))  # type: ignore
        logging.info("Precomputed %d reference text embeddings.", self._n_reference_text)

        self._precompute_reference_teacher_embeddings(reference_dataset)
    
    def _precompute_reference_teacher_embeddings(self, reference_dataset) -> None:
        """Cache teacher features for reference data to avoid redundant GPU passes."""

        try:
            dataset_size = len(reference_dataset)  # type: ignore[arg-type]
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

        with autocast('cuda', **self._autocast_kwargs):
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
                    use_feature_kd,
                )
                l2_term = self.l2_protection_loss()
                try:
                    prior_term = (
                        self.network.model.vision_model.regularization_loss()
                        if getattr(self.network, "train_mode", "lora") == "lora"
                        else torch.zeros((), device=self.device)
                    )
                except AttributeError:
                    prior_term = torch.zeros((), device=self.device)
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
            correct=int(n_correct),
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

        # 条件检查
        if (
            not use_feature_kd
            or reference_images is None
            or student_ref_feats is None
            or self.reference_text_embeddings is None
            or self.reference_text_labels is None
            or reference_labels is None
        ):
            return zero, metrics

        # 1) 教师特征：图像 -> 教师网络
        with torch.no_grad():
            with autocast('cuda', **self._autocast_kwargs):
                teacher_feats = self.teacher_network.encode_image(reference_images.to(self.device, non_blocking=True))
            teacher_feats = teacher_feats / teacher_feats.norm(dim=-1, keepdim=True)

        # 2) 学生特征已在外部算好，标准化保障可比性
        student_feats = student_ref_feats
        if student_feats is None:
            return zero, metrics
        # student_feats 在调用处已标准化，这里作为保险再做一次
        student_feats = student_feats / student_feats.norm(dim=-1, keepdim=True)

        # 3) 取与每张 reference 图像对应的文本特征
        if isinstance(reference_labels, torch.Tensor):
            ref_labels_tensor = reference_labels.to(dtype=torch.long, device="cpu")
        elif isinstance(reference_labels, (list, tuple)):
            ref_labels_tensor = torch.tensor(reference_labels, dtype=torch.long)
        else:
            ref_labels_tensor = torch.tensor([int(reference_labels)], dtype=torch.long)

        if (
            ref_labels_tensor.numel() == 0
            or ref_labels_tensor.min().item() < 0
            or ref_labels_tensor.max().item() >= self._n_reference_text
        ):
            return zero, metrics

        ref_indices = self.reference_text_labels[ref_labels_tensor]
        reference_text_feats = self.reference_text_embeddings[ref_indices].to(self.device)

        # 4) 蒸馏项计算
        logit_scale = self.network.model.logit_scale
        # a) 特征对齐（L2 + 余弦）
        ref_feature_l2_dist = F.mse_loss(student_feats, teacher_feats)
        ref_feature_cosine_sim = F.cosine_similarity(student_feats, teacher_feats).mean()

        # b) 通过与文本的对齐分布做 KL 蒸馏（teacher logits / student logits）
        teacher_logits_ref = logit_scale.exp() * (teacher_feats @ reference_text_feats.T)
        student_logits_ref = logit_scale.exp() * (student_feats @ reference_text_feats.T)

        prob_teacher_ref = F.softmax(teacher_logits_ref, dim=-1)
        prob_student_ref = F.softmax(student_logits_ref, dim=-1)

        temperature = 2.0
        teacher_probs = F.softmax(teacher_logits_ref / temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits_ref / temperature, dim=-1)
        ref_raw_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)

        # 总 KD 项（保持你原先的加权）
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
    
    def run_training_loop(
        self,
        train_loader: DataLoader,
        zeroshot_weights: torch.Tensor,
        optimizer: optim.Optimizer,
        scheduler: _LRScheduler,
        log_interval: int,
        cur_task: int,
        current_global_to_local: Optional[Dict[int, int]],
        current_num_classes: int,
        use_feature_kd: bool,
        monitor_ema: Dict[str, Any],
        history: Dict[str, List],
    ) -> None:
        """Iterate over the training dataloader for the configured number of steps."""

        start = time.time()
        train_iter = iter(train_loader)
        self.reference_iter = iter(self.reference_loader) if self.reference_loader is not None else None
        zeroshot_weights = zeroshot_weights.to(device=self.device, dtype=self.compute_dtype)

        for iteration in range(1, self.iterations + 1):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)

            reference_batch = self.next_reference_batch()
            metrics = self.run_training_step(
                inputs,
                targets,
                zeroshot_weights,
                reference_batch,
                optimizer,
                current_global_to_local,
                current_num_classes,
                use_feature_kd,
            )

            effective_batch_size = metrics.batch_size if metrics.batch_size > 0 else inputs.size(0)
            accuracy = metrics.correct / effective_batch_size if effective_batch_size else 0.0
            monitor_ema["ema_acc"].update(accuracy)
            history["ema_acc"].append(monitor_ema["ema_acc"].get())

            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else self.lrate
            history["iteration"].append(iteration)
            history["train_loss"].append(metrics.loss)
            history["lr"].append(current_lr)

            if iteration % log_interval == 0:
                self._log_iteration(iteration, current_lr, monitor_ema, cur_task)

            scheduler.step()

        self._timings.train = time.time() - start  # type: ignore
    
    def _log_iteration(
        self,
        iteration: int,
        learning_rate: float,
        monitor_ema: Dict[str, Any],
        cur_task: int
    ) -> None:
        """Emit a structured log message for important training iterations."""

        logging.info(
            "Task %d Iter %d/%d | lr=%.6g | acc=%.4f | pos_cos=%.6f | neg_cos=%.6f | ref_L2=%.6f | ref_cos=%.6f | ref_KL=%.6f",
            cur_task,
            iteration,
            self.iterations,
            learning_rate,
            monitor_ema["ema_acc"].get(),
            monitor_ema["input_feature_positive_cosine"].get(),
            monitor_ema["input_feature_negative_cosine"].get(),
            monitor_ema["ref_feature_l2"].get(),
            monitor_ema["ref_feature_cosine"].get(),
            monitor_ema["ref_raw_kl"].get(),
        )
