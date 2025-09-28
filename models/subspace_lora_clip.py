
from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, _LRScheduler
from torch.utils.data import DataLoader

from models.base import BaseLearner
from utils.inc_net import CLIP_BaseNet

from utils.clip_incremental_manager import ClipIncrementalDataManager
from utils.data1 import basic_templates as DATA1_BASIC_TEMPLATES
# from clip_datasets.imagenet1k import ImageNet1K
from utils.flickr8k_ref import Flickr8kRefDataset
from models.subspace_lora import EMASmooth, feature_distillation_loss
from lora import FeatureCovarianceCalculator


@dataclass
class Timing:
    """Track the elapsed time of key training stages for a task."""
    train: float = 0.0
    drift: float = 0.0
    total: float = 0.0


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for optimisation and scheduling."""
    optimizer_type: str
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    iterations: int
    eta_min: float = 1e-5


@dataclass(frozen=True)
class TrainingLoopConfig:
    batch_size: int
    log_interval: int
    ema_alpha: float


@dataclass(frozen=True)
class RegularizationConfig:
    """Weights and toggles for auxiliary regularisation terms."""
    gamma_kd: float
    gamma_norm: float
    gamma_prior: float
    l2_enabled: bool
    l2_lambda: float


@dataclass(frozen=True)
class ReferenceConfig:
    """Reference dataset configuration used for knowledge distillation."""
    enabled: bool
    dataset_type: str
    dataset_path: Optional[str]
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class TrainingStepMetrics:
    """Metrics captured after completing a single optimisation step."""

    loss: float
    correct: int
    kd_value: float
    prior_value: float
    batch_size: int


@dataclass
class ReferenceBatch:
    """Container for an optional reference batch used during KD."""
    images: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]

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


class SubspaceLoRAClipLearner(BaseLearner):
    """Incremental CLIP learner enhanced with Subspace-LoRA adapters."""

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)
        self.args = args

        (
            self.optim_cfg,
            self.loop_cfg,
            self.reg_cfg,
            self.reference_cfg,
        ) = self._build_configs(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = CLIP_BaseNet(args, train_mode="lora").to(self.device)
        self.prev_network = CLIP_BaseNet(args, train_mode="frozen").to(self.device)

        amp_requested = args.get("amp", True)
        amp_dtype_str = str(args.get("amp_dtype", "fp16")).lower()
        self.use_amp = bool(amp_requested) and self.device.type == "cuda"
        if amp_dtype_str not in {"fp16", "bf16"}:
            logging.warning("Unknown amp_dtype '%s'; defaulting to fp16", amp_dtype_str)
            amp_dtype_str = "fp16"
        if self.use_amp and amp_dtype_str == "bf16":
            bf16_support = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            if not bf16_support:
                logging.warning("bf16 AMP requested but not supported on this device; falling back to fp16")
                amp_dtype_str = "fp16"
        self.amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
        self.grad_scaler = GradScaler() if self.use_amp and self.amp_dtype == torch.float16 else None
        self._autocast_kwargs = {"enabled": self.use_amp, "dtype": self.amp_dtype}
        self.compute_dtype = self.amp_dtype if self.use_amp else torch.float32
        if self.use_amp:
            logging.info("AMP enabled (dtype=%s)", "bf16" if self.amp_dtype == torch.bfloat16 else "fp16")

        if hasattr(torch, "compile"):
            try:
                self.network = torch.compile(self.network)
                logging.info("Compiled SubspaceLoRAClipLearner network via torch.compile")
            except Exception as exc:  # pragma: no cover - fallback path
                logging.warning("torch.compile failed for SubspaceLoRAClipLearner: %s", exc)

        dataset_sequence = args.get("clip_dataset_sequence")
        if not dataset_sequence:
            raise ValueError("clip_dataset_sequence must contain at least one dataset name.")

        self.seed: int = args.get("seed", 1990)
        self.clip_manager = ClipIncrementalDataManager(
            dataset_sequence,
            shuffle=args.get("clip_dataset_shuffle", False),
            seed=args.get("clip_dataset_seed", self.seed),
            log_level=logging.getLogger().getEffectiveLevel(),
        )
        self.dataset_names = self.clip_manager.task_names

        self.use_reference_data: bool = self.reference_cfg.enabled
        self.clip_num_workers: int = self.reference_cfg.num_workers
        self.clip_pin_memory: bool = self.reference_cfg.pin_memory

        self.train_transform = self.network.valid_preprocess
        self.test_transform = self.network.valid_preprocess
        self._fallback_templates = list(DATA1_BASIC_TEMPLATES)

        self.reference_loader: Optional[DataLoader] = None
        self.reference_iter: Optional[Iterator] = None
        self.reference_text_embeddings: Optional[torch.Tensor] = None
        self.reference_text_labels: Optional[torch.Tensor] = None
        self.reference_teacher_embeddings: Optional[torch.Tensor] = None
        self._n_reference_text: int = 0

        self._task_label_mappings: Dict[int, Tuple[Dict[int, int], int]] = {}
        self._current_global_to_local: Optional[Dict[int, int]] = None
        self._current_num_classes: int = 0
        self._last_valid_batch_size: int = 0

        self._timings: Timing = Timing()
        self.time_history: List[Dict[str, float]] = []
        self.history = {
            "iteration": [],
            "train_loss": [],
            "ema_acc": [],
            "lr": [],
            "zeroshot_acc": [],
        }

        self.batch_size: int = self.loop_cfg.batch_size
        self.lrate: float = self.optim_cfg.learning_rate
        self.weight_decay: float = self.optim_cfg.weight_decay
        self.optimizer_type: str = self.optim_cfg.optimizer_type
        self.warmup_steps: int = self.optim_cfg.warmup_steps
        self.iterations: int = self.optim_cfg.iterations
        self.eta_min: float = self.optim_cfg.eta_min

        self.kd_loss_fn = feature_distillation_loss
        self.gamma_kd: float = self.reg_cfg.gamma_kd
        self.gamma_norm: float = self.reg_cfg.gamma_norm
        self.gamma_prior: float = self.reg_cfg.gamma_prior
        self.l2_protection: bool = self.reg_cfg.l2_enabled
        self.l2_lambda: float = self.reg_cfg.l2_lambda

        self.covariances: Optional[Dict[str, torch.Tensor]] = None
        self.prev_params: Optional[Dict[str, torch.Tensor]] = None

        self.weight_interpolation_alpha: float = args.get("weight_interpolation_alpha", 0.5)
        self.model_snapshot: Optional[Dict[str, torch.Tensor]] = None

        self.task_count: int = 0
        self.log_interval: int = self.loop_cfg.log_interval
        self.ema_alpha: float = self.loop_cfg.ema_alpha
        self.monitor_ema = self._build_metric_smoothers(self.ema_alpha)

        self.use_feature_kd: bool = self.gamma_kd > 0.0 and self.use_reference_data
        self.reference_batch_size: int = self.reference_cfg.batch_size

    @staticmethod
    def _build_configs(
        args: Dict[str, Any]
    ) -> Tuple[OptimizationConfig, TrainingLoopConfig, RegularizationConfig, ReferenceConfig]:
        """Assemble strongly-typed configuration objects from the raw argument dictionary."""

        optim_cfg = OptimizationConfig(
            optimizer_type=str(args["optimizer"]),
            learning_rate=float(args["lrate"]),
            weight_decay=float(args["weight_decay"]),
            warmup_steps=int(args["warmup_steps"]),
            iterations=int(args["iterations"]),
            eta_min=float(args.get("lora_eta_min", 1e-7)),
        )

        loop_cfg = TrainingLoopConfig(
            batch_size=int(args["batch_size"]),
            log_interval=int(args.get("log_interval", 10)),
            ema_alpha=float(args.get("ema_alpha", 0.90)),
        )

        reg_cfg = RegularizationConfig(
            gamma_kd=float(args["gamma_kd"]),
            gamma_norm=float(args.get("gamma_norm", 0.0)),
            gamma_prior=float(args["kl_gamma"]),
            l2_enabled=bool(args.get("l2_protection", False)),
            l2_lambda=float(args.get("l2_protection_lambda", 0.0)),
        )

        reference_cfg = ReferenceConfig(
            enabled=bool(args.get("clip_use_reference_data", False)),
            dataset_type=str(args.get("aux_dataset_type", "imagenet")).lower(),
            dataset_path=args.get("auxiliary_data_path"),
            batch_size=int(args.get("reference_batch_size", args["batch_size"])),
            num_workers=int(args.get("clip_num_workers", 0)),
            pin_memory=bool(args.get("clip_pin_memory", True)),
        )
        return optim_cfg, loop_cfg, reg_cfg, reference_cfg

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
        model: CLIP_BaseNet,
    ) -> torch.Tensor:
        """Build a zeroshot classifier from CLIP text embedddings."""

        template_fns = self._resolve_templates(templates)
        zeroshot_weights: List[torch.Tensor] = []
        for classname in classnames:
            texts = [template(classname) for template in template_fns]
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights_tensor = torch.stack(zeroshot_weights, dim=1).to(self.device)
        zeroshot_weights_tensor = zeroshot_weights_tensor / zeroshot_weights_tensor.norm(dim=0, keepdim=True)
        return zeroshot_weights_tensor

    def _resolve_templates(self, templates: Optional[Iterable[Any]]) -> List[Any]:
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

    @staticmethod
    def _collate_clip_batch(batch: Iterable[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function aware of the CLIP-style dataset outputs."""

        images: List[torch.Tensor] = []
        labels: List[int] = []
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

    def _build_dataloader(self, dataset, *, train: bool) -> DataLoader:
        """Instantiate a DataLoader with the learner defaults."""

        return DataLoader(
            dataset,
            batch_size=self.loop_cfg.batch_size,
            shuffle=train,
            num_workers=self.clip_num_workers,
            pin_memory=self.clip_pin_memory,
            drop_last=False,
            collate_fn=self._collate_clip_batch,
        )

    def _get_task_label_mapping(self, task_idx: int) -> Tuple[Dict[int, int], int]:
        """Return cached global->local mapping for the given task along with class count."""

        if task_idx not in self._task_label_mappings:
            task_labels = self.clip_manager.get_task_labels(task_idx, cumulative=False)
            mapping = {int(global_label): idx for idx, global_label in enumerate(task_labels)}
            self._task_label_mappings[task_idx] = (mapping, len(task_labels))
        return self._task_label_mappings[task_idx]

    @staticmethod
    def _remap_targets_to_local(
        targets: torch.Tensor,
        mapping: Dict[int, int],
        num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map global labels to local contiguous indices for the task."""

        mapped_list = [mapping.get(int(t), -1) for t in targets.detach().cpu().tolist()]
        mapped = torch.tensor(mapped_list, device=targets.device, dtype=torch.long)
        valid_mask = (mapped >= 0) & (mapped < num_classes)
        return mapped, valid_mask

    @torch.no_grad()
    def evaluate_zeroshot(self, task_idx: int) -> float:
        """Evaluate zeroshot accuracy on the specified task index."""

        class_names = self.clip_manager.get_task_class_names(task_idx, cumulative=False)
        templates = self._resolve_templates(self.clip_manager.get_dataset_templates(task_idx))
        zeroshot_weights = self.zeroshot_classifier(class_names, templates, self.network)
        label_mapping, num_classes = self._get_task_label_mapping(task_idx)

        test_dataset = self.clip_manager.get_task_dataset(
            task_idx,
            source="test",
            cumulative=False,
            transform=self.test_transform,
        )
        test_loader = self._build_dataloader(test_dataset, train=False)

        prev_mode = self.network.training
        self.network.eval()

        total = 0
        correct = 0

        for images, targets in test_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            local_targets, valid_mask = self._remap_targets_to_local(targets, label_mapping, num_classes)
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

    def save_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""

        param_dict = {
            name: param.detach().cpu()
            for name, param in self.network.named_parameters()
            if param.requires_grad
        }
        payload = {"task": self._cur_task, "model_state_dict": param_dict}
        path = f"{prefix}_after_task_{self._cur_task}.pth"
        torch.save(payload, path)
        logging.info("Checkpoint saved to %s", path)

    def store_model_snapshot(self) -> None:
        """Save a full snapshot of the current model state before training."""

        self.model_snapshot = {k: v.clone().detach() for k, v in self.network.state_dict().items()}
        logging.info("Model snapshot saved before task %d", self._cur_task + 1)

    def after_task(self) -> None:
        """Update class counters after finishing a task."""

        self._known_classes = self._total_classes
        self.task_count += 1

    def incremental_train(
        self,
        train_loader: DataLoader,
        zeroshot_weights: torch.Tensor,
        reference_loader: Optional[DataLoader],
    ) -> None:
        """Entry-point for training on a new task with optional weight interpolation."""

        start_time = time.time()
        # Before starting a new task, fuse current LoRA into backbone
        # and re-initialise LoRA parameters for the next task.
        # try:
        #     vm = getattr(self.network.model, "vision_model", None)
        #     if vm is not None and hasattr(vm, "merge_lora_weights"):
        #         vm.merge_lora_weights()
        #         if hasattr(vm, "reset_parameters_svd"):
        #             vm.reset_parameters_svd()
        # except Exception as exc:
        #     logging.warning("LoRA merge/reinit before new task failed: %s", exc)

        self._cur_task += 1

        self._current_global_to_local, self._current_num_classes = self._get_task_label_mapping(self._cur_task)

        self.store_prev_params()
        self._run_training_loop(train_loader, zeroshot_weights, reference_loader)

        self._timings.total = time.time() - start_time
        logging.info(
            "Task %d finished | total: %.2f s | train: %.2f s | drift: %.2f s",
            self._cur_task,
            self._timings.total,
            self._timings.train,
            self._timings.drift,
        )
        self.update_projection_matrices()

    def _configure_optimizer(
        self,
        params: Iterable[torch.nn.Parameter],
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

        if self.warmup_steps > 0:
            eta_min = self.eta_min

            def lr_lambda(step: int) -> float:
                if step < self.warmup_steps:
                    return step / max(1, self.warmup_steps)
                progress = (step - self.warmup_steps) / max(1, self.iterations - self.warmup_steps)
                lr_ratio = eta_min / self.lrate
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return lr_ratio + cosine_decay * (1.0 - lr_ratio)

            scheduler: _LRScheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=self.eta_min)

        return optimizer, scheduler

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

    def _run_training_loop(
        self,
        train_loader: DataLoader,
        zeroshot_weights: torch.Tensor,
        reference_loader: Optional[DataLoader],
    ) -> None:
        """Iterate over the training dataloader for the configured number of steps."""

        params: List[torch.nn.Parameter] = [param for param in self.network.parameters() if param.requires_grad]
        optimizer, scheduler = self._configure_optimizer(params)

        start = time.time()
        train_iter = iter(train_loader)
        self.reference_iter = iter(reference_loader) if reference_loader is not None else None
        zeroshot_weights = zeroshot_weights.to(device=self.device, dtype=self.compute_dtype)

        for iteration in range(1, self.iterations + 1):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)

            metrics = self.train_one_iteration(
                inputs,
                targets,
                zeroshot_weights,
                reference_loader,
                optimizer,
                iteration,
            )

            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else self.lrate
            self.history["iteration"].append(iteration)
            self.history["train_loss"].append(metrics.loss)
            self.history["lr"].append(current_lr)

            if iteration % self.log_interval == 0:
                self._log_iteration(iteration, current_lr)

            scheduler.step()

        self._timings.train = time.time() - start

    def train_one_iteration(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        zeroshot_weights: torch.Tensor,
        reference_loader: Optional[DataLoader],
        optimizer: optim.Optimizer,
        iteration: int,
    ) -> TrainingStepMetrics:
        """Execute a single optimisation step and update EMA statistics."""

        reference_batch = self._next_reference_batch(reference_loader)
        metrics = self._run_training_step(
            inputs,
            targets,
            zeroshot_weights,
            reference_batch,
            optimizer,
        )

        effective_batch_size = metrics.batch_size if metrics.batch_size > 0 else inputs.size(0)
        accuracy = metrics.correct / effective_batch_size if effective_batch_size else 0.0
        self.monitor_ema["ema_acc"].update(accuracy)
        self.history["ema_acc"].append(self.monitor_ema["ema_acc"].get())
        return metrics

    def _next_reference_batch(self, reference_loader: Optional[DataLoader]) -> ReferenceBatch:
        """Retrieve the next reference batch, rewinding the iterator if necessary."""

        if not self.use_feature_kd or reference_loader is None:
            return ReferenceBatch(images=None, labels=None)

        assert self.reference_iter is not None, "reference_iter must be initialised when reference data is enabled"
        try:
            batch = next(self.reference_iter)
        except StopIteration:
            self.reference_iter = iter(reference_loader)
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

    def _run_training_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        zeroshot_weights: torch.Tensor,
        reference_batch: ReferenceBatch,
        optimizer: optim.Optimizer,
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
            )

            if not valid_mask.any():
                invalid_batch = True
                loss = combined_img_feats.new_zeros((), dtype=combined_img_feats.dtype, device=combined_img_feats.device)
            else:
                logits_modified = logits_per_image.clone()
                batch_size = logits_modified.size(0)
                row_indices = torch.arange(batch_size, device=logits_modified.device)
                logits_modified[row_indices, local_targets] += 1.0

                ce_loss = F.cross_entropy(logits_modified, local_targets, label_smoothing=0.1)
                kd_term, kd_metrics = self._compute_reference_regularisation(
                    reference_img_feats,
                    reference_batch.labels,
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

        self._update_metric_smoothers(similarity_per_image, local_targets, kd_metrics)

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
                if isinstance(self._cur_task, int):
                    self._task_label_mappings[self._cur_task] = (mapping, len(mapping))
                    self._current_global_to_local = mapping
                    self._current_num_classes = len(mapping)
                    logging.warning(
                        "Label mapping was empty; built a temporary mapping with %d classes for task %d.",
                        len(mapping), self._cur_task,
                    )
            except Exception:
                pass
        num_classes = self._current_num_classes or logits_per_image.size(1)
        local_targets, valid_mask = self._remap_targets_to_local(targets, mapping, num_classes)

        if not valid_mask.all():
            if not hasattr(self, "_warned_oob_targets"):
                self._warned_oob_targets = True
                logging.warning(
                    "Dropping %d training samples due to invalid labels (task=%d, num_classes=%d)",
                    valid_mask.numel() - valid_mask.sum().item(),
                    self._cur_task,
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

    def _compute_reference_regularisation(
        self,
        reference_img_feats: Optional[torch.Tensor],
        reference_labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute KD-based regularisation terms when reference data is provided."""

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
            not self.use_feature_kd
            or reference_img_feats is None
            or self.reference_text_embeddings is None
            or self.reference_text_labels is None
            or reference_labels is None
        ):
            return zero, metrics

        with torch.no_grad():
            reference_inputs_prev = reference_img_feats
            if self.prev_network is not None:
                prev_feats = self.prev_network.encode_image(reference_img_feats)
                reference_inputs_prev = prev_feats / prev_feats.norm(dim=-1, keepdim=True)

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

        logit_scale = self.network.model.logit_scale
        ref_feature_l2_dist = F.mse_loss(reference_img_feats, reference_inputs_prev)
        ref_feature_cosine_sim = F.cosine_similarity(reference_img_feats, reference_inputs_prev).mean()

        teacher_logits_ref = logit_scale.exp() * (reference_inputs_prev @ reference_text_feats.T)
        student_logits_ref = logit_scale.exp() * (reference_img_feats @ reference_text_feats.T)

        prob_teacher_ref = F.softmax(teacher_logits_ref, dim=-1)
        prob_student_ref = F.softmax(student_logits_ref, dim=-1)

        temperature = 5.0
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

    def _update_metric_smoothers(
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

    @staticmethod
    def norm_loss(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """MSE between L2-norms of teacher / student feature vectors."""

        t_norm = t_feat.norm(p=2, dim=1)
        s_norm = s_feat.norm(p=2, dim=1)
        return F.mse_loss(t_norm, s_norm)

    def update_projection_matrices(self, initial_weight = 1.0, incremental_weight = 0.9) -> None:
        """Update OSGP projection matrices using the current training data."""

        new_covs = compute_covariances(self.network.model.vision_model, self.train_loader_test_mode)

        if self.covariances is None:
            self.covariances = new_covs
            for key, item in self.covariances.items():
                self.covariances[key] = initial_weight * self.covariances[key]
        else:
            for key in self.covariances:
                self.covariances[key] = incremental_weight * self.covariances[key] + new_covs[key]

        self.network.model.vision_model.update_projection_matrices(self.covariances)

    def _log_iteration(self, iteration: int, learning_rate: float) -> None:
        """Emit a structured log message for important training iterations."""

        ema = self.monitor_ema
        logging.info(
            "Task %d Iter %d/%d | lr=%.6g | acc=%.4f | pos_cos=%.6f | neg_cos=%.6f | ref_L2=%.6f | ref_cos=%.6f | ref_KL=%.6f",
            self._cur_task,
            iteration,
            self.iterations,
            learning_rate,
            ema["ema_acc"].get(),
            ema["input_feature_positive_cosine"].get(),
            ema["input_feature_negative_cosine"].get(),
            ema["ref_feature_l2"].get(),
            ema["ref_feature_cosine"].get(),
            ema["ref_raw_kl"].get(),
        )

    def loop(self) -> Dict[str, List[float | None]]:
        """Run incremental training across the configured dataset sequence."""

        self._initialise_reference_components()

        # If using auxiliary/reference data, update projection matrices
        # before any task starts so auxiliary features are protected.
        if self.use_reference_data:
            if self.reference_loader is not None:
                self.train_loader_test_mode = self.reference_loader
            self.update_projection_matrices(initial_weight=1.0)

        for task_idx, dataset_name in enumerate(self.dataset_names):
            task_meta = self.clip_manager.get_task_metadata(task_idx)
            logging.info(
                "Starting task %d/%d: %s (train=%d, test=%d, classes=%d)",
                task_idx + 1,
                self.clip_manager.nb_tasks,
                dataset_name,
                task_meta["train_size"],
                task_meta["test_size"],
                task_meta["num_classes"],
            )

            class_names = self.clip_manager.get_task_class_names(task_idx, cumulative=False)
            templates = self._resolve_templates(self.clip_manager.get_dataset_templates(task_idx))
            zeroshot_weights = self.zeroshot_classifier(class_names, templates, self.network)

            train_dataset = self.clip_manager.get_task_dataset(
                task_idx,
                source="train",
                cumulative=False,
                transform=self.train_transform,
            )
            train_loader = self._build_dataloader(train_dataset, train=True)

            self.train_loader_test_mode = self._build_dataloader(
                self.clip_manager.get_task_dataset(
                    task_idx,
                    source="train",
                    cumulative=False,
                    transform=self.test_transform,
                ),
                train=False,
            )

            self.incremental_train(train_loader, zeroshot_weights, self.reference_loader)

            logging.info("Evaluating zeroshot performance after task %d", task_idx + 1)
            zeroshot_results = {}
            for eval_idx in range(task_idx + 1):
                accuracy = self.evaluate_zeroshot(eval_idx)
                eval_name = self.dataset_names[eval_idx]
                zeroshot_results[eval_name] = accuracy
                logging.info("  %s: %.2f%%", eval_name, accuracy)

            avg_zeroshot = (
                sum(zeroshot_results.values()) / len(zeroshot_results)
                if zeroshot_results
                else 0.0
            )

            self.history["zeroshot_acc"].append(avg_zeroshot)
            logging.info("Average zeroshot accuracy after task %d: %.2f%%", task_idx + 1, avg_zeroshot)

            self.after_task()

            if self.args.get("save_checkpoints", False):
                self.save_checkpoint(f"checkpoint_task_{task_idx + 1}")
        return self.history

    def _initialise_reference_components(self) -> None:
        """Prepare reference dataloaders and cached embeddings if KD is enabled."""

        self.reference_loader = None
        self.reference_iter = None
        self.reference_text_embeddings = None
        self.reference_text_labels = None
        self.reference_teacher_embeddings = None
        self._n_reference_text = 0

        if not self.use_reference_data:
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
                    feats = self.prev_network.encode_text(list(captions))
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    mean_feat = feats.mean(dim=0)
                    mean_feat = mean_feat / mean_feat.norm()
                    per_image_feats.append(mean_feat)
                text_features = torch.stack(per_image_feats, dim=0)
            else:
                text_features = self.prev_network.encode_text(unique_ref_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.reference_text_embeddings = text_features.to(dtype=torch.float32).cpu().contiguous()
            try:
                self.reference_text_labels = torch.as_tensor(unique_ref_labels, dtype=torch.long)
            except Exception:  # pragma: no cover - fallback path for exotic types
                self.reference_text_labels = torch.tensor(list(unique_ref_labels), dtype=torch.long)
            self.reference_text_labels = self.reference_text_labels.to(dtype=torch.long, device="cpu")

            self._n_reference_text = int(self.reference_text_embeddings.size(0))
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
                feats = self.prev_network.encode_image(images)
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
            return Flickr8kRefDataset(aux_path, transform=self.network.valid_preprocess)

        # return ImageNet1K(self.network.valid_preprocess)

    def get_training_history(self) -> Dict[str, List[float | None]]:
        """Expose training history for external visualisation utilities."""

        return self.history

    def plot_training_progress(self) -> None:
        """Optionally plot training progress, if matplotlib is available."""

        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(self.history["iteration"], self.history["train_loss"])
            plt.xlabel("Iteration")
            plt.ylabel("Training Loss")
            plt.title("Training Loss over Iterations")
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.plot(self.history["iteration"], self.history["ema_acc"], label="EMA Acc", linewidth=2, color="blue")
            plt.xlabel("Iteration")
            plt.ylabel("Training Accuracy")
            plt.title("Training Accuracy (EMA) over Iterations")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.plot(self.history["iteration"], self.history["lr"])
            plt.xlabel("Iteration")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(
                range(1, len(self.history["zeroshot_acc"]) + 1),
                self.history["zeroshot_acc"],
                "o-",
                linewidth=2,
                markersize=6,
            )
            plt.xlabel("Task")
            plt.ylabel("Zeroshot Accuracy (%)")
            plt.title("Zeroshot Accuracy per Task")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig("training_progress.png")
            plt.close()
        except ImportError:
            logging.warning("Matplotlib not available, skipping plotting")


# Backwards compatibility alias
SubspaceLoRA_CLIP = SubspaceLoRAClipLearner
