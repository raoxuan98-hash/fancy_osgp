"""Core SubspaceLoRA CLIP learner class with refactored components."""

import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import BaseLearner
from models.config import (
    OptimizationConfig, TrainingLoopConfig, RegularizationConfig,
    ReferenceConfig, Timing, TrainingStepMetrics, ReferenceBatch
)
from models.clip_utils import (
    norm_loss, store_prev_params, l2_protection_loss, update_projection_matrices,
    save_checkpoint, store_model_snapshot, weight_interpolation, build_metric_smoothers
)
from models.data_and_evaluation import DataAndEvaluationManager
from models.training_and_reference import TrainingAndReferenceManager
from utils.inc_net import CLIP_BaseNet
from utils.clip_incremental_manager import ClipIncrementalDataManager


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
        self.teacher_network = CLIP_BaseNet(args, train_mode="frozen").to(self.device)

        # AMP configuration
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
        self._autocast_kwargs = {"enabled": self.use_amp, "dtype": self.amp_dtype}
        self.compute_dtype = self.amp_dtype if self.use_amp else torch.float32
        if self.use_amp:
            logging.info("AMP enabled (dtype=%s)", "bf16" if self.amp_dtype == torch.bfloat16 else "fp16")

        # Model compilation
        if hasattr(torch, "compile"):
            try:
                self.network = torch.compile(self.network)
                logging.info("Compiled SubspaceLoRAClipLearner network via torch.compile")
            except Exception as exc:  # pragma: no cover - fallback path
                logging.warning("torch.compile failed for SubspaceLoRAClipLearner: %s", exc)

        # Dataset configuration
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

        # Reference data configuration
        self.use_reference_data: bool = self.reference_cfg.enabled
        self.clip_num_workers: int = self.reference_cfg.num_workers
        self.clip_pin_memory: bool = self.reference_cfg.pin_memory

        # Transforms
        self.train_transform = getattr(self.network, 'valid_preprocess', None)
        self.test_transform = getattr(self.network, 'valid_preprocess', None)

        # Initialize component managers
        self.data_eval_manager = DataAndEvaluationManager(
            self.network, self.device, self.clip_num_workers, self.clip_pin_memory,
            self.loop_cfg.batch_size, self.loop_cfg
        )
        
        self.training_ref_manager = TrainingAndReferenceManager(
            self.network, self.teacher_network, self.device, self.optim_cfg, self.reg_cfg,
            self.reference_cfg, self.use_amp, self.amp_dtype, self._autocast_kwargs,
            self.clip_num_workers, self.clip_pin_memory
        )
        
        # 设置layer-wise特征收集器
        self.training_ref_manager.setup_layerwise_collectors(self.teacher_network, self.network)

        # Training configuration
        self.batch_size: int = self.loop_cfg.batch_size
        self.lrate: float = self.optim_cfg.learning_rate
        self.weight_decay: float = self.optim_cfg.weight_decay
        self.optimizer_type: str = self.optim_cfg.optimizer_type
        self.warmup_steps: int = self.optim_cfg.warmup_steps
        self.iterations: int = self.optim_cfg.iterations
        self.eta_min: float = self.optim_cfg.eta_min

        # Regularization
        self.kd_loss_fn = self.training_ref_manager.kd_loss_fn
        self.gamma_kd: float = self.reg_cfg.gamma_kd
        self.gamma_norm: float = self.reg_cfg.gamma_norm
        self.gamma_prior: float = self.reg_cfg.gamma_prior
        self.l2_protection: bool = self.reg_cfg.l2_enabled
        self.l2_lambda: float = self.reg_cfg.l2_lambda

        # State variables
        self.covariances: Optional[Dict[str, torch.Tensor]] = None
        self.prev_params: Optional[Dict[str, torch.Tensor]] = None
        self.weight_interpolation_alpha: float = args.get("weight_interpolation_alpha", 0.5)
        self.model_snapshot: Optional[Dict[str, torch.Tensor]] = None
        self.task_count: int = 0
        self.log_interval: int = self.loop_cfg.log_interval
        self.ema_alpha: float = self.loop_cfg.ema_alpha
        self.monitor_ema = build_metric_smoothers(self.ema_alpha)
        self.use_feature_kd: bool = self.gamma_kd > 0.0 and self.use_reference_data
        self.reference_batch_size: int = self.reference_cfg.batch_size
        
        # 添加调试信息
        logging.info(f"参考数据配置 - enabled: {self.use_reference_data}")
        logging.info(f"知识蒸馏配置 - gamma_kd: {self.gamma_kd}")
        logging.info(f"特征知识蒸馏启用状态 - use_feature_kd: {self.use_feature_kd}")

        # Timing and history
        self._timings: Timing = Timing()
        self.time_history: List[Dict[str, float]] = []
        self.history = {
            "iteration": [],
            "train_loss": [],
            "ema_acc": [],
            "lr": [],
            "zeroshot_acc": [],
        }

        # Set timings reference for training manager
        self.training_ref_manager._timings = self._timings

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
            bidirectional_kd=bool(args.get("bidirectional_kd", False)),
            layerwise_kd_enabled=bool(args.get("layerwise_kd_enabled", False)),
            layerwise_kd_weight=float(args.get("layerwise_kd_weight", 1.0)),
            layerwise_kd_layers=None,  # 可以从args中获取，目前使用None表示所有层
            layerwise_kd_pooling=str(args.get("layerwise_kd_pooling", "mean")),
            layerwise_kd_loss_type=str(args.get("layerwise_kd_loss_type", "mse")),
            layerwise_kd_weight_strategy=str(args.get("layerwise_kd_weight_strategy", "uniform")),
        )

        # 处理自动检测选项
        aux_dataset_type = args.get("aux_dataset_type", "imagenet")
        auto_detect = bool(args.get("aux_auto_detect", False))
        
        # 如果指定了"auto"类型，则启用自动检测
        if aux_dataset_type.lower() == "auto":
            auto_detect = True
            aux_dataset_type = None  # 将由自动检测确定
        
        reference_cfg = ReferenceConfig(
            enabled=bool(args.get("clip_use_reference_data", False)),
            dataset_type=str(aux_dataset_type).lower() if aux_dataset_type else "imagenet",
            dataset_path=args.get("auxiliary_data_path"),
            batch_size=int(args.get("reference_batch_size", args["batch_size"])),
            num_workers=int(args.get("clip_num_workers", 3)),
            pin_memory=bool(args.get("clip_pin_memory", True)),
            # 新增配置选项
            auto_detect=auto_detect,
            type_hint=args.get("aux_type_hint"),
            num_samples=args.get("aux_num_samples"),
            split=args.get("aux_split", "val")
        )
        return optim_cfg, loop_cfg, reg_cfg, reference_cfg

    @torch.no_grad()
    def zeroshot_classifier(
        self,
        classnames: Iterable[str],
        templates: Iterable[Any],
    ) -> torch.Tensor:
        """Build a zeroshot classifier from CLIP text embedddings."""
        return self.data_eval_manager.zeroshot_classifier(classnames, templates)

    def evaluate_zeroshot(self, task_idx: int) -> float:
        """Evaluate zeroshot accuracy on the specified task index."""
        return self.data_eval_manager.evaluate_zeroshot(task_idx, self.clip_manager, self.test_transform)

    def save_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        save_checkpoint(self.network, self._cur_task, prefix)

    def store_model_snapshot(self) -> None:
        """Save a full snapshot of the current model state before training."""
        self.model_snapshot = store_model_snapshot(self.network, self._cur_task)

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
        try:
            vm = getattr(self.network, "model", None)
            if vm is not None:
                vm = getattr(vm, "vision_model", None)
                if vm is not None and hasattr(vm, "merge_lora_weights"):
                    vm.merge_lora_weights()
        except AttributeError:
            pass

        self._cur_task += 1
        self.data_eval_manager.set_current_task_mapping(self._cur_task, self.clip_manager)

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

    def store_prev_params(self) -> None:
        """Snapshot of trainable weights (used for L2-protection)."""
        self.prev_params = store_prev_params(self.network, self.l2_protection)  # type: ignore

    def l2_protection_loss(self) -> torch.Tensor:
        """L2-penalty that keeps current weights close to the snapshot."""
        return l2_protection_loss(self.network, self.prev_params, self.l2_lambda, self.device)  # type: ignore

    def _run_training_loop(
        self,
        train_loader: DataLoader,
        zeroshot_weights: torch.Tensor,
        reference_loader: Optional[DataLoader],
    ) -> None:
        """Iterate over the training dataloader for the configured number of steps."""

        try:
            params: List[nn.Parameter] = [param for param in self.network.parameters() if param.requires_grad]  # type: ignore
        except AttributeError:
            params = []
        optimizer, scheduler = self.training_ref_manager.configure_optimizer(params)

        self.training_ref_manager.run_training_loop(
            train_loader,
            zeroshot_weights,
            optimizer,
            scheduler,
            self.log_interval,
            self._cur_task,
            self.data_eval_manager._current_global_to_local,
            self.data_eval_manager._current_num_classes,
            self.use_feature_kd,
            self.monitor_ema,
            self.history
        )

    def update_projection_matrices(self, initial_weight = 1.0, incremental_weight = 0.9) -> None:
        """Update OSGP projection matrices using the current training data."""
        self.covariances = update_projection_matrices(
            self.network, self.train_loader_test_mode, self.covariances, initial_weight, incremental_weight  # type: ignore
        )

    def loop(self) -> Dict[str, List[Union[float, None]]]:
        """Run incremental training across the configured dataset sequence."""

        self._initialise_reference_components()

        # If using auxiliary/reference data, update projection matrices
        # before any task starts so auxiliary features are protected.
        if self.use_reference_data:
            if self.training_ref_manager.reference_loader is not None:
                self.train_loader_test_mode = self.training_ref_manager.reference_loader
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
            templates = self.data_eval_manager.resolve_templates(self.clip_manager.get_dataset_templates(task_idx))
            zeroshot_weights = self.zeroshot_classifier(class_names, templates)

            train_dataset = self.clip_manager.get_task_dataset(
                task_idx,
                source="train",
                cumulative=False,
                transform=self.train_transform,
            )
            train_loader = self.data_eval_manager.build_dataloader(train_dataset, train=True)

            self.train_loader_test_mode = self.data_eval_manager.build_dataloader(
                self.clip_manager.get_task_dataset(
                    task_idx,
                    source="train",
                    cumulative=False,
                    transform=self.test_transform,
                ),
                train=False,
            )

            self.incremental_train(train_loader, zeroshot_weights, self.training_ref_manager.reference_loader)

            logging.info("Evaluating zeroshot performance after task %d", task_idx + 1)
            zeroshot_results = {}

            for eval_idx in range(len(self.dataset_names)):
            # for eval_idx in range(task_idx + 1):
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
        self.training_ref_manager.initialise_reference_components()

    def get_training_history(self) -> Dict[str, List[Union[float, None]]]:
        """Expose training history for external visualisation utilities."""
        return self.history


# Backwards compatibility alias
SubspaceLoRA_CLIP = SubspaceLoRAClipLearner