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
from models.training_components import feature_distillation_loss, bidirectional_kl_loss
from models.layerwise_distillation import LayerwiseFeatureCollector, layerwise_feature_distillation_loss, create_layer_weights
from models.reference_dataset import ReferenceDatasetFactory, DatasetDetectionError, DatasetLoadError

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
        self.bidirectional_kd = reg_cfg.bidirectional_kd
        
        # Layer-wise特征蒸馏配置
        self.layerwise_kd_enabled = reg_cfg.layerwise_kd_enabled
        self.layerwise_kd_weight = reg_cfg.layerwise_kd_weight
        self.layerwise_kd_layers = reg_cfg.layerwise_kd_layers
        self.layerwise_kd_pooling = reg_cfg.layerwise_kd_pooling
        self.layerwise_kd_loss_type = reg_cfg.layerwise_kd_loss_type
        self.layerwise_kd_weight_strategy = reg_cfg.layerwise_kd_weight_strategy
        
        # 初始化特征收集器（稍后设置）
        self.teacher_feature_collector = None
        self.student_feature_collector = None
        
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
        
        # 只有在启用layer-wise蒸馏时才初始化特征收集器
        self.teacher_feature_collector = None
        self.student_feature_collector = None
            
    def setup_layerwise_collectors(self, teacher_network, student_network):
        """设置layer-wise特征收集器"""
        logging.info(f"尝试设置layer-wise特征收集器，layerwise_kd_enabled: {self.layerwise_kd_enabled}")
        logging.info(f"layerwise_kd_weight: {self.layerwise_kd_weight}")
        logging.info(f"layerwise_kd_pooling: {self.layerwise_kd_pooling}")
        logging.info(f"layerwise_kd_loss_type: {self.layerwise_kd_loss_type}")
        logging.info(f"layerwise_kd_weight_strategy: {self.layerwise_kd_weight_strategy}")
        
        # 确保特征收集器初始化为None
        self.teacher_feature_collector = None
        self.student_feature_collector = None
        
        if self.layerwise_kd_enabled:
            try:
                self.teacher_feature_collector = LayerwiseFeatureCollector(
                    teacher_network,
                    self.layerwise_kd_layers,
                    self.layerwise_kd_pooling
                )
                self.student_feature_collector = LayerwiseFeatureCollector(
                    student_network,
                    self.layerwise_kd_layers,
                    self.layerwise_kd_pooling
                )
                logging.info(f"Layer-wise特征蒸馏已启用，池化方式: {self.layerwise_kd_pooling}")
                logging.info(f"教师特征收集器: {self.teacher_feature_collector is not None}")
                logging.info(f"学生特征收集器: {self.student_feature_collector is not None}")
            except Exception as e:
                logging.warning(f"无法初始化layer-wise特征收集器: {e}")
                self.layerwise_kd_enabled = False
                self.teacher_feature_collector = None
                self.student_feature_collector = None
        else:
            logging.info("Layer-wise特征蒸馏未启用，跳过特征收集器设置")
    
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
        
        logging.info(f"初始化参考组件 - enabled: {self.reference_cfg.enabled}")
        logging.info(f"参考数据集路径: {getattr(self.reference_cfg, 'dataset_path', 'None')}")
        logging.info(f"参考数据集类型: {getattr(self.reference_cfg, 'dataset_type', 'None')}")
        logging.info(f"自动检测: {getattr(self.reference_cfg, 'auto_detect', 'None')}")
        
        self.reference_loader = None
        self.reference_iter = None
        self.reference_text_embeddings = None
        self.reference_text_labels = None
        self.reference_teacher_embeddings = None
        self._n_reference_text = 0

        if not self.reference_cfg.enabled:
            logging.info("Reference dataset disabled; skipping data loader.")
            return

        try:
            reference_dataset = self._build_reference_dataset()
            logging.info(f"参考数据集创建成功，样本数量: {len(reference_dataset)}")
            ref_workers = int(self.reference_cfg.num_workers)
            self.reference_loader = DataLoader(
                reference_dataset,
                batch_size=self.reference_cfg.batch_size,
                shuffle=True,
                num_workers=ref_workers,
                pin_memory=self.reference_cfg.pin_memory
            )
            self.reference_iter = iter(self.reference_loader)
            logging.info("参考数据加载器创建成功")
            
        except Exception as e:
            logging.error(f"参考数据集初始化失败: {e}")
            self.reference_loader = None
            self.reference_iter = None
            return

        logging.info("Precomputing reference text embeddings ...")
        with torch.no_grad():
            unique_ref_labels, unique_ref_prompts = reference_dataset.return_labels_and_prompts()
            
            # 应用样本数量限制到文本嵌入计算
            num_samples = getattr(self.reference_cfg, 'num_samples', None)
            if num_samples is not None and len(unique_ref_labels) > num_samples:
                logging.info(f"文本嵌入计算限制为前 {num_samples} 个样本（原始: {len(unique_ref_labels)}）")
                unique_ref_labels = unique_ref_labels[:num_samples]
                unique_ref_prompts = unique_ref_prompts[:num_samples]
            
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
        """Factory method returning the configured reference dataset using the new dataset factory."""
        
        aux_path = self.reference_cfg.dataset_path
        
        # 检查是否启用自动检测
        auto_detect = getattr(self.reference_cfg, 'auto_detect', False)
        
        # 确保num_samples参数被正确传递
        num_samples = getattr(self.reference_cfg, 'num_samples', None)
        if num_samples is not None:
            logging.info(f"参考数据集样本数量限制为: {num_samples}")
        
        if auto_detect:
            # 使用自动检测模式
            type_hint = getattr(self.reference_cfg, 'type_hint', None)
            try:
                logging.info("使用自动检测模式创建参考数据集")
                return ReferenceDatasetFactory.create_dataset_auto_detect(
                    dataset_path=aux_path,
                    transform=self.network.valid_preprocess,
                    num_samples=num_samples,
                    type_hint=type_hint,
                    split=getattr(self.reference_cfg, 'split', 'val')
                )
            except (DatasetDetectionError, DatasetLoadError) as e:
                logging.error(f"自动检测数据集失败: {e}")
                raise ValueError(f"无法自动检测和创建数据集: {e}")
        else:
            # 使用指定的数据集类型
            aux_type = self.reference_cfg.dataset_type
            if not aux_type:
                raise ValueError("当auto_detect为False时，必须指定dataset_type")
            
            if not aux_path:
                raise ValueError("dataset_path必须提供")
            
            try:
                logging.info(f"使用指定类型创建参考数据集: {aux_type}")
                return ReferenceDatasetFactory.create_dataset(
                    dataset_type=aux_type,
                    dataset_path=aux_path,
                    transform=self.network.valid_preprocess,
                    num_samples=num_samples,
                    split=getattr(self.reference_cfg, 'split', 'val')
                )
            except (ValueError, DatasetLoadError) as e:
                logging.error(f"创建指定类型数据集失败: {e}")
                raise ValueError(f"无法创建{aux_type}数据集: {e}")
    
    def next_reference_batch(self) -> ReferenceBatch:
        """Retrieve the next reference batch, rewinding the iterator if necessary."""

        # 添加详细调试日志
        logging.debug(f"[DEBUG] next_reference_batch called")
        logging.debug(f"[DEBUG] reference_loader is None: {self.reference_loader is None}")
        logging.debug(f"[DEBUG] reference_cfg.enabled: {self.reference_cfg.enabled}")

        if not self.reference_cfg.enabled or self.reference_loader is None:
            logging.debug(f"[DEBUG] Returning empty reference batch: enabled={self.reference_cfg.enabled}, loader={self.reference_loader is not None}")
            return ReferenceBatch(images=None, labels=None)

        assert self.reference_iter is not None, "reference_iter must be initialised when reference data is enabled"
        try:
            batch = next(self.reference_iter)
        except StopIteration:
            logging.debug("[DEBUG] Reference iterator exhausted, rewinding...")
            self.reference_iter = iter(self.reference_loader)
            batch = next(self.reference_iter)

        if isinstance(batch, dict):
            images = batch.get("images")
            labels = batch.get("labels")
        else:
            images, labels = batch

        # 添加数据形状和类型的调试信息
        if isinstance(images, torch.Tensor):
            logging.debug(f"[DEBUG] reference_images shape: {images.shape}, dtype: {images.dtype}")
            images = images.to(self.device, non_blocking=True)
        else:
            logging.debug(f"[DEBUG] reference_images is not a tensor: {type(images)}")
            
        if isinstance(labels, torch.Tensor):
            logging.debug(f"[DEBUG] reference_labels shape: {labels.shape}, dtype: {labels.dtype}")
            labels = labels.to(dtype=torch.long, device="cpu")
        elif isinstance(labels, (list, tuple)):
            logging.debug(f"[DEBUG] reference_labels is list/tuple with length: {len(labels)}")
            labels = torch.as_tensor(labels, dtype=torch.long)
        else:
            logging.debug(f"[DEBUG] reference_labels is unexpected type: {type(labels)}")
            
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
    ) -> Tuple[TrainingStepMetrics, Dict[str, float]]:
        """Forward/backward pass, including optional reference distillation."""

        # 添加详细调试日志
        logging.debug(f"[DEBUG] run_training_step called with use_feature_kd: {use_feature_kd}")
        logging.debug(f"[DEBUG] reference_batch.images is None: {reference_batch.images is None}")
        if reference_batch.images is not None:
            logging.debug(f"[DEBUG] reference_batch.images shape: {reference_batch.images.shape}")
        logging.debug(f"[DEBUG] reference_batch.labels is None: {reference_batch.labels is None}")
        if reference_batch.labels is not None:
            if isinstance(reference_batch.labels, torch.Tensor):
                logging.debug(f"[DEBUG] reference_batch.labels shape: {reference_batch.labels.shape}")
            else:
                logging.debug(f"[DEBUG] reference_batch.labels type: {type(reference_batch.labels)}")

        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        reference_images = reference_batch.images
        if isinstance(reference_images, torch.Tensor) and reference_images.device != self.device:
            reference_images = reference_images.to(self.device, non_blocking=True)

        kd_metrics = {
            "ref_feature_l2": 0.0,
            "ref_feature_cosine": 0.0,
            "ref_raw_kl": 0.0,
            "layerwise_kd_loss": 0.0,  # 添加layerwise蒸馏损失的初始化
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

                logging.debug(f"调用参考损失计算 - use_feature_kd: {use_feature_kd}, reference_images: {reference_images is not None}, reference_img_feats: {reference_img_feats is not None}")
                kd_term, kd_metrics = self._compute_reference_regularisation(
                    reference_images,
                    reference_img_feats,
                    reference_batch.labels,
                    use_feature_kd,
                )
                logging.debug(f"参考损失计算结果 - kd_term: {kd_term.item() if torch.is_tensor(kd_term) else kd_term}")
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
            return TrainingStepMetrics(loss=0.0, correct=0, kd_value=0.0, prior_value=0.0, batch_size=0), kd_metrics

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
        ), kd_metrics
    
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
        logging.debug(f"_compute_classification_logits - input_img_feats dtype: {input_img_feats.dtype}, zeroshot_weights dtype: {zeroshot_weights.dtype}")
        # 确保input_img_feats与zeroshot_weights的数据类型一致
        if input_img_feats.dtype != zeroshot_weights.dtype:
            input_img_feats = input_img_feats.to(dtype=zeroshot_weights.dtype)
            logging.debug(f"Converted input_img_feats to dtype: {input_img_feats.dtype}")
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

        # 添加详细调试日志
        logging.debug(f"[DEBUG] _compute_reference_regularisation called")
        logging.debug(f"[DEBUG] use_feature_kd: {use_feature_kd}")
        logging.debug(f"[DEBUG] reference_images is None: {reference_images is None}")
        logging.debug(f"[DEBUG] student_ref_feats is None: {student_ref_feats is None}")
        logging.debug(f"[DEBUG] reference_text_embeddings is None: {self.reference_text_embeddings is None}")
        logging.debug(f"[DEBUG] reference_text_labels is None: {self.reference_text_labels is None}")
        logging.debug(f"[DEBUG] reference_labels is None: {reference_labels is None}")
        
        zero = torch.tensor(0.0, device=self.device)
        metrics = {
            "ref_feature_l2": 0.0,
            "ref_feature_cosine": 0.0,
            "ref_raw_kl": 0.0,
            "layerwise_kd_loss": 0.0,  # 添加layerwise蒸馏损失的初始化
            "teacher_ref_probs_min": 0.0,
            "teacher_ref_probs_max": 0.0,
            "student_ref_probs_min": 0.0,
            "student_ref_probs_max": 0.0,
        }

        # 条件检查 - 添加调试日志
        if not use_feature_kd:
            logging.debug("参考损失计算跳过: use_feature_kd = False")
            return zero, metrics
        if reference_images is None:
            logging.debug("参考损失计算跳过: reference_images is None")
            return zero, metrics
        if student_ref_feats is None:
            logging.debug("参考损失计算跳过: student_ref_feats is None")
            return zero, metrics
        if self.reference_text_embeddings is None:
            logging.debug("参考损失计算跳过: reference_text_embeddings is None")
            return zero, metrics
        if self.reference_text_labels is None:
            logging.debug("参考损失计算跳过: reference_text_labels is None")
            return zero, metrics
        if reference_labels is None:
            logging.debug("参考损失计算跳过: reference_labels is None")
            return zero, metrics
            
        # 检查layerwise蒸馏是否应该计算（即使其他条件满足，layerwise蒸馏可能单独计算）
        if self.layerwise_kd_enabled and self.teacher_feature_collector and self.student_feature_collector:
            logging.debug("[DEBUG] Layer-wise蒸馏条件满足，将计算layerwise损失")

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
        if self.bidirectional_kd:
            # 使用双向KL散度
            ref_raw_kl = bidirectional_kl_loss(teacher_logits_ref, student_logits_ref, temperature)
        else:
            # 使用原有的单向KL散度
            teacher_probs = F.softmax(teacher_logits_ref / temperature, dim=-1).detach()
            student_log_probs = F.log_softmax(student_logits_ref / temperature, dim=-1)
            ref_raw_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)

        # 计算layer-wise特征蒸馏损失
        layerwise_kd_loss = torch.tensor(0.0, device=self.device)
        logging.debug(f"[DEBUG] Layer-wise蒸馏检查 - enabled: {self.layerwise_kd_enabled}, teacher_collector: {self.teacher_feature_collector is not None}, student_collector: {self.student_feature_collector is not None}")
        
        # 检查特征收集器是否存在，避免调用None对象的方法
        if self.teacher_feature_collector is None or self.student_feature_collector is None:
            # logging.warning("特征收集器未初始化，跳过layerwise蒸馏")
            layerwise_kd_loss = torch.tensor(0.0, device=self.device)
        elif self.layerwise_kd_enabled and self.teacher_feature_collector and self.student_feature_collector:
            try:
                # 在教师和学生模型前向传播后立即获取特征
                # 教师特征：重新运行教师模型以捕获中间层特征
                with torch.no_grad():
                    with autocast('cuda', **self._autocast_kwargs):
                        _ = self.teacher_network.encode_image(reference_images.to(self.device, non_blocking=True))
                
                # 学生特征：重新运行学生模型以捕获中间层特征
                with autocast('cuda', **self._autocast_kwargs):
                    _ = self.network.encode_image(reference_images.to(self.device, non_blocking=True))
                
                # 获取教师和学生的多层特征
                teacher_layer_features = self.teacher_feature_collector.get_layer_features_list()
                student_layer_features = self.student_feature_collector.get_layer_features_list()
                
                logging.debug(f"[DEBUG] 获取到的特征 - teacher_layers: {len(teacher_layer_features)}, student_layers: {len(student_layer_features)}")
                
                if teacher_layer_features and student_layer_features and len(teacher_layer_features) > 0 and len(student_layer_features) > 0:
                    # 创建层权重
                    layer_weights = create_layer_weights(
                        len(teacher_layer_features),
                        self.layerwise_kd_weight_strategy
                    )
                    
                    # 计算layer-wise蒸馏损失
                    layerwise_kd_loss = layerwise_feature_distillation_loss(
                        teacher_layer_features,
                        student_layer_features,
                        layer_weights,
                        self.layerwise_kd_loss_type
                    )
                    
                    logging.info(f"[INFO] Layer-wise蒸馏损失计算成功: {layerwise_kd_loss.item():.6f}")
                    
                    # 清空特征缓存
                    self.teacher_feature_collector.clear_features()
                    self.student_feature_collector.clear_features()
                else:
                    logging.warning(f"[WARNING] 教师或学生特征为空，跳过layerwise蒸馏损失计算 - teacher: {len(teacher_layer_features) if teacher_layer_features else 0}, student: {len(student_layer_features) if student_layer_features else 0}")
                    
            except Exception as e:
                logging.warning(f"Layer-wise蒸馏损失计算失败: {e}")
                layerwise_kd_loss = torch.tensor(0.0, device=self.device)
        else:
            logging.debug("[DEBUG] Layer-wise蒸馏条件不满足，跳过计算")

        # 总KD损失 = 原有特征损失 + layer-wise特征损失 + KL损失
        kd_term = ref_feature_l2_dist + self.layerwise_kd_weight * layerwise_kd_loss + 2.0 * ref_raw_kl

        # 添加详细的损失计算结果日志
        logging.debug(f"[DEBUG] 参考损失计算结果:")
        logging.debug(f"[DEBUG] ref_feature_l2_dist: {ref_feature_l2_dist.item():.6f}")
        logging.debug(f"[DEBUG] ref_feature_cosine_sim: {ref_feature_cosine_sim.item():.6f}")
        logging.debug(f"[DEBUG] ref_raw_kl: {ref_raw_kl.item():.6f}")
        logging.debug(f"[DEBUG] layerwise_kd_loss: {layerwise_kd_loss.item():.6f}")
        logging.debug(f"[DEBUG] layerwise_kd_weight: {self.layerwise_kd_weight}")
        logging.debug(f"[DEBUG] 最终 kd_term: {kd_term.item():.6f}")

        metrics.update(
            ref_feature_l2=float(ref_feature_l2_dist.item()),
            ref_feature_cosine=float(ref_feature_cosine_sim.item()),
            ref_raw_kl=float(ref_raw_kl.item()),
            layerwise_kd_loss=float(layerwise_kd_loss.item()),
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
        logging.debug(f"Before conversion - zeroshot_weights dtype: {zeroshot_weights.dtype}, compute_dtype: {self.compute_dtype}")
        zeroshot_weights = zeroshot_weights.to(device=self.device, dtype=self.compute_dtype)
        logging.debug(f"After conversion - zeroshot_weights dtype: {zeroshot_weights.dtype}")

        for iteration in range(1, self.iterations + 1):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)

            reference_batch = self.next_reference_batch()
            metrics, kd_metrics = self.run_training_step(
                inputs,
                targets,
                zeroshot_weights,
                reference_batch,
                optimizer,
                current_global_to_local,
                current_num_classes,
                use_feature_kd,
            )
            
            # 更新参考数据相关指标到monitor_ema
            if kd_metrics:
                if "ref_feature_l2" in kd_metrics:
                    monitor_ema["ref_feature_l2"].update(kd_metrics["ref_feature_l2"])
                if "ref_feature_cosine" in kd_metrics:
                    monitor_ema["ref_feature_cosine"].update(kd_metrics["ref_feature_cosine"])
                if "ref_raw_kl" in kd_metrics:
                    monitor_ema["ref_raw_kl"].update(kd_metrics["ref_raw_kl"])
                # 添加layerwise蒸馏损失的监控更新
                if "layerwise_kd_loss" in kd_metrics:
                    monitor_ema["layerwise_kd_loss"].update(kd_metrics["layerwise_kd_loss"])

            effective_batch_size = metrics.batch_size if metrics.batch_size > 0 else inputs.size(0)
            accuracy = metrics.correct / effective_batch_size if effective_batch_size else 0.0
            monitor_ema["ema_acc"].update(accuracy)
            history["ema_acc"].append(monitor_ema["ema_acc"].get())

            # 计算并更新余弦相似度指标
            with torch.no_grad():
                # 获取当前批次的图像特征
                inputs_device = inputs.to(self.device, non_blocking=True)
                current_img_feats = self.network.encode_image(inputs_device)
                current_img_feats = current_img_feats / current_img_feats.norm(dim=-1, keepdim=True)
                
                # 计算与zeroshot权重的余弦相似度
                if zeroshot_weights is not None:
                    # 添加数据类型检查日志
                    logging.debug(f"current_img_feats dtype: {current_img_feats.dtype}, zeroshot_weights dtype: {zeroshot_weights.dtype}")
                    # 确保current_img_feats与zeroshot_weights的数据类型一致
                    if current_img_feats.dtype != zeroshot_weights.dtype:
                        current_img_feats = current_img_feats.to(dtype=zeroshot_weights.dtype)
                        logging.debug(f"Converted current_img_feats to dtype: {current_img_feats.dtype}")
                    similarity = current_img_feats @ zeroshot_weights
                    # 获取正样本（正确类别）和负样本（错误类别）的相似度
                    batch_size = similarity.size(0)
                    for i in range(batch_size):
                        target_idx = targets[i] if targets[i] < similarity.size(1) else 0
                        # 正样本相似度（正确类别的相似度）
                        pos_sim = similarity[i, target_idx].item()
                        # 负样本相似度（除正确类别外的最大相似度）
                        mask = torch.ones(similarity.size(1), dtype=torch.bool, device=similarity.device)
                        mask[target_idx] = False
                        neg_sim = similarity[i, mask].max().item() if mask.any() else 0.0
                        
                        monitor_ema["input_feature_positive_cosine"].update(pos_sim)
                        monitor_ema["input_feature_negative_cosine"].update(neg_sim)

            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else self.lrate
            history["iteration"].append(iteration)
            history["train_loss"].append(metrics.loss)
            history["lr"].append(current_lr)
            
            # 记录layerwise蒸馏损失到历史记录
            if "layerwise_kd_loss" in history and self.layerwise_kd_enabled:
                if "layerwise_kd_loss" in kd_metrics:
                    history["layerwise_kd_loss"].append(kd_metrics["layerwise_kd_loss"])
                else:
                    history["layerwise_kd_loss"].append(0.0)
            elif "layerwise_kd_loss" in history:
                history["layerwise_kd_loss"].append(0.0)

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
            "Task %d Iter %d/%d | lr=%.6g | acc=%.4f | pos_cos=%.6f | neg_cos=%.6f | ref_L2=%.6f | ref_cos=%.6f | ref_KL=%.6f | layerwise_KD=%.6f",
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
            monitor_ema['layerwise_kd_loss'].get()
        )