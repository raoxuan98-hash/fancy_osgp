# -*- coding: utf-8 -*-
"""
SubspaceLoRA implementation with a handful of performance improvements.
All public API (class name, method signatures, etc.) stays identical to the
original code base.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from models.base import BaseLearner
from models.sldc_modules import Drift_Compensator
from utils.inc_net import BaseNet
from lora import compute_covariances
from collections import namedtuple
import math 
# ----------------------------------------------------------------------
#  Loss utilities
# ----------------------------------------------------------------------


def symmetric_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sce_a: float = 0.5,
    sce_b: float = 0.5) -> torch.Tensor:

    logsoftmax = F.log_softmax(logits, dim=1)
    softmax = logsoftmax.exp()

    oh = F.one_hot(targets, num_classes=logits.size(1)).float()
    oh = torch.clamp(oh, min=1e-4, max=1.0)

    ce  = -(oh * logsoftmax).sum(dim=1).mean()
    rce = -(softmax * oh).sum(dim=1).mean()
    return sce_a * ce + sce_b * rce

def feature_distillation_loss(
    teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
    """L2 distance between teacher and student feature maps."""
    return ((teacher_feat - student_feat) ** 2).mean()

def cosine_similarity_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """1 – cosine similarity, averaged over the batch."""
    return (1.0 - F.cosine_similarity(x1, x2, dim=-1)).mean()
# ----------------------------------------------------------------------
#  Main learner
# ----------------------------------------------------------------------


@dataclass
class Timing:
    train: float = 0.0
    drift: float = 0.0
    total: float = 0.0

class SubspaceLoRA(BaseLearner):
    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        # ------------------------------------------------------------------
        #  Device handling
        # ------------------------------------------------------------------
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------
        #  Base net (use pretrained weights)
        # ------------------------------------------------------------------
        self.network = BaseNet(args, pretrained=True).to(self._device)
        self.args = args

        # Optional JIT compilation (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                self.network = torch.compile(self.network)
                logging.info("Compiled network with torch.compile")
            except Exception as e:
                logging.warning(f"torch.compile failed: {e}")

        self._timings: Timing = Timing()
        self.time_history: List[Dict[str, float]] = []

        self.sce_a: float = args["sce_a"]
        self.sce_b: float = args["sce_b"]

        self.batch_size: int = args["batch_size"]
        self.iterations: int = args["iterations"]
        self.warmup_steps: int = args["warmup_steps"]
        self.ca_epochs: int = args["ca_epochs"]
        self.lrate: float = args["lrate"]
        self.weight_decay: float = args["weight_decay"]
        self.optimizer_type: str = args["optimizer"]
        self.head_scale: float = args["head_scale"]
        self.osgp_scale: float = args["osgp_scale"]
        self.compensate: bool = args["compensate"]

        # ------------------------------------------------------------------
        #  Knowledge Distillation
        # ------------------------------------------------------------------Avg Prior
        kd_type = args["kd_type"]

        if kd_type == "feat":
            self.kd_loss_fn = feature_distillation_loss
        elif kd_type == "cos":
            self.kd_loss_fn = cosine_similarity_loss
        else:
            raise ValueError(f"Unsupported kd_type = {kd_type}")

        self.gamma_kd: float = args["gamma_kd"]
        self.use_feature_kd: bool = self.gamma_kd > 0.0
        self.gamma_norm: float = args["gamma_norm"]
        self.gamma_prior: float = args["kl_gamma"]

        self.l2_protection: bool = args["l2_protection"]
        self.l2_lambda: float = args["l2_protection_lambda"]

        self.covariances: Dict[str, torch.Tensor] | None = None
        self.drift_compensator = Drift_Compensator(args)

        self.prev_params: Dict[str, torch.Tensor] | None = None
        self.prev_network: BaseNet | None = None
        self.original_fc: nn.Module | None = None
        self.linear_fc: nn.Module | None = None

        self.seed: int = args["seed"]
        self.task_count: int = 0

        self._eval_tasks: set[int] = set()
        logging.info(f"Optimizer instantiated: lrate={self.lrate}, wd={self.weight_decay}, optimizer={self.optimizer_type}")

    # ------------------------------------------------------------------
    #  Check‑point utilities
    # ------------------------------------------------------------------
    def save_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        param_dict = {
            n: p.detach().cpu() for n, p in self.network.named_parameters() if p.requires_grad}
        
        payload = {"task": self._cur_task, "model_state_dict": param_dict}
        path = f"{prefix}/after_task_{self._cur_task}.pth"
        torch.save(payload, path)
        logging.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        path = f"{prefix}/after_task_{self._cur_task}.pth"
        param_dict = torch.load(path)['model_state_dict']
        self.network.load_state_dict(param_dict, strict=False)
        logging.info(f"Checkpoint loaded from: {path}")

    def _compute_eval_milestones(self, nb_tasks: int) -> set[int]:
        import math
        raw = [math.ceil(nb_tasks * 0.4), math.ceil(nb_tasks * 0.7), nb_tasks]
        return set(min(max(1, t), nb_tasks) for t in raw)


    def handle_drift_compensation(self) -> None:
        """Handle the drift compensation and update classifiers."""
        drift_start = time.time()

        # Update drift compensator stats
        self.drift_compensator.build_all_variants(
            self._cur_task,
            self.prev_network.vit,
            self.network.vit,
            self.train_loader_test_mode)

        # Record the drift compensation time
        self._timings.drift = time.time() - drift_start

    def refine_classifiers(self):
        # Optionally refine the classifiers based on drift compensation
        self.fc_dict = self.drift_compensator.refine_classifiers_from_variants(self.network.fc, self._cur_task, self.ca_epochs)

    def after_task(self) -> None:
        """Update class counters after finishing a task."""
        self._known_classes = self._total_classes
        self.update_projection_matrices()
        self.task_count += 1

    def incremental_train(self, data_manager) -> None:
        """Entry‑point for training on a new task."""
        start_time = time.time()
        self._cur_task += 1

        # ---- 1️⃣  task split -------------------------------------------------
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self.topk = min(self._total_classes, 5)

        train_set = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train")
        
        test_set = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        
        train_set_test_mode = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test")

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True,
            persistent_workers=True)
        
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=3,
            pin_memory=True,
            persistent_workers=True)
        
        dataset_size = len(train_set_test_mode)
        max_samples = getattr(self, 'max_train_test_samples', 1000)  # 默认采样1000个样本，可配置

        sampler = None
        if dataset_size > max_samples:
            # 随机采样 max_samples 个样本索引
            indices = torch.randperm(dataset_size)[:max_samples].tolist()
            sampler = SubsetRandomSampler(indices)
            print(f"⚠️ Dataset too large ({dataset_size}), sampling {max_samples} examples for test-mode training set.")

        # 创建 DataLoader，注意：使用 sampler 时 shuffle 必须为 False
        self.train_loader_test_mode = DataLoader(
            train_set_test_mode,
            batch_size=self.batch_size,
            shuffle=False if sampler else True,  # 有 sampler 时不能 shuffle
            sampler=sampler,
            num_workers=3,
            pin_memory=True,
            persistent_workers=True)

        # ---- 3️⃣  keep a copy of the model before training ------------------
        self.prev_network = copy.deepcopy(self.network).to(self._device)
        self.prev_network.vit.finalize_without_lora()

        self._store_prev_params()

        # ---- 4️⃣  expand classifier head ------------------------------------
        self.network.update_fc(task_size)
        self.network.fc.to(self._device)

        # ---- 5️⃣  system‑level training (backbone + head) --------------------
        logging.info(
            "System training on classes %d-%d (%s)",
            self._known_classes,
            self._total_classes,
            data_manager.dataset_name.lower())
        
        if self.args['eval_only']:
            self.load_checkpoint(self.args["log_path"])
        else:
            self._system_training(self.train_loader)
            self.save_checkpoint(self.args["log_path"])

        # ---- 6️⃣  drift compensation -----------------------------------------
        self.handle_drift_compensation()

        # ---- 7️⃣  timing ----------------------------------------------------
        self._timings.total = time.time() - start_time

        logging.info(
            "Task %d finished – total: %.2f s | train: %.2f s | drift: %.2f s",
            self._cur_task,
            self._timings.total,
            self._timings.train,
            self._timings.drift)

    
    # ------------------------------------------------------------------
    #  Optimiser factory
    def _make_optimizer(
        self,
        lora_params: List[torch.nn.Parameter],
        fc_params: List[torch.nn.Parameter],
        osgp_params: List[torch.nn.Parameter] | None = None) -> optim.Optimizer:
        """Create optimizer according to ``self.optimizer_type``."""
        if osgp_params is None:
            param_groups = [
                {
                    "params": lora_params,
                    "lr": self.lrate,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": fc_params,
                    "lr": 1e-3,
                    "weight_decay": self.weight_decay,
                },
            ]
        else:
            param_groups = [
                {
                    "params": lora_params,
                    "lr": self.lrate,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": fc_params,
                    "lr": 1e-3,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": osgp_params,
                    "lr": 1e-3 * self.osgp_scale,
                    "weight_decay": 1e-4,
                },
            ]

        if self.optimizer_type == "sgd":
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif self.optimizer_type == "adamw":
            optimizer = optim.AdamW(param_groups)
        elif self.optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

        if self.warmup_steps > 0:
            base_lr = self.lrate

            # 定义每个参数组的学习率调度函数
            def lora_lr_lambda(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))  # 0 → 1
                else:
                    # 余弦退火：从 1.0 → 0.5
                    progress = (step - self.warmup_steps) / max(1, self.iterations - self.warmup_steps)
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1.0 → 0.0
                    return 0.5 + 0.5 * cosine_decay  # 1.0 → 0.5

            # fc 和 osgp 保持恒定学习率（乘以 1.0）
            def const_lr_lambda(step):
                return 1.0

            # 构造 lr_lambda 列表：每个 param_group 一个函数
            lr_lambdas = [lora_lr_lambda]  # group 0: lora
            lr_lambdas.append(const_lr_lambda)  # group 1: fc
            if osgp_params is not None:
                lr_lambdas.append(const_lr_lambda)  # group 2: osgp

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas, last_epoch=-1)

        else:
            # 无 warmup：所有参数组使用统一 cosine 调度
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.iterations, eta_min=self.lrate * 0.5)

        return optimizer, scheduler
    # ------------------------------------------------------------------
    #  L2‑Protection helpers
    # ------------------------------------------------------------------
    def _store_prev_params(self) -> None:
        """Snapshot of trainable weights (used for L2‑protection)."""
        if not self.l2_protection:
            self.prev_params = None
            return

        # Only store the *trainable* part (skip classifier)
        self.prev_params = {
            name: p.clone().detach()
            for name, p in self.network.named_parameters()
            if p.requires_grad and "fc" not in name}

    def _l2_protection_loss(self) -> torch.Tensor:
        """L2‑penalty that keeps current weights close to the snapshot."""
        if not self.l2_protection or self.prev_params is None or self._cur_task == 0:
            return torch.tensor(0.0, device=self._device)

        loss = 0.0
        for name, param in self.network.named_parameters():
            if not param.requires_grad or name.startswith("fc"):
                continue
            old = self.prev_params.get(name)
            if old is None:
                continue
            loss = loss + ((param - old.to(self._device)) ** 2).sum()
        return self.l2_lambda * loss

    # ------------------------------------------------------------------
    #  System‑level training (backbone + head)
    # ------------------------------------------------------------------
    def _system_training(self, train_loader: DataLoader) -> None:
        """Train ViT + new classifier head for ``self.epochs`` epochs."""
        # Collect trainable parameter groups
        fc_params = list(self.network.fc.parameters())

        if self.network.vit.optimizable:
            lora_params, osgp_params = self.network.vit.collect_vit_and_delta_params()
        else:
            osgp_params = None
            lora_params = [p for p in self.network.vit.parameters() if p.requires_grad]

        optimizer, scheduler = self._make_optimizer(lora_params, fc_params, osgp_params)
        start = time.time()
        self.network.train()
        
        done = False
        step = 1
        while True:
            for batch in self.train_loader:
                inputs, targets = batch[1], batch[2]
                loss, n_corr, kd_term, prior_term = self.process_batch(inputs, targets, optimizer)

                if step == 1:
                    total_loss = loss
                    total_kd_loss = kd_term
                    total_prior_loss = prior_term
                    total_correct = n_corr/inputs.size(0)
                else:
                    total_loss = 0.9*total_loss + 0.1*loss
                    total_kd_loss = 0.9*total_kd_loss + 0.1*kd_term
                    total_prior_loss = 0.9*total_prior_loss + 0.1*prior_term
                    total_correct = 0.9*total_correct + 0.1*n_corr/inputs.size(0)

                if step % 10 == 0:
                    logging.info('step: %d, loss: %.4f, kd_loss: %.4f, prior_loss: %.4f, acc: %.4f' % (step, total_loss, total_kd_loss, total_prior_loss, total_correct))
    
                scheduler.step()
                step += 1

                if step == self.iterations:
                    done = True
                    break

            if done:
                break

        self._timings.train = time.time() - start

    def process_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: optim.Optimizer) -> Tuple[float, int, float, float]:
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        # Forward with autocast (auto‑mixed precision)
        feats = self.network.vit(inputs)          # (B, D)
        logits = self.network.fc(feats)           # (B, C)

        kd_term = 0.0
        if self.use_feature_kd and self._cur_task > 0:
            with torch.no_grad():
                prev_feats = self.prev_network.vit(inputs)   # keep on same device

            kd_feat = self.kd_loss_fn(prev_feats, feats)
            kd_norm = self._norm_loss(prev_feats, feats)
            kd_term = self.gamma_kd * kd_feat + self.gamma_norm * kd_norm

        new_targets_rel = torch.where(
            targets - self._known_classes >= 0,
            targets - self._known_classes, -100)
        new_logits = logits[:, self._known_classes :]
        
        sce = symmetric_cross_entropy_loss(
            new_logits, new_targets_rel, self.sce_a, self.sce_b)

        l2_term = self._l2_protection_loss()
        prior_term = self.network.vit.kl_regularization()
        loss = sce + kd_term + l2_term + prior_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            n_correct = (pred == targets).sum().item()

        # Return scalar loss + raw KD / prior contributions (for logging)
        kd_raw = (kd_term.item() / self.gamma_kd if isinstance(kd_term, torch.Tensor) and self.gamma_kd != 0 else float(kd_term))
        prior_raw = (prior_term.item() if isinstance(prior_term, torch.Tensor) else float(prior_term))
        
        return loss.item(), n_correct, kd_raw, prior_raw

    @staticmethod
    def _norm_loss(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """MSE between L2‑norms of teacher / student feature vectors."""
        t_norm = t_feat.norm(p=2, dim=1)
        s_norm = s_feat.norm(p=2, dim=1)
        return F.mse_loss(t_norm, s_norm)

    def evaluate(
        self,
        loader: DataLoader,
        fc_dict):

        """Run inference on ``loader`` and report accuracy for both classifiers."""
        self.network.eval()
        total = 0
        corrects = {}
        for name, fc in fc_dict.items():
            corrects[name] = 0

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                feats = self.network.vit(inputs)

                for name, fc in fc_dict.items():
                    preds = fc(feats).argmax(dim=1).cpu()
                    corrects[name] += (preds == targets).sum().item()

                total += targets.size(0)
            
        for name, correct in corrects.items():
            corrects[name] = float(np.around(100 * correct / total, 2))
        
        return corrects

    def eval_task(self):
        """Evaluate the current task using both classifiers."""
        
        results = self.evaluate(
            self.test_loader,
            fc_dict=self.fc_dict)

        # 动态格式化输出所有分类器的结果
        result_str = ", ".join([f"{name}: {acc:.2f}%" for name, acc in results.items()])
        logging.info(f"Task {self._cur_task} – Eval → {result_str}")

        if not hasattr(self, "all_task_results"):
            self.all_task_results: Dict[int, Dict[str, float]] = {}
        self.all_task_results[self._cur_task] = results  # 保存整个字典

        return results

    # ------------------------------------------------------------------
    #  Projection matrix update (kept – only the core update, no logging)
    # ------------------------------------------------------------------
    def update_projection_matrices(self):
        """Update OSGP projection matrices using the current training data."""
        if self._cur_task >= 0 and self.network.vit.use_projection:
            new_covs = compute_covariances(self.network.vit, self.train_loader_test_mode)

            if self.covariances is None:
                self.covariances = new_covs
            else:
                for k in self.covariances:
                    self.covariances[k] = 0.9 * self.covariances[k] + new_covs[k]

            self.network.vit.update_projection_matrices(new_covs)

    # ------------------------------------------------------------------
    #  Full incremental loop (called by the training script)
    # ------------------------------------------------------------------
    def loop(self, data_manager) -> Dict[str, List[float | None]]:

        agg: Dict[str, List[float | None]] = {
            "original_fc": [],
            "linear_fc": []}

        self.data_manager = data_manager
        self._eval_tasks = self._compute_eval_milestones(data_manager.nb_tasks)
        logging.info(f"Classifier refinement scheduled at tasks: {sorted(self._eval_tasks)}")

        for _ in range(data_manager.nb_tasks):
            self.incremental_train(data_manager)
            # if (self._cur_task + 1) in self._eval_tasks:
            self.refine_classifiers()
            logging.info(f"Evaluating after task {self._cur_task}...")
            task_res = self.eval_task()
            self.after_task()
        return task_res
