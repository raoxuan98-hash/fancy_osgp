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
from torch.utils.data import DataLoader

from models.base import BaseLearner
from models.sldc_modules import Drift_Compensator
from utils.inc_net import BaseNet
from lora import compute_covariances
from collections import namedtuple

# ----------------------------------------------------------------------
#  Loss utilities
# ----------------------------------------------------------------------


def symmetric_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sce_a: float = 0.5,
    sce_b: float = 0.5,
) -> torch.Tensor:
    """
    SCE = a·CE + b·Reverse‑CE.

    The implementation aggressively re‑uses the log‑softmax and softmax
    tensors to avoid an extra softmax/log call pair.
    """
    logsoftmax = F.log_softmax(logits, dim=1)
    softmax = logsoftmax.exp()

    # one‑hot labels, very small values are guaranteed by torch
    oh = F.one_hot(targets, num_classes=logits.size(1)).float()
    oh = torch.clamp(oh, min=1e-4, max=1.0)

    ce  = -(oh * logsoftmax).sum(dim=1).mean()
    rce = -(softmax * oh).sum(dim=1).mean()
    return sce_a * ce + sce_b * rce


def feature_distillation_loss(
    teacher_feat: torch.Tensor, student_feat: torch.Tensor
) -> torch.Tensor:
    """L2 distance between teacher and student feature maps."""
    return ((teacher_feat - student_feat) ** 2).mean()


def cosine_similarity_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """1 – cosine similarity, averaged over the batch."""
    return (1.0 - F.cosine_similarity(x1, x2, dim=-1)).mean()


def soft_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float = 2.0,
) -> torch.Tensor:
    """KL‑divergence between softened student / teacher logits."""
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T ** 2)


EvalResult = namedtuple("EvalResult", ["original_fc", "linear_fc"])


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


        # ------------------------------------------------------------------
        #  Checkpoint / bookkeeping
        # ------------------------------------------------------------------
        self._timings: Timing = Timing()
        self.time_history: List[Dict[str, float]] = []

        # ------------------------------------------------------------------
        #  Hyper‑parameters
        # ------------------------------------------------------------------
        self.sce_a: float = args["sce_a"]
        self.sce_b: float = args["sce_b"]

        self.batch_size: int = args["batch_size"]
        self.epochs: int = args["epochs"]
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

        # ------------------------------------------------------------------
        #  Projection / L2 protection
        # ------------------------------------------------------------------
        self.l2_protection: bool = args["l2_protection"]
        self.l2_lambda: float = args["l2_protection_lambda"]

        self.covariances: Dict[str, torch.Tensor] | None = None

        # ------------------------------------------------------------------
        #  Drift compensator
        # ------------------------------------------------------------------
        self.drift_compensator = Drift_Compensator(args)

        # ------------------------------------------------------------------
        #  Other bookkeeping
        # ------------------------------------------------------------------
        self.prev_params: Dict[str, torch.Tensor] | None = None
        self.prev_network: BaseNet | None = None
        self.original_fc: nn.Module | None = None
        self.linear_fc: nn.Module | None = None

        self.seed: int = args["seed"]
        self.task_count: int = 0

        self._eval_tasks: set[int] = set()


        logging.info(
            f"Optimizer instantiated: lrate={self.lrate}, "
            f"wd={self.weight_decay}, optimizer={self.optimizer_type}")

    # ------------------------------------------------------------------
    #  Check‑point utilities
    # ------------------------------------------------------------------
    def save_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        param_dict = {
            n: p.detach().cpu() for n, p in self.network.named_parameters() if p.requires_grad
        }
        payload = {"task": self._cur_task, "model_state_dict": param_dict}
        path = f"{prefix}_after_task_{self._cur_task}.pth"
        torch.save(payload, path)
        logging.info(f"Checkpoint saved to: {path}")

    def _compute_eval_milestones(self, nb_tasks: int) -> set[int]:
        """
        Return a set of task indices (1-based) at ~40%, ~70%, and 100%
        of the total number of tasks. We use ceil to avoid 0.
        """
        import math
        raw = [math.ceil(nb_tasks * 0.4), math.ceil(nb_tasks * 0.7), nb_tasks]
        return set(min(max(1, t), nb_tasks) for t in raw)


    def handle_drift_compensation(self) -> None:
        """Handle the drift compensation and update classifiers."""
        drift_start = time.time()

        # Update drift compensator stats
        self.drift_compensator.update_stats(
            self._cur_task,
            self.prev_network.vit,
            self.network.vit,
            self.train_loader_test_mode,
            self.compensate)

        # Record the drift compensation time
        self._timings.drift = time.time() - drift_start

    def refine_classifiers(self):
        # Optionally refine the classifiers based on drift compensation
        fc_dict = self.drift_compensator.refine_classifiers(self.network.fc, self._cur_task, self.ca_epochs)
        self.original_fc = fc_dict["original"]
        self.linear_fc = fc_dict["linear_compensate"]
        logging.info(f"Refining classifiers at milestone task {self._cur_task} / {self.data_manager.nb_tasks}")


    # ------------------------------------------------------------------
    #  Task‑level bookkeeping
    # ------------------------------------------------------------------
    def after_task(self) -> None:
        """Update class counters after finishing a task."""
        self._known_classes = self._total_classes
        self.update_projection_matrices()
        self.task_count += 1

    # ------------------------------------------------------------------
    #  Incremental training driver (public API)
    # ------------------------------------------------------------------
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
            mode="train",
        )
        
        test_set = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        train_set_test_mode = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )

        # ---- 2️⃣  DataLoaders ------------------------------------------------
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        self.train_loader_test_mode = DataLoader(
            train_set_test_mode,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # ---- 3️⃣  keep a copy of the model before training ------------------
        self.prev_network = copy.deepcopy(self.network).to(self._device)
        self.prev_network.vit.finalize_without_lora()

        self._store_prev_params()

        # ---- 4️⃣  expand classifier head ------------------------------------
        self.network.update_fc(task_size)
        self.network.fc.to(self._device)

        # ---- 5️⃣  system‑level training (backbone + head) --------------------
        logging.info(
            "System training on classes %d‑%d (%s)",
            self._known_classes,
            self._total_classes,
            data_manager.dataset_name.lower())
        
        self._system_training(self.train_loader)

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
    # ------------------------------------------------------------------
    def _make_optimizer(
        self,
        lora_params: List[torch.nn.Parameter],
        fc_params: List[torch.nn.Parameter],
        osgp_params: List[torch.nn.Parameter] | None = None,
    ) -> optim.Optimizer:
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
            return optim.SGD(param_groups, momentum=0.9)
        if self.optimizer_type == "adamw":
            return optim.AdamW(param_groups)
        if self.optimizer_type == "rmsprop":
            return optim.RMSprop(param_groups)
        raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

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
            if p.requires_grad and "fc" not in name
        }

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

        optimizer = self._make_optimizer(lora_params, fc_params, osgp_params)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lrate / 2)

        start = time.time()
        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(train_loader, optimizer, epoch)
            scheduler.step()
        self._timings.train = time.time() - start

    # ------------------------------------------------------------------
    #  Epoch‑wise training
    # ------------------------------------------------------------------
    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch_id: int,
    ) -> None:
        self.network.train()
        total_loss, correct, n_samples = 0.0, 0, 0

        total_kd_loss, total_prior_loss = 0.0, 0.0

        for batch_idx, (_, inputs, targets) in enumerate(loader):
            loss, n_corr, kd_term, prior_term = self._process_batch(
                inputs, targets, optimizer
            )
            total_loss += loss
            correct += n_corr
            n_samples += targets.size(0)

            total_kd_loss += kd_term
            total_prior_loss += prior_term

        avg_loss = total_loss / len(loader)
        accuracy = correct / n_samples

        msg = (
            f"Task {self._cur_task} – Epoch {epoch_id}/{self.epochs} – "
            f"Loss: {avg_loss:.7f} – Acc: {accuracy:.7f}"
        )
        if self.use_feature_kd and self._cur_task > 0:
            avg_kd = total_kd_loss / len(loader)
            msg += f" – Avg KD loss: {avg_kd / self.args['gamma_kd']:.7f}"
        avg_prior = total_prior_loss / len(loader)
        msg += f" – Avg Prior loss: {avg_prior / self.args['kl_gamma']:.7f}"
        logging.info(msg)

    # ------------------------------------------------------------------
    #  Process single batch (forward → loss → backward → step)
    # ------------------------------------------------------------------
    def _process_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> Tuple[float, int, float, float]:
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        # Forward with autocast (auto‑mixed precision)
        feats = self.network.vit(inputs)          # (B, D)
        logits = self.network.fc(feats)           # (B, C)

        # ------------------------------------------------------------------
        #  Knowledge Distillation
        # ------------------------------------------------------------------
        kd_term = 0.0
        if self.use_feature_kd and self._cur_task > 0:
            with torch.no_grad():
                prev_feats = self.prev_network.vit(inputs)   # keep on same device

            kd_feat = self.kd_loss_fn(prev_feats, feats)
            kd_norm = self._norm_loss(prev_feats, feats)
            kd_term = self.gamma_kd * kd_feat + self.gamma_norm * kd_norm

        # ------------------------------------------------------------------
        #  SCE on newly introduced classes
        # ------------------------------------------------------------------
        new_targets_rel = torch.where(
            targets - self._known_classes >= 0,
            targets - self._known_classes,
            -100,
        )
        new_logits = logits[:, self._known_classes :]
        
        sce = symmetric_cross_entropy_loss(
            new_logits, new_targets_rel, self.sce_a, self.sce_b
        )

        # ------------------------------------------------------------------
        #  L2‑Protection & Prior
        # ------------------------------------------------------------------
        l2_term = self._l2_protection_loss()
        prior_term = self.network.vit.kl_regularization()

        # prior_term = self.network.vit.directional_reg()

        # ------------------------------------------------------------------
        #  Total loss & optimisation step
        # ------------------------------------------------------------------
        loss = sce + kd_term + l2_term + prior_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------------------------------------------------------------
        #  Statistics
        # ------------------------------------------------------------------
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            n_correct = (pred == targets).sum().item()

        # Return scalar loss + raw KD / prior contributions (for logging)
        kd_raw = (
            kd_term.item() / self.gamma_kd
            if isinstance(kd_term, torch.Tensor) and self.gamma_kd != 0
            else float(kd_term)
        )
        prior_raw = (prior_term.item() if isinstance(prior_term, torch.Tensor) else float(prior_term))
        
        return loss.item(), n_correct, kd_raw, prior_raw

    # ------------------------------------------------------------------
    #  Helper for norm‑based KD (used together with feature KD)
    # ------------------------------------------------------------------
    @staticmethod
    def _norm_loss(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """MSE between L2‑norms of teacher / student feature vectors."""
        t_norm = t_feat.norm(p=2, dim=1)
        s_norm = s_feat.norm(p=2, dim=1)
        return F.mse_loss(t_norm, s_norm)

    # ------------------------------------------------------------------
    #  Evaluation utilities
    # ------------------------------------------------------------------
    def evaluate(
        self,
        loader: DataLoader,
        original_fc: nn.Module | None = None,
        linear_fc: nn.Module | None = None,
    ) -> EvalResult:
        """Run inference on ``loader`` and report accuracy for both classifiers."""
        self.network.eval()
        total = 0
        correct_orig, correct_lin = 0, 0

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                feats = self.network.vit(inputs)

                if original_fc is not None:
                    preds = original_fc(feats).argmax(dim=1).cpu()
                    correct_orig += (preds == targets).sum().item()

                if linear_fc is not None:
                    preds = linear_fc(feats).argmax(dim=1).cpu()
                    correct_lin += (preds == targets).sum().item()

                total += targets.size(0)

        def _pct(correct: int) -> float:
            return float(np.around(100.0 * correct / total, 2))

        return EvalResult(original_fc=_pct(correct_orig), linear_fc=_pct(correct_lin))

    def eval_task(self) -> EvalResult:
        """Evaluate the current task using both classifiers."""
        
        results = self.evaluate(
            self.test_loader,
            original_fc=self.original_fc,
            linear_fc=self.linear_fc,
        )

        logging.info(
            f"Task {self._cur_task} – Eval → Orig: {results.original_fc:.2f}%, "
            f"Compensated: {results.linear_fc:.2f}%"
        )
        if not hasattr(self, "all_task_results"):
            self.all_task_results: Dict[int, EvalResult] = {}
        self.all_task_results[self._cur_task] = results
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
                # print(1)
            else:
                for k in self.covariances:
                    # print(0.9 * self.covariances[k].diag().mean())
                    # print(0.3 * self.covariances[k].diag().mean())
                    # print("________")
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
            if (self._cur_task + 1) in self._eval_tasks:
                self.refine_classifiers()
                logging.info(f"Evaluating after task {self._cur_task}...")
                task_res = self.eval_task()
                agg["original_fc"].append(task_res.original_fc)
                agg["linear_fc"].append(task_res.linear_fc)

            self.after_task()

        return agg
