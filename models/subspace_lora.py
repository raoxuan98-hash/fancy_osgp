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
from models.sldc_modules2 import Drift_Compensator
from utils.inc_net import BaseNet
from lora import compute_covariances
import math 

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
    return ((teacher_feat - student_feat) ** 2).mean()

def cosine_similarity_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(x1, x2, dim=-1)).mean()

@dataclass
class Timing:
    train: float = 0.0
    drift: float = 0.0
    total: float = 0.0

class SubspaceLoRA(BaseLearner):
    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = BaseNet(args, pretrained=True).to(self._device)
        self.args = args

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
        self.compensate: bool = args["compensate"]

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

        self.l2_protection: bool = args["l2_protection"]
        self.l2_lambda: float = args["l2_protection_lambda"]

        self.covariances: Dict[str, torch.Tensor] | None = None
        self.drift_compensator = Drift_Compensator(args)

        self.prev_params: Dict[str, torch.Tensor] | None = None
        self.prev_network = None
        self.original_fc: nn.Module | None = None
        self.seed: int = args["seed"]
        self.task_count: int = 0

        self._eval_tasks: set[int] = set()

        self.current_task_id = 0
        logging.info(f"Optimizer instantiated: lrate={self.lrate}, wd={self.weight_decay}, optimizer={self.optimizer_type}")

    def save_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        param_dict = {
            n: p.detach().cpu() for n, p in self.network.named_parameters() if p.requires_grad}
        payload = {"task": self.current_task_id, "model_state_dict": param_dict}
        path = f"{prefix}/after_task_{self.current_task_id}.pth"
        torch.save(payload, path)
        logging.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        path = f"{prefix}/after_task_{self.current_task_id}.pth"
        param_dict = torch.load(path)['model_state_dict']
        self.network.load_state_dict(param_dict, strict=False)
        logging.info(f"Checkpoint loaded from: {path}")

    def compute_eval_milestones(self, nb_tasks: int) -> set[int]:
        import math
        raw = [math.ceil(nb_tasks * 0.4), math.ceil(nb_tasks * 0.7), nb_tasks]
        return set(min(max(1, t), nb_tasks) for t in raw)

    def handle_drift_compensation(self) -> None:
        """Handle the drift compensation and update classifiers."""
        drift_start = time.time()
        
        self.drift_compensator.build_all_variants(
            self.current_task_id,
            self.prev_network.vit,
            self.network.vit,
            self.train_loader_test_mode)

        self._timings.drift = time.time() - drift_start

    def refine_classifiers(self):
        self.fc_dict = self.drift_compensator.refine_classifiers_from_variants(self.network.fc, self.ca_epochs)
        self.network.fc = next(iter(self.fc_dict.values()))

    def after_task(self) -> None:
        """Update class counters after finishing a task."""
        self._known_classes = self._total_classes
        self.update_projection_matrices()
        self.task_count += 1

    def incremental_train(self, data_manager) -> None:
        start_time = time.time()
        task_size = data_manager.get_task_size(self.current_task_id)
        self.current_task_id += 1
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
            num_workers=4,
            pin_memory=True,
            persistent_workers=True)
        
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True)
        
        dataset_size = len(train_set_test_mode)
        max_samples = getattr(self, 'max_train_test_samples', 5000)

        sampler = None
        if dataset_size > max_samples:
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

        if self.use_feature_kd or self.compensate:
            self.prev_network = copy.deepcopy(self.network).to(self._device)
            self.prev_network.vit.finalize_without_lora()

        self.store_prev_params()
        self.network.update_fc(task_size)
        self.network.fc.to(self._device)

        logging.info(
            "System training on classes %d-%d (%s)",
            self._known_classes,
            self._total_classes,
            data_manager.dataset_name.lower())
        
        if self.args['eval_only']:
            self.load_checkpoint(self.args["log_path"])
        else:
            self.system_training(self.train_loader)
            self.save_checkpoint(self.args["log_path"])

        self.handle_drift_compensation()
        self._timings.total = time.time() - start_time

        logging.info(
            "Task %d finished – total: %.2f s | train: %.2f s | drift: %.2f s",
            self.current_task_id,
            self._timings.total,
            self._timings.train,
            self._timings.drift)

    
    def make_optimizer(
        self,
        lora_params: List[torch.nn.Parameter],
        scale_params: List[torch.nn.Parameter],
        fc_params: List[torch.nn.Parameter]) -> optim.Optimizer:

        """Create optimizer according to ``self.optimizer_type``."""
        
        param_groups = [
            {"params": lora_params, "lr": self.lrate, "weight_decay": self.weight_decay},
            {"params": fc_params, "lr": 1e-3 if self.optimizer_type == "adamw" else 5e-3, "weight_decay": self.weight_decay},
            {"params": scale_params, "lr": 0.01 if self.optimizer_type == "adamw" else 0.1, "weight_decay": 0.0}]

        if self.optimizer_type == "sgd":
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif self.optimizer_type == "adamw":
            optimizer = optim.AdamW(param_groups)
        elif self.optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

        if self.warmup_steps > 0:
            def lora_lr_lambda(step):
                if step < self.warmup_steps:
                    return step / max(1, self.warmup_steps)
                else:
                    progress = (step - self.warmup_steps) / max(1, self.iterations - self.warmup_steps)
                    initial_lr = self.lrate
                    eta_min = getattr(self, 'lora_eta_min', 1e-4)
                    lr_ratio = eta_min / initial_lr
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return lr_ratio + cosine_decay * (1.0 - lr_ratio)
            def const_lr_lambda(step):
                return 1.0
            
            lr_lambdas = [lora_lr_lambda, const_lr_lambda, const_lr_lambda]
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas, last_epoch=-1)
        return optimizer, scheduler

    def store_prev_params(self) -> None:
        """Snapshot of trainable weights (used for L2‑protection)."""
        if not self.l2_protection:
            self.prev_params = None
            return

        self.prev_params = {
            name: p.clone().detach()
            for name, p in self.network.named_parameters()
            if p.requires_grad and "fc" not in name}

    def l2_protection_loss(self) -> torch.Tensor:
        if not self.l2_protection or self.prev_params is None:
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

    def system_training(self, train_loader: DataLoader) -> None:
        """Train ViT + new classifier head for ``self.epochs`` epochs."""
        # Collect trainable parameter groups
        fc_params = list(self.network.fc.parameters())
        lora_params = self.network.model.visual.get_param_groups()
        optimizer, scheduler = self.make_optimizer(lora_params['others'], lora_params['scales'], fc_params)
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

                if step % 200 == 0 or step == self.iterations:
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
        
        if self.use_feature_kd and self.current_task_id > 0:
            with torch.no_grad():
                prev_feats = self.prev_network.vit(inputs)

            kd_feat = self.kd_loss_fn(prev_feats, feats)
            kd_norm = self.norm_loss(prev_feats, feats)
            kd_term = self.gamma_kd * kd_feat + self.gamma_norm * kd_norm

        new_targets_rel = torch.where(
            targets - self._known_classes >= 0,
            targets - self._known_classes, -100)
        
        new_logits = logits[:, self._known_classes :]
        
        sce = symmetric_cross_entropy_loss(new_logits, new_targets_rel, self.sce_a, self.sce_b)

        l2_term = self.l2_protection_loss()
        prior_term = self.network.vit.regularization_loss()
        loss = sce + kd_term + l2_term + prior_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            n_correct = (pred == targets).sum().item()

        kd_raw = (kd_term.item() / self.gamma_kd if isinstance(kd_term, torch.Tensor) and self.gamma_kd != 0 else float(kd_term))
        prior_raw = (prior_term.item() if isinstance(prior_term, torch.Tensor) else float(prior_term))
        
        return loss.item(), n_correct, kd_raw, prior_raw

    @staticmethod
    def norm_loss(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """MSE between L2‑norms of teacher / student feature vectors."""
        t_norm = t_feat.norm(p=2, dim=1)
        s_norm = s_feat.norm(p=2, dim=1)
        return F.mse_loss(t_norm, s_norm)

    def evaluate(
        self,
        loader: DataLoader,
        fc_dict):

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
        results = self.evaluate(
            self.test_loader,
            fc_dict=self.fc_dict)
        if not hasattr(self, "all_task_results"):
            self.all_task_results: Dict[int, Dict[str, float]] = {}
        self.all_task_results[self.current_task_id] = results
        return results

    def update_projection_matrices(self):
        if self.current_task_id >= 0 and self.network.vit.use_projection:
            new_covs = compute_covariances(self.network.vit, self.train_loader_test_mode)
            if self.covariances is None:
                self.covariances = new_covs
            else:
                for k in self.covariances:
                    self.covariances[k] = 0.9 * self.covariances[k] + new_covs[k]
            self.network.vit.update_projection_matrices(new_covs)

    def loop(self, data_manager) -> Dict[str, List[float | None]]:
        self.data_manager = data_manager
        for _ in range(data_manager.nb_tasks):
            self.incremental_train(data_manager)
            self.refine_classifiers()
            logging.info(f"Evaluating after task {self.current_task_id}...")
            self.eval_task()
            self.analyze_task_results(self.all_task_results)
            self.after_task()
        return self.all_task_results

    def analyze_task_results(self, all_task_results: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze incremental learning evaluation results.
        Logs and returns final-task accuracy and average accuracy across tasks for each variant.
        """
        if not all_task_results:
            logging.info("📊 Task evaluation results are empty. Nothing to analyze.")
            return {
                "last_task_id": None,
                "last_task_accuracies": {},
                "average_accuracies": {}}

        # Sort task IDs and get the last one
        task_ids = sorted(all_task_results.keys())
        last_task_id = task_ids[-1]

        # Collect all variant names across tasks (robust to dynamic variant sets)
        variant_names = set()
        for task_dict in all_task_results.values():
            variant_names.update(task_dict.keys())
        variant_names = sorted(variant_names)

        # Compute final-task accuracies
        last_task_accuracies = {
            variant: all_task_results[last_task_id].get(variant, 0.0)
            for variant in variant_names
        }

        # Compute average accuracies across all tasks
        average_accuracies = {}
        for variant in variant_names:
            accs = [all_task_results[task_id].get(variant, 0.0) for task_id in task_ids]
            average_accuracies[variant] = float(np.mean(accs))

        # === 📝 Log Results in Structured Format ===
        logging.info("📊 Incremental Learning Evaluation Analysis:")
        logging.info(f"   Last Task ID: {last_task_id}")

        logging.info("  ── Final Task Accuracy (%) ──")
        for variant in variant_names:
            logging.info(f"      {variant:<20} : {last_task_accuracies[variant]:.2f}%")

        logging.info("   ── Average Accuracy Across Tasks (%) ──")
        for variant in variant_names:
            logging.info(f"      {variant:<20} : {average_accuracies[variant]:.2f}%")

        # Optional: Identify best variants and log summary
        best_last = max(last_task_accuracies, key=last_task_accuracies.get)
        best_avg = max(average_accuracies, key=average_accuracies.get)

        if best_last == best_avg:
            summary = f"🏆 Variant '{best_last}' is best in both final task and average performance."
        else:
            summary = f"🥇 Best in Final Task: '{best_last}' | 📈 Best Average: '{best_avg}'"

        logging.info("   ── Summary ──")
        logging.info(f"      {summary}")

        # Return structured data for further use
        return {
            "last_task_id": last_task_id,
            "last_task_accuracies": last_task_accuracies,
            "average_accuracies": average_accuracies}
