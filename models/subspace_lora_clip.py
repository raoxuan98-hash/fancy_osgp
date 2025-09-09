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
from utils.inc_net import CLIP_BaseNet
from lora import compute_covariances
from collections import namedtuple

import clip_datasets
from clip_datasets.common import get_dataloader
from models.subspace_lora import symmetric_cross_entropy_loss, feature_distillation_loss
import clip
from torchvision import datasets, transforms

EvalResult = namedtuple("EvalResult", ["original_fc", "linear_fc"])


@dataclass
class Timing:
    train: float = 0.0
    drift: float = 0.0
    total: float = 0.0


class CLIP_SubspaceLoRA(BaseLearner):
    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = CLIP_BaseNet(args).to(self._device)
        
        self.args = args

        # Optional JIT compilation (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                self.network = torch.compile(self.network)
                print("Compiled network with torch.compile")
            except Exception as e:
                logging.warning(f"torch.compile failed: {e}")

        self.dataset_names = ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]
        
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
        self.iterations_per_task: int = args.get("iterations_per_task", 1000)  # 每个任务的迭代次数
        self.lrate: float = args["lrate"]
        self.weight_decay: float = args["weight_decay"]
        self.optimizer_type: str = args["optimizer"]
        self.head_scale: float = args["head_scale"]
        self.osgp_scale: float = args["osgp_scale"]
        self.compensate: bool = args["compensate"]
        self.data_location: str = args["data_location"]

        self.kd_loss_fn = feature_distillation_loss

        self.gamma_kd: float = args["gamma_kd"]
        self.use_feature_kd: bool = self.gamma_kd > 0.0
        self.gamma_norm: float = args["gamma_norm"]
        self.gamma_prior: float = args["kl_gamma"]

        self.l2_protection: bool = args["l2_protection"]
        self.l2_lambda: float = args["l2_protection_lambda"]

        self.covariances: Dict[str, torch.Tensor] | None = None

        self.drift_compensator = Drift_Compensator(args)

        self.prev_params: Dict[str, torch.Tensor] | None = None
        self.prev_network = None

        self.original_fc: nn.Module | None = None
        self.linear_fc: nn.Module | None = None

        self.seed: int = args["seed"] if "seed" in args else 1990
        self.task_count: int = 0

        self._eval_tasks: set[int] = set()

        self.log_interval: int = args.get("log_interval", 10)  # 每多少个 iteration 打印一次
        self.ema_alpha: float = args.get("ema_alpha", 0.9)  # EMA平滑系数
        
        self.history: Dict[str, list] = {
            "task": [],
            "iteration": [],
            "train_loss": [],
            "train_acc": [],
            "ema_acc": [],  # 添加EMA准确率记录
            "avg_kd": [],
            "avg_prior": [],
            "lr": [],
            "zeroshot_acc": [],
        }

    @torch.no_grad()
    def zeroshot_classifier(self, classnames, templates, model):
        if not isinstance(templates, list):
            templates = [templates]
        zeroshot_weights = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        print(zeroshot_weights.shape)
        return zeroshot_weights
    
    @torch.no_grad()
    def evaluate_zeroshot(self, dataset_name):
        """评估零样本学习能力"""
        dataset_class = getattr(clip_datasets, dataset_name)
        dataset = dataset_class(
            self.network.valid_preprocess,
            location=self.args['data_location'],
            batch_size=self.batch_size)
        
        test_loader = get_dataloader(dataset, is_train=False)
        zeroshot_weights = self.zeroshot_classifier(dataset.classnames, dataset.templates, self.network)

        self.network.eval()
        total = 0
        correct = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                features = self.network.encode_image(inputs)

                features = features / features.norm(dim=-1, keepdim=True)
                
                logit_scale = self.network.model.clip.logit_scale
                logits_per_image = logit_scale.exp() * features @ zeroshot_weights
                
                preds = logits_per_image.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy

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
        print(f"Checkpoint saved to: {path}")

    def _compute_eval_milestones(self, nb_tasks: int) -> set[int]:
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
        print(f"Refining classifiers at milestone task {self._cur_task} / {self.data_manager.nb_tasks}")


    # ------------------------------------------------------------------
    #  Task‑level bookkeeping
    # ------------------------------------------------------------------
    def after_task(self) -> None:
        """Update class counters after finishing a task."""
        self._known_classes = self._total_classes
        # self.update_projection_matrices()
        self.task_count += 1

    # ------------------------------------------------------------------
    #  Incremental training driver (public API)
    # ------------------------------------------------------------------
    def incremental_train(self, train_loader, zeroshot_weights, reference_loader):
        """Entry‑point for training on a new task."""
        start_time = time.time()
        self._cur_task += 1

        self.prev_network = copy.deepcopy(self.network).to(self._device)
        self.prev_network.model.finalize_without_lora()

        self._store_prev_params()
        self._system_training(train_loader, zeroshot_weights, reference_loader)

        # ---- 6️⃣  drift compensation -----------------------------------------
        # self.handle_drift_compensation()

        # ---- 7️⃣  timing ----------------------------------------------------
        self._timings.total = time.time() - start_time

        print("Task %d finished – total: %.2f s | train: %.2f s | drift: %.2f s", self._cur_task, self._timings.total, self._timings.train, self._timings.drift)

    
    # ------------------------------------------------------------------
    #  Optimiser factory
    # ------------------------------------------------------------------
    def _make_optimizer(
        self,
        lora_params: List[torch.nn.Parameter],
        osgp_params: List[torch.nn.Parameter] | None = None,
    ) -> optim.Optimizer:
        """Create optimizer according to ``self.optimizer_type``."""
        if osgp_params is None:
            param_groups = [
                {
                    "params": lora_params,
                    "lr": self.lrate,
                    "weight_decay": self.weight_decay,
                }
            ]
        else:
            param_groups = [
                {
                    "params": lora_params,
                    "lr": self.lrate,
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
    def _system_training(self, train_loader, zeroshot_weights, reference_loader) -> None:
        """Train ViT + new classifier head for ``self.iterations_per_task`` iterations."""
        if self.network.model.optimizable:
            lora_params, osgp_params = self.network.model.collect_vit_and_delta_params()
        else:
            osgp_params = None
            lora_params = [p for p in self.network.model.parameters() if p.requires_grad]

        optimizer = self._make_optimizer(lora_params, osgp_params)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations_per_task, eta_min=self.lrate / 2)

        start = time.time()

        # 创建无限数据迭代器
        train_iter = iter(train_loader)
        
        # 初始化EMA准确率
        ema_acc = 0.0
        
        for iteration in range(1, self.iterations_per_task + 1):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)
                
            avg_loss, acc, avg_kd, avg_prior = self._train_one_iteration(
                inputs, targets, zeroshot_weights, reference_loader, optimizer, iteration)
            
            # 更新EMA准确率
            if iteration == 1:
                ema_acc = acc
            else:
                ema_acc = self.ema_alpha * ema_acc + (1 - self.ema_alpha) * acc
            
            # 记录和打印 LR
            if len(optimizer.param_groups) > 0:
                cur_lr = optimizer.param_groups[0]["lr"]
            else:
                cur_lr = self.lrate

            if iteration % self.log_interval == 0:
                print(f"[Task {self._cur_task} Iter {iteration}/{self.iterations_per_task}] lr={cur_lr:.6g} | loss={avg_loss:.6f} | acc={acc:.4f} | ema_acc={ema_acc:.4f} | kd={avg_kd:.6f} | prior={avg_prior:.6f}")

            # 保存到历史
            self.history["task"].append(self._cur_task)
            self.history["iteration"].append(iteration)
            self.history["train_loss"].append(float(avg_loss))
            self.history["train_acc"].append(float(acc))
            self.history["ema_acc"].append(float(ema_acc))  # 记录EMA准确率
            self.history["avg_kd"].append(float(avg_kd))
            self.history["avg_prior"].append(float(avg_prior))
            self.history["lr"].append(float(cur_lr))

            scheduler.step()

        self._timings.train = time.time() - start

    def _train_one_iteration(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        zeroshot_weights: torch.Tensor,
        reference_loader: DataLoader,
        optimizer: optim.Optimizer,
        iteration: int
    ) -> Tuple[float, float, float, float]:
        """
        处理单个iteration的训练
        返回：loss, accuracy, kd_loss, prior_loss
        """
        # 获取reference batch
        try:
            reference_inputs, _ = next(self.reference_iter)
        except StopIteration:
            self.reference_iter = iter(reference_loader)
            reference_inputs, _ = next(self.reference_iter)

        loss, n_corr, kd_raw, prior_raw = self._process_batch(
            inputs, targets, zeroshot_weights, reference_inputs, optimizer)

        bs = inputs.size(0)
        accuracy = n_corr / bs if bs > 0 else 0.0

        return loss, accuracy, kd_raw, prior_raw

    # ------------------------------------------------------------------
    #  Process single batch (forward → loss → backward → step)
    # ------------------------------------------------------------------
    def _process_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        zeroshot_weights: torch.Tensor,
        reference_inputs: torch.Tensor,
        optimizer: optim.Optimizer) -> Tuple[float, int, float, float]:

        inputs, targets = inputs.to(self._device), targets.to(self._device)
        zeroshot_weights = zeroshot_weights.to(self._device)
        reference_inputs = reference_inputs.to(self._device)

        total_inputs = torch.cat([inputs, reference_inputs], dim=0)
        total_feats = self.network.encode_image(total_inputs)
        total_feats = total_feats / total_feats.norm(dim=-1, keepdim=True)

        feats = total_feats[:inputs.shape[0]]


        with torch.no_grad():
            logit_scale = self.network.model.clip.logit_scale
            total_feats_prev = self.prev_network.encode_image(total_inputs)
            total_feats_prev = total_feats_prev / total_feats_prev.norm(dim=-1, keepdim=True)
            # 选 1：用“同一个温度”做教师分布（常见也可用教师自身的 logit_scale；两者择一即可，保持一致即可）
            # logit_scale_t = self.prev_network.model.clip.logit_scale.exp()
            # teacher_logits = logit_scale_t * (total_feats_prev @ zeroshot_weights)
            teacher_logits = logit_scale * (total_feats_prev @ zeroshot_weights)
            prob_teacher = F.softmax(teacher_logits, dim=-1)  # 确保不反传

        kd_feat = self.kd_loss_fn(total_feats, total_feats_prev)
        kd_norm = self._norm_loss(total_feats, total_feats_prev)
        kd_term = self.gamma_kd * kd_feat + self.gamma_norm * kd_norm

        logits_per_image = logit_scale.exp() * feats @ zeroshot_weights

        prob = F.softmax(total_feats @ zeroshot_weights, dim=-1)

        kl_loss = F.kl_div(
            prob.log(),          # log(p_student)
            prob_teacher,        # p_teacher
            reduction="batchmean")

        sce_loss = symmetric_cross_entropy_loss(logits_per_image, targets)

        l2_term = self._l2_protection_loss() + 0.5 * kl_loss
        prior_term = self.network.model.kl_regularization()

        loss = sce_loss + kd_term + l2_term + prior_term 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------------------------------------------------------------
        #  Statistics
        # ------------------------------------------------------------------
        with torch.no_grad():
            pred = logits_per_image.argmax(dim=1)
            n_correct = (pred == targets).sum().item()

        # Return scalar loss + raw KD / prior contributions (for logging)
        kd_raw = (kd_term.item() / self.gamma_kd if isinstance(kd_term, torch.Tensor) and self.gamma_kd != 0 else float(kd_term))
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
        linear_fc: nn.Module | None = None) -> EvalResult:
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

        print(
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
            else:
                for k in self.covariances:
                    self.covariances[k] = 0.9 * self.covariances[k] + new_covs[k]

            self.network.vit.update_projection_matrices(new_covs)

    def loop(self) -> Dict[str, List[float | None]]:
        """主循环，按任务顺序训练并评估"""
        agg: Dict[str, List[float | None]] = {
            "original_fc": [],
            "linear_fc": [],
            "zeroshot_acc": [] }
        
        # 初始化参考数据集
        reference_dataset = datasets.ImageFolder(self.args['reference_dataset_path'], transform=self.network.valid_preprocess)
        self.reference_loader = DataLoader(reference_dataset, batch_size=self.args['reference_batch_size'], shuffle=True, num_workers=4)
        self.reference_iter = iter(self.reference_loader)

        for i, dataset_name in enumerate(self.dataset_names):
            print(f"Starting task {i+1}/{len(self.dataset_names)}: {dataset_name}")
            
            # 准备数据集
            dataset_class = getattr(clip_datasets, dataset_name)
            dataset = dataset_class(
                self.network.valid_preprocess,
                location=self.args['data_location'],
                batch_size=self.batch_size)
            
            # 获取零样本权重
            zeroshot_weights = self.zeroshot_classifier(dataset.classnames, dataset.templates, self.network)
            train_loader = get_dataloader(dataset, is_train=True)
            
            # 训练当前任务
            self.incremental_train(train_loader, zeroshot_weights, self.reference_loader)
            
            # 评估零样本学习能力（在所有已见数据集上）
            print(f"Evaluating zeroshot performance after task {i+1}")
            zeroshot_results = {}
            for j in range(i + 1):  # 在所有已学习的任务上评估
                test_dataset_name = self.dataset_names[j]
                accuracy = self.evaluate_zeroshot(test_dataset_name)
                zeroshot_results[test_dataset_name] = accuracy
                print(f"  {test_dataset_name}: {accuracy:.2f}%")
            
            # 记录平均零样本准确率
            avg_zeroshot = sum(zeroshot_results.values()) / len(zeroshot_results) if zeroshot_results else 0
            self.history["zeroshot_acc"].append(avg_zeroshot)
            agg["zeroshot_acc"].append(avg_zeroshot)
            
            print(f"Average zeroshot accuracy after task {i+1}: {avg_zeroshot:.2f}%")
            
            # 任务结束后处理
            self.after_task()
            
            # 保存检查点
            if self.args.get("save_checkpoints", False):
                self.save_checkpoint(f"checkpoint_task_{i+1}")
        
        return agg

    def get_training_history(self) -> Dict[str, list]:
        """获取训练历史记录"""
        return self.history

    def plot_training_progress(self):
        """绘制训练进度图（可选）"""
        try:
            import matplotlib.pyplot as plt
            
            # 绘制训练损失
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(self.history["iteration"], self.history["train_loss"])
            plt.xlabel("Iteration")
            plt.ylabel("Training Loss")
            plt.title("Training Loss over Iterations")
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(self.history["iteration"], self.history["train_acc"], label='Raw Acc')
            plt.plot(self.history["iteration"], self.history["ema_acc"], label='EMA Acc', linewidth=2)
            plt.xlabel("Iteration")
            plt.ylabel("Training Accuracy")
            plt.title("Training Accuracy over Iterations")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(self.history["iteration"], self.history["lr"])
            plt.xlabel("Iteration")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            # 按任务分组显示零样本准确率
            task_indices = []
            task_accuracies = []
            for i, task_num in enumerate(self.history["task"]):
                if i == 0 or task_num != self.history["task"][i-1]:
                    task_indices.append(i)
                    task_accuracies.append(self.history["zeroshot_acc"][i] if i < len(self.history["zeroshot_acc"]) else 0)
            
            plt.plot(task_indices, task_accuracies, 'o-')
            plt.xlabel("Task")
            plt.ylabel("Zeroshot Accuracy")
            plt.title("Zeroshot Accuracy per Task")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig("training_progress.png")
            plt.close()
            
        except ImportError:
            logging.warning("Matplotlib not available, skipping plotting")

