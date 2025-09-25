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
from utils.inc_net import CLIP_BaseNet
import math

import clip_datasets
from clip_datasets.common import get_dataloader
from clip_datasets.imagenet1k import ImageNet1K
from models.subspace_lora import feature_distillation_loss, symmetric_cross_entropy_loss, EMASmooth
from lora import FeatureCovarianceCalculator

@dataclass
class Timing:
    train: float = 0.0
    drift: float = 0.0
    total: float = 0.0

def compute_covariances(model, data_loader, device='cuda'):
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

class SubspaceLoRA_CLIP(BaseLearner):
    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = CLIP_BaseNet(args, train_mode="lora").to(self.device)
        self.prev_network = CLIP_BaseNet(args, train_mode="frozen").to(self.device)

        self.args = args

        if hasattr(torch, "compile"):
            try:
                self.network = torch.compile(self.network)
                print("Compiled network with torch.compile")
            except Exception as e:
                logging.warning(f"torch.compile failed: {e}")

        self.dataset_names = ["Aircraft", "Caltech101", "DTD", "EuroSAT", "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]
        
        self._timings: Timing = Timing()
        self.time_history: List[Dict[str, float]] = []
        self.batch_size: int = args["batch_size"]
        self.lrate: float = args["lrate"]
        self.weight_decay: float = args["weight_decay"]
        self.optimizer_type: str = args["optimizer"]
        self.data_location: str = args["data_location"]
        self.warmup_steps: int = args["warmup_steps"]
        self.iterations: int = args["iterations"]

        self.kd_loss_fn = feature_distillation_loss

        self.gamma_kd: float = args["gamma_kd"]
        self.use_feature_kd: bool = self.gamma_kd > 0.0
        self.gamma_norm: float = args["gamma_norm"]
        self.gamma_prior: float = args["kl_gamma"]

        self.l2_protection: bool = args["l2_protection"]
        self.l2_lambda: float = args["l2_protection_lambda"]

        self.covariances: Dict[str, torch.Tensor] | None = None
        self.prev_params: Dict[str, torch.Tensor] | None = None

        # === 新增：权重融合超参数 ===
        self.weight_interpolation_alpha: float = args.get("weight_interpolation_alpha", 0.5)  # 0.0 = no fusion, 0.5 = average
        self.model_snapshot: Dict[str, torch.Tensor] | None = None  # 用于保存训练前的完整模型状态

        self.seed: int = args["seed"] if "seed" in args else 1990
        self.task_count: int = 0

        self.log_interval: int = args.get("log_interval", 10)
        self.ema_alpha: float = args.get("ema_alpha", 0.90)

        self.monitor_ema = {
            'input_feature_positive_cosine': EMASmooth(alpha=self.ema_alpha),
            'input_feature_negative_cosine': EMASmooth(alpha=self.ema_alpha),

            'ref_feature_l2': EMASmooth(alpha=self.ema_alpha),
            'ref_feature_cosine': EMASmooth(alpha=self.ema_alpha),
            'ref_raw_kl': EMASmooth(alpha=self.ema_alpha),

            'teacher_ref_probs_min': EMASmooth(alpha=self.ema_alpha),
            'teacher_ref_probs_max': EMASmooth(alpha=self.ema_alpha),
            'student_ref_probs_min': EMASmooth(alpha=self.ema_alpha),
            'student_ref_probs_max': EMASmooth(alpha=self.ema_alpha),
            
            'ema_acc': EMASmooth(alpha=self.ema_alpha)}

    @torch.no_grad()
    def zeroshot_classifier(self, classnames, templates, model):
        if not isinstance(templates, list):
            templates = [templates]
        zeroshot_weights = []
        for classname in classnames:
            texts = [template(classname) for template in templates]
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(dim=0, keepdim=True)
        print(zeroshot_weights.shape)
        return zeroshot_weights
    
    @torch.no_grad()
    def evaluate_zeroshot(self, dataset_name):
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
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                features = self.network.encode_image(inputs)

                features = features / features.norm(dim=-1, keepdim=True)
                logit_scale = self.network.model.logit_scale
                logits_per_image = logit_scale.exp() * features @ zeroshot_weights
                
                preds = logits_per_image.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy

    def save_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        param_dict = {
            n: p.detach().cpu() for n, p in self.network.named_parameters() if p.requires_grad}
        
        payload = {"task": self._cur_task, "model_state_dict": param_dict}
        path = f"{prefix}_after_task_{self._cur_task}.pth"
        torch.save(payload, path)
        print(f"Checkpoint saved to: {path}")

    def store_model_snapshot(self) -> None:
        """Save a full snapshot of the current model state before training."""
        self.model_snapshot = copy.deepcopy(self.network.state_dict())
        print(f"✅ Model snapshot saved before task {self._cur_task + 1}")

    def after_task(self) -> None:
        """Update class counters after finishing a task."""
        self._known_classes = self._total_classes
        self.task_count += 1

    def incremental_train(self, train_loader, zeroshot_weights, reference_loader):
        """Entry‑point for training on a new task with optional weight interpolation."""
        start_time = time.time()
        self._cur_task += 1

        self.store_prev_params()
        self.system_training(train_loader, zeroshot_weights, reference_loader)

        self._timings.total = time.time() - start_time
        print(f"Task {self._cur_task} finished – total: {self._timings.total:.2f} s | train: {self._timings.train:.2f} s | drift: {self._timings.drift:.2f} s")

    
    def make_optimizer(
        self,
        params) -> optim.Optimizer:

        if self.optimizer_type == "sgd":
            optimizer = optim.SGD(params, lr=self.lrate, momentum=0.9)
        elif self.optimizer_type == "adamw":
            optimizer = optim.AdamW(params, lr=self.lrate)
        elif self.optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(params, lr=self.lrate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

        if self.warmup_steps > 0:
            eta_min = getattr(self, 'lora_eta_min', 1e-7)
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / max(1, self.warmup_steps)
                else:
                    progress = (step - self.warmup_steps) / max(1, self.iterations - self.warmup_steps)
                    initial_lr = self.lrate
                    lr_ratio = eta_min / initial_lr
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return lr_ratio + cosine_decay * (1.0 - lr_ratio)
            lr_lambdas = [lr_lambda]
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas, last_epoch=-1)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=eta_min)


        return optimizer, scheduler

    def store_prev_params(self) -> None:
        """Snapshot of trainable weights (used for L2‑protection)."""
        if not self.l2_protection:
            self.prev_params = None
            return

        # Only store the *trainable* part (skip classifier)
        self.prev_params = {
            name: p.clone().detach()
            for name, p in self.network.named_parameters()
            if p.requires_grad and "fc" not in name}

    def l2_protection_loss(self) -> torch.Tensor:
        """L2‑penalty that keeps current weights close to the snapshot."""
        if not self.l2_protection or self.prev_params is None:
            return torch.tensor(0.0, device=self.device)

        loss = 0.0
        for name, param in self.network.named_parameters():
            if not param.requires_grad or name.startswith("fc"):
                continue
            old = self.prev_params.get(name)
            if old is None:
                continue
            loss = loss + ((param - old.to(self.device)) ** 2).sum()
        return self.l2_lambda * loss


    def system_training(self, train_loader, zeroshot_weights, reference_loader) -> None:
        params = []
        param_names = []
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                params.append(p)
                param_names.append(n)

        optimizer, scheduler = self.make_optimizer(params)

        start = time.time()
        train_iter = iter(train_loader)
        self.reference_iter = iter(reference_loader)

        for iteration in range(1, self.iterations + 1):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)

            self.train_one_iteration(
                inputs, targets, zeroshot_weights, reference_loader, optimizer, iteration)

            cur_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else self.lrate
            
            if iteration == 0 or iteration % self.log_interval == 0:
                ema = self.monitor_ema
                print(f"\n=== [Task {self._cur_task} Iter {iteration}/{self.iterations}] EMA Metrics ===")
                print(f"  LR: {cur_lr:.6g} | Acc: {ema['ema_acc']:.4f}")
                print(f"  Input Postive cos: {ema['input_feature_positive_cosine'].get():.6f}' | Input Negative cos: {ema['input_feature_negative_cosine'].get():.6f}")
                print(f"  Ref Feature L2: {ema['ref_feature_l2'].get():.6f} | Ref Feature Cos: {ema['ref_feature_cosine'].get():.6f}")
                print(f"  Ref KL: {ema['ref_raw_kl'].get():.6f}")
                print(f"  Teacher Probs: [{ema['teacher_ref_probs_min'].get():.4f}, {ema['teacher_ref_probs_max'].get():.4f}]")
                print(f"  Student Probs: [{ema['student_ref_probs_min'].get():.4f}, {ema['student_ref_probs_max'].get():.4f}]")
                print("-" * 80)

            scheduler.step()

        self._timings.train = time.time() - start

    def train_one_iteration(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        zeroshot_weights: torch.Tensor,
        reference_loader: DataLoader,
        optimizer: optim.Optimizer,
        iteration: int) -> Tuple[float, float, float, float]:

        try:
            reference_batchs = next(self.reference_iter)
        except StopIteration:
            self.reference_iter = iter(reference_loader)
            reference_batchs = next(self.reference_iter)

        loss, n_corr, kd_raw, prior_raw = self.process_batch(
            inputs, targets, zeroshot_weights, reference_batchs, optimizer)

        bs = inputs.size(0)
        accuracy = n_corr / bs if bs > 0 else 0.0
        self.monitor_ema['ema_acc'].update(accuracy)

        return 

    def process_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        zeroshot_weights: torch.Tensor,
        reference_batch: dict,
        optimizer: optim.Optimizer) -> Tuple[float, int, float, float]:

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        zeroshot_weights = zeroshot_weights.to(self.device)

        reference_inputs = reference_batch["images"]
        reference_inputs = reference_inputs.to(self.device)

        combined_inputs = torch.cat([inputs, reference_inputs], dim=0)
        combined_img_feats = self.network.encode_image(combined_inputs)

        input_img_feats = combined_img_feats[:inputs.size(0)]
        reference_img_feats = combined_img_feats[inputs.size(0):]

        input_img_feats = input_img_feats / input_img_feats.norm(dim=-1, keepdim=True)
        reference_img_feats = reference_img_feats / reference_img_feats.norm(dim=-1, keepdim=True)
        
        with torch.no_grad():
            logit_scale = self.network.model.logit_scale
            
            reference_img_feats_prev = self.prev_network.encode_image(reference_inputs)
            reference_img_feats_prev = reference_img_feats_prev / reference_img_feats_prev.norm(dim=-1, keepdim=True)

            ref_labels = reference_batch["labels"]
            ref_indices = self.reference_text_labels[ref_labels].to(self.device)
            reference_text_feats = self.reference_text_embeddings[ref_indices]

            ref_feature_l2_dist = F.mse_loss(reference_img_feats, reference_img_feats_prev)
            ref_feature_cosine_sim = F.cosine_similarity(reference_img_feats, reference_img_feats_prev).mean()

            teacher_logits_ref = logit_scale.exp() * (reference_img_feats_prev @ reference_text_feats.T)
            student_logits_ref = logit_scale.exp() * (reference_img_feats @ reference_text_feats.T)

            prob_teacher_ref = F.softmax(teacher_logits_ref, dim=-1)
            prob_student_ref = F.softmax(student_logits_ref, dim=-1)

            t = 5.0
                                    
            teacher_probs = F.softmax(teacher_logits_ref / t, dim=-1).detach()
            student_log_probs = F.log_softmax(student_logits_ref / t, dim=-1)
            ref_raw_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (t * t)

            teacher_ref_probs_min = prob_teacher_ref.min().item()
            teacher_ref_probs_max = prob_teacher_ref.max().item()
            student_ref_probs_min = prob_student_ref.min().item()
            student_ref_probs_max = prob_student_ref.max().item()

        # ========== 损失计算 ==========
        # 交叉熵损失（当前任务数据）
        similarity_per_image = input_img_feats @ zeroshot_weights
        logits_per_image = logit_scale.exp() * similarity_per_image

        # ce_loss = F.cross_entropy(logits_per_image, targets, label_smoothing=0.2)

        # ========== 损失计算 ==========
        # 交叉熵损失（当前任务数据）—— 修改版：增强正例对齐
        similarity_per_image = input_img_feats @ zeroshot_weights
        logits_per_image = logit_scale.exp() * similarity_per_image

        # --- 新增：正例 logit 增强 ---
        delta = 1.0  # 可作为超参 self.positive_emphasis_delta
        logits_modified = logits_per_image.clone()
        batch_size = logits_per_image.size(0)
        row_indices = torch.arange(batch_size, device=logits_per_image.device)
        logits_modified[row_indices, targets] += delta

        ce_loss = F.cross_entropy(logits_modified, targets, label_smoothing=0.1)

        # ce_loss = symmetric_cross_entropy_loss(logits_per_image, targets)

        # 知识蒸馏损失（仅参考数据）
        kd_term = 1.0 * ref_feature_l2_dist + 2.0 * ref_raw_kl
        
        # 正则化项
        l2_term = self.l2_protection_loss()
        # print(l2_term.item())
        if self.network.train_mode == "lora":
            prior_term = self.network.model.vision_model.regularization_loss()
        else:
            prior_term = torch.tensor([0.0]).to(self.device)

        # 总损失
        loss = ce_loss + self.gamma_kd * kd_term + l2_term + prior_term 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits_per_image.argmax(dim=1)
            n_correct = (pred == targets).sum().item()

        kd_raw = kd_term.item() + l2_term.item()
        prior_raw = prior_term.item() if isinstance(prior_term, torch.Tensor) else float(prior_term)

        # ========== 更新 EMA ==========
        ema = self.monitor_ema

        batch_size = targets.size(0)
        row_indices = torch.arange(batch_size, device=targets.device)

        # 正类相似度：每个样本对应标签的相似度
        postive_cosine = similarity_per_image[row_indices, targets].mean().item()

        # 负类相似度：所有非目标类的平均相似度
        mask = torch.ones_like(similarity_per_image)
        mask[row_indices, targets] = 0
        nonpositive_cosine = (similarity_per_image * mask).sum() / mask.sum()
        nonpositive_cosine = nonpositive_cosine.item()

        ema['input_feature_positive_cosine'].update(postive_cosine)
        ema['input_feature_negative_cosine'].update(nonpositive_cosine)

        ema['ref_feature_l2'].update(ref_feature_l2_dist.item())
        ema['ref_feature_cosine'].update(ref_feature_cosine_sim.item())
        ema['ref_raw_kl'].update(ref_raw_kl.item())
        ema['teacher_ref_probs_min'].update(teacher_ref_probs_min)
        ema['teacher_ref_probs_max'].update(teacher_ref_probs_max)
        ema['student_ref_probs_min'].update(student_ref_probs_min)
        ema['student_ref_probs_max'].update(student_ref_probs_max)

        return loss.item(), n_correct, kd_raw, prior_raw
        
    @staticmethod
    def norm_loss(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """MSE between L2‑norms of teacher / student feature vectors."""
        t_norm = t_feat.norm(p=2, dim=1)
        s_norm = s_feat.norm(p=2, dim=1)
        return F.mse_loss(t_norm, s_norm)

    def update_projection_matrices(self):
        """Update OSGP projection matrices using the current training data."""
        if self._cur_task >= 0:
            new_covs = compute_covariances(self.network.model.vision_model, self.train_loader_test_mode)

            if self.covariances is None:
                self.covariances = new_covs
            else:
                for k in self.covariances:
                    self.covariances[k] = 0.9 * self.covariances[k] + new_covs[k]

            self.network.model.vision_model.update_projection_matrices(new_covs)

    def loop(self) -> Dict[str, List[float | None]]:

        reference_dataset = ImageNet1K(self.network.valid_preprocess)
        self.reference_loader = DataLoader(reference_dataset, batch_size=self.args['reference_batch_size'], shuffle=True, num_workers=4)
        self.reference_iter = iter(self.reference_loader)

        print("Precomputing reference text embeddings...")
        with torch.no_grad():
            unique_ref_labels, unique_ref_prompts = reference_dataset.return_labels_and_prompts()
            text_features = self.prev_network.encode_text(unique_ref_prompts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.reference_text_embeddings = text_features
            self.reference_text_labels = unique_ref_labels

        print(f"Precomputed {len(self.reference_text_embeddings)} reference text embeddings.")

        for i, dataset_name in enumerate(self.dataset_names):
            print(f"Starting task {i+1}/{len(self.dataset_names)}: {dataset_name}")
            dataset_class = getattr(clip_datasets, dataset_name)
            dataset = dataset_class(
                self.network.valid_preprocess,
                location=self.args['data_location'],
                batch_size=self.batch_size)
            
            zeroshot_weights = self.zeroshot_classifier(dataset.classnames, dataset.templates, self.network)
            train_loader = get_dataloader(dataset, is_train=True)
            
            self.incremental_train(train_loader, zeroshot_weights, self.reference_loader)
            
            print(f"Evaluating zeroshot performance after task {i+1}")
            zeroshot_results = {}
            for j in range(i + 1):
                test_dataset_name = self.dataset_names[j]
                accuracy = self.evaluate_zeroshot(test_dataset_name)
                zeroshot_results[test_dataset_name] = accuracy
                print(f"  {test_dataset_name}: {accuracy:.2f}%")
            
            avg_zeroshot = sum(zeroshot_results.values()) / len(zeroshot_results) if zeroshot_results else 0
            
            # ✅ 记录每个任务的平均零样本准确率（只记录一次，任务级别）
            self.history["zeroshot_acc"].append(avg_zeroshot)

            print(f"Average zeroshot accuracy after task {i+1}: {avg_zeroshot:.2f}%")
            
            self.after_task()
            
            if self.args.get("save_checkpoints", False):
                self.save_checkpoint(f"checkpoint_task_{i+1}")
        
        return

    def get_training_history(self) -> Dict[str, list]:
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
            # ✅ 使用 EMA 准确率，而不是原始准确率
            plt.plot(self.history["iteration"], self.history["ema_acc"], label='EMA Acc', linewidth=2, color='blue')
            # 可选：原始准确率淡色显示
            # plt.plot(self.history["iteration"], self.history["train_acc"], label='Raw Acc', alpha=0.3, color='gray')
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
            # 按任务分组显示零样本准确率
            plt.plot(range(1, len(self.history["zeroshot_acc"]) + 1), self.history["zeroshot_acc"], 'o-', linewidth=2, markersize=6)
            plt.xlabel("Task")
            plt.ylabel("Zeroshot Accuracy (%)")
            plt.title("Zeroshot Accuracy per Task")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig("training_progress.png")
            plt.close()
            
        except ImportError:
            logging.warning("Matplotlib not available, skipping plotting")