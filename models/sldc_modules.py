import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import copy
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset

def cholesky_manual_parallel(matrix):
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    for j in range(n):
        s_diag = torch.sum(L[j, :j] ** 2, dim=0)
        diag = matrix[j, j] - s_diag
        L[j, j] = torch.sqrt(torch.clamp(diag, min=1e-8))
        if j < n - 1:
            s_off = torch.mm(L[j+1:, :j], L[j, :j].unsqueeze(1)).squeeze(1)
            L[j+1:, j] = (matrix[j+1:, j] - s_off) / L[j, j]
    return L

def sample_torch_cholesky(n_samples, mean, L, given_Z=None):
    if given_Z is not None:
        random_indices = torch.randperm(given_Z.shape[0] )
        Z = given_Z[random_indices][0:n_samples].to(mean.device)
    else:
        Z = torch.randn(n_samples, mean.size(0), device=mean.device)
    X = Z @ L.T + mean.unsqueeze(0)
    return X

def symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5):
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0) 
    label_one_hot = torch.nn.functional.one_hot(targets, pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    ce_loss = -torch.sum(label_one_hot * torch.log(pred), dim=1).mean()
    rce_loss = -torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
    total_loss = sce_a * ce_loss + sce_b * rce_loss
    return total_loss

class GaussianStatistics:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.L = cholesky_manual_parallel(cov + 1e-3 * torch.eye(cov.size(0), device=cov.device))
        
        self.mean = self.mean.cpu()
        self.cov = self.cov.cpu()
        self.L = self.L.cpu()
        
    def kl_divergence(self, other, eps=1e-6):
        """计算KL散度"""
        d = self.mean.size(0)
        cov2_inv = torch.linalg.inv(other.cov + eps * torch.eye(d, device=other.cov.device))
    
        diff = self.mean - other.mean
        kl = 0.5 * (
            torch.logdet(other.cov + eps * torch.eye(d, device=other.cov.device)) -
            torch.logdet(self.cov + eps * torch.eye(d, device=self.cov.device)) +
            torch.trace(cov2_inv @ self.cov) + diff @ cov2_inv @ diff - d)
        return kl
    
class NonlinearCompensator(nn.Module):
    def __init__(self, dim):
        super(NonlinearCompensator, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.fc1.weight)

        self.fc2 = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=True))
        self.alphas =  nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
        self.weight = 0.0

    def forward(self, x):
        weights = F.softmax(self.alphas, dim=0)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y = weights[0]*x + weights[1] * y1 + weights[2] * y2
        return (1.0 - self.weight) * y + self.weight * x
    
    def reg_loss(self):
        weights = F.softmax(self.alphas, dim=0)
        return (weights[0] + weights[1] - 1.0) ** 2
    
class Drift_Compensator(object):
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.original_stats = {}
        self.linear_stats = {}

        self.alpha_t = args['alpha_t']
        self.gamma_1 = args['gamma_1']
        self.auxiliary_data_size = args['auxiliary_data_size']
        self.args = args

    def update_stats(self, task_id, model_before, model_after, data_loader, compensate=False):
        feats_before, feats_after, targets = self.extract_features_before_after(model_before, model_after, data_loader)

        if task_id == 0:
            try:
                self.cached_Z = torch.load('cached_data/cached_Gaussian_samples.pt', weights_only=False)
            except:
                self.cached_Z = torch.randn(50000, feats_before.size(1))
                torch.save(self.cached_Z, 'cached_data/cached_Gaussian_samples.pt', weights_only=False)

        original_stats = self.compute_class_statistics(feats_after, targets)

        assert set(original_stats.keys()).isdisjoint(set(self.original_stats.keys())), \
            "original_stats keys should not overlap with self.original_stats keys"
        
        self.original_stats.update(original_stats)

        if compensate:
            aux_loader = self.get_aux_loader(self.args)
            feats_aux_before, feats_aux_after = self.extract_features_before_after_for_auxiliary_data(
                model_before, model_after, aux_loader)
        
            feats_before_with_aux = torch.cat([feats_before, feats_aux_before], dim=0)
            feats_after_with_aux = torch.cat([feats_after, feats_aux_after], dim=0)
                
            if task_id > 0:
                self.linear_stats = self.update_statistics_with_linear_transform(
                    self.linear_stats, feats_before_with_aux, feats_after_with_aux)
                # self.linear_stats = self.update_statistics_with_weak_nonlinear_transform(
                #     self.linear_stats, feats_before_with_aux, feats_after_with_aux)
                    
        self.linear_stats.update(original_stats)

        
    @torch.no_grad()
    def extract_features_before_after(self, model_before, model_after, data_loader):
        """对比模型训练前后的特征变化"""
        model_before, model_after = model_before.to(self.device), model_after.to(self.device)
        model_before.eval(); model_after.eval()
        
        feats_before, feats_after, targets, indices = [], [], [], []
        
        for batch_indices, inputs, batch_targets in data_loader:
            inputs = inputs.to(self.device)
            feats_before.append(model_before(inputs).cpu())
            feats_after.append(model_after(inputs).cpu())
            targets.append(batch_targets)
            indices.append(batch_indices)

        feats_before = torch.cat(feats_before)
        feats_after = torch.cat(feats_after)
        targets = torch.cat(targets)
        indices = torch.cat(indices)
        return feats_before[indices], feats_after[indices], targets[indices]

    @torch.no_grad()
    def extract_features_before_after_for_auxiliary_data(self, model_before, model_after, data_loader):
        model_before, model_after = model_before.to(self.device), model_after.to(self.device)
        model_before.eval(); model_after.eval()
        
        feats_before, feats_after= [], []
        for inputs, batch_targets in data_loader:
            inputs = inputs.to(self.device)
            feats_before.append(model_before(inputs).cpu())
            feats_after.append(model_after(inputs).cpu())

        feats_before = torch.cat(feats_before)
        feats_after = torch.cat(feats_after)
        return feats_before, feats_after
    
    def compute_class_statistics(self, features, labels):
        unique_labels = torch.unique(labels)
        stats_dict = {}
        for lbl in unique_labels:
            mask = (labels == lbl)
            class_features = features[mask]
            mean = class_features.mean(dim=0)
            centered = class_features - mean
            cov = (centered.T @ centered) / (class_features.size(0) - 1)
            stats_dict[int(lbl.item())] = GaussianStatistics(mean, cov)
        return stats_dict
    
    def update_statistics_with_linear_transform(self, stats, features_before, features_after):
        print("基于当前任务的前后特征构建线性补偿器(alpha_1-SLDC)")
        features_before = features_before.to(self.device)
        features_after = features_after.to(self.device)
        X = F.normalize(features_before, dim=1)
        Y = F.normalize(features_after, dim=1)
        XTX = X.T @ X + self.gamma_1 * torch.eye(X.size(1), device=self.device)
        XTY = X.T @ Y
        W_global = torch.linalg.solve(XTX, XTY)

        dim = features_before.size(1)
        sample_num = features_before.size(0)

        weight = math.exp(- sample_num / (self.alpha_t*dim))
        print(weight)
        W_global = (1 - weight)  * W_global + weight * torch.eye(dim, device=self.device)
        feats_new_after_pred = features_before @ W_global
        feat_diffs = (features_after - features_before).norm(dim=1).mean().item()
        feat_diffs_pred = (features_after - feats_new_after_pred).norm(dim=1).mean().item()

        s = torch.linalg.svdvals(W_global)
        max_singular = s[0].item()
        min_singular = s[-1].item()
        print(f"线性变换矩阵对角线元素平均值：{W_global.diag().mean().item():.4f}, 加权权重：{weight:.4f}, 样本数量：{sample_num}")
        print(f"线性修正前特征差异:{feat_diffs:.4f}; 修正后差异:{feat_diffs_pred:.4f}; 变换矩阵最大奇异值:{max_singular:.2f}; 最小奇异值:{min_singular:.2f}")

        updated_stats = {}
        for class_id, gauss in stats.items():
            old_mean = gauss.mean.to(self.device)
            old_cov = gauss.cov.to(self.device)
            new_mean = old_mean @ W_global
            new_cov = W_global.T @ old_cov @ W_global + 1e-2 * torch.eye(old_cov.size(0), device=old_cov.device)
            updated_stats[class_id] = GaussianStatistics(new_mean, new_cov)
        return updated_stats

    def optimize_nonlinear_transform(self, features_before, features_after):
        features_after = F.normalize(features_after, dim=1)
        features_before = F.normalize(features_before, dim=1)
        model = NonlinearCompensator(features_after.size(1)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        steps = 5000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=5e-4)
        for step in range(steps):
            random_indices = torch.randint(0, features_before.shape[0], (32, ))
            features_before_batch = features_before[random_indices]
            features_after_batch = features_after[random_indices]
            optimizer.zero_grad()
            output = model(features_before_batch)
            loss = criterion(output, features_after_batch)
            loss_total = loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (step + 1) % 2500 == 0:
                print(f"弱非线性补偿器训练: Step {step + 1}, Loss_total: {loss_total.item():.4f}, Loss: {loss.item():.4f}")
        model.eval()
        return model, tuple(F.softmax(model.alphas, dim=0).detach().cpu().tolist())

    def update_statistics_with_weak_nonlinear_transform(self, stats, features_before, features_after):
        features_before = features_before.to(self.device)
        features_after = features_after.to(self.device)
        print("基于当前任务的前后特征构建弱非线性补偿器")
        mapping, alpha = self.optimize_nonlinear_transform(features_before, features_after)
        
        # 评估修正效果（可选）
        with torch.no_grad():
            feats_new_after_pred = mapping(F.normalize(features_before, dim=1)) * features_before.norm(dim=1, keepdim=True)
            feat_diffs_pred = (features_after - feats_new_after_pred).norm(dim=1).mean().item()
            print(f"弱非线性修正后特征差异: {feat_diffs_pred:.4f}, Alpha: {alpha}")
        
        # 使用雅可比矩阵近似更新高斯分布
        new_stats = {}
        for class_id, gauss in stats.items():
            class_mean = gauss.mean.to(self.device)
            class_cov = gauss.cov.to(self.device)
            
            # 计算雅可比矩阵
            J = self.compute_jacobian(mapping, class_mean)
            
            # 计算新均值
            with torch.no_grad():
                x_norm = F.normalize(class_mean.unsqueeze(0), dim=1)
                y_norm = mapping(x_norm)
                new_mean = y_norm.squeeze(0) * class_mean.norm()
            
            # 计算新协方差
            A = mapping.fc1.weight
            linear_part = alpha[1] * A  # 线性部分的权重
            nonlinear_part = alpha[2] * J  # MLP部分的雅可比近似
            total_transform = alpha[0] * torch.eye(class_mean.size(0), device=self.device) + linear_part + nonlinear_part
            
            # 应用变换：cov_new = J @ cov_old @ J^T
            new_cov = total_transform @ class_cov @ total_transform.T
            new_cov = new_cov + 1e-3 * torch.eye(new_cov.size(0), device=new_cov.device)  # 正则化
            
            new_stats[class_id] = GaussianStatistics(new_mean.cpu(), new_cov.cpu())
        
        return new_stats

    def compute_jacobian(self, model, x):
        """
        计算模型在输入x处的雅可比矩阵
        """
        x_norm = F.normalize(x.unsqueeze(0), dim=1).detach().requires_grad_(True)
        
        # 前向传播
        y = model(x_norm)
        
        # 计算雅可比矩阵
        J = torch.zeros(y.size(1), x_norm.size(1), device=self.device)
        for i in range(y.size(1)):
            grad_output = torch.zeros_like(y)
            grad_output[0, i] = 1.0
            grad_input = torch.autograd.grad(y, x_norm, grad_outputs=grad_output, 
                                            retain_graph=True, create_graph=False)[0]
            J[i] = grad_input.squeeze(0)
        
        return J.detach()

    def refine_classifiers(self, fc, task_id, epochs=5):
        if task_id > 0:
            # 不使用辅助数据的分类器
            original_fc = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.original_stats, epochs)
            linear_fc = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.linear_stats, epochs) 

        elif task_id == 0:
            # 初始任务，直接复制分类器
            original_fc = self.train_classifier_with_cached_samples(copy.deepcopy(fc), self.original_stats, epochs)
            linear_fc = copy.deepcopy(original_fc)
        
        return {'original': original_fc,
                'linear_compensate': linear_fc}

    def get_aux_loader(self, args):
        if hasattr(self, 'aux_loader'):
            return self.aux_loader
        
        aux_dataset_type = args.get('aux_dataset_type', 'image_folder')
        num_samples = args.get('auxiliary_data_size', 1024)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if aux_dataset_type == 'imagenet':
          if 'auxiliary_data_path' not in args:
            raise ValueError("当 aux_dataset_type='image_folder' 时，必须提供 auxiliary_data_path")
          dataset = datasets.ImageFolder(args['auxiliary_data_path'], transform=transform)
        elif aux_dataset_type == 'cifar10':
          dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        elif aux_dataset_type == 'svhn':
          dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        else:
          raise ValueError(f"不支持的 aux_dataset_type: {aux_dataset_type}")

        torch.manual_seed(1)
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        train_subset = Subset(dataset, indices)

        self.aux_loader = DataLoader(train_subset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        return self.aux_loader

    def train_classifier_with_cached_samples(self, fc, stats, epochs):
        epochs = 6
        num_samples_per_class = 1024
        batch_size = 32 * len(stats) // 10
        lr = 5e-4     
        fc.to(self.device)
        optimizer = torch.optim.Adam(fc.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=lr/10)
        
        cached_Z = self.cached_Z.to(self.device)

        # 准备数据 - 确保不原地修改
        all_samples, all_targets = [], []
        for class_id, gauss in stats.items():
            class_mean = gauss.mean.to(self.device)
            class_L = gauss.L.to(self.device)
            
            # 使用新的随机索引避免修改缓存
            start_idx = (class_id * num_samples_per_class) % cached_Z.size(0)
            end_idx = start_idx + num_samples_per_class
            if end_idx > cached_Z.size(0):
                Z = torch.cat([cached_Z[start_idx:], cached_Z[:end_idx-cached_Z.size(0)]], dim=0)
            else:
                Z = cached_Z[start_idx:end_idx]
            
            samples = class_mean + torch.mm(Z, class_L.t())
            targets = torch.full((num_samples_per_class,), class_id, device=self.device)
            
            all_samples.append(samples)
            all_targets.append(targets)

        # 拼接并克隆张量
        inputs = torch.cat(all_samples, dim=0).detach().clone()  # 确保不共享内存
        targets = torch.cat(all_targets, dim=0).detach().clone()

        # 训练循环
        for epoch in range(epochs):
            # 创建新的随机索引而不修改原数据
            perm = torch.randperm(inputs.size(0), device=self.device)
            inputs_shuffled = inputs[perm]
            targets_shuffled = targets[perm]

            losses = 0.0
            num_samples = inputs.size(0)
            num_complete_batches = num_samples // batch_size
            
            for batch_idx in range(num_complete_batches + 1):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                
                if start_idx >= end_idx:
                    continue
                    
                inp = inputs_shuffled[start_idx:end_idx]
                tgt = targets_shuffled[start_idx:end_idx]
                
                optimizer.zero_grad()
                output = fc(inp)
                loss = symmetric_cross_entropy_loss(output, tgt)
                
                # 启用异常检测(调试用)
                # with torch.autograd.detect_anomaly():
                loss.backward()
                optimizer.step()
                
                losses += loss.item() * (end_idx - start_idx)
            
            loss = losses / num_samples
            if (epoch + 1) % 3 == 0:
                print(f"分类器矫正训练 (cached samples): Epoch {epoch + 1}, Loss: {loss:.4f}")
            scheduler.step()
        
        return fc


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import copy
# from collections import defaultdict

# # --------------------------------------------------------------
# # 1️⃣ 统计量容器（支持增量更新 + 线性映射）
# # --------------------------------------------------------------
# class ClassStat:
#     """
#     保存一个类别在当前累计特征空间中的统计量。
#     只保存:
#         N  : 已累计样本数（标量）
#         mu : 均值               (d,)
#         C  : 未中心化二阶矩   (d, d)   = Σ_i x_i x_i^T
#     """
#     def __init__(self, device: torch.device):
#         self.device = device
#         self.N   = 0                         # int
#         self.mu  = None                      # (d,)
#         self.C   = None                      # (d, d)

#     def update(self, new_mu: torch.Tensor, new_cov: torch.Tensor, new_N: int):
#         """
#         参数:
#             new_mu   – (d,) 新任务均值（已经在 *当前* 特征空间里）
#             new_cov  – (d, d) 新任务协方差（已经在 *当前* 特征空间里）
#             new_N    – int   新任务样本数
#         """
#         # 先把新任务的未中心化二阶矩算出来
#         new_C = new_cov * (new_N - 1) + new_N * torch.ger(new_mu, new_mu)   # Σ_i x_i x_i^T

#         if self.N == 0:                 # 第一次收到该类的数据
#             self.N  = new_N
#             self.mu = new_mu.clone()
#             self.C  = new_C.clone()
#             return

#         # 累计公式
#         total_N = self.N + new_N
#         mu_new = (self.N * self.mu + new_N * new_mu) / total_N
#         C_new  = self.C + new_C

#         self.N  = total_N
#         self.mu = mu_new
#         self.C  = C_new

#     def apply_linear(self, W: torch.Tensor):
#         """把已经累计好的统计量映射到 W 的新空间（就地修改）。"""
#         if self.N == 0:
#             return
#         # μ' = μ @ W
#         self.mu = self.mu @ W
#         # C' = W.T @ C @ W   （未中心化矩阵的线性变换）
#         self.C = W.t() @ self.C @ W

#     def get_cov(self, eps: float = 1e-6) -> torch.Tensor:
#         """返回协方差 Σ = (C - N μ μ^T) / (N-1)."""
#         if self.N <= 1:
#             d = self.mu.shape[0]
#             return torch.eye(d, device=self.device) * eps
#         cov = (self.C - self.N * torch.ger(self.mu, self.mu)) / (self.N - 1)
#         # 为数值稳定性加一点对角线噪声
#         cov = cov + eps * torch.eye(cov.shape[0])
#         return cov

# # --------------------------------------------------------------
# # 2️⃣ Drift_Compensator（增量统计 + LDA）
# # --------------------------------------------------------------
# class Drift_Compensator:
#     def __init__(self, args):
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         # 每个任务结束后累计的类统计量 (dict[int, ClassStat])
#         self.class_stats = defaultdict(lambda: ClassStat(self.device))

#         self.alpha_t = args['alpha_t']
#         self.gamma_1 = args['gamma_1']
#         self.auxiliary_data_size = args['auxiliary_data_size']
#         self.args = args

#     # -----------------------------------------------------------------
#     # 2.1   统计量更新（包括线性补偿）
#     # -----------------------------------------------------------------
#     @torch.no_grad()
#     def update_stats(self, task_id, model_before, model_after, data_loader):
#         # ------------------- 1) 提取本任务特征 -------------------
#         feats_before, feats_after, targets = self.extract_features_before_after(
#             model_before, model_after, data_loader)

#         # ------------------- 2) 线性映射 (if task_id>0) -------------------
#         if task_id > 0:
#             # 这里返回的 W_global 同时用于对已有统计量进行映射
#             W_global = self._compute_linear_transform(feats_before, feats_after)
#             # 把之前累计的所有类统计量“迁移”到新的特征空间
#             for stat in self.class_stats.values():
#                 stat.apply_linear(W_global)

#         # ------------------- 3) 计算本任务的类统计（已在 final 空间） -------------------
#         # 下面的 `new_stats` 为 dict[int, (mean, cov, N)]
#         new_stats = {}
#         unique_labels = torch.unique(targets)
#         for lbl in unique_labels:
#             mask = (targets == lbl)
#             class_feats = feats_after[mask]                 # 已经是“after”特征
#             N = class_feats.size(0)
#             mu = class_feats.mean(dim=0)                    # (d,)
#             centered = class_feats - mu
#             cov = (centered.t() @ centered) / (N - 1)       # (d,d)
#             new_stats[int(lbl.item())] = (mu, cov, N)

#         # ------------------- 4) 增量合并 -------------------
#         for cid, (mu, cov, N) in new_stats.items():
#             self.class_stats[cid].update(mu, cov, N)

#     # -----------------------------------------------------------------
#     # 2.2   计算线性映射矩阵（与原来实现保持一致）
#     # -----------------------------------------------------------------
#     def _compute_linear_transform(self, X_before, X_after):
#         """返回 W_global ∈ ℝ^{d×d} 用于把 before 映射到 after。"""
#         X = X_before.to(self.device)
#         Y = X_after.to(self.device)
#         d = X.size(1)

#         XTX = X.t() @ X + self.gamma_1 * torch.eye(d, device=self.device)
#         XTY = X.t() @ Y
#         W = torch.linalg.solve(XTX, XTY)

#         # α_t‑weighting 讓矩陣更保守（原实现中的 weight ）
#         weight = math.exp(- X.size(0) / (self.alpha_t * d))
#         W = (1 - weight) * W + weight * torch.eye(d, device=self.device)
#         return W.cpu()

#     # -----------------------------------------------------------------
#     # 2.3   特征提取（保持原实现）
#     # -----------------------------------------------------------------
#     @torch.no_grad()
#     def extract_features_before_after(self, model_before, model_after, data_loader):
#         model_before, model_after = model_before.to(self.device), model_after.to(self.device)
#         model_before.eval(); model_after.eval()

#         feats_before, feats_after, targets, indices = [], [], [], []

#         for batch_indices, inputs, batch_targets in data_loader:
#             inputs = inputs.to(self.device)
#             feats_before.append(model_before(inputs).cpu())
#             feats_after.append(model_after(inputs).cpu())
#             targets.append(batch_targets)
#             indices.append(batch_indices)

#         feats_before = torch.cat(feats_before)
#         feats_after = torch.cat(feats_after)
#         targets = torch.cat(targets)
#         indices = torch.cat(indices)

#         return feats_before[indices], feats_after[indices], targets[indices]

#     # -----------------------------------------------------------------
#     # 3️⃣ 生成 LDA 分类器（不再进行梯度学习）
#     # -----------------------------------------------------------------
#     def _build_lda_classifier(self, use_linear_stats: bool = False):
#         """
#         使用累计的 **类均值 & 类协方差** 直接构造 LDA 分类器。
#         参数 `use_linear_stats` 只在 `refine_classifiers` 中用来标识
#         “原始统计量” vs “线性补偿后统计量”。这里两者共享同一
#         `self.class_stats`，因为在 `update_stats` 中已经把线性变换
#         应用到了累计统计量上。
#         """
#         # 1) 收集统计量
#         class_ids = sorted(self.class_stats.keys())
#         num_classes = len(class_ids)
#         d = next(iter(self.class_stats.values())).mu.shape[0]

#         N_vec   = torch.tensor([self.class_stats[c].N for c in class_ids],
#                                dtype=torch.float32)
#         mu_vec  = torch.stack([self.class_stats[c].mu for c in class_ids])   # (C, d)
#         cov_vec = torch.stack([self.class_stats[c].get_cov() for c in class_ids])  # (C, d, d)

#         total_N = N_vec.sum()

#         # 2) 类内散度矩阵 S_w
#         S_w = cov_vec.sum(dim=0) + 1e-6 * torch.eye(d)

#         # 3) 整体均值 μ
#         overall_mu = (N_vec.unsqueeze(1) * mu_vec).sum(dim=0) / total_N   # (d,)

#         # 4) 类间散度矩阵 S_b
#         diff = mu_vec - overall_mu.unsqueeze(0)          # (C, d)
#         S_b = (N_vec.unsqueeze(1).unsqueeze(2) *
#                diff.unsqueeze(2) @ diff.unsqueeze(1)).sum(dim=0)   # (d, d)

#         # 5) 广义特征分解： S_w^{-1} S_b
#         # 为了数值稳定，使用 torch.linalg.eig on a symmetric matrix:
#         Sw_inv_Sb = torch.linalg.solve(S_w, S_b)

#         eigvals, eigvecs = torch.linalg.eig(Sw_inv_Sb)
#         eigvals = eigvals.real
#         eigvecs = eigvecs.real

#         # 取前 (C-1) 个最大的特征向量作为投影矩阵
#         k = max(1, num_classes - 1)  # 二分类时 k=1
#         idx = torch.argsort(eigvals, descending=True)[:k]
#         W_lda = eigvecs[:, idx]                     # (d, k)

#         # 6) 把每个类的均值投影到 LDA 子空间
#         mu_proj = mu_vec @ W_lda                     # (C, k)

#         # 7) 在投影空间直接求最小二乘解把投影空间映射到 “one‑hot”标签
#         #    这相当于一次性的线性分类层：  W_fc = (Y^+) * mu_proj.T
#         #    Y 为 one‑hot (C, C)
#         Y_onehot = torch.eye(num_classes)           # (C, C)
#         # 使用伪逆得到最小二乘解
#         W_fc = torch.linalg.pinv(mu_proj) @ Y_onehot                     # (k, C)
#         b_fc = torch.zeros(num_classes)              # 偏置设为 0

#         # 8) 包装成 nn.Module
#         class LDA_Linear(nn.Module):
#             def __init__(self, proj, weight, bias):
#                 super().__init__()
#                 self.register_buffer('proj', proj)   # (d, k)
#                 self.fc = nn.Linear(weight.shape[0], weight.shape[1], bias=True)
#                 self.fc.weight.data = weight.t()    # nn.Linear expects (out, in)
#                 self.fc.bias.data = bias
#             def forward(self, x):
#                 # x : (B, d)
#                 x = x @ self.proj                 # (B, k)
#                 return self.fc(x)                 # (B, C)

#         lda_classifier = LDA_Linear(W_lda, W_fc, b_fc).to(self.device)
#         return lda_classifier

#     # -----------------------------------------------------------------
#     # 4️⃣ 重新构造分类器（不再进行梯度训练）
#     # -----------------------------------------------------------------
#     def refine_classifiers(self, fc, task_id):
#         """
#         `fc` 是原始模型在当前特征空间的线性层（用于获取维度信息）。
#         这里我们直接返回两套 LDA 分类器：
#             - "original"          : 直接使用累计统计量（已经被线性变换同步）
#             - "linear_compensate" : 与上面相同，因为在线性补偿阶段我们已经把
#                                    统计量映射到了最新空间，所以两者相等。
#         若想保留 “未做线性补偿”的版本，可在 `update_stats` 前
#         另存一份统计量副本再构造，但在大多数实验中只需要
#         当前（已补偿）的 LDA 即可。
#         """
#         # 1) 把原始的全连接层丢掉，只保留特征维度信息
#         #    (这里我们不需要对 fc 进行任何训练)
#         # 2) 直接使用累计统计量构造 LDA 分类器
#         lda_original = self._build_lda_classifier(use_linear_stats=False)

#         # 如果在 `update_stats` 中已经对累计统计量做了线性映射，
#         # 那么 “linear_compensate” 与 “original” 完全相同。
#         # 为了兼容 API，仍然返回两个对象。
#         lda_linear = copy.deepcopy(lda_original)

#         return {'original': lda_original,
#                 'linear_compensate': lda_linear}
