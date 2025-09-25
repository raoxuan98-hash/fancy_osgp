import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import copy
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from torch.nn.functional import softmax

def cholesky_decomposition(matrix):
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    for j in range(n):
        diag_sum = torch.sum(L[j, :j] ** 2)
        diagonal_value = matrix[j, j] - diag_sum
        L[j, j] = torch.sqrt(torch.clamp(diagonal_value, min=1e-8))
        if j < n - 1:
            off_diag_sum = torch.mm(L[j+1:, :j], L[j, :j].unsqueeze(1)).squeeze(1)
            L[j+1:, j] = (matrix[j+1:, j] - off_diag_sum) / L[j, j]
    return L

def symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5):
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    label_one_hot = F.one_hot(targets, num_classes=pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    ce_loss = -torch.sum(label_one_hot * torch.log(pred), dim=1).mean()
    rce_loss = -torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
    total_loss = sce_a * ce_loss + sce_b * rce_loss
    return total_loss

class MultiMeanSharedCovStatistics:
    def __init__(self, means, original_covariance_matrix):
        """
        初始化时，只保存原始协方差矩阵。
        加权协方差将由 Drift_Compensator 管理和缓存。
        """
        self.means = [m.cpu() for m in means]
        if isinstance(original_covariance_matrix, list):
            original_covariance_matrix = torch.stack(original_covariance_matrix, dim=0).mean(dim=0)
        self.original_covariance = original_covariance_matrix.cpu().clone()

    def get_original_covariance(self):
        """获取原始协方差矩阵"""
        return self.original_covariance

    def get_representative_mean(self):
        """获取该类的代表均值（所有子均值的平均）"""
        return torch.stack(self.means, dim=0).mean(dim=0)


class Drift_Compensator(object):
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.class_statistics = {}
        self.args = args
        self.covariance_sharing_mode = 'per_class'
        self.temp = 1.0
        self.noise_cache = None
        # ===== 新增：加权协方差缓存字典 =====
        self.weighted_cov_cache = {} # {class_id: torch.Tensor}

    def update_statistics(self, task_id, model_after, data_loader):
        feature_vectors, targets = self.extract_features(model_after, data_loader)

        # === Step 1: 为当前任务的每个类计算并保存其原始统计量 ===
        new_class_stats = self.compute_class_statistics(feature_vectors, targets)
        
        # 检查类ID是否冲突
        assert set(new_class_stats.keys()).isdisjoint(set(self.class_statistics.keys())), \
            f"Class IDs {set(new_class_stats.keys()) & set(self.class_statistics.keys())} already exist."
        
        # 更新全局类统计字典
        self.class_statistics.update(new_class_stats)

        # === Step 2: 为所有现有类（包括新加入的）重新计算并缓存加权协方差 ===
        self._update_weighted_covariance_cache()

        # === 初始化噪声缓存（仅在第一次任务时）===
        if task_id == 0:
            feat_dim = feature_vectors.size(1)
            self.noise_cache = torch.randn(50000, feat_dim)

    def _update_weighted_covariance_cache(self):
        """
        为 self.class_statistics 中的所有类计算加权协方差，并更新缓存。
        此方法在每次 update_statistics 后被调用。
        """
        if len(self.class_statistics) == 0:
            self.weighted_cov_cache = {}
            return

        # 收集所有类的代表均值和原始协方差
        all_class_means = []
        all_original_covariances = []
        all_class_ids = []

        for class_id, stat in self.class_statistics.items():
            rep_mean = stat.get_representative_mean().numpy()
            orig_cov = stat.get_original_covariance().numpy()
            all_class_means.append(rep_mean)
            all_original_covariances.append(orig_cov)
            all_class_ids.append(class_id)

        all_class_means = np.stack(all_class_means, axis=0) # (num_classes, d)
        all_original_covariances = np.stack(all_original_covariances, axis=0) # (num_classes, d, d)

        # 为每个类计算加权协方差并缓存
        for i, class_id in enumerate(all_class_ids):
            target_mean = all_class_means[i]
            target_mean_norm = target_mean / (np.linalg.norm(target_mean) + 1e-8)
            all_class_means_norm = all_class_means / (np.linalg.norm(all_class_means, axis=1, keepdims=True) + 1e-8)
            cosine_similarities = target_mean_norm @ all_class_means_norm.T  # (num_classes,)
            temperature = self.temp
            weights = softmax(torch.from_numpy(cosine_similarities / temperature).float(), dim=0).numpy()
            weighted_cov = np.einsum('i,ijk->jk', weights, all_original_covariances)
            weighted_cov += np.eye(weighted_cov.shape[0]) * 1e-4
            self.weighted_cov_cache[class_id] = torch.from_numpy(weighted_cov).float()

    def compute_class_statistics(self, feature_vectors, labels):
        unique_labels = torch.unique(labels)
        stats_dict = {}

        n_clusters_per_class = self.args.get('n_clusters_per_class', 3)
        min_cluster_distance = self.args.get('min_cluster_distance', 0.5)

        for label in unique_labels:
            mask = (labels == label)
            class_features = feature_vectors[mask].cpu().numpy()
            class_mean = class_features.mean(axis=0)  # (d,)

            if len(class_features) < n_clusters_per_class:
                means = [torch.from_numpy(class_mean).float()]
                # 计算该类的原始协方差
                if len(class_features) > 1:
                    original_cov = np.cov(class_features, rowvar=False, ddof=1)
                else:
                    original_cov = np.eye(class_features.shape[1]) * 1e-6
                original_cov = torch.from_numpy(original_cov).float()
            else:
                # 类内聚类
                kmeans_class = KMeans(
                    n_clusters=n_clusters_per_class,
                    random_state=0,
                    n_init=20,
                    init='k-means++').fit(class_features)
                class_centers = kmeans_class.cluster_centers_
                class_centers = self._enforce_min_distance(class_centers, min_cluster_distance)
                means = [torch.from_numpy(center).float() for center in class_centers]

                # 使用类内所有样本计算原始协方差
                if len(class_features) > 1:
                    original_cov = np.cov(class_features, rowvar=False, ddof=1)
                else:
                    original_cov = np.eye(class_features.shape[1]) * 1e-6
                original_cov = torch.from_numpy(original_cov).float()

            # 创建统计对象，只传入原始协方差
            stat_obj = MultiMeanSharedCovStatistics(means, original_cov)
            stats_dict[int(label.item())] = stat_obj

        return stats_dict

    def _compute_class_statistics_fallback(self, feature_vectors, labels):
        return self.compute_class_statistics(feature_vectors, labels)
    
    def _enforce_min_distance(self, centers, min_distance):
        """确保聚类中心之间满足最小距离约束"""
        centers = centers.copy()
        n_centers = len(centers)
        for i in range(n_centers):
            for j in range(i + 1, n_centers):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_distance:
                    direction = centers[i] - centers[j]
                    if np.linalg.norm(direction) > 1e-8:
                        direction /= np.linalg.norm(direction)
                    else:
                        direction = np.random.randn(len(centers[i]))
                        direction /= np.linalg.norm(direction)
                    move = (min_distance - dist) * direction * 0.5
                    centers[i] += move
                    centers[j] -= move
        return centers

    def refine_classifier(self, classifier, task_id, epochs=6):
        num_samples_per_mean = 512
        batch_size = 32 * sum(len(stat.means) for stat in self.class_statistics.values()) // 10
        lr = 5e-4

        classifier.to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

        synthetic_samples, synthetic_targets = [], []
        noise_cache = self.noise_cache.to(self.device)

        for class_id, stat in self.class_statistics.items():
            # 关键修改：直接从缓存中获取加权协方差
            if class_id not in self.weighted_cov_cache:
                raise RuntimeError(f"Weighted covariance for class {class_id} is not cached. Call update_statistics first.")
            
            weighted_cov = self.weighted_cov_cache[class_id]
            # 为这个加权协方差计算 Cholesky 分解
            I = torch.eye(weighted_cov.size(0), device=weighted_cov.device)
            cholesky_factor = cholesky_decomposition(weighted_cov + 1e-4 * I).to(self.device)

            for mean_vec in stat.means:
                mean_vec = mean_vec.to(self.device)
                
                start_idx = (class_id * len(stat.means) + len(synthetic_samples)) * num_samples_per_mean
                start_idx %= noise_cache.size(0)
                end_idx = start_idx + num_samples_per_mean
                
                if end_idx > noise_cache.size(0):
                    Z = torch.cat([noise_cache[start_idx:], noise_cache[:end_idx - noise_cache.size(0)]], dim=0)
                else:
                    Z = noise_cache[start_idx:end_idx]

                samples = mean_vec + torch.mm(Z, cholesky_factor.t())
                targets = torch.full((num_samples_per_mean,), class_id, device=self.device)

                synthetic_samples.append(samples)
                synthetic_targets.append(targets)

        inputs = torch.cat(synthetic_samples, dim=0).detach().clone()
        targets = torch.cat(synthetic_targets, dim=0).detach().clone()

        for epoch in range(epochs):
            perm = torch.randperm(inputs.size(0), device=self.device)
            inputs_shuffled = inputs[perm]
            targets_shuffled = targets[perm]

            total_loss = 0.0
            num_samples = inputs.size(0)
            num_batches = (num_samples + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_samples)
                if start >= end:
                    continue

                batch_x = inputs_shuffled[start:end]
                batch_y = targets_shuffled[start:end]

                optimizer.zero_grad()
                logits = classifier(batch_x)
                loss = symmetric_cross_entropy_loss(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * (end - start)

            avg_loss = total_loss / num_samples
            if (epoch + 1) % 3 == 0:
                print(f"[Classifier Refinement] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            scheduler.step()
        return classifier

    # ... (get_auxiliary_loader, extract_features_before_after, extract_features 方法保持不变) ...
    def get_auxiliary_loader(self, args):
        """
        获取辅助数据加载器（目前未使用，但保留接口以兼容未来扩展）
        """
        if hasattr(self, '_aux_loader') and self._aux_loader is not None:
            return self._aux_loader

        aux_dataset_type = args['aux_dataset']
        num_samples = args['auxiliary_data_size']

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if aux_dataset_type == 'imagenet':
            if 'auxiliary_data_path' not in args:
                raise ValueError("必须提供 auxiliary_data_path")
            dataset = datasets.ImageFolder(args['auxiliary_data_path'], transform=transform)
        elif aux_dataset_type == 'cifar10':
            dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        elif aux_dataset_type == 'svhn':
            dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        elif aux_dataset_type == 'flickr8k':
            dataset = datasets.ImageFolder(args['auxiliary_data_path'], transform=transform)
        else:
            raise ValueError(f"不支持的 aux_dataset_type: {aux_dataset_type}")

        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        self._aux_loader = loader
        return loader

    @torch.no_grad()
    def extract_features_before_after(self, model_before, model_after, data_loader):
        model_before, model_after = model_before.to(self.device), model_after.to(self.device)
        model_before.eval(); model_after.eval()
        
        feats_before, feats_after, targets = [], [], []
        
        for batch_indices, inputs, batch_targets in data_loader:
            inputs = inputs.to(self.device)
            feats_before.append(model_before(inputs).cpu())
            feats_after.append(model_after(inputs).cpu())
            targets.append(batch_targets)

        feats_before = torch.cat(feats_before)
        feats_after = torch.cat(feats_after)
        targets = torch.cat(targets)
        return feats_before, feats_after, targets

    @torch.no_grad()
    def extract_features(self, model, data_loader):
        model.eval()
        model.to(self.device)
        
        features_list, labels_list = [], []
        for batch_indices, inputs, batch_targets in data_loader:
            inputs = inputs.to(self.device)
            features = model(inputs).cpu()
            features_list.append(features)
            labels_list.append(batch_targets)

        all_features = torch.cat(features_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        return all_features, all_labels 