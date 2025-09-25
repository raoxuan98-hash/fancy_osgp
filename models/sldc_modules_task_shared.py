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
    def __init__(self, means, shared_covariance_matrix):
        self.means = [m.cpu() for m in means]  # 多个均值向量
        if isinstance(shared_covariance_matrix, list):
            shared_covariance_matrix = torch.stack(shared_covariance_matrix, dim=0).mean(dim=0)
        self.covariance = shared_covariance_matrix.cpu()
        I = torch.eye(self.covariance.size(0), device=self.covariance.device)
        self.cholesky_factor = cholesky_decomposition(self.covariance + 1e-4 * I).cpu()

    def update_shared_covariance(self, new_shared_covariance_matrix):
        if isinstance(new_shared_covariance_matrix, list):
            new_shared_covariance_matrix = torch.stack(new_shared_covariance_matrix, dim=0).mean(dim=0)
        self.covariance = new_shared_covariance_matrix.cpu()
        I = torch.eye(self.covariance.size(0), device=self.covariance.device)
        self.cholesky_factor = cholesky_decomposition(self.covariance + 1e-4 * I).cpu()

class Drift_Compensator(object):
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.class_statistics = {}
        self.args = args
        self.covariance_sharing_mode = args.get('covariance_sharing_mode', 'task_wise')
        assert self.covariance_sharing_mode in {'per_class', 'task_wise'}
        self.noise_cache = None

    def update_statistics(self, task_id, model_after, data_loader):
        feature_vectors, targets = self.extract_features(model_after, data_loader)
        if task_id == 0:
            feat_dim = feature_vectors.size(1)
            self.noise_cache = torch.randn(50000, feat_dim)

        class_stats = self.compute_class_statistics(feature_vectors, targets)
        assert set(class_stats.keys()).isdisjoint(set(self.class_statistics.keys())), \
            f"Class IDs {set(class_stats.keys()) & set(self.class_statistics.keys())} already exist."
        
        self.class_statistics.update(class_stats)

    def compute_class_statistics(self, feature_vectors, labels):
        feature_covariance = torch.cov(feature_vectors.T)
        unique_labels = torch.unique(labels)
        stats_dict = {}
        n_clusters_per_class = self.args.get('n_clusters_per_class', 3)
        min_cluster_distance = self.args.get('min_cluster_distance', 0.5)
        for label in unique_labels:
            mask = (labels == label)
            class_features = feature_vectors[mask].cpu().numpy()
            kmeans_class = KMeans(n_clusters=n_clusters_per_class, random_state=0, n_init=20, init='k-means++').fit(class_features)
            class_centers = kmeans_class.cluster_centers_
            class_centers = self._enforce_min_distance(class_centers, min_cluster_distance)
            means = [torch.from_numpy(center).float() for center in class_centers]
            stats_dict[int(label.item())] = MultiMeanSharedCovStatistics(means, feature_covariance)
        return stats_dict
    
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
            class_cov_cholesky = stat.cholesky_factor.to(self.device)
            for mean_vec in stat.means:
                mean_vec = mean_vec.to(self.device)
                
                start_idx = (class_id * len(stat.means) + len(synthetic_samples)) * num_samples_per_mean
                start_idx %= noise_cache.size(0)
                end_idx = start_idx + num_samples_per_mean
                
                if end_idx > noise_cache.size(0):
                    Z = torch.cat([noise_cache[start_idx:], noise_cache[:end_idx - noise_cache.size(0)]], dim=0)
                else:
                    Z = noise_cache[start_idx:end_idx]

                samples = mean_vec + torch.mm(Z, class_cov_cholesky.t())
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
