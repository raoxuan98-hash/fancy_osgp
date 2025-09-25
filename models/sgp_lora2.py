import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple, Callable, Any
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from scipy.stats import spearmanr

# ================= 工具函数：计算特征值权重 =================
def compute_eigval_weights(
    eigvals: torch.Tensor,
    temp: float = 5.0,
    weight_kind: str = "log1p",
    weight_p: float = 2.0,
    weight_alpha: float = 0.5,
    weight_kappa: float = 2.0
) -> torch.Tensor:
    """
    根据特征值计算缩放权重。
    """
    x = eigvals
    beta = temp

    if weight_kind == "exp":
        weights = torch.exp(-beta * x)
    elif weight_kind == "rational1":
        weights = 1.0 / (1.0 + beta * x)
    elif weight_kind == "rational2":
        weights = 1.0 / (1.0 + beta * (x ** 2))
    elif weight_kind == "sqrt_rational2":
        weights = 1.0 / torch.sqrt(1.0 + beta * (x ** 2))
    elif weight_kind == "log1p":
        weights = 1.0 / (1.0 + beta * torch.log1p(x ** weight_p))
    elif weight_kind == "power_family":
        weights = (1.0 + beta * (x ** weight_p)) ** (-weight_alpha)
    elif weight_kind == "stretched_exp":
        weights = torch.exp(- (beta * x) ** weight_kappa)
    else:
        raise ValueError(f"Unknown weight_kind='{weight_kind}'")

    max_weight = weights.max()
    weights = weights / (max_weight + 1e-8)
    return weights

# ================= 固定正交基投影（支持动态更新） =================
class FixedBasisProjection(nn.Module):
    def __init__(self, basis: torch.Tensor):
        super().__init__()
        assert basis.ndim == 2 and basis.size(0) == basis.size(1), "basis must be square matrix"
        self.register_buffer("basis", basis)

    def forward(self) -> torch.Tensor:
        return self.basis  # shape: (d, d)

    def update_basis(self, new_basis: torch.Tensor) -> None:
        """动态更新正交基，保持设备/类型一致"""
        self.basis.copy_(new_basis)


# ================= SGPLinear：带可训练 scales 的 LoRA =================
class SGPLinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, proj: nn.Module, proj_temp: float = 2.0):
        super().__init__()
        self.linear = linear
        self.dim = linear.in_features
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj  # FixedBasisProjection
        self.proj_temp = proj_temp

        # LoRA matrices
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

        # 可训练方向缩放
        self.scales = nn.Parameter(torch.ones(self.dim))
        self.register_buffer("eigvals", torch.ones(self.dim))
        self.register_buffer("importance_weights", torch.ones(self.dim))  # 默认无加权

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    @property
    def target_weight(self) -> torch.Tensor:
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.linear(x)
        P_basis = self.P()  # (d, d)
        A_eff_base = self.A @ P_basis  # (r, d)
        A_eff = A_eff_base * self.scales.unsqueeze(0)  # 应用可学习缩放
        h = F.linear(x, A_eff)
        lora_update = F.linear(h, self.B)
        return orig_out + lora_update

    @property
    def scales_init(self) -> torch.Tensor:
        return compute_eigval_weights(self.eigvals, self.proj_temp, "log1p", weight_p=1)

    def merge_lora_weights(self) -> None:
        with torch.no_grad():
            P_basis = self.P()
            A_eff = self.A @ P_basis * self.scales.unsqueeze(0)
            delta = self.B @ A_eff  # (out, d)
            self.linear.weight += delta.to(self.linear.weight.device)
            self.B.zero_()

    def scales_regularization_loss(self, p: float = 2.0, importance_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.pow(torch.abs(self.scales - self.scales_init), p)
        if importance_weights is not None:
            assert importance_weights.shape == self.scales.shape, "Importance weights shape mismatch"
            diff = diff * importance_weights
        return diff.mean()


# ================= SGPQKV：用于 Attention QKV 的适配器 =================
class SGPQKV(nn.Module):
    def __init__(self, qkv: nn.Linear, r: int, proj: nn.Module, proj_temp: float = 2.0):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r
        self.P = proj
        self.proj_temp = proj_temp

        self.A = nn.Parameter(torch.zeros(r, self.dim))
        self.B = nn.Parameter(torch.zeros(3 * self.dim, r))

        self.scales = nn.Parameter(torch.ones(self.dim))
        self.register_buffer("eigvals", torch.ones(self.dim))
        self.register_buffer("importance_weights", torch.ones(self.dim))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    @property
    def target_weight(self) -> torch.Tensor:
        return self.qkv.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_qkv = self.qkv(x)
        P_basis = self.P()
        A_eff_base = self.A @ P_basis
        A_eff = A_eff_base * self.scales.unsqueeze(0)
        h = F.linear(x, A_eff)
        lora_update = F.linear(h, self.B)
        return orig_qkv + lora_update

    def merge_lora_weights(self) -> None:
        with torch.no_grad():
            P_basis = self.P()
            A_eff = self.A @ P_basis * self.scales.unsqueeze(0)
            delta = self.B @ A_eff
            self.qkv.weight += delta.to(self.qkv.weight.device)
            self.B.zero_()

    @property
    def scales_init(self) -> torch.Tensor:
        return compute_eigval_weights(self.eigvals, self.proj_temp, "log1p", weight_p=1)

    def scales_regularization_loss(self, p: float = 2.0, importance_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.pow(torch.abs(self.scales - self.scales_init), p)
        if importance_weights is not None:
            assert importance_weights.shape == self.scales.shape
            diff = diff * importance_weights
        return diff.mean()


# ================= 基础 LoRA ViT 类 =================
class BaseLoRAViT(nn.Module):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        qkv_adapter_cls: type = SGPQKV,
        linear_adapter_cls: type = SGPLinear,
        placeholder_proj_factory: Optional[Callable[[int, torch.device, torch.dtype], nn.Module]] = None,
        include_norm: bool = True,
        proj_temp: float = 2.0):
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.r = r
        try:
            self.feature_dim = vit_model.embed_dim
        except AttributeError:
            self.feature_dim = 768
        self.use_projection = True
        self.lora_layer = list(lora_layer) if lora_layer is not None else list(range(len(vit_model.blocks)))
        self.proj_temp = proj_temp

        # 冻结原始参数
        for n, p in vit_model.named_parameters():
            p.requires_grad_(include_norm and "norm" in n)

        self.lora_modules = nn.ModuleDict()
        dev = vit_model.patch_embed.proj.weight.device

        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer:
                continue

            def make_placeholder(d: int, dtype: torch.dtype) -> nn.Module:
                if placeholder_proj_factory is not None:
                    return placeholder_proj_factory(d, dev, dtype)
                else:
                    return FixedBasisProjection(torch.eye(d, device=dev, dtype=dtype))

            # QKV
            qkv_in, qkv_dtype = blk.attn.qkv.in_features, blk.attn.qkv.weight.dtype
            qkv_proj = make_placeholder(qkv_in, qkv_dtype)
            new_qkv = qkv_adapter_cls(blk.attn.qkv, r, qkv_proj, proj_temp=self.proj_temp)
            blk.attn.qkv = new_qkv
            self.lora_modules[f"block_{idx}_attn_qkv"] = new_qkv

            # Proj
            proj_in, proj_dtype = blk.attn.proj.in_features, blk.attn.proj.weight.dtype
            proj_proj = make_placeholder(proj_in, proj_dtype)
            new_proj = linear_adapter_cls(blk.attn.proj, r, proj_proj, proj_temp=self.proj_temp)
            blk.attn.proj = new_proj
            self.lora_modules[f"block_{idx}_attn_proj"] = new_proj

            # MLP fc1
            fc1_in, fc1_dtype = blk.mlp.fc1.in_features, blk.mlp.fc1.weight.dtype
            fc1_proj = make_placeholder(fc1_in, fc1_dtype)
            new_fc1 = linear_adapter_cls(blk.mlp.fc1, r, fc1_proj, proj_temp=self.proj_temp)
            blk.mlp.fc1 = new_fc1
            self.lora_modules[f"block_{idx}_mlp_fc1"] = new_fc1

            # MLP fc2
            fc2_in, fc2_dtype = blk.mlp.fc2.in_features, blk.mlp.fc2.weight.dtype
            fc2_proj = make_placeholder(fc2_in, fc2_dtype)
            new_fc2 = linear_adapter_cls(blk.mlp.fc2, r, fc2_proj, proj_temp=self.proj_temp)
            blk.mlp.fc2 = new_fc2
            self.lora_modules[f"block_{idx}_mlp_fc2"] = new_fc2

        self.lora_vit = vit_model
        self.reset_parameters_svd()

    def reset_parameters_svd(self) -> None:
        for module in self.lora_modules.values():
            if hasattr(module, 'target_weight'):
                W = module.target_weight
                _, _, Vh = torch.linalg.svd(W, full_matrices=False)
                module.A.data = Vh[:self.r, :].clone()
                module.B.data.zero_()

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        """根据协方差矩阵更新投影基和 scales 初始化"""
        self.merge_lora_weights()  # 先合并，避免干扰
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            module = self.lora_modules[name]

            eps = 1e-6
            device, dtype = cov.device, cov.dtype
            d = cov.size(0)
            cov_stable = cov + eps * torch.eye(d, device=device, dtype=dtype)
            eigvals, eigvecs = torch.linalg.eigh(cov_stable)
            eigvals = torch.abs(eigvals)

            # 归一化特征值
            scale_ = d / (eigvals.sum() + eps)
            eigvals = eigvals * scale_

            # 更新正交基
            module.P.update_basis(eigvecs)

            # 计算初始 scales
            scales_init = compute_eigval_weights(
                eigvals, temp=self.proj_temp, weight_kind="log1p", weight_p=1)

            # 设置 scales 和重要性权重（按特征值比例加权）
            module.scales.data.copy_(scales_init)
            module.eigvals.copy_(eigvals)

            # log1p_eigvals = torch.log1p(eigvals) + 0.01
            # module.importance_weights.copy_(log1p_eigvals)
            module.importance_weights.copy_(torch.ones_like(eigvals))

    def regularization_loss(self) -> torch.Tensor:
        """返回 scales 正则化损失（按 importance_weights 加权）"""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for module in self.lora_modules.values():
            imp_weights = getattr(module, 'importance_weights', None)
            loss = module.scales_regularization_loss(p=1.0, importance_weights=imp_weights)
            total_loss += loss
            count += 1
        
        total_loss /= count
        rhos = self.compute_spearman_stats(only_first=True)
        avg_rho = torch.tensor([rho for key, rho in rhos.items()]).mean().squeeze().item()

        # print(f"Total: {1000 * total_loss.item():.7f}, "
        #       f"Avg spearman rho: {avg_rho:.4f}")
        
        return 0.1 * self.scales_reg_weight * total_loss if hasattr(self, 'scales_reg_weight') else total_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_vit(x)

    def get_module_names(self) -> list:
        return list(self.lora_modules.keys())

    def merge_lora_weights(self) -> None:
        self.eval()
        with torch.no_grad():
            for mod in self.lora_modules.values():
                mod.merge_lora_weights()

    def finalize_without_lora(self) -> None:
        self.eval()
        with torch.no_grad():
            for _, mod in self.lora_modules.items():
                mod.merge_lora_weights()

            for idx, blk in enumerate(self.lora_vit.blocks):
                name_qkv = f"block_{idx}_attn_qkv"
                if name_qkv in self.lora_modules:
                    adapter = self.lora_modules[name_qkv]
                    blk.attn.qkv = adapter.qkv

                name_proj = f"block_{idx}_attn_proj"
                if name_proj in self.lora_modules:
                    adapter = self.lora_modules[name_proj]
                    blk.attn.proj = adapter.linear

                name_fc1 = f"block_{idx}_mlp_fc1"
                if name_fc1 in self.lora_modules:
                    adapter = self.lora_modules[name_fc1]
                    blk.mlp.fc1 = adapter.linear

                name_fc2 = f"block_{idx}_mlp_fc2"
                if name_fc2 in self.lora_modules:
                    adapter = self.lora_modules[name_fc2]
                    blk.mlp.fc2 = adapter.linear

            self.lora_modules = nn.ModuleDict()

    def compute_spearman_stats(self, only_first=False) -> Dict[str, float]:
        stats = {}
        with torch.no_grad():
            for name, module in self.lora_modules.items():
                try:
                    rho_result = spearmanr(
                        module.scales.cpu().numpy(), 
                        module.scales_init.cpu().numpy())
                    rho = rho_result.statistic

                    # === 紧凑+增强调试输出 ===
                    s = module.scales.cpu()
                    si = module.scales_init.cpu()

                    s_min, s_max = s.min().item(), s.max().item()
                    si_min, si_max = si.min().item(), si.max().item()
                    s_mean, s_std = s.mean().item(), s.std().item()
                    si_mean, si_std = si.mean().item(), si.std().item()

                    # === 计算 KL 散度：将 s 和 si 视为分布 ===
                    def kl_divergence(p, q):
                        # p, q: 1D tensors, 将被 softmax 归一化
                        p_norm = torch.softmax(p, dim=0)
                        q_norm = torch.softmax(q, dim=0)
                        # 避免 log(0)
                        kl = torch.sum(p_norm * torch.log((p_norm + 1e-12) / (q_norm + 1e-12))).item()
                        return kl

                    kl_s_given_si = kl_divergence(s, si)   # KL(s || si)
                    kl_si_given_s = kl_divergence(si, s)   # KL(si || s)
                    js_divergence = (kl_s_given_si + kl_si_given_s) / 2  # 可选：对称 JS 散度

                    # print(f"[DEBUG] {name} | scales: [{s_min:.4f}, {s_max:.4f}] μ±σ={s_mean:.4f}±{s_std:.4f} | "
                    #     f"scales_init: [{si_min:.4f}, {si_max:.4f}] μ±σ={si_mean:.4f}±{si_std:.4f}")
                    # print(f"[DEBUG] {name} | KL(s||si): {kl_s_given_si:.4f} | KL(si||s): {kl_si_given_s:.4f} | JS: {js_divergence:.4f} | Spearman ρ: {rho:.4f}")

                except Exception as e:
                    print(f"[Warning] Spearman or KL failed for {name}: {e}")
                    rho = 0.0

                stats[name] = rho
                if only_first:
                    break

        return stats


# ================= SGPLoRAViT：支持 scales 初始化 + 正则 =================
class SGPLoRAViT(BaseLoRAViT):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        proj_temp: float = 2.0,
        k: float = 1.0,
        use_soft_projection: bool = True,
        weight_kind: str = "log1p",
        weight_p: float = 2.0,
        nsp_eps: float = 0.05,
        nsp_weight: float = 0.0,
        scales_reg_weight: float = 1.0,
        scales_reg_p: float = 2.0):
        super().__init__(
            vit_model, r, lora_layer,
            qkv_adapter_cls=SGPQKV,
            linear_adapter_cls=SGPLinear,
            placeholder_proj_factory=lambda d, dev, dtype: FixedBasisProjection(torch.eye(d, device=dev, dtype=dtype)),
            proj_temp=proj_temp)
        self.proj_temp = proj_temp
        self.use_soft_projection = use_soft_projection
        self.k = k
        self.weight_kind = weight_kind
        self.weight_p = weight_p
        self.nsp_eps = nsp_eps
        self.nsp_weight = nsp_weight
        self.scales_reg_weight = scales_reg_weight
        self.scales_reg_p = scales_reg_p

    def get_param_groups(self) -> Dict[str, list]:
        """将参数分为 scales 和其他参数两组，用于不同学习率/正则"""
        scale_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ".scales" in name:  # 精确匹配
                scale_params.append(param)
            else:
                other_params.append(param)
        return {"scales": scale_params, "others": other_params}