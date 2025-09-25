import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple
from timm.models.vision_transformer import VisionTransformer as timm_ViT


# ================= 工具函数：计算特征值权重 =================
def compute_eigval_weights(
    eigvals: torch.Tensor,
    temp: float = 5.0,
    weight_kind: str = "exp",
    weight_p: float = 2.0,
    weight_alpha: float = 0.5,
    weight_kappa: float = 2.0) -> torch.Tensor:
    """
    根据特征值计算缩放权重，与原始 build_projection 逻辑一致。
    """
    beta = temp
    x = eigvals

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
    weights = weights / (max_weight + 1e-8)  # 避免除零
    return weights


# ================= 固定正交基投影 =================
class FixedBasisProjection(nn.Module):
    def __init__(self, basis: torch.Tensor):
        super().__init__()
        assert basis.ndim == 2 and basis.size(0) == basis.size(1), "basis must be square matrix"
        self.register_buffer("basis", basis)  # 正交基，固定不变

    def forward(self) -> torch.Tensor:
        return self.basis  # shape: (d, d)


# ================= SGPLinear：带可训练 scales 的 LoRA =================
class SGPLinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, proj: nn.Module):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj  # FixedBasisProjection, returns (d, d)

        # LoRA matrices
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

        # 可训练方向缩放 + 初始值 buffer
        self.scales = nn.Parameter(torch.ones(self.in_features))  # (d,)
        self.register_buffer("scales_init", torch.ones(self.in_features))  # 保存初始值

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.linear(x)
        P_basis = self.P()  # (d, d)
        A_eff_base = self.A @ P_basis  # (r, d)
        A_eff = A_eff_base * self.scales.unsqueeze(0)  # (r, d) ← 关键：应用可学习缩放
        h = F.linear(x, A_eff)              # (·, r)
        lora_update = F.linear(h, self.B)   # (·, out)
        return orig_out + lora_update

    def merge_lora_weights(self) -> None:
        with torch.no_grad():
            P_basis = self.P()
            A_eff = self.A @ P_basis * self.scales.unsqueeze(0)  # 应用当前 scales
            delta = self.B @ A_eff  # (out, d)
            self.linear.weight += delta.to(self.linear.weight.device)
            self.B.zero_()

    def scales_regularization_loss(self, p: float = 1.0, importance_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.pow(self.scales - self.scales_init, p)
        if importance_weights is not None:
            assert importance_weights.shape == self.scales.shape, "Importance weights shape mismatch"
            diff = diff * importance_weights
        return diff.mean()


# ================= SGPQKV：用于 Attention QKV 的适配器 =================
class SGPQKV(nn.Module):
    def __init__(self, qkv: nn.Linear, r: int, proj: nn.Module):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r
        self.P = proj
        self.A = nn.Parameter(torch.zeros(r, self.dim))
        self.B = nn.Parameter(torch.zeros(3 * self.dim, r))
        self.scales = nn.Parameter(torch.ones(self.dim))  # (d,)
        self.register_buffer("scales_init", torch.ones(self.dim))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_qkv = self.qkv(x)
        P_basis = self.P()
        A_eff_base = self.A @ P_basis
        A_eff = A_eff_base * self.scales.unsqueeze(0)  # (r, d)
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

    def scales_regularization_loss(self, p: float = 1.0, importance_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.pow(self.scales - self.scales_init, p)
        if importance_weights is not None:
            assert importance_weights.shape == self.scales.shape, "Importance weights shape mismatch"
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
        placeholder_proj_factory: Optional[callable] = None,  # (d, device, dtype) -> nn.Module
        include_norm: bool = True
    ):
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.r = r
        try:
            self.feature_dim = vit_model.embed_dim
        except:
            self.feature_dim = 768
        self.use_projection = True
        self.lora_layer = (list(lora_layer) if lora_layer is not None else list(range(len(vit_model.blocks))))

        for n, p in vit_model.named_parameters():
            if include_norm and "norm" in n:
                p.requires_grad_(True)
            else:
                p.requires_grad = False

        self.lora_modules = nn.ModuleDict()

        dev = vit_model.patch_embed.proj.weight.device

        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer:
                continue

            # 缓存 dtype / in_features
            qkv_in = blk.attn.qkv.in_features
            qkv_dtype = blk.attn.qkv.weight.dtype

            proj_in = blk.attn.proj.in_features
            proj_dtype = blk.attn.proj.weight.dtype

            fc1_in = blk.mlp.fc1.in_features
            fc1_dtype = blk.mlp.fc1.weight.dtype

            fc2_in = blk.mlp.fc2.in_features
            fc2_dtype = blk.mlp.fc2.weight.dtype

            # 占位投影工厂
            def make_placeholder(d, dtype):
                if placeholder_proj_factory is not None:
                    return placeholder_proj_factory(d, dev, dtype)
                else:
                    return FixedBasisProjection(torch.eye(d, device=dev, dtype=dtype))

            # QKV
            qkv_proj = make_placeholder(qkv_in, qkv_dtype)
            new_qkv = qkv_adapter_cls(blk.attn.qkv, r, qkv_proj)
            blk.attn.qkv = new_qkv
            self.lora_modules[f"block_{idx}_attn_qkv"] = new_qkv

            # Attention Proj
            proj_proj = make_placeholder(proj_in, proj_dtype)
            new_proj = linear_adapter_cls(blk.attn.proj, r, proj_proj)
            blk.attn.proj = new_proj
            self.lora_modules[f"block_{idx}_attn_proj"] = new_proj

            # MLP fc1
            fc1_proj = make_placeholder(fc1_in, fc1_dtype)
            new_fc1 = linear_adapter_cls(blk.mlp.fc1, r, fc1_proj)
            blk.mlp.fc1 = new_fc1
            self.lora_modules[f"block_{idx}_mlp_fc1"] = new_fc1

            # MLP fc2
            fc2_proj = make_placeholder(fc2_in, fc2_dtype)
            new_fc2 = linear_adapter_cls(blk.mlp.fc2, r, fc2_proj)
            blk.mlp.fc2 = new_fc2
            self.lora_modules[f"block_{idx}_mlp_fc2"] = new_fc2

        self.lora_vit = vit_model
        self.reset_parameters_svd()

    def reset_parameters_svd(self) -> None:
        for _, module in self.lora_modules.items():
            if isinstance(module, (SGPQKV, SGPLinear)):
                W = module.qkv.weight if hasattr(module, 'qkv') else module.linear.weight
                _, _, Vh = torch.linalg.svd(W, full_matrices=False)
                module.A.data = Vh[: self.r, :].clone()
                module.B.data.zero_()
                # scales 保留默认 1.0，实际初始化在 update_projection_matrices 中完成

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def kl_regularization(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_vit(x)

    def get_module_names(self):
        return list(self.lora_modules.keys())

    def merge_lora_weights(self) -> None:
        self.eval()
        with torch.no_grad():
            for _, mod in self.lora_modules.items():
                mod.merge_lora_weights()

    def finalize_without_lora(self) -> None:
        self.eval()
        with torch.no_grad():
            for _, mod in self.lora_modules.items():
                mod.merge_lora_weights()

            for idx, blk in enumerate(self.lora_vit.blocks):
                # QKV
                name_qkv = f"block_{idx}_attn_qkv"
                if name_qkv in self.lora_modules:
                    adapter = self.lora_modules[name_qkv]
                    blk.attn.qkv = adapter.qkv

                # Proj
                name_proj = f"block_{idx}_attn_proj"
                if name_proj in self.lora_modules:
                    adapter = self.lora_modules[name_proj]
                    blk.attn.proj = adapter.linear

                # MLP fc1
                name_fc1 = f"block_{idx}_mlp_fc1"
                if name_fc1 in self.lora_modules:
                    adapter = self.lora_modules[name_fc1]
                    blk.mlp.fc1 = adapter.linear

                # MLP fc2
                name_fc2 = f"block_{idx}_mlp_fc2"
                if name_fc2 in self.lora_modules:
                    adapter = self.lora_modules[name_fc2]
                    blk.mlp.fc2 = adapter.linear

            self.lora_modules = nn.ModuleDict()


# ================= SGPLoRAViT：支持 scales 初始化 + 正则（按特征值加权）=================
class SGPLoRAViT(BaseLoRAViT):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        proj_temp: float = 1.0,
        k: float = 0.5,
        use_soft_projection: bool = True,
        weight_kind: str = "exp",
        weight_p: float = 2.0,
        nsp_eps: float = 0.05,
        nsp_weight: float = 0.0,

        # —————— 新增：scales 正则超参 ——————
        scales_reg_weight: float = 1.0,
        scales_reg_p: float = 2.0):

        super().__init__(
            vit_model, r, lora_layer,
            qkv_adapter_cls=SGPQKV,
            linear_adapter_cls=SGPLinear,
            placeholder_proj_factory=lambda d, dev, dtype: FixedBasisProjection(torch.eye(d, device=dev, dtype=dtype)))
        
        self.proj_temp = proj_temp
        self.use_soft_projection = use_soft_projection
        self.k = k
        self.weight_kind = weight_kind
        self.weight_p = weight_p
        self.nsp_eps = nsp_eps
        self.nsp_weight = nsp_weight

        # 正则相关
        self.scales_reg_weight = scales_reg_weight
        self.scales_reg_p = scales_reg_p

    @torch.no_grad()
    def _ensure_merged_before_rebuild(self):
        self.merge_lora_weights()

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        self._ensure_merged_before_rebuild()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            module = self.lora_modules[name]

            # 计算特征分解
            eps = 1e-6
            device = cov.device
            dtype = cov.dtype
            d = cov.size(0)
            cov_stable = cov + eps * torch.eye(d, device=device, dtype=dtype)
            eigvals, eigvecs = torch.linalg.eigh(cov_stable)  # ascending
            eigvals = torch.abs(eigvals)

            # 归一化特征值
            sum_vals = eigvals.sum()
            scale_ = d / (sum_vals + eps)
            eigvals = eigvals * scale_

            # 设置固定正交基
            module.P = FixedBasisProjection(eigvecs)

            # 计算初始 scales 权重
            weights = compute_eigval_weights(
                eigvals,
                temp=self.proj_temp,
                weight_kind=self.weight_kind,
                weight_p=self.weight_p,
                weight_alpha=0.5,
                weight_kappa=2.0)
            
            scales_init = weights

            # 赋值给参数和 buffer
            module.scales.data.copy_(scales_init)
            module.scales_init.copy_(scales_init)

            # —————— ✅ 新增：保存重要性权重（用于正则化损失加权）——————
            # 使用归一化特征值作为重要性权重（特征值越大，惩罚越重）
            importance_weights = (1 / weights) ** 3  # shape: (d,)
            # importance_weights = eigvals / eigvals.max()
            # print(importance_weights)
            module.register_buffer("importance_weights", importance_weights)

    def kl_regularization(self) -> torch.Tensor:
        """
        返回所有模块 scales 与初始值的平均 Lp 距离惩罚，按特征值重要性加权。
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, module in self.lora_modules.items():
            imp_weights = getattr(module, 'importance_weights', None)
            loss = module.scales_regularization_loss(
                p=self.scales_reg_p,
                importance_weights=imp_weights)
            total_loss += loss
        print(100 *  total_loss)

        return 0.01*self.scales_reg_weight * total_loss

    def get_param_groups(self):
        """
        将模型参数分为 scales 和其他参数两组
        """
        scale_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "scale" in name.lower():   # 名字里包含 scale
                scale_params.append(param)
            else:
                other_params.append(param)
        
        return {
            "scales": scale_params,
            "others": other_params}