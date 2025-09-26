# ==============================================================
#  OSGP‑LoRA  (optimizable‑projection)  +  SGP‑LoRA switch
# ==============================================================
# In[]
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple
from timm.models.vision_transformer import VisionTransformer as timm_ViT

class FixedProjection(nn.Module):
    def __init__(self, P: torch.Tensor):
        super().__init__()
        self.register_buffer("P", P)
    def forward(self) -> torch.Tensor:
        return self.P
    
    def kl_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.P.device)
    
class OSGPProjection(nn.Module):
    def __init__(
        self,
        cov: torch.Tensor,
        temp: float = 2.0,
        weight_p: float = 2.0,
        trace_k: float = 0.1): 
        super().__init__()

        self.weight_p = weight_p
        self.register_buffer("prior_raw_temp", torch.tensor(math.log(temp)))
        raw_temp = math.log(temp)
        self.raw_temp = nn.Parameter(torch.tensor(raw_temp), requires_grad=True)
        self.trace_k = trace_k

        self.eps = 1e-6
        d = cov.size(0)
        cov = cov + self.eps * torch.eye(d, device=cov.device, dtype=cov.dtype)

        # 谱分解（升序）
        eigvals, eigvecs = torch.linalg.eigh(cov)
        scale = d / (eigvals.sum() + self.eps)
        eigvals =eigvals * scale

        # 缓冲：本征向量与固定先验谱
        self.register_buffer("U", eigvecs)
        self.register_buffer("eigvals", eigvals)

        # 可学习偏移 δ（相对 λ 的微调），初始化为 0
        self.delta = nn.Parameter(0.01*torch.randn_like(self.eigvals), requires_grad=True)
        self.mse = nn.MSELoss(reduction="mean")

    @property
    def temp(self):
        return torch.exp(self.raw_temp)

    def compute_weights(self, x: torch.Tensor) -> torch.Tensor:
            # 1 / (1 + β log(1 + x)) ：对数级超慢衰减（比任何幂律都慢）
        return 1.0 / (1.0 + self.temp * torch.log1p(x**self.weight_p))
        

    def project_weights(self) -> torch.Tensor:
        weight = self.compute_weights(self.eigvals * torch.exp(self.delta)) * self.trace_k
        return weight

    # -------------------- P = U diag(w) U^T --------------------
    def forward(self) -> torch.Tensor:
        w = self.project_weights()                        # (d,)
        UW = self.U * w.unsqueeze(0)                       # U @ diag(w)
        P = UW @ self.U.t()
        return P

    # -------- 高效乘法：A @ P，避免显式构造 P ----------
    def apply_to_A(self, A: torch.Tensor) -> torch.Tensor:
        w = self.project_weights()                        # (d,)
        AU = A @ self.U                                    # (r, d)
        AUW = AU * w.unsqueeze(0)                          # (r, d)
        A_eff = AUW @ self.U.t()                           # (r, d)
        return A_eff

    # ------------------------ KL 正则（未缩放） ------------------------
    def kl_loss(self) -> torch.Tensor:
        return self.delta.abs().mean() + (self.raw_temp - self.prior_raw_temp).pow(2) / 0.25

    
# ========= 占位投影（仅 OSGP 初始化用） =========
class IdentityProjection(nn.Module):
    def __init__(self, d: int, device=None, dtype=None):
        super().__init__()
        I = torch.eye(d, device=device, dtype=dtype)
        self.register_buffer("I", I)

    def forward(self) -> torch.Tensor:
        return self.I

    def apply_to_A(self, A: torch.Tensor) -> torch.Tensor:
        return A
    
    def kl_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.I.device)

# ========= 线性适配器（分 OSGP / SGP） =========
class OSGPLinear(nn.Module):
    """
    仅用于 OSGP：P 必须实现 apply_to_A(A)。
    """
    def __init__(self, linear: nn.Linear, r: int, proj: nn.Module):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.P = proj
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.linear(x)
        # 这里不再做 hasattr 判断
        A_eff = self.P.apply_to_A(self.A)
        h = F.linear(x, A_eff)
        lora_update = F.linear(h, self.B) * self.scale
        return orig_out + lora_update

    def merge_lora_weights(self) -> None:
        with torch.no_grad():
            # 对于合并，直接使用全矩阵 P()
            delta = self.scale * (self.B @ self.A @ self.P())
            self.linear.weight += delta.to(self.linear.weight.device)
            self.B.zero_()

class SGPLinear(nn.Module):
    """
    仅用于 SGP：P 是 FixedProjection（或返回 P() 的非优化投影），不要求 apply_to_A。
    """
    def __init__(self, linear: nn.Linear, r: int, proj: nn.Module):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.P = proj
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.linear(x)
        A_eff = self.A @ self.P()
        h = F.linear(x, A_eff)
        lora_update = F.linear(h, self.B) * self.scale
        return orig_out + lora_update

    def merge_lora_weights(self) -> None:
        with torch.no_grad():
            delta = self.scale * (self.B @ self.A @ self.P())
            self.linear.weight += delta.to(self.linear.weight.device)
            self.B.zero_()


# ========= QKV 适配器（分 OSGP / SGP） =========
class OSGPQKV(nn.Module):
    """
    仅用于 OSGP：P 必须实现 apply_to_A(A)。
    """
    def __init__(self, qkv: nn.Linear, r: int, proj: nn.Module):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.P = proj
        self.A = nn.Parameter(torch.zeros(r, self.dim))
        self.B = nn.Parameter(torch.zeros(3 * self.dim, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_qkv = self.qkv(x)
        A_eff = self.P.apply_to_A(self.A)
        h = F.linear(x, A_eff)
        lora_update = F.linear(h, self.B) * self.scale
        return orig_qkv + lora_update

    def merge_lora_weights(self) -> None:
        with torch.no_grad():
            delta = self.scale * (self.B @ self.A @ self.P())
            self.qkv.weight += delta.to(self.qkv.weight.device)
            self.B.zero_()


class SGPQKV(nn.Module):
    """
    仅用于 SGP：P 是 FixedProjection（或返回 P() 的非优化投影），不要求 apply_to_A。
    """
    def __init__(self, qkv: nn.Linear, r: int, proj: nn.Module):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.P = proj
        self.A = nn.Parameter(torch.zeros(r, self.dim))
        self.B = nn.Parameter(torch.zeros(3 * self.dim, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_qkv = self.qkv(x)
        A_eff = self.A @ self.P()
        h = F.linear(x, A_eff)
        lora_update = F.linear(h, self.B) * self.scale
        return orig_qkv + lora_update

    def merge_lora_weights(self) -> None:
        with torch.no_grad():
            delta = self.scale * (self.B @ self.A @ self.P())
            self.qkv.weight += delta.to(self.qkv.weight.device)
            self.B.zero_()


class BaseLoRAViT(nn.Module):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        # 新增：适配器工厂 & 占位投影工厂
        qkv_adapter_cls: type = SGPQKV,
        linear_adapter_cls: type = SGPLinear,
        placeholder_proj_factory: Optional[callable] = None,  # (d, device, dtype) -> nn.Module
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

        for p in vit_model.parameters():
            p.requires_grad = False

        self.lora_modules = nn.ModuleDict()

        dev = vit_model.patch_embed.proj.weight.device

        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer:
                continue

            # ---- 先缓存各层的 dtype / in_features（在替换任何模块之前）----
            qkv_in = blk.attn.qkv.in_features
            qkv_dtype = blk.attn.qkv.weight.dtype

            fc1_in = blk.mlp.fc1.in_features
            fc1_dtype = blk.mlp.fc1.weight.dtype

            fc2_in = blk.mlp.fc2.in_features
            fc2_dtype = blk.mlp.fc2.weight.dtype

            # 根据工厂决定占位投影
            def make_placeholder(d, dtype):
                if placeholder_proj_factory is not None:
                    return placeholder_proj_factory(d, dev, dtype)
                else:
                    return FixedProjection(torch.eye(d, device=dev, dtype=dtype))

            # ---- QKV ----
            qkv_proj = make_placeholder(qkv_in, qkv_dtype)
            new_qkv = qkv_adapter_cls(blk.attn.qkv, r, qkv_proj)
            blk.attn.qkv = new_qkv
            self.lora_modules[f"block_{idx}_attn_qkv"] = new_qkv

            # ---- MLP fc1 ----
            fc1_proj = make_placeholder(fc1_in, fc1_dtype)
            new_fc1 = linear_adapter_cls(blk.mlp.fc1, r, fc1_proj)
            blk.mlp.fc1 = new_fc1
            self.lora_modules[f"block_{idx}_mlp_fc1"] = new_fc1

            # ---- MLP fc1 ----
            fc2_proj = make_placeholder(fc2_in, fc2_dtype)
            new_fc2 = linear_adapter_cls(blk.mlp.fc2, r, fc2_proj)
            blk.mlp.fc2 = new_fc2
            self.lora_modules[f"block_{idx}_mlp_fc2"] = new_fc2


        self.lora_vit = vit_model
        self.reset_parameters_svd()

    def reset_parameters_svd(self) -> None:
        for _, module in self.lora_modules.items():
            if isinstance(module, (OSGPQKV, SGPQKV)):
                W = module.qkv.weight
            elif isinstance(module, (OSGPLinear, SGPLinear)):
                W = module.linear.weight
            else:
                continue  # 安全跳过未知类型
            _, _, Vh = torch.linalg.svd(W, full_matrices=False)
            module.A.data = Vh[: self.r, :].clone()
            module.B.data.zero_()

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
                # ---- QKV ----
                name_qkv = f"block_{idx}_attn_qkv"
                if name_qkv in self.lora_modules:
                    adapter = self.lora_modules[name_qkv]
                    blk.attn.qkv = adapter.qkv

                # ---- MLP fc1 ----
                name_fc1 = f"block_{idx}_mlp_fc1"
                if name_fc1 in self.lora_modules:
                    adapter = self.lora_modules[name_fc1]
                    blk.mlp.fc1 = adapter.linear

                # ---- MLP fc2 ----
                name_fc2 = f"block_{idx}_mlp_fc2"
                if name_fc2 in self.lora_modules:
                    adapter = self.lora_modules[name_fc2]
                    blk.mlp.fc2 = adapter.linear

            self.lora_modules = nn.ModuleDict()

        if hasattr(self, "optimizable"):
            self.optimizable = False
# ------------------------------------------------------------------

class OSGPLoRAViT(BaseLoRAViT):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        proj_temp: float = 1.0,
        trace_k: float = 0.5,
        weight_p: float = 2.0,
        kl_gamma: float = 1.0,
        
    ):
        super().__init__(
            vit_model, r, lora_layer,
            qkv_adapter_cls=OSGPQKV,
            linear_adapter_cls=OSGPLinear,
            placeholder_proj_factory=lambda d, dev, dtype: IdentityProjection(d, device=dev, dtype=dtype),
        )

        self.proj_temp = proj_temp
        self.trace_k = trace_k
        self.kl_gamma = kl_gamma
        self.optimizable = True
        self.weight_p = weight_p

    @torch.no_grad()
    def _ensure_merged_before_rebuild(self):
        self.merge_lora_weights()

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        self._ensure_merged_before_rebuild()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            proj = OSGPProjection(
                cov,
                temp=self.proj_temp,
                trace_k=self.trace_k,
                weight_p=self.weight_p,
            )
            self.lora_modules[name].P = proj

    def kl_regularization(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for mod in self.lora_modules.values():
            if hasattr(mod.P, "kl_loss"):
                total = total + mod.P.kl_loss()
        return self.kl_gamma * total

    def get_projection_learnable_params(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for mod in self.lora_modules.values():
            if isinstance(mod.P, OSGPProjection):
                params.append(mod.P.delta)
                params.append(mod.P.raw_temp)
        return params


    def collect_vit_and_delta_params(self):
        vit_params = []
        # 只从 self.lora_vit 中取，而不是整个 self（避免把 A/B 也捞进来）
        for name, p in self.lora_vit.named_parameters():
            if not p.requires_grad:
                continue
            # 显式排除 LoRA 模块参数与投影 delta
            if name.endswith(".A") or name.endswith(".B"):
                vit_params.append(p)

        delta_params = self.get_projection_learnable_params()
        return vit_params, delta_params


class SGPLoRAViT(BaseLoRAViT):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        proj_temp: float = 1.0,
        k: float = 0.5,
        use_soft_projection: bool = True,
        weight_kind: str = None,
        weight_p: float = 2.0,

        nsp_eps: float = 0.05,
        nsp_weight: float = 0.0,
    
    ):
        super().__init__(
            vit_model, r, lora_layer,
            qkv_adapter_cls=SGPQKV,
            linear_adapter_cls=SGPLinear,
            placeholder_proj_factory=lambda d, dev, dtype: FixedProjection(torch.eye(d, device=dev, dtype=dtype)))
        
        self.proj_temp = proj_temp
        self.use_soft_projection = use_soft_projection
        self.k = k
        self.optimizable = False
        self.weight_kind = weight_kind
        self.weight_p = weight_p

        self.nsp_eps = nsp_eps
        self.nsp_weight = nsp_weight

    @torch.no_grad()
    def _ensure_merged_before_rebuild(self):
        self.merge_lora_weights()

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        self._ensure_merged_before_rebuild()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            P = build_projection(
                cov,
                temp=self.proj_temp,
                soft=self.use_soft_projection,
                k=self.k,
                weight_kind=self.weight_kind,
                weight_p=self.weight_p,
                nsp_eps=self.nsp_eps,
                nsp_weight=self.nsp_weight)

            self.lora_modules[name].P = FixedProjection(P)

    # 继承 Base 的 kl_regularization=0

# ------------------------------------------------------------------
#  Original SGP projection builder (kept unchanged)
# ------------------------------------------------------------------
def build_projection(
    cov: torch.Tensor,
    soft: bool = True,
    temp: float = 5.0,
    k: float = 0.5,
    nsp_eps = 0.05, 
    nsp_weight = 0.0,
    *,

    # 新增：可切换的权重函数及其超参（全部可选）
    weight_kind: str = "stretched_exp",
    weight_alpha: float = 0.5,   # 用于 power_family: (1 + β x^p)^(-alpha)
    weight_p: float = 2.0,       # 用于 rational2/power_family: x^p
    weight_kappa: float = 2    # 用于 stretched_exp: exp[- (β x)^kappa]
) -> torch.Tensor:
    """
    Construct the *soft* or *hard* projection matrix from a covariance.
    Kept backward-compatible by default (weight_kind='exp').
    """
    eps = 1e-6
    cov = cov + eps * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov)          # ascending order
    eigvals = torch.abs(eigvals)
    d = cov.size(0)
    sum_vals = eigvals.sum()
    scale_ = d / (sum_vals + eps)
    eigvals = eigvals * scale_

    # print(eigvals)
    # ============ 新增：可切换权重函数 ============
    def compute_weights(x: torch.Tensor) -> torch.Tensor:
        beta = temp
        if weight_kind == "exp":
            # 原版：exp(-β x) → 指数尾部（最快）
            return torch.exp(-beta * x)

        elif weight_kind == "rational1":
            # 1 / (1 + β x) → 一次幂律尾部（~ 1/x）
            return 1.0 / (1.0 + beta * x)

        elif weight_kind == "rational2":
            # 1 / (1 + β x^2) → 二次幂律尾部（~ 1/x^2）
            return 1.0 / (1.0 + beta * (x ** 2))

        elif weight_kind == "sqrt_rational2":
            # 1 / sqrt(1 + β x^2) ：小 x 二次起步，尾部 ~ 1/x
            return 1.0 / torch.sqrt(1.0 + beta * (x ** 2))

        elif weight_kind == "log1p":
            # 1 / (1 + β log(1 + x)) ：对数级超慢衰减（比任何幂律都慢）
            return 1.0 / (1.0 + beta * torch.log1p(x**weight_p))

        elif weight_kind == "power_family":
            # (1 + β x^p)^(-alpha) ：统一族；alpha、p 可调
            # 例：alpha=0.5,p=2 → 1/sqrt(1 + β x^2)
            return (1.0 + beta * (x ** weight_p)) ** (-weight_alpha)

        elif weight_kind == "stretched_exp":
            # exp( - (β x)^kappa ) ：拉伸指数，kappa∈(0,1) 比指数慢、比幂律快
            return torch.exp(- (beta * x) ** weight_kappa)

        else:
            raise ValueError(
                f"Unknown weight_kind='{weight_kind}'. "
                f"Choose from ['exp','rational1','rational2','sqrt_rational2','log1p','power_family','stretched_exp']")

    if soft:
        weights = compute_weights(eigvals)  # <— 关键替换点
        # print(weights)

        # --- 归一化熵缩放，与原版保持一致 ---
        # prob = eigvals / eigvals.sum()
        # entropy = -torch.sum((prob + 1e-7) * torch.log(prob + 1e-7))
        # max_entropy = math.log(d)
        # normalized_entropy = (max_entropy - entropy) / max_entropy

        # total = eigvals.sum()
        # cumsum = torch.cumsum(eigvals, dim=0)
        # ratio = cumsum / (total + 1e-12)
        # idx = (ratio >= 0.05).nonzero(as_tuple=False)
        # m = idx[0].item() if idx.numel() > 0 else eigvals.numel()
        # normalized_entropy = m / d
        # print(normalized_entropy)
        diag_w = torch.diag(weights) * k #* normalized_entropy
        P = eigvecs @ diag_w @ eigvecs.t()

    else:
        eps_hard = nsp_eps
        total = eigvals.sum()
        cumsum = torch.cumsum(eigvals, dim=0)
        ratio = cumsum / (total + 1e-12)
        idx = (ratio >= eps_hard).nonzero(as_tuple=False)
        m = idx[0].item() if idx.numel() > 0 else eigvals.numel()
        V_keep = eigvecs[:, :m]
        P = V_keep @ V_keep.t()
        I = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
        P = (1 - nsp_weight) * P + nsp_weight * I
    return P

# # %%
# import timm
# vit = timm.create_model('vit_base_patch16_224', pretrained=False)
# osgp_vit = OSGPLoRAViT(vit, 4, None, True, True, 1, 0.1, 1)


# covariances = {}
# for name, mod in osgp_vit.lora_modules.items():
#     x = torch.randn([768, 1000])
#     cov = x @ x.t() + torch.eye(768)
#     covariances[name] = cov

# %%
