# ==============================================================
#  SGP‑LoRA (Static Gaussian Projection LoRA) — QKV + fc1 + fc2 全部 LoRA
# ==============================================================
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


# ========= 线性适配器（SGP） =========
class SGPLinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, proj: nn.Module):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.linear(x)
        A_eff = self.A @ self.P()
        h = F.linear(x, A_eff)
        lora_update = F.linear(h, self.B)  # 无 scale
        return orig_out + lora_update

    def merge_lora_weights(self) -> None:
        with torch.no_grad():
            delta = self.B @ self.A @ self.P()
            self.linear.weight += delta.to(self.linear.weight.device)
            self.B.zero_()


# ========= QKV 适配器（SGP） =========
class SGPQKV(nn.Module):
    def __init__(self, qkv: nn.Linear, r: int, proj: nn.Module):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r
        self.P = proj
        self.A = nn.Parameter(torch.zeros(r, self.dim))
        self.B = nn.Parameter(torch.zeros(3 * self.dim, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_qkv = self.qkv(x)
        A_eff = self.A @ self.P()
        h = F.linear(x, A_eff)
        lora_update = F.linear(h, self.B)
        return orig_qkv + lora_update

    def merge_lora_weights(self) -> None:
        with torch.no_grad():
            delta = self.B @ self.A @ self.P()
            self.qkv.weight += delta.to(self.qkv.weight.device)
            self.B.zero_()


class BaseLoRAViT(nn.Module):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        qkv_adapter_cls: type = SGPQKV,
        linear_adapter_cls: type = SGPLinear,
    ):
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.r = r
        try:
            self.feature_dim = vit_model.embed_dim * 2
        except:
            self.feature_dim = 768 * 2
        self.use_projection = True

        self.lora_layer = (list(lora_layer) if lora_layer is not None else list(range(len(vit_model.blocks))))

        for p in vit_model.parameters():
            p.requires_grad = False

        self.lora_modules = nn.ModuleDict()

        dev = vit_model.patch_embed.proj.weight.device

        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer:
                continue

            # ---- 缓存各层参数（在替换前）----
            qkv_in = blk.attn.qkv.in_features
            qkv_dtype = blk.attn.qkv.weight.dtype

            fc1_in = blk.mlp.fc1.in_features
            fc1_dtype = blk.mlp.fc1.weight.dtype

<<<<<<< HEAD
            fc2_in = blk.mlp.fc2.in_features
            fc2_dtype = blk.mlp.fc2.weight.dtype

            # 根据工厂决定占位投影
            def make_placeholder(d, dtype):
                if placeholder_proj_factory is not None:
                    return placeholder_proj_factory(d, dev, dtype)
                else:
                    return FixedProjection(torch.eye(d, device=dev, dtype=dtype))
=======
            fc2_in = blk.mlp.fc2.in_features   # ← 启用 fc2
            fc2_dtype = blk.mlp.fc2.weight.dtype
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601

            # ---- QKV ----
            qkv_proj = FixedProjection(torch.eye(qkv_in, device=dev, dtype=qkv_dtype))
            new_qkv = qkv_adapter_cls(blk.attn.qkv, r, qkv_proj)
            blk.attn.qkv = new_qkv
            self.lora_modules[f"block_{idx}_attn_qkv"] = new_qkv

            # ---- MLP fc1 ----
            fc1_proj = FixedProjection(torch.eye(fc1_in, device=dev, dtype=fc1_dtype))
            new_fc1 = linear_adapter_cls(blk.mlp.fc1, r, fc1_proj)
            blk.mlp.fc1 = new_fc1
            self.lora_modules[f"block_{idx}_mlp_fc1"] = new_fc1

<<<<<<< HEAD
            # ---- MLP fc1 ----
            fc2_proj = make_placeholder(fc2_in, fc2_dtype)
            new_fc2 = linear_adapter_cls(blk.mlp.fc2, r, fc2_proj)
            blk.mlp.fc2 = new_fc2
            self.lora_modules[f"block_{idx}_mlp_fc2"] = new_fc2

=======
            # ---- MLP fc2 ---- ← 启用并注册
            fc2_proj = FixedProjection(torch.eye(fc2_in, device=dev, dtype=fc2_dtype))
            new_fc2 = linear_adapter_cls(blk.mlp.fc2, r, fc2_proj)
            blk.mlp.fc2 = new_fc2
            self.lora_modules[f"block_{idx}_mlp_fc2"] = new_fc2
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601

        self.lora_vit = vit_model
        self.reset_parameters_svd()

    def reset_parameters_svd(self) -> None:
<<<<<<< HEAD
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
=======
        for name, module in self.lora_modules.items():
            if isinstance(module, (SGPQKV, SGPLinear)):
                if "qkv" in name:
                    W = module.qkv.weight
                elif "fc1" in name or "fc2" in name:
                    W = module.linear.weight
                else:
                    continue
                _, _, Vh = torch.linalg.svd(W, full_matrices=False)
                module.A.data = Vh[: self.r, :].clone()
                module.B.data.zero_()
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def kl_regularization(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lora_vit.patch_embed(x)
        if self.lora_vit.cls_token is not None:
            x = torch.cat((self.lora_vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.lora_vit.pos_drop(x + self.lora_vit.pos_embed)

        cls_outputs = []
        for idx, blk in enumerate(self.lora_vit.blocks):
            x = blk(x)
            if idx >= len(self.lora_vit.blocks) - 2:
                cls_token = x[:, 0, :]
                cls_outputs.append(cls_token)

        output = torch.cat(cls_outputs, dim=-1)
        return output

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

<<<<<<< HEAD
                # ---- MLP fc2 ----
=======
                # ---- MLP fc2 ---- ← 恢复原始层
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601
                name_fc2 = f"block_{idx}_mlp_fc2"
                if name_fc2 in self.lora_modules:
                    adapter = self.lora_modules[name_fc2]
                    blk.mlp.fc2 = adapter.linear

            self.lora_modules = nn.ModuleDict()


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
            linear_adapter_cls=SGPLinear)
        
        self.proj_temp = proj_temp
        self.use_soft_projection = use_soft_projection
        self.k = k
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


# ------------------------------------------------------------------
#  SGP projection builder
# ------------------------------------------------------------------
def build_projection(
    cov: torch.Tensor,
    soft: bool = True,
    temp: float = 5.0,
    k: float = 0.5,
    nsp_eps = 0.05, 
    nsp_weight = 0.0,
    *,

    weight_kind: str = "stretched_exp",
    weight_alpha: float = 0.5,
    weight_p: float = 2.0,
    weight_kappa: float = 2
) -> torch.Tensor:
    eps = 1e-6
    cov = cov + eps * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.abs(eigvals)
    d = cov.size(0)
    sum_vals = eigvals.sum()
    scale_ = d / (sum_vals + eps)
    eigvals = eigvals * scale_

    def compute_weights(x: torch.Tensor) -> torch.Tensor:
        beta = temp
        if weight_kind == "exp":
            return torch.exp(-beta * x)
        elif weight_kind == "rational1":
            return 1.0 / (1.0 + beta * x)
        elif weight_kind == "rational2":
            return 1.0 / (1.0 + beta * (x ** 2))
        elif weight_kind == "sqrt_rational2":
            return 1.0 / torch.sqrt(1.0 + beta * (x ** 2))
        elif weight_kind == "log1p":
            return 1.0 / (1.0 + beta * torch.log1p(x**weight_p))
        elif weight_kind == "power_family":
            return (1.0 + beta * (x ** weight_p)) ** (-weight_alpha)
        elif weight_kind == "stretched_exp":
            return torch.exp(- (beta * x) ** weight_kappa)
        else:
            raise ValueError(
                f"Unknown weight_kind='{weight_kind}'. "
                f"Choose from ['exp','rational1','rational2','sqrt_rational2','log1p','power_family','stretched_exp']")

    if soft:
<<<<<<< HEAD
        weights = compute_weights(eigvals)  # <— 关键替换点
        # print(weights)

        # --- 归一化熵缩放，与原版保持一致 ---
=======
        weights = compute_weights(eigvals)
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601
        # prob = eigvals / eigvals.sum()
        # entropy = -torch.sum((prob + 1e-7) * torch.log(prob + 1e-7))
        # max_entropy = math.log(d)
        # normalized_entropy = (max_entropy - entropy) / max_entropy
<<<<<<< HEAD

        # total = eigvals.sum()
        # cumsum = torch.cumsum(eigvals, dim=0)
        # ratio = cumsum / (total + 1e-12)
        # idx = (ratio >= 0.05).nonzero(as_tuple=False)
        # m = idx[0].item() if idx.numel() > 0 else eigvals.numel()
        # normalized_entropy = m / d
        # print(normalized_entropy)
=======
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601
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
# %%
