#  SGP-LoRA (post-projection only, no Base class)
# ==============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch.autograd import Function

class LoRAFunction(Function):
    @staticmethod
    def forward(ctx, x, A, B, scale, proj):
        h = F.linear(x, A)                 # (..., r)
        lora_out = F.linear(h, B) * scale  # (..., d_out)
        ctx.save_for_backward(x, h, A, B, scale, proj)
        return lora_out

    @staticmethod
    def backward(ctx, grad_output):
        x, h, A, B, scale, proj = ctx.saved_tensors
        # Δ = G^T @ X  → (d_out, d_in)
        # grad_output: (..., d_out), x: (..., d_in)
        grad_delta = torch.einsum('...o,...i->oi', grad_output, x)

        # —— 关键1：做“后投影” —— 
        if proj is not None:
            # 右乘 P，将梯度限制在输入子空间（post-projection）
            grad_delta = grad_delta @ proj

        # —— 关键2：对 A、B 的梯度别忘了 scale —— 
        grad_A = scale * (B.t() @ grad_delta)   # (r, d_in)
        grad_B = scale * (grad_delta @ A.t())   # (d_out, r)

        # scale 的梯度：对未乘 scale 的 LoRA 输出求内积
        lora_no_scale = torch.einsum('...r,dr->...d', h, B)   # (..., d_out)
        grad_scale = torch.sum(grad_output * lora_no_scale)

        # —— 关键3：对输入 x 的梯度也要乘 scale —— 
        weight = scale * (B @ A)                         # (d_out, d_in)
        grad_x = torch.einsum('...o,oi->...i', grad_output, weight)

        # proj 不需要梯度
        return grad_x, grad_A, grad_B, grad_scale, None



# =========================
# LoRA wrappers with post-projection
# =========================
class SGPLinearBackProj(nn.Module):
    def __init__(self, linear: nn.Linear, r: int):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r

        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)

        self.scale = nn.Parameter(torch.tensor(0.8), requires_grad=True)

        self.register_buffer(
            "proj_matrix",
            torch.eye(self.in_features, dtype=self.linear.weight.dtype,
                      device=self.linear.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + LoRAFunction.apply(x, self.A, self.B, self.scale, self.proj_matrix)

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        delta = self.scale * (self.B @ self.A)
        self.linear.weight += delta.to(self.linear.weight.dtype)
        self.B.zero_()


class SGPQKVBackProj(nn.Module):
    def __init__(self, qkv: nn.Linear, r: int):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r

        self.A = nn.Parameter(torch.zeros(r, self.dim))
        self.B = nn.Parameter(torch.zeros(3 * self.dim, r))
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)

        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.register_buffer(
            "proj_matrix",
            torch.eye(self.dim, dtype=self.qkv.weight.dtype,
                      device=self.qkv.weight.device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qkv(x) + LoRAFunction.apply(x, self.A, self.B, self.scale, self.proj_matrix)

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        delta = self.scale * (self.B @ self.A)
        self.qkv.weight += delta.to(self.qkv.weight.dtype)
        self.B.zero_()


# =========================
# SGP-LoRA wrapper for ViT
# =========================
class SGPLoRAViT(nn.Module):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        proj_temp: float = 1.0,
        proj_gamma: float = 0.5,
        use_soft_projection: bool = True,
        k: float = 0.5
    ):
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.r = r
        self.vit = vit_model
        self.proj_temp = proj_temp
        self.proj_gamma = proj_gamma
        self.use_soft_projection = use_soft_projection
        self.k = k
        self.feature_dim = vit_model.embed_dim
        self.optimizable = False

        self.lora_layer = (
            list(lora_layer) if lora_layer is not None
            else list(range(len(vit_model.blocks)))
        )
        for p in vit_model.parameters():
            p.requires_grad = False

        self.lora_modules = nn.ModuleDict()

        dev = vit_model.patch_embed.proj.weight.device

        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer:
                continue

            # QKV
            new_qkv = SGPQKVBackProj(blk.attn.qkv, r)
            blk.attn.qkv = new_qkv
            self.lora_modules[f"block_{idx}_attn_qkv"] = new_qkv

            # MLP fc1
            new_fc1 = SGPLinearBackProj(blk.mlp.fc1, r)
            blk.mlp.fc1 = new_fc1
            self.lora_modules[f"block_{idx}_mlp_fc1"] = new_fc1

        self.reset_parameters_svd()

    def reset_parameters_svd(self) -> None:
        for _, module in self.lora_modules.items():
            W = module.qkv.weight if hasattr(module, "qkv") else module.linear.weight
            _, _, Vh = torch.linalg.svd(W, full_matrices=False)
            module.A.data = Vh[: self.r, :].clone()
            module.B.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights()

    @torch.no_grad()
    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        self.merge_lora_weights()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            P = build_projection(
                cov,
                temp=self.proj_temp,
                gamma=self.proj_gamma,
                soft=self.use_soft_projection,
                k=self.k)

            mod = self.lora_modules[name]
            mod.proj_matrix.data.copy_(P.to(mod.proj_matrix.dtype).to(mod.proj_matrix.device))

            # print(1)

    def kl_regularization(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_module_names(self):
        return list(self.lora_modules.keys())

# =========================
# Projection builder
# =========================
def build_projection(
    cov: torch.Tensor,
    gamma: float = 0.5,
    soft: bool = True,
    temp: float = 5.0,
    k: float = 0.5) -> torch.Tensor:

    cov = cov + 1e-6 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.abs(eigvals)
    d = cov.size(0)
    eigvals = eigvals * (d / eigvals.sum())

    prob = torch.log(eigvals + 1e-7 + 1.0)
    prob = prob / prob.sum()
    entropy = -torch.sum((prob + 1e-7) * torch.log(prob + 1e-7))
    normalized_entropy_res = 1.0 - entropy / math.log(d)

    scale = (1 + normalized_entropy_res * gamma)

    if soft:
        weights = torch.exp(-temp * eigvals * scale)
        diag_w = torch.diag(weights) * k
        return eigvecs @ diag_w @ eigvecs.t()
    
    else:
        eps = 0.05
        cumsum = torch.cumsum(eigvals, dim=0)
        idx = (cumsum / (eigvals.sum() + 1e-12) >= eps).nonzero(as_tuple=False)
        m = idx[0].item() if idx.numel() > 0 else eigvals.numel()
        V_keep = eigvecs[:, :m]
        return V_keep @ V_keep.t()
