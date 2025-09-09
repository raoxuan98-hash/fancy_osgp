# ==============================================================
#  Plain LoRA-ViT  —— 仅 A、B 参数（无投影 P）
#  依赖：torch, timm
# ==============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Dict
from timm.models.vision_transformer import VisionTransformer as timm_ViT


# -----------------------------
# 基础 LoRA 适配器（线性层）
# -----------------------------
class LoRALinear(nn.Module):
    """
    标准 LoRA 线性适配器：W <- W + scale * (B @ A)
      - A: (r, in_features)
      - B: (out_features, r)
      - 两段式前向：h = x @ A^T;  y = h @ B^T;  out = orig + scale * y
    """
    def __init__(self, linear: nn.Linear, r: int, alpha: Optional[float] = None):
        super().__init__()
        assert r > 0
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r

        # LoRA 缩放：若给定 alpha，则 scale = alpha / r；否则为 1.0
        scale_val = float(alpha) / float(r) if alpha is not None else 1.0
        self.register_buffer("scale", torch.tensor(scale_val), persistent=False)

        # LoRA 参数
        self.A = nn.Parameter(torch.zeros(r, self.in_features, dtype=linear.weight.dtype, device=linear.weight.device))
        self.B = nn.Parameter(torch.zeros(self.out_features, r, dtype=linear.weight.dtype, device=linear.weight.device))

        # 常见做法：A Kaiming 初始化，B 全零，确保初始等价于恒等
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # 冻结原始线性层参数（典型 LoRA 训练设置）
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.linear(x)
        h = F.linear(x, self.A)                 # (..., r)
        lora = F.linear(h, self.B) * self.scale # (..., out_features)
        return orig + lora

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        """
        将 LoRA 权重合并进原始层：W <- W + scale * (B @ A)
        合并后将 B 置零（保持可重复合并的幂等性）。
        """
        delta = (self.B @ self.A) * self.scale  # (out, in)
        self.linear.weight.add_(delta.to(self.linear.weight.dtype))
        self.B.zero_()

    @torch.no_grad()
    def reset_parameters_svd(self) -> None:
        """
        用原权重的右奇异向量初始化 A；B 置零。
        """
        W = self.linear.weight
        # torch.linalg.svd 返回 U, S, Vh；Vh 形状 (out,in)→(in,in) 取前 r 行
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        self.A.copy_(Vh[: self.r, :])
        self.B.zero_()


# -----------------------------
# QKV 专用 LoRA 适配器
# -----------------------------
class LoRAQKV(nn.Module):
    """
    timm ViT 中的 qkv 是一个 Linear(in=dim, out=3*dim)。
    这里做与 LoRALinear 相同的增量：W_qkv <- W_qkv + scale * (B @ A)
      - A: (r, dim)
      - B: (3*dim, r)
    """
    def __init__(self, qkv: nn.Linear, r: int, alpha: Optional[float] = None):
        super().__init__()
        assert r > 0
        self.qkv = qkv
        self.dim = qkv.in_features
        assert qkv.out_features % 3 == 0 and qkv.out_features == 3 * self.dim, \
            "Expect qkv.out_features == 3 * qkv.in_features for ViT."

        scale_val = float(alpha) / float(r) if alpha is not None else 1.0
        self.register_buffer("scale", torch.tensor(scale_val), persistent=False)
        self.r = r

        self.A = nn.Parameter(torch.zeros(r, self.dim, dtype=qkv.weight.dtype, device=qkv.weight.device))
        self.B = nn.Parameter(torch.zeros(3 * self.dim, r, dtype=qkv.weight.dtype, device=qkv.weight.device))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.qkv.weight.requires_grad_(False)
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.qkv(x)
        h = F.linear(x, self.A)                # (..., r)
        lora = F.linear(h, self.B) * self.scale
        return orig + lora

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        delta = (self.B @ self.A) * self.scale  # (3*dim, dim)
        self.qkv.weight.add_(delta.to(self.qkv.weight.dtype))
        self.B.zero_()

    @torch.no_grad()
    def reset_parameters_svd(self) -> None:
        W = self.qkv.weight
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        self.A.copy_(Vh[: self.r, :])
        self.B.zero_()


# -----------------------------
# 主包装器：PlainLoRAViT
# -----------------------------
class PlainLoRAViT(nn.Module):
    """
    仅含 LoRA(A,B) 的 ViT 包装器：
      - 将每个指定 block 的 attn.qkv 与 mlp.fc1（及可选 fc2）替换为 LoRA 版本
      - 冻结原始 ViT 参数，仅训练 LoRA 的 A、B
      - 支持 SVD 初始化、权重合并、最终去 LoRA 化
    """
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,  # 需要注入的 block 索引列表；默认全体
        alpha: Optional[float] = None,               # LoRA scale = alpha / r（默认 1.0）
    ):
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"

        self.r = r
        self.alpha = alpha

        # 默认所有 block
        self.lora_layer = (
            list(lora_layer) if lora_layer is not None
            else list(range(len(vit_model.blocks)))
        )

        # 冻结 ViT 原始参数
        for p in vit_model.parameters():
            p.requires_grad = False

        # 替换 qkv / fc1(/fc2) 为 LoRA 版本
        self.lora_modules = nn.ModuleDict()
        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer:
                continue

            # --- QKV ---
            qkv_adapter = LoRAQKV(blk.attn.qkv, r=self.r, alpha=self.alpha)
            blk.attn.qkv = qkv_adapter
            self.lora_modules[f"block_{idx}_attn_qkv"] = qkv_adapter

            # --- MLP fc1 ---
            fc1_adapter = LoRALinear(blk.mlp.fc1, r=self.r, alpha=self.alpha)
            blk.mlp.fc1 = fc1_adapter
            self.lora_modules[f"block_{idx}_mlp_fc1"] = fc1_adapter


            # # --- MLP fc2 ---
            # fc2_adapter = LoRALinear(blk.mlp.fc2, r=self.r, alpha=self.alpha)
            # blk.mlp.fc2 = fc2_adapter
            # self.lora_modules[f"block_{idx}_mlp_fc2"] = fc2_adapter

        self.vit = vit_model

        # SVD 初始化（A 取 Vh 前 r 行，B 置零）
        self.reset_parameters_svd()
        self.feature_dim = vit_model.embed_dim
        self.optimizable = False
        self.use_projection = False

    # ---------- 训练/推理 ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    # ---------- 工具方法 ----------
    @torch.no_grad()
    def reset_parameters_svd(self) -> None:
        for name, mod in self.lora_modules.items():
            mod.reset_parameters_svd()

    def get_module_names(self):
        return list(self.lora_modules.keys())

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        for _, mod in self.lora_modules.items():
            yield mod.A
            yield mod.B

    def kl_regularization(self) -> torch.Tensor:
        # 普通 LoRA 不需要 KL 正则
        return torch.tensor(0.0, device=next(self.parameters()).device)

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        """
        将所有 LoRA 增量合并进原模型权重。
        """
        self.eval()
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights()

    @torch.no_grad()
    def finalize_without_lora(self) -> None:
        """
        1) 先合并 LoRA 权重；
        2) 再把适配器替换回原生 nn.Linear，得到“无 LoRA”的 ViT。
        使用后 self.lora_modules 将被清空。
        """
        self.merge_lora_weights()

        for idx, blk in enumerate(self.vit.blocks):
            # qkv
            name_qkv = f"block_{idx}_attn_qkv"
            if name_qkv in self.lora_modules:
                adapter: LoRAQKV = self.lora_modules[name_qkv]
                blk.attn.qkv = adapter.qkv  # 合并后的原生 Linear

            # fc1
            name_fc1 = f"block_{idx}_mlp_fc1"
            if name_fc1 in self.lora_modules:
                adapter: LoRALinear = self.lora_modules[name_fc1]
                blk.mlp.fc1 = adapter.linear

            # # fc2
            # name_fc2 = f"block_{idx}_mlp_fc2"
            # if name_fc2 in self.lora_modules:
            #     adapter: LoRALinear = self.lora_modules[name_fc2]
            #     blk.mlp.fc2 = adapter.linear

        self.lora_modules = nn.ModuleDict()  # 清空


# -----------------------------
# 用法示例（伪代码）
# -----------------------------
if __name__ == "__main__":
    # 假设已有 timm ViT
    # from timm import create_model
    # vit = create_model("vit_base_patch16_224", pretrained=True)
    # 这里直接假设 vit 存在：
    vit: timm_ViT = ...  # 请替换为实际模型

    # 在所有 blocks 上对 qkv 与 mlp.fc1 注入 LoRA；rank=8；alpha=16；也给 fc2 注入
    lora_vit = PlainLoRAViT(vit, r=8, lora_layer=None, alpha=16, include_mlp_fc2=False)

    # 只优化 LoRA 参数
    optimizer = torch.optim.AdamW(lora_vit.lora_parameters(), lr=1e-4, weight_decay=0.0)

    # 训练若干步后，若需要将 LoRA 合并回模型并移除适配器：
    # lora_vit.finalize_without_lora()
