import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple, List
from collections import OrderedDict
from models.osgp_lora import OSGPLinear, OSGPProjection, IdentityProjection, SGPLinear, FixedProjection


class BaseLoRACLIPVisionEncoder(nn.Module):
    def __init__(
        self,
        clip: nn.Module,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        # 新增：适配器工厂 & 占位投影工厂
        linear_adapter_cls: type = SGPLinear,
        placeholder_proj_factory: Optional[callable] = None,  # (d, device, dtype) -> nn.Module
    ):
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.r = r
        
        # 获取CLIP视觉编码器的特征维度
        try:
            self.feature_dim = clip.visual.proj.shape[1]
        except Exception:
            try:
                self.feature_dim = clip.visual.output_dim
            except Exception:
                self.feature_dim = 768

        self.lora_layer = (list(lora_layer) if lora_layer is not None else 
                           list(range(len(clip.visual.transformer.resblocks))))

        # 冻结原始模型参数
        for p in clip.parameters():
            p.requires_grad = False

        self.lora_modules = nn.ModuleDict()
        dev = next(clip.parameters()).device

        blocks = clip.visual.transformer.resblocks

        def make_placeholder(d, dtype):
            if placeholder_proj_factory is not None:
                return placeholder_proj_factory(d, dev, dtype)
            else:
                return FixedProjection(torch.eye(d, device=dev, dtype=dtype))

        for idx, blk in enumerate(blocks):
            if idx not in self.lora_layer:
                continue

            # --------- 统一取出 fc1 引用与元信息 ----------
            fc1_ref = None
            try:
                # OpenAI-CLIP
                fc1_ref = ('c_fc', blk.mlp.c_fc)
            except Exception:
                # timm
                try:
                    fc1_ref = ('fc1', blk.mlp.fc1)
                except Exception:
                    # 列表式
                    try:
                        fc1_ref = (0, blk.mlp[0])
                    except Exception:
                        fc1_ref = None

            # --------- 统一取出 fc2 引用与元信息 ----------
            fc2_ref = None
            try:
                # OpenAI-CLIP
                fc2_ref = ('c_proj', blk.mlp.c_proj)
            except Exception:
                # timm
                try:
                    fc2_ref = ('fc2', blk.mlp.fc2)
                except Exception:
                    # 列表式：通常 [Linear, Act, Linear]
                    try:
                        fc2_ref = (2, blk.mlp[2])
                    except Exception:
                        fc2_ref = None

            # ---- MLP fc1 ----
            if fc1_ref is not None and isinstance(fc1_ref[1], nn.Linear):
                name, fc1 = fc1_ref
                fc1_in = fc1.in_features
                fc1_dtype = fc1.weight.dtype
                fc1_proj = make_placeholder(fc1_in, fc1_dtype)

                new_fc1 = linear_adapter_cls(fc1, r, fc1_proj)

                # 回写
                if name == 'c_fc':
                    blk.mlp.c_fc = new_fc1
                elif name == 'fc1':
                    blk.mlp.fc1 = new_fc1
                else:
                    # 列表式
                    blk.mlp[0] = new_fc1

                self.lora_modules[f"block_{idx}_mlp_fc1"] = new_fc1

            # ---- MLP fc2 ----  ← 新增部分
            if fc2_ref is not None and isinstance(fc2_ref[1], nn.Linear):
                name, fc2 = fc2_ref
                fc2_in = fc2.in_features
                fc2_dtype = fc2.weight.dtype
                fc2_proj = make_placeholder(fc2_in, fc2_dtype)

                new_fc2 = linear_adapter_cls(fc2, r, fc2_proj)

                # 回写
                if name == 'c_proj':
                    blk.mlp.c_proj = new_fc2
                elif name == 'fc2':
                    blk.mlp.fc2 = new_fc2
                else:
                    # 列表式
                    blk.mlp[2] = new_fc2

                self.lora_modules[f"block_{idx}_mlp_fc2"] = new_fc2

        self.clip = clip
        self.reset_parameters_svd()

    def reset_parameters_svd(self) -> None:
        for _, module in self.lora_modules.items():
            # SGPLinear 提供 .linear
            W = getattr(module, 'linear', None)
            if isinstance(W, nn.Linear):
                W = W.weight
            elif hasattr(module, 'qkv'):
                W = module.qkv.weight
            else:
                # 兜底：跳过无法识别权重的模块
                continue

            # 用右奇异向量初始化 A，B 置零
            try:
                _, _, Vh = torch.linalg.svd(W, full_matrices=False)
                module.A.data = Vh[: self.r, :].clone()
            except Exception:
                # 退化到随机正交初始化
                with torch.no_grad():
                    nn.init.orthogonal_(module.A)
            module.B.data.zero_()

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def kl_regularization(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

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

            # 获取transformer blocks
            try:
                blocks = self.clip.visual.transformer.resblocks
            except Exception:
                blocks = self.clip.transformer.resblocks

            for idx, blk in enumerate(blocks):
                # ---- MLP fc1 ----
                name_fc1 = f"block_{idx}_mlp_fc1"
                if name_fc1 in self.lora_modules:
                    adapter = self.lora_modules[name_fc1]
                    # 回写原始 Linear
                    if hasattr(blk.mlp, 'c_fc'):
                        blk.mlp.c_fc = adapter.linear
                    elif hasattr(blk.mlp, 'fc1'):
                        blk.mlp.fc1 = adapter.linear
                    else:
                        blk.mlp[0] = adapter.linear

                # ---- MLP fc2 ----  ← 新增回写
                name_fc2 = f"block_{idx}_mlp_fc2"
                if name_fc2 in self.lora_modules:
                    adapter = self.lora_modules[name_fc2]
                    if hasattr(blk.mlp, 'c_proj'):
                        blk.mlp.c_proj = adapter.linear
                    elif hasattr(blk.mlp, 'fc2'):
                        blk.mlp.fc2 = adapter.linear
                    else:
                        blk.mlp[2] = adapter.linear

            # 清空字典
            self.lora_modules = nn.ModuleDict()

        if hasattr(self, "optimizable"):
            self.optimizable = False


class OSGPLoRAViT_CLIP(BaseLoRACLIPVisionEncoder):
    def __init__(
        self,
        clip: nn.Module,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        proj_temp: float = 1.0,
        trace_k: float = 0.5,
        weight_p: float = 2.0,
        kl_gamma: float = 1.0,
    ):
        super().__init__(
            clip, r, lora_layer,
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

    def get_projection_learnable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for mod in self.lora_modules.values():
            if isinstance(mod.P, OSGPProjection):
                params.append(mod.P.delta)
                params.append(mod.P.raw_temp)
        return params

    def collect_vit_and_delta_params(self):
        vit_params = []
        # 只从 self.lora_clip 中取，而不是整个 self（避免把 A/B 也捞进来）
        for name, p in self.clip.named_parameters():
            if not p.requires_grad:
                continue
            # 显式排除 LoRA 模块参数与投影 delta
            if name.endswith(".A") or name.endswith(".B"):
                vit_params.append(p)

        delta_params = self.get_projection_learnable_params()
        return vit_params, delta_params


class SGPLoRAViT_CLIP(BaseLoRACLIPVisionEncoder):
    def __init__(
        self,
        clip: nn.Module,
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
            clip, r, lora_layer,
            linear_adapter_cls=SGPLinear,
            placeholder_proj_factory=lambda d, dev, dtype: FixedProjection(torch.eye(d, device=dev, dtype=dtype))
        )
        
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
    nsp_eps: float = 0.05, 
    nsp_weight: float = 0.0,
    *,
    weight_kind: str = "stretched_exp",
    weight_alpha: float = 0.5,
    weight_p: float = 2.0,
    weight_kappa: float = 2
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
        weights = compute_weights(eigvals)
        prob = eigvals / eigvals.sum()
        entropy = -torch.sum((prob + 1e-7) * torch.log(prob + 1e-7))
        max_entropy = math.log(d)
        normalized_entropy = (max_entropy - entropy) / max_entropy
        
        diag_w = torch.diag(weights) * k * normalized_entropy
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
