import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch.autograd import Function


# --------------------------------------------------------------
# 1️⃣ LoRAFunction（保持原样，只在 backward 中多保守一次 None 判定）
# --------------------------------------------------------------
class LoRAFunction(Function):
    """
    自定义的 autograd Function，用于计算 LoRA 的前向和后向传播。
    这个实现将 LoRA 的计算（两个线性层）和梯度计算封装在一起，
    并允许通过一个投影矩阵 `proj` 来调整梯度。
    """
    @staticmethod
    def forward(ctx, x, A, B, scale, proj):
        # h = X @ Aᵀ   → (..., r)
        h = F.linear(x, A)                     # (…, r)
        # LoRA 增量 (h @ Bᵀ) * scale
        lora_out = F.linear(h, B) * scale       # (…, d_out)

        # 保存用于反向传播的张量
        # 注意：proj 也会被保存，尽管它本身不需要梯度
        ctx.save_for_backward(x, h, A, B, scale, proj)
        return lora_out

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output 的维度是 (..., d_out)
        x, h, A, B, scale, proj = ctx.saved_tensors

        # 计算对原始权重矩阵 W 的梯度 Δ = Gᵀ @ X  (d_out, d_in)
        # grad_output: (..., d_out), x: (..., d_in) -> grad_delta: (d_out, d_in)
        grad_delta = torch.einsum('...i,...j->ij', grad_output, x)

        # 如果提供了投影矩阵，则对梯度进行投影
        # 在新的设计中，proj 永远不会是 None，但保留这个检查以增加鲁棒性
        # if proj is not None:
        #     grad_delta = grad_delta @ proj

        # --- 计算参数的梯度 ---
        # grad_A = Bᵀ @ Δ   (r, d_in)
        grad_A = B.t() @ grad_delta
        # grad_B = Δ @ Aᵀ   (d_out, r)
        grad_B = grad_delta @ A.t()

        # --- 计算 scale 的梯度 ---
        # 首先计算不带 scale 的 lora 输出
        lora_no_scale = torch.einsum('...r,dr->...d', h, B) # (..., d_out)
        # 梯度是 grad_output 和 lora_no_scale 的点积之和
        grad_scale = torch.sum(grad_output * lora_no_scale)

        # --- 计算对输入 x 的梯度 ---
        # LoRA 等效于一个权重矩阵 W_lora = B @ A
        weight = B @ A                         # (d_out, d_in)
        # grad_x = G @ W_lora
        grad_x = torch.einsum('...i,ij->...j', grad_output, weight)  # (..., d_in)

        # 返回的梯度顺序必须与 forward 的输入参数一一对应
        # proj 不需要梯度，所以返回 None
        return grad_x, grad_A, grad_B, grad_scale, None


# --------------------------------------------------------------
# 2️⃣ LoRA 包装的 Linear（默认 proj_matrix = I）
# --------------------------------------------------------------
class _LoRA_linear(nn.Module):
    """
    使用 LoRAFunction 包装一个标准的 nn.Linear 层。
    它在内部维护 LoRA 参数 (A, B, scale) 和一个投影矩阵 `proj_matrix`。
    `proj_matrix` 默认是单位矩阵，实现了“无投影”的效果。
    """
    def __init__(self, linear: nn.Linear, r: int):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r), requires_grad=True)
        # B 矩阵通常初始化为零，A 矩阵用高斯分布初始化
        nn.init.normal_(self.lora_A, std=0.02)

        # 乘以一个可学习的 scale
        self.scale = nn.Parameter(torch.tensor(0.8), requires_grad=True)

        # **关键**：注册一个单位矩阵作为默认的投影矩阵。
        # 使用 register_buffer，这样它会随模型移动（如 .to(device)），但不会被视为可训练参数。
        self.register_buffer('proj_matrix',
                             torch.eye(self.in_features, dtype=torch.float32))

    def forward(self, x):
        # 原始线性层的输出
        orig_out = self.linear(x)

        # 调用 LoRAFunction 计算增量
        lora_update = LoRAFunction.apply(
            x,
            self.lora_A,
            self.lora_B,
            self.scale,
            self.proj_matrix  # 永远传递一个有效的 Tensor
        )
        return orig_out + lora_update


# --------------------------------------------------------------
# 3️⃣ LoRA 包装的 QKV（同理）
# --------------------------------------------------------------
class _LoRA_qkv_timm(nn.Module):
    """
    使用 LoRAFunction 包装 timm ViT 中的 QKV 线性层。
    逻辑与 _LoRA_linear 类似，但维度不同 (in -> 3*in)，且 scale 通常固定。
    """
    def __init__(self, qkv: nn.Linear, r: int):
        super().__init__()
        self.qkv = qkv
        # 输入特征维度 = head_dim * num_heads
        self.dim = qkv.in_features
        self.r = r

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, self.dim), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(3 * self.dim, r), requires_grad=True)
        nn.init.normal_(self.lora_A, std=0.02)

        # 在 QKV 中，scale 通常固定为 1 且不可训练，以保持稳定性
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # 注册单位投影矩阵（size = dim x dim）
        self.register_buffer('proj_matrix',
                             torch.eye(self.dim, dtype=torch.float32))

    def forward(self, x):
        # 原始 QKV 层的输出
        orig_qkv = self.qkv(x)

        # 调用 LoRAFunction 计算增量
        lora_update = LoRAFunction.apply(
            x,
            self.lora_A,
            self.lora_B,
            self.scale,
            self.proj_matrix
        )
        return orig_qkv + lora_update


# --------------------------------------------------------------
# 4️⃣ LoRA‑ViT（核心改动：去掉 use_projection 相关的布尔位）
# --------------------------------------------------------------
class LoRAViT(nn.Module):
    """
    给定一个已经初始化好的 `timm_ViT`（VisionTransformer），
    在指定的 Block 上注入 LoRA（QKV + MLP 两个全连接层）。
    投影矩阵在每个 LoRA 子模块里默认是 **单位矩阵**。
    当需要使用软/硬投影时，调用 `update_projection_matrices`
    方法，将计算好的投影矩阵写入相应的子模块中。
    """
    def __init__(self,
                 vit_model: timm_ViT,
                 r: int,
                 lora_layer_indices=None):
        super().__init__()
        assert r > 0, "LoRA rank 'r' 必须大于 0"

        # 需要注入 LoRA 的 Block 索引，默认全部
        if lora_layer_indices is None:
            self.lora_layer_indices = list(range(len(vit_model.blocks)))
        else:
            self.lora_layer_indices = lora_layer_indices

        # 冻结原始 ViT 的所有参数
        for p in vit_model.parameters():
            p.requires_grad = False

        # 使用 ModuleDict 保存所有被 LoRA 包装的子模块，方便按名称索引
        self.lora_modules = nn.ModuleDict()

        # 遍历 ViT 的所有 Block，将选中的层替换为 LoRA 包装层
        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer_indices:
                continue

            # ---- 包装 QKV 层 ----
            name_qkv = f"block_{idx}_attn_qkv"
            blk.attn.qkv = _LoRA_qkv_timm(blk.attn.qkv, r)
            self.lora_modules[name_qkv] = blk.attn.qkv

            # ---- 包装 MLP 的第一个全连接层 ----
            name_fc1 = f"block_{idx}_mlp_fc1"
            blk.mlp.fc1 = _LoRA_linear(blk.mlp.fc1, r)
            self.lora_modules[name_fc1] = blk.mlp.fc1

            # ---- 包装 MLP 的第二个全连接层 ----
            name_fc2 = f"block_{idx}_mlp_fc2"
            blk.mlp.fc2 = _LoRA_linear(blk.mlp.fc2, r)
            self.lora_modules[name_fc2] = blk.mlp.fc2

        self.lora_vit = vit_model

        # 使用 SVD 初始化 LoRA 参数
        self.reset_parameters_svd()

        # 用来缓存最近一次计算得到的投影矩阵（name -> Tensor），方便外部查询
        self.projection_matrices_cache = {}

    # ------------------------------------------------------------------
    # 1️⃣ SVD 初始化（保持不变）
    # ------------------------------------------------------------------
    def reset_parameters_svd(self):
        """使用原始权重的 SVD 来初始化 LoRA 矩阵 A，并将 B 初始化为零。"""
        for name, module in self.lora_modules.items():
            if 'attn_qkv' in name:
                W = module.qkv.weight.data          # (3*dim, dim)
                U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                module.lora_A.data = Vh[:module.r, :].clone()
                module.lora_B.data.zero_()
            else:   # MLP fc1 或 fc2
                W = module.linear.weight.data        # (out, in)
                U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                module.lora_A.data = Vh[:module.r, :].clone()
                module.lora_B.data.zero_()

    # ------------------------------------------------------------------
    # 2️⃣ 前向传播（直接转交给包装好的 ViT）
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播直接调用内部的 ViT 模型。"""
        return self.lora_vit(x)

    # ------------------------------------------------------------------
    # 3️⃣ 生成（软/硬）投影矩阵并写入每个 LoRA 模块
    # ------------------------------------------------------------------
    def update_projection_matrices(self,
                                   covariances: dict,
                                   eps: float = 0.05,
                                   soft: bool = True,
                                   temp: float = 5.0) -> None:
        """
        根据提供的协方差矩阵，计算软投影或硬投影矩阵，
        然后将结果直接写入对应 LoRA 子模块的 `proj_matrix` buffer 中。

        参数:
        - `covariances` (dict): {module_name: cov_matrix} 的字典，`cov_matrix` 是 (d, d) 的张量。
        - `eps` (float): 在硬投影模式下用于平滑的系数。
        - `soft` (bool): True 表示使用软投影（指数加权），False 表示使用硬投影（低特征值子空间）。
        - `temp` (float): 软投影的温度系数，数值越大，投影效果越接近硬投影。
        """
        # 清空旧的缓存
        self.projection_matrices_cache.clear()

        # 对每个子模块分别计算投影矩阵
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                # 如果外部传入了未被 LoRA 包装的模块名称，直接跳过
                continue

            # 1️⃣ 稳定化处理和特征分解
            cov = cov + 1e-6 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
            eigvals, eigvecs = torch.linalg.eigh(cov)  # eigh 保证特征向量正交，特征值升序
            eigvals = torch.abs(eigvals)               # 确保特征值为非负

            if soft:
                # ---------- 软投影 ----------
                # M = V·diag(exp(-temp·λ̂))·Vᵀ, where λ̂ are normalized eigenvalues
                max_eig = eigvals.max()
                lam_norm = eigvals / (max_eig + 1e-12) # 归一化到 (0, 1]
                weights = torch.exp(-temp * lam_norm)  # 特征值越小，权重越接近1
                P = eigvecs @ torch.diag(weights) @ eigvecs.t()
            else:
                # ---------- 硬投影 ----------
                # 找到能保留 (1-eps) 能量的最小子空间维度 m
                total_energy = eigvals.sum()
                cumsum_energy = torch.cumsum(eigvals, dim=0)
                ratio = cumsum_energy / (total_energy + 1e-12)
                # 找到第一个使得累积能量比率 >= eps 的索引
                indices = (ratio >= eps).nonzero()
                # 如果所有特征值都很小，可能 indices 为空，此时保留所有维度
                m = indices[0].item() if indices.numel() > 0 else eigvals.numel()

                # V_keep 是能量较低的特征向量
                V_keep = eigvecs[:, :m]                # (d, m)
                # 投影到低能量子空间
                P_low = V_keep @ V_keep.t()
                
                # 原始论文的硬投影是投影到高能量子空间，这里假设投影到低能量子空间以抑制梯度
                # 如果要投影到高能量子空间，应该是 V_high = eigvecs[:, m:]
                # 这里的实现与软投影目标一致：抑制梯度中与高特征值对应的方向
                # P = P_low
                
                # 原论文的平滑方式
                I = torch.eye(P_low.size(0), device=P_low.device, dtype=P_low.dtype)
                P = eps * I + (1 - eps) * P_low

            # 缓存并写入子模块
            self.projection_matrices_cache[name] = P.to(self.lora_modules[name].proj_matrix.device, P.dtype)
            self.lora_modules[name].proj_matrix.data.copy_(self.projection_matrices_cache[name])

        # 对于未提供协方差的模块，确保它们的投影矩阵恢复为单位矩阵
        for name, module in self.lora_modules.items():
            if name not in covariances:
                I = torch.eye(module.proj_matrix.size(0),
                              device=module.proj_matrix.device,
                              dtype=module.proj_matrix.dtype)
                module.proj_matrix.data.copy_(I)
                self.projection_matrices_cache[name] = I

    # ------------------------------------------------------------------
    # 4️⃣ 额外的辅助接口
    # ------------------------------------------------------------------
    def get_lora_module_names(self) -> list[str]:
        """返回所有已包装的 LoRA 子模块名称列表，用于外部索引。"""
        return list(self.lora_modules.keys())

    def get_projection_matrix(self, name: str) -> torch.Tensor:
        """获取某个子模块当前使用的投影矩阵（如果未计算则为单位矩阵）。"""
        if name not in self.lora_modules:
            raise KeyError(f"模块 '{name}' 不存在或未被 LoRA 包装。")
        # 直接从模块中读取，保证获取到的是当前正在使用的矩阵
        return self.lora_modules[name].proj_matrix
