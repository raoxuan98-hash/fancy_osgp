
import argparse
import os
from trainer_clip import train


# --------------------------------------------------------------
# 2️⃣  主入口
# --------------------------------------------------------------
def main(args):
    """把已经解析好的 ``Namespace`` 交给 trainer。"""
    train(args)


# --------------------------------------------------------------
# 3️⃣  参数解析（按功能分组、每条 add_argument 为单行）
# --------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='SGP-LoRA experiments unified management (grouped arguments).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ------------------------------------------------------------------
    # 基础选项
    # ------------------------------------------------------------------
    basic = parser.add_argument_group('basic', 'General / high‑level options')
    basic.add_argument('--smart_defaults', action='store_true', default=False, help='If set, overwrite a few hyper‑parameters according to the dataset.')
    basic.add_argument('--dataset', type=str, default='CLIP-CL')
    basic.add_argument('--prefix', type=str, default='original', help='Prefix for output folders / logs.')
    basic.add_argument('--test', action=argparse.BooleanOptionalAction, default=False, help='Run in test mode (single seed, quick).')
    basic.add_argument('--user', type=str, default='raoxuan', choices=['null'], help='User identifier (currently unused).')
    basic.add_argument('--data_location', type=str, default="/home/raoxuan/projects/SubspaceLoRA/data/mtil")
    basic.add_argument('--reference_dataset_path', type=str, default='/data1/open_datasets/ImageNet-2012/train/')
    basic.add_argument('--reference_batch_size', type=int, default=24)
    
    # ------------------------------------------------------------------
    # Memory 参数
    # ------------------------------------------------------------------
    mem = parser.add_argument_group('memory', 'Memory / replay buffer')
    mem.add_argument('--memory_size', type=int, default=0, help='Total memory budget.')
    mem.add_argument('--memory_per_class', type=int, default=0, help='Memory allocated per class.')
    mem.add_argument('--fixed_memory', action='store_true', default=False, help='If set, memory size does not grow with new classes.')
    mem.add_argument('--shuffle', action='store_true', default=True, help='Shuffle replay buffer before each epoch.')

    # ------------------------------------------------------------------
    # 类别增量参数
    # ------------------------------------------------------------------
    cls = parser.add_argument_group('class', 'Class increment settings')
    cls.add_argument('--init_cls', type=int, default=20, help='Number of classes in the first task.')
    cls.add_argument('--increment', type=int, default=20, help='Number of new classes added per task.')

    # ------------------------------------------------------------------
    # 模型相关参数
    # ------------------------------------------------------------------
    model = parser.add_argument_group('model', 'Backbone & LoRA settings')
    model.add_argument('--model_name', type=str, default='sldc', help='Model identifier.')
    model.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    model.add_argument('--device', nargs='+', default=['0'], help='CUDA device ids, e.g. --device 0 1 2')
    model.add_argument('--vit_type', type=str, default='clip-vit-b-16')
    
    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    model.add_argument('--lora_rank', type=int, default=8, help='LoRA rank.')
    model.add_argument('--lora_type', type=str, default="sgp_lora", choices=['basic_lora', 'osgp_lora', 'sgp_lora', 'nsp_lora', 'full'], help='Type of LoRA adaptor.')
    model.add_argument('--weight_temp', type=float, default=4, help='Projection temperature.')

    # NSP相关的参数
    model.add_argument('--nsp_eps', type=float, default=0.05, choices=[0.05, 0.10])
    model.add_argument('--nsp_weight', type=float, default=0.0, choices=[0.0, 0.02, 0.05])
    
    # SGP相关的参数
    model.add_argument('--trace_k', type=float, default=0.2 , help='Flag for osgp_lora.')
    model.add_argument('--weight_kind', type=str, default='log1p', choices=["exp", "log1p", "rational1", "rational2", "sqrt_rational2", "power_family", "stretched_exp"])
    model.add_argument('--weight_p', type=float, default=1.0, help='Weight p.')

    # OSGP相关的参数
    model.add_argument('--kl_gamma', type=float, default=1e-2, help='KL‑div weight for projection.')
    model.add_argument('--osgp_scale', type=float, default=1.0, help='Scale factor when using osgp_lora.')

    # ------------------------------------------------------------------
    # 训练相关参数
    # ------------------------------------------------------------------
    train_grp = parser.add_argument_group('training', 'Optimisation & schedule')
    train_grp.add_argument('--sce_a', type=float, default=0.5, help='Symmetric cross‑entropy A.')
    train_grp.add_argument('--sce_b', type=float, default=0.5, help='Symmetric cross‑entropy B.')
    train_grp.add_argument('--seed_list', nargs='+', type=int, default=[1993, 1996, 1997], help='Random seeds for multiple runs.')
    train_grp.add_argument('--iterations', type=int, default=50, help='Training iterations per task.')
    train_grp.add_argument('--warmup_steps', type=int, default=100, help='Warm‑up steps.')
    train_grp.add_argument('--ca_epochs', type=int, default=5, help='Class‑augmentation epochs.')
    train_grp.add_argument('--optimizer', type=str, default='adamw', help='Optimizer name (adamw / sgd).')
    train_grp.add_argument('--lrate', type=float, default=1e-5, help='Learning rate.')
    train_grp.add_argument('--head_scale', type=float, default=1.0, help='Scale for the classifier head.')
    train_grp.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    train_grp.add_argument('--evaluate_final_only', action=argparse.BooleanOptionalAction, default=True)
    train_grp.add_argument('--gamma_norm', type=float, default=0.1, help='Norm regularisation weight.')
    train_grp.add_argument('--gamma_kd', type=float, default=1.0, help='Knowledge‑distillation weight.')
    train_grp.add_argument('--kd_type', type=str, default='feat', help='KD type (feat / logit).')
    train_grp.add_argument('--alpha_t', type=float, default=1.0, help='Auxiliary loss weight.')
    train_grp.add_argument('--gamma_1', type=float, default=1e-4, help='Additional regularisation weight.')
    train_grp.add_argument('--compensate', type=bool, default=True)

    # ------------------------------------------------------------------
    # 辅助数据集参数
    # ------------------------------------------------------------------
    aux = parser.add_argument_group('auxiliary', 'External / auxiliary dataset')
    aux.add_argument('--auxiliary_data_path', type=str, default='/data1/open_datasets/ImageNet-2012/train/', help='Root path of the auxiliary dataset.')
    aux.add_argument('--aux_dataset_type', type=str, default='imagenet', help='Dataset type for auxiliary data (e.g. imagenet, cifar).')
    aux.add_argument('--auxiliary_data_size', type=int, default=2048, help='Number of samples drawn from the auxiliary dataset each epoch.')

    # ------------------------------------------------------------------
    # 正则化 / L2‑Protection
    # ------------------------------------------------------------------
    reg = parser.add_argument_group('regularisation', 'Extra regularisation terms') 
    reg.add_argument('--l2_protection', action='store_true', default=True, help='Enable L2‑protection between the current and previous network.')
    reg.add_argument('--l2_protection_lambda', type=float, default=1.0, help='Weight for the L2‑protection term (higher → stronger regularisation). When `--l2_protection` is off, this will be automatically set to 0.0.')

    return parser

# In[]
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args = vars(args)
    main(args)