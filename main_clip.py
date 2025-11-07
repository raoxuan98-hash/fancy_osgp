import argparse
import os
from trainer_clip import train

os.environ['CUDA_VISIBLE_DEVICES'] = "4"

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
    basic.add_argument('--dataset', type=str, default='CLIP-CL')
    basic.add_argument('--user', type=str, default='raoxuan', choices=['null'], help='User identifier (currently unused).')
    

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
    model.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay.')
    model.add_argument('--device', nargs='+', default=['0'], help='CUDA device ids, e.g. --device 0 1 2')
    model.add_argument('--vit_type', type=str, default='clip-vit-b-16')

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    model.add_argument('--lora_rank', type=int, default=8, help='LoRA rank.')
    model.add_argument('--lora_type', type=str, default="nsp_lora", choices=['basic_lora', 'sgp_lora', 'nsp_lora', 'full'], help='Type of LoRA adaptor.')


    # NSP相关的参数
    model.add_argument('--nsp_eps', type=float, default=0.05, choices=[0.05, 0.10])
    model.add_argument('--nsp_weight', type=float, default=0.02, choices=[0.0, 0.02, 0.05])
    
    # SGP相关的参数
    model.add_argument('--weight_temp', type=float, default=1, help='Projection temperature.')
    model.add_argument('--weight_kind', type=str, default='log1p', choices=["exp", "log1p", "rational1", "rational2", "sqrt_rational2", "power_family", "stretched_exp"])
    model.add_argument('--weight_p', type=float, default=1.0, help='Weight p.')

    train_grp = parser.add_argument_group('training', 'Optimisation & schedule')
    train_grp.add_argument('--seed_list', nargs='+', type=int, default=[1993], help='Random seeds for multiple runs.')
    train_grp.add_argument('--iterations', type=int, default=800, help='Training iterations per task.')
    train_grp.add_argument('--warmup_steps', type=int, default=0, help='Warm‑up steps.')
    train_grp.add_argument('--optimizer', type=str, default='adamw', help='Optimizer name (adamw / sgd).')
    train_grp.add_argument('--lrate', type=float, default=5e-4, help='Learning rate.')
    train_grp.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    train_grp.add_argument('--gamma_norm', type=float, default=0.1, help='Norm regularisation weight.')
    train_grp.add_argument('--gamma_kd', type=float, default=5.0, help='Knowledge‑distillation weight.')
    train_grp.add_argument('--kd_type', type=str, default='feat', help='KD type (feat / logit).')
    train_grp.add_argument('--kl_gamma', type=float, default=1.0, help='KL divergence regularisation weight.')
    train_grp.add_argument('--bidirectional_kd', action='store_true', default=False, help='Enable bidirectional KL divergence for knowledge distillation.')
    train_grp.add_argument('--layerwise_kd_enabled', action='store_true', default=False, help='Enable layer-wise feature distillation.')
    train_grp.add_argument('--layerwise_kd_weight', type=float, default=1.0, help='Weight for layer-wise feature distillation.')
    train_grp.add_argument('--layerwise_kd_pooling', type=str, default='mean', choices=['mean', 'cls', 'max'], help='Pooling method for layer-wise features.')
    train_grp.add_argument('--layerwise_kd_loss_type', type=str, default='mse', choices=['mse', 'cosine', 'mse_cosine'], help='Loss type for layer-wise distillation.')
    train_grp.add_argument('--layerwise_kd_weight_strategy', type=str, default='uniform', choices=['uniform', 'linear', 'exponential'], help='Weight strategy for different layers.')
    train_grp.add_argument('--compensate', type=bool, default=True)
    train_grp.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True, help='Enable torch.cuda.amp mixed precision when CUDA is available.')
    train_grp.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16'], help='AMP compute dtype to request when mixed precision is enabled.')
    train_grp.add_argument('--debug_mode', action=argparse.BooleanOptionalAction, default=False, help='Enable debug mode to show detailed debug logs.')

    # ------------------------------------------------------------------
    # CLIP dataset sequence
    # ------------------------------------------------------------------
    clip_data = parser.add_argument_group('clip-data', 'CLIP dataset sequencing')
    clip_data.add_argument('--clip_dataset_sequence', nargs='+', default=['fgvc_aircraft', 'caltech-101', 'dtd', 'stanford_cars'], help='Dataset names (defined in utils.data1) composing the CLIP incremental tasks.')
    clip_data.add_argument('--clip_dataset_shuffle', action=argparse.BooleanOptionalAction, default=False, help='Shuffle the dataset order before training.')
    clip_data.add_argument('--clip_dataset_seed', type=int, default=0, help='Random seed used when shuffling the dataset order.')
    clip_data.add_argument('--clip_num_workers', type=int, default=4, help='Number of worker processes for CLIP dataloaders.')
    clip_data.add_argument('--clip_pin_memory', action=argparse.BooleanOptionalAction, default=False, help='Pin dataloader memory for CLIP tasks.')
    clip_data.add_argument('--clip_use_reference_data', action=argparse.BooleanOptionalAction, default=True, help='Use ImageNet1K/Flickr8k reference data for distillation.')

    # ------------------------------------------------------------------
    # 辅助数据集参数
    # ------------------------------------------------------------------
    aux = parser.add_argument_group('auxiliary', 'External / auxiliary dataset')
    aux.add_argument('--auxiliary_data_path', type=str, default='/data1/open_datasets/flickr8k', help='Root path of the auxiliary dataset. Example for Flickr8k: D:/projects/datasets/flickr8k')
    aux.add_argument('--aux_dataset_type', type=str, default='flickr8k', choices=['imagenet', 'flickr8k', 'auto'], help='Dataset type for auxiliary data (imagenet, flickr8k, or auto for automatic detection).')
    aux.add_argument('--aux_auto_detect', action=argparse.BooleanOptionalAction, default=True, help='Enable automatic dataset type detection based on path keywords and directory structure.')
    aux.add_argument('--aux_type_hint', type=str, default=None, choices=['imagenet', 'flickr8k'], help='Optional hint for automatic dataset type detection.')
    aux.add_argument('--aux_num_samples', type=int, default=1024, help='Limit the number of samples from the reference dataset. If not specified, use all samples.')
    aux.add_argument('--aux_split', type=str, default='val', choices=['train', 'val'], help='Dataset split to use (for datasets that support multiple splits like ImageNet).')
    aux.add_argument('--reference_batch_size', type=int, default=16, help='Batch size for the reference dataset. If not specified, uses the same value as the main training batch size.')

    # ------------------------------------------------------------------
    # 正则化 / L2‑Protection
    # ------------------------------------------------------------------
    reg = parser.add_argument_group('regularisation', 'Extra regularisation terms') 
    reg.add_argument('--l2_protection', action='store_true', default=False, help='Enable L2‑protection between the current and previous network.')
    reg.add_argument('--l2_protection_lambda', type=float, default=1.0, help='Weight for the L2‑protection term (higher → stronger regularisation). When `--l2_protection` is off, this will be automatically set to 0.0.')
    return parser

# In[]
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    parser = build_parser()
    args = parser.parse_args()
    args = vars(args)
    main(args)