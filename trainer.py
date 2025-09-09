import os
import sys
import logging
import torch
import random
import numpy as np
from models.subspace_lora import SubspaceLoRA
from utils.data_manager import DataManager
from utils.toolkit import count_parameters

def train(args):
    # Set device and seed for randomness upfront
    device = set_device(args['device'])
    all_results = {}
    
    # 用于存储最终结果的列表
    original_fc_final = []
    linear_fc_final = []
    for run_id, seed in enumerate(args['seed_list']):
        args['seed'], args['run_id'] = seed, run_id
        args['device'] = device
        
        logfile_head, logfile_name = build_log_dirs(args)
        args['log_path'] = logfile_name
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(filename)s] => %(message)s',
            handlers=[
                logging.FileHandler(filename=os.path.join(logfile_name, 'record.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        results = train_single_run(args)
        all_results[f"seed_{seed}"] = results
        
        # 收集最终结果
        original_fc_final.append(results['original_fc'][-1])
        linear_fc_final.append(results['linear_fc'][-1])
    # 计算平均值和标准差
    original_fc_mean = np.mean(original_fc_final)
    original_fc_std = np.std(original_fc_final)
    linear_fc_mean = np.mean(linear_fc_final)
    linear_fc_std = np.std(linear_fc_final)
    
    # 打印每个种子的结果
    for seed, results in all_results.items():
        print(f"Seed {seed}:")
        print(f"  Original FC: {results['original_fc'][-1]:.2f}%")
        print(f"  Linear FC: {results['linear_fc'][-1]:.2f}%")
    
    # 打印并记录平均值和标准差
    summary_msg = f"\nSummary across {len(args['seed_list'])} seeds:"
    summary_msg += f"\n  Original FC: {original_fc_mean:.2f}% ± {original_fc_std:.2f}%"
    summary_msg += f"\n  Linear FC: {linear_fc_mean:.2f}% ± {linear_fc_std:.2f}%"
    
    print(summary_msg)
    logging.info(summary_msg)
    
    # 将汇总结果也添加到all_results中
    all_results['summary'] = {
        'original_fc_mean': original_fc_mean,
        'original_fc_std': original_fc_std,
        'linear_fc_mean': linear_fc_mean,
        'linear_fc_std': linear_fc_std,
        'num_seeds': len(args['seed_list'])
    }
    return all_results

def train_single_run(args):
    # Setting random seed and device for reproducibility
    set_random(args['seed'])
    print_args(args)
    
    # Initialize data manager and model
    data_manager = DataManager(
        dataset_name=args['dataset'],
        shuffle=args['shuffle'],
        seed=args['seed'],
        init_cls=args['init_cls'],
        increment=args['increment'],
        args=args
    )
    
    model = SubspaceLoRA(args)
    logging.info(f'All params: {count_parameters(model.network)}')
    logging.info(f'Trainable params: {count_parameters(model.network, True)}')

    final_results = model.loop(data_manager)
    return final_results

def Bayesian_evaluate(args):
    """
    Similar to `train_single_run`, but evaluates the model every 5 tasks and returns the evaluation result.
    
    Args:
        args: Configuration arguments (same as in train_single_run)
        data_manager: DataManager object that handles datasets and task splits
    
    Yields:
        Task results after every 5 tasks for evaluation.
    """
    # Setting random seed and device for reproducibility
    set_random(args['seed'])
    device = set_device(args['device'])
    args['device'] = device

    print_args(args)

    logfile_head, logfile_name = build_log_dirs(args)
    args['log_path'] = logfile_name

    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=os.path.join(logfile_name, 'record.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )


    # Initialize data manager and model
    data_manager = DataManager(
        dataset_name=args['dataset'],
        shuffle=args['shuffle'],
        seed=args['seed'],
        init_cls=args['init_cls'],
        increment=args['increment']
    )
    
    model = SubspaceLoRA(args)
    logging.info(f'All params: {count_parameters(model.network)}')
    logging.info(f'Trainable params: {count_parameters(model.network, True)}')

    # Initialize result storage
    task_results = {
        "original_fc": [],
        "linear_fc": []}
    
    model._eval_tasks = model._compute_eval_milestones(data_manager.nb_tasks)

    logging.info(f"Classifier refinement scheduled at tasks: {sorted(model._eval_tasks)}")

    model.data_manager = data_manager
    # Train and evaluate in tasks

    for task_id in range(data_manager.nb_tasks):
        # Incremental training
        model.incremental_train(data_manager)
        if (model._cur_task + 1) in [5, 10]:
            model.refine_classifiers()
            # logging.info(f"Evaluating after task {model._cur_task}...")
            eval_result = model.eval_task()
            # Store the evaluation results
            task_results["original_fc"].append(eval_result.original_fc)
            task_results["linear_fc"].append(eval_result.linear_fc)
            # Yield evaluation results after every 5 tasks
            logging.info(f"Evaluation after task {task_id + 1} -> Original FC: {eval_result.original_fc:.2f}% | Compensated: {eval_result.linear_fc:.2f}%")
            
            if (model._cur_task + 1) == 5:
                flag = 0
            elif (model._cur_task + 1) == 10:
                flag = 1
            yield task_results, flag

        model.after_task()
    # Return the aggregated task results after all tasks
    return task_results

def set_device(device_type):
    """Properly set the device (either CPU or GPU) based on input"""
    if isinstance(device_type, (list, tuple)):
        return [torch.device(f'cuda:{d}' if d != -1 else 'cpu') for d in device_type]
    return torch.device('cuda' if device_type != -1 else 'cpu')

def set_random(seed):
    """Set random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    """Log the arguments for this run"""
    for key, value in args.items():
        logging.info(f'{key}: {value}')

# --------- NEW: compact float → short string ----------
def _fmt(x, *, digits=4):
    """
    压缩数值到短字符串：0.5 -> 0p5, 1e-3 -> 1e-03, 0.200 -> 0p2
    作用：减少路径长度、避免小数点过多。
    """
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    try:
        s = f"{float(x):.{digits}g}"
    except Exception:
        s = str(x)
    s = s.replace('.', 'p')
    return s

# --------- NEW: build log head & name based on args ----------
def build_log_dirs(args: dict):
    """
    根据数据集、ViT、LoRA 类型与关键超参构建日志目录层级。
    结构：
      {model_name}_logs_{user}/
        {dataset}_{vit_type}/
          {init/inc + lora关键参}/
            optim-..._lr-..._bz-..._epoch-..._seed-...
    """
    # 顶层与二级目录（不变）
    logfile_head = os.path.join(
        f"{args['model_name']}_logs_{args['user']}",
        f"{args['dataset']}_{args['vit_type']}"
    )

    # ---- 关键：在第三层目录揉入 LoRA 相关核心超参 ----
    base_tags = [
        f"init-{args['init_cls']}",
        f"inc-{args['increment']}",
        f"rank-{args.get('lora_rank', 'NA')}",
        f"lt-{args.get('lora_type', 'NA')}",
    ]

    # 某些公共但重要的控制项（按需保留，避免太长）
    # 投影温度对 SGP/OSGP 都重要：
    if 'proj_temp' in args:
        base_tags.append(f"T-{_fmt(args['proj_temp'])}")

    # 是否启用 classifier compensate
    if 'compensate' in args:
        base_tags.append(f"comp-{_fmt(args['compensate'])}")

    # LoRA 类型专属参数
    lora_type = args.get('lora_type', '')
    if lora_type in ('sgp_lora', 'osgp_lora'):
        if 'weight_kind' in args:
            base_tags.append(f"wk-{args['weight_kind']}")
        if 'weight_p' in args:
            base_tags.append(f"wp-{_fmt(args['weight_p'])}")
        if 'trace_k' in args:
            base_tags.append(f"tk-{_fmt(args['trace_k'])}")

    if lora_type == 'osgp_lora':
        if 'kl_gamma' in args:
            base_tags.append(f"klg-{_fmt(args['kl_gamma'])}")
        if 'osgp_scale' in args:
            base_tags.append(f"scale-{_fmt(args['osgp_scale'])}")

    if lora_type == 'nsp_lora':
        if 'nsp_eps' in args:
            base_tags.append(f"eps-{_fmt(args['nsp_eps'])}")
        if 'nsp_weight' in args:
            base_tags.append(f"nw-{_fmt(args['nsp_weight'])}")

    # KD / 正则（如果与你的实验常变动，也可以加进去）
    if 'kd_type' in args:
        base_tags.append(f"kd-{args['kd_type']}")
    if 'gamma_kd' in args:
        base_tags.append(f"gkd-{_fmt(args['gamma_kd'])}")
    if 'gamma_norm' in args:
        base_tags.append(f"gn-{_fmt(args['gamma_norm'])}")

    # 组装第三层目录名（更语义化）
    log_suffix = "_".join(base_tags)

    # 最底层（优化器/学习率/批量/轮次/seed）
    logfile_name = os.path.join(
        logfile_head,
        f"{log_suffix}_optim-{args['optimizer']}_lr-{_fmt(args['lrate'])}"
        f"_bz-{args['batch_size']}_epoch-{args['epochs']}_seed-{args['seed']}"
    )

    # 创建目录
    os.makedirs(logfile_head, exist_ok=True)
    os.makedirs(logfile_name, exist_ok=True)

    return logfile_head, logfile_name
