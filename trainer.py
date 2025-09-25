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
    device = set_device(args['device'])
    all_results = {}
    
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
                logging.StreamHandler(sys.stdout)])
        
        args['log_path'] = logfile_name
        results = train_single_run(args)
        all_results[f"seed_{seed}"] = results
    aggregate_seed_results(all_results)
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
        args=args)
    
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
        increment=args['increment'])
    
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
    logfile_head = os.path.join(
        f"{args['model_name']}_logs_{args['user']}",
        f"{args['dataset']}_{args['vit_type']}")

    # ---- 关键：在第三层目录揉入 LoRA 相关核心超参 ----
    base_tags = [
        f"init-{args['init_cls']}",
        f"inc-{args['increment']}",
        f"rank-{args.get('lora_rank', 'NA')}",
        f"lt-{args.get('lora_type', 'NA')}"]


    if 'weight_temp' in args:
        base_tags.append(f"T-{_fmt(args['weight_temp'])}")

    if 'compensate' in args:
        base_tags.append(f"comp-{_fmt(args['compensate'])}")

    lora_type = args.get('lora_type', '')
    if lora_type in ('sgp_lora', 'osgp_lora'):
        if 'weight_kind' in args:
            base_tags.append(f"wk-{args['weight_kind']}")
        if 'weight_p' in args:
            base_tags.append(f"wp-{_fmt(args['weight_p'])}")

    if lora_type == 'nsp_lora':
        if 'nsp_eps' in args:
            base_tags.append(f"eps-{_fmt(args['nsp_eps'])}")
        if 'nsp_weight' in args:
            base_tags.append(f"nw-{_fmt(args['nsp_weight'])}")

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
        f"_bz-{args['batch_size']}_iter-{args['iterations']}_seed-{args['seed']}")

    # 创建目录
    os.makedirs(logfile_head, exist_ok=True)
    os.makedirs(logfile_name, exist_ok=True)

    return logfile_head, logfile_name

def aggregate_seed_results(seed_results):
    if not seed_results:
        logging.warning("⚠️ No seed results provided for aggregation.")
        return {"final_task": {}, "average_across_tasks": {}}

    # Collect all variant names across all seeds
    all_variants = set()
    for res in seed_results:
        all_variants.update(res.get("last_task_accuracies", {}).keys())
        all_variants.update(res.get("average_accuracies", {}).keys())
    all_variants = sorted(all_variants)

    # Initialize containers
    final_task_values = {variant: [] for variant in all_variants}
    avg_task_values = {variant: [] for variant in all_variants}

    # Populate with data from each seed
    for res in seed_results:
        last_acc = res.get("last_task_accuracies", {})
        avg_acc = res.get("average_accuracies", {})

        for variant in all_variants:
            final_task_values[variant].append(last_acc.get(variant, 0.0))
            avg_task_values[variant].append(avg_acc.get(variant, 0.0))

    # Compute mean and std
    final_task_stats = {}
    avg_task_stats = {}

    for variant in all_variants:
        f_vals = np.array(final_task_values[variant])
        a_vals = np.array(avg_task_values[variant])

        final_task_stats[variant] = (float(np.mean(f_vals)), float(np.std(f_vals)))
        avg_task_stats[variant] = (float(np.mean(a_vals)), float(np.std(a_vals)))

    # === 📊 Log Aggregated Results ===
    logging.info("📈 Aggregated Results Across Random Seeds:")
    logging.info("  ── Final Task Accuracy (Mean ± Std) ──")
    for variant in all_variants:
        mean, std = final_task_stats[variant]
        logging.info(f"      {variant:<20} : {mean:.2f}% ± {std:.2f}%")

    logging.info("  ── Average Accuracy Across Tasks (Mean ± Std) ──")
    for variant in all_variants:
        mean, std = avg_task_stats[variant]
        logging.info(f"      {variant:<20} : {mean:.2f}% ± {std:.2f}%")

    # Return structured stats
    return {
        "final_task": final_task_stats,
        "average_across_tasks": avg_task_stats}