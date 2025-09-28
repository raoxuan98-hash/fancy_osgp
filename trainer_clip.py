import os
import sys
import logging
import torch
import random
import numpy as np
from models.subspace_lora_clip import SubspaceLoRA_CLIP
from utils.toolkit import count_parameters


def train(args):
    device = set_device(args['device'])
    all_results = {}

    # Friendly startup hint for reference data usage
    try:
        if bool(args.get('clip_use_reference_data', False)):
            ref_type = args.get('aux_dataset_type', 'imagenet')
            ref_path = args.get('auxiliary_data_path')
            nw = args.get('clip_num_workers')
            pm = args.get('clip_pin_memory')
            print(f"[Reference Data] enabled | type={ref_type} | path={ref_path} | num_workers={nw} | pin_memory={pm} | captions=multi-avg(if supported)")
        else:
            print("[Reference Data] disabled")
    except Exception:
        pass

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
    # Aggregate expects an iterable of per-seed dicts; pass values()
    try:
        aggregate_seed_results(list(all_results.values()))
    except Exception as e:
        logging.warning(f"Aggregation failed: {e}")

    return all_results

def train_single_run(args):
    # Setting random seed and device for reproducibility
    set_random(args['seed'])
    print_args(args)
    
    model = SubspaceLoRA_CLIP(args)
    logging.info(f'All params: {count_parameters(model.network)}')
    logging.info(f'Trainable params: {count_parameters(model.network, True)}')
    final_results = model.loop()
    # Ensure a dict is returned for aggregation
    if isinstance(final_results, dict):
        return final_results
    # Fallback: package minimal results from model history if available
    try:
        hist = getattr(model, 'history', {})
        return {
            'average_accuracies': {'zeroshot': float(hist.get('zeroshot_acc', [-1])[-1]) if hist.get('zeroshot_acc') else None},
            'last_task_accuracies': {'zeroshot': float(hist.get('zeroshot_acc', [-1])[-1]) if hist.get('zeroshot_acc') else None},
        }
    except Exception:
        return {'average_accuracies': {}, 'last_task_accuracies': {}}

def set_device(device_type):
    """Properly set the device (either CPU or GPU) based on input.

    Accepts values like -1, "-1", 0, "0", or a list such as ["0", "1"].
    Any entry equal to -1 (after int cast) maps to CPU; otherwise maps to CUDA:{id}.
    """
    def _to_device(d):
        try:
            did = int(d)
        except Exception:
            # fallback: treat non-numeric as cpu
            return torch.device('cpu')
        if did == -1:
            return torch.device('cpu')
        return torch.device(f'cuda:{did}')

    if isinstance(device_type, (list, tuple)):
        return [_to_device(d) for d in device_type]
    return _to_device(device_type)

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

def build_log_dirs(args: dict):
    logfile_head = os.path.join(
        f"{args['model_name']}_logs_{args['user']}",
        f"{args['dataset']}_{args['vit_type']}")

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
