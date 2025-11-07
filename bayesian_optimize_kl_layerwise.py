#!/usr/bin/env python3
"""
贝叶斯优化脚本，用于优化双向KL损失和Layer-wise蒸馏损失参数。

此脚本使用optuna进行超参数优化，主要优化以下参数：
1. bidirectional_kd: 是否使用双向KL损失
2. layerwise_kd_enabled: 是否启用Layer-wise蒸馏损失
3. layerwise_kd_weight: Layer-wise蒸馏损失的权重
4. layerwise_kd_loss_type: Layer-wise蒸馏损失的类型
5. layerwise_kd_weight_strategy: Layer-wise蒸馏权重的分配策略
"""

import argparse
import json
import logging
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import optuna

from main_clip import build_parser
from trainer_clip import train


def objective(trial: optuna.Trial, base_args: Dict[str, Any], search_space: Dict[str, list]) -> float:
    """
    优化目标函数，优化所有任务的平均零样本准确率。
    
    Args:
        trial: optuna试验对象
        base_args: 基础参数字典
        search_space: 搜索空间定义
        
    Returns:
        平均零样本准确率
    """
    # 从搜索空间中采样参数
    params = {}
    
    # 双向KL损失参数
    if 'bidirectional_kd' in search_space:
        params['bidirectional_kd'] = trial.suggest_categorical('bidirectional_kd', search_space['bidirectional_kd'])
    
    # Layer-wise蒸馏损失参数 - 处理条件依赖关系
    if 'layerwise_kd_enabled' in search_space:
        params['layerwise_kd_enabled'] = trial.suggest_categorical('layerwise_kd_enabled', search_space['layerwise_kd_enabled'])
        
        # 只有在启用Layer-wise蒸馏时才采样相关参数
        if params['layerwise_kd_enabled']:
            if 'layerwise_kd_weight' in search_space:
                params['layerwise_kd_weight'] = trial.suggest_categorical('layerwise_kd_weight',
                                                              search_space['layerwise_kd_weight'])
            
            if 'layerwise_kd_loss_type' in search_space:
                params['layerwise_kd_loss_type'] = trial.suggest_categorical('layerwise_kd_loss_type',
                                                                        search_space['layerwise_kd_loss_type'])
            
            if 'layerwise_kd_weight_strategy' in search_space:
                params['layerwise_kd_weight_strategy'] = trial.suggest_categorical('layerwise_kd_weight_strategy',
                                                                             search_space['layerwise_kd_weight_strategy'])
        else:
            # 如果禁用Layer-wise蒸馏，设置默认值
            params['layerwise_kd_weight'] = 1.0
            params['layerwise_kd_loss_type'] = 'mse'
            params['layerwise_kd_weight_strategy'] = 'uniform'
    
    # 其他基础参数
    for name, choices in search_space.items():
        if name not in params:
            if isinstance(choices, list) and all(isinstance(x, (int, float, str, bool)) for x in choices):
                params[name] = trial.suggest_categorical(name, choices)
            elif name == 'layerwise_kd_weight' and isinstance(choices, list):
                # 特殊处理layerwise_kd_weight参数，使用离散值
                params[name] = trial.suggest_categorical(name, choices)
    
    # 构建完整的参数字典
    args = deepcopy(base_args)
    args.update(params)
    args["prefix"] = f"optuna_kl_layerwise_trial_{trial.number}"
    
    # 记录试验参数
    logging.info(f"试验 {trial.number} 参数: {params}")
    
    start = time.perf_counter()
    try:
        # 运行训练
        raw_results = train(args)  # 期望返回: {seed: history_dict}
        duration = time.perf_counter() - start
        
        # 计算每个种子的平均零样本准确率
        mean_accuracies_per_seed = []
        for seed, history in (raw_results or {}).items():
            if not isinstance(history, dict):
                continue
            zeroshot_acc_list = history.get("zeroshot_acc")
            if zeroshot_acc_list and len(zeroshot_acc_list) > 0:
                # 计算所有任务的平均零样本准确率
                avg_inc_acc = float(np.mean(zeroshot_acc_list))
                mean_accuracies_per_seed.append(avg_inc_acc)
        
        if not mean_accuracies_per_seed:
            raise ValueError("在任何种子中未找到有效的zeroshot_acc历史记录。")
        
        final_objective = float(np.mean(mean_accuracies_per_seed))  # 跨种子平均
        
        # 记录试验属性
        trial.set_user_attr("duration_sec", duration)
        trial.set_user_attr("mean_accuracies_per_seed", mean_accuracies_per_seed)
        trial.set_user_attr("final_objective", final_objective)
        trial.set_user_attr("failed", False)
        trial.set_user_attr("params", params)
        
        # 记录layerwise蒸馏损失历史（如果存在）
        layerwise_kd_losses = []
        for seed, history in (raw_results or {}).items():
            if isinstance(history, dict) and "layerwise_kd_loss" in history:
                layerwise_kd_losses.extend(history["layerwise_kd_loss"])
        
        if layerwise_kd_losses:
            trial.set_user_attr("avg_layerwise_kd_loss", float(np.mean(layerwise_kd_losses)))
        
        if math.isnan(final_objective):
            raise optuna.TrialPruned("目标值为NaN")
        
        logging.info(f"试验 {trial.number} 完成，准确率: {final_objective:.4f}")
        return final_objective  # 最大化这个值
        
    except Exception as e:
        duration = time.perf_counter() - start
        trial.set_user_attr("duration_sec", duration)
        trial.set_user_attr("failed", True)
        trial.set_user_attr("error_message", str(e))
        trial.set_user_attr("params", params)
        logging.exception(f"试验 {trial.number} 失败: {e}")
        raise


def prepare_args(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """准备基础参数字典。"""
    parser = build_parser()
    args = vars(parser.parse_args([]))
    args.update(overrides)
    return args


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("kl_layerwise_optuna_results.json"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--study-name", type=str, default="kl-layerwise-optimization2")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # 添加自定义搜索空间选项
    parser.add_argument("--optimize-bidirectional-kd", type=bool, default=True,  
                       help="是否优化双向KL损失参数")
    parser.add_argument("--optimize-layerwise-kd", type=bool, default=True, 
                       help="是否优化Layer-wise蒸馏损失参数")
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), 
                       format="%(asctime)s - %(levelname)s - %(message)s")
    
    # 基础训练参数
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.seeds:
        overrides["seed_list"] = args.seeds
    base_args = prepare_args(overrides)
    
    # 构建搜索空间 - 仅包含要优化的参数，其他使用默认值
    search_space = {}
    
    # 添加双向KL损失参数到搜索空间
    if args.optimize_bidirectional_kd:
        search_space["bidirectional_kd"] = [True, False]
        logging.info("优化双向KL损失参数")
    
    # 添加Layer-wise蒸馏损失参数到搜索空间
    if args.optimize_layerwise_kd:
        search_space["layerwise_kd_enabled"] = [True, False]
        search_space["layerwise_kd_weight"] = [1.0, 2.0, 5.0]  # 离散值
        search_space["layerwise_kd_loss_type"] = ["mse", "cosine", "mse_cosine"]
        search_space["layerwise_kd_weight_strategy"] = ["uniform", "linear", "exponential"]
        logging.info("优化Layer-wise蒸馏损失参数")
    
    # 存储和采样器配置
    storage = args.storage or f"sqlite:///{args.study_name}.db"
    logging.info(f"使用存储: {storage}")
    
    sampler = (
        optuna.samplers.RandomSampler()
        if args.sampler == "random"
        else optuna.samplers.TPESampler(multivariate=True)
    )
    
    # 创建研究
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",  # 最大化准确率
        sampler=sampler,
        load_if_exists=True,
    )
    logging.info(f"加载了 {len(study.trials)} 个现有试验。")
    
    # 运行优化
    study.optimize(
        lambda t: objective(t, base_args, search_space),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    
    # 保存完整结果
    results = []
    for trial in study.trials:
        result = {
            "index": trial.number,
            "parameters": trial.params,
            "value": trial.value,
            "duration_sec": trial.user_attrs.get("duration_sec", math.nan),
            "failed": trial.user_attrs.get("failed", trial.state != optuna.trial.TrialState.COMPLETE),
            "error_message": trial.user_attrs.get("error_message"),
        }
        
        # 添加自定义属性
        if "params" in trial.user_attrs:
            result["optimized_params"] = trial.user_attrs["params"]
        if "mean_accuracies_per_seed" in trial.user_attrs:
            result["mean_accuracies_per_seed"] = trial.user_attrs["mean_accuracies_per_seed"]
        if "avg_layerwise_kd_loss" in trial.user_attrs:
            result["avg_layerwise_kd_loss"] = trial.user_attrs["avg_layerwise_kd_loss"]
        
        results.append(result)
    
    # 保存结果到文件
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # 打印最佳试验结果
    if study.best_trial:
        logging.info(f"最佳试验 {study.best_trial.number}: 值={study.best_trial.value:.4f}, 参数={study.best_trial.params}")
        
        # 分析最佳参数配置
        best_params = study.best_trial.params
        logging.info("最佳参数配置分析:")
        
        if args.optimize_bidirectional_kd and "bidirectional_kd" in best_params:
            logging.info(f"  双向KL损失: {'启用' if best_params['bidirectional_kd'] else '禁用'}")
        
        if args.optimize_layerwise_kd:
            if "layerwise_kd_enabled" in best_params:
                logging.info(f"  Layer-wise蒸馏损失: {'启用' if best_params['layerwise_kd_enabled'] else '禁用'}")
            if "layerwise_kd_weight" in best_params:
                logging.info(f"  Layer-wise蒸馏损失权重: {best_params['layerwise_kd_weight']:.4f}")
            if "layerwise_kd_loss_type" in best_params:
                logging.info(f"  Layer-wise蒸馏损失类型: {best_params['layerwise_kd_loss_type']}")
            if "layerwise_kd_weight_strategy" in best_params:
                logging.info(f"  Layer-wise蒸馏权重策略: {best_params['layerwise_kd_weight_strategy']}")
    
    logging.info(f"保存了 {len(results)} 个试验结果到 {args.output}")
    
    # 参数重要性分析
    if len(study.trials) > 1:
        try:
            importance = optuna.importance.get_param_importances(study)
            logging.info("参数重要性:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"  {param}: {imp:.4f}")
        except Exception as e:
            logging.warning(f"无法计算参数重要性: {e}")


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    main()