#!/usr/bin/env python3
"""
测试贝叶斯优化功能的简单脚本，用于验证代码的正确性。

此脚本运行少量试验，测试双向KL损失和Layer-wise蒸馏损失的优化功能。
"""

import argparse
import json
import logging
import time
from pathlib import Path

from bayesian_optimize_kl_layerwise import objective, prepare_args


def mock_train(args):
    """模拟训练函数，返回模拟结果。"""
    # 模拟训练延迟
    time.sleep(1)
    
    # 根据参数生成模拟结果
    base_acc = 0.5  # 基础准确率
    
    # 双向KL损失的影响
    if args.get('bidirectional_kd', False):
        base_acc += 0.02
    
    # Layer-wise蒸馏损失的影响
    if args.get('layerwise_kd_enabled', False):
        layerwise_weight = args.get('layerwise_kd_weight', 1.0)
        layerwise_loss_type = args.get('layerwise_kd_loss_type', 'mse')
        
        if layerwise_loss_type == 'mse_cosine':
            base_acc += 0.03 * layerwise_weight
        elif layerwise_loss_type == 'cosine':
            base_acc += 0.025 * layerwise_weight
        else:  # mse
            base_acc += 0.02 * layerwise_weight
        
        layerwise_strategy = args.get('layerwise_kd_weight_strategy', 'uniform')
        if layerwise_strategy == 'exponential':
            base_acc += 0.01
        elif layerwise_strategy == 'linear':
            base_acc += 0.005
    
    # 学习率的影响
    lrate = args.get('lrate', 1e-4)
    if lrate == 5e-4:
        base_acc += 0.01
    elif lrate == 1e-3:
        base_acc += 0.005
    
    # 添加一些随机性
    import random
    base_acc += random.uniform(-0.01, 0.01)
    
    # 返回模拟结果
    return {
        1993: {
            "zeroshot_acc": [base_acc],
            "layerwise_kd_loss": [0.1 * args.get('layerwise_kd_weight', 1.0)] if args.get('layerwise_kd_enabled', False) else []
        }
    }


def test_objective_function():
    """测试目标函数。"""
    logging.info("测试目标函数...")
    
    # 准备基础参数
    base_args = prepare_args({})
    
    # 定义搜索空间 - 仅包含要优化的参数，其他使用默认值
    search_space = {}
    
    # 创建模拟试验对象
    from typing import Any, Dict
    
    class MockTrial:
        def __init__(self, params):
            self.number = 0
            self.params = params
            self.user_attrs = {}
        
        def suggest_categorical(self, name: str, choices: list) -> Any:
            return choices[0] if isinstance(choices, list) else choices
        
        def suggest_float(self, name: str, low: float, high: float) -> float:
            return (low + high) / 2
        
        def set_user_attr(self, name: str, value: Any) -> None:
            self.user_attrs[name] = value
        
        # 添加optuna.Trial的其他必需方法
        def report(self, value: float, step: int) -> None:
            pass
        
        def should_prune(self) -> bool:
            return False
    
    # 测试不同的参数组合
    test_cases = [
        {
            'bidirectional_kd': True,
            'layerwise_kd_enabled': True,
            'layerwise_kd_weight': 0.5,
            'layerwise_kd_loss_type': 'mse_cosine',
            'layerwise_kd_weight_strategy': 'exponential',
            'lrate': 5e-4,
        },
        {
            'bidirectional_kd': False,
            'layerwise_kd_enabled': False,
            'lrate': 1e-4,
        },
        {
            'bidirectional_kd': True,
            'layerwise_kd_enabled': False,
            'lrate': 5e-4,
        },
        {
            'bidirectional_kd': False,
            'layerwise_kd_enabled': True,
            'layerwise_kd_weight': 1.0,
            'layerwise_kd_loss_type': 'cosine',
            'layerwise_kd_weight_strategy': 'linear',
            'lrate': 1e-3,
        },
    ]
    
    # 临时替换train函数
    import bayesian_optimize_kl_layerwise
    original_train = bayesian_optimize_kl_layerwise.train
    bayesian_optimize_kl_layerwise.train = mock_train
    
    try:
        for i, params in enumerate(test_cases):
            logging.info(f"测试用例 {i+1}: {params}")
            
            trial = MockTrial(params)
            
            # 调用目标函数
            result = objective(trial, base_args, search_space)  # type: ignore
            
            logging.info(f"结果: {result:.4f}")
            logging.info(f"用户属性: {trial.user_attrs}")
            logging.info("-" * 50)
    
    finally:
        # 恢复原始train函数
        bayesian_optimize_kl_layerwise.train = original_train
    
    logging.info("目标函数测试完成")


def test_optimization_script():
    """测试优化脚本。"""
    logging.info("测试优化脚本...")
    
    # 创建临时输出目录
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 临时替换train函数
    import bayesian_optimize_kl_layerwise
    original_train = bayesian_optimize_kl_layerwise.train
    bayesian_optimize_kl_layerwise.train = mock_train
    
    try:
        # 运行优化
        import optuna
        
        # 准备基础参数
        base_args = prepare_args({})
        
        # 定义搜索空间
        search_space = {
            'lrate': [1e-4, 5e-4],
            'bidirectional_kd': [True, False],
            'layerwise_kd_enabled': [True, False],
            'layerwise_kd_weight': [0.1, 1.0],
            'layerwise_kd_loss_type': ["mse", "cosine"],
            'layerwise_kd_weight_strategy': ["uniform", "linear"],
        }
        
        # 创建研究
        study = optuna.create_study(direction="maximize")
        
        # 运行优化
        study.optimize(
            lambda t: objective(t, base_args, search_space),
            n_trials=5,
        )
        
        # 保存结果
        results = []
        for trial in study.trials:
            results.append({
                "index": trial.number,
                "parameters": trial.params,
                "value": trial.value,
                "duration_sec": trial.user_attrs.get("duration_sec", 0),
                "failed": trial.user_attrs.get("failed", trial.state != optuna.trial.TrialState.COMPLETE),
                "error_message": trial.user_attrs.get("error_message"),
                "optimized_params": trial.user_attrs.get("params", {}),
            })
        
        # 保存结果到文件
        output_file = output_dir / "test_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"测试结果保存到: {output_file}")
        
        # 打印最佳试验
        if study.best_trial:
            logging.info(f"最佳试验: 值={study.best_trial.value:.4f}, 参数={study.best_trial.params}")
    
    finally:
        # 恢复原始train函数
        bayesian_optimize_kl_layerwise.train = original_train
    
    logging.info("优化脚本测试完成")


def main():
    parser = argparse.ArgumentParser(description='测试贝叶斯优化功能')
    parser.add_argument('--test', choices=['objective', 'script', 'all'], default='all', help='测试类型')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    if args.test in ['objective', 'all']:
        test_objective_function()
    
    if args.test in ['script', 'all']:
        test_optimization_script()
    
    logging.info("所有测试完成")


if __name__ == "__main__":
    main()