#!/usr/bin/env python3
"""
分析贝叶斯优化结果的脚本，用于评估双向KL损失和Layer-wise蒸馏损失的效果。

此脚本可以：
1. 加载和可视化优化结果
2. 比较不同参数配置的性能
3. 分析参数重要性
4. 生成最佳配置建议
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_results(file_path: Path) -> List[Dict[str, Any]]:
    """加载优化结果文件。"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析优化结果。"""
    if not results:
        return {}
    
    # 过滤成功的试验
    successful_trials = [r for r in results if not r.get('failed', True)]
    
    if not successful_trials:
        logging.warning("没有成功的试验")
        return {}
    
    # 找到最佳试验
    best_trial = max(successful_trials, key=lambda x: x.get('value', float('-inf')))
    
    # 计算统计信息
    values = [r.get('value', 0) for r in successful_trials]
    
    analysis = {
        'total_trials': len(results),
        'successful_trials': len(successful_trials),
        'failed_trials': len(results) - len(successful_trials),
        'best_trial': best_trial,
        'best_value': best_trial.get('value', 0),
        'mean_value': np.mean(values),
        'std_value': np.std(values),
        'min_value': np.min(values),
        'max_value': np.max(values),
    }
    
    return analysis

def compare_bidirectional_kd(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """比较双向KL损失的效果。"""
    # 分离双向KL损失启用和禁用的试验
    enabled_trials = [r for r in results 
                     if not r.get('failed', True) and 
                     r.get('optimized_params', {}).get('bidirectional_kd') is True]
    
    disabled_trials = [r for r in results 
                      if not r.get('failed', True) and 
                      r.get('optimized_params', {}).get('bidirectional_kd') is False]
    
    if not enabled_trials or not disabled_trials:
        return {}
    
    enabled_values = [r.get('value', 0) for r in enabled_trials]
    disabled_values = [r.get('value', 0) for r in disabled_trials]
    
    comparison = {
        'enabled': {
            'count': len(enabled_trials),
            'mean': np.mean(enabled_values),
            'std': np.std(enabled_values),
            'min': np.min(enabled_values),
            'max': np.max(enabled_values),
        },
        'disabled': {
            'count': len(disabled_trials),
            'mean': np.mean(disabled_values),
            'std': np.std(disabled_values),
            'min': np.min(disabled_values),
            'max': np.max(disabled_values),
        },
        'improvement': np.mean(enabled_values) - np.mean(disabled_values),
        'improvement_percent': (np.mean(enabled_values) - np.mean(disabled_values)) / np.mean(disabled_values) * 100,
    }
    
    return comparison

def compare_layerwise_kd(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """比较Layer-wise蒸馏损失的效果。"""
    # 分离Layer-wise蒸馏损失启用和禁用的试验
    enabled_trials = [r for r in results 
                     if not r.get('failed', True) and 
                     r.get('optimized_params', {}).get('layerwise_kd_enabled') is True]
    
    disabled_trials = [r for r in results 
                      if not r.get('failed', True) and 
                      r.get('optimized_params', {}).get('layerwise_kd_enabled') is False]
    
    if not enabled_trials or not disabled_trials:
        return {}
    
    enabled_values = [r.get('value', 0) for r in enabled_trials]
    disabled_values = [r.get('value', 0) for r in disabled_trials]
    
    comparison = {
        'enabled': {
            'count': len(enabled_trials),
            'mean': np.mean(enabled_values),
            'std': np.std(enabled_values),
            'min': np.min(enabled_values),
            'max': np.max(enabled_values),
        },
        'disabled': {
            'count': len(disabled_trials),
            'mean': np.mean(disabled_values),
            'std': np.std(disabled_values),
            'min': np.min(disabled_values),
            'max': np.max(disabled_values),
        },
        'improvement': np.mean(enabled_values) - np.mean(disabled_values),
        'improvement_percent': (np.mean(enabled_values) - np.mean(disabled_values)) / np.mean(disabled_values) * 100,
    }
    
    return comparison

def analyze_layerwise_kd_types(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析不同Layer-wise蒸馏损失类型的效果。"""
    loss_types = {}
    
    for result in results:
        if result.get('failed', True):
            continue
            
        params = result.get('optimized_params', {})
        if params.get('layerwise_kd_enabled') is True:
            loss_type = params.get('layerwise_kd_loss_type', 'unknown')
            if loss_type not in loss_types:
                loss_types[loss_type] = []
            loss_types[loss_type].append(result.get('value', 0))
    
    analysis = {}
    for loss_type, values in loss_types.items():
        analysis[loss_type] = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    return analysis

def analyze_layerwise_kd_strategies(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析不同Layer-wise蒸馏权重策略的效果。"""
    strategies = {}
    
    for result in results:
        if result.get('failed', True):
            continue
            
        params = result.get('optimized_params', {})
        if params.get('layerwise_kd_enabled') is True:
            strategy = params.get('layerwise_kd_weight_strategy', 'unknown')
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(result.get('value', 0))
    
    analysis = {}
    for strategy, values in strategies.items():
        analysis[strategy] = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    return analysis

def plot_results(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """绘制优化结果图表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 过滤成功的试验
    successful_trials = [r for r in results if not r.get('failed', True)]
    
    if not successful_trials:
        logging.warning("没有成功的试验可以绘制")
        return
    
    # 1. 试验值分布
    values = [r.get('value', 0) for r in successful_trials]
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=20, alpha=0.7)
    plt.xlabel('准确率')
    plt.ylabel('试验数量')
    plt.title('优化结果分布')
    plt.grid(True)
    plt.savefig(output_dir / 'value_distribution.png')
    plt.close()
    
    # 2. 试验进度
    trial_indices = [r.get('index', 0) for r in successful_trials]
    plt.figure(figsize=(10, 6))
    plt.plot(trial_indices, values, 'o-')
    plt.xlabel('试验索引')
    plt.ylabel('准确率')
    plt.title('优化进度')
    plt.grid(True)
    plt.savefig(output_dir / 'optimization_progress.png')
    plt.close()
    
    # 3. 双向KL损失比较
    bidirectional_comparison = compare_bidirectional_kd(results)
    if bidirectional_comparison:
        labels = ['禁用', '启用']
        means = [bidirectional_comparison['disabled']['mean'], bidirectional_comparison['enabled']['mean']]
        stds = [bidirectional_comparison['disabled']['std'], bidirectional_comparison['enabled']['std']]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, means, yerr=stds, capsize=5)
        plt.ylabel('平均准确率')
        plt.title('双向KL损失效果比较')
        plt.grid(True)
        
        # 添加数值标签
        for i, (bar, mean) in enumerate(zip(bars, means)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom')
        
        plt.savefig(output_dir / 'bidirectional_kd_comparison.png')
        plt.close()
    
    # 4. Layer-wise蒸馏损失比较
    layerwise_comparison = compare_layerwise_kd(results)
    if layerwise_comparison:
        labels = ['禁用', '启用']
        means = [layerwise_comparison['disabled']['mean'], layerwise_comparison['enabled']['mean']]
        stds = [layerwise_comparison['disabled']['std'], layerwise_comparison['enabled']['std']]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, means, yerr=stds, capsize=5)
        plt.ylabel('平均准确率')
        plt.title('Layer-wise蒸馏损失效果比较')
        plt.grid(True)
        
        # 添加数值标签
        for i, (bar, mean) in enumerate(zip(bars, means)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom')
        
        plt.savefig(output_dir / 'layerwise_kd_comparison.png')
        plt.close()
    
    # 5. Layer-wise蒸馏损失类型比较
    loss_type_analysis = analyze_layerwise_kd_types(results)
    if loss_type_analysis:
        types = list(loss_type_analysis.keys())
        means = [loss_type_analysis[t]['mean'] for t in types]
        stds = [loss_type_analysis[t]['std'] for t in types]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(types, means, yerr=stds, capsize=5)
        plt.ylabel('平均准确率')
        plt.title('不同Layer-wise蒸馏损失类型比较')
        plt.grid(True)
        
        # 添加数值标签
        for i, (bar, mean) in enumerate(zip(bars, means)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom')
        
        plt.savefig(output_dir / 'layerwise_loss_types_comparison.png')
        plt.close()
    
    # 6. Layer-wise蒸馏权重策略比较
    strategy_analysis = analyze_layerwise_kd_strategies(results)
    if strategy_analysis:
        strategies = list(strategy_analysis.keys())
        means = [strategy_analysis[s]['mean'] for s in strategies]
        stds = [strategy_analysis[s]['std'] for s in strategies]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, means, yerr=stds, capsize=5)
        plt.ylabel('平均准确率')
        plt.title('不同Layer-wise蒸馏权重策略比较')
        plt.grid(True)
        
        # 添加数值标签
        for i, (bar, mean) in enumerate(zip(bars, means)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom')
        
        plt.savefig(output_dir / 'layerwise_weight_strategies_comparison.png')
        plt.close()

def generate_report(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """生成分析报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis = analyze_results(results)
    bidirectional_comparison = compare_bidirectional_kd(results)
    layerwise_comparison = compare_layerwise_kd(results)
    loss_type_analysis = analyze_layerwise_kd_types(results)
    strategy_analysis = analyze_layerwise_kd_strategies(results)
    
    report = []
    report.append("# 贝叶斯优化结果分析报告\n")
    
    # 基本统计信息
    report.append("## 基本统计信息\n")
    report.append(f"- 总试验数: {analysis.get('total_trials', 0)}")
    report.append(f"- 成功试验数: {analysis.get('successful_trials', 0)}")
    report.append(f"- 失败试验数: {analysis.get('failed_trials', 0)}")
    report.append(f"- 最佳准确率: {analysis.get('best_value', 0):.4f}")
    report.append(f"- 平均准确率: {analysis.get('mean_value', 0):.4f} ± {analysis.get('std_value', 0):.4f}")
    report.append(f"- 准确率范围: [{analysis.get('min_value', 0):.4f}, {analysis.get('max_value', 0):.4f}]\n")
    
    # 最佳配置
    best_trial = analysis.get('best_trial', {})
    if best_trial:
        report.append("## 最佳配置\n")
        params = best_trial.get('optimized_params', {})
        for param, value in params.items():
            report.append(f"- {param}: {value}")
        report.append("")
    
    # 双向KL损失分析
    if bidirectional_comparison:
        report.append("## 双向KL损失分析\n")
        report.append(f"- 启用双向KL损失的平均准确率: {bidirectional_comparison['enabled']['mean']:.4f} ± {bidirectional_comparison['enabled']['std']:.4f}")
        report.append(f"- 禁用双向KL损失的平均准确率: {bidirectional_comparison['disabled']['mean']:.4f} ± {bidirectional_comparison['disabled']['std']:.4f}")
        report.append(f"- 改进: {bidirectional_comparison['improvement']:.4f} ({bidirectional_comparison['improvement_percent']:.2f}%)\n")
    
    # Layer-wise蒸馏损失分析
    if layerwise_comparison:
        report.append("## Layer-wise蒸馏损失分析\n")
        report.append(f"- 启用Layer-wise蒸馏损失的平均准确率: {layerwise_comparison['enabled']['mean']:.4f} ± {layerwise_comparison['enabled']['std']:.4f}")
        report.append(f"- 禁用Layer-wise蒸馏损失的平均准确率: {layerwise_comparison['disabled']['mean']:.4f} ± {layerwise_comparison['disabled']['std']:.4f}")
        report.append(f"- 改进: {layerwise_comparison['improvement']:.4f} ({layerwise_comparison['improvement_percent']:.2f}%)\n")
    
    # Layer-wise蒸馏损失类型分析
    if loss_type_analysis:
        report.append("## Layer-wise蒸馏损失类型分析\n")
        for loss_type, stats in loss_type_analysis.items():
            report.append(f"- {loss_type}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        report.append("")
    
    # Layer-wise蒸馏权重策略分析
    if strategy_analysis:
        report.append("## Layer-wise蒸馏权重策略分析\n")
        for strategy, stats in strategy_analysis.items():
            report.append(f"- {strategy}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        report.append("")
    
    # 保存报告
    with open(output_dir / 'analysis_report.md', 'w') as f:
        f.write('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(description='分析贝叶斯优化结果')
    parser.add_argument('input_files', nargs='+', type=Path, help='输入结果文件路径')
    parser.add_argument('--output-dir', type=Path, default=Path('analysis_output'), help='输出目录')
    parser.add_argument('--plot', action='store_true', help='生成图表')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析每个文件
    for input_file in args.input_files:
        if not input_file.exists():
            logging.warning(f"文件不存在: {input_file}")
            continue
        
        logging.info(f"分析文件: {input_file}")
        results = load_results(input_file)
        
        # 创建文件特定的输出目录
        file_output_dir = args.output_dir / input_file.stem
        
        # 生成报告
        generate_report(results, file_output_dir)
        
        # 生成图表
        if args.plot:
            plot_results(results, file_output_dir)
        
        logging.info(f"分析完成，结果保存在: {file_output_dir}")
    
    # 如果有多个文件，生成综合比较
    if len(args.input_files) > 1:
        logging.info("生成综合比较报告...")
        # 这里可以添加多文件比较的逻辑

if __name__ == "__main__":
    main()