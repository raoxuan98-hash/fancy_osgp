#!/usr/bin/env python3
"""
Bayesian Optimization for OSGP-LoRA hyperparameters using Optuna.

This script optimizes proj_gamma and proj_temp hyperparameters to maximize
the sum of results from cars196, cifar100_224, and imagenet-r datasets.
Supports checkpointing and parallel optimization.
"""

import os
import json
import logging
import argparse
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.storages import RDBStorage
from typing import Dict, Any, Optional
import torch
import numpy as np
from trainer import Bayesian_evaluate
from main import set_smart_defaults

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """Bayesian optimizer for OSGP-LoRA hyperparameters with checkpoint support."""

    def __init__(self, n_trials: int = 20, timeout: int = 3600000, checkpoint_dir: Optional[str] = None,
                 study_name: str = "sldc_optimization", pruner: Optional[optuna.pruners.BasePruner] = None):
        self.n_trials = n_trials
        self.timeout = timeout
        # self.datasets = ['imagenet-r', 'cifar100_224', 'cub200_224', 'cars196_224']
        self.datasets = ["imagenet-r", "cars196_224"]
        self.checkpoint_dir = checkpoint_dir
        self.study_name = study_name
        self.pruner = pruner 

        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def get_storage(self) -> Optional[RDBStorage]:
        """Get RDB storage backend if checkpointing is enabled."""
        if not self.checkpoint_dir:
            return None
        db_path = os.path.join(self.checkpoint_dir, "optuna_study.db")
        return RDBStorage(
            url=f"sqlite:///{db_path}",
            heartbeat_interval=60,
            grace_period=120)
    
    def load_study(self) -> Optional[optuna.Study]:
        """Load existing study from checkpoint if available."""
        if not self.checkpoint_dir:
            return None
        storage = self.get_storage()
        try:
            return optuna.load_study(study_name=self.study_name, storage=storage)
        except Exception as e:
            logger.warning(f"Could not load existing study: {e}")
            return None
    
    def create_base_args(self, proj_temp, lrate, temp):
        """Create base arguments for training."""
        return {
            'smart_defaults': True,
            'prefix': 'bayesian_opt',
            'test': True,
            'user': 'raoxuan',
            'memory_size': 0,
            'memory_per_class': 0,
            'fixed_memory': False,
            'shuffle': True,
            'model_name': 'sldc',
            'vit_type': 'vit-b-p16-mocov3',
            'weight_decay': 0.0,
            'device': ['0'],
            'lora_rank': 4,
            'lora_type': "sgp_lora",
            'use_projection': True,
            'proj_temp': proj_temp,
            'proj_gamma': 0.0,
            'weight_kind': "log1p",
            'iterations': 2000,
            "warmup_steps": 200,
            "weight_p": 1,
            'kl_gamma': 0.1,
            'osgp_scale': 1.0,
            "trace_k": 1.0,
            'sce_a': 0.5,
            'sce_b': 0.5,
            'seed_list': [1993],
            "seed": 1993,
            'ca_epochs': 5,
            'optimizer': "adamw",
            'lrate': lrate,
            'head_scale': 1.0,
            'batch_size': 24,
            # 'evaluate_final_only': True,
            'gamma_norm': 0.1,
            'gamma_kd': 0.5,
            'kd_type': 'feat',
            'alpha_t': temp,
            'gamma_1': 1e-4,
            'compensate': True,
            'auxiliary_data_path': '/data1/open_datasets/ImageNet-2012/train/',
            'aux_dataset': 'imagenet',
            'auxiliary_data_size': 512,
            'l2_protection': False,
            'l2_protection_lambda': 0.0}
    
    def run_single_experiment(self, dataset: str, args: Dict[str, Any]):
        """Run a single experiment for a given dataset and return its final accuracy."""
        # Set dataset-specific parameters
        args['dataset'] = dataset
        args_copy = args.copy()
        from types import SimpleNamespace
        args = SimpleNamespace(**args_copy)
        args = set_smart_defaults(args)
        args_copy = vars(args)
        
        for task_results, flag in Bayesian_evaluate(args_copy):
            multi_objective_acc = task_results['original_fc'][-1]
            yield multi_objective_acc, flag

    def objective(self, trial: optuna.Trial) -> float:
        proj_temp = trial.suggest_categorical('proj_temp', [1.0])
        lrate = trial.suggest_categorical('lrate', [1e-4, 2e-4, 5e-4])
        temp = trial.suggest_categorical('temp', [0.1, 0.5, 1.0, 2.0])
        args = self.create_base_args(proj_temp, lrate, temp)
        # Run experiments for all datasets
        total_score = 0.0
        total_step = 0
        for i, dataset in enumerate(self.datasets, start=1):
            torch.cuda.empty_cache()
            if dataset == "imagenet-r" or dataset == "cub200_224":
                args['vit_type'] = "vit-b-p16"
            
            elif dataset == "cifar100_224" or dataset == "cars196_224":
                args['vit_type'] = "vit-b-p16-mocov3"

            for acc, flag in self.run_single_experiment(dataset, args):
                total_step += 1
                trial.report(acc, step=total_step)
                if flag == 1:
                    total_score += acc

        return total_score

    def optimize(self) -> None:
        """Run the Bayesian optimization process."""
        # Load or create study
        study = self.load_study() or optuna.create_study(
            direction='maximize',
            sampler=RandomSampler(seed=0),
            storage=self.get_storage(),
            study_name=self.study_name,
            load_if_exists=True)
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Print results
        best_trial = study.best_trial
        logger.info(f"Best trial: Value: {best_trial.value:.4f}, Params: {best_trial.params}")
        
        # Save final checkpoint
        if self.checkpoint_dir:
            logger.info(f"Optimization results saved to {self.checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=str, default="optuna/checkpoints_osgp_0913")
    parser.add_argument("--study_name", type=str, default="sgp_optimization_3")
    args = parser.parse_args()
    
    optim = BayesianOptimizer(
        n_trials=args.n_trials,
        timeout=3600000,
        checkpoint_dir=args.checkpoint_dir,
        study_name=args.study_name
    )
    optim.optimize()