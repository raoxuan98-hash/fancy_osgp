"""Bayesian hyperparameter optimization for SubspaceLoRA-CLIP, targeting final average zeroshot accuracy."""

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
    """Optimize the average of all per-task average zeroshot accuracies (i.e., mean of history['zeroshot_acc'])."""
    params = {name: trial.suggest_categorical(name, choices) for name, choices in search_space.items()}
    args = deepcopy(base_args)
    args.update(params)
    args["prefix"] = f"optuna_trial_{trial.number}"

    start = time.perf_counter()
    try:
        raw_results = train(args)  # Expected: {seed: history_dict}
        duration = time.perf_counter() - start

        # For each seed, compute mean of all per-task average zeroshot accuracies
        mean_accuracies_per_seed = []
        for seed, history in (raw_results or {}).items():
            if not isinstance(history, dict):
                continue
            zeroshot_acc_list = history.get("zeroshot_acc")
            if zeroshot_acc_list and len(zeroshot_acc_list) > 0:
                # Average over all tasks' average zeroshot accuracies
                avg_inc_acc = float(np.mean(zeroshot_acc_list))
                mean_accuracies_per_seed.append(avg_inc_acc)

        if not mean_accuracies_per_seed:
            raise ValueError("No valid zeroshot_acc history found in any seed.")

        final_objective = float(np.mean(mean_accuracies_per_seed))  # Mean across seeds

        trial.set_user_attr("duration_sec", duration)
        trial.set_user_attr("mean_accuracies_per_seed", mean_accuracies_per_seed)
        trial.set_user_attr("final_objective", final_objective)
        trial.set_user_attr("failed", False)

        if math.isnan(final_objective):
            raise optuna.TrialPruned("Objective is NaN")

        return final_objective  # Maximize this

    except Exception as e:
        duration = time.perf_counter() - start
        trial.set_user_attr("duration_sec", duration)
        trial.set_user_attr("failed", True)
        trial.set_user_attr("error_message", str(e))
        logging.exception(f"Trial {trial.number} failed: {e}")
        raise


def prepare_args(overrides: Dict[str, Any]) -> Dict[str, Any]:
    parser = build_parser()
    args = vars(parser.parse_args([]))
    args.update(overrides)
    return args


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("clip_optuna_zeroshot_results.json"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--study-name", type=str, default="clip-zeroshot-optimization")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    # Base training args
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.seeds:
        overrides["seed_list"] = args.seeds
    base_args = prepare_args(overrides)

    # Discrete search space (aligned with your learner)
    search_space = {
        "lrate": [1e-4, 5e-4, 1e-3],
        "weight_temp": [1.0, 4.0],
        "iterations": [600, 1200],
        "gamma_kd": [0.5, 1.0, 2.0],
        "sgp_soft_projection": [True, False],
    }

    # Storage & sampler
    storage = args.storage or f"sqlite:///{args.study_name}.db"
    logging.info(f"Using storage: {storage}")

    sampler = (
        optuna.samplers.RandomSampler()
        if args.sampler == "random"
        else optuna.samplers.TPESampler(multivariate=True)
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",  # ← Crucial: we maximize accuracy
        sampler=sampler,
        load_if_exists=True,
    )
    logging.info(f"Loaded {len(study.trials)} existing trials.")

    # Run optimization
    study.optimize(
        lambda t: objective(t, base_args, search_space),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    # Save full results
    results = []
    for trial in study.trials:
        results.append({
            "index": trial.number,
            "parameters": trial.params,
            "value": trial.value,
            "duration_sec": trial.user_attrs.get("duration_sec", math.nan),
            "failed": trial.user_attrs.get("failed", trial.state != optuna.trial.TrialState.COMPLETE),
            "error_message": trial.user_attrs.get("error_message"),
            "final_accuracies": trial.user_attrs.get("final_accuracies", []),
            "mean_final_acc": trial.user_attrs.get("mean_final_acc"),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    if study.best_trial:
        logging.info(f"Best trial {study.best_trial.number}: value={study.best_trial.value:.4f}, params={study.best_trial.params}")
    logging.info(f"Saved {len(results)} trials to {args.output}")


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    main()