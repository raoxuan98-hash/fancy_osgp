"""Optuna-based hyperparameter optimisation for CLIP continual learning."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import optuna

from main_clip import build_parser
from trainer_clip import train


# ---------------------------------------------------------------------------
# Dataclasses mirroring the JSON payload emitted at the end of the search.
# ---------------------------------------------------------------------------


@dataclass
class TrialMetrics:
    """Summary statistics computed from raw per-seed metrics."""

    mean: float
    std: float
    minimum: float
    maximum: float
    values: Sequence[float] = field(default_factory=list)


@dataclass
class TrialResult:
    """Container storing the outcome of an Optuna trial."""

    index: int
    parameters: Mapping[str, Any]
    duration_sec: float
    metrics: Mapping[str, TrialMetrics] = field(default_factory=dict)
    failed: bool = False
    error_message: Optional[str] = None
    metric_key: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "index": self.index,
            "parameters": dict(self.parameters),
            "duration_sec": self.duration_sec,
            "failed": self.failed,
        }
        if self.metric_key is not None:
            payload["metric_key"] = self.metric_key
        if self.error_message:
            payload["error_message"] = self.error_message
        if self.metrics:
            payload["metrics"] = {
                name: asdict(metric)
                for name, metric in self.metrics.items()
            }
        return payload


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _summarise_seed_results(all_results: Mapping[str, Any]) -> Dict[str, TrialMetrics]:
    """Extract aggregate statistics from the raw ``train`` outputs."""

    metric_buckets: MutableMapping[str, list[float]] = {}

    for seed_name, seed_payload in (all_results or {}).items():
        if not isinstance(seed_payload, Mapping):
            logging.debug("Seed %s produced non-dict payload: %r", seed_name, seed_payload)
            continue

        for section_name, metrics in seed_payload.items():
            if not isinstance(metrics, Mapping):
                continue

            for metric_name, value in metrics.items():
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    logging.debug(
                        "Skipping non-numeric metric %s.%s=%r for seed %s",
                        section_name,
                        metric_name,
                        value,
                        seed_name,
                    )
                    continue

                key = f"{section_name}.{metric_name}"
                metric_buckets.setdefault(key, []).append(numeric_value)

    summary: Dict[str, TrialMetrics] = {}
    for metric_name, values in metric_buckets.items():
        arr = np.asarray(values, dtype=float)
        summary[metric_name] = TrialMetrics(
            mean=float(arr.mean()) if arr.size else math.nan,
            std=float(arr.std(ddof=0)) if arr.size else math.nan,
            minimum=float(arr.min()) if arr.size else math.nan,
            maximum=float(arr.max()) if arr.size else math.nan,
            values=[float(v) for v in values],
        )

    return summary


def _default_search_space() -> Mapping[str, Sequence[Any]]:
    return {
        "lrate": (1e-4, 5e-4, 1e-3),
        "weight_temp": (1.0, 4.0),
        "iterations": (600, 1200),
        "gamma_kd": (0.5, 1.0, 2.0),
        "sgp_soft_projection": (True, False),
    }


def _prepare_base_arguments(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    parser = build_parser()
    base_args = vars(parser.parse_args([]))
    base_args.update(overrides)
    return base_args


def _coerce_device(device: Optional[str]) -> Optional[list[str]]:
    if device is None:
        return None
    if device.strip() == "":
        return None
    return [device]


def _select_metric(summary: Mapping[str, TrialMetrics], preferred: Optional[str]) -> Tuple[str, TrialMetrics]:
    if preferred:
        metric = summary.get(preferred)
        if metric is None:
            raise KeyError(
                f"Requested metric '{preferred}' was not produced by the trainer."
            )
        return preferred, metric

    # Attempt to pick a sensible default: prefer average accuracies, then final task.
    priority_keywords = (
        "average_across_tasks",
        "average_accuracies",
        "avg",
        "final_task",
        "last_task",
        "acc",
    )

    for keyword in priority_keywords:
        for name, metric in summary.items():
            if keyword in name:
                return name, metric

    if summary:
        name, metric = next(iter(summary.items()))
        logging.warning(
            "Falling back to first available metric '%s' for optimisation.",
            name,
        )
        return name, metric

    raise RuntimeError("Trainer did not return any numeric metrics to optimise.")


# ---------------------------------------------------------------------------
# Optuna integration
# ---------------------------------------------------------------------------


def _objective(
    trial: optuna.Trial,
    *,
    base_args: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    metric_name: Optional[str],
) -> float:
    params: Dict[str, Any] = {}
    for name, choices in search_space.items():
        params[name] = trial.suggest_categorical(name, list(choices))

    args = deepcopy(dict(base_args))
    args.update(params)
    args["prefix"] = f"optuna_trial_{trial.number}"

    logging.info("[Optuna trial %d] parameters=%s", trial.number, params)

    start_time = time.perf_counter()
    try:
        raw_results = train(args)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        duration = time.perf_counter() - start_time
        trial.set_user_attr("duration_sec", duration)
        trial.set_user_attr("metrics", {})
        trial.set_user_attr("failed", True)
        trial.set_user_attr("error_message", str(exc))
        logging.exception("Trial %d failed with an exception", trial.number)
        raise

    duration = time.perf_counter() - start_time
    summary = _summarise_seed_results(raw_results)

    try:
        metric_key, metric = _select_metric(summary, metric_name)
    except Exception as exc:  # pragma: no cover - defensive path
        trial.set_user_attr("duration_sec", duration)
        trial.set_user_attr(
            "metrics",
            {name: asdict(value) for name, value in summary.items()},
        )
        trial.set_user_attr("failed", True)
        trial.set_user_attr("error_message", str(exc))
        logging.exception("Trial %d did not produce a usable metric", trial.number)
        raise optuna.TrialPruned(str(exc))

    trial.set_user_attr("duration_sec", duration)
    trial.set_user_attr(
        "metrics", {name: asdict(value) for name, value in summary.items()}
    )
    trial.set_user_attr("failed", False)
    trial.set_user_attr("error_message", None)
    trial.set_user_attr("metric_key", metric_key)

    if math.isnan(metric.mean):
        raise optuna.TrialPruned(
            f"Metric '{metric_key}' evaluated to NaN for trial {trial.number}."
        )

    return float(metric.mean)


def _compile_results(study: optuna.Study) -> Sequence[TrialResult]:
    results: list[TrialResult] = []
    for trial in study.trials:
        metrics_payload = trial.user_attrs.get("metrics", {})
        metrics = {
            name: TrialMetrics(**metric_dict)
            for name, metric_dict in metrics_payload.items()
        }
        result = TrialResult(
            index=trial.number,
            parameters=trial.params,
            duration_sec=float(trial.user_attrs.get("duration_sec", math.nan)),
            metrics=metrics,
            failed=bool(trial.user_attrs.get("failed", trial.state != optuna.trial.TrialState.COMPLETE)),
            error_message=trial.user_attrs.get("error_message"),
            metric_key=trial.user_attrs.get("metric_key"),
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna search over CLIP continual-learning hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clip_hparam_optuna_results.json"),
        help="Destination JSON file for aggregated results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device identifier passed to the trainer (e.g. -1 for CPU).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of random seeds to evaluate. Defaults to parser defaults.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials to run. Defaults to exhaustive enumeration of the discrete space.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Stop the optimisation after the given number of seconds.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Fully qualified metric key to optimise (e.g. average_accuracies.zeroshot).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=("maximize", "minimize"),
        default="maximize",
        help="Optimisation direction passed to Optuna.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional Optuna study name (useful when reusing a storage backend).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///study.db). If omitted, in-memory storage is used.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=("tpe", "random"),
        default="tpe",
        help="Sampler to use for navigating the discrete search space.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        help="Configure logging verbosity for the search runner.",
    )

    return parser


def _build_sampler(name: str) -> optuna.samplers.BaseSampler:
    if name == "random":
        return optuna.samplers.RandomSampler()
    return optuna.samplers.TPESampler(multivariate=True)


def main() -> None:
    cli_parser = _build_cli_parser()
    cli_args = cli_parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, cli_args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    overrides: Dict[str, Any] = {}
    device_override = _coerce_device(cli_args.device)
    if device_override is not None:
        overrides["device"] = device_override

    if cli_args.seeds:
        overrides["seed_list"] = list(cli_args.seeds)

    base_args = _prepare_base_arguments(overrides)
    search_space = _default_search_space()

    sampler = _build_sampler(cli_args.sampler)

    study = optuna.create_study(
        study_name=cli_args.study_name,
        storage=cli_args.storage,
        direction=cli_args.direction,
        sampler=sampler,
        load_if_exists=bool(cli_args.study_name and cli_args.storage),
    )

    study.optimize(
        lambda trial: _objective(
            trial,
            base_args=base_args,
            search_space=search_space,
            metric_name=cli_args.metric,
        ),
        n_trials=cli_args.n_trials,
        timeout=cli_args.timeout,
    )

    results = _compile_results(study)

    output_path = cli_args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump([result.to_json() for result in results], fp, indent=2)

    if study.trials:
        best_trial = study.best_trial
        logging.info(
            "Best trial %d (value=%f): %s",
            best_trial.number,
            best_trial.value,
            best_trial.params,
        )
    logging.info("Saved %d trial results to %s", len(results), output_path)


if __name__ == "__main__":
    main()

