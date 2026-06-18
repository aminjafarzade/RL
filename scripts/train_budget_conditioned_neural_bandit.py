#!/usr/bin/env python
"""Train one neural continuation bandit conditioned on requested cost."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

try:
    from continuation_gate_features import as_bool
    from continuation_gate_features import as_float
    from continuation_gate_features import feature_dict as base_feature_dict
    from continuation_gate_features import read_jsonl
    from export_continuation_gate_dataset import FEATURE_KEYS
    from export_continuation_gate_dataset import validate_feature_keys
    from train_neural_continuation_bandit import FORBIDDEN_INPUT_FIELDS
    from train_neural_continuation_bandit import NeuralContinuationBandit
    from train_neural_continuation_bandit import actual_extra_steps
    from train_neural_continuation_bandit import bandit_loss_from_logits
    from train_neural_continuation_bandit import baseline_results
    from train_neural_continuation_bandit import bootstrap_confidence_intervals
    from train_neural_continuation_bandit import evaluate_policy
    from train_neural_continuation_bandit import feature_names
    from train_neural_continuation_bandit import planned_continuation_steps
    from train_neural_continuation_bandit import resolve_device
    from train_neural_continuation_bandit import save_preprocessor
    from train_neural_continuation_bandit import select_threshold
    from train_neural_continuation_bandit import set_seed
    from train_neural_continuation_bandit import split_by_mode
    from train_neural_continuation_bandit import split_train_validation
    from train_neural_continuation_bandit import train_epoch
except ImportError:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from continuation_gate_features import as_bool
    from continuation_gate_features import as_float
    from continuation_gate_features import feature_dict as base_feature_dict
    from continuation_gate_features import read_jsonl
    from export_continuation_gate_dataset import FEATURE_KEYS
    from export_continuation_gate_dataset import validate_feature_keys
    from train_neural_continuation_bandit import FORBIDDEN_INPUT_FIELDS
    from train_neural_continuation_bandit import NeuralContinuationBandit
    from train_neural_continuation_bandit import actual_extra_steps
    from train_neural_continuation_bandit import bandit_loss_from_logits
    from train_neural_continuation_bandit import baseline_results
    from train_neural_continuation_bandit import bootstrap_confidence_intervals
    from train_neural_continuation_bandit import evaluate_policy
    from train_neural_continuation_bandit import feature_names
    from train_neural_continuation_bandit import planned_continuation_steps
    from train_neural_continuation_bandit import resolve_device
    from train_neural_continuation_bandit import save_preprocessor
    from train_neural_continuation_bandit import select_threshold
    from train_neural_continuation_bandit import set_seed
    from train_neural_continuation_bandit import split_by_mode
    from train_neural_continuation_bandit import split_train_validation
    from train_neural_continuation_bandit import train_epoch


REQUESTED_COST_FEATURE = "requested_cost_lambda"
DEFAULT_TRAIN_COST_LAMBDAS = "0.0,0.0025,0.005,0.01,0.02"
DEFAULT_EVAL_COST_LAMBDAS = "0.0,0.0025,0.005,0.01,0.02"


def parse_lambda_list(value: str) -> list[float]:
    lambdas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not lambdas:
        raise ValueError("at least one cost lambda is required")
    return lambdas


def validate_no_feature_leakage() -> None:
    validate_feature_keys(FEATURE_KEYS)
    feature_keys = set(FEATURE_KEYS) | {REQUESTED_COST_FEATURE}
    leaked = sorted(FORBIDDEN_INPUT_FIELDS.intersection(feature_keys))
    if leaked:
        raise ValueError(f"Leaky fields configured as features: {leaked}")


def feature_dict(row: dict[str, Any]) -> dict[str, Any]:
    features = dict(base_feature_dict(row))
    features[REQUESTED_COST_FEATURE] = as_float(row.get(REQUESTED_COST_FEATURE))
    return features


def budget_conditioned_rewards(
    row: dict[str, Any],
    *,
    harm_penalty: float,
    correctness_reward: float,
) -> dict[str, Any]:
    requested_lambda = as_float(row.get(REQUESTED_COST_FEATURE))
    base_correct = as_bool(row.get("base_correct"))
    continued_correct = as_bool(row.get("continued_correct"))
    steps = actual_extra_steps(row)
    r_stop = float(correctness_reward) * float(base_correct)
    r_continue = (
        float(correctness_reward) * float(continued_correct)
        - requested_lambda * steps
        - float(harm_penalty) * float(base_correct and not continued_correct)
    )
    delta_reward = r_continue - r_stop
    return {
        "requested_cost_lambda": requested_lambda,
        "base_correct": base_correct,
        "continued_correct": continued_correct,
        "actual_extra_steps": steps,
        "planned_extra_steps": planned_continuation_steps(row),
        "r_stop": r_stop,
        "r_continue": r_continue,
        "delta_reward": delta_reward,
        "oracle_continue": delta_reward > 0.0,
    }


def duplicate_rows_for_lambdas(
    rows: list[dict[str, Any]],
    lambdas: list[float],
    *,
    harm_penalty: float,
    correctness_reward: float,
) -> list[dict[str, Any]]:
    duplicated: list[dict[str, Any]] = []
    for row in rows:
        for lambda_value in lambdas:
            conditioned = {**row, REQUESTED_COST_FEATURE: float(lambda_value)}
            duplicated.append(
                {
                    **conditioned,
                    **budget_conditioned_rewards(
                        conditioned,
                        harm_penalty=harm_penalty,
                        correctness_reward=correctness_reward,
                    ),
                }
            )
    return duplicated


def duplicate_rows_by_lambda(
    rows: list[dict[str, Any]],
    lambdas: list[float],
    *,
    harm_penalty: float,
    correctness_reward: float,
) -> dict[float, list[dict[str, Any]]]:
    return {
        float(lambda_value): duplicate_rows_for_lambdas(
            rows,
            [float(lambda_value)],
            harm_penalty=harm_penalty,
            correctness_reward=correctness_reward,
        )
        for lambda_value in lambdas
    }


def fit_feature_preprocessor(rows: list[dict[str, Any]]):
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("scikit-learn is required for feature preprocessing") from exc
    preprocessor = make_pipeline(
        DictVectorizer(sparse=False),
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(),
    )
    preprocessor.fit([feature_dict(row) for row in rows])
    return preprocessor


def transform_features(preprocessor, rows: list[dict[str, Any]]) -> np.ndarray:
    return preprocessor.transform([feature_dict(row) for row in rows]).astype(
        np.float32,
        copy=False,
    )


def build_tensors(
    preprocessor,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    features = torch.tensor(
        transform_features(preprocessor, rows),
        dtype=torch.float32,
        device=device,
    )
    r_stop = torch.tensor(
        [float(row["r_stop"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    r_continue = torch.tensor(
        [float(row["r_continue"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    oracle = torch.tensor(
        [float(row["oracle_continue"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    return {
        "features": features,
        "r_stop": r_stop,
        "r_continue": r_continue,
        "oracle": oracle,
        "delta": r_continue - r_stop,
    }


def score_rows(
    model: NeuralContinuationBandit,
    preprocessor,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> list[float]:
    if not rows:
        return []
    model.eval()
    features = torch.tensor(
        transform_features(preprocessor, rows),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        scores = torch.sigmoid(model(features)).detach().cpu().numpy()
    return [float(score) for score in scores]


def summarize_validation(
    model: NeuralContinuationBandit,
    preprocessor,
    validation_by_lambda: dict[float, list[dict[str, Any]]],
    *,
    threshold_mode: str,
    harm_penalty: float,
    correctness_reward: float,
    device: torch.device,
) -> dict[str, Any]:
    thresholds: dict[float, float] = {}
    metrics_by_lambda: dict[float, dict[str, Any]] = {}
    for lambda_value, rows in validation_by_lambda.items():
        scores = score_rows(model, preprocessor, rows, device)
        if threshold_mode == "per_lambda_validation":
            selected = select_threshold(
                rows,
                scores,
                cost_lambda=lambda_value,
                harm_penalty=harm_penalty,
                correctness_reward=correctness_reward,
            )
            threshold = float(selected["selected_threshold"])
            metrics = selected["selected_validation_metrics"]
        else:
            threshold = 0.5
            metrics = {
                "threshold": threshold,
                **evaluate_policy(
                    rows,
                    scores,
                    threshold,
                    lambda_value,
                    harm_penalty,
                    correctness_reward,
                ),
            }
        thresholds[lambda_value] = threshold
        metrics_by_lambda[lambda_value] = metrics
    mean_utility = mean(
        float(metrics["policy_utility_mean"])
        for metrics in metrics_by_lambda.values()
    )
    mean_accuracy = mean(
        float(metrics["policy_accuracy"])
        for metrics in metrics_by_lambda.values()
    )
    mean_extra_steps = mean(
        float(metrics["avg_extra_steps"])
        for metrics in metrics_by_lambda.values()
    )
    total_harms = sum(int(metrics["harms"]) for metrics in metrics_by_lambda.values())
    return {
        "thresholds": thresholds,
        "metrics_by_lambda": metrics_by_lambda,
        "mean_policy_utility": mean_utility,
        "mean_policy_accuracy": mean_accuracy,
        "mean_avg_extra_steps": mean_extra_steps,
        "total_harms": total_harms,
    }


def _json_lambda_map(mapping: dict[float, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in sorted(mapping.items())}


def final_eval_for_lambdas(
    model: NeuralContinuationBandit,
    preprocessor,
    final_by_lambda: dict[float, list[dict[str, Any]]],
    thresholds: dict[float, float],
    *,
    harm_penalty: float,
    correctness_reward: float,
    bootstrap_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[float, list[float]]]:
    results_by_lambda: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    scores_by_lambda: dict[float, list[float]] = {}
    for lambda_value, rows in sorted(final_by_lambda.items()):
        threshold = float(thresholds.get(lambda_value, 0.5))
        scores = score_rows(model, preprocessor, rows, device)
        scores_by_lambda[lambda_value] = scores
        baselines = baseline_results(
            rows,
            cost_lambda=lambda_value,
            harm_penalty=harm_penalty,
            correctness_reward=correctness_reward,
        )
        learned = {
            "policy": "budget_conditioned_neural_policy",
            **evaluate_policy(
                rows,
                scores,
                threshold,
                lambda_value,
                harm_penalty,
                correctness_reward,
            ),
        }
        baselines["budget_conditioned_neural_policy"] = learned
        baselines["bootstrap_ci"] = bootstrap_confidence_intervals(
            rows,
            scores,
            threshold,
            cost_lambda=lambda_value,
            harm_penalty=harm_penalty,
            correctness_reward=correctness_reward,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        results_by_lambda[str(lambda_value)] = {
            "eval_cost_lambda": lambda_value,
            "chosen_threshold": threshold,
            **baselines,
        }
        table.append(
            {
                "eval_cost_lambda": lambda_value,
                "chosen_threshold": threshold,
                "policy_accuracy": learned["policy_accuracy"],
                "policy_utility_mean": learned["policy_utility_mean"],
                "trigger_rate": learned["trigger_rate"],
                "avg_extra_steps": learned["avg_extra_steps"],
                "avg_planned_extra_steps": learned["avg_planned_extra_steps"],
                "fixes": learned["fixes"],
                "harms": learned["harms"],
                "net_fix_minus_harm": learned["net_fix_minus_harm"],
                "missed_fixes_vs_always": learned["missed_fixes_vs_always"],
                "avoided_harms_vs_always": learned["avoided_harms_vs_always"],
                "compute_saved_vs_always_actual": (
                    learned["compute_saved_vs_always_actual"]
                ),
                "compute_saved_vs_always_planned": (
                    learned["compute_saved_vs_always_planned"]
                ),
                "policy_utility_gain_vs_base": (
                    learned["policy_utility_gain_vs_base"]
                ),
                "policy_utility_gain_vs_always": (
                    learned["policy_utility_gain_vs_always"]
                ),
                "policy_regret_vs_oracle": learned["policy_regret_vs_oracle"],
                "precision_for_positive_delta": (
                    learned["precision_for_positive_delta"]
                ),
                "recall_for_positive_delta": learned["recall_for_positive_delta"],
                "auc": learned["auc"],
            }
        )
    return results_by_lambda, table, scores_by_lambda


def write_predictions(
    path: Path,
    final_by_lambda: dict[float, list[dict[str, Any]]],
    scores_by_lambda: dict[float, list[float]],
    thresholds: dict[float, float],
    *,
    harm_penalty: float,
    correctness_reward: float,
) -> None:
    with path.open("w") as f:
        for lambda_value, rows in sorted(final_by_lambda.items()):
            threshold = float(thresholds.get(lambda_value, 0.5))
            for row, score in zip(rows, scores_by_lambda[lambda_value]):
                rewards = budget_conditioned_rewards(
                    row,
                    harm_penalty=harm_penalty,
                    correctness_reward=correctness_reward,
                )
                policy_continue = float(score) >= threshold
                oracle_continue = bool(rewards["oracle_continue"])
                record = {
                    "example_id": row.get("example_id"),
                    "eval_cost_lambda": float(lambda_value),
                    "requested_cost_lambda": float(row[REQUESTED_COST_FEATURE]),
                    "score": float(score),
                    "threshold": threshold,
                    "triggered": bool(policy_continue),
                    "policy_continue": bool(policy_continue),
                    "oracle_continue": oracle_continue,
                    "base_correct": rewards["base_correct"],
                    "continued_correct": rewards["continued_correct"],
                    "actual_extra_steps": rewards["actual_extra_steps"],
                    "planned_extra_steps": rewards["planned_extra_steps"],
                    "r_stop": rewards["r_stop"],
                    "r_continue": rewards["r_continue"],
                    "delta_reward": rewards["delta_reward"],
                    "policy_correct": bool(
                        rewards["continued_correct"]
                        if policy_continue
                        else rewards["base_correct"]
                    ),
                    "policy_utility": float(
                        rewards["r_continue"] if policy_continue else rewards["r_stop"]
                    ),
                    "oracle_utility": float(
                        rewards["r_continue"] if oracle_continue else rewards["r_stop"]
                    ),
                }
                f.write(json.dumps(record, sort_keys=True) + "\n")


def write_pretty_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# One budget-conditioned neural policy controls the accuracy/compute tradeoff at test time.",
        "",
        "| lambda | accuracy | utility | trigger rate | avg extra steps | compute saved vs always | fixes | harms | net |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["final_eval_table"]:
        lines.append(
            "| {lam:.4g} | {acc:.4f} | {utility:.4f} | {trigger:.4f} | {steps:.4f} | {saved:.4f} | {fixes} | {harms} | {net} |".format(
                lam=row["eval_cost_lambda"],
                acc=row["policy_accuracy"],
                utility=row["policy_utility_mean"],
                trigger=row["trigger_rate"],
                steps=row["avg_extra_steps"],
                saved=row["compute_saved_vs_always_actual"],
                fixes=row["fixes"],
                harms=row["harms"],
                net=row["net_fix_minus_harm"],
            )
        )
    lines.extend(
        [
            "",
            "Unlike the fixed-lambda neural bandits, this model uses "
            "`requested_cost_lambda` as an input and reuses the same checkpoint "
            "for all operating points.",
            "",
            "## Training Config",
            "",
            f"- Train cost lambdas: `{result['config']['train_cost_lambdas']}`",
            f"- Eval cost lambdas: `{result['config']['eval_cost_lambdas']}`",
            f"- Threshold mode: `{result['config']['threshold_mode']}`",
            f"- Hidden dim: `{result['config']['hidden_dim']}`",
            f"- Num layers: `{result['config']['num_layers']}`",
            f"- Dropout: `{result['config']['dropout']}`",
            f"- Learning rate: `{result['config']['lr']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def train_budget_conditioned_bandit(
    *,
    dataset: str | Path,
    output_dir: str | Path,
    train_cost_lambdas: list[float],
    eval_cost_lambdas: list[float],
    harm_penalty: float = 0.0,
    correctness_reward: float = 1.0,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    batch_size: int = 128,
    validation_fraction: float = 0.2,
    patience: int = 40,
    entropy_coef: float = 0.01,
    aux_bce_weight: float = 0.25,
    threshold_mode: str = "per_lambda_validation",
    bootstrap_samples: int = 100,
    seed: int = 123,
    device: str = "auto",
) -> dict[str, Any]:
    validate_no_feature_leakage()
    set_seed(seed)
    torch_device = resolve_device(device)
    rows = read_jsonl(dataset)
    if not rows:
        raise ValueError("dataset is empty")

    train_pool_raw, final_eval_raw, fallback_split = split_by_mode(rows)
    if fallback_split:
        print(
            "warning: mode split missing or incomplete; using first 80% train / last 20% eval",
            file=sys.stderr,
        )
    train_raw, validation_raw = split_train_validation(
        train_pool_raw,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    train_rows = duplicate_rows_for_lambdas(
        train_raw,
        train_cost_lambdas,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    validation_by_lambda = duplicate_rows_by_lambda(
        validation_raw,
        eval_cost_lambdas,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    final_by_lambda = duplicate_rows_by_lambda(
        final_eval_raw,
        eval_cost_lambdas,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )

    preprocessor = fit_feature_preprocessor(train_rows)
    train_tensors = build_tensors(preprocessor, train_rows, torch_device)
    input_dim = int(train_tensors["features"].shape[1])
    model = NeuralContinuationBandit(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.jsonl"
    best_state = copy.deepcopy(model.state_dict())
    best_thresholds = {float(value): 0.5 for value in eval_cost_lambdas}
    best_key = None
    best_epoch = 0
    best_validation: dict[str, Any] | None = None
    stale_epochs = 0

    with history_path.open("w") as history_file:
        for epoch in range(1, epochs + 1):
            train_metrics = train_epoch(
                model,
                train_tensors,
                optimizer,
                batch_size=batch_size,
                entropy_coef=entropy_coef,
                aux_bce_weight=aux_bce_weight,
            )
            validation = summarize_validation(
                model,
                preprocessor,
                validation_by_lambda,
                threshold_mode=threshold_mode,
                harm_penalty=harm_penalty,
                correctness_reward=correctness_reward,
                device=torch_device,
            )
            key = (
                validation["mean_policy_utility"],
                validation["mean_policy_accuracy"],
                -validation["mean_avg_extra_steps"],
                -validation["total_harms"],
            )
            improved = best_key is None or key > best_key
            if improved:
                best_key = key
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_thresholds = dict(validation["thresholds"])
                best_validation = validation
                stale_epochs = 0
            else:
                stale_epochs += 1
            history = {
                "epoch": epoch,
                **train_metrics,
                "validation_mean_policy_utility": validation["mean_policy_utility"],
                "validation_mean_policy_accuracy": validation["mean_policy_accuracy"],
                "validation_mean_avg_extra_steps": validation["mean_avg_extra_steps"],
                "validation_total_harms": validation["total_harms"],
                "thresholds": _json_lambda_map(validation["thresholds"]),
                "best_epoch": best_epoch,
            }
            history_file.write(json.dumps(history, sort_keys=True) + "\n")
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    final_results, final_table, final_scores = final_eval_for_lambdas(
        model,
        preprocessor,
        final_by_lambda,
        best_thresholds,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        device=torch_device,
    )

    model_path = output_dir / "budget_conditioned_neural_bandit_policy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "thresholds": _json_lambda_map(best_thresholds),
            "feature_keys": [*FEATURE_KEYS, REQUESTED_COST_FEATURE],
            "train_cost_lambdas": train_cost_lambdas,
            "eval_cost_lambdas": eval_cost_lambdas,
        },
        model_path,
    )
    preprocessor_path = output_dir / "feature_preprocessor.joblib"
    preprocessor_saved = save_preprocessor(preprocessor, preprocessor_path)
    feature_names_path = output_dir / "feature_names.json"
    feature_names_path.write_text(json.dumps(feature_names(preprocessor), indent=2) + "\n")
    prediction_path = output_dir / "budget_conditioned_eval_predictions.jsonl"
    write_predictions(
        prediction_path,
        final_by_lambda,
        final_scores,
        best_thresholds,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )

    result = {
        "status": "ok",
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "selected_thresholds": _json_lambda_map(best_thresholds),
        "best_epoch": best_epoch,
        "split": {
            "total_rows": len(rows),
            "train_pool_rows": len(train_pool_raw),
            "train_rows_base": len(train_raw),
            "train_rows_duplicated": len(train_rows),
            "validation_rows_base": len(validation_raw),
            "validation_rows_per_lambda": {
                str(key): len(value) for key, value in validation_by_lambda.items()
            },
            "final_eval_rows_base": len(final_eval_raw),
            "final_eval_rows_per_lambda": {
                str(key): len(value) for key, value in final_by_lambda.items()
            },
            "fallback_mode_split": fallback_split,
        },
        "config": {
            "train_cost_lambdas": train_cost_lambdas,
            "eval_cost_lambdas": eval_cost_lambdas,
            "harm_penalty": harm_penalty,
            "correctness_reward": correctness_reward,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_fraction": validation_fraction,
            "patience": patience,
            "entropy_coef": entropy_coef,
            "aux_bce_weight": aux_bce_weight,
            "threshold_mode": threshold_mode,
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "device": str(torch_device),
        },
        "validation": (
            {
                "mean_policy_utility": best_validation["mean_policy_utility"],
                "mean_policy_accuracy": best_validation["mean_policy_accuracy"],
                "mean_avg_extra_steps": best_validation["mean_avg_extra_steps"],
                "total_harms": best_validation["total_harms"],
                "metrics_by_lambda": _json_lambda_map(
                    best_validation["metrics_by_lambda"]
                ),
            }
            if best_validation is not None
            else None
        ),
        "final_eval_by_lambda": final_results,
        "final_eval_table": final_table,
        "same_checkpoint_eval_lambdas": eval_cost_lambdas,
        "artifacts": {
            "model": str(model_path),
            "feature_preprocessor": str(preprocessor_path) if preprocessor_saved else None,
            "feature_names": str(feature_names_path),
            "training_history": str(history_path),
            "result_json": str(output_dir / "budget_conditioned_result.json"),
            "pretty_markdown": str(output_dir / "budget_conditioned_result_pretty.md"),
            "eval_predictions": str(prediction_path),
        },
    }
    result_path = output_dir / "budget_conditioned_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_pretty_markdown(output_dir / "budget_conditioned_result_pretty.md", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_cost_lambdas", default=DEFAULT_TRAIN_COST_LAMBDAS)
    parser.add_argument("--eval_cost_lambdas", default=DEFAULT_EVAL_COST_LAMBDAS)
    parser.add_argument("--harm_penalty", type=float, default=0.0)
    parser.add_argument("--correctness_reward", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--aux_bce_weight", type=float, default=0.25)
    parser.add_argument(
        "--threshold_mode",
        choices=["per_lambda_validation", "fixed_0_5"],
        default="per_lambda_validation",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = train_budget_conditioned_bandit(
        dataset=args.dataset,
        output_dir=args.output_dir,
        train_cost_lambdas=parse_lambda_list(args.train_cost_lambdas),
        eval_cost_lambdas=parse_lambda_list(args.eval_cost_lambdas),
        harm_penalty=args.harm_penalty,
        correctness_reward=args.correctness_reward,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        entropy_coef=args.entropy_coef,
        aux_bce_weight=args.aux_bce_weight,
        threshold_mode=args.threshold_mode,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result["final_eval_table"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
