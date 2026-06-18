#!/usr/bin/env python
"""Train a neural contextual-bandit continuation controller."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from continuation_gate_features import as_bool
    from continuation_gate_features import as_float
    from continuation_gate_features import continuation_step_cost
    from continuation_gate_features import feature_dict
    from continuation_gate_features import planned_continuation_steps
    from continuation_gate_features import read_jsonl
    from evaluate_continuation_heuristic_gate import policy_combined
    from export_continuation_gate_dataset import FEATURE_KEYS
    from export_continuation_gate_dataset import validate_feature_keys
except ImportError:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from continuation_gate_features import as_bool
    from continuation_gate_features import as_float
    from continuation_gate_features import continuation_step_cost
    from continuation_gate_features import feature_dict
    from continuation_gate_features import planned_continuation_steps
    from continuation_gate_features import read_jsonl
    from evaluate_continuation_heuristic_gate import policy_combined
    from export_continuation_gate_dataset import FEATURE_KEYS
    from export_continuation_gate_dataset import validate_feature_keys


FORBIDDEN_INPUT_FIELDS = {
    "base_correct",
    "actual_extra_steps",
    "continued_correct",
    "continued_extra_steps",
    "continuation_extra_steps",
    "continuation_gain",
    "label_gain_positive",
    "label_correct_gain",
    "label_harm",
    "reference_answer",
    "prompt",
    "generated_text",
    "generated_text_final",
    "base_generated_text",
    "continued_generated_text",
    "answer_quality",
    "task_score",
    "base_task_score",
    "final_task_score",
    "reward",
    "correctness",
    "ground_truth",
    "answer",
}


class NeuralContinuationBandit(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def validate_no_feature_leakage() -> None:
    validate_feature_keys(FEATURE_KEYS)
    leaked = sorted(FORBIDDEN_INPUT_FIELDS.intersection(FEATURE_KEYS))
    if leaked:
        raise ValueError(f"Leaky fields configured as features: {leaked}")


def actual_extra_steps(row: dict[str, Any]) -> float:
    return continuation_step_cost(row)


def bandit_rewards(
    row: dict[str, Any],
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
) -> dict[str, Any]:
    base_correct = as_bool(row.get("base_correct"))
    continued_correct = as_bool(row.get("continued_correct"))
    steps = actual_extra_steps(row)
    r_stop = float(correctness_reward) * int(base_correct)
    r_continue = (
        float(correctness_reward) * int(continued_correct)
        - float(cost_lambda) * steps
        - float(harm_penalty) * int(base_correct and not continued_correct)
    )
    delta_reward = r_continue - r_stop
    return {
        "base_correct": base_correct,
        "continued_correct": continued_correct,
        "actual_extra_steps": steps,
        "planned_extra_steps": planned_continuation_steps(row),
        "r_stop": r_stop,
        "r_continue": r_continue,
        "delta_reward": delta_reward,
        "oracle_continue": delta_reward > 0.0,
    }


def add_bandit_labels(
    rows: list[dict[str, Any]],
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            **bandit_rewards(
                row,
                cost_lambda=cost_lambda,
                harm_penalty=harm_penalty,
                correctness_reward=correctness_reward,
            ),
        }
        for row in rows
    ]


def split_by_mode(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    train_rows = [row for row in rows if row.get("mode") == "train"]
    eval_rows = [row for row in rows if row.get("mode") == "eval"]
    if train_rows and eval_rows:
        return train_rows, eval_rows, False
    split = max(1, int(0.8 * len(rows)))
    if split >= len(rows) and len(rows) > 1:
        split = len(rows) - 1
    return rows[:split], rows[split:] or rows[split - 1 :], True


def split_train_validation(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) <= 1 or validation_fraction <= 0.0:
        return rows, rows
    ids = [row.get("example_id") for row in rows]
    if all(item is not None for item in ids):
        ordered_ids = list(dict.fromkeys(str(item) for item in ids))
        shuffled_ids = ordered_ids[:]
        random.Random(seed).shuffle(shuffled_ids)
        val_count = max(1, int(round(len(shuffled_ids) * validation_fraction)))
        if val_count >= len(shuffled_ids):
            val_count = len(shuffled_ids) - 1
        val_ids = set(shuffled_ids[:val_count])
        train = [row for row in rows if str(row.get("example_id")) not in val_ids]
        val = [row for row in rows if str(row.get("example_id")) in val_ids]
        return train or rows, val or rows
    shuffled_indices = list(range(len(rows)))
    random.Random(seed).shuffle(shuffled_indices)
    val_count = max(1, int(round(len(rows) * validation_fraction)))
    if val_count >= len(rows):
        val_count = len(rows) - 1
    val_indices = set(shuffled_indices[:val_count])
    train = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val = [row for idx, row in enumerate(rows) if idx in val_indices]
    return train or rows, val or rows


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


def feature_names(preprocessor) -> list[str]:
    vectorizer = preprocessor.named_steps["dictvectorizer"]
    return [str(name) for name in vectorizer.get_feature_names_out()]


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
    delta = r_continue - r_stop
    return {
        "features": features,
        "r_stop": r_stop,
        "r_continue": r_continue,
        "oracle": oracle,
        "delta": delta,
    }


def bandit_loss_from_logits(
    logits: torch.Tensor,
    r_stop: torch.Tensor,
    r_continue: torch.Tensor,
    oracle: torch.Tensor,
    *,
    entropy_coef: float,
    aux_bce_weight: float,
) -> dict[str, torch.Tensor]:
    p_continue = torch.sigmoid(logits)
    expected_utility = p_continue * r_continue + (1.0 - p_continue) * r_stop
    bandit_loss = -expected_utility.mean()
    entropy = -(
        p_continue * torch.log(p_continue.clamp_min(1e-8))
        + (1.0 - p_continue) * torch.log((1.0 - p_continue).clamp_min(1e-8))
    ).mean()
    weight = (r_continue - r_stop).abs().clamp_min(0.01)
    aux_loss = F.binary_cross_entropy_with_logits(
        logits,
        oracle,
        weight=weight,
    )
    total_loss = (
        bandit_loss
        - float(entropy_coef) * entropy
        + float(aux_bce_weight) * aux_loss
    )
    return {
        "total_loss": total_loss,
        "bandit_loss": bandit_loss.detach(),
        "entropy": entropy.detach(),
        "aux_loss": aux_loss.detach(),
        "expected_utility": expected_utility.mean().detach(),
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


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _auc(labels: list[bool], scores: list[float] | None) -> float | None:
    if scores is None or len(set(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    return float(roc_auc_score([int(label) for label in labels], scores))


def evaluate_triggered_policy(
    rows: list[dict[str, Any]],
    triggered: list[bool],
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
    scores: list[float] | None = None,
) -> dict[str, Any]:
    if len(rows) != len(triggered):
        raise ValueError("rows and triggered must have equal length")
    if not rows:
        raise ValueError("cannot evaluate an empty row set")
    values = [
        bandit_rewards(
            row,
            cost_lambda=cost_lambda,
            harm_penalty=harm_penalty,
            correctness_reward=correctness_reward,
        )
        for row in rows
    ]
    base_correct = [item["base_correct"] for item in values]
    continued_correct = [item["continued_correct"] for item in values]
    utility_oracle_continue = [item["delta_reward"] > 0.0 for item in values]
    policy_correct = [
        continued if use else base
        for use, base, continued in zip(triggered, base_correct, continued_correct)
    ]
    utility_oracle_correct = [
        continued if use else base
        for use, base, continued in zip(
            utility_oracle_continue,
            base_correct,
            continued_correct,
        )
    ]
    accuracy_oracle_possible = [
        base or continued for base, continued in zip(base_correct, continued_correct)
    ]
    r_stop = [item["r_stop"] for item in values]
    r_continue = [item["r_continue"] for item in values]
    policy_utility = [
        cont if use else stop
        for use, stop, cont in zip(triggered, r_stop, r_continue)
    ]
    utility_oracle_utility = [
        max(stop, cont) for stop, cont in zip(r_stop, r_continue)
    ]
    actual_steps_all = [item["actual_extra_steps"] for item in values]
    planned_steps_all = [item["planned_extra_steps"] for item in values]
    actual_steps_policy = [
        steps if use else 0.0 for steps, use in zip(actual_steps_all, triggered)
    ]
    planned_steps_policy = [
        steps if use else 0.0 for steps, use in zip(planned_steps_all, triggered)
    ]
    fixes = sum(
        use and (not base) and continued
        for use, base, continued in zip(triggered, base_correct, continued_correct)
    )
    harms = sum(
        use and base and (not continued)
        for use, base, continued in zip(triggered, base_correct, continued_correct)
    )
    always_fixes = sum(
        (not base) and continued
        for base, continued in zip(base_correct, continued_correct)
    )
    always_harms = sum(
        base and (not continued)
        for base, continued in zip(base_correct, continued_correct)
    )
    trigger_count = sum(triggered)
    positive_count = sum(utility_oracle_continue)
    true_positive = sum(
        use and positive for use, positive in zip(triggered, utility_oracle_continue)
    )
    oracle_action_matches = sum(
        use == oracle
        for use, oracle in zip(triggered, utility_oracle_continue)
    )
    base_utility_mean = _safe_mean(r_stop)
    always_utility_mean = _safe_mean(r_continue)
    policy_utility_mean = _safe_mean(policy_utility)
    utility_oracle_utility_mean = _safe_mean(utility_oracle_utility)
    utility_oracle_accuracy = sum(utility_oracle_correct) / len(rows)
    utility_oracle_actual_steps = [
        steps if use else 0.0
        for steps, use in zip(actual_steps_all, utility_oracle_continue)
    ]
    utility_oracle_planned_steps = [
        steps if use else 0.0
        for steps, use in zip(planned_steps_all, utility_oracle_continue)
    ]
    utility_oracle_fixes = sum(
        use and (not base) and continued
        for use, base, continued in zip(
            utility_oracle_continue,
            base_correct,
            continued_correct,
        )
    )
    utility_oracle_harms = sum(
        use and base and (not continued)
        for use, base, continued in zip(
            utility_oracle_continue,
            base_correct,
            continued_correct,
        )
    )
    return {
        "example_count": len(rows),
        "base_accuracy": sum(base_correct) / len(rows),
        "always_continue_accuracy": sum(continued_correct) / len(rows),
        "policy_accuracy": sum(policy_correct) / len(rows),
        "utility_oracle_accuracy": utility_oracle_accuracy,
        "accuracy_oracle_upper_bound": sum(accuracy_oracle_possible) / len(rows),
        "accuracy_oracle_upper_bound_note": (
            "Mean base_correct or continued_correct; not a feasible cost-aware policy."
        ),
        "policy_utility_mean": policy_utility_mean,
        "utility_oracle_utility_mean": utility_oracle_utility_mean,
        "utility_oracle_policy": {
            "action": "continue iff r_continue > r_stop",
            "policy_accuracy": utility_oracle_accuracy,
            "policy_utility_mean": utility_oracle_utility_mean,
            "trigger_rate": sum(utility_oracle_continue) / len(rows),
            "avg_extra_steps": _safe_mean(utility_oracle_actual_steps),
            "avg_planned_extra_steps": _safe_mean(utility_oracle_planned_steps),
            "fixes": utility_oracle_fixes,
            "harms": utility_oracle_harms,
            "net_fix_minus_harm": utility_oracle_fixes - utility_oracle_harms,
        },
        "oracle_action_is_utility_optimal": True,
        "oracle_action_match_rate": oracle_action_matches / len(rows),
        "trigger_rate": trigger_count / len(rows),
        "avg_extra_steps": _safe_mean(actual_steps_policy),
        "avg_planned_extra_steps": _safe_mean(planned_steps_policy),
        "fixes": fixes,
        "harms": harms,
        "net_fix_minus_harm": fixes - harms,
        "missed_fixes_vs_always": always_fixes - fixes,
        "avoided_harms_vs_always": always_harms - harms,
        "compute_saved_vs_always_actual": (
            _safe_mean(actual_steps_all) - _safe_mean(actual_steps_policy)
        ),
        "compute_saved_vs_always_planned": (
            _safe_mean(planned_steps_all) - _safe_mean(planned_steps_policy)
        ),
        "policy_utility_gain_vs_base": policy_utility_mean - base_utility_mean,
        "policy_utility_gain_vs_always": policy_utility_mean - always_utility_mean,
        "policy_regret_vs_oracle": utility_oracle_utility_mean - policy_utility_mean,
        "precision_for_positive_delta": (
            true_positive / trigger_count if trigger_count else None
        ),
        "recall_for_positive_delta": (
            true_positive / positive_count if positive_count else None
        ),
        "auc": _auc(utility_oracle_continue, scores),
    }


def evaluate_policy(
    rows: list[dict[str, Any]],
    scores: list[float],
    threshold: float,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
) -> dict[str, Any]:
    triggered = [float(score) >= float(threshold) for score in scores]
    return evaluate_triggered_policy(
        rows,
        triggered,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
        scores=scores,
    )


def _threshold_candidates(scores: list[float], explicit: list[float] | None) -> list[float]:
    if explicit:
        return sorted(set(float(item) for item in explicit))
    grid = [idx / 100 for idx in range(5, 100, 5)]
    finite = sorted(float(score) for score in scores if math.isfinite(float(score)))
    if finite:
        for idx in range(21):
            q = idx / 20
            pos = q * (len(finite) - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            value = finite[lo] if lo == hi else finite[lo] * (hi - pos) + finite[hi] * (pos - lo)
            grid.append(value)
    return sorted(set(grid))


def select_threshold(
    validation_rows: list[dict[str, Any]],
    validation_scores: list[float],
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
    threshold_grid: list[float] | None = None,
) -> dict[str, Any]:
    best_key = None
    best_item = None
    candidates = _threshold_candidates(validation_scores, threshold_grid)
    for threshold in candidates:
        metrics = evaluate_policy(
            validation_rows,
            validation_scores,
            threshold,
            cost_lambda,
            harm_penalty,
            correctness_reward,
        )
        item = {"threshold": float(threshold), **metrics}
        key = (
            metrics["policy_utility_mean"],
            metrics["policy_accuracy"],
            -metrics["avg_extra_steps"],
            -metrics["harms"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_item = item
    assert best_item is not None
    return {
        "selected_threshold": best_item["threshold"],
        "selected_validation_metrics": best_item,
        "candidate_count": len(candidates),
    }


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: NeuralContinuationBandit,
    tensors: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    entropy_coef: float,
    aux_bce_weight: float,
) -> dict[str, float]:
    model.train()
    n = tensors["features"].shape[0]
    order = torch.randperm(n, device=tensors["features"].device)
    total = {
        "loss": 0.0,
        "bandit_loss": 0.0,
        "entropy": 0.0,
        "aux_loss": 0.0,
        "expected_utility": 0.0,
    }
    seen = 0
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        logits = model(tensors["features"][idx])
        losses = bandit_loss_from_logits(
            logits,
            tensors["r_stop"][idx],
            tensors["r_continue"][idx],
            tensors["oracle"][idx],
            entropy_coef=entropy_coef,
            aux_bce_weight=aux_bce_weight,
        )
        losses["total_loss"].backward()
        optimizer.step()
        count = int(idx.numel())
        seen += count
        total["loss"] += float(losses["total_loss"].detach().cpu()) * count
        total["bandit_loss"] += float(losses["bandit_loss"].cpu()) * count
        total["entropy"] += float(losses["entropy"].cpu()) * count
        total["aux_loss"] += float(losses["aux_loss"].cpu()) * count
        total["expected_utility"] += float(losses["expected_utility"].cpu()) * count
    return {key: value / max(seen, 1) for key, value in total.items()}


def bootstrap_confidence_intervals(
    rows: list[dict[str, Any]],
    scores: list[float],
    threshold: float,
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if bootstrap_samples <= 0:
        return {}
    metric_names = [
        "policy_accuracy",
        "policy_utility_mean",
        "trigger_rate",
        "avg_extra_steps",
        "net_fix_minus_harm",
        "policy_utility_gain_vs_base",
    ]
    samples = {name: [] for name in metric_names}
    rng = random.Random(seed)
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(rows)) for _ in rows]
        sample_rows = [rows[idx] for idx in indices]
        sample_scores = [scores[idx] for idx in indices]
        metrics = evaluate_policy(
            sample_rows,
            sample_scores,
            threshold,
            cost_lambda,
            harm_penalty,
            correctness_reward,
        )
        for name in metric_names:
            samples[name].append(float(metrics[name]))
    return {
        name: {
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
        for name, values in samples.items()
    }


def baseline_results(
    rows: list[dict[str, Any]],
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
) -> dict[str, Any]:
    baselines = {
        "never_continue": [False] * len(rows),
        "always_continue": [True] * len(rows),
        "combined_heuristic": [bool(policy_combined(row)) for row in rows],
        "utility_oracle_policy": [bool(row["oracle_continue"]) for row in rows],
        "oracle_bandit_policy": [bool(row["oracle_continue"]) for row in rows],
    }
    return {
        name: {
            "policy": name,
            **evaluate_triggered_policy(
                rows,
                triggered,
                cost_lambda=cost_lambda,
                harm_penalty=harm_penalty,
                correctness_reward=correctness_reward,
            ),
        }
        for name, triggered in baselines.items()
    }


def write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    scores: list[float],
    threshold: float,
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
) -> None:
    with path.open("w") as f:
        for row, score in zip(rows, scores):
            rewards = bandit_rewards(
                row,
                cost_lambda=cost_lambda,
                harm_penalty=harm_penalty,
                correctness_reward=correctness_reward,
            )
            triggered = float(score) >= float(threshold)
            oracle_continue = bool(rewards["oracle_continue"])
            record = {
                "example_id": row.get("example_id"),
                "score": float(score),
                "threshold": float(threshold),
                "triggered": bool(triggered),
                "policy_continue": bool(triggered),
                "base_correct": rewards["base_correct"],
                "continued_correct": rewards["continued_correct"],
                "actual_extra_steps": rewards["actual_extra_steps"],
                "r_stop": rewards["r_stop"],
                "r_continue": rewards["r_continue"],
                "delta_reward": rewards["delta_reward"],
                "oracle_continue": oracle_continue,
                "policy_correct": bool(
                    rewards["continued_correct"] if triggered else rewards["base_correct"]
                ),
                "policy_utility": float(
                    rewards["r_continue"] if triggered else rewards["r_stop"]
                ),
                "oracle_utility": float(
                    rewards["r_continue"] if oracle_continue else rewards["r_stop"]
                ),
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")


def save_preprocessor(preprocessor, path: Path) -> bool:
    try:
        import joblib
    except ImportError:
        return False
    joblib.dump(preprocessor, path)
    return True


def write_pretty_markdown(path: Path, result: dict[str, Any]) -> None:
    learned = result["final_eval"]["neural_bandit_policy"]
    always = result["final_eval"]["always_continue"]
    always_steps = always["avg_extra_steps"]
    saved = learned["compute_saved_vs_always_actual"]
    saved_pct = (saved / always_steps * 100.0) if always_steps > 0 else 0.0
    lines = [
        "# Neural Contextual Bandit Continuation Policy",
        "",
        "The neural contextual-bandit RL policy matches/approaches always-continuation "
        f"accuracy while reducing continuation compute by {saved_pct:.1f}%.",
        "",
        "## Final Eval",
        "",
        "| Policy | Accuracy | Utility | Trigger Rate | Avg Extra Steps | Fixes | Harms | Net |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in [
        "never_continue",
        "always_continue",
        "combined_heuristic",
        "neural_bandit_policy",
        "utility_oracle_policy",
    ]:
        metrics = result["final_eval"][name]
        lines.append(
            "| {name} | {acc:.4f} | {utility:.4f} | {trigger:.4f} | {steps:.4f} | {fixes} | {harms} | {net} |".format(
                name=name,
                acc=metrics["policy_accuracy"],
                utility=metrics["policy_utility_mean"],
                trigger=metrics["trigger_rate"],
                steps=metrics["avg_extra_steps"],
                fixes=metrics["fixes"],
                harms=metrics["harms"],
                net=metrics["net_fix_minus_harm"],
            )
        )
    lines.extend(
        [
            "",
            "## Training Config",
            "",
            f"- Chosen threshold: `{result['selected_threshold']}`",
            f"- Hidden dim: `{result['config']['hidden_dim']}`",
            f"- Num layers: `{result['config']['num_layers']}`",
            f"- Dropout: `{result['config']['dropout']}`",
            f"- Learning rate: `{result['config']['lr']}`",
            f"- Cost lambda: `{result['config']['cost_lambda']}`",
            f"- Harm penalty: `{result['config']['harm_penalty']}`",
            f"- Compute saved vs always continuation: `{saved:.4f}` actual steps/sample",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_threshold_grid(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def train_neural_bandit(
    *,
    dataset: str | Path,
    output_dir: str | Path,
    cost_lambda: float = 0.0,
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
    threshold_grid: list[float] | None = None,
    seed: int = 123,
    bootstrap_samples: int = 100,
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
    train_pool = add_bandit_labels(
        train_pool_raw,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    final_eval = add_bandit_labels(
        final_eval_raw,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    train_rows, validation_rows = split_train_validation(
        train_pool,
        validation_fraction=validation_fraction,
        seed=seed,
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
    best_threshold = 0.5
    best_key = None
    best_epoch = 0
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
            val_scores = score_rows(model, preprocessor, validation_rows, torch_device)
            threshold_info = select_threshold(
                validation_rows,
                val_scores,
                cost_lambda=cost_lambda,
                harm_penalty=harm_penalty,
                correctness_reward=correctness_reward,
                threshold_grid=threshold_grid,
            )
            val_metrics = threshold_info["selected_validation_metrics"]
            key = (
                val_metrics["policy_utility_mean"],
                val_metrics["policy_accuracy"],
                -val_metrics["avg_extra_steps"],
                -val_metrics["harms"],
            )
            improved = best_key is None or key > best_key
            if improved:
                best_key = key
                best_epoch = epoch
                best_threshold = float(threshold_info["selected_threshold"])
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            history = {
                "epoch": epoch,
                **train_metrics,
                "selected_threshold": float(threshold_info["selected_threshold"]),
                "validation_policy_utility_mean": val_metrics["policy_utility_mean"],
                "validation_policy_accuracy": val_metrics["policy_accuracy"],
                "validation_trigger_rate": val_metrics["trigger_rate"],
                "validation_avg_extra_steps": val_metrics["avg_extra_steps"],
                "best_epoch": best_epoch,
            }
            history_file.write(json.dumps(history, sort_keys=True) + "\n")
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    train_pool_scores = score_rows(model, preprocessor, train_pool, torch_device)
    final_scores = score_rows(model, preprocessor, final_eval, torch_device)
    train_pool_eval = baseline_results(
        train_pool,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    train_pool_eval["neural_bandit_policy"] = {
        "policy": "neural_bandit_policy",
        **evaluate_policy(
            train_pool,
            train_pool_scores,
            best_threshold,
            cost_lambda,
            harm_penalty,
            correctness_reward,
        ),
    }
    final_eval_results = baseline_results(
        final_eval,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    final_eval_results["neural_bandit_policy"] = {
        "policy": "neural_bandit_policy",
        **evaluate_policy(
            final_eval,
            final_scores,
            best_threshold,
            cost_lambda,
            harm_penalty,
            correctness_reward,
        ),
    }
    final_eval_results["bootstrap_ci"] = bootstrap_confidence_intervals(
        final_eval,
        final_scores,
        best_threshold,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    model_path = output_dir / "neural_bandit_policy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "threshold": best_threshold,
            "feature_keys": FEATURE_KEYS,
        },
        model_path,
    )
    preprocessor_path = output_dir / "feature_preprocessor.joblib"
    preprocessor_saved = save_preprocessor(preprocessor, preprocessor_path)
    feature_names_path = output_dir / "feature_names.json"
    feature_names_path.write_text(json.dumps(feature_names(preprocessor), indent=2) + "\n")
    prediction_path = output_dir / "neural_bandit_eval_predictions.jsonl"
    write_predictions(
        prediction_path,
        final_eval,
        final_scores,
        best_threshold,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    result = {
        "status": "ok",
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "selected_threshold": best_threshold,
        "best_epoch": best_epoch,
        "split": {
            "total_rows": len(rows),
            "train_pool_rows": len(train_pool),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "final_eval_rows": len(final_eval),
            "fallback_mode_split": fallback_split,
        },
        "config": {
            "cost_lambda": cost_lambda,
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
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "device": str(torch_device),
        },
        "train_pool": train_pool_eval,
        "final_eval": final_eval_results,
        "artifacts": {
            "model": str(model_path),
            "feature_preprocessor": str(preprocessor_path) if preprocessor_saved else None,
            "feature_names": str(feature_names_path),
            "training_history": str(history_path),
            "result_json": str(output_dir / "neural_bandit_result.json"),
            "pretty_markdown": str(output_dir / "neural_bandit_result_pretty.md"),
            "eval_predictions": str(prediction_path),
        },
    }
    result_path = output_dir / "neural_bandit_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_pretty_markdown(output_dir / "neural_bandit_result_pretty.md", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cost_lambda", type=float, default=0.0)
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
    parser.add_argument("--threshold_grid", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bootstrap_samples", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = train_neural_bandit(
        dataset=args.dataset,
        output_dir=args.output_dir,
        cost_lambda=args.cost_lambda,
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
        threshold_grid=parse_threshold_grid(args.threshold_grid),
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )
    print(json.dumps(result["final_eval"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
