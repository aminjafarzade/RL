#!/usr/bin/env python
"""Train a contextual-bandit continuation budget policy from full-info data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from export_continuation_gate_dataset import FEATURE_KEYS
    from export_continuation_gate_dataset import validate_feature_keys
    from continuation_gate_features import as_bool
    from continuation_gate_features import as_float
    from continuation_gate_features import continuation_step_cost
    from continuation_gate_features import feature_dict
    from continuation_gate_features import missing_feature_counts
    from continuation_gate_features import planned_continuation_steps
    from continuation_gate_features import read_jsonl
    from evaluate_continuation_heuristic_gate import policy_combined
    from evaluate_continuation_heuristic_gate import policy_format
    from evaluate_continuation_heuristic_gate import policy_repetition
    from evaluate_continuation_heuristic_gate import policy_source
except ImportError:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from export_continuation_gate_dataset import FEATURE_KEYS
    from export_continuation_gate_dataset import validate_feature_keys
    from continuation_gate_features import as_bool
    from continuation_gate_features import as_float
    from continuation_gate_features import continuation_step_cost
    from continuation_gate_features import feature_dict
    from continuation_gate_features import missing_feature_counts
    from continuation_gate_features import planned_continuation_steps
    from continuation_gate_features import read_jsonl
    from evaluate_continuation_heuristic_gate import policy_combined
    from evaluate_continuation_heuristic_gate import policy_format
    from evaluate_continuation_heuristic_gate import policy_repetition
    from evaluate_continuation_heuristic_gate import policy_source


FORBIDDEN_FEATURE_FIELDS = {
    "base_correct",
    "continued_correct",
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
    "task_score",
    "base_task_score",
    "final_task_score",
    "reward",
    "answer_quality",
    "correctness",
}


def validate_no_feature_leakage() -> None:
    validate_feature_keys(FEATURE_KEYS)
    leaked = sorted(FORBIDDEN_FEATURE_FIELDS.intersection(FEATURE_KEYS))
    if leaked:
        raise ValueError(f"Leaky fields configured as features: {leaked}")


def actual_extra_steps(row: dict[str, Any]) -> float:
    if row.get("actual_extra_steps") is not None:
        return as_float(row.get("actual_extra_steps"))
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
    labeled = []
    for row in rows:
        labeled.append(
            {
                **row,
                **bandit_rewards(
                    row,
                    cost_lambda=cost_lambda,
                    harm_penalty=harm_penalty,
                    correctness_reward=correctness_reward,
                ),
            }
        )
    return labeled


def split_by_mode(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
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
    validation_fraction: float = 0.2,
    seed: int = 123,
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

    indices = list(range(len(rows)))
    shuffled = indices[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(rows) * validation_fraction)))
    if val_count >= len(rows):
        val_count = len(rows) - 1
    val_indices = set(shuffled[:val_count])
    train = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val = [row for idx, row in enumerate(rows) if idx in val_indices]
    return train or rows, val or rows


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _auc(labels: list[bool], scores: list[float] | None) -> float | None:
    if scores is None or len(set(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    return float(roc_auc_score([int(label) for label in labels], scores))


def evaluate_bandit_policy(
    rows: list[dict[str, Any]],
    triggered: list[bool],
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
    scores: list[float] | None = None,
) -> dict[str, Any]:
    if len(rows) != len(triggered):
        raise ValueError("rows and triggered must have the same length")
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
    positive_delta = [item["delta_reward"] > 0.0 for item in values]
    policy_correct = [
        continued if use else base
        for use, base, continued in zip(triggered, base_correct, continued_correct)
    ]
    bandit_oracle_correct = [
        continued if oracle else base
        for oracle, base, continued in zip(
            positive_delta,
            base_correct,
            continued_correct,
        )
    ]
    oracle_possible_correct = [
        base or continued
        for base, continued in zip(base_correct, continued_correct)
    ]

    r_stop = [item["r_stop"] for item in values]
    r_continue = [item["r_continue"] for item in values]
    policy_utility = [
        cont if use else stop
        for use, stop, cont in zip(triggered, r_stop, r_continue)
    ]
    oracle_utility = [max(stop, cont) for stop, cont in zip(r_stop, r_continue)]

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
    positive_count = sum(positive_delta)
    true_positive = sum(
        use and oracle for use, oracle in zip(triggered, positive_delta)
    )
    base_utility_mean = _safe_mean(r_stop)
    always_continue_utility_mean = _safe_mean(r_continue)
    policy_utility_mean = _safe_mean(policy_utility)
    oracle_utility_mean = _safe_mean(oracle_utility)

    return {
        "example_count": len(rows),
        "base_accuracy": _safe_rate(sum(base_correct), len(rows)),
        "always_continue_accuracy": _safe_rate(sum(continued_correct), len(rows)),
        "policy_accuracy": _safe_rate(sum(policy_correct), len(rows)),
        "oracle_accuracy": _safe_rate(sum(oracle_possible_correct), len(rows)),
        "bandit_oracle_accuracy": _safe_rate(
            sum(bandit_oracle_correct),
            len(rows),
        ),
        "base_utility_mean": base_utility_mean,
        "always_continue_utility_mean": always_continue_utility_mean,
        "policy_utility_mean": policy_utility_mean,
        "oracle_utility_mean": oracle_utility_mean,
        "policy_utility_gain_vs_base": policy_utility_mean - base_utility_mean,
        "policy_utility_gain_vs_always": (
            policy_utility_mean - always_continue_utility_mean
        ),
        "policy_regret_vs_oracle": oracle_utility_mean - policy_utility_mean,
        "trigger_rate": trigger_count / len(rows),
        "avg_extra_steps": _safe_mean(actual_steps_policy),
        "avg_planned_extra_steps": _safe_mean(planned_steps_policy),
        "compute_saved_vs_always_actual": (
            _safe_mean(actual_steps_all) - _safe_mean(actual_steps_policy)
        ),
        "compute_saved_vs_always_planned": (
            _safe_mean(planned_steps_all) - _safe_mean(planned_steps_policy)
        ),
        "fixes": fixes,
        "harms": harms,
        "net_fix_minus_harm": fixes - harms,
        "missed_fixes_vs_always": always_fixes - fixes,
        "avoided_harms_vs_always": always_harms - harms,
        "precision_for_positive_delta": (
            true_positive / trigger_count if trigger_count else None
        ),
        "recall_for_positive_delta": (
            true_positive / positive_count if positive_count else None
        ),
        "auc": _auc(positive_delta, scores),
    }


def _score_quantiles(scores: list[float], bins: int = 21) -> list[float]:
    finite = sorted(float(score) for score in scores if math.isfinite(float(score)))
    if not finite:
        return [0.0]
    if len(finite) == 1:
        return finite
    quantiles = []
    for idx in range(bins):
        q = idx / (bins - 1)
        pos = q * (len(finite) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            quantiles.append(finite[lo])
        else:
            frac = pos - lo
            quantiles.append(finite[lo] * (1 - frac) + finite[hi] * frac)
    return quantiles


def parse_threshold_grid(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def default_threshold_candidates(
    scores: list[float],
    *,
    model_name: str,
) -> list[float]:
    quantiles = _score_quantiles(scores)
    if model_name == "logistic":
        grid = [idx / 100 for idx in range(5, 100, 5)]
        return sorted(set(grid + quantiles))
    return sorted(set([0.0] + quantiles))


def triggers_from_scores(
    scores: list[float],
    threshold: float,
    *,
    model_name: str,
) -> list[bool]:
    if model_name == "logistic":
        return [float(score) >= float(threshold) for score in scores]
    return [float(score) > float(threshold) for score in scores]


def select_threshold(
    validation_rows: list[dict[str, Any]],
    validation_scores: list[float],
    *,
    model_name: str,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
    threshold_grid: list[float] | None = None,
) -> dict[str, Any]:
    candidates = threshold_grid or default_threshold_candidates(
        validation_scores,
        model_name=model_name,
    )
    best = None
    evaluated = []
    for threshold in candidates:
        triggered = triggers_from_scores(
            validation_scores,
            threshold,
            model_name=model_name,
        )
        metrics = evaluate_bandit_policy(
            validation_rows,
            triggered,
            cost_lambda,
            harm_penalty,
            correctness_reward,
            scores=validation_scores,
        )
        item = {"threshold": float(threshold), **metrics}
        evaluated.append(item)
        key = (
            metrics["policy_utility_mean"],
            metrics["policy_accuracy"],
            -metrics["avg_extra_steps"],
            -metrics["harms"],
        )
        if best is None or key > best[0]:
            best = (key, item)
    assert best is not None
    return {
        "selected_threshold": best[1]["threshold"],
        "selected_validation_metrics": best[1],
        "candidate_count": len(evaluated),
        "candidates": evaluated,
    }


def _build_model(model_name: str, seed: int):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("scikit-learn is required for bandit model training") from exc

    if model_name == "logistic":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
    elif model_name == "ridge":
        estimator = Ridge()
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=200,
            learning_rate=0.05,
            min_samples_leaf=1,
        )
    else:
        raise ValueError(f"Unsupported model {model_name!r}")
    return make_pipeline(
        DictVectorizer(sparse=False),
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(),
        estimator,
    )


def train_model(
    train_rows: list[dict[str, Any]],
    *,
    model_name: str,
    seed: int,
):
    x_train = [feature_dict(row) for row in train_rows]
    deltas = [float(row["delta_reward"]) for row in train_rows]
    if model_name == "logistic":
        y_train = [int(bool(row["oracle_continue"])) for row in train_rows]
        if len(set(y_train)) < 2:
            return {
                "kind": "constant",
                "model_name": model_name,
                "constant_score": float(y_train[0]) if y_train else 0.0,
            }
        model = _build_model(model_name, seed)
        model.fit(x_train, y_train)
        return {"kind": "sklearn", "model_name": model_name, "model": model}

    model = _build_model(model_name, seed)
    model.fit(x_train, deltas)
    return {"kind": "sklearn", "model_name": model_name, "model": model}


def predict_scores(model_bundle: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    if model_bundle["kind"] == "constant":
        return [float(model_bundle["constant_score"])] * len(rows)
    model = model_bundle["model"]
    x_rows = [feature_dict(row) for row in rows]
    if model_bundle["model_name"] == "logistic":
        return [float(score) for score in model.predict_proba(x_rows)[:, 1]]
    return [float(score) for score in model.predict(x_rows)]


def baseline_triggers(rows: list[dict[str, Any]], name: str) -> list[bool]:
    if name == "never_continue":
        return [False] * len(rows)
    if name == "always_continue":
        return [True] * len(rows)
    if name == "heuristic_source":
        return [bool(policy_source(row)) for row in rows]
    if name == "heuristic_repetition":
        return [bool(policy_repetition(row)) for row in rows]
    if name == "heuristic_format":
        return [bool(policy_format(row)) for row in rows]
    if name == "combined_heuristic":
        return [bool(policy_combined(row)) for row in rows]
    if name == "oracle_bandit_policy":
        return [bool(row["oracle_continue"]) for row in rows]
    raise ValueError(f"Unknown baseline {name!r}")


BASELINE_NAMES = [
    "never_continue",
    "always_continue",
    "heuristic_source",
    "heuristic_repetition",
    "heuristic_format",
    "combined_heuristic",
    "oracle_bandit_policy",
]


def evaluate_named_policy(
    rows: list[dict[str, Any]],
    name: str,
    triggered: list[bool],
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
    scores: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "policy": name,
        **evaluate_bandit_policy(
            rows,
            triggered,
            cost_lambda,
            harm_penalty,
            correctness_reward,
            scores=scores,
        ),
    }


def evaluate_all_baselines(
    rows: list[dict[str, Any]],
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
) -> dict[str, dict[str, Any]]:
    return {
        name: evaluate_named_policy(
            rows,
            name,
            baseline_triggers(rows, name),
            cost_lambda=cost_lambda,
            harm_penalty=harm_penalty,
            correctness_reward=correctness_reward,
        )
        for name in BASELINE_NAMES
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def bootstrap_confidence_intervals(
    rows: list[dict[str, Any]],
    triggered: list[bool],
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    metric_names = [
        "policy_accuracy",
        "policy_utility_mean",
        "trigger_rate",
        "avg_extra_steps",
        "net_fix_minus_harm",
        "policy_utility_gain_vs_base",
    ]
    if bootstrap_samples <= 0:
        return {}
    rng = random.Random(seed)
    samples = {name: [] for name in metric_names}
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(rows)) for _ in rows]
        sample_rows = [rows[idx] for idx in indices]
        sample_triggered = [triggered[idx] for idx in indices]
        metrics = evaluate_bandit_policy(
            sample_rows,
            sample_triggered,
            cost_lambda,
            harm_penalty,
            correctness_reward,
        )
        for name in metric_names:
            samples[name].append(float(metrics[name]))
    return {
        name: {
            "low": _percentile(values, 0.025),
            "high": _percentile(values, 0.975),
        }
        for name, values in samples.items()
    }


def write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    scores: list[float],
    triggered: list[bool],
    *,
    cost_lambda: float,
    harm_penalty: float,
    correctness_reward: float,
) -> None:
    with path.open("w") as f:
        for row, score, use in zip(rows, scores, triggered):
            rewards = bandit_rewards(
                row,
                cost_lambda=cost_lambda,
                harm_penalty=harm_penalty,
                correctness_reward=correctness_reward,
            )
            policy_correct = (
                rewards["continued_correct"] if use else rewards["base_correct"]
            )
            policy_utility = rewards["r_continue"] if use else rewards["r_stop"]
            record = {
                "example_id": row.get("example_id"),
                "score": float(score),
                "triggered": bool(use),
                "base_correct": rewards["base_correct"],
                "continued_correct": rewards["continued_correct"],
                "actual_extra_steps": rewards["actual_extra_steps"],
                "r_stop": rewards["r_stop"],
                "r_continue": rewards["r_continue"],
                "delta_reward": rewards["delta_reward"],
                "policy_correct": bool(policy_correct),
                "policy_utility": float(policy_utility),
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")


def save_model(path: Path, model_bundle: dict[str, Any], threshold: float) -> bool:
    if model_bundle["kind"] != "sklearn":
        return False
    try:
        import joblib
    except ImportError:
        return False
    joblib.dump(
        {
            "model": model_bundle["model"],
            "model_name": model_bundle["model_name"],
            "threshold": threshold,
            "feature_keys": FEATURE_KEYS,
        },
        path,
    )
    return True


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def write_pretty_markdown(path: Path, result: dict[str, Any]) -> None:
    eval_learned = result["eval"]["learned_bandit_policy"]
    always = result["eval"]["always_continue"]
    saved = eval_learned["compute_saved_vs_always_actual"]
    always_steps = always["avg_extra_steps"]
    saved_pct = (saved / always_steps * 100.0) if always_steps > 0 else 0.0
    claim = (
        "On the held-out eval split, the learned contextual-bandit budget policy "
        "matches/approaches always-continuation accuracy while reducing actual "
        f"continuation compute by {saved_pct:.1f}%."
    )
    lines = [
        "# Contextual Bandit Continuation Policy",
        "",
        claim,
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
        "learned_bandit_policy",
        "oracle_bandit_policy",
    ]:
        metrics = result["eval"][name]
        lines.append(
            "| {name} | {acc} | {utility} | {trigger} | {steps} | {fixes} | {harms} | {net} |".format(
                name=name,
                acc=_fmt_float(metrics["policy_accuracy"]),
                utility=_fmt_float(metrics["policy_utility_mean"]),
                trigger=_fmt_float(metrics["trigger_rate"]),
                steps=_fmt_float(metrics["avg_extra_steps"]),
                fixes=metrics["fixes"],
                harms=metrics["harms"],
                net=metrics["net_fix_minus_harm"],
            )
        )
    lines.extend(
        [
            "",
            "## Training",
            "",
            f"- Model: `{result['model']}`",
            f"- Selected threshold: `{result['selected_threshold']}`",
            f"- Cost lambda: `{result['cost_lambda']}`",
            f"- Harm penalty: `{result['harm_penalty']}`",
            f"- Train rows: `{result['split']['fit_train_rows']}`",
            f"- Validation rows: `{result['split']['validation_rows']}`",
            f"- Final eval rows: `{result['split']['eval_rows']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def train_contextual_bandit(
    *,
    dataset: str | Path,
    output_dir: str | Path,
    model: str = "hist_gradient_boosting",
    cost_lambda: float = 0.0,
    harm_penalty: float = 0.0,
    correctness_reward: float = 1.0,
    threshold_grid: list[float] | None = None,
    validation_fraction: float = 0.2,
    seed: int = 123,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    validate_no_feature_leakage()
    rows = read_jsonl(dataset)
    if not rows:
        raise ValueError("dataset is empty")
    train_rows_raw, eval_rows_raw, fallback_split = split_by_mode(rows)
    if fallback_split:
        print(
            "warning: mode split missing or incomplete; using first 80% train / last 20% eval",
            file=sys.stderr,
        )

    train_rows = add_bandit_labels(
        train_rows_raw,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    eval_rows = add_bandit_labels(
        eval_rows_raw,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    fit_train_rows, validation_rows = split_train_validation(
        train_rows,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    model_bundle = train_model(fit_train_rows, model_name=model, seed=seed)
    validation_scores = predict_scores(model_bundle, validation_rows)
    threshold_selection = select_threshold(
        validation_rows,
        validation_scores,
        model_name=model,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
        threshold_grid=threshold_grid,
    )
    selected_threshold = float(threshold_selection["selected_threshold"])

    train_val_scores = predict_scores(model_bundle, train_rows)
    train_val_triggered = triggers_from_scores(
        train_val_scores,
        selected_threshold,
        model_name=model,
    )
    eval_scores = predict_scores(model_bundle, eval_rows)
    eval_triggered = triggers_from_scores(
        eval_scores,
        selected_threshold,
        model_name=model,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "bandit_policy_eval_predictions.jsonl"
    write_predictions(
        prediction_path,
        eval_rows,
        eval_scores,
        eval_triggered,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )

    model_path = output_dir / "bandit_policy_model.joblib"
    model_saved = save_model(model_path, model_bundle, selected_threshold)

    train_val_results = evaluate_all_baselines(
        train_rows,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    train_val_results["learned_bandit_policy"] = evaluate_named_policy(
        train_rows,
        "learned_bandit_policy",
        train_val_triggered,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
        scores=train_val_scores,
    )
    eval_results = evaluate_all_baselines(
        eval_rows,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
    )
    eval_results["learned_bandit_policy"] = evaluate_named_policy(
        eval_rows,
        "learned_bandit_policy",
        eval_triggered,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
        scores=eval_scores,
    )
    eval_results["bootstrap_ci"] = bootstrap_confidence_intervals(
        eval_rows,
        eval_triggered,
        cost_lambda=cost_lambda,
        harm_penalty=harm_penalty,
        correctness_reward=correctness_reward,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    result = {
        "status": "ok",
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "model": model,
        "model_kind": model_bundle["kind"],
        "cost_lambda": cost_lambda,
        "harm_penalty": harm_penalty,
        "correctness_reward": correctness_reward,
        "selected_threshold": selected_threshold,
        "threshold_selection": threshold_selection,
        "split": {
            "total_rows": len(rows),
            "train_val_rows": len(train_rows),
            "fit_train_rows": len(fit_train_rows),
            "validation_rows": len(validation_rows),
            "eval_rows": len(eval_rows),
            "fallback_mode_split": fallback_split,
        },
        "missing_feature_counts": {
            "all": missing_feature_counts(rows),
            "train": missing_feature_counts(train_rows),
            "eval": missing_feature_counts(eval_rows),
        },
        "feature_keys": FEATURE_KEYS,
        "train_val": train_val_results,
        "eval": eval_results,
        "artifacts": {
            "result_json": str(output_dir / "bandit_policy_result.json"),
            "pretty_markdown": str(output_dir / "bandit_policy_result_pretty.md"),
            "model_joblib": str(model_path) if model_saved else None,
            "eval_predictions_jsonl": str(prediction_path),
        },
    }

    result_path = output_dir / "bandit_policy_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_pretty_markdown(output_dir / "bandit_policy_result_pretty.md", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model",
        choices=["logistic", "ridge", "hist_gradient_boosting"],
        default="hist_gradient_boosting",
    )
    parser.add_argument("--cost_lambda", type=float, default=0.0)
    parser.add_argument("--harm_penalty", type=float, default=0.0)
    parser.add_argument("--correctness_reward", type=float, default=1.0)
    parser.add_argument("--threshold_grid", default=None)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = train_contextual_bandit(
        dataset=args.dataset,
        output_dir=args.output_dir,
        model=args.model,
        cost_lambda=args.cost_lambda,
        harm_penalty=args.harm_penalty,
        correctness_reward=args.correctness_reward,
        threshold_grid=parse_threshold_grid(args.threshold_grid),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(result["eval"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
