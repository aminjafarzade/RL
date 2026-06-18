#!/usr/bin/env python
"""Train a small offline continuation gate from exported gate data."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from export_continuation_gate_dataset import FEATURE_KEYS
    from export_continuation_gate_dataset import validate_feature_keys
except ImportError:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from export_continuation_gate_dataset import FEATURE_KEYS
    from export_continuation_gate_dataset import validate_feature_keys


TERNARY_CATEGORICAL_FEATURES = {"base_arithmetic_ok"}
NUMERIC_FEATURES = {
    "target_budget",
    "base_nfe",
    "remaining_budget",
    "base_verifier_score",
    "base_repetition_penalty",
    "base_visible_generated_char_count",
    "base_answer_span_confidence",
    "continuation_k",
    "continuation_remasked_token_count",
}
NUMERIC_FEATURE_PREFIXES = (
    "base_entropy_stats.",
    "base_confidence_stats.",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _is_missing(value: Any) -> bool:
    return value is None or _is_nan(value)


def _is_numeric_feature(prefix: str) -> bool:
    return prefix in NUMERIC_FEATURES or prefix.startswith(NUMERIC_FEATURE_PREFIXES)


def _ternary_bool_value(value: Any) -> str:
    if _is_missing(value):
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return "true"
        if normalized in {"false", "0", "no", "n"}:
            return "false"
        if normalized in {"", "none", "null", "nan", "unknown"}:
            return "unknown"
    return "true" if bool(value) else "false"


def _add_numeric_feature(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if _is_missing(value):
        out[prefix] = 0.0
        out[f"{prefix}__missing"] = 1.0
        return
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        out[prefix] = 0.0
        out[f"{prefix}__missing"] = 1.0
        return
    if math.isnan(numeric_value):
        out[prefix] = 0.0
        out[f"{prefix}__missing"] = 1.0
        return
    out[prefix] = numeric_value
    out[f"{prefix}__missing"] = 0.0


def flatten_feature_value(prefix: str, value, out: dict[str, Any]) -> None:
    if prefix in TERNARY_CATEGORICAL_FEATURES:
        out[prefix] = _ternary_bool_value(value)
        return
    if isinstance(value, dict):
        if not value:
            out[prefix] = "__missing__"
        else:
            for key, child in value.items():
                flatten_feature_value(f"{prefix}.{key}", child, out)
    elif _is_numeric_feature(prefix):
        _add_numeric_feature(prefix, value, out)
    elif isinstance(value, bool):
        out[prefix] = int(value)
    elif _is_missing(value):
        out[prefix] = "__missing__"
    elif isinstance(value, (int, float, str)):
        out[prefix] = "__missing__" if value == "" else value
    else:
        out[prefix] = str(value)


def validate_feature_dict_values(features: dict[str, Any]) -> None:
    invalid = []
    for key, value in features.items():
        if value is None:
            invalid.append(key)
        elif isinstance(value, float) and math.isnan(value):
            invalid.append(key)
    if invalid:
        raise ValueError(
            f"Feature vector contains missing/NaN values for keys: {sorted(invalid)}"
        )


def feature_dict(row: dict[str, Any]) -> dict[str, Any]:
    validate_feature_keys(FEATURE_KEYS)
    features: dict[str, Any] = {}
    for key in FEATURE_KEYS:
        flatten_feature_value(key, row.get(key), features)
    validate_feature_dict_values(features)
    return features


def _collect_missing(prefix: str, value: Any, counts: dict[str, int]) -> None:
    if isinstance(value, dict):
        if not value:
            counts[prefix] += 1
        for key, child in value.items():
            _collect_missing(f"{prefix}.{key}", child, counts)
    elif _is_missing(value):
        counts[prefix] += 1


def missing_feature_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for key in FEATURE_KEYS:
            _collect_missing(key, row.get(key), counts)
    return dict(sorted(counts.items()))


def label_positive_rate(rows: list[dict[str, Any]], target_label: str) -> float | None:
    if not rows:
        return None
    return sum(bool(row.get(target_label)) for row in rows) / len(rows)


def _as_bool(value) -> bool:
    return bool(value)


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def continuation_step_cost(row: dict) -> float:
    for key in ("continued_extra_steps", "continuation_extra_steps"):
        if row.get(key) is not None:
            return _as_float(row.get(key))
    return _as_float(row.get("continuation_k"))


def planned_continuation_steps(row: dict) -> float:
    return _as_float(row.get("continuation_k"))


def evaluate_gate(rows: list[dict], triggered: list[bool], name: str) -> dict:
    base_correct = [_as_bool(row.get("base_correct")) for row in rows]
    continued_correct = [_as_bool(row.get("continued_correct")) for row in rows]
    gain_positive = [_as_float(row.get("continuation_gain")) > 0.0 for row in rows]
    gated_correct = [
        continued if use else base
        for use, base, continued in zip(triggered, base_correct, continued_correct)
    ]
    oracle_correct = [
        base or continued
        for base, continued in zip(base_correct, continued_correct)
    ]
    fixes = sum(
        use and (not base) and continued
        for use, base, continued in zip(triggered, base_correct, continued_correct)
    )
    harms = sum(
        use and base and (not continued)
        for use, base, continued in zip(triggered, base_correct, continued_correct)
    )
    extra_steps = [
        continuation_step_cost(row) if use else 0.0
        for row, use in zip(rows, triggered)
    ]
    planned_extra_steps = [
        planned_continuation_steps(row) if use else 0.0
        for row, use in zip(rows, triggered)
    ]
    trigger_count = sum(triggered)
    gain_count = sum(gain_positive)
    true_gain_triggers = sum(
        use and gain for use, gain in zip(triggered, gain_positive)
    )
    base_accuracy = sum(base_correct) / len(rows)
    gated_accuracy = sum(gated_correct) / len(rows)
    always_continue_accuracy = sum(continued_correct) / len(rows)
    avg_extra_steps = mean(extra_steps)
    return {
        "policy": name,
        "trigger_rate": trigger_count / len(rows),
        "base_accuracy": base_accuracy,
        "gated_accuracy": gated_accuracy,
        "always_continue_accuracy": always_continue_accuracy,
        "oracle_continue_accuracy": sum(oracle_correct) / len(rows),
        "fixes": fixes,
        "harms": harms,
        "net_fix_minus_harm": fixes - harms,
        "avg_extra_steps": avg_extra_steps,
        "avg_planned_extra_steps": mean(planned_extra_steps),
        "accuracy_per_extra_step": (
            (gated_accuracy - base_accuracy) / avg_extra_steps
            if avg_extra_steps > 0.0
            else None
        ),
        "precision_for_gain": (
            true_gain_triggers / trigger_count if trigger_count else None
        ),
        "recall_for_gain": true_gain_triggers / gain_count if gain_count else None,
    }


def train_and_evaluate(
    rows: list[dict[str, Any]],
    *,
    target_label: str,
    threshold: float,
) -> dict:
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        return {
            "status": "sklearn_unavailable",
            "todo": (
                "Install scikit-learn or use the exported feature dataset with "
                "another offline classifier."
            ),
            "error": str(exc),
        }

    train_rows = [row for row in rows if row.get("mode") == "train"]
    eval_rows = [row for row in rows if row.get("mode") == "eval"]
    if not train_rows or not eval_rows:
        split = max(1, int(0.8 * len(rows)))
        train_rows = rows[:split]
        eval_rows = rows[split:] or rows[split - 1 :]

    diagnostics = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "label_positive_rate_train": label_positive_rate(train_rows, target_label),
        "label_positive_rate_eval": label_positive_rate(eval_rows, target_label),
        "missing_feature_counts": missing_feature_counts(rows),
        "missing_feature_counts_train": missing_feature_counts(train_rows),
        "missing_feature_counts_eval": missing_feature_counts(eval_rows),
    }

    y_train = [int(bool(row.get(target_label))) for row in train_rows]
    if len(set(y_train)) < 2:
        return {
            "status": "single_class_train_labels",
            "target_label": target_label,
            "positive_train_labels": sum(y_train),
            **diagnostics,
        }

    model = make_pipeline(
        DictVectorizer(sparse=False),
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit([feature_dict(row) for row in train_rows], y_train)
    probabilities = model.predict_proba([feature_dict(row) for row in eval_rows])[:, 1]
    triggered = [float(prob) >= float(threshold) for prob in probabilities]
    result = evaluate_gate(eval_rows, triggered, "learned_logistic_gate")
    result.update(
        {
            "status": "ok",
            "target_label": target_label,
            "threshold": threshold,
            **diagnostics,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--target_label",
        choices=["label_gain_positive", "label_correct_gain"],
        default="label_gain_positive",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    result = train_and_evaluate(
        rows,
        target_label=args.target_label,
        threshold=args.threshold,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
