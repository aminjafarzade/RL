"""Shared feature utilities for offline continuation gate policies."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", "", "none", "null", "nan"}:
            return False
    return bool(value)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


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


def flatten_feature_value(prefix: str, value: Any, out: dict[str, Any]) -> None:
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


def continuation_step_cost(row: dict[str, Any]) -> float:
    for key in ("continued_extra_steps", "continuation_extra_steps"):
        if row.get(key) is not None:
            return as_float(row.get(key))
    return as_float(row.get("continuation_k"))


def planned_continuation_steps(row: dict[str, Any]) -> float:
    return as_float(row.get("continuation_k"))
