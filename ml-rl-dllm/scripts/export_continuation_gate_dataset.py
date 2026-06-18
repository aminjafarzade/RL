#!/usr/bin/env python
"""Export flattened continuation records into an offline gate dataset."""

import argparse
import json
from pathlib import Path
from typing import Any


FEATURE_KEYS = [
    "target_budget",
    "base_nfe",
    "remaining_budget",
    "base_parse_ok",
    "base_format_ok",
    "base_arithmetic_ok",
    "base_verifier_score",
    "base_reward_answer_source",
    "base_repetition_penalty",
    "base_visible_generated_char_count",
    "base_answer_span_confidence",
    "base_entropy_stats",
    "base_confidence_stats",
    "continuation_k",
    "continuation_remasked_token_count",
    "continuation_over_budget",
]

LABEL_KEYS = [
    "continuation_gain",
    "label_gain_positive",
    "label_correct_gain",
    "label_harm",
    "base_correct",
    "continued_correct",
]

METRIC_KEYS = [
    "continued_extra_steps",
    "continuation_extra_steps",
]

FORBIDDEN_FEATURE_KEYS = {
    "answer",
    "base_correct",
    "base_generated_text",
    "base_task_score",
    "continued_correct",
    "continued_generated_text",
    "continuation_gain",
    "final_task_score",
    "generated_text",
    "ground_truth",
    "prompt",
    "reference_answer",
    "reward",
    "reward_after_quality_caps",
    "reward_before_quality_caps",
}

LEAKY_FEATURE_SUBSTRINGS = (
    "answer_quality",
    "task_score",
    "correct",
    "reward",
    "gain",
    "reference",
    "ground_truth",
)

ALLOWED_SUBSTRING_FEATURE_KEYS = {"base_reward_answer_source"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_from_per_example(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        base = {k: v for k, v in row.items() if k != "continuation_results"}
        for item in row.get("continuation_results") or []:
            if isinstance(item, dict):
                flattened.append({**base, **item})
    return flattened


def make_gate_row(record: dict[str, Any]) -> dict[str, Any]:
    validate_feature_keys(FEATURE_KEYS)
    features = {key: record.get(key) for key in FEATURE_KEYS}
    forbidden_present = FORBIDDEN_FEATURE_KEYS.intersection(features)
    if forbidden_present:
        raise ValueError(
            f"Forbidden feature keys present: {sorted(forbidden_present)}"
        )
    gain = float(record.get("continuation_gain") or 0.0)
    labels = {
        "continuation_gain": gain,
        "label_gain_positive": gain > 0.0,
        "label_correct_gain": bool(record.get("continuation_correct_gain")),
        "label_harm": bool(record.get("continuation_harm")),
        "base_correct": bool(record.get("base_correct")),
        "continued_correct": bool(record.get("continued_correct")),
    }
    metrics = {key: record.get(key) for key in METRIC_KEYS}
    return {
        "example_id": record.get("example_id"),
        "mode": record.get("mode"),
        **features,
        **labels,
        **metrics,
    }


def validate_feature_keys(feature_keys: list[str]) -> None:
    label_keys = set(LABEL_KEYS)
    for key in feature_keys:
        if key in label_keys:
            raise ValueError(f"Label key cannot be exported as a feature: {key}")
        if key in FORBIDDEN_FEATURE_KEYS:
            raise ValueError(f"Forbidden key cannot be exported as a feature: {key}")
        if key in ALLOWED_SUBSTRING_FEATURE_KEYS:
            continue
        for substring in LEAKY_FEATURE_SUBSTRINGS:
            if substring in key:
                raise ValueError(
                    f"Potentially leaked feature key {key!r} contains {substring!r}"
                )


def export_gate_dataset(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    records = read_jsonl(output_dir / "continuation_records.jsonl")
    if not records:
        records = flatten_from_per_example(read_jsonl(output_dir / "per_example.jsonl"))
    if not records:
        raise FileNotFoundError(
            f"No continuation records found under {output_dir}"
        )
    out_path = output_dir / "continuation_gate_dataset.jsonl"
    with out_path.open("w") as f:
        for record in records:
            row = make_gate_row(record)
            for forbidden in FORBIDDEN_FEATURE_KEYS:
                if forbidden in row and forbidden not in LABEL_KEYS:
                    raise ValueError(f"Forbidden key leaked into gate row: {forbidden}")
            f.write(json.dumps(row) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out_path = export_gate_dataset(args.output_dir)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
