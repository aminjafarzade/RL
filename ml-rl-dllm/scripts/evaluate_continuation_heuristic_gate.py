#!/usr/bin/env python
"""Evaluate simple continuation gate heuristics on exported gate data."""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Callable


ANSWER_REGION_SOURCES = {"incomplete_answer_tag", "final_window", "final_line"}


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def policy_never(_row: dict) -> bool:
    return False


def policy_always(_row: dict) -> bool:
    return True


def policy_format(row: dict) -> bool:
    return row.get("base_format_ok") is False


def policy_source(row: dict) -> bool:
    return row.get("base_reward_answer_source") in ANSWER_REGION_SOURCES


def policy_repetition(row: dict) -> bool:
    return _as_float(row.get("base_repetition_penalty"), 1.0) < 1.0


def policy_combined(row: dict) -> bool:
    has_budget = _as_float(row.get("remaining_budget")) >= _as_float(
        row.get("continuation_k")
    )
    trigger = (
        policy_format(row)
        or policy_source(row)
        or policy_repetition(row)
    )
    return bool(trigger and has_budget)


POLICIES: dict[str, Callable[[dict], bool]] = {
    "never_continue": policy_never,
    "always_continue": policy_always,
    "heuristic_format": policy_format,
    "heuristic_source": policy_source,
    "heuristic_repetition": policy_repetition,
    "combined_heuristic": policy_combined,
}


def evaluate_policy(rows: list[dict], name: str, fn: Callable[[dict], bool]) -> dict:
    if not rows:
        raise ValueError("dataset is empty")
    triggered = [bool(fn(row)) for row in rows]
    base_correct = [_as_bool(row.get("base_correct")) for row in rows]
    continued_correct = [_as_bool(row.get("continued_correct")) for row in rows]
    gains = [_as_float(row.get("continuation_gain")) for row in rows]
    gain_positive = [gain > 0.0 for gain in gains]
    gated_correct = [
        continued if use_continuation else base
        for use_continuation, base, continued in zip(
            triggered,
            base_correct,
            continued_correct,
        )
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
    avg_extra_steps = mean(extra_steps)
    base_accuracy = sum(base_correct) / len(rows)
    gated_accuracy = sum(gated_correct) / len(rows)
    return {
        "policy": name,
        "trigger_rate": trigger_count / len(rows),
        "base_accuracy": base_accuracy,
        "gated_accuracy": gated_accuracy,
        "oracle_accuracy": sum(oracle_correct) / len(rows),
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
        "gain_rate": gain_count / len(rows),
        "trigger_gain_precision": (
            true_gain_triggers / trigger_count if trigger_count else None
        ),
        "trigger_gain_recall": (
            true_gain_triggers / gain_count if gain_count else None
        ),
    }


def evaluate_policies(rows: list[dict]) -> list[dict]:
    return [
        evaluate_policy(rows, name, fn)
        for name, fn in POLICIES.items()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    for result in evaluate_policies(rows):
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
