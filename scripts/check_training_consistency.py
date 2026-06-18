#!/usr/bin/env python
import argparse
import ast
import json
import math
import os
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_config(path: Path) -> dict:
    try:
        import yaml

        with path.open() as f:
            return yaml.safe_load(f) or {}
    except Exception:
        config = {}
        for raw_line in path.read_text().splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line or line.startswith("-"):
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if not value:
                continue
            try:
                config[key.strip()] = ast.literal_eval(value)
            except Exception:
                lowered = value.lower()
                if lowered == "true":
                    config[key.strip()] = True
                elif lowered == "false":
                    config[key.strip()] = False
                elif lowered in {"none", "null"}:
                    config[key.strip()] = None
                else:
                    config[key.strip()] = value
        return config


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"_decode_error": f"{path}:{line_no}: {exc}"})
    return rows


def latest_trainer_state(output_dir: Path) -> dict:
    states = []
    for path in output_dir.glob("checkpoint-*/trainer_state.json"):
        match = re.search(r"checkpoint-(\d+)$", path.parent.name)
        step = int(match.group(1)) if match else -1
        states.append((step, path))
    if not states:
        return {}
    _, path = max(states)
    with path.open() as f:
        return json.load(f)


def parse_train_log_counts(output_dir: Path) -> dict[str, int]:
    path = output_dir / "train.log"
    counts = {}
    if not path.exists():
        return counts
    text = path.read_text(errors="replace")
    for name in ("Train", "Eval"):
        match = re.search(rf"{name} samples:\s*(\d+)", text)
        if match:
            counts[name.lower()] = int(match.group(1))
    return counts


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_nan(value) -> bool:
    value = as_float(value)
    return value is not None and math.isnan(value)


def finite_values(values: list) -> list[float]:
    result = []
    for value in values:
        value = as_float(value)
        if value is not None and math.isfinite(value):
            result.append(value)
    return result


def finite_summary(values: list) -> dict[str, float] | None:
    finite = finite_values(values)
    if not finite:
        return None
    return {
        "mean": mean(finite),
        "min": min(finite),
        "max": max(finite),
        "std": _population_std(finite),
    }


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return math.sqrt(mean([(value - avg) ** 2 for value in values]))


def observed_values(rows: list[dict], key: str) -> list:
    return [row.get(key) for row in rows if row.get(key) is not None]


def bool_rate(rows: list[dict], key: str) -> float | None:
    values = [row.get(key) for row in rows if key in row and row.get(key) is not None]
    if not values:
        return None
    return sum(value is True for value in values) / len(values)


def truthy_rate(values: list) -> float | None:
    if not values:
        return None
    truthy = 0
    for value in values:
        if isinstance(value, bool):
            truthy += int(value)
        else:
            numeric = as_float(value)
            truthy += int(numeric is not None and numeric > 0.5)
    return truthy / len(values)


def text_for_row(row: dict) -> str:
    return (
        row.get("generated_text_final")
        or row.get("generated_text")
        or row.get("generation")
        or ""
    )


def has_strict_format(row: dict) -> bool:
    if row.get("has_boxed_answer") is True or row.get("has_complete_answer_tag") is True:
        return True
    text = text_for_row(row)
    return "\\boxed" in text or (
        "<answer>" in text.lower() and "</answer>" in text.lower()
    )


def has_empty_answer_tag(row: dict) -> bool:
    text = text_for_row(row)
    return any(
        not match.group(1).strip()
        for match in re.finditer(
            r"<answer>(.*?)</answer>",
            text or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def avg_nfe_by_budget(rows: list[dict]) -> dict[int, float]:
    by_budget = defaultdict(list)
    for row in rows:
        budget = row.get("target_budget")
        nfe = row.get("actual_nfe", row.get("nfe_used"))
        nfe_float = as_float(nfe)
        if budget is not None and nfe_float is not None:
            by_budget[int(budget)].append(nfe_float)
    return {budget: mean(values) for budget, values in sorted(by_budget.items())}


def mean_by_target_budget(rows: list[dict], key: str) -> dict[int, float]:
    by_budget = defaultdict(list)
    for row in rows:
        budget = row.get("target_budget")
        value = as_float(row.get(key))
        if budget is not None and value is not None and math.isfinite(value):
            by_budget[int(budget)].append(value)
    return {budget: mean(values) for budget, values in sorted(by_budget.items())}


def positive_reward_rate_by_target_budget(rows: list[dict]) -> dict[int, float]:
    by_budget = defaultdict(list)
    for row in rows:
        budget = row.get("target_budget")
        reward = as_float(row.get("reward"))
        if budget is not None and reward is not None and math.isfinite(reward):
            by_budget[int(budget)].append(reward > 1e-8)
    return {
        budget: sum(values) / len(values)
        for budget, values in sorted(by_budget.items())
        if values
    }


def value_distribution_by_target_budget(rows: list[dict], key: str) -> dict[int, dict]:
    by_budget = defaultdict(Counter)
    for row in rows:
        budget = row.get("target_budget")
        value = row.get(key)
        if budget is not None and value is not None:
            by_budget[int(budget)][str(value)] += 1
    return {budget: dict(counter) for budget, counter in sorted(by_budget.items())}


def mean_by_category(rows: list[dict], category_key: str, value_key: str) -> dict[str, float]:
    by_category = defaultdict(list)
    for row in rows:
        category = row.get(category_key)
        value = as_float(row.get(value_key))
        if category is not None and value is not None and math.isfinite(value):
            by_category[str(category)].append(value)
    return {
        category: mean(values)
        for category, values in sorted(by_category.items())
        if values
    }


def positive_reward_rate_by_category(
    rows: list[dict],
    category_key: str,
) -> dict[str, float]:
    by_category = defaultdict(list)
    for row in rows:
        category = row.get(category_key)
        reward = as_float(row.get("reward"))
        if category is not None and reward is not None and math.isfinite(reward):
            by_category[str(category)].append(reward > 1e-8)
    return {
        category: sum(values) / len(values)
        for category, values in sorted(by_category.items())
        if values
    }


def value_distribution_by_category(
    rows: list[dict],
    category_key: str,
    value_key: str,
) -> dict[str, dict]:
    by_category = defaultdict(Counter)
    for row in rows:
        category = row.get(category_key)
        value = row.get(value_key)
        if category is not None and value is not None:
            by_category[str(category)][str(value)] += 1
    return {
        category: dict(counter)
        for category, counter in sorted(by_category.items())
    }


def positive_reward_source_distribution(rows: list[dict]) -> dict[str, int]:
    counter = Counter()
    for row in rows:
        reward = as_float(row.get("reward"))
        source = row.get("reward_answer_source")
        if reward is not None and reward > 1e-8 and source is not None:
            counter[str(source)] += 1
    return dict(counter)


def positive_reward_mean_by_source(rows: list[dict]) -> dict[str, float]:
    by_source = defaultdict(list)
    for row in rows:
        reward = as_float(row.get("reward"))
        source = row.get("reward_answer_source")
        if (
            reward is not None
            and math.isfinite(reward)
            and reward > 1e-8
            and source is not None
        ):
            by_source[str(source)].append(reward)
    return {source: mean(values) for source, values in sorted(by_source.items())}


def task_score_for_row(row: dict) -> float:
    for key in ("final_task_score", "task_score_after_quality", "task_score_before_compute"):
        value = as_float(row.get(key))
        if value is not None and math.isfinite(value):
            return value
    reward = as_float(row.get("reward"))
    return reward if reward is not None and math.isfinite(reward) else 0.0


def paired_rollout_counts(rows: list[dict], margin: float = 0.05) -> dict[str, int]:
    counts = {
        "patient_better_than_normal_count": 0,
        "patient_correct_normal_wrong_count": 0,
        "normal_correct_patient_wrong_count": 0,
        "both_good_count": 0,
        "both_bad_count": 0,
    }
    for group_rows in _group_rows(rows).values():
        normal = next(
            (
                row
                for row in group_rows
                if row.get("rollout_compute_style") == "normal"
            ),
            None,
        )
        patient = next(
            (
                row
                for row in group_rows
                if row.get("rollout_compute_style") == "patient"
            ),
            None,
        )
        if normal is None or patient is None:
            continue
        normal_score = task_score_for_row(normal)
        patient_score = task_score_for_row(patient)
        if patient_score > normal_score + margin:
            counts["patient_better_than_normal_count"] += 1
        if patient_score > 0.0 and normal_score <= 0.0:
            counts["patient_correct_normal_wrong_count"] += 1
        if normal_score > 0.0 and patient_score <= 0.0:
            counts["normal_correct_patient_wrong_count"] += 1
        if normal_score > 0.0 and patient_score > 0.0:
            counts["both_good_count"] += 1
        if normal_score <= 0.0 and patient_score <= 0.0:
            counts["both_bad_count"] += 1
    return counts


def continuation_entries(rows: list[dict]) -> list[tuple[dict, dict]]:
    entries = []
    for row in rows:
        for item in row.get("continuation_results") or []:
            if isinstance(item, dict):
                entries.append((row, item))
    return entries


def mean_continuation_gain_by(
    entries: list[tuple[dict, dict]],
    key_fn,
) -> dict:
    grouped = defaultdict(list)
    for row, item in entries:
        key = key_fn(row, item)
        value = as_float(item.get("continuation_gain"))
        if key is not None and value is not None and math.isfinite(value):
            grouped[key].append(value)
    return {
        key: mean(values)
        for key, values in sorted(grouped.items(), key=lambda pair: str(pair[0]))
        if values
    }


def _row_base_correct(row: dict) -> bool | None:
    if row.get("base_correct") is not None:
        return row.get("base_correct") is True
    score = task_score_for_row(row)
    return score > 0.0


def continuation_accuracy_summary(rows: list[dict]) -> dict[str, object]:
    base_values = []
    selected_values = []
    best_values = []
    oracle_values = []
    selected_fix_count = 0
    selected_harm_count = 0

    for row in rows:
        items = [
            item
            for item in (row.get("continuation_results") or [])
            if isinstance(item, dict)
        ]
        base_correct = _row_base_correct(row)
        if base_correct is None:
            continue
        base_values.append(base_correct)

        selected_correct = None
        if items:
            selected = items[0]
            if selected.get("continued_correct") is not None:
                selected_correct = selected.get("continued_correct") is True
        elif row.get("continued_correct") is not None:
            selected_correct = row.get("continued_correct") is True

        best_correct = None
        if row.get("best_continuation_correct") is not None:
            best_correct = row.get("best_continuation_correct") is True
        elif items:
            best_item = max(
                items,
                key=lambda item: (
                    as_float(item.get("continued_task_score")) or 0.0,
                    -int(item.get("continuation_k") or 0),
                ),
            )
            if best_item.get("continued_correct") is not None:
                best_correct = best_item.get("continued_correct") is True
        elif selected_correct is not None:
            best_correct = selected_correct

        if selected_correct is not None:
            selected_values.append(selected_correct)
            if selected_correct and not base_correct:
                selected_fix_count += 1
            if base_correct and not selected_correct:
                selected_harm_count += 1
        if best_correct is not None:
            best_values.append(best_correct)
            oracle_values.append(base_correct or best_correct)
        else:
            oracle_values.append(base_correct)

    return {
        "base_accuracy": sum(base_values) / len(base_values)
        if base_values
        else None,
        "continued_accuracy": sum(selected_values) / len(selected_values)
        if selected_values
        else None,
        "best_continuation_accuracy": sum(best_values) / len(best_values)
        if best_values
        else None,
        "oracle_continue_accuracy": sum(oracle_values) / len(oracle_values)
        if oracle_values
        else None,
        "continuation_fix_count": selected_fix_count,
        "continuation_harm_count": selected_harm_count,
        "continued_correct_base_wrong_count": selected_fix_count,
        "base_correct_continued_wrong_count": selected_harm_count,
    }


def _trainer_log_history_values(trainer_state: dict, key: str) -> list:
    return [
        item.get(key)
        for item in trainer_state.get("log_history", [])
        if item.get(key) is not None
    ]


def analyze_training_consistency(
    *,
    config: dict,
    rows: list[dict],
    generations: list[dict] | None = None,
    verifier_rows: list[dict] | None = None,
    trainer_state: dict | None = None,
    log_counts: dict[str, int] | None = None,
    min_examples: int = 1,
    output_dir: Path | None = None,
    checkpoint_count: int = 0,
) -> dict:
    generations = generations or []
    verifier_rows = verifier_rows or []
    trainer_state = trainer_state or {}
    log_counts = log_counts or {}
    inconsistencies: list[str] = []
    warnings: list[str] = []

    if output_dir is not None:
        expected_output_dir = config.get("output_dir")
        if expected_output_dir and os.path.normpath(expected_output_dir) != os.path.normpath(
            str(output_dir)
        ):
            inconsistencies.append(
                f"output_dir mismatch: config={expected_output_dir} observed={output_dir}"
            )
        if not output_dir.exists():
            inconsistencies.append(f"output_dir does not exist: {output_dir}")
        if checkpoint_count == 0:
            warnings.append("no checkpoint directories found yet")

    if len(rows) < min_examples:
        inconsistencies.append(
            f"too few per-example rows: found {len(rows)}, expected at least {min_examples}"
        )
    if any("_decode_error" in row for row in rows):
        inconsistencies.extend(
            row["_decode_error"] for row in rows if "_decode_error" in row
        )

    if config.get("dataset") not in {None, "gsm8k"}:
        inconsistencies.append(f"dataset is not gsm8k: {config.get('dataset')}")
    for key, log_key in (("max_train_samples", "train"), ("max_eval_samples", "eval")):
        if key in config and config.get(key) is not None:
            observed = log_counts.get(log_key)
            if observed is None:
                warnings.append(f"{key} set but train.log count was not found")
            elif int(config[key]) != observed:
                inconsistencies.append(
                    f"{key} mismatch: config={config[key]} observed={observed}"
                )

    config_target_budgets = set(config.get("target_budgets") or [])
    config_train_budgets = set(config.get("train_budget_sampling") or [])
    target_budgets = [int(v) for v in observed_values(rows, "target_budget")]
    target_budget_set = set(target_budgets)
    if not target_budgets:
        inconsistencies.append("target_budget missing from all per-example logs")
    elif config_target_budgets and not target_budget_set.issubset(config_target_budgets):
        inconsistencies.append(
            "observed target budgets "
            f"{sorted(target_budget_set)} not subset of config target_budgets "
            f"{sorted(config_target_budgets)}"
        )
    train_rows = [row for row in rows if row.get("mode") == "train"]
    eval_rows = [row for row in rows if row.get("mode") == "eval"]
    train_target_set = set(int(v) for v in observed_values(train_rows, "target_budget"))
    if train_target_set and config_train_budgets and not train_target_set.issubset(
        config_train_budgets
    ):
        inconsistencies.append(
            "train target budgets "
            f"{sorted(train_target_set)} not subset of train_budget_sampling "
            f"{sorted(config_train_budgets)}"
        )

    group_budget_mismatches = []
    for (step, group_id), group_rows in _group_rows(train_rows).items():
        budgets = {row.get("target_budget") for row in group_rows}
        if len(group_rows) >= int(config.get("num_generations") or 1) and len(budgets) > 1:
            group_budget_mismatches.append((step, group_id, sorted(budgets)))
    if group_budget_mismatches:
        inconsistencies.append(
            "target_budget varies within GRPO group: "
            f"{group_budget_mismatches[:5]}"
        )

    for key in (
        "max_completion_length",
        "num_generations",
        "generation_batch_size",
        "per_device_train_batch_size",
    ):
        values = set(observed_values(rows, key))
        if rows and not values:
            inconsistencies.append(f"{key} missing from per-example logs")
        elif values and config.get(key) is not None and values != {config.get(key)}:
            inconsistencies.append(
                f"{key} mismatch: config={config.get(key)} observed={sorted(values)}"
            )

    remask_counts = [int(v) for v in observed_values(rows, "remask_count")]
    if config.get("enable_remasking") is False and any(v != 0 for v in remask_counts):
        inconsistencies.append(
            f"remask_count nonzero while enable_remasking=false: {Counter(remask_counts)}"
        )

    verifier_calls = [int(v) for v in observed_values(rows, "verifier_calls")]
    if config.get("enable_verifier") is True:
        if config.get("verifier_type") not in {None, "math"}:
            inconsistencies.append(f"verifier_type is not math: {config.get('verifier_type')}")
        if not verifier_calls:
            inconsistencies.append("verifier_calls missing from all per-example logs")
        expected_calls = 1
        if config.get("enable_remasking") is True:
            expected_calls = 2
        if config.get("verifier_schedule") == "final_only" and verifier_calls:
            if max(verifier_calls) > expected_calls:
                inconsistencies.append(
                    f"verifier_calls too high for final_only: {Counter(verifier_calls)}"
                )

    nfe_values = [
        as_float(row.get("actual_nfe", row.get("nfe_used")))
        for row in rows
        if row.get("actual_nfe", row.get("nfe_used")) is not None
    ]
    nfe_values = [value for value in nfe_values if value is not None]
    avg_nfe_budget = avg_nfe_by_budget(rows)
    if not nfe_values:
        inconsistencies.append("nfe_used/actual_nfe missing from all per-example logs")
    elif (
        len(avg_nfe_budget) > 1
        and config.get("hard_stop_at_target_budget") is False
    ):
        rounded = {round(value, 6) for value in avg_nfe_budget.values()}
        if len(rounded) == 1:
            inconsistencies.append(
                "all avg_nfe values are identical while "
                f"hard_stop_at_target_budget=false: {avg_nfe_budget}"
            )

    rewards = observed_values(rows, "reward")
    reward_summary = finite_summary(rewards)
    reward_max = reward_summary["max"] if reward_summary else None
    positive_rewards = [value for value in finite_values(rewards) if value > 1e-8]
    if any(is_nan(v) for v in rewards):
        inconsistencies.append("reward contains NaN")
    if len(rows) >= min_examples and (reward_max is None or reward_max <= 0.0):
        inconsistencies.append(
            "No positive rewards observed. Check reward parser, exact-match "
            "extractor, generation format, or config difficulty."
        )

    for item in trainer_state.get("log_history", []):
        if is_nan(item.get("loss")) or is_nan(item.get("train_loss")):
            inconsistencies.append(f"trainer loss contains NaN at step {item.get('step')}")
        if is_nan(item.get("reward")) or is_nan(item.get("eval_reward")):
            inconsistencies.append(f"trainer reward contains NaN at step {item.get('step')}")

    extractor = (
        config.get("reward_gsm_extractor")
        or _first_observed(rows, "reward_gsm_extractor")
        or "robust"
    )
    robust_correct_but_reward_zero = [
        row
        for row in rows
        if row.get("robust_correct") is True
        and (as_float(row.get("reward")) is None or as_float(row.get("reward")) <= 1e-8)
    ]
    if extractor == "robust" and robust_correct_but_reward_zero:
        inconsistencies.append(
            "robust_correct_but_reward_zero: "
            f"{len(robust_correct_but_reward_zero)} rows"
        )

    strict_format_rate = None
    if rows:
        strict_format_rate = sum(has_strict_format(row) for row in rows) / len(rows)
        if extractor == "strict" and strict_format_rate == 0.0 and len(rows) >= min_examples:
            inconsistencies.append(
                "strict format rate is 0 while reward_gsm_extractor=strict"
            )

    positive_wrong = 0
    for row in rows:
        reward = as_float(row.get("reward"))
        if reward is None or reward <= 1e-8:
            continue
        exact_used = row.get("reward_exact_match_used")
        if exact_used is False:
            positive_wrong += 1
    if positive_wrong:
        inconsistencies.append(
            f"{positive_wrong} rows have positive reward but reward_exact_match_used=false"
        )

    parse_rate = bool_rate(rows, "parse_ok")
    format_rate = bool_rate(rows, "format_ok")
    arithmetic_rate = bool_rate(rows, "arithmetic_ok")
    if rows and parse_rate is None:
        inconsistencies.append("parse_ok missing from per-example logs")
    elif parse_rate == 0.0:
        inconsistencies.append("parse_ok is always false")

    generated_texts = [text_for_row(row) for row in rows]
    generated_present = sum(bool(text) for text in generated_texts)
    char_counts = [
        int(row.get("visible_generated_char_count", len(text_for_row(row)) or 0))
        for row in rows
    ]
    short_rows = sum(count < 8 for count in char_counts)
    if rows and generated_present / len(rows) < 0.5:
        inconsistencies.append("generated_text empty for most rows")
    elif rows and short_rows / len(rows) > 0.5:
        inconsistencies.append("generated_text empty/short for most rows")

    masked_fraction_values = finite_values(
        observed_values(rows, "masked_token_fraction")
    )
    masked_token_fraction_mean = (
        mean(masked_fraction_values) if masked_fraction_values else None
    )
    still_masked_sample_rate = None
    if rows:
        still_masked_flags = []
        for row in rows:
            masked_count = as_float(row.get("masked_token_count"))
            masked_fraction = as_float(row.get("masked_token_fraction"))
            if masked_count is not None:
                still_masked_flags.append(masked_count > 0)
            elif masked_fraction is not None:
                still_masked_flags.append(masked_fraction > 0.0)
        if still_masked_flags:
            still_masked_sample_rate = sum(still_masked_flags) / len(still_masked_flags)
    budget_is_hard_cap_values = observed_values(rows, "budget_is_hard_cap")
    stop_reasons = [str(value) for value in observed_values(rows, "stop_reason")]
    boxed_rate = bool_rate(rows, "has_boxed_answer")
    answer_tag_rate = bool_rate(rows, "has_complete_answer_tag")
    robust_correct_rate = bool_rate(rows, "robust_correct")
    answer_span_correct_rate = bool_rate(rows, "answer_span_correct")
    strict_correct_rate = bool_rate(rows, "strict_correct")
    robust_only_correct_rows = [
        row
        for row in rows
        if row.get("robust_correct") is True
        and row.get("answer_span_correct") is not True
    ]
    robust_only_positive_reward_rows = [
        row
        for row in robust_only_correct_rows
        if (as_float(row.get("reward")) or 0.0) > 1e-8
    ]
    if extractor == "answer_span" and robust_only_positive_reward_rows:
        inconsistencies.append(
            "robust_only_positive_reward_count should be 0 when "
            "reward_gsm_extractor=answer_span, got "
            f"{len(robust_only_positive_reward_rows)}"
        )
    format_false_positive_rows = [
        row
        for row in rows
        if row.get("format_ok") is False and (as_float(row.get("reward")) or 0.0) > 1e-8
    ]
    empty_answer_tag_rows = [row for row in rows if has_empty_answer_tag(row)]
    malformed_positive_rows = [
        row
        for row in rows
        if (as_float(row.get("reward")) or 0.0) > 1e-8
        and (
            has_empty_answer_tag(row)
            or row.get("reward_answer_source") in {"robust_anywhere", "none"}
        )
    ]
    avg_nfe_by_style = mean_by_category(
        rows,
        "rollout_compute_style",
        "actual_nfe",
    )
    reward_mean_by_style = mean_by_category(rows, "rollout_compute_style", "reward")
    positive_reward_rate_by_style = positive_reward_rate_by_category(
        rows,
        "rollout_compute_style",
    )
    avg_task_score_by_style = mean_by_category(
        rows,
        "rollout_compute_style",
        "final_task_score",
    )
    masked_fraction_by_style = mean_by_category(
        rows,
        "rollout_compute_style",
        "masked_token_fraction",
    )
    stop_reason_by_style = value_distribution_by_category(
        rows,
        "rollout_compute_style",
        "stop_reason",
    )
    pair_counts = paired_rollout_counts(
        train_rows,
        margin=float(config.get("patient_win_margin", 0.05) or 0.05),
    )
    deliberation_gains = finite_values(observed_values(rows, "deliberation_gain"))
    early_task_scores = finite_values(observed_values(rows, "early_task_score"))
    final_task_scores = finite_values(observed_values(rows, "final_task_score"))
    patient_cap_values = observed_values(rows, "patient_cap_applied")
    malformed_reward_cap_values = observed_values(
        rows,
        "malformed_reward_cap_applied",
    )
    repetitive_reward_cap_values = observed_values(
        rows,
        "repetitive_reward_cap_applied",
    )
    reward_before_caps_summary = finite_summary(
        observed_values(rows, "reward_before_quality_caps")
    )
    reward_after_caps_summary = finite_summary(
        observed_values(rows, "reward_after_quality_caps")
    )
    cont_entries = continuation_entries(rows)
    cont_gains = finite_values(
        [item.get("continuation_gain") for _, item in cont_entries]
    )
    best_cont_gains = finite_values(observed_values(rows, "best_continuation_gain"))
    cont_extra_steps = finite_values(
        [item.get("continued_extra_steps") for _, item in cont_entries]
    )
    cont_over_budget = [
        item.get("continuation_over_budget")
        for _, item in cont_entries
        if item.get("continuation_over_budget") is not None
    ]
    cont_remask_counts = finite_values(
        [item.get("continuation_remasked_token_count") for _, item in cont_entries]
    )
    base_quality_values = finite_values(observed_values(rows, "base_quality_score"))
    continued_quality_values = finite_values(
        [item.get("continued_quality_score") for _, item in cont_entries]
    )
    best_quality_values = finite_values(
        observed_values(rows, "best_continuation_task_score")
    )
    base_format_values = observed_values(rows, "base_format_ok")
    continued_format_values = [
        item.get("continued_format_ok")
        for _, item in cont_entries
        if item.get("continued_format_ok") is not None
    ]
    continuation_acc = continuation_accuracy_summary(rows)
    high_reward_malformed_rows = [
        row
        for row in rows
        if (as_float(row.get("reward")) or 0.0) > 0.5
        and (
            row.get("format_ok") is False
            or not has_strict_format(row)
            or row.get("reward_answer_source")
            not in {"strict_boxed", "strict_answer_tag"}
        )
    ]
    high_reward_repetitive_rows = [
        row
        for row in rows
        if (as_float(row.get("reward")) or 0.0) > 0.5
        and (
            row.get("repeated_char_run_detected") is True
            or (as_float(row.get("repetition_penalty")) or 1.0) < 1.0
        )
    ]
    positive_source_dist = positive_reward_source_distribution(rows)
    quality_mode = (
        config.get("reward_quality_mode")
        or _first_observed(rows, "reward_quality_mode")
        or "none"
    )
    if avg_nfe_by_style.get("patient") is not None and avg_nfe_by_style.get("normal") is not None:
        if avg_nfe_by_style["patient"] <= avg_nfe_by_style["normal"]:
            warnings.append("patient exploration is not slower than normal rollout")
    if "patient" in avg_nfe_by_style and pair_counts["patient_better_than_normal_count"] == 0:
        warnings.append("extra compute did not improve observed task score")
    if len(rows) >= min_examples and not positive_rewards:
        warnings.append("training signal is dead: positive_reward_rate is 0")
    if (
        quality_mode == "format_weighted"
        and positive_rewards
        and positive_source_dist.get("incomplete_answer_tag", 0) / len(positive_rewards)
        > 0.5
    ):
        warnings.append(
            "format_weighted reward still relies on malformed incomplete_answer_tag "
            "answers for more than 50% of positive rewards"
        )
    if high_reward_malformed_rows:
        warnings.append("quality weighting is too weak: high-reward malformed rows found")
    continuation_trigger_rate = bool_rate(rows, "continuation_triggered")
    if config.get("enable_posthoc_continuation") is True:
        if continuation_trigger_rate is None or continuation_trigger_rate == 0.0:
            warnings.append("continuation_trigger_rate is 0")
        if not cont_entries or not any(value > 1e-8 for value in cont_gains):
            warnings.append("continuation_gain_rate is 0")
        if (
            continuation_acc["continuation_harm_count"]
            > continuation_acc["continuation_fix_count"]
        ):
            warnings.append(
                "continuation harms outnumber continued-correct/base-wrong gains"
            )
        if cont_remask_counts and sum(value == 0 for value in cont_remask_counts) / len(cont_remask_counts) > 0.5:
            warnings.append("continuation remasked_token_count is 0 for most candidates")
        if cont_over_budget and sum(value is True for value in cont_over_budget) / len(cont_over_budget) > 0.5:
            warnings.append("continuation candidates are mostly over budget")
    checkpoint_requested = (
        config.get("policy_checkpoint_path")
        or _first_observed(rows, "policy_checkpoint_path_requested")
    )
    checkpoint_loaded_values = observed_values(rows, "policy_checkpoint_loaded")
    checkpoint_loaded = any(value is True for value in checkpoint_loaded_values)
    if checkpoint_requested and not checkpoint_loaded:
        warnings.append(
            "policy_checkpoint_loaded is false even though a checkpoint path was requested"
        )
    if masked_token_fraction_mean is not None and masked_token_fraction_mean > 0.02:
        warnings.append("Draft generation incomplete; do not run continuation.")
    if still_masked_sample_rate is not None and still_masked_sample_rate > 0.1:
        warnings.append("Many samples still masked; draft policy too conservative.")
    hard_generation_budget = as_float(config.get("hard_generation_budget"))
    avg_nfe = mean(nfe_values) if nfe_values else None
    if (
        avg_nfe is not None
        and hard_generation_budget is not None
        and hard_generation_budget > 0
        and masked_token_fraction_mean is not None
        and masked_token_fraction_mean > 0.0
        and avg_nfe >= 0.9 * hard_generation_budget
    ):
        warnings.append("Generation is hitting cap without finishing.")
    zero_std_values = finite_values(
        _trainer_log_history_values(trainer_state, "zero_std_ratio")
        + _trainer_log_history_values(trainer_state, "eval_zero_std_ratio")
    )
    if config.get("hard_stop_at_target_budget") is True:
        block_length = int(config.get("block_length") or 0)
        target_budget_values = list(config.get("target_budgets") or target_budget_set)
        has_low_target = (
            block_length > 0
            and any(int(budget) < block_length * 2 for budget in target_budget_values)
        )
        zero_reward = reward_max is not None and reward_max <= 0.0
        high_mask_fraction = (
            masked_token_fraction_mean is not None and masked_token_fraction_mean > 0.25
        )
        if has_low_target or zero_reward or high_mask_fraction:
            warnings.append(
                "hard_stop_at_target_budget=true may prevent generation from reaching "
                "final answer spans during early policy training. For reward recovery, "
                "use hard_stop_at_target_budget=false and hard_generation_budget >= 128."
            )

    summary = {
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "hard_stop_at_target_budget": config.get("hard_stop_at_target_budget"),
        "hard_generation_budget": config.get("hard_generation_budget"),
        "budget_is_hard_cap_rate": truthy_rate(budget_is_hard_cap_values),
        "policy_smart_init": config.get("policy_smart_init"),
        "policy_checkpoint_path": config.get("policy_checkpoint_path"),
        "policy_checkpoint_path_requested": checkpoint_requested,
        "policy_checkpoint_loaded": checkpoint_loaded if checkpoint_requested else None,
        "policy_checkpoint_path_effective": _first_observed(
            rows,
            "policy_checkpoint_path_effective",
        ),
        "avg_nfe": avg_nfe,
        "still_masked_sample_rate": still_masked_sample_rate,
        "generations_rows": len(generations),
        "verifier_rows": len(verifier_rows),
        "reward_summary": reward_summary,
        "positive_reward_count": len(positive_rewards),
        "positive_reward_rate": len(positive_rewards) / len(rows) if rows else None,
        "positive_reward_rate_by_target_budget": (
            positive_reward_rate_by_target_budget(rows)
        ),
        "avg_nfe_by_rollout_compute_style": avg_nfe_by_style,
        "reward_mean_by_rollout_compute_style": reward_mean_by_style,
        "positive_reward_rate_by_rollout_compute_style": (
            positive_reward_rate_by_style
        ),
        "avg_task_score_by_rollout_compute_style": avg_task_score_by_style,
        "masked_token_fraction_by_rollout_compute_style": masked_fraction_by_style,
        "stop_reason_by_rollout_compute_style": stop_reason_by_style,
        **pair_counts,
        "deliberation_gain_rate": (
            sum(value > 1e-8 for value in deliberation_gains) / len(deliberation_gains)
            if deliberation_gains
            else None
        ),
        "early_task_score_mean": mean(early_task_scores)
        if early_task_scores
        else None,
        "final_task_score_mean": mean(final_task_scores)
        if final_task_scores
        else None,
        "avg_deliberation_gain": mean(deliberation_gains)
        if deliberation_gains
        else None,
        "patient_cap_applied_rate": truthy_rate(patient_cap_values),
        "avg_selected_unmask_count_before_cap": _mean_observed(
            rows,
            "selected_unmask_count_before_cap",
        ),
        "avg_selected_unmask_count_after_cap": _mean_observed(
            rows,
            "selected_unmask_count_after_cap",
        ),
        "avg_delayed_low_confidence_count": _mean_observed(
            rows,
            "delayed_low_confidence_count",
        ),
        "avg_delayed_final_window_count": _mean_observed(
            rows,
            "delayed_final_window_count",
        ),
        "avg_patient_selected_before_delay": _mean_observed(
            rows,
            "patient_selected_before_delay",
        ),
        "avg_patient_selected_after_delay": _mean_observed(
            rows,
            "patient_selected_after_delay",
        ),
        "avg_answer_quality": _mean_observed(rows, "answer_quality"),
        "avg_format_factor": _mean_observed(rows, "format_factor"),
        "avg_repetition_penalty": _mean_observed(rows, "repetition_penalty"),
        "malformed_reward_cap_applied_count": sum(
            value is True for value in malformed_reward_cap_values
        ),
        "repetitive_reward_cap_applied_count": sum(
            value is True for value in repetitive_reward_cap_values
        ),
        "high_reward_malformed_count": len(high_reward_malformed_rows),
        "high_reward_repetitive_count": len(high_reward_repetitive_rows),
        "reward_before_quality_caps_mean": (
            reward_before_caps_summary["mean"]
            if reward_before_caps_summary
            else None
        ),
        "reward_before_quality_caps_max": (
            reward_before_caps_summary["max"]
            if reward_before_caps_summary
            else None
        ),
        "reward_after_quality_caps_mean": (
            reward_after_caps_summary["mean"]
            if reward_after_caps_summary
            else None
        ),
        "reward_after_quality_caps_max": (
            reward_after_caps_summary["max"]
            if reward_after_caps_summary
            else None
        ),
        "positive_reward_source_distribution": positive_source_dist,
        "positive_reward_mean_by_source": positive_reward_mean_by_source(rows),
        "continuation_trigger_rate": continuation_trigger_rate,
        "continuation_candidate_count": len(cont_entries),
        "continuation_over_budget_rate": (
            sum(value is True for value in cont_over_budget) / len(cont_over_budget)
            if cont_over_budget
            else None
        ),
        "continuation_zero_remask_rate": (
            sum(value == 0 for value in cont_remask_counts) / len(cont_remask_counts)
            if cont_remask_counts
            else None
        ),
        "continuation_gain_rate": (
            sum(value > 1e-8 for value in cont_gains) / len(cont_gains)
            if cont_gains
            else None
        ),
        "avg_continuation_gain": mean(cont_gains) if cont_gains else None,
        "avg_best_continuation_gain": mean(best_cont_gains)
        if best_cont_gains
        else None,
        "continued_correct_base_wrong_count": continuation_acc[
            "continued_correct_base_wrong_count"
        ],
        "base_correct_continued_wrong_count": continuation_acc[
            "base_correct_continued_wrong_count"
        ],
        "continuation_fix_count": continuation_acc["continuation_fix_count"],
        "continuation_harm_count": continuation_acc["continuation_harm_count"],
        "net_fix_minus_harm": (
            continuation_acc["continuation_fix_count"]
            - continuation_acc["continuation_harm_count"]
        ),
        "avg_extra_steps": mean(cont_extra_steps) if cont_extra_steps else None,
        "continuation_gain_by_k": mean_continuation_gain_by(
            cont_entries,
            lambda _row, item: item.get("continuation_k"),
        ),
        "continuation_gain_by_trigger_reason": mean_continuation_gain_by(
            cont_entries,
            lambda _row, item: item.get("continuation_trigger_reason"),
        ),
        "continuation_gain_by_remask_mode": mean_continuation_gain_by(
            cont_entries,
            lambda _row, item: item.get("continuation_remask_mode"),
        ),
        "continuation_gain_by_budget": mean_continuation_gain_by(
            cont_entries,
            lambda row, _item: row.get("target_budget"),
        ),
        "base_accuracy": continuation_acc["base_accuracy"],
        "continued_accuracy": continuation_acc["continued_accuracy"],
        "best_continuation_accuracy": continuation_acc[
            "best_continuation_accuracy"
        ],
        "oracle_continue_accuracy": continuation_acc[
            "oracle_continue_accuracy"
        ],
        "oracle_best_accuracy": continuation_acc["oracle_continue_accuracy"],
        "oracle_possible_gain": (
            continuation_acc["oracle_continue_accuracy"]
            - continuation_acc["base_accuracy"]
            if continuation_acc["oracle_continue_accuracy"] is not None
            and continuation_acc["base_accuracy"] is not None
            else None
        ),
        "always_continue_gain": (
            continuation_acc["continued_accuracy"]
            - continuation_acc["base_accuracy"]
            if continuation_acc["continued_accuracy"] is not None
            and continuation_acc["base_accuracy"] is not None
            else None
        ),
        "quality_before_after": {
            "base_mean": mean(base_quality_values)
            if base_quality_values
            else None,
            "continued_mean": mean(continued_quality_values)
            if continued_quality_values
            else None,
            "oracle_best_mean": mean(best_quality_values)
            if best_quality_values
            else None,
        },
        "format_before_after": {
            "base_format_ok_rate": truthy_rate(base_format_values),
            "continued_format_ok_rate": truthy_rate(continued_format_values),
        },
        "target_budget_values": sorted(target_budget_set),
        "avg_nfe_by_target_budget": avg_nfe_budget,
        "stop_reason_distribution": dict(Counter(stop_reasons)),
        "stop_reason_distribution_by_target_budget": (
            value_distribution_by_target_budget(rows, "stop_reason")
        ),
        "parse_ok_rate": parse_rate,
        "format_ok_rate": format_rate,
        "arithmetic_ok_rate": arithmetic_rate,
        "strict_correct_rate": strict_correct_rate,
        "answer_span_correct_rate": answer_span_correct_rate,
        "robust_correct_rate": robust_correct_rate,
        "robust_only_correct_rate": len(robust_only_correct_rows) / len(rows)
        if rows
        else None,
        "robust_only_positive_reward_count": len(robust_only_positive_reward_rows),
        "format_ok_false_reward_positive_count": len(format_false_positive_rows),
        "empty_answer_tag_count": len(empty_answer_tag_rows),
        "malformed_answer_positive_count": len(malformed_positive_rows),
        "reward_answer_source_distribution": dict(
            Counter(str(value) for value in observed_values(rows, "reward_answer_source"))
        ),
        "robust_correct_but_reward_zero": len(robust_correct_but_reward_zero),
        "remask_count_distribution": dict(Counter(remask_counts)),
        "verifier_calls_distribution": dict(Counter(verifier_calls)),
        "has_boxed_answer_rate": boxed_rate,
        "has_complete_answer_tag_rate": answer_tag_rate,
        "strict_format_rate": strict_format_rate,
        "masked_token_fraction_mean": masked_token_fraction_mean,
        "masked_token_fraction_mean_by_target_budget": mean_by_target_budget(
            rows,
            "masked_token_fraction",
        ),
        "visible_generated_char_count_mean": mean(char_counts) if char_counts else None,
        "visible_generated_char_count_mean_by_target_budget": mean_by_target_budget(
            rows,
            "visible_generated_char_count",
        ),
        "zero_std_ratio_values": zero_std_values,
    }
    summary.update(_eval_summary(eval_rows))
    eval_zero_std_values = finite_values(
        _trainer_log_history_values(trainer_state, "eval_zero_std_ratio")
    )
    if eval_zero_std_values:
        summary["eval_zero_std_ratio"] = eval_zero_std_values[-1]
    return {
        "summary": summary,
        "warnings": warnings,
        "inconsistencies": inconsistencies,
    }


def _group_rows(rows: list[dict]) -> dict[tuple, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        if "group_id" not in row:
            continue
        grouped[(row.get("trainer_global_step"), row.get("group_id"))].append(row)
    return grouped


def _first_observed(rows: list[dict], key: str):
    for row in rows:
        if row.get(key) is not None:
            return row.get(key)
    return None


def _eval_summary(eval_rows: list[dict]) -> dict:
    rewards = observed_values(eval_rows, "reward")
    reward_summary = finite_summary(rewards)
    positive = [value for value in finite_values(rewards) if value > 1e-8]
    nfe_values = finite_values(
        [
            row.get("actual_nfe", row.get("nfe_used"))
            for row in eval_rows
            if row.get("actual_nfe", row.get("nfe_used")) is not None
        ]
    )
    return {
        "eval_reward_mean": reward_summary["mean"] if reward_summary else None,
        "eval_reward_std": reward_summary["std"] if reward_summary else None,
        "eval_reward_max": reward_summary["max"] if reward_summary else None,
        "eval_positive_reward_rate": len(positive) / len(eval_rows)
        if eval_rows
        else None,
        "eval_exact_match_strict": bool_rate(eval_rows, "strict_correct"),
        "eval_exact_match_answer_span": bool_rate(eval_rows, "answer_span_correct"),
        "eval_exact_match_robust": bool_rate(eval_rows, "robust_correct"),
        "eval_parse_ok_rate": bool_rate(eval_rows, "parse_ok"),
        "eval_format_ok_rate": bool_rate(eval_rows, "format_ok"),
        "eval_arithmetic_ok_rate": bool_rate(eval_rows, "arithmetic_ok"),
        "eval_avg_nfe": mean(nfe_values) if nfe_values else None,
        "eval_avg_nfe_by_target_budget": avg_nfe_by_budget(eval_rows),
        "eval_zero_std_ratio": None,
        "eval_masked_token_fraction_mean": _mean_observed(
            eval_rows,
            "masked_token_fraction",
        ),
        "eval_has_boxed_answer_rate": bool_rate(eval_rows, "has_boxed_answer"),
        "eval_has_complete_answer_tag_rate": bool_rate(
            eval_rows,
            "has_complete_answer_tag",
        ),
        "eval_visible_generated_char_count_mean": _mean_observed(
            eval_rows,
            "visible_generated_char_count",
        ),
    }


def _mean_observed(rows: list[dict], key: str) -> float | None:
    values = finite_values(observed_values(rows, key))
    return mean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--min_examples", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    config = load_config(config_path)
    rows = read_jsonl(output_dir / "per_example.jsonl")
    generations = read_jsonl(output_dir / "generations.jsonl")
    verifier_rows = read_jsonl(output_dir / "verifier_results.jsonl")
    trainer_state = latest_trainer_state(output_dir)
    log_counts = parse_train_log_counts(output_dir)
    checkpoint_count = len(list(output_dir.glob("checkpoint-*"))) if output_dir.exists() else 0

    result = analyze_training_consistency(
        config=config,
        rows=rows,
        generations=generations,
        verifier_rows=verifier_rows,
        trainer_state=trainer_state,
        log_counts=log_counts,
        min_examples=args.min_examples,
        output_dir=output_dir,
        checkpoint_count=checkpoint_count,
    )

    print(f"config: {config_path}")
    print(f"output_dir: {output_dir}")
    print(f"log sample counts: {log_counts}")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")
    if result["warnings"]:
        print("warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
    if result["inconsistencies"]:
        print("inconsistencies:")
        for inconsistency in result["inconsistencies"]:
            print(f"  - {inconsistency}")
        return 1
    print("consistency: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
