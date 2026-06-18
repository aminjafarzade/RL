import importlib.util
import json
from pathlib import Path

import pytest


def _load_budget_bandit():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_budget_conditioned_neural_bandit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_budget_conditioned_neural_bandit",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(**overrides):
    row = {
        "example_id": "ex-0",
        "mode": "train",
        "target_budget": 32,
        "base_nfe": 20,
        "remaining_budget": 12,
        "base_parse_ok": True,
        "base_format_ok": False,
        "base_arithmetic_ok": None,
        "base_verifier_score": 0.2,
        "base_reward_answer_source": "final_window",
        "base_repetition_penalty": 1.0,
        "base_visible_generated_char_count": 100,
        "base_answer_span_confidence": 0.7,
        "base_entropy_stats": {"mean": 0.2, "max": 0.4},
        "base_confidence_stats": {"mean": 0.8, "min": 0.4},
        "continuation_k": 1,
        "continuation_remasked_token_count": 1,
        "continuation_over_budget": False,
        "continued_extra_steps": 1,
        "continuation_extra_steps": 1,
        "actual_extra_steps": 1,
        "base_correct": False,
        "continued_correct": True,
    }
    row.update(overrides)
    return row


def _pattern_rows(mode="train", repeats=8):
    patterns = [
        {
            "name": "cheap_fix",
            "base_correct": False,
            "continued_correct": True,
            "base_format_ok": False,
            "base_verifier_score": 0.1,
            "base_reward_answer_source": "final_window",
            "continuation_k": 1,
            "continued_extra_steps": 1,
            "actual_extra_steps": 1,
        },
        {
            "name": "expensive_fix",
            "base_correct": False,
            "continued_correct": True,
            "base_format_ok": False,
            "base_verifier_score": 0.15,
            "base_reward_answer_source": "final_window",
            "continuation_k": 20,
            "continued_extra_steps": 20,
            "actual_extra_steps": 20,
        },
        {
            "name": "harm",
            "base_correct": True,
            "continued_correct": False,
            "base_format_ok": True,
            "base_verifier_score": 0.95,
            "base_reward_answer_source": "strict_answer_tag",
            "continuation_k": 1,
            "continued_extra_steps": 1,
            "actual_extra_steps": 1,
        },
        {
            "name": "both_correct",
            "base_correct": True,
            "continued_correct": True,
            "base_format_ok": True,
            "base_verifier_score": 0.9,
            "base_reward_answer_source": "strict_answer_tag",
            "continuation_k": 1,
            "continued_extra_steps": 1,
            "actual_extra_steps": 1,
        },
        {
            "name": "both_wrong",
            "base_correct": False,
            "continued_correct": False,
            "base_format_ok": False,
            "base_verifier_score": 0.55,
            "base_reward_answer_source": "answer_marker",
            "continuation_k": 1,
            "continued_extra_steps": 1,
            "actual_extra_steps": 1,
        },
    ]
    rows = []
    for rep in range(repeats):
        for pattern in patterns:
            name = pattern["name"]
            fields = {key: value for key, value in pattern.items() if key != "name"}
            rows.append(_row(example_id=f"{mode}-{name}-{rep}", mode=mode, **fields))
    return rows


def test_row_duplication_expands_by_train_lambdas():
    bandit = _load_budget_bandit()
    rows = [_row(example_id="a"), _row(example_id="b")]
    duplicated = bandit.duplicate_rows_for_lambdas(
        rows,
        [0.0, 0.1, 0.2],
        harm_penalty=0.0,
        correctness_reward=1.0,
    )

    assert len(duplicated) == len(rows) * 3
    assert [row["requested_cost_lambda"] for row in duplicated] == [
        0.0,
        0.1,
        0.2,
        0.0,
        0.1,
        0.2,
    ]


def test_requested_cost_lambda_is_present_in_input_features():
    bandit = _load_budget_bandit()
    features = bandit.feature_dict(_row(requested_cost_lambda=0.2))

    assert features["requested_cost_lambda"] == pytest.approx(0.2)


def test_changing_requested_cost_lambda_changes_rewards():
    bandit = _load_budget_bandit()
    low = _row(requested_cost_lambda=0.0, continuation_k=20, continued_extra_steps=20)
    high = _row(requested_cost_lambda=0.2, continuation_k=20, continued_extra_steps=20)

    low_rewards = bandit.budget_conditioned_rewards(
        low,
        harm_penalty=0.0,
        correctness_reward=1.0,
    )
    high_rewards = bandit.budget_conditioned_rewards(
        high,
        harm_penalty=0.0,
        correctness_reward=1.0,
    )

    assert low_rewards["r_continue"] == pytest.approx(1.0)
    assert high_rewards["r_continue"] == pytest.approx(-3.0)
    assert low_rewards["oracle_continue"] is True
    assert high_rewards["oracle_continue"] is False


def test_actual_and_continued_extra_steps_are_not_input_features():
    bandit = _load_budget_bandit()
    features = bandit.feature_dict(
        _row(
            requested_cost_lambda=0.1,
            continued_extra_steps=999,
            actual_extra_steps=999,
            continued_correct=True,
        )
    )

    assert "continued_extra_steps" not in features
    assert "actual_extra_steps" not in features
    assert "continued_correct" not in features
    assert "continuation_k" in features


def test_per_lambda_validation_thresholds_use_validation_only():
    bandit = _load_budget_bandit()
    validation_rows = bandit.duplicate_rows_by_lambda(
        [
            _row(example_id="val-fix", base_correct=False, continued_correct=True),
            _row(example_id="val-harm", base_correct=True, continued_correct=False),
        ],
        [0.0, 0.2],
        harm_penalty=1.0,
        correctness_reward=1.0,
    )
    final_rows = bandit.duplicate_rows_by_lambda(
        [_row(example_id="eval-harm", base_correct=True, continued_correct=False)],
        [0.0, 0.2],
        harm_penalty=1.0,
        correctness_reward=1.0,
    )

    selected = bandit.select_threshold(
        validation_rows[0.0],
        [0.9, 0.6],
        cost_lambda=0.0,
        harm_penalty=1.0,
        correctness_reward=1.0,
        threshold_grid=[0.5, 0.8],
    )

    assert selected["selected_threshold"] == pytest.approx(0.8)
    assert selected["selected_validation_metrics"]["example_count"] == 2
    assert len(final_rows[0.0]) == 1


def test_same_checkpoint_is_evaluated_for_multiple_lambdas(tmp_path):
    bandit = _load_budget_bandit()
    dataset = tmp_path / "gate.jsonl"
    rows = _pattern_rows(mode="train", repeats=6) + _pattern_rows(mode="eval", repeats=2)
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = bandit.train_budget_conditioned_bandit(
        dataset=dataset,
        output_dir=tmp_path / "out",
        train_cost_lambdas=[0.0, 0.2],
        eval_cost_lambdas=[0.0, 0.2],
        harm_penalty=0.5,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        lr=0.03,
        weight_decay=0.0,
        epochs=80,
        batch_size=16,
        validation_fraction=0.2,
        patience=20,
        entropy_coef=0.0,
        aux_bce_weight=0.5,
        threshold_mode="per_lambda_validation",
        bootstrap_samples=0,
        seed=123,
        device="cpu",
    )

    assert result["artifacts"]["model"].endswith(
        "budget_conditioned_neural_bandit_policy.pt"
    )
    assert result["same_checkpoint_eval_lambdas"] == [0.0, 0.2]
    assert sorted(result["final_eval_by_lambda"]) == ["0.0", "0.2"]


def test_output_files_are_written(tmp_path):
    bandit = _load_budget_bandit()
    dataset = tmp_path / "gate.jsonl"
    rows = _pattern_rows(mode="train", repeats=4) + _pattern_rows(mode="eval", repeats=2)
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output_dir = tmp_path / "out"

    result = bandit.train_budget_conditioned_bandit(
        dataset=dataset,
        output_dir=output_dir,
        train_cost_lambdas=[0.0, 0.2],
        eval_cost_lambdas=[0.0, 0.2],
        harm_penalty=0.5,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        lr=0.03,
        weight_decay=0.0,
        epochs=30,
        batch_size=8,
        validation_fraction=0.2,
        patience=10,
        entropy_coef=0.0,
        aux_bce_weight=0.5,
        threshold_mode="fixed_0_5",
        bootstrap_samples=2,
        seed=123,
        device="cpu",
    )

    assert (output_dir / "budget_conditioned_neural_bandit_policy.pt").exists()
    assert (output_dir / "feature_names.json").exists()
    assert (output_dir / "training_history.jsonl").exists()
    assert (output_dir / "budget_conditioned_result.json").exists()
    assert (output_dir / "budget_conditioned_result_pretty.md").exists()
    assert (output_dir / "budget_conditioned_eval_predictions.jsonl").exists()
    assert result["artifacts"]["feature_preprocessor"] is None or (
        output_dir / "feature_preprocessor.joblib"
    ).exists()


def test_higher_requested_cost_lambda_reduces_trigger_rate(tmp_path):
    bandit = _load_budget_bandit()
    dataset = tmp_path / "gate.jsonl"
    rows = _pattern_rows(mode="train", repeats=12) + _pattern_rows(mode="eval", repeats=4)
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = bandit.train_budget_conditioned_bandit(
        dataset=dataset,
        output_dir=tmp_path / "out",
        train_cost_lambdas=[0.0, 0.2],
        eval_cost_lambdas=[0.0, 0.2],
        harm_penalty=0.5,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        lr=0.03,
        weight_decay=0.0,
        epochs=120,
        batch_size=16,
        validation_fraction=0.2,
        patience=30,
        entropy_coef=0.0,
        aux_bce_weight=0.5,
        threshold_mode="fixed_0_5",
        bootstrap_samples=0,
        seed=123,
        device="cpu",
    )

    low = result["final_eval_by_lambda"]["0.0"]["budget_conditioned_neural_policy"]
    high = result["final_eval_by_lambda"]["0.2"]["budget_conditioned_neural_policy"]
    assert high["trigger_rate"] < low["trigger_rate"]
