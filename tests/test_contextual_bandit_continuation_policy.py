import importlib.util
from pathlib import Path

import pytest


def _load_bandit():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_contextual_bandit_continuation_policy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_contextual_bandit_continuation_policy",
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
        "base_verifier_score": 0.55,
        "base_reward_answer_source": "final_window",
        "base_repetition_penalty": 1.0,
        "base_visible_generated_char_count": 100,
        "base_answer_span_confidence": 0.7,
        "base_entropy_stats": {"mean": 0.2, "max": 0.4},
        "base_confidence_stats": {"mean": 0.8, "min": 0.4},
        "continuation_k": 4,
        "continuation_remasked_token_count": 1,
        "continuation_over_budget": False,
        "continued_extra_steps": 2,
        "continuation_extra_steps": 4,
        "base_correct": False,
        "continued_correct": True,
    }
    row.update(overrides)
    return row


def test_bandit_reward_and_oracle_actions_cover_synthetic_cases():
    bandit = _load_bandit()
    rows = [
        _row(example_id="A", base_correct=False, continued_correct=True),
        _row(example_id="B", base_correct=True, continued_correct=False),
        _row(example_id="C", base_correct=True, continued_correct=True),
        _row(example_id="D", base_correct=False, continued_correct=False),
        _row(
            example_id="E",
            base_correct=False,
            continued_correct=True,
            continued_extra_steps=2,
        ),
    ]

    low_cost = [
        bandit.bandit_rewards(
            row,
            cost_lambda=0.1,
            harm_penalty=0.5,
            correctness_reward=1.0,
        )
        for row in rows[:4]
    ]
    high_cost = bandit.bandit_rewards(
        rows[4],
        cost_lambda=0.6,
        harm_penalty=0.0,
        correctness_reward=1.0,
    )

    assert low_cost[0]["delta_reward"] == pytest.approx(0.8)
    assert low_cost[0]["oracle_continue"] is True
    assert low_cost[1]["oracle_continue"] is False
    assert low_cost[2]["oracle_continue"] is False
    assert low_cost[3]["oracle_continue"] is False
    assert high_cost["delta_reward"] == pytest.approx(-0.2)
    assert high_cost["oracle_continue"] is False


def test_evaluate_bandit_policy_computes_utility_fixes_and_harms():
    bandit = _load_bandit()
    rows = [
        _row(example_id="A", base_correct=False, continued_correct=True),
        _row(example_id="B", base_correct=True, continued_correct=False),
        _row(example_id="C", base_correct=True, continued_correct=True),
        _row(example_id="D", base_correct=False, continued_correct=False),
    ]
    triggered = [True, False, False, False]

    result = bandit.evaluate_bandit_policy(
        rows,
        triggered,
        cost_lambda=0.1,
        harm_penalty=0.5,
        correctness_reward=1.0,
    )

    assert result["example_count"] == 4
    assert result["base_accuracy"] == pytest.approx(0.5)
    assert result["always_continue_accuracy"] == pytest.approx(0.5)
    assert result["policy_accuracy"] == pytest.approx(0.75)
    assert result["policy_utility_mean"] == pytest.approx((0.8 + 1 + 1 + 0) / 4)
    assert result["fixes"] == 1
    assert result["harms"] == 0
    assert result["missed_fixes_vs_always"] == 0
    assert result["avoided_harms_vs_always"] == 1


def test_feature_dict_ignores_leaky_row_fields():
    bandit = _load_bandit()
    row = _row(
        prompt="do not use",
        reference_answer="42",
        generated_text="do not use",
        reward=1.0,
        task_score=1.0,
        answer_quality=1.0,
    )

    bandit.validate_no_feature_leakage()
    features = bandit.feature_dict(row)

    for key in bandit.FORBIDDEN_FEATURE_FIELDS:
        assert key not in features
    assert "base_reward_answer_source" in features


def test_mode_split_and_train_validation_split_by_example_id():
    bandit = _load_bandit()
    rows = [
        _row(example_id="t1", mode="train", continuation_k=2),
        _row(example_id="t1", mode="train", continuation_k=4),
        _row(example_id="t2", mode="train", continuation_k=4),
        _row(example_id="e1", mode="eval", continuation_k=4),
    ]

    train_rows, eval_rows, fallback = bandit.split_by_mode(rows)
    fit_rows, validation_rows = bandit.split_train_validation(
        train_rows,
        validation_fraction=0.5,
        seed=123,
    )

    assert fallback is False
    assert len(train_rows) == 3
    assert len(eval_rows) == 1
    assert len(fit_rows) + len(validation_rows) == 3
    assert {row["example_id"] for row in fit_rows}.isdisjoint(
        {row["example_id"] for row in validation_rows}
    )
    assert fit_rows == [row for row in train_rows if row in fit_rows]
    assert validation_rows == [row for row in train_rows if row in validation_rows]


def test_missing_mode_split_falls_back_to_ordered_80_20():
    bandit = _load_bandit()
    rows = [_row(example_id=f"ex-{idx}", mode=None) for idx in range(5)]

    train_rows, eval_rows, fallback = bandit.split_by_mode(rows)

    assert fallback is True
    assert [row["example_id"] for row in train_rows] == [
        "ex-0",
        "ex-1",
        "ex-2",
        "ex-3",
    ]
    assert [row["example_id"] for row in eval_rows] == ["ex-4"]


def test_threshold_selection_uses_validation_metrics_not_eval_rows():
    bandit = _load_bandit()
    validation_rows = bandit.add_bandit_labels(
        [
            _row(example_id="val-fix", base_correct=False, continued_correct=True),
            _row(example_id="val-harm", base_correct=True, continued_correct=False),
        ],
        cost_lambda=0.0,
        harm_penalty=1.0,
        correctness_reward=1.0,
    )
    validation_scores = [0.9, 0.6]

    selected = bandit.select_threshold(
        validation_rows,
        validation_scores,
        model_name="logistic",
        cost_lambda=0.0,
        harm_penalty=1.0,
        correctness_reward=1.0,
        threshold_grid=[0.5, 0.8],
    )

    assert selected["selected_threshold"] == pytest.approx(0.8)
    assert selected["selected_validation_metrics"]["fixes"] == 1
    assert selected["selected_validation_metrics"]["harms"] == 0


def test_eval_rows_counted_exactly_once():
    bandit = _load_bandit()
    eval_rows = [
        _row(
            example_id="eval-1",
            mode="eval",
            base_correct=False,
            continued_correct=True,
        ),
        _row(
            example_id="eval-2",
            mode="eval",
            base_correct=True,
            continued_correct=False,
        ),
    ]

    result = bandit.evaluate_bandit_policy(
        eval_rows,
        [True, False],
        cost_lambda=0.0,
        harm_penalty=0.0,
        correctness_reward=1.0,
    )

    assert result["example_count"] == 2
    assert result["fixes"] == 1
    assert result["harms"] == 0
