import importlib.util
import json
from pathlib import Path

import pytest
import torch


def _load_neural_bandit():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_neural_continuation_bandit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_neural_continuation_bandit",
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
            "name": "fix",
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
        {
            "name": "too_costly",
            "base_correct": False,
            "continued_correct": True,
            "base_format_ok": False,
            "base_verifier_score": 0.15,
            "base_reward_answer_source": "final_window",
            "continuation_k": 20,
            "continued_extra_steps": 20,
            "actual_extra_steps": 20,
        },
    ]
    rows = []
    for rep in range(repeats):
        for pattern in patterns:
            name = pattern["name"]
            fields = {key: value for key, value in pattern.items() if key != "name"}
            rows.append(_row(example_id=f"{mode}-{name}-{rep}", mode=mode, **fields))
    return rows


def test_reward_construction_and_oracle_cases():
    bandit = _load_neural_bandit()
    rows = [
        _row(example_id="A", base_correct=False, continued_correct=True),
        _row(example_id="B", base_correct=True, continued_correct=False),
        _row(example_id="C", base_correct=True, continued_correct=True),
        _row(example_id="D", base_correct=False, continued_correct=False),
        _row(
            example_id="E",
            base_correct=False,
            continued_correct=True,
            continuation_k=20,
            continued_extra_steps=20,
            actual_extra_steps=20,
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
        cost_lambda=0.2,
        harm_penalty=0.0,
        correctness_reward=1.0,
    )

    assert low_cost[0]["r_stop"] == pytest.approx(0.0)
    assert low_cost[0]["r_continue"] == pytest.approx(0.9)
    assert low_cost[0]["oracle_continue"] is True
    assert low_cost[1]["oracle_continue"] is False
    assert low_cost[2]["oracle_continue"] is False
    assert low_cost[3]["oracle_continue"] is False
    assert high_cost["r_continue"] == pytest.approx(-3.0)
    assert high_cost["oracle_continue"] is False


def test_no_forbidden_feature_fields_are_used_as_inputs():
    bandit = _load_neural_bandit()
    row = _row(
        prompt="hidden",
        reference_answer="42",
        generated_text="hidden",
        generated_text_final="hidden",
        base_generated_text="hidden",
        continued_generated_text="hidden",
        answer_quality=1.0,
        task_score=1.0,
        reward=1.0,
        ground_truth="42",
        answer="42",
        continued_extra_steps=999,
        continuation_extra_steps=999,
        actual_extra_steps=999,
    )

    bandit.validate_no_feature_leakage()
    features = bandit.feature_dict(row)

    for key in bandit.FORBIDDEN_INPUT_FIELDS:
        assert key not in features
    assert "continuation_k" in features


def test_forbidden_post_action_fields_are_not_model_inputs():
    bandit = _load_neural_bandit()
    row = _row(
        continued_extra_steps=999,
        actual_extra_steps=999,
        continued_correct=True,
        continuation_gain=1.0,
        label_gain_positive=True,
        label_harm=False,
        reward=1.0,
        generated_text="hidden",
        reference_answer="42",
    )

    features = bandit.feature_dict(row)

    for key in {
        "continued_extra_steps",
        "actual_extra_steps",
        "continued_correct",
        "continuation_gain",
        "label_gain_positive",
        "label_harm",
        "reward",
        "generated_text",
        "reference_answer",
    }:
        assert key not in features


def test_expected_utility_loss_decreases_on_synthetic_data():
    bandit = _load_neural_bandit()
    bandit.set_seed(123)
    rows = bandit.add_bandit_labels(
        _pattern_rows(repeats=6),
        cost_lambda=0.1,
        harm_penalty=0.5,
        correctness_reward=1.0,
    )
    device = torch.device("cpu")
    preprocessor = bandit.fit_feature_preprocessor(rows)
    tensors = bandit.build_tensors(preprocessor, rows, device)
    model = bandit.NeuralContinuationBandit(
        input_dim=tensors["features"].shape[1],
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)

    with torch.no_grad():
        initial = bandit.bandit_loss_from_logits(
            model(tensors["features"]),
            tensors["r_stop"],
            tensors["r_continue"],
            tensors["oracle"],
            entropy_coef=0.0,
            aux_bce_weight=0.5,
        )["total_loss"].item()

    for _ in range(80):
        bandit.train_epoch(
            model,
            tensors,
            optimizer,
            batch_size=16,
            entropy_coef=0.0,
            aux_bce_weight=0.5,
        )

    with torch.no_grad():
        final = bandit.bandit_loss_from_logits(
            model(tensors["features"]),
            tensors["r_stop"],
            tensors["r_continue"],
            tensors["oracle"],
            entropy_coef=0.0,
            aux_bce_weight=0.5,
        )["total_loss"].item()

    assert final < initial


def test_trained_neural_policy_learns_oracle_actions_on_synthetic_data(tmp_path):
    bandit = _load_neural_bandit()
    dataset = tmp_path / "gate.jsonl"
    rows = _pattern_rows(mode="train", repeats=10) + _pattern_rows(
        mode="eval",
        repeats=3,
    )
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = bandit.train_neural_bandit(
        dataset=dataset,
        output_dir=tmp_path / "out",
        cost_lambda=0.1,
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
        bootstrap_samples=0,
        seed=123,
        device="cpu",
    )

    learned = result["final_eval"]["neural_bandit_policy"]
    oracle = result["final_eval"]["utility_oracle_policy"]
    assert learned["example_count"] == 15
    assert learned["oracle_action_is_utility_optimal"] is True
    assert oracle["oracle_action_is_utility_optimal"] is True
    assert learned["policy_utility_mean"] >= oracle["policy_utility_mean"] - 5e-2
    assert learned["policy_regret_vs_oracle"] <= 5e-2
    assert learned["oracle_action_match_rate"] >= 0.95


def test_threshold_selection_uses_validation_not_final_eval():
    bandit = _load_neural_bandit()
    validation_rows = bandit.add_bandit_labels(
        [
            _row(example_id="val-fix", base_correct=False, continued_correct=True),
            _row(example_id="val-harm", base_correct=True, continued_correct=False),
        ],
        cost_lambda=0.0,
        harm_penalty=1.0,
        correctness_reward=1.0,
    )

    selected = bandit.select_threshold(
        validation_rows,
        [0.9, 0.6],
        cost_lambda=0.0,
        harm_penalty=1.0,
        correctness_reward=1.0,
        threshold_grid=[0.5, 0.8],
    )

    assert selected["selected_threshold"] == pytest.approx(0.8)
    assert selected["selected_validation_metrics"]["fixes"] == 1
    assert selected["selected_validation_metrics"]["harms"] == 0


def test_final_eval_rows_counted_exactly_once():
    bandit = _load_neural_bandit()
    rows = [
        _row(example_id="eval-1", mode="eval", base_correct=False, continued_correct=True),
        _row(example_id="eval-2", mode="eval", base_correct=True, continued_correct=False),
    ]

    metrics = bandit.evaluate_policy(
        rows,
        [0.9, 0.1],
        threshold=0.5,
        cost_lambda=0.0,
        harm_penalty=0.0,
        correctness_reward=1.0,
    )

    assert metrics["example_count"] == 2
    assert metrics["fixes"] == 1
    assert metrics["harms"] == 0


def test_bootstrap_does_not_retrain_model(monkeypatch):
    bandit = _load_neural_bandit()

    def fail_train_epoch(*_args, **_kwargs):
        raise AssertionError("bootstrap should not train")

    monkeypatch.setattr(bandit, "train_epoch", fail_train_epoch)
    rows = bandit.add_bandit_labels(
        _pattern_rows(mode="eval", repeats=2),
        cost_lambda=0.1,
        harm_penalty=0.5,
        correctness_reward=1.0,
    )
    intervals = bandit.bootstrap_confidence_intervals(
        rows,
        [0.9 if row["oracle_continue"] else 0.1 for row in rows],
        0.5,
        cost_lambda=0.1,
        harm_penalty=0.5,
        correctness_reward=1.0,
        bootstrap_samples=5,
        seed=123,
    )

    assert "policy_accuracy" in intervals


def test_output_files_are_written(tmp_path):
    bandit = _load_neural_bandit()
    dataset = tmp_path / "gate.jsonl"
    rows = _pattern_rows(mode="train", repeats=4) + _pattern_rows(mode="eval", repeats=2)
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output_dir = tmp_path / "out"

    result = bandit.train_neural_bandit(
        dataset=dataset,
        output_dir=output_dir,
        cost_lambda=0.1,
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
        bootstrap_samples=2,
        seed=123,
        device="cpu",
    )

    assert (output_dir / "neural_bandit_policy.pt").exists()
    assert (output_dir / "feature_names.json").exists()
    assert (output_dir / "training_history.jsonl").exists()
    assert (output_dir / "neural_bandit_result.json").exists()
    assert (output_dir / "neural_bandit_result_pretty.md").exists()
    assert (output_dir / "neural_bandit_eval_predictions.jsonl").exists()
    assert result["artifacts"]["feature_preprocessor"] is None or (
        output_dir / "feature_preprocessor.joblib"
    ).exists()
