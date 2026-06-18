import importlib.util
import json
import math
from pathlib import Path

import pytest


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_continuation_analysis_help_includes_resume_options():
    analysis = _load_script("run_continuation_analysis.py")
    help_text = analysis.build_arg_parser().format_help()

    assert "--resume" in help_text
    assert "--start_train_index" in help_text


def test_run_continuation_analysis_resume_preserves_jsonl_files(tmp_path):
    analysis = _load_script("run_continuation_analysis.py")
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    per_example_path = output_dir / "per_example.jsonl"
    generations_path = output_dir / "generations.jsonl"
    failed_path = output_dir / "failed_batches.jsonl"
    per_example_path.write_text("existing per-example\n")
    generations_path.write_text("existing generation\n")
    failed_path.write_text("existing failure\n")

    analysis.prepare_output_files(
        output_dir,
        resume=True,
        failed_batches_path=failed_path,
    )

    assert per_example_path.read_text() == "existing per-example\n"
    assert generations_path.read_text() == "existing generation\n"
    assert failed_path.read_text() == "existing failure\n"


def test_run_continuation_analysis_resume_index_range_uses_max_as_end():
    analysis = _load_script("run_continuation_analysis.py")

    assert analysis.split_index_range(
        config={"max_train_samples": 3000},
        split="train",
        dataset_len=7473,
        resume=True,
        start_index=2392,
        end_index=None,
    ) == (2392, 3000)
    assert analysis.split_index_range(
        config={"max_train_samples": 3000},
        split="train",
        dataset_len=7473,
        resume=True,
        start_index=2392,
        end_index=2400,
    ) == (2392, 2400)


def test_export_continuation_gate_dataset_excludes_reference_answer(tmp_path):
    exporter = _load_script("export_continuation_gate_dataset.py")
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    row = {
        "example_id": "train-0",
        "mode": "train",
        "prompt": "question text",
        "reference_answer": "42",
        "target_budget": 32,
        "base_nfe": 20,
        "remaining_budget": 12,
        "base_parse_ok": True,
        "base_format_ok": False,
        "base_arithmetic_ok": None,
        "base_verifier_score": 0.5,
        "base_reward_answer_source": "final_window",
        "base_repetition_penalty": 1.0,
        "base_answer_quality": 0.45,
        "base_format_factor": 0.5,
        "base_visible_generated_char_count": 100,
        "base_answer_span_confidence": 0.7,
        "base_entropy_stats": {"mean": 0.2},
        "base_confidence_stats": {"mean": 0.8},
        "base_correct": False,
        "continuation_results": [
            {
                "continuation_k": 4,
                "continuation_gain": 0.45,
                "continuation_correct_gain": True,
                "continuation_harm": False,
                "continued_correct": True,
                "continued_generated_text": "hidden from features",
            }
        ],
    }
    with (output_dir / "per_example.jsonl").open("w") as f:
        f.write(json.dumps(row) + "\n")

    out_path = exporter.export_gate_dataset(output_dir)
    exported = json.loads(out_path.read_text().strip())

    assert exported["label_gain_positive"] is True
    assert exported["label_correct_gain"] is True
    assert "base_correct" in exported
    assert "continued_correct" in exported
    assert "continuation_gain" in exported
    forbidden = {
        "base_answer_quality",
        "base_format_factor",
        "reference_answer",
        "answer",
        "prompt",
        "generated_text",
        "base_generated_text",
        "continued_generated_text",
    }
    assert forbidden.isdisjoint(exported)


def test_export_continuation_gate_dataset_rejects_leaky_feature_names():
    exporter = _load_script("export_continuation_gate_dataset.py")

    with pytest.raises(ValueError, match="Forbidden key"):
        exporter.validate_feature_keys(["target_budget", "base_task_score"])

    with pytest.raises(ValueError, match="Forbidden key"):
        exporter.validate_feature_keys(["reference_answer"])

    with pytest.raises(ValueError, match="Potentially leaked feature key"):
        exporter.validate_feature_keys(["target_budget", "some_reward_like_feature"])


def test_heuristic_gate_metrics_compute_fixes_and_harms():
    evaluator = _load_script("evaluate_continuation_heuristic_gate.py")
    rows = [
        {
            "base_correct": False,
            "continued_correct": True,
            "continuation_gain": 0.45,
            "base_format_ok": False,
            "base_reward_answer_source": "answer_marker",
            "base_repetition_penalty": 1.0,
            "remaining_budget": 8,
            "continuation_k": 4,
        },
        {
            "base_correct": True,
            "continued_correct": False,
            "continuation_gain": -1.0,
            "base_format_ok": False,
            "base_reward_answer_source": "answer_marker",
            "base_repetition_penalty": 1.0,
            "remaining_budget": 8,
            "continuation_k": 4,
        },
        {
            "base_correct": False,
            "continued_correct": False,
            "continuation_gain": 0.0,
            "base_format_ok": True,
            "base_reward_answer_source": "strict_answer_tag",
            "base_repetition_penalty": 1.0,
            "remaining_budget": 8,
            "continuation_k": 4,
        },
    ]

    result = evaluator.evaluate_policy(
        rows,
        "heuristic_format",
        evaluator.policy_format,
    )

    assert result["trigger_rate"] == pytest.approx(2 / 3)
    assert result["fixes"] == 1
    assert result["harms"] == 1
    assert result["net_fix_minus_harm"] == 0
    assert result["trigger_gain_precision"] == pytest.approx(0.5)
    assert result["trigger_gain_recall"] == pytest.approx(1.0)


def test_policy_checkpoint_path_validation_raises_for_missing_path(tmp_path):
    analysis = _load_script("run_continuation_analysis.py")
    missing = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError, match="policy_checkpoint_path does not exist"):
        analysis.resolve_policy_checkpoint_file(str(missing))


def _gate_row(**overrides):
    row = {
        "example_id": "train-0",
        "mode": "train",
        "target_budget": 32,
        "base_nfe": 20,
        "remaining_budget": 12,
        "base_parse_ok": True,
        "base_format_ok": False,
        "base_arithmetic_ok": None,
        "base_verifier_score": 0.5,
        "base_reward_answer_source": "final_window",
        "base_repetition_penalty": 1.0,
        "base_visible_generated_char_count": 100,
        "base_answer_span_confidence": None,
        "base_entropy_stats": {"mean": 0.2, "max": None},
        "base_confidence_stats": {"mean": 0.8},
        "continuation_k": 4,
        "continuation_gain": 0.45,
        "label_gain_positive": True,
        "label_correct_gain": True,
        "label_harm": False,
        "base_correct": False,
        "continued_correct": True,
    }
    row.update(overrides)
    return row


def _assert_no_none_or_nan(features):
    for key, value in features.items():
        assert value is not None, key
        if isinstance(value, float):
            assert not math.isnan(value), key


def test_train_gate_feature_dict_handles_ternary_and_numeric_missing_values():
    gate = _load_script("train_continuation_gate.py")
    features = gate.feature_dict(
        _gate_row(base_arithmetic_ok=None, base_answer_span_confidence=None)
    )

    assert features["base_arithmetic_ok"] == "unknown"
    assert features["base_answer_span_confidence"] == 0.0
    assert features["base_answer_span_confidence__missing"] == 1.0
    assert features["base_entropy_stats.max"] == 0.0
    assert features["base_entropy_stats.max__missing"] == 1.0
    _assert_no_none_or_nan(features)


def test_train_gate_feature_dict_marks_present_numeric_values_not_missing():
    gate = _load_script("train_continuation_gate.py")
    features = gate.feature_dict(
        _gate_row(base_arithmetic_ok=True, base_answer_span_confidence=0.73)
    )

    assert features["base_arithmetic_ok"] == "true"
    assert features["base_answer_span_confidence"] == pytest.approx(0.73)
    assert features["base_answer_span_confidence__missing"] == 0.0
    _assert_no_none_or_nan(features)


def test_train_gate_tiny_dataset_fits_with_missing_values():
    pytest.importorskip("sklearn")
    gate = _load_script("train_continuation_gate.py")
    rows = [
        _gate_row(
            mode="train",
            label_gain_positive=True,
            continuation_gain=0.2,
            base_arithmetic_ok=None,
            base_answer_span_confidence=None,
        ),
        _gate_row(
            mode="train",
            label_gain_positive=False,
            continuation_gain=-0.1,
            base_arithmetic_ok=False,
            base_answer_span_confidence=0.1,
            base_reward_answer_source="strict_answer_tag",
        ),
        _gate_row(
            mode="train",
            label_gain_positive=True,
            continuation_gain=0.3,
            base_arithmetic_ok=True,
            base_answer_span_confidence=0.8,
        ),
        _gate_row(
            mode="train",
            label_gain_positive=False,
            continuation_gain=0.0,
            base_arithmetic_ok=None,
            base_answer_span_confidence=None,
            base_reward_answer_source="strict_answer_tag",
        ),
        _gate_row(
            mode="eval",
            label_gain_positive=True,
            continuation_gain=0.15,
            base_arithmetic_ok=None,
            base_answer_span_confidence=None,
        ),
        _gate_row(
            mode="eval",
            label_gain_positive=False,
            continuation_gain=0.0,
            base_arithmetic_ok=False,
            base_answer_span_confidence=0.2,
            base_reward_answer_source="strict_answer_tag",
        ),
    ]

    result = gate.train_and_evaluate(
        rows,
        target_label="label_gain_positive",
        threshold=0.5,
    )

    assert result["status"] == "ok"
    assert result["train_rows"] == 4
    assert result["eval_rows"] == 2
    assert result["label_positive_rate_train"] == pytest.approx(0.5)
    assert result["missing_feature_counts"]["base_arithmetic_ok"] == 3
    assert result["missing_feature_counts"]["base_answer_span_confidence"] == 3
