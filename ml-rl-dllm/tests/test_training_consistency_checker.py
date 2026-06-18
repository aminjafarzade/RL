import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_training_consistency.py"
_SPEC = importlib.util.spec_from_file_location("check_training_consistency", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
analyze_training_consistency = _MODULE.analyze_training_consistency


def _base_config():
    return {
        "dataset": "gsm8k",
        "target_budgets": [32],
        "train_budget_sampling": [32],
        "num_generations": 1,
        "max_completion_length": 64,
        "block_length": 32,
        "generation_batch_size": 1,
        "per_device_train_batch_size": 1,
        "reward_type": "multiplicative_budget",
        "reward_gsm_extractor": "robust",
        "enable_verifier": True,
        "verifier_type": "math",
        "verifier_schedule": "final_only",
        "enable_remasking": False,
        "hard_stop_at_target_budget": True,
    }


def _base_row(**overrides):
    row = {
        "mode": "train",
        "trainer_global_step": 0,
        "group_id": 0,
        "rollout_index": 0,
        "num_generations": 1,
        "target_budget": 32,
        "nfe_used": 32,
        "actual_nfe": 32,
        "reward": 0.0,
        "generated_text_final": "The answer is 100",
        "reference_answer": "100",
        "robust_correct": True,
        "answer_span_correct": True,
        "strict_correct": False,
        "reward_exact_match_used": True,
        "parse_ok": True,
        "format_ok": False,
        "arithmetic_ok": None,
        "remask_count": 0,
        "verifier_calls": 1,
        "has_boxed_answer": False,
        "has_complete_answer_tag": False,
        "masked_token_fraction": 0.0,
        "visible_generated_char_count": 17,
        "stop_reason": "target_budget_exhausted",
        "budget_is_hard_cap": True,
        "reward_answer_source": "answer_marker",
        "reward_quality_mode": "none",
        "rollout_compute_style": "normal",
        "final_task_score": 0.0,
        "early_task_score": 0.0,
        "deliberation_gain": 0.0,
        "answer_quality": 0.0,
        "format_factor": 1.0,
        "repetition_penalty": 1.0,
        "reward_before_quality_caps": 0.0,
        "reward_after_quality_caps": 0.0,
        "malformed_reward_cap_applied": False,
        "repetitive_reward_cap_applied": False,
        "patient_cap_applied": False,
        "delayed_low_confidence_count": 0,
        "delayed_final_window_count": 0,
        "patient_selected_before_delay": 0.0,
        "patient_selected_after_delay": 0.0,
        "max_completion_length": 64,
        "generation_batch_size": 1,
        "per_device_train_batch_size": 1,
    }
    row.update(overrides)
    return row


def test_consistency_checker_flags_robust_correct_but_reward_zero():
    result = analyze_training_consistency(
        config=_base_config(),
        rows=[_base_row()],
        min_examples=1,
    )

    assert any(
        "robust_correct_but_reward_zero" in item
        for item in result["inconsistencies"]
    )


def test_consistency_checker_flags_all_zero_reward():
    result = analyze_training_consistency(
        config=_base_config(),
        rows=[
            _base_row(
                robust_correct=False,
                answer_span_correct=False,
                reward_exact_match_used=False,
            )
        ],
        min_examples=1,
    )

    assert any(
        "No positive rewards observed" in item
        for item in result["inconsistencies"]
    )


def test_consistency_checker_flags_group_budget_mismatch():
    config = _base_config()
    config["num_generations"] = 2
    rows = [
        _base_row(num_generations=2, rollout_index=0, target_budget=16),
        _base_row(num_generations=2, rollout_index=1, target_budget=32),
    ]

    result = analyze_training_consistency(
        config=config,
        rows=rows,
        min_examples=1,
    )

    assert any(
        "target_budget varies within GRPO group" in item
        for item in result["inconsistencies"]
    )


def test_consistency_checker_ignores_eval_rows_for_group_budget_mismatch():
    config = _base_config()
    config["num_generations"] = 2
    rows = [
        _base_row(num_generations=2, rollout_index=0, target_budget=16),
        _base_row(num_generations=2, rollout_index=1, target_budget=16),
        _base_row(
            mode="eval",
            num_generations=2,
            rollout_index=0,
            target_budget=16,
            reward=1.0,
        ),
        _base_row(
            mode="eval",
            num_generations=2,
            rollout_index=1,
            target_budget=32,
            reward=1.0,
        ),
    ]

    result = analyze_training_consistency(
        config=config,
        rows=rows,
        min_examples=1,
    )

    assert not any(
        "target_budget varies within GRPO group" in item
        for item in result["inconsistencies"]
    )


def test_consistency_checker_warns_about_hard_cap_reward_recovery_risk():
    result = analyze_training_consistency(
        config=_base_config(),
        rows=[
            _base_row(
                robust_correct=False,
                answer_span_correct=False,
                reward_exact_match_used=False,
            )
        ],
        min_examples=1,
    )

    assert any(
        "hard_stop_at_target_budget=true may prevent generation" in item
        for item in result["warnings"]
    )
    assert result["summary"]["hard_stop_at_target_budget"] is True
    assert result["summary"]["hard_generation_budget"] is None
    assert result["summary"]["budget_is_hard_cap_rate"] == 1.0
    assert result["summary"]["stop_reason_distribution"] == {
        "target_budget_exhausted": 1
    }
    assert result["summary"]["positive_reward_rate_by_target_budget"] == {32: 0.0}
    assert result["summary"]["masked_token_fraction_mean_by_target_budget"] == {
        32: 0.0
    }


def test_consistency_checker_flags_robust_only_positive_in_answer_span_mode():
    config = _base_config()
    config["reward_gsm_extractor"] = "answer_span"
    row = _base_row(
        reward=0.4,
        robust_correct=True,
        answer_span_correct=False,
        reward_exact_match_used=False,
        reward_answer_source="robust_anywhere",
        generated_text_final="reasoning has 1 ... <answer>\n</answer>",
    )

    result = analyze_training_consistency(
        config=config,
        rows=[row],
        min_examples=1,
    )

    assert any(
        "robust_only_positive_reward_count should be 0" in item
        for item in result["inconsistencies"]
    )
    assert result["summary"]["answer_span_correct_rate"] == 0.0
    assert result["summary"]["robust_only_correct_rate"] == 1.0
    assert result["summary"]["robust_only_positive_reward_count"] == 1
    assert result["summary"]["empty_answer_tag_count"] == 1
    assert result["summary"]["malformed_answer_positive_count"] == 1


def test_consistency_checker_reports_patient_pair_metrics_and_warning():
    config = _base_config()
    config["num_generations"] = 2
    rows = [
        _base_row(
            num_generations=2,
            rollout_index=0,
            rollout_compute_style="normal",
            actual_nfe=18,
            nfe_used=18,
            reward=0.2,
            final_task_score=0.2,
        ),
        _base_row(
            num_generations=2,
            rollout_index=1,
            rollout_compute_style="patient",
            actual_nfe=17,
            nfe_used=17,
            reward=0.2,
            final_task_score=0.2,
        ),
    ]

    result = analyze_training_consistency(
        config=config,
        rows=rows,
        min_examples=1,
    )

    assert result["summary"]["avg_nfe_by_rollout_compute_style"] == {
        "normal": 18.0,
        "patient": 17.0,
    }
    assert result["summary"]["patient_better_than_normal_count"] == 0
    assert any(
        "patient exploration is not slower" in item for item in result["warnings"]
    )


def test_consistency_checker_reports_patient_better_count():
    config = _base_config()
    config["num_generations"] = 2
    rows = [
        _base_row(
            num_generations=2,
            rollout_index=0,
            rollout_compute_style="normal",
            actual_nfe=12,
            nfe_used=12,
            reward=0.0,
            final_task_score=0.0,
        ),
        _base_row(
            num_generations=2,
            rollout_index=1,
            rollout_compute_style="patient",
            actual_nfe=20,
            nfe_used=20,
            reward=0.4,
            final_task_score=0.4,
        ),
    ]

    result = analyze_training_consistency(
        config=config,
        rows=rows,
        min_examples=1,
    )

    assert result["summary"]["patient_better_than_normal_count"] == 1
    assert result["summary"]["patient_correct_normal_wrong_count"] == 1
    assert result["summary"]["normal_correct_patient_wrong_count"] == 0


def test_consistency_checker_reports_quality_counts():
    config = _base_config()
    config["reward_quality_mode"] = "format_weighted"
    row = _base_row(
        reward=0.75,
        reward_quality_mode="format_weighted",
        format_ok=False,
        reward_answer_source="incomplete_answer_tag",
        repetition_penalty=0.5,
        repeated_char_run_detected=True,
        final_task_score=0.75,
    )

    result = analyze_training_consistency(
        config=config,
        rows=[row],
        min_examples=1,
    )

    assert result["summary"]["high_reward_malformed_count"] == 1
    assert result["summary"]["high_reward_repetitive_count"] == 1
    assert result["summary"]["positive_reward_source_distribution"] == {
        "incomplete_answer_tag": 1
    }
    assert any("quality weighting is too weak" in item for item in result["warnings"])


def test_consistency_checker_reports_reward_cap_counts_and_summaries():
    row = _base_row(
        reward=0.3,
        reward_before_quality_caps=0.8,
        reward_after_quality_caps=0.3,
        malformed_reward_cap_applied=True,
        repetitive_reward_cap_applied=True,
        repetition_penalty=0.5,
        repeated_char_run_detected=True,
    )

    result = analyze_training_consistency(
        config=_base_config(),
        rows=[row],
        min_examples=1,
    )

    assert result["summary"]["malformed_reward_cap_applied_count"] == 1
    assert result["summary"]["repetitive_reward_cap_applied_count"] == 1
    assert result["summary"]["reward_before_quality_caps_mean"] == 0.8
    assert result["summary"]["reward_before_quality_caps_max"] == 0.8
    assert result["summary"]["reward_after_quality_caps_mean"] == 0.3
    assert result["summary"]["reward_after_quality_caps_max"] == 0.3


def test_consistency_checker_reports_selective_patient_delay_metrics():
    rows = [
        _base_row(
            rollout_compute_style="normal",
            actual_nfe=20,
            reward=0.2,
            masked_token_fraction=0.0,
            stop_reason="all_unmasked",
        ),
        _base_row(
            rollout_compute_style="patient",
            actual_nfe=24,
            reward=0.3,
            masked_token_fraction=0.05,
            stop_reason="all_unmasked",
            delayed_low_confidence_count=3,
            delayed_final_window_count=2,
            patient_selected_before_delay=5.0,
            patient_selected_after_delay=3.0,
        ),
    ]

    result = analyze_training_consistency(
        config=_base_config(),
        rows=rows,
        min_examples=1,
    )

    assert result["summary"]["avg_delayed_low_confidence_count"] == 1.5
    assert result["summary"]["avg_delayed_final_window_count"] == 1.0
    assert result["summary"]["masked_token_fraction_by_rollout_compute_style"] == {
        "normal": 0.0,
        "patient": 0.05,
    }
    assert result["summary"]["stop_reason_by_rollout_compute_style"] == {
        "normal": {"all_unmasked": 1},
        "patient": {"all_unmasked": 1},
    }
    assert result["summary"]["avg_patient_selected_before_delay"] == 2.5
    assert result["summary"]["avg_patient_selected_after_delay"] == 1.5


def test_consistency_checker_reports_continuation_metrics_from_fake_rows():
    config = _base_config()
    config["enable_posthoc_continuation"] = True
    rows = [
        _base_row(
            reward=0.2,
            base_correct=False,
            base_task_score=0.0,
            base_quality_score=0.0,
            best_continuation_correct=True,
            best_continuation_gain=0.45,
            best_continuation_task_score=0.45,
            continuation_triggered=True,
            continuation_remasked_token_count=3,
            continuation_results=[
                {
                    "continuation_k": 4,
                    "continuation_gain": 0.45,
                    "continuation_correct_gain": True,
                    "continuation_harm": False,
                    "continuation_quality_gain": 0.45,
                    "continued_extra_steps": 4,
                    "continued_correct": True,
                    "continued_task_score": 0.45,
                    "continued_quality_score": 0.45,
                    "continued_format_ok": True,
                    "continuation_trigger_reason": "format_bad",
                    "continuation_remask_mode": "answer_span",
                    "continuation_over_budget": False,
                    "continuation_remasked_token_count": 3,
                }
            ],
        ),
        _base_row(
            reward=0.8,
            base_correct=True,
            base_task_score=1.0,
            base_quality_score=1.0,
            best_continuation_correct=False,
            best_continuation_gain=-1.0,
            best_continuation_task_score=0.0,
            continuation_triggered=True,
            continuation_remasked_token_count=2,
            continuation_results=[
                {
                    "continuation_k": 8,
                    "continuation_gain": -1.0,
                    "continuation_correct_gain": False,
                    "continuation_harm": True,
                    "continuation_quality_gain": -1.0,
                    "continued_extra_steps": 8,
                    "continued_correct": False,
                    "continued_task_score": 0.0,
                    "continued_quality_score": 0.0,
                    "continued_format_ok": False,
                    "continuation_trigger_reason": "parse_bad",
                    "continuation_remask_mode": "answer_span",
                    "continuation_over_budget": True,
                    "continuation_remasked_token_count": 2,
                }
            ],
        ),
    ]

    result = analyze_training_consistency(
        config=config,
        rows=rows,
        min_examples=1,
    )

    summary = result["summary"]
    assert summary["continuation_trigger_rate"] == 1.0
    assert summary["continuation_candidate_count"] == 2
    assert summary["continuation_gain_rate"] == 0.5
    assert summary["continued_correct_base_wrong_count"] == 1
    assert summary["base_correct_continued_wrong_count"] == 1
    assert summary["continuation_harm_count"] == 1
    assert summary["net_fix_minus_harm"] == 0
    assert summary["avg_extra_steps"] == 6.0
    assert summary["continuation_gain_by_k"] == {4: 0.45, 8: -1.0}
    assert (
        abs(summary["continuation_gain_by_remask_mode"]["answer_span"] + 0.275)
        < 1e-12
    )
    assert abs(summary["continuation_gain_by_budget"][32] + 0.275) < 1e-12
    assert summary["base_accuracy"] == 0.5
    assert summary["continued_accuracy"] == 0.5
    assert summary["best_continuation_accuracy"] == 0.5
    assert summary["oracle_continue_accuracy"] == 1.0
    assert summary["oracle_best_accuracy"] == 1.0
    assert summary["oracle_possible_gain"] == 0.5
    assert summary["always_continue_gain"] == 0.0
    assert summary["continuation_over_budget_rate"] == 0.5
    assert summary["continuation_zero_remask_rate"] == 0.0


def test_consistency_checker_oracle_falls_back_to_base_when_continuation_harms():
    config = _base_config()
    config["enable_posthoc_continuation"] = True
    rows = [
        _base_row(
            base_correct=True,
            reward=1.0,
            continuation_results=[
                {
                    "continuation_k": 4,
                    "continued_correct": False,
                    "continued_task_score": 0.0,
                    "continuation_gain": -1.0,
                    "continuation_harm": True,
                    "continuation_correct_gain": False,
                }
            ],
        ),
        _base_row(
            base_correct=False,
            reward=0.0,
            continuation_results=[
                {
                    "continuation_k": 4,
                    "continued_correct": True,
                    "continued_task_score": 1.0,
                    "continuation_gain": 1.0,
                    "continuation_harm": False,
                    "continuation_correct_gain": True,
                }
            ],
        ),
        _base_row(
            base_correct=False,
            reward=0.0,
            continuation_results=[
                {
                    "continuation_k": 4,
                    "continued_correct": False,
                    "continued_task_score": 0.0,
                    "continuation_gain": 0.0,
                    "continuation_harm": False,
                    "continuation_correct_gain": False,
                }
            ],
        ),
    ]

    result = analyze_training_consistency(
        config=config,
        rows=rows,
        min_examples=1,
    )

    summary = result["summary"]
    assert summary["base_accuracy"] == 1 / 3
    assert summary["continued_accuracy"] == 1 / 3
    assert summary["best_continuation_accuracy"] == 1 / 3
    assert summary["oracle_continue_accuracy"] == 2 / 3
    assert summary["oracle_best_accuracy"] == 2 / 3
    assert summary["continuation_harm_count"] == 1
    assert summary["continuation_fix_count"] == 1


def test_consistency_checker_warns_when_requested_checkpoint_not_loaded():
    config = _base_config()
    config["policy_checkpoint_path"] = "missing/checkpoint-best"
    row = _base_row(
        policy_checkpoint_path_requested="missing/checkpoint-best",
        policy_checkpoint_loaded=False,
    )

    result = analyze_training_consistency(
        config=config,
        rows=[row],
        min_examples=1,
    )

    assert result["summary"]["policy_checkpoint_path_requested"] == (
        "missing/checkpoint-best"
    )
    assert result["summary"]["policy_checkpoint_loaded"] is False
    assert any(
        "policy_checkpoint_loaded is false" in item
        for item in result["warnings"]
    )


def test_consistency_checker_warns_when_draft_generation_is_incomplete():
    config = _base_config()
    config["hard_stop_at_target_budget"] = False
    config["hard_generation_budget"] = 48
    row = _base_row(
        nfe_used=46,
        actual_nfe=46,
        masked_token_count=8,
        masked_token_fraction=0.07,
        stop_reason="generation_budget_exhausted",
    )

    result = analyze_training_consistency(
        config=config,
        rows=[row],
        min_examples=1,
    )

    assert result["summary"]["avg_nfe"] == 46.0
    assert result["summary"]["still_masked_sample_rate"] == 1.0
    assert any(
        "Draft generation incomplete; do not run continuation." in item
        for item in result["warnings"]
    )
    assert any(
        "Many samples still masked; draft policy too conservative." in item
        for item in result["warnings"]
    )
    assert any(
        "Generation is hitting cap without finishing." in item
        for item in result["warnings"]
    )
