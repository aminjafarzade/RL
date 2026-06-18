import math
from types import SimpleNamespace

import pytest
import torch

from common.rewards import additive_proposal_reward
from common.rewards import apply_reward_quality_caps
from common.rewards import compute_budget_reward
from common.rewards import compute_gsm_reward_parse_fields
from common.rewards import compute_reward_quality
from common.rewards import deliberation_bonus
from common.rewards import multiplicative_budget_reward
from common.rewards import patient_win_bonus_applies
from common.rewards import select_reward_functions_from_config
from common.rewards import validate_reward_type_matches_functions
from train.reward_func import mixed_multiplicative_budget_reward_func
import train.reward_func as reward_func_module


def test_additive_proposal_reward_normalized_formula():
    reward = additive_proposal_reward(
        task_score=1.0,
        nfe_used=32,
        max_steps=128,
        lambda_compute=0.2,
        normalize_compute=True,
    )
    assert reward == pytest.approx(1.0 - 0.2 * (32 / 128))


def test_additive_proposal_reward_raw_formula():
    reward = additive_proposal_reward(
        task_score=1.0,
        nfe_used=32,
        max_steps=128,
        lambda_compute=0.01,
        normalize_compute=False,
    )
    assert reward == pytest.approx(1.0 - 0.01 * 32)


def test_additive_proposal_wrong_fast_is_less_penalized_than_wrong_slow():
    wrong_fast = additive_proposal_reward(0.0, 8, 128, 0.2, True)
    wrong_slow = additive_proposal_reward(0.0, 64, 128, 0.2, True)
    assert wrong_fast > wrong_slow


def test_multiplicative_budget_wrong_answers_get_zero_reward():
    assert multiplicative_budget_reward(False, 8, 16, 128) == 0.0
    assert multiplicative_budget_reward(False, 64, 16, 128) == 0.0


def test_multiplicative_budget_penalizes_over_target_budget():
    within = multiplicative_budget_reward(True, 8, 16, 128)
    over = multiplicative_budget_reward(True, 64, 16, 128)
    assert within > over


def test_multiplicative_budget_penalizes_remasking():
    no_remask = multiplicative_budget_reward(True, 16, 16, 128, remask_count=0)
    two_remasks = multiplicative_budget_reward(True, 16, 16, 128, remask_count=2)
    assert no_remask > two_remasks


def test_multiplicative_budget_larger_target_reduces_budget_penalty():
    small_budget = multiplicative_budget_reward(True, 64, 16, 128)
    large_budget = multiplicative_budget_reward(True, 64, 64, 128)
    assert large_budget > small_budget


def test_multiplicative_budget_is_finite_for_edge_cases():
    reward = multiplicative_budget_reward(True, 0, 1, 1)
    assert math.isfinite(reward)


def test_config_selects_additive_proposal_reward_variant():
    selection = select_reward_functions_from_config(
        SimpleNamespace(
            reward_type="additive_proposal",
            reward_gsm_extractor="strict",
            reward_lambda_compute=0.2,
            reward_normalize_compute=True,
        )
    )
    assert selection.reward_function_names == ["mixed_additive_proposal_reward_func"]
    assert selection.parameters["reward_lambda_compute"] == 0.2
    assert selection.parameters["reward_gsm_extractor"] == "strict"


def test_config_selects_multiplicative_budget_reward_variant():
    selection = select_reward_functions_from_config(
        SimpleNamespace(
            reward_type="multiplicative_budget",
            reward_gsm_extractor="robust",
            reward_beta=2.0,
            reward_mu=0.5,
            reward_rho=0.1,
            reward_quality_mode="source_aware_anti_junk",
            source_incomplete_answer_tag_credit=0.03,
            enable_negative_junk_penalty=True,
            repeated_junk_negative_reward=0.02,
        )
    )
    assert selection.reward_function_names == [
        "mixed_multiplicative_budget_reward_func"
    ]
    assert selection.parameters["reward_beta"] == 2.0
    assert selection.parameters["reward_gsm_extractor"] == "robust"
    assert selection.parameters["reward_quality_mode"] == "source_aware_anti_junk"
    assert selection.parameters["source_incomplete_answer_tag_credit"] == 0.03
    assert selection.parameters["enable_negative_junk_penalty"] is True
    assert selection.parameters["repeated_junk_negative_reward"] == 0.02


def test_config_selects_task_only_reward_variant():
    selection = select_reward_functions_from_config(
        SimpleNamespace(
            reward_type="task_only",
            reward_gsm_extractor="answer_span",
            reward_quality_mode="source_aware_anti_junk",
            reward_budget_mode="cap",
            source_answer_marker_credit=0.35,
        )
    )
    assert selection.reward_function_names == [
        "mixed_multiplicative_budget_reward_func"
    ]
    assert selection.parameters["reward_budget_mode"] == "none"
    assert selection.parameters["reward_beta"] == 0.0
    assert selection.parameters["source_answer_marker_credit"] == 0.35
    assert "max_reward_if_malformed" not in selection.parameters
    assert "max_reward_if_repetitive" not in selection.parameters


def test_config_selection_does_not_expose_quality_cap_kwargs():
    for reward_type in (
        "additive_proposal",
        "multiplicative_budget",
        "task_only",
    ):
        selection = select_reward_functions_from_config(
            SimpleNamespace(
                reward_type=reward_type,
                max_reward_if_malformed=0.3,
                max_reward_if_repetitive=0.05,
            )
        )

        assert "max_reward_if_malformed" not in selection.parameters
        assert "max_reward_if_repetitive" not in selection.parameters


def test_reward_type_validation_accepts_matching_multiplicative_function():
    validate_reward_type_matches_functions(
        SimpleNamespace(
            reward_type="multiplicative_budget",
            reward_functions=["mixed_multiplicative_budget_reward_func"],
        )
    )


def test_reward_type_validation_rejects_mismatched_function():
    with pytest.raises(ValueError, match="reward_type=multiplicative_budget expects"):
        validate_reward_type_matches_functions(
            SimpleNamespace(
                reward_type="multiplicative_budget",
                reward_functions=["mixed_additive_proposal_reward_func"],
            )
        )


def test_reward_type_validation_accepts_matching_additive_function():
    validate_reward_type_matches_functions(
        SimpleNamespace(
            reward_type="additive_proposal",
            reward_functions=["mixed_additive_proposal_reward_func"],
        )
    )


def test_reward_type_validation_accepts_task_only_with_budget_reward_function():
    validate_reward_type_matches_functions(
        SimpleNamespace(
            reward_type="task_only",
            reward_functions=["mixed_multiplicative_budget_reward_func"],
        )
    )


def test_quality_weighting_strict_clean_answer_gets_full_task_score():
    quality = compute_reward_quality(
        "<answer>350</answer>",
        "350",
        extractor_mode="answer_span",
        reward_quality_mode="format_weighted",
    )

    assert quality.task_score == pytest.approx(1.0)
    assert quality.answer_quality == pytest.approx(1.0)
    assert quality.format_factor == pytest.approx(1.0)


def test_quality_weighting_strict_correct_overrides_false_format_ok():
    quality = compute_reward_quality(
        "<answer>350</answer>",
        "350",
        extractor_mode="answer_span",
        reward_quality_mode="format_weighted",
        format_ok=False,
    )

    assert quality.task_score == pytest.approx(1.0)
    assert quality.answer_quality == pytest.approx(1.0)
    assert quality.format_factor == pytest.approx(1.0)


def test_quality_weighting_answer_span_malformed_gets_partial_task_score():
    quality = compute_reward_quality(
        "Reasoning.\nTherefore the answer is 350.",
        "350",
        extractor_mode="answer_span",
        reward_quality_mode="format_weighted",
        answer_span_partial_credit=0.4,
        malformed_answer_factor=0.5,
    )

    assert quality.task_score == pytest.approx(0.2)
    assert quality.answer_quality == pytest.approx(0.4)
    assert quality.format_factor == pytest.approx(0.5)


def test_quality_weighting_repeated_junk_gets_lower_task_score():
    repeated = " ".join(["hulaula"] * 40 + ["Therefore the answer is 350."])
    clean = "Reasoning.\nTherefore the answer is 350."

    repeated_quality = compute_reward_quality(
        repeated,
        "350",
        extractor_mode="answer_span",
        reward_quality_mode="format_weighted",
        enable_repetition_penalty=True,
        repetition_unique_ratio_threshold=0.25,
        repetition_max_freq_threshold=0.25,
        repetition_penalty_factor=0.5,
    )
    clean_quality = compute_reward_quality(
        clean,
        "350",
        extractor_mode="answer_span",
        reward_quality_mode="format_weighted",
        enable_repetition_penalty=True,
        repetition_unique_ratio_threshold=0.25,
        repetition_max_freq_threshold=0.25,
        repetition_penalty_factor=0.5,
    )

    assert repeated_quality.task_score < clean_quality.task_score
    assert repeated_quality.repetition_penalty < 1.0


def test_quality_weighting_wrong_answer_gets_zero():
    quality = compute_reward_quality(
        "<answer>351</answer>",
        "350",
        extractor_mode="answer_span",
        reward_quality_mode="format_weighted",
    )

    assert quality.task_score == 0.0


def test_quality_mode_none_preserves_selected_extractor_behavior():
    quality = compute_reward_quality(
        "Reasoning.\nTherefore the answer is 350.",
        "350",
        extractor_mode="answer_span",
        reward_quality_mode="none",
    )

    assert quality.task_score == pytest.approx(1.0)


def test_source_aware_clean_strict_answer_gets_full_task_score():
    quality = compute_reward_quality(
        "<answer>42</answer>",
        "42",
        extractor_mode="answer_span",
        reward_quality_mode="source_aware_anti_junk",
    )

    assert quality.task_score == pytest.approx(1.0)
    assert quality.answer_quality == pytest.approx(1.0)
    assert quality.format_factor == pytest.approx(1.0)


def test_source_aware_answer_marker_gets_configured_credit():
    quality = compute_reward_quality(
        "Therefore the answer is 42.",
        "42",
        extractor_mode="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        source_answer_marker_credit=0.20,
    )

    assert quality.answer_source == "answer_marker"
    assert quality.task_score == pytest.approx(0.20)


def test_source_aware_final_window_gets_small_credit():
    quality = compute_reward_quality(
        "Reasoning without numeric details. Final result follows: 42",
        "42",
        extractor_mode="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        source_final_window_credit=0.08,
    )

    assert quality.answer_source == "final_window"
    assert quality.task_score == pytest.approx(0.08)


def test_source_aware_incomplete_answer_tag_gets_tiny_credit():
    quality = compute_reward_quality(
        "<answer> blah 42",
        "42",
        extractor_mode="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        source_incomplete_answer_tag_credit=0.03,
    )

    assert quality.answer_source == "incomplete_answer_tag"
    assert quality.task_score == pytest.approx(0.03)


def test_source_aware_repeated_numeric_junk_is_penalized():
    clean = compute_reward_quality(
        "Therefore the answer is 42.",
        "42",
        extractor_mode="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        enable_repetition_penalty=True,
        source_answer_marker_credit=0.20,
    )
    repeated = compute_reward_quality(
        "Therefore the answer is 42 42 42 42",
        "42",
        extractor_mode="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        enable_repetition_penalty=True,
        source_answer_marker_credit=0.20,
        repetition_penalty_factor=0.5,
    )

    assert repeated.repeated_number_run_detected is True
    assert repeated.task_score < clean.task_score
    assert repeated.task_score <= clean.task_score * 0.5


def test_source_aware_repeated_answer_span_can_zero_reward():
    quality = compute_reward_quality(
        "<answer>42 42 42</answer>",
        "42",
        extractor_mode="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        zero_reward_if_repeated_answer_span=True,
    )

    assert quality.repeated_answer_span_detected is True
    assert quality.task_score <= 0.02


def test_task_only_reward_strict_answer_gets_full_quality_score():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["<answer>42</answer>"],
        answer=["42"],
        n_steps=torch.tensor([96]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([16]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_budget_mode="none",
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([1.0])


def test_task_only_reward_ignores_legacy_pos_reward_scaling():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["<answer>42</answer>"],
        answer=["42"],
        n_steps=torch.tensor([128]),
        L=128,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([16]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_type="task_only",
        pos_reward=1.0 / 128.0,
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([1.0])


def test_task_only_reward_answer_marker_gets_source_credit():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["Therefore the answer is 42."],
        answer=["42"],
        n_steps=torch.tensor([96]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([16]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_budget_mode="none",
        source_answer_marker_credit=0.35,
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([0.35])


def test_task_only_reward_answer_marker_ignores_completion_length_scaling():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["Therefore the answer is 42."],
        answer=["42"],
        n_steps=torch.tensor([128]),
        L=128,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([16]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_type="task_only",
        pos_reward=1.0 / 128.0,
        source_answer_marker_credit=0.35,
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([0.35])


def test_task_only_reward_incomplete_answer_tag_gets_source_credit():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["<answer> blah 42"],
        answer=["42"],
        n_steps=torch.tensor([96]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([16]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_budget_mode="none",
        source_incomplete_answer_tag_credit=0.08,
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([0.08])


def test_task_only_reward_incomplete_answer_tag_ignores_completion_length_scaling():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["<answer> blah 42"],
        answer=["42"],
        n_steps=torch.tensor([128]),
        L=128,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([16]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_type="task_only",
        pos_reward=1.0 / 128.0,
        source_incomplete_answer_tag_credit=0.08,
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([0.08])


def test_task_only_reward_wrong_normal_answer_is_zero():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["Therefore the answer is 41."],
        answer=["42"],
        n_steps=torch.tensor([8]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([96]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_budget_mode="none",
        source_answer_marker_credit=0.35,
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([0.0])


def test_task_only_reward_is_not_changed_by_target_budget_or_nfe():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=["", ""],
        completions=["<answer>42</answer>", "<answer>42</answer>"],
        answer=["42", "42"],
        n_steps=torch.tensor([8, 96]),
        L=192,
        dataset_type=["gsm8k", "gsm8k"],
        target_budget=torch.tensor([16, 48]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_budget_mode="none",
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([1.0, 1.0])


def test_task_only_reward_same_source_is_length_invariant_without_repetition_penalty():
    short = "Therefore the answer is 42."
    long = (
        "We compare the quantities carefully, check the arithmetic, and keep the "
        "final response separate from the reasoning. Therefore the answer is 42."
    )
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=["", ""],
        completions=[short, long],
        answer=["42", "42"],
        n_steps=torch.tensor([8, 128]),
        L=128,
        dataset_type=["gsm8k", "gsm8k"],
        target_budget=torch.tensor([16, 64]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_type="task_only",
        pos_reward=1.0 / 128.0,
        source_answer_marker_credit=0.35,
        enable_repetition_penalty=False,
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([0.35, 0.35])


def test_task_only_reward_repeated_answer_span_stays_capped_low():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["<answer>42 42 42</answer>"],
        answer=["42"],
        n_steps=torch.tensor([128]),
        L=128,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([16]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_type="task_only",
        pos_reward=1.0 / 128.0,
        zero_reward_if_repeated_answer_span=True,
        max_reward_if_repetitive=0.05,
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards[0] <= 0.05


def test_task_only_reward_type_disables_budget_when_mode_is_omitted():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["<answer>42</answer>"],
        answer=["42"],
        n_steps=torch.tensor([96]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([16]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_type="task_only",
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([1.0])


def test_task_only_reward_ignores_malformed_cap_and_stays_on_source_scale():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["Therefore the answer is 42."],
        answer=["42"],
        n_steps=torch.tensor([96]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([16]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_budget_mode="none",
        source_answer_marker_credit=0.35,
        max_reward_if_malformed=0.30,
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert rewards == pytest.approx([0.35])


def _task_only_trainer_args(**overrides):
    defaults = {
        "reward_type": "task_only",
        "reward_budget_mode": "cap",
        "reward_gsm_extractor": "answer_span",
        "reward_quality_mode": "source_aware_anti_junk",
        "answer_span_partial_credit": 0.4,
        "malformed_answer_factor": 0.5,
        "source_answer_marker_credit": 0.35,
        "source_final_line_credit": 0.25,
        "source_final_window_credit": 0.15,
        "source_incomplete_answer_tag_credit": 0.08,
        "source_other_answer_span_credit": 0.10,
        "enable_repetition_penalty": False,
        "repetition_min_tokens": 8,
        "repetition_number_run_threshold": 3,
        "repetition_token_run_threshold": 4,
        "repetition_bigram_run_threshold": 3,
        "repetition_answer_span_number_threshold": 2,
        "repetition_unique_ratio_threshold": 0.35,
        "repetition_max_freq_threshold": 0.25,
        "repetition_penalty_factor": 0.25,
        "zero_reward_if_repeated_answer_span": True,
        "max_reward_if_malformed": 0.30,
        "max_reward_if_repetitive": 0.05,
        "target_budget": 16,
        "max_completion_length": 128,
        "reward_beta": 99.0,
        "reward_mu": 99.0,
        "reward_rho": 99.0,
        "threshold_reward_hard_zero": False,
        "deliberation_bonus_weight": 0.0,
        "alpha_correctness_reward": 1.0 / 128.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_trainer_task_only_logs_strict_answer_on_quality_scale():
    fields = compute_gsm_reward_parse_fields(
        "<answer>42</answer>",
        "42",
        _task_only_trainer_args(),
        {"actual_nfe": 128, "target_budget": 16, "max_steps": 128},
    )

    assert fields["quality_task_score_before_scaling"] == pytest.approx(1.0)
    assert fields["quality_task_score_after_caps"] == pytest.approx(1.0)
    assert fields["raw_task_score"] == pytest.approx(1.0)
    assert fields["final_reward"] == pytest.approx(1.0)
    assert fields["completion_length_normalizer"] == pytest.approx(1.0)
    assert fields["budget_multiplier"] == pytest.approx(1.0)
    assert fields["reward_budget_mode"] == "none"
    assert fields["reward_type/task_only"] == pytest.approx(1.0)


def test_trainer_task_only_logs_answer_marker_without_length_scaling():
    fields = compute_gsm_reward_parse_fields(
        "Therefore the answer is 42.",
        "42",
        _task_only_trainer_args(),
        {"actual_nfe": 128, "target_budget": 16, "max_steps": 128},
    )

    assert fields["quality_task_score_before_scaling"] == pytest.approx(0.35)
    assert fields["quality_task_score_after_caps"] == pytest.approx(0.35)
    assert fields["raw_task_score"] == pytest.approx(0.35)
    assert fields["final_reward"] == pytest.approx(0.35)
    assert fields["completion_length_normalizer"] == pytest.approx(1.0)
    assert fields["budget_multiplier"] == pytest.approx(1.0)


def test_trainer_task_only_logs_incomplete_tag_without_length_scaling():
    fields = compute_gsm_reward_parse_fields(
        "<answer> blah 42",
        "42",
        _task_only_trainer_args(),
        {"actual_nfe": 128, "target_budget": 16, "max_steps": 128},
    )

    assert fields["answer_source"] == "incomplete_answer_tag"
    assert fields["quality_task_score_before_scaling"] == pytest.approx(0.08)
    assert fields["quality_task_score_after_caps"] == pytest.approx(0.08)
    assert fields["raw_task_score"] == pytest.approx(0.08)
    assert fields["final_reward"] == pytest.approx(0.08)
    assert fields["completion_length_normalizer"] == pytest.approx(1.0)
    assert fields["budget_multiplier"] == pytest.approx(1.0)


def test_multiplicative_budget_reward_still_applies_pos_reward_scaling():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["<answer>42</answer>"],
        answer=["42"],
        n_steps=torch.tensor([12]),
        L=128,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([24]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_budget_mode="threshold",
        pos_reward=0.5,
        reward_beta=0.0,
        reward_mu=0.0,
        reward_rho=0.0,
    )

    assert rewards == pytest.approx([0.5])


def test_reward_func_does_not_pass_cap_kwargs_to_compute_reward_quality(monkeypatch):
    seen_kwargs = []

    def fake_compute_reward_quality(text, reference_answer, **kwargs):
        seen_kwargs.append(kwargs)
        return SimpleNamespace(
            task_score=1.0,
            parsed_answer="42",
            answer_source="strict_answer_tag",
            answer_span_source="strict_answer_tag",
            strict_correct=True,
            answer_span_correct=True,
            robust_correct=True,
            format_ok=kwargs.get("format_ok"),
            repetition_penalty=1.0,
            repeated_char_run_detected=False,
            repeated_number_run_detected=False,
            repeated_token_run_detected=False,
            repeated_ngram_detected=False,
            repeated_answer_span_detected=False,
            low_unique_ratio_detected=False,
            high_token_freq_detected=False,
            repeated_junk_detected=False,
        )

    monkeypatch.setattr(
        reward_func_module,
        "compute_reward_quality",
        fake_compute_reward_quality,
    )

    rewards = reward_func_module.mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["<answer>42</answer>"],
        answer=["42"],
        n_steps=torch.tensor([96]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([16]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_budget_mode="none",
        max_reward_if_malformed=0.30,
        max_reward_if_repetitive=0.05,
    )

    assert rewards == pytest.approx([1.0])
    assert seen_kwargs
    assert all("max_reward_if_malformed" not in item for item in seen_kwargs)
    assert all("max_reward_if_repetitive" not in item for item in seen_kwargs)


def test_budget_reward_none_returns_task_score_without_budget_penalty():
    reward = compute_budget_reward(
        task_score=0.35,
        nfe_used=96,
        target_budget=16,
        max_steps=192,
        reward_budget_mode="none",
        reward_beta=99.0,
        reward_mu=99.0,
        reward_rho=99.0,
    )

    assert reward.reward == pytest.approx(0.35)
    assert reward.budget_multiplier == pytest.approx(1.0)


def test_threshold_reward_has_no_under_budget_speed_penalty():
    under = compute_budget_reward(
        task_score=1.0,
        nfe_used=13,
        target_budget=24,
        max_steps=192,
        reward_budget_mode="threshold",
        reward_beta=4.0,
        reward_mu=99.0,
    )
    exact = compute_budget_reward(
        task_score=1.0,
        nfe_used=24,
        target_budget=24,
        max_steps=192,
        reward_budget_mode="threshold",
        reward_beta=4.0,
        reward_mu=0.0,
    )

    assert under.reward == pytest.approx(1.0)
    assert exact.reward == pytest.approx(1.0)
    assert under.threshold_gate == pytest.approx(1.0)


def test_threshold_reward_penalizes_over_budget_correct_answer():
    over = compute_budget_reward(
        task_score=1.0,
        nfe_used=36,
        target_budget=24,
        max_steps=192,
        reward_budget_mode="threshold",
        reward_beta=4.0,
    )

    assert 0.0 < over.reward < 1.0


def test_threshold_reward_wrong_answer_stays_zero():
    wrong = compute_budget_reward(
        task_score=0.0,
        nfe_used=13,
        target_budget=24,
        max_steps=192,
        reward_budget_mode="threshold",
    )

    assert wrong.reward == 0.0


def test_threshold_reward_ignores_reward_mu_by_default():
    low_mu = compute_budget_reward(
        task_score=1.0,
        nfe_used=13,
        target_budget=24,
        max_steps=192,
        reward_budget_mode="threshold",
        reward_mu=0.0,
    )
    high_mu = compute_budget_reward(
        task_score=1.0,
        nfe_used=13,
        target_budget=24,
        max_steps=192,
        reward_budget_mode="threshold",
        reward_mu=99.0,
    )

    assert high_mu.reward == pytest.approx(low_mu.reward)


def test_deliberation_bonus_positive_only_for_quality_gain():
    gain, bonus = deliberation_bonus(
        early_task_score=0.0,
        final_task_score=0.4,
        threshold_gate=1.0,
        deliberation_bonus_weight=0.1,
    )
    no_gain, no_bonus = deliberation_bonus(
        early_task_score=0.4,
        final_task_score=0.4,
        threshold_gate=1.0,
        deliberation_bonus_weight=0.1,
    )
    wrong_gain, wrong_bonus = deliberation_bonus(
        early_task_score=0.0,
        final_task_score=0.0,
        threshold_gate=1.0,
        deliberation_bonus_weight=0.1,
    )

    assert gain == pytest.approx(0.4)
    assert bonus == pytest.approx(0.04)
    assert no_gain == 0.0
    assert no_bonus == 0.0
    assert wrong_gain == 0.0
    assert wrong_bonus == 0.0


def test_patient_win_bonus_requires_patient_margin_and_budget():
    assert patient_win_bonus_applies(
        patient_task_score=0.4,
        normal_task_score=0.0,
        patient_nfe=20,
        target_budget=24,
        patient_win_margin=0.05,
    )
    assert not patient_win_bonus_applies(
        patient_task_score=0.4,
        normal_task_score=0.4,
        patient_nfe=20,
        target_budget=24,
        patient_win_margin=0.05,
    )
    assert not patient_win_bonus_applies(
        patient_task_score=0.4,
        normal_task_score=0.0,
        patient_nfe=25,
        target_budget=24,
        patient_win_margin=0.05,
    )


def test_malformed_answer_span_reward_is_capped():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["Reasoning.\nTherefore the answer is 350."],
        answer=["350"],
        n_steps=torch.tensor([12]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([24]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="format_weighted",
        answer_span_partial_credit=1.0,
        malformed_answer_factor=1.0,
        format_ok=[False],
        reward_budget_mode="threshold",
        reward_beta=4.0,
        reward_mu=0.0,
        reward_rho=0.0,
        max_reward_if_malformed=0.4,
    )

    assert rewards == pytest.approx([0.4])


def test_wrong_repeated_junk_gets_negative_reward_when_enabled():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["Therefore the answer is 41 41 41 41"],
        answer=["42"],
        n_steps=torch.tensor([12]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([24]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_budget_mode="threshold",
        reward_beta=4.0,
        reward_mu=0.0,
        reward_rho=0.0,
        enable_negative_junk_penalty=True,
        repeated_junk_negative_reward=0.02,
        malformed_junk_negative_reward=0.01,
        incomplete_answer_tag_negative_reward=0.01,
        min_reward_floor=-0.05,
    )

    assert rewards[0] < 0.0


def test_wrong_normal_answer_stays_zero_with_negative_junk_enabled():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["Therefore the answer is 41."],
        answer=["42"],
        n_steps=torch.tensor([12]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([24]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="source_aware_anti_junk",
        reward_budget_mode="threshold",
        reward_beta=4.0,
        reward_mu=0.0,
        reward_rho=0.0,
        enable_negative_junk_penalty=True,
        repeated_junk_negative_reward=0.02,
        malformed_junk_negative_reward=0.01,
        incomplete_answer_tag_negative_reward=0.01,
        min_reward_floor=-0.05,
    )

    assert rewards == pytest.approx([0.0])


def test_repetitive_answer_reward_is_capped():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["aaaaaaaa <answer>350</answer>"],
        answer=["350"],
        n_steps=torch.tensor([12]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([24]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="format_weighted",
        format_ok=[True],
        enable_repetition_penalty=True,
        repetition_penalty_factor=0.5,
        reward_budget_mode="threshold",
        reward_beta=4.0,
        reward_mu=0.0,
        reward_rho=0.0,
        max_reward_if_repetitive=0.3,
    )

    assert rewards == pytest.approx([0.3])


def test_clean_strict_answer_can_exceed_quality_caps():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["<answer>350</answer>"],
        answer=["350"],
        n_steps=torch.tensor([12]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([24]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="format_weighted",
        format_ok=[False],
        enable_repetition_penalty=True,
        reward_budget_mode="threshold",
        reward_beta=4.0,
        reward_mu=0.0,
        reward_rho=0.0,
        max_reward_if_malformed=0.4,
        max_reward_if_repetitive=0.3,
    )

    assert rewards == pytest.approx([1.0])


def test_deliberation_bonus_cannot_push_malformed_above_cap():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=[""],
        completions=["Reasoning.\nTherefore the answer is 350."],
        answer=["350"],
        n_steps=torch.tensor([12]),
        L=192,
        dataset_type=["gsm8k"],
        target_budget=torch.tensor([24]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="format_weighted",
        answer_span_partial_credit=1.0,
        malformed_answer_factor=0.5,
        format_ok=[False],
        reward_budget_mode="threshold",
        reward_beta=4.0,
        reward_mu=0.0,
        reward_rho=0.0,
        enable_deliberation_probe=True,
        early_generated_text=["No final answer yet."],
        deliberation_bonus_weight=0.1,
        max_reward_if_malformed=0.4,
    )

    assert rewards == pytest.approx([0.4])


def test_patient_win_bonus_cannot_push_repetitive_above_cap():
    rewards = mixed_multiplicative_budget_reward_func(
        prompts=["", ""],
        completions=["<answer>0</answer>", "aaaaaaaa <answer>350</answer>"],
        answer=["350", "350"],
        n_steps=torch.tensor([12, 12]),
        L=192,
        dataset_type=["gsm8k", "gsm8k"],
        target_budget=torch.tensor([24, 24]),
        reward_gsm_extractor="answer_span",
        reward_quality_mode="format_weighted",
        format_ok=[True, True],
        enable_repetition_penalty=True,
        repetition_penalty_factor=0.5,
        reward_budget_mode="threshold",
        reward_beta=4.0,
        reward_mu=0.0,
        reward_rho=0.0,
        rollout_compute_style=["normal", "patient"],
        num_generations=2,
        patient_win_bonus=0.1,
        patient_win_margin=0.05,
        max_reward_if_repetitive=0.3,
    )

    assert rewards == pytest.approx([0.0, 0.3])


def test_reward_quality_caps_report_before_and_after_values():
    cap = apply_reward_quality_caps(
        reward=0.8,
        is_malformed=True,
        repeated_junk_detected=True,
        max_reward_if_malformed=0.4,
        max_reward_if_repetitive=0.3,
    )

    assert cap.reward_before_quality_caps == pytest.approx(0.8)
    assert cap.reward_after_quality_caps == pytest.approx(0.3)
    assert cap.malformed_reward_cap_applied is True
    assert cap.repetitive_reward_cap_applied is True
