import torch

from common.budgeting import assign_rollout_compute_styles
from common.budgeting import sample_group_target_budgets
from common.generation.generation import apply_patient_unmask_controls
from common.generation.generation import continuation_metric_fields
from common.generation.generation import continuation_trigger_reasons
from common.generation.generation import select_continuation_remask_indices


def test_sample_group_budgets_repeats_within_group():
    torch.manual_seed(0)
    budgets = sample_group_target_budgets(
        expanded_batch_size=8,
        num_generations=2,
        budget_choices=[16, 32, 64],
        device="cpu",
    )

    assert budgets.shape[0] == 8
    assert budgets[0] == budgets[1]
    assert budgets[2] == budgets[3]
    assert budgets[4] == budgets[5]
    assert budgets[6] == budgets[7]


def test_sample_group_budgets_falls_back_for_ambiguous_layout():
    budgets = sample_group_target_budgets(
        expanded_batch_size=5,
        num_generations=2,
        budget_choices=[16],
        device="cpu",
    )

    assert budgets.tolist() == [16, 16, 16, 16, 16]


def test_assign_rollout_compute_styles_pairs_normal_and_patient():
    styles = assign_rollout_compute_styles(
        expanded_batch_size=6,
        num_generations=2,
        styles=["normal", "patient"],
    )

    assert styles == [
        "normal",
        "patient",
        "normal",
        "patient",
        "normal",
        "patient",
    ]


def test_patient_unmask_guard_caps_fake_masked_sequence():
    samples = torch.tensor([[True, True, True, True, True, True]])
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    sampling_mask = torch.ones_like(samples)
    full_mask_index = torch.ones_like(samples)
    adjusted, stats = apply_patient_unmask_controls(
        samples,
        sampling_logits=logits,
        sampling_mask=sampling_mask,
        full_mask_index=full_mask_index,
        steps_taken=torch.tensor([0]),
        target_budget_values=torch.tensor([10]),
        rollout_compute_style=["patient"],
        patient_max_unmask_fraction=0.5,
        patient_min_steps_frac=0.7,
    )

    assert adjusted.sum().item() < samples.sum().item()
    assert adjusted.sum().item() == 1
    assert adjusted[0, -1].item() is True
    assert stats[0]["patient_cap_applied"] is True
    assert stats[0]["selected_unmask_count_before_cap"] == 6
    assert stats[0]["selected_unmask_count_after_cap"] == 1


def test_selective_patient_delay_does_not_delay_all_tokens():
    samples = torch.ones((1, 10), dtype=torch.bool)
    confidence = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    adjusted, stats = apply_patient_unmask_controls(
        samples,
        sampling_logits=confidence,
        sampling_mask=torch.ones_like(samples),
        full_mask_index=torch.ones_like(samples),
        confidence=confidence,
        steps_taken=torch.tensor([0]),
        target_budget_values=torch.tensor([20]),
        rollout_compute_style=["patient"],
        patient_global_slowdown=False,
        patient_delay_low_confidence=True,
        patient_low_confidence_quantile=0.3,
    )

    assert adjusted.sum().item() == 7
    assert stats[0]["delayed_low_confidence_count"] == 3
    assert stats[0]["patient_cap_applied"] is False


def test_low_confidence_delay_delays_only_configured_quantile():
    samples = torch.ones((1, 5), dtype=torch.bool)
    confidence = torch.tensor([[0.05, 0.10, 0.90, 0.80, 0.70]])
    adjusted, stats = apply_patient_unmask_controls(
        samples,
        sampling_logits=confidence,
        sampling_mask=torch.ones_like(samples),
        full_mask_index=torch.ones_like(samples),
        confidence=confidence,
        steps_taken=torch.tensor([0]),
        target_budget_values=torch.tensor([20]),
        rollout_compute_style=["patient"],
        patient_global_slowdown=False,
        patient_delay_low_confidence=True,
        patient_low_confidence_quantile=0.4,
    )

    assert adjusted.tolist() == [[False, False, True, True, True]]
    assert stats[0]["delayed_low_confidence_count"] == 2


def test_final_window_delay_blocks_before_min_step_and_allows_after():
    samples = torch.ones((1, 10), dtype=torch.bool)
    kwargs = {
        "sampling_logits": torch.arange(10, dtype=torch.float32).view(1, 10),
        "sampling_mask": torch.ones_like(samples),
        "full_mask_index": torch.ones_like(samples),
        "target_budget_values": torch.tensor([10]),
        "rollout_compute_style": ["patient"],
        "patient_global_slowdown": False,
        "patient_delay_final_window": True,
        "patient_final_window_tokens": 3,
        "patient_final_window_min_steps_frac": 0.6,
    }

    before, before_stats = apply_patient_unmask_controls(
        samples,
        steps_taken=torch.tensor([0]),
        **kwargs,
    )
    after, after_stats = apply_patient_unmask_controls(
        samples,
        steps_taken=torch.tensor([6]),
        **kwargs,
    )

    assert before.tolist() == [[True, True, True, True, True, True, True, False, False, False]]
    assert before_stats[0]["delayed_final_window_count"] == 3
    assert before_stats[0]["final_window_min_step"] == 6
    assert after.tolist() == [[True] * 10]
    assert after_stats[0]["delayed_final_window_count"] == 0


def test_selective_patient_keeps_non_final_non_low_confidence_tokens():
    samples = torch.ones((1, 8), dtype=torch.bool)
    confidence = torch.tensor([[0.1, 0.2, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]])
    adjusted, stats = apply_patient_unmask_controls(
        samples,
        sampling_logits=confidence,
        sampling_mask=torch.ones_like(samples),
        full_mask_index=torch.ones_like(samples),
        confidence=confidence,
        steps_taken=torch.tensor([0]),
        target_budget_values=torch.tensor([10]),
        rollout_compute_style=["patient"],
        patient_global_slowdown=False,
        patient_delay_low_confidence=True,
        patient_low_confidence_quantile=0.25,
        patient_delay_final_window=True,
        patient_final_window_tokens=2,
        patient_final_window_min_steps_frac=0.6,
    )

    assert adjusted.tolist() == [[False, False, True, True, True, True, False, False]]
    assert stats[0]["delayed_low_confidence_count"] == 2
    assert stats[0]["delayed_final_window_count"] == 2


def test_continuation_heuristic_trigger_fires_on_bad_verifier_fields():
    reasons = continuation_trigger_reasons(
        trigger="heuristic",
        verifier_result={
            "format_ok": False,
            "parse_ok": False,
            "arithmetic_ok": False,
            "verifier_score": 0.2,
        },
        answer_span={"token_indices": [1, 2], "mean_confidence": 0.2},
        reward_answer_source="final_window",
        repetition_penalty=0.5,
        confidence_threshold=0.5,
        verifier_threshold=0.7,
    )

    assert "format_bad" in reasons
    assert "parse_bad" in reasons
    assert "arithmetic_bad" in reasons
    assert "source_final_window" in reasons
    assert "answer_span_low_confidence" in reasons
    assert "repetitive" in reasons
    assert "verifier_low" in reasons


def test_continuation_answer_span_remask_uses_generated_local_indices_only():
    generated_tokens = torch.tensor([10, 11, 12, 13, 14])
    indices = select_continuation_remask_indices(
        mode="answer_span",
        generated_tokens=generated_tokens,
        answer_span_indices=[-3, 1, 2, 100],
        mask_id=-1,
        special_token_ids={99},
    )

    assert indices == [1, 2]
    assert all(0 <= idx < generated_tokens.numel() for idx in indices)


def test_continuation_final_window_remasks_only_generated_tokens_before_eos():
    generated_tokens = torch.tensor([10, 11, 12, 13, 14, 15, 99, 16])
    indices = select_continuation_remask_indices(
        mode="final_window",
        generated_tokens=generated_tokens,
        mask_id=-1,
        special_token_ids={99},
        final_window_tokens=3,
    )

    assert indices == [3, 4, 5]


def test_continuation_low_confidence_window_selects_lowest_confidence_fraction():
    generated_tokens = torch.tensor([10, 11, 12, 13])
    confidences = torch.tensor([0.9, 0.1, 0.7, 0.2])
    indices = select_continuation_remask_indices(
        mode="low_confidence_answer_window",
        generated_tokens=generated_tokens,
        answer_span_indices=[0, 1, 2, 3],
        confidences=confidences,
        mask_id=-1,
        special_token_ids={99},
        low_confidence_quantile=0.5,
    )

    assert indices == [1, 3]


def test_continuation_metric_fields_compute_gain_and_harm():
    gain = continuation_metric_fields(
        base_task_score=0.0,
        continued_task_score=0.45,
        base_correct=False,
        continued_correct=True,
        base_quality_score=0.0,
        continued_quality_score=0.45,
    )
    harm = continuation_metric_fields(
        base_task_score=1.0,
        continued_task_score=0.0,
        base_correct=True,
        continued_correct=False,
        base_quality_score=1.0,
        continued_quality_score=0.0,
    )

    assert gain["continuation_gain"] == 0.45
    assert gain["continuation_correct_gain"] is True
    assert gain["continuation_harm"] is False
    assert harm["continuation_gain"] == -1.0
    assert harm["continuation_correct_gain"] is False
    assert harm["continuation_harm"] is True
