#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from common.parsing.parse_and_get_acc import extract_gsm_answer_with_source
from common.parsing.parse_and_get_acc import normalize_gsm_numeric_answer
from common.parsing.parse_and_get_acc import numeric_equal


_ANSWER_TAG_BODY_RE = re.compile(
    r"<answer>(.*?)</answer>",
    re.IGNORECASE | re.DOTALL,
)
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
_REPEATED_CHAR_RUN_RE = re.compile(r"(.)\1{7,}")

SOURCE_AWARE_ANSWER_CREDITS = {
    "strict_answer_tag": 1.0,
    "strict_boxed": 1.0,
    "answer_marker": 0.20,
    "final_line": 0.15,
    "final_window": 0.08,
    "incomplete_answer_tag": 0.03,
    "other": 0.05,
}


def additive_proposal_reward(
    task_score: float,
    nfe_used: int | float,
    max_steps: int | float | None = None,
    lambda_compute: float = 0.01,
    normalize_compute: bool = True,
) -> float:
    if normalize_compute:
        if max_steps is None or max_steps <= 0:
            raise ValueError("max_steps must be positive when normalize_compute=True")
        compute_cost = float(nfe_used) / float(max_steps)
    else:
        compute_cost = float(nfe_used)
    return float(task_score) - float(lambda_compute) * compute_cost


def multiplicative_budget_reward(
    exact_match: int | bool | float,
    nfe_used: int | float,
    target_budget: int | float,
    max_steps: int | float,
    remask_count: int | float = 0,
    beta: float = 2.0,
    mu: float = 0.5,
    rho: float = 0.1,
    eps: float = 1e-8,
) -> float:
    if float(exact_match) <= 0.0:
        return 0.0
    safe_target_budget = max(float(target_budget), eps)
    safe_max_steps = max(float(max_steps), eps)
    over_budget = max(0.0, float(nfe_used) - safe_target_budget)
    budget_penalty = math.exp(-float(beta) * over_budget / safe_target_budget)
    compute_penalty = math.exp(-float(mu) * float(nfe_used) / safe_max_steps)
    remask_penalty = math.exp(-float(rho) * float(remask_count))
    return float(exact_match) * budget_penalty * compute_penalty * remask_penalty


@dataclass(frozen=True)
class RepetitionMetrics:
    penalty: float
    unique_token_ratio: float | None
    max_token_freq_ratio: float | None
    repeated_char_run_detected: bool
    repeated_number_run_detected: bool
    repeated_token_run_detected: bool
    repeated_ngram_detected: bool
    repeated_answer_span_detected: bool
    low_unique_ratio_detected: bool
    high_token_freq_detected: bool

    @property
    def any_repetition_detected(self) -> bool:
        return any(
            (
                self.repeated_char_run_detected,
                self.repeated_number_run_detected,
                self.repeated_token_run_detected,
                self.repeated_ngram_detected,
                self.repeated_answer_span_detected,
                self.low_unique_ratio_detected,
                self.high_token_freq_detected,
            )
        )


@dataclass(frozen=True)
class RewardQualityResult:
    task_score: float
    answer_quality: float
    format_factor: float
    format_ok: bool | None
    repetition_penalty: float
    unique_token_ratio: float | None
    max_token_freq_ratio: float | None
    repeated_char_run_detected: bool
    repeated_number_run_detected: bool
    repeated_token_run_detected: bool
    repeated_ngram_detected: bool
    repeated_answer_span_detected: bool
    low_unique_ratio_detected: bool
    high_token_freq_detected: bool
    repeated_junk_detected: bool
    parsed_answer: str | None
    answer_source: str
    answer_span_source: str
    strict_correct: bool
    answer_span_correct: bool
    robust_correct: bool


@dataclass(frozen=True)
class BudgetRewardResult:
    reward: float
    threshold_gate: float
    budget_over_penalty: float
    budget_multiplier: float
    nfe_target_ratio: float


@dataclass(frozen=True)
class RewardCapResult:
    reward_before_quality_caps: float
    reward_after_quality_caps: float
    malformed_reward_cap_applied: bool
    repetitive_reward_cap_applied: bool


@dataclass(frozen=True)
class NegativeJunkPenaltyResult:
    reward_before_negative_junk_penalty: float
    reward_after_negative_junk_penalty: float
    negative_junk_penalty: float
    repeated_junk_negative_penalty_applied: bool
    malformed_junk_negative_penalty_applied: bool
    incomplete_answer_tag_negative_penalty_applied: bool


def effective_reward_budget_mode(config: Any) -> str:
    if getattr(config, "reward_type", None) == "task_only":
        return "none"
    return getattr(config, "reward_budget_mode", "cap")


def uses_task_only_quality_scale(config: Any) -> bool:
    return effective_reward_budget_mode(config) == "none"


def reward_quality_kwargs_from_config(config: Any) -> dict[str, Any]:
    return {
        "extractor_mode": getattr(config, "reward_gsm_extractor", "robust"),
        "reward_quality_mode": getattr(config, "reward_quality_mode", "none"),
        "answer_span_partial_credit": getattr(
            config,
            "answer_span_partial_credit",
            0.4,
        ),
        "malformed_answer_factor": getattr(config, "malformed_answer_factor", 0.5),
        "source_answer_marker_credit": getattr(
            config,
            "source_answer_marker_credit",
            0.20,
        ),
        "source_final_line_credit": getattr(config, "source_final_line_credit", 0.15),
        "source_final_window_credit": getattr(
            config,
            "source_final_window_credit",
            0.08,
        ),
        "source_incomplete_answer_tag_credit": getattr(
            config,
            "source_incomplete_answer_tag_credit",
            0.03,
        ),
        "source_other_answer_span_credit": getattr(
            config,
            "source_other_answer_span_credit",
            0.05,
        ),
        "enable_repetition_penalty": getattr(
            config,
            "enable_repetition_penalty",
            False,
        ),
        "repetition_min_tokens": getattr(config, "repetition_min_tokens", 8),
        "repetition_number_run_threshold": getattr(
            config,
            "repetition_number_run_threshold",
            3,
        ),
        "repetition_token_run_threshold": getattr(
            config,
            "repetition_token_run_threshold",
            4,
        ),
        "repetition_bigram_run_threshold": getattr(
            config,
            "repetition_bigram_run_threshold",
            3,
        ),
        "repetition_answer_span_number_threshold": getattr(
            config,
            "repetition_answer_span_number_threshold",
            2,
        ),
        "repetition_unique_ratio_threshold": getattr(
            config,
            "repetition_unique_ratio_threshold",
            0.25,
        ),
        "repetition_max_freq_threshold": getattr(
            config,
            "repetition_max_freq_threshold",
            0.25,
        ),
        "repetition_penalty_factor": getattr(
            config,
            "repetition_penalty_factor",
            0.5,
        ),
        "zero_reward_if_repeated_answer_span": getattr(
            config,
            "zero_reward_if_repeated_answer_span",
            True,
        ),
    }


def compute_repetition_metrics(
    text: str | None,
    *,
    enable_repetition_penalty: bool = False,
    repetition_min_tokens: int = 8,
    repetition_number_run_threshold: int = 3,
    repetition_token_run_threshold: int = 4,
    repetition_bigram_run_threshold: int = 3,
    repetition_answer_span_number_threshold: int = 2,
    repetition_unique_ratio_threshold: float = 0.25,
    repetition_max_freq_threshold: float = 0.25,
    repetition_penalty_factor: float = 0.5,
) -> RepetitionMetrics:
    raw_text = text or ""
    lowered_words = [
        word.lower()
        for word in _WORD_TOKEN_RE.findall(raw_text)
        if word.strip()
    ]
    unique_ratio = None
    max_freq_ratio = None

    normalized_numbers = [
        normalized
        for normalized in (
            normalize_gsm_numeric_answer(match.group(0))
            for match in re.finditer(
                r"[-+]?\s*\$?\s*(?:\d[\d,]*|\.\d+)(?:\.\d+)?"
                r"(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?",
                raw_text,
            )
        )
        if normalized is not None
    ]
    repeated_number_run_detected = any(
        count >= int(repetition_number_run_threshold)
        for count in Counter(normalized_numbers).values()
    )

    repeated_token_run_detected = False
    current_token = None
    current_token_count = 0
    for token in lowered_words:
        if token == current_token:
            current_token_count += 1
        else:
            current_token = token
            current_token_count = 1
        if current_token_count >= int(repetition_token_run_threshold):
            repeated_token_run_detected = True
            break

    repeated_ngram_detected = False
    if len(lowered_words) >= 2:
        bigrams = list(zip(lowered_words, lowered_words[1:]))
        repeated_ngram_detected = any(
            count >= int(repetition_bigram_run_threshold)
            for count in Counter(bigrams).values()
        )

    repeated_answer_span_detected = False
    answer_regions = [
        match.group(1)
        for match in _ANSWER_TAG_BODY_RE.finditer(raw_text)
    ]
    lower_text = raw_text.lower()
    last_open = lower_text.rfind("<answer>")
    last_close = lower_text.rfind("</answer>")
    if last_open >= 0 and last_open > last_close:
        answer_regions.append(raw_text[last_open + len("<answer>") :])
    for region in answer_regions:
        region_numbers = [
            normalized
            for normalized in (
                normalize_gsm_numeric_answer(match.group(0))
                for match in re.finditer(
                    r"[-+]?\s*\$?\s*(?:\d[\d,]*|\.\d+)(?:\.\d+)?"
                    r"(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?",
                    region,
                )
            )
            if normalized is not None
        ]
        if any(
            count >= int(repetition_answer_span_number_threshold)
            for count in Counter(region_numbers).values()
        ):
            repeated_answer_span_detected = True
            break

    low_unique_ratio_detected = False
    high_token_freq_detected = False
    if len(lowered_words) >= int(repetition_min_tokens):
        counts = Counter(lowered_words)
        unique_ratio = len(counts) / len(lowered_words)
        max_freq_ratio = max(counts.values()) / len(lowered_words)
        low_unique_ratio_detected = unique_ratio < repetition_unique_ratio_threshold
        high_token_freq_detected = max_freq_ratio > repetition_max_freq_threshold

    repeated_char_run_detected = bool(_REPEATED_CHAR_RUN_RE.search(raw_text))

    penalty = 1.0
    if enable_repetition_penalty:
        signals = (
            repeated_char_run_detected,
            repeated_number_run_detected,
            repeated_token_run_detected,
            repeated_ngram_detected,
            repeated_answer_span_detected,
            low_unique_ratio_detected,
            high_token_freq_detected,
        )
        for detected in signals:
            if detected:
                penalty *= float(repetition_penalty_factor)
        penalty = max(0.0, float(penalty))

    return RepetitionMetrics(
        penalty=float(penalty),
        unique_token_ratio=unique_ratio,
        max_token_freq_ratio=max_freq_ratio,
        repeated_char_run_detected=repeated_char_run_detected,
        repeated_number_run_detected=repeated_number_run_detected,
        repeated_token_run_detected=repeated_token_run_detected,
        repeated_ngram_detected=repeated_ngram_detected,
        repeated_answer_span_detected=repeated_answer_span_detected,
        low_unique_ratio_detected=low_unique_ratio_detected,
        high_token_freq_detected=high_token_freq_detected,
    )


def _source_aware_credit(
    source: str,
    *,
    source_answer_marker_credit: float = 0.20,
    source_final_line_credit: float = 0.15,
    source_final_window_credit: float = 0.08,
    source_incomplete_answer_tag_credit: float = 0.03,
    source_other_answer_span_credit: float = 0.05,
) -> float:
    if source in {"strict_answer_tag", "strict_boxed"}:
        return 1.0
    if source == "answer_marker":
        return float(source_answer_marker_credit)
    if source == "final_line":
        return float(source_final_line_credit)
    if source == "final_window":
        return float(source_final_window_credit)
    if source == "incomplete_answer_tag":
        return float(source_incomplete_answer_tag_credit)
    return float(source_other_answer_span_credit)


def has_malformed_answer_structure(text: str | None) -> bool:
    """Detect incomplete or malformed answer-tag structure without judging content."""
    raw_text = text or ""
    lowered = raw_text.lower()
    has_answer_fragment = "<answer" in lowered or "</answer>" in lowered
    if not has_answer_fragment:
        return False
    complete_matches = list(_ANSWER_TAG_BODY_RE.finditer(raw_text))
    if not complete_matches:
        return True
    if any(not match.group(1).strip() for match in complete_matches):
        return True
    outside_complete = _ANSWER_TAG_BODY_RE.sub("", raw_text).lower()
    return "<answer" in outside_complete or "</answer>" in outside_complete


def apply_negative_junk_penalty(
    *,
    reward: float,
    strict_correct: bool,
    repeated_junk_detected: bool,
    malformed_junk_detected: bool,
    answer_source: str | None,
    enable_negative_junk_penalty: bool = False,
    repeated_junk_negative_reward: float = 0.02,
    malformed_junk_negative_reward: float = 0.01,
    incomplete_answer_tag_negative_reward: float = 0.01,
    min_reward_floor: float = -0.05,
) -> NegativeJunkPenaltyResult:
    reward_before = float(reward)
    if not enable_negative_junk_penalty or strict_correct:
        return NegativeJunkPenaltyResult(
            reward_before_negative_junk_penalty=reward_before,
            reward_after_negative_junk_penalty=reward_before,
            negative_junk_penalty=0.0,
            repeated_junk_negative_penalty_applied=False,
            malformed_junk_negative_penalty_applied=False,
            incomplete_answer_tag_negative_penalty_applied=False,
        )

    repeated_applied = bool(repeated_junk_detected)
    malformed_applied = bool(malformed_junk_detected)
    incomplete_applied = answer_source == "incomplete_answer_tag"
    penalty = 0.0
    if repeated_applied:
        penalty += float(repeated_junk_negative_reward)
    if malformed_applied:
        penalty += float(malformed_junk_negative_reward)
    if incomplete_applied:
        penalty += float(incomplete_answer_tag_negative_reward)

    reward_after = max(float(min_reward_floor), reward_before - penalty)
    return NegativeJunkPenaltyResult(
        reward_before_negative_junk_penalty=reward_before,
        reward_after_negative_junk_penalty=float(reward_after),
        negative_junk_penalty=float(penalty),
        repeated_junk_negative_penalty_applied=repeated_applied,
        malformed_junk_negative_penalty_applied=malformed_applied,
        incomplete_answer_tag_negative_penalty_applied=incomplete_applied,
    )


def compute_reward_quality(
    text: str | None,
    reference_answer,
    *,
    extractor_mode: str = "answer_span",
    reward_quality_mode: str = "none",
    format_ok: bool | None = None,
    answer_span_partial_credit: float = 0.4,
    malformed_answer_factor: float = 0.5,
    enable_repetition_penalty: bool = False,
    repetition_min_tokens: int = 8,
    repetition_number_run_threshold: int = 3,
    repetition_token_run_threshold: int = 4,
    repetition_bigram_run_threshold: int = 3,
    repetition_answer_span_number_threshold: int = 2,
    repetition_unique_ratio_threshold: float = 0.25,
    repetition_max_freq_threshold: float = 0.25,
    repetition_penalty_factor: float = 0.5,
    source_answer_marker_credit: float = 0.20,
    source_final_line_credit: float = 0.15,
    source_final_window_credit: float = 0.08,
    source_incomplete_answer_tag_credit: float = 0.03,
    source_other_answer_span_credit: float = 0.05,
    zero_reward_if_repeated_answer_span: bool = True,
) -> RewardQualityResult:
    strict = extract_gsm_answer_with_source(text or "", mode="strict")
    answer_span = extract_gsm_answer_with_source(text or "", mode="answer_span")
    robust = extract_gsm_answer_with_source(text or "", mode="robust")
    selected = extract_gsm_answer_with_source(text or "", mode=extractor_mode)

    strict_correct = numeric_equal(strict.answer, reference_answer)
    answer_span_correct = numeric_equal(answer_span.answer, reference_answer)
    robust_correct = numeric_equal(robust.answer, reference_answer)
    selected_correct = numeric_equal(selected.answer, reference_answer)

    repetition = compute_repetition_metrics(
        text,
        enable_repetition_penalty=enable_repetition_penalty,
        repetition_min_tokens=repetition_min_tokens,
        repetition_number_run_threshold=repetition_number_run_threshold,
        repetition_token_run_threshold=repetition_token_run_threshold,
        repetition_bigram_run_threshold=repetition_bigram_run_threshold,
        repetition_answer_span_number_threshold=repetition_answer_span_number_threshold,
        repetition_unique_ratio_threshold=repetition_unique_ratio_threshold,
        repetition_max_freq_threshold=repetition_max_freq_threshold,
        repetition_penalty_factor=repetition_penalty_factor,
    )

    if reward_quality_mode == "none":
        answer_quality = 1.0 if selected_correct else 0.0
        format_factor = 1.0
        repetition_penalty = 1.0
    elif reward_quality_mode == "format_weighted":
        if strict_correct:
            answer_quality = 1.0
            format_factor = 1.0
        elif answer_span_correct:
            answer_quality = float(answer_span_partial_credit)
            format_factor = (
                1.0 if bool(format_ok) else float(malformed_answer_factor)
            )
        else:
            answer_quality = 0.0
            format_factor = 1.0
        repetition_penalty = repetition.penalty
    elif reward_quality_mode == "source_aware_anti_junk":
        if strict_correct:
            answer_quality = 1.0
            format_factor = 1.0
        elif answer_span_correct:
            answer_quality = _source_aware_credit(
                answer_span.source,
                source_answer_marker_credit=source_answer_marker_credit,
                source_final_line_credit=source_final_line_credit,
                source_final_window_credit=source_final_window_credit,
                source_incomplete_answer_tag_credit=(
                    source_incomplete_answer_tag_credit
                ),
                source_other_answer_span_credit=source_other_answer_span_credit,
            )
            format_factor = 1.0
        else:
            answer_quality = 0.0
            format_factor = 1.0
        if zero_reward_if_repeated_answer_span and repetition.repeated_answer_span_detected:
            answer_quality = 0.0
        repetition_penalty = repetition.penalty
    else:
        raise ValueError(
            f"Unknown reward_quality_mode={reward_quality_mode!r}; "
            "expected 'none', 'format_weighted', or 'source_aware_anti_junk'"
        )

    if answer_quality <= 0.0:
        task_score = 0.0
    else:
        task_score = answer_quality * format_factor * repetition_penalty

    return RewardQualityResult(
        task_score=float(task_score),
        answer_quality=float(answer_quality),
        format_factor=float(format_factor),
        format_ok=format_ok,
        repetition_penalty=float(repetition_penalty),
        unique_token_ratio=repetition.unique_token_ratio,
        max_token_freq_ratio=repetition.max_token_freq_ratio,
        repeated_char_run_detected=repetition.repeated_char_run_detected,
        repeated_number_run_detected=repetition.repeated_number_run_detected,
        repeated_token_run_detected=repetition.repeated_token_run_detected,
        repeated_ngram_detected=repetition.repeated_ngram_detected,
        repeated_answer_span_detected=repetition.repeated_answer_span_detected,
        low_unique_ratio_detected=repetition.low_unique_ratio_detected,
        high_token_freq_detected=repetition.high_token_freq_detected,
        repeated_junk_detected=repetition.any_repetition_detected,
        parsed_answer=selected.answer,
        answer_source=selected.source,
        answer_span_source=answer_span.source,
        strict_correct=strict_correct,
        answer_span_correct=answer_span_correct,
        robust_correct=robust_correct,
    )


def apply_reward_quality_caps(
    *,
    reward: float,
    is_malformed: bool,
    repeated_junk_detected: bool,
    max_reward_if_malformed: float | None = None,
    max_reward_if_repetitive: float | None = None,
) -> RewardCapResult:
    reward_before = float(reward)
    capped_reward = reward_before
    malformed_cap_applied = False
    repetitive_cap_applied = False

    if (
        is_malformed
        and max_reward_if_malformed is not None
        and capped_reward > float(max_reward_if_malformed)
    ):
        capped_reward = float(max_reward_if_malformed)
        malformed_cap_applied = True

    if (
        repeated_junk_detected
        and max_reward_if_repetitive is not None
        and capped_reward > float(max_reward_if_repetitive)
    ):
        capped_reward = float(max_reward_if_repetitive)
        repetitive_cap_applied = True

    return RewardCapResult(
        reward_before_quality_caps=reward_before,
        reward_after_quality_caps=float(capped_reward),
        malformed_reward_cap_applied=malformed_cap_applied,
        repetitive_reward_cap_applied=repetitive_cap_applied,
    )


def compute_budget_reward(
    *,
    task_score: float,
    nfe_used: int | float,
    target_budget: int | float,
    max_steps: int | float,
    remask_count: int | float = 0,
    reward_budget_mode: str = "cap",
    reward_beta: float = 2.0,
    reward_mu: float = 0.5,
    reward_rho: float = 0.1,
    threshold_reward_hard_zero: bool = False,
    eps: float = 1e-8,
) -> BudgetRewardResult:
    safe_target_budget = max(float(target_budget), eps)
    nfe = float(nfe_used)
    nfe_target_ratio = nfe / safe_target_budget

    if reward_budget_mode == "none":
        return BudgetRewardResult(
            reward=float(task_score),
            threshold_gate=1.0,
            budget_over_penalty=1.0,
            budget_multiplier=1.0,
            nfe_target_ratio=float(nfe_target_ratio),
        )

    if float(task_score) <= 0.0:
        return BudgetRewardResult(
            reward=0.0,
            threshold_gate=0.0,
            budget_over_penalty=0.0,
            budget_multiplier=0.0,
            nfe_target_ratio=float(nfe_target_ratio),
        )

    safe_max_steps = max(float(max_steps), eps)
    remasks = float(remask_count)
    over = max(0.0, nfe - safe_target_budget) / safe_target_budget

    if reward_budget_mode in {"cap", "band"}:
        budget_over_penalty = math.exp(-float(reward_beta) * over)
        compute_penalty = math.exp(-float(reward_mu) * nfe / safe_max_steps)
        remask_penalty = math.exp(-float(reward_rho) * remasks)
        budget_multiplier = budget_over_penalty * compute_penalty * remask_penalty
        reward = float(task_score) * budget_multiplier
        return BudgetRewardResult(
            reward=float(reward),
            threshold_gate=float(budget_over_penalty),
            budget_over_penalty=float(budget_over_penalty),
            budget_multiplier=float(budget_multiplier),
            nfe_target_ratio=float(nfe_target_ratio),
        )

    if reward_budget_mode == "threshold":
        if threshold_reward_hard_zero:
            threshold_gate = 1.0 if nfe <= safe_target_budget else 0.0
        else:
            threshold_gate = math.exp(-float(reward_beta) * over)
        remask_penalty = math.exp(-float(reward_rho) * remasks)
        budget_multiplier = threshold_gate * remask_penalty
        reward = float(task_score) * budget_multiplier
        return BudgetRewardResult(
            reward=float(reward),
            threshold_gate=float(threshold_gate),
            budget_over_penalty=float(threshold_gate),
            budget_multiplier=float(budget_multiplier),
            nfe_target_ratio=float(nfe_target_ratio),
        )

    raise ValueError(
        f"Unknown reward_budget_mode={reward_budget_mode!r}; "
        "expected 'none', 'cap', 'band', or 'threshold'"
    )


def deliberation_bonus(
    *,
    early_task_score: float,
    final_task_score: float,
    threshold_gate: float,
    deliberation_bonus_weight: float,
) -> tuple[float, float]:
    gain = max(0.0, float(final_task_score) - float(early_task_score))
    return gain, float(deliberation_bonus_weight) * gain * float(threshold_gate)


def patient_win_bonus_applies(
    *,
    patient_task_score: float,
    normal_task_score: float,
    patient_nfe: int | float,
    target_budget: int | float,
    patient_win_margin: float = 0.05,
) -> bool:
    return (
        float(patient_task_score) > float(normal_task_score) + float(patient_win_margin)
        and float(patient_nfe) <= float(target_budget)
    )


def _has_complete_answer_tag(text: str | None) -> bool:
    lowered = (text or "").lower()
    return "<answer>" in lowered and "</answer>" in lowered


def _continuation_metric_fields(
    *,
    base_task_score: float,
    continued_task_score: float,
    base_correct: bool,
    continued_correct: bool,
    base_quality_score: float,
    continued_quality_score: float,
) -> dict[str, Any]:
    return {
        "continuation_gain": float(continued_task_score) - float(base_task_score),
        "continuation_correct_gain": bool(continued_correct and not base_correct),
        "continuation_harm": bool(base_correct and not continued_correct),
        "continuation_quality_gain": float(continued_quality_score)
        - float(base_quality_score),
    }


def compute_gsm_reward_parse_fields(
    generated_text: str | None,
    reference_answer,
    config: Any,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    text = generated_text or ""
    strict = extract_gsm_answer_with_source(text, mode="strict")
    answer_span = extract_gsm_answer_with_source(text, mode="answer_span")
    robust = extract_gsm_answer_with_source(text, mode="robust")
    selected = extract_gsm_answer_with_source(
        text,
        mode=getattr(config, "reward_gsm_extractor", "robust"),
    )
    strict_correct = numeric_equal(strict.answer, reference_answer)
    answer_span_correct = numeric_equal(answer_span.answer, reference_answer)
    robust_correct = numeric_equal(robust.answer, reference_answer)
    reward_exact_match_used = numeric_equal(selected.answer, reference_answer)
    quality = compute_reward_quality(
        text,
        reference_answer,
        format_ok=metadata.get("format_ok"),
        **reward_quality_kwargs_from_config(config),
    )
    effective_mode = effective_reward_budget_mode(config)
    task_only_quality_scale = uses_task_only_quality_scale(config)
    completion_length_normalizer = 1.0
    if task_only_quality_scale:
        assert completion_length_normalizer == 1.0
    raw_task_score = quality.task_score
    is_clean_format = bool(quality.strict_correct) or bool(quality.format_ok)
    repeated_junk_detected = bool(quality.repeated_junk_detected) or (
        float(quality.repetition_penalty) < 1.0
    )
    if task_only_quality_scale:
        task_score = raw_task_score
        reward_before_quality_caps = raw_task_score
        malformed_reward_cap_applied = False
        repetitive_reward_cap_applied = False
    else:
        task_score_cap = apply_reward_quality_caps(
            reward=raw_task_score,
            is_malformed=not is_clean_format,
            repeated_junk_detected=repeated_junk_detected,
            max_reward_if_malformed=getattr(config, "max_reward_if_malformed", None),
            max_reward_if_repetitive=getattr(config, "max_reward_if_repetitive", None),
        )
        task_score = task_score_cap.reward_after_quality_caps
        reward_before_quality_caps = task_score_cap.reward_before_quality_caps
        malformed_reward_cap_applied = task_score_cap.malformed_reward_cap_applied
        repetitive_reward_cap_applied = task_score_cap.repetitive_reward_cap_applied
    budget = compute_budget_reward(
        task_score=task_score,
        nfe_used=metadata.get("actual_nfe", metadata.get("nfe_used", 0)),
        target_budget=metadata.get(
            "target_budget",
            getattr(config, "target_budget", 1),
        ),
        max_steps=metadata.get(
            "max_steps",
            getattr(config, "max_completion_length", 1),
        ),
        remask_count=metadata.get("remask_count", 0),
        reward_budget_mode=effective_mode,
        reward_beta=getattr(config, "reward_beta", 2.0),
        reward_mu=getattr(config, "reward_mu", 0.5),
        reward_rho=getattr(config, "reward_rho", 0.1),
        threshold_reward_hard_zero=getattr(
            config,
            "threshold_reward_hard_zero",
            False,
        ),
    )
    early_text = metadata.get("early_generated_text")
    early_quality = (
        compute_reward_quality(
            early_text,
            reference_answer,
            format_ok=None,
            **reward_quality_kwargs_from_config(config),
        )
        if early_text is not None
        else None
    )
    early_task_score = early_quality.task_score if early_quality is not None else 0.0
    gain, _ = deliberation_bonus(
        early_task_score=early_task_score,
        final_task_score=task_score,
        threshold_gate=budget.threshold_gate,
        deliberation_bonus_weight=getattr(config, "deliberation_bonus_weight", 0.0),
    )
    base_correct = task_score > 0.0
    continuation_results = []
    for raw_result in metadata.get("continuation_results", []) or []:
        continued_text = raw_result.get("continued_generated_text")
        continued_quality = compute_reward_quality(
            continued_text,
            reference_answer,
            format_ok=raw_result.get("continued_format_ok"),
            **reward_quality_kwargs_from_config(config),
        )
        continued_correct = continued_quality.task_score > 0.0
        metric_fields = _continuation_metric_fields(
            base_task_score=task_score,
            continued_task_score=continued_quality.task_score,
            base_correct=base_correct,
            continued_correct=continued_correct,
            base_quality_score=task_score,
            continued_quality_score=continued_quality.task_score,
        )
        continuation_results.append(
            {
                **raw_result,
                "continued_parsed_answer": continued_quality.parsed_answer,
                "continued_task_score": continued_quality.task_score,
                "continued_correct": continued_correct,
                "continued_quality_score": continued_quality.task_score,
                **metric_fields,
            }
        )
    best_continuation = None
    if continuation_results:
        best_continuation = max(
            continuation_results,
            key=lambda item: (
                float(item.get("continued_task_score") or 0.0),
                -int(item.get("continuation_k") or 0),
            ),
        )
    return {
        "strict_parsed_answer": strict.answer,
        "answer_span_parsed_answer": answer_span.answer,
        "robust_parsed_answer": robust.answer,
        "parsed_answer_used_for_reward": selected.answer,
        "final_parsed_answer": selected.answer,
        "early_parsed_answer": early_quality.parsed_answer
        if early_quality is not None
        else None,
        "strict_correct": strict_correct,
        "answer_span_correct": answer_span_correct,
        "robust_correct": robust_correct,
        "reward_exact_match_used": reward_exact_match_used,
        "format_ok": metadata.get("format_ok"),
        "reward_answer_source": selected.source,
        "answer_source": selected.source,
        "reward_quality_mode": getattr(config, "reward_quality_mode", "none"),
        "answer_quality": quality.answer_quality,
        "format_factor": quality.format_factor,
        "repetition_penalty": quality.repetition_penalty,
        "unique_token_ratio": quality.unique_token_ratio,
        "max_token_freq_ratio": quality.max_token_freq_ratio,
        "repeated_char_run_detected": quality.repeated_char_run_detected,
        "repeated_number_run_detected": quality.repeated_number_run_detected,
        "repeated_token_run_detected": quality.repeated_token_run_detected,
        "repeated_ngram_detected": quality.repeated_ngram_detected,
        "repeated_answer_span_detected": quality.repeated_answer_span_detected,
        "low_unique_ratio_detected": quality.low_unique_ratio_detected,
        "high_token_freq_detected": quality.high_token_freq_detected,
        "repeated_junk_detected": quality.repeated_junk_detected,
        "quality_task_score_before_scaling": raw_task_score,
        "quality_task_score_after_caps": task_score,
        "completion_length_normalizer": completion_length_normalizer,
        "raw_task_score": raw_task_score,
        "task_score_before_compute": task_score,
        "task_score_after_quality": task_score,
        "base_task_score": task_score,
        "base_correct": base_correct,
        "base_quality_score": task_score,
        "base_repetition_penalty": quality.repetition_penalty,
        "early_task_score": early_task_score,
        "final_task_score": task_score,
        "deliberation_gain": gain,
        "early_nfe": metadata.get("early_nfe"),
        "final_nfe": metadata.get("actual_nfe", metadata.get("nfe_used")),
        "final_reward": budget.reward,
        "reward_budget_mode": effective_mode,
        "reward_type": getattr(config, "reward_type", None),
        "reward_type/task_only": float(
            getattr(config, "reward_type", None) == "task_only"
        ),
        "threshold_gate": budget.threshold_gate,
        "budget_over_penalty": budget.budget_over_penalty,
        "budget_factor": budget.budget_multiplier,
        "budget_multiplier": budget.budget_multiplier,
        "budgeted_reward": budget.reward,
        "reward_before_quality_caps": reward_before_quality_caps,
        "reward_after_quality_caps": task_score,
        "malformed_reward_cap_applied": malformed_reward_cap_applied,
        "repetitive_reward_cap_applied": repetitive_reward_cap_applied,
        "nfe_target_ratio": budget.nfe_target_ratio,
        "continuation_results": continuation_results,
        "best_continuation_k": best_continuation.get("continuation_k")
        if best_continuation
        else None,
        "best_continuation_task_score": best_continuation.get(
            "continued_task_score"
        )
        if best_continuation
        else None,
        "best_continuation_gain": best_continuation.get("continuation_gain")
        if best_continuation
        else None,
        "best_continuation_correct": best_continuation.get("continued_correct")
        if best_continuation
        else None,
        "best_continuation_text": best_continuation.get("continued_generated_text")
        if best_continuation
        else None,
        "has_boxed_answer": "\\boxed" in text,
        "has_complete_answer_tag": _has_complete_answer_tag(text),
        "has_malformed_answer_structure": has_malformed_answer_structure(text),
    }


@dataclass(frozen=True)
class RewardSelection:
    reward_type: str
    reward_function_names: list[str]
    parameters: dict[str, Any]


def validate_reward_type_matches_functions(config: Any) -> None:
    """Reject mismatches between reward_type and explicit reward_functions."""
    funcs = list(getattr(config, "reward_functions", None) or [])
    if not funcs:
        return

    expected = {
        "multiplicative_budget": "mixed_multiplicative_budget_reward_func",
        "additive_proposal": "mixed_additive_proposal_reward_func",
        "task_only": "mixed_multiplicative_budget_reward_func",
    }
    reward_type = getattr(config, "reward_type", "multiplicative_budget")
    expected_func = expected.get(reward_type)
    if expected_func is None:
        raise ValueError(f"Unknown reward_type={reward_type}")
    if funcs != [expected_func]:
        raise ValueError(
            f"reward_type={reward_type} expects "
            f"reward_functions=[{expected_func}], but got reward_functions={funcs}. "
            "Fix the YAML to avoid misleading logs."
        )


def select_reward_functions_from_config(config: Any) -> RewardSelection:
    reward_type = getattr(config, "reward_type", "multiplicative_budget")
    if reward_type == "additive_proposal":
        return RewardSelection(
            reward_type=reward_type,
            reward_function_names=["mixed_additive_proposal_reward_func"],
            parameters={
                "reward_gsm_extractor": getattr(
                    config,
                    "reward_gsm_extractor",
                    "robust",
                ),
                "reward_lambda_compute": getattr(
                    config, "reward_lambda_compute", 0.01
                ),
                "reward_normalize_compute": getattr(
                    config, "reward_normalize_compute", True
                ),
                "reward_quality_mode": getattr(config, "reward_quality_mode", "none"),
                "answer_span_partial_credit": getattr(
                    config,
                    "answer_span_partial_credit",
                    0.4,
                ),
                "malformed_answer_factor": getattr(
                    config,
                    "malformed_answer_factor",
                    0.5,
                ),
                "source_answer_marker_credit": getattr(
                    config,
                    "source_answer_marker_credit",
                    0.20,
                ),
                "source_final_line_credit": getattr(
                    config,
                    "source_final_line_credit",
                    0.15,
                ),
                "source_final_window_credit": getattr(
                    config,
                    "source_final_window_credit",
                    0.08,
                ),
                "source_incomplete_answer_tag_credit": getattr(
                    config,
                    "source_incomplete_answer_tag_credit",
                    0.03,
                ),
                "source_other_answer_span_credit": getattr(
                    config,
                    "source_other_answer_span_credit",
                    0.05,
                ),
                "enable_repetition_penalty": getattr(
                    config,
                    "enable_repetition_penalty",
                    False,
                ),
                "repetition_min_tokens": getattr(config, "repetition_min_tokens", 8),
                "repetition_number_run_threshold": getattr(
                    config,
                    "repetition_number_run_threshold",
                    3,
                ),
                "repetition_token_run_threshold": getattr(
                    config,
                    "repetition_token_run_threshold",
                    4,
                ),
                "repetition_bigram_run_threshold": getattr(
                    config,
                    "repetition_bigram_run_threshold",
                    3,
                ),
                "repetition_answer_span_number_threshold": getattr(
                    config,
                    "repetition_answer_span_number_threshold",
                    2,
                ),
                "zero_reward_if_repeated_answer_span": getattr(
                    config,
                    "zero_reward_if_repeated_answer_span",
                    True,
                ),
            },
        )
    if reward_type == "multiplicative_budget":
        return RewardSelection(
            reward_type=reward_type,
            reward_function_names=["mixed_multiplicative_budget_reward_func"],
            parameters={
                "reward_gsm_extractor": getattr(
                    config,
                    "reward_gsm_extractor",
                    "robust",
                ),
                "reward_beta": getattr(config, "reward_beta", 2.0),
                "reward_mu": getattr(config, "reward_mu", 0.5),
                "reward_rho": getattr(config, "reward_rho", 0.1),
                "reward_quality_mode": getattr(config, "reward_quality_mode", "none"),
                "answer_span_partial_credit": getattr(
                    config,
                    "answer_span_partial_credit",
                    0.4,
                ),
                "malformed_answer_factor": getattr(
                    config,
                    "malformed_answer_factor",
                    0.5,
                ),
                "source_answer_marker_credit": getattr(
                    config,
                    "source_answer_marker_credit",
                    0.20,
                ),
                "source_final_line_credit": getattr(
                    config,
                    "source_final_line_credit",
                    0.15,
                ),
                "source_final_window_credit": getattr(
                    config,
                    "source_final_window_credit",
                    0.08,
                ),
                "source_incomplete_answer_tag_credit": getattr(
                    config,
                    "source_incomplete_answer_tag_credit",
                    0.03,
                ),
                "source_other_answer_span_credit": getattr(
                    config,
                    "source_other_answer_span_credit",
                    0.05,
                ),
                "enable_repetition_penalty": getattr(
                    config,
                    "enable_repetition_penalty",
                    False,
                ),
                "repetition_min_tokens": getattr(config, "repetition_min_tokens", 8),
                "repetition_number_run_threshold": getattr(
                    config,
                    "repetition_number_run_threshold",
                    3,
                ),
                "repetition_token_run_threshold": getattr(
                    config,
                    "repetition_token_run_threshold",
                    4,
                ),
                "repetition_bigram_run_threshold": getattr(
                    config,
                    "repetition_bigram_run_threshold",
                    3,
                ),
                "repetition_answer_span_number_threshold": getattr(
                    config,
                    "repetition_answer_span_number_threshold",
                    2,
                ),
                "zero_reward_if_repeated_answer_span": getattr(
                    config,
                    "zero_reward_if_repeated_answer_span",
                    True,
                ),
                "reward_budget_mode": getattr(config, "reward_budget_mode", "cap"),
                "threshold_reward_hard_zero": getattr(
                    config,
                    "threshold_reward_hard_zero",
                    False,
                ),
                "enable_negative_junk_penalty": getattr(
                    config,
                    "enable_negative_junk_penalty",
                    False,
                ),
                "repeated_junk_negative_reward": getattr(
                    config,
                    "repeated_junk_negative_reward",
                    0.02,
                ),
                "malformed_junk_negative_reward": getattr(
                    config,
                    "malformed_junk_negative_reward",
                    0.01,
                ),
                "incomplete_answer_tag_negative_reward": getattr(
                    config,
                    "incomplete_answer_tag_negative_reward",
                    0.01,
                ),
                "min_reward_floor": getattr(config, "min_reward_floor", -0.05),
            },
        )
    if reward_type == "task_only":
        return RewardSelection(
            reward_type=reward_type,
            reward_function_names=["mixed_multiplicative_budget_reward_func"],
            parameters={
                "reward_gsm_extractor": getattr(
                    config,
                    "reward_gsm_extractor",
                    "robust",
                ),
                "reward_beta": 0.0,
                "reward_mu": 0.0,
                "reward_rho": 0.0,
                "reward_quality_mode": getattr(config, "reward_quality_mode", "none"),
                "answer_span_partial_credit": getattr(
                    config,
                    "answer_span_partial_credit",
                    0.4,
                ),
                "malformed_answer_factor": getattr(
                    config,
                    "malformed_answer_factor",
                    0.5,
                ),
                "source_answer_marker_credit": getattr(
                    config,
                    "source_answer_marker_credit",
                    0.20,
                ),
                "source_final_line_credit": getattr(
                    config,
                    "source_final_line_credit",
                    0.15,
                ),
                "source_final_window_credit": getattr(
                    config,
                    "source_final_window_credit",
                    0.08,
                ),
                "source_incomplete_answer_tag_credit": getattr(
                    config,
                    "source_incomplete_answer_tag_credit",
                    0.03,
                ),
                "source_other_answer_span_credit": getattr(
                    config,
                    "source_other_answer_span_credit",
                    0.05,
                ),
                "enable_repetition_penalty": getattr(
                    config,
                    "enable_repetition_penalty",
                    False,
                ),
                "repetition_min_tokens": getattr(config, "repetition_min_tokens", 8),
                "repetition_number_run_threshold": getattr(
                    config,
                    "repetition_number_run_threshold",
                    3,
                ),
                "repetition_token_run_threshold": getattr(
                    config,
                    "repetition_token_run_threshold",
                    4,
                ),
                "repetition_bigram_run_threshold": getattr(
                    config,
                    "repetition_bigram_run_threshold",
                    3,
                ),
                "repetition_answer_span_number_threshold": getattr(
                    config,
                    "repetition_answer_span_number_threshold",
                    2,
                ),
                "repetition_unique_ratio_threshold": getattr(
                    config,
                    "repetition_unique_ratio_threshold",
                    0.25,
                ),
                "repetition_max_freq_threshold": getattr(
                    config,
                    "repetition_max_freq_threshold",
                    0.25,
                ),
                "repetition_penalty_factor": getattr(
                    config,
                    "repetition_penalty_factor",
                    0.5,
                ),
                "zero_reward_if_repeated_answer_span": getattr(
                    config,
                    "zero_reward_if_repeated_answer_span",
                    True,
                ),
                "reward_budget_mode": "none",
                "threshold_reward_hard_zero": False,
                "enable_negative_junk_penalty": False,
                "repeated_junk_negative_reward": getattr(
                    config,
                    "repeated_junk_negative_reward",
                    0.02,
                ),
                "malformed_junk_negative_reward": getattr(
                    config,
                    "malformed_junk_negative_reward",
                    0.01,
                ),
                "incomplete_answer_tag_negative_reward": getattr(
                    config,
                    "incomplete_answer_tag_negative_reward",
                    0.01,
                ),
                "min_reward_floor": getattr(config, "min_reward_floor", 0.0),
            },
        )
    raise ValueError(
        f"Unknown reward_type '{reward_type}'. "
        "Expected 'additive_proposal', 'multiplicative_budget', or 'task_only'."
    )
