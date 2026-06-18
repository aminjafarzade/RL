#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import torch

from common.remasking.span_finder import AnswerSpan
from common.verifiers.base import VerifierResult


@dataclass
class RemaskDecision:
    token_indices: list[int] = field(default_factory=list)
    reason: str = "none"
    span: AnswerSpan | None = None

    @property
    def should_remask(self) -> bool:
        return bool(self.token_indices)


@dataclass
class RemaskState:
    remask_count: int = 0
    last_remask_step: int | None = None
    remasked_spans: set[tuple[int, ...]] = field(default_factory=set)


class NoRemask:
    def select(self, *args, **kwargs) -> RemaskDecision:
        return RemaskDecision()


class LowConfidenceRemask:
    def __init__(
        self,
        max_tokens: int = 1,
        confidence_threshold: float = 0.2,
    ):
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold

    def select(
        self,
        confidences: Sequence[float] | torch.Tensor,
        eligible_mask: Sequence[bool] | torch.Tensor,
    ) -> RemaskDecision:
        if isinstance(confidences, torch.Tensor):
            confidence_tensor = confidences.detach().float().cpu()
        else:
            confidence_tensor = torch.tensor(list(confidences), dtype=torch.float32)
        if isinstance(eligible_mask, torch.Tensor):
            eligible = eligible_mask.detach().bool().cpu()
        else:
            eligible = torch.tensor(list(eligible_mask), dtype=torch.bool)

        candidate_mask = eligible & (confidence_tensor < self.confidence_threshold)
        if not candidate_mask.any():
            return RemaskDecision()
        candidate_scores = confidence_tensor.masked_fill(~candidate_mask, float("inf"))
        k = min(self.max_tokens, int(candidate_mask.sum().item()))
        token_indices = torch.topk(candidate_scores, k=k, largest=False).indices
        return RemaskDecision(
            token_indices=[int(idx.item()) for idx in token_indices],
            reason="low_confidence",
        )


class VerifierAnswerRemask:
    def __init__(
        self,
        max_remasks_per_sample: int = 1,
        remask_cooldown_steps: int = 2,
        min_steps_before_remask: int = 0,
        verifier_threshold: float = 0.7,
    ):
        self.max_remasks_per_sample = max_remasks_per_sample
        self.remask_cooldown_steps = remask_cooldown_steps
        self.min_steps_before_remask = min_steps_before_remask
        self.verifier_threshold = verifier_threshold

    def select(
        self,
        verifier_result: VerifierResult,
        span: AnswerSpan | None,
        state: RemaskState,
        current_step: int,
        prompt_token_count: int = 0,
    ) -> RemaskDecision:
        if span is None or not span.token_indices:
            return RemaskDecision(reason="no_answer_span")
        if verifier_result.verifier_score >= self.verifier_threshold:
            return RemaskDecision(reason="verifier_above_threshold", span=span)
        if current_step < self.min_steps_before_remask:
            return RemaskDecision(reason="before_min_step", span=span)
        if state.remask_count >= self.max_remasks_per_sample:
            return RemaskDecision(reason="max_remasks_reached", span=span)
        if state.last_remask_step is not None:
            elapsed = current_step - state.last_remask_step
            if elapsed < self.remask_cooldown_steps:
                return RemaskDecision(reason="cooldown", span=span)

        token_indices = [idx for idx in span.token_indices if idx >= prompt_token_count]
        if not token_indices:
            return RemaskDecision(reason="prompt_span_rejected", span=span)
        span_key = tuple(token_indices)
        if span_key in state.remasked_spans:
            return RemaskDecision(reason="repeat_span_rejected", span=span)

        return RemaskDecision(
            token_indices=token_indices,
            reason="verifier_answer_span",
            span=span,
        )

    def record(self, state: RemaskState, decision: RemaskDecision, step: int) -> None:
        if not decision.should_remask:
            return
        state.remask_count += 1
        state.last_remask_step = step
        state.remasked_spans.add(tuple(decision.token_indices))
