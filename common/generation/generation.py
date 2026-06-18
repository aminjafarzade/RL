#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import math

from typing import Any
from typing import NamedTuple

import torch
import torch.nn.functional as F

from common.generation.sampling import bernoulli_sample
from common.generation.sampling import dpls_sample
from common.parsing.parse_and_get_acc import extract_gsm_answer_with_source
from common.policy_features import build_policy_extra_features
from common.rewards import compute_repetition_metrics
from common.remasking.span_finder import find_final_answer_span
from common.remasking.strategies import RemaskState
from common.remasking.strategies import VerifierAnswerRemask
from common.verifiers.math_verifier import MathVerifier
from common.verifiers.math_verifier import has_answer_marker


class GenerationResult(NamedTuple):
    sequences: torch.Tensor  # Generated sequences (B, prompt_L + gen_L)
    steps_taken: torch.Tensor  # Steps taken per batch item

    # Policy training data (None for non-policy modes)
    sampling_inputs: torch.Tensor | None = None  # (B, T, BL)
    samples: torch.Tensor | None = None  # (B, T, BL)
    sampling_masks: torch.Tensor | None = None  # (B, T, BL)
    policy_inputs: tuple[torch.Tensor, ...] | None = None
    still_masked: torch.Tensor | None = None  # (B,)
    metadata: list[dict[str, Any]] | None = None
    step_traces: list[list[dict[str, Any]]] | None = None


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _target_budget_limit(
    target_budget: int | torch.Tensor | None,
    max_steps: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if target_budget is None:
        return torch.full((batch_size,), int(max_steps), dtype=torch.long, device=device)
    if isinstance(target_budget, torch.Tensor):
        return (
            target_budget.to(device=device, dtype=torch.long)
            .view(batch_size)
            .clamp(min=0, max=max_steps)
        )
    return torch.full(
        (batch_size,),
        min(max_steps, int(target_budget)),
        dtype=torch.long,
        device=device,
    )


def _target_budget_values(
    target_budget: int | torch.Tensor | None,
    default_budget: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if target_budget is None:
        return torch.full(
            (batch_size,),
            int(default_budget),
            dtype=torch.long,
            device=device,
        )
    if isinstance(target_budget, torch.Tensor):
        return target_budget.to(device=device, dtype=torch.long).view(batch_size)
    return torch.full((batch_size,), int(target_budget), dtype=torch.long, device=device)


def _generation_step_limit(
    target_budget: int | torch.Tensor | None,
    max_steps: int,
    batch_size: int,
    device: torch.device,
    hard_stop_at_target_budget: bool = True,
    hard_generation_budget: int | None = None,
) -> torch.Tensor:
    if hard_stop_at_target_budget:
        return _target_budget_limit(target_budget, max_steps, batch_size, device)
    if hard_generation_budget is not None:
        return torch.full(
            (batch_size,),
            min(max_steps, int(hard_generation_budget)),
            dtype=torch.long,
            device=device,
        )
    return torch.full((batch_size,), int(max_steps), dtype=torch.long, device=device)


def _decode_generation(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(
        [int(x) for x in token_ids.detach().cpu().tolist()],
        skip_special_tokens=True,
    )


def _normalize_rollout_styles(
    rollout_compute_style: str | list[str] | tuple[str, ...] | None,
    batch_size: int,
) -> list[str]:
    if rollout_compute_style is None:
        return ["normal"] * batch_size
    if isinstance(rollout_compute_style, str):
        return [rollout_compute_style] * batch_size
    styles = [str(style) for style in rollout_compute_style]
    if len(styles) != batch_size:
        raise ValueError(
            f"rollout_compute_style length {len(styles)} does not match batch size "
            f"{batch_size}"
        )
    return styles


def _verifier_to_dict(verifier_result) -> dict[str, Any] | None:
    if verifier_result is None:
        return None
    if hasattr(verifier_result, "to_dict"):
        return verifier_result.to_dict()
    return dict(verifier_result)


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float()
    if values.numel() == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def continuation_trigger_reasons(
    *,
    trigger: str,
    verifier_result: dict[str, Any] | None = None,
    answer_span: dict[str, Any] | None = None,
    reward_answer_source: str | None = None,
    repetition_penalty: float = 1.0,
    confidence_threshold: float = 0.5,
    verifier_threshold: float = 0.7,
) -> list[str]:
    verifier_result = verifier_result or {}
    answer_span = answer_span or {}
    reasons: list[str] = []
    span_confidence = answer_span.get("mean_confidence")
    verifier_score = verifier_result.get("verifier_score")
    format_ok = verifier_result.get("format_ok")
    parse_ok = verifier_result.get("parse_ok")
    arithmetic_ok = verifier_result.get("arithmetic_ok")

    if format_ok is False:
        reasons.append("format_bad")
    if parse_ok is False:
        reasons.append("parse_bad")
    if arithmetic_ok is False:
        reasons.append("arithmetic_bad")
    if reward_answer_source in {"incomplete_answer_tag", "final_window", "final_line"}:
        reasons.append(f"source_{reward_answer_source}")
    if span_confidence is not None and float(span_confidence) < float(confidence_threshold):
        reasons.append("answer_span_low_confidence")
    if float(repetition_penalty) < 1.0:
        reasons.append("repetitive")
    if verifier_score is not None and float(verifier_score) < float(verifier_threshold):
        reasons.append("verifier_low")

    has_span = bool(answer_span.get("token_indices"))
    low_confidence = "answer_span_low_confidence" in reasons
    verifier_bad = any(
        reason in reasons
        for reason in ("format_bad", "parse_bad", "arithmetic_bad", "verifier_low")
    )

    if trigger == "always":
        return ["always"]
    if trigger == "always_for_analysis":
        return ["always_for_analysis"]
    if trigger == "answer_span":
        return ["answer_span"] if has_span else []
    if trigger == "low_confidence":
        return ["answer_span_low_confidence"] if low_confidence else []
    if trigger == "verifier_bad":
        return [reason for reason in reasons if reason in {"format_bad", "parse_bad", "arithmetic_bad", "verifier_low"}]
    if trigger == "answer_span_or_low_confidence":
        selected = []
        if has_span:
            selected.append("answer_span")
        if low_confidence:
            selected.append("answer_span_low_confidence")
        return selected
    if trigger == "heuristic":
        return reasons
    raise ValueError(f"Unknown continuation_trigger: {trigger}")


def continuation_metric_fields(
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


def _valid_generated_token_mask(
    generated_tokens: torch.Tensor,
    *,
    mask_id: int,
    special_token_ids: set[int] | None = None,
) -> torch.Tensor:
    valid = generated_tokens != int(mask_id)
    for token_id in special_token_ids or set():
        valid &= generated_tokens != int(token_id)
    return valid


def select_continuation_remask_indices(
    *,
    mode: str,
    generated_tokens: torch.Tensor,
    answer_span_indices: list[int] | None = None,
    confidences: torch.Tensor | None = None,
    mask_id: int,
    special_token_ids: set[int] | None = None,
    final_window_tokens: int = 32,
    low_confidence_quantile: float = 0.3,
) -> list[int]:
    valid = _valid_generated_token_mask(
        generated_tokens,
        mask_id=mask_id,
        special_token_ids=special_token_ids,
    )
    length = int(generated_tokens.numel())

    if mode == "answer_span":
        indices = [
            int(idx)
            for idx in (answer_span_indices or [])
            if 0 <= int(idx) < length and bool(valid[int(idx)].item())
        ]
        return sorted(set(indices))

    if mode == "final_window":
        end = length
        for idx, token in enumerate(generated_tokens.detach().cpu().tolist()):
            if special_token_ids and int(token) in special_token_ids:
                end = idx
                break
        start = max(0, end - int(final_window_tokens))
        return [idx for idx in range(start, end) if bool(valid[idx].item())]

    if mode == "low_confidence_answer_window":
        window = select_continuation_remask_indices(
            mode="answer_span",
            generated_tokens=generated_tokens,
            answer_span_indices=answer_span_indices,
            confidences=confidences,
            mask_id=mask_id,
            special_token_ids=special_token_ids,
            final_window_tokens=final_window_tokens,
            low_confidence_quantile=low_confidence_quantile,
        )
        if not window:
            window = select_continuation_remask_indices(
                mode="final_window",
                generated_tokens=generated_tokens,
                answer_span_indices=answer_span_indices,
                confidences=confidences,
                mask_id=mask_id,
                special_token_ids=special_token_ids,
                final_window_tokens=final_window_tokens,
                low_confidence_quantile=low_confidence_quantile,
            )
        if not window or confidences is None:
            return window
        keep_count = max(
            1,
            math.ceil(len(window) * float(low_confidence_quantile)),
        )
        keep_count = min(len(window), keep_count)
        window_tensor = torch.tensor(window, device=confidences.device, dtype=torch.long)
        values = confidences[window_tensor]
        selected = torch.topk(values, k=keep_count, largest=False).indices
        return sorted(int(window_tensor[idx].item()) for idx in selected)

    raise ValueError(f"Unknown continuation_remask_mode: {mode}")


def _tokenizer_special_ids(tokenizer, extra_ids: set[int] | None = None) -> set[int]:
    ids = set(extra_ids or set())
    if tokenizer is None:
        return ids
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            ids.add(int(value))
    for value in getattr(tokenizer, "all_special_ids", []) or []:
        if value is not None:
            ids.add(int(value))
    return ids


def _select_confident_unmasks(
    confidence: torch.Tensor,
    active_mask: torch.Tensor,
    remaining_steps: int,
) -> torch.Tensor:
    selected = torch.zeros_like(active_mask, dtype=torch.bool)
    remaining = int(active_mask.sum().item())
    if remaining <= 0:
        return selected
    k = max(1, math.ceil(remaining / max(1, int(remaining_steps))))
    masked_conf = torch.where(active_mask, confidence, -torch.inf)
    topk = torch.topk(masked_conf, k=min(k, remaining)).indices
    selected[topk] = True
    return selected


def _run_continuation_candidate(
    *,
    model,
    base_sequence: torch.Tensor,
    prompt_L: int,
    attention_mask: torch.Tensor | None,
    model_type: str | None,
    temperature: float,
    mask_id: int,
    tokenizer,
    verifier: MathVerifier | None,
    remask_indices: list[int],
    extra_steps: int,
    base_nfe: int,
    last_confidence: torch.Tensor,
    last_entropy: torch.Tensor,
    remask_answer_span_mode: str,
) -> dict[str, Any]:
    x_cont = base_sequence.detach().clone().unsqueeze(0)
    device = x_cont.device
    generation_tokens = x_cont[0, prompt_L:]
    safe_indices = [
        int(idx)
        for idx in remask_indices
        if 0 <= int(idx) < generation_tokens.numel()
    ]
    if safe_indices:
        index_tensor = torch.tensor(safe_indices, dtype=torch.long, device=device)
        x_cont[0, prompt_L + index_tensor] = mask_id

    continued_confidence = last_confidence.detach().clone()
    continued_entropy = last_entropy.detach().clone()
    actual_extra_steps = 0
    for step_idx in range(int(extra_steps)):
        active_mask = torch.zeros_like(generation_tokens, dtype=torch.bool)
        if safe_indices:
            idx = torch.tensor(safe_indices, dtype=torch.long, device=device)
            active_mask[idx] = x_cont[0, prompt_L + idx] == mask_id
        if not bool(active_mask.any().item()):
            break

        model_output = model(
            x_cont,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        if model_type == "Dream":
            logits = model_output.logits[:, prompt_L - 1 : -1]
        else:
            logits = model_output.logits[:, prompt_L:]
        logits = logits[0]
        logits_with_noise = add_gumbel_noise(logits.unsqueeze(0), temperature)[0]
        x0 = torch.argmax(logits_with_noise, dim=-1)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(
            dim=-1
        )
        selected = _select_confident_unmasks(
            confidence,
            active_mask,
            remaining_steps=int(extra_steps) - step_idx,
        )
        x_cont[0, prompt_L:] = torch.where(selected, x0, x_cont[0, prompt_L:])
        continued_confidence = confidence.detach().float()
        continued_entropy = entropy.detach().float()
        actual_extra_steps += 1

    continued_tokens = x_cont[0, prompt_L:]
    continued_text = _decode_generation(tokenizer, continued_tokens)
    verifier_result = verifier.verify(continued_text) if verifier is not None else None
    verifier_dict = _verifier_to_dict(verifier_result)
    span = None
    if verifier_result is not None:
        span = find_final_answer_span(
            tokenizer=tokenizer,
            decoded_text=continued_text,
            generated_token_ids=continued_tokens,
            verifier_result=verifier_result,
            confidences=continued_confidence,
            min_token_index=0,
            special_token_ids={mask_id},
            remask_span_mode=remask_answer_span_mode,
        )
    span_dict = span.to_dict() if span is not None else None
    return {
        "continued_generated_text": continued_text,
        "continued_nfe": int(base_nfe) + int(actual_extra_steps),
        "continued_extra_steps": int(actual_extra_steps),
        "continued_parse_ok": verifier_dict["parse_ok"] if verifier_dict else None,
        "continued_format_ok": verifier_dict["format_ok"] if verifier_dict else None,
        "continued_arithmetic_ok": verifier_dict["arithmetic_ok"]
        if verifier_dict
        else None,
        "continued_verifier_score": verifier_dict["verifier_score"]
        if verifier_dict
        else None,
        "continued_answer_span": span_dict,
        "continued_answer_span_confidence": span_dict.get("mean_confidence")
        if span_dict
        else None,
        "continued_answer_span_entropy": _tensor_stats(
            continued_entropy[span.token_indices]
        )["mean"]
        if span is not None and span.token_indices
        else None,
    }


def _should_run_verifier(
    schedule: str,
    current_step: int,
    num_block: int,
    num_blocks: int,
    text: str,
    every_k: int,
) -> bool:
    if schedule == "none":
        return False
    if schedule == "final_only":
        return False
    if schedule == "every_k_steps":
        return every_k > 0 and current_step % every_k == 0
    if schedule == "final_block_only":
        return num_block == num_blocks - 1
    if schedule == "after_answer_detected":
        return has_answer_marker(text)
    raise ValueError(f"Unknown verifier_schedule: {schedule}")


@torch.no_grad()
def generate_unified(
    model,
    prompt: torch.Tensor,
    remasking: str,
    policy=None,
    thres: float | torch.Tensor | None = None,
    steps: int | None = None,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_id: int = 126336,
    sampling_mode: str = "bernoulli",
    dpls_stop_logit: float = 0.0,
    model_type: str | None = None,
    attention_mask: torch.Tensor | None = None,
    temperature_policy: float = 1.0,
    full_context: bool = False,
    confidences_top_p: int = 1,
    tokenizer=None,
    enable_verifier: bool = False,
    verifier_type: str = "math",
    verifier_schedule: str = "final_only",
    verifier_every_k_steps: int = 8,
    verifier_threshold: float = 0.7,
    enable_remasking: bool = False,
    remask_strategy: str = "none",
    max_remasks_per_sample: int = 1,
    remask_cooldown_steps: int = 2,
    min_steps_before_remask: int = 0,
    remask_answer_span_mode: str = "numeric_only",
    target_budget: int | torch.Tensor | None = None,
    hard_stop_at_target_budget: bool = True,
    hard_generation_budget: int | None = None,
    policy_extra_feature_names: list[str] | None = None,
    log_step_traces: bool = False,
    rollout_compute_style: str | list[str] | tuple[str, ...] | None = None,
    patient_unmask_logit_bias: float = 0.0,
    patient_policy_temperature: float = 1.0,
    patient_max_unmask_fraction: float = 1.0,
    patient_min_steps_frac: float = 0.0,
    patient_global_slowdown: bool = True,
    patient_delay_low_confidence: bool = False,
    patient_low_confidence_quantile: float = 0.3,
    patient_delay_final_window: bool = False,
    patient_final_window_tokens: int = 32,
    patient_final_window_min_steps_frac: float = 0.6,
    enable_deliberation_probe: bool = False,
    deliberation_probe_frac: float = 0.5,
    enable_posthoc_continuation: bool = False,
    continuation_steps: list[int] | None = None,
    continuation_trigger: str = "heuristic",
    continuation_remask_mode: str = "answer_span",
    continuation_final_window_tokens: int = 32,
    continuation_low_confidence_quantile: float = 0.3,
    continuation_min_remaining_budget: int = 1,
    continuation_use_oracle_for_analysis: bool = False,
    continuation_force_for_analysis: bool = False,
    continuation_select_best_k_oracle: bool = False,
    continuation_confidence_threshold: float = 0.5,
    continuation_verifier_threshold: float = 0.7,
) -> GenerationResult:
    if remasking == "policy":
        if policy is None:
            raise ValueError("policy must be provided for remasking='policy'")
    elif remasking == "fastdllm":
        if thres is None:
            raise ValueError("thres must be provided for remasking='fastdllm'")
    elif remasking in ["low_confidence", "random"]:
        if steps is None:
            raise ValueError(f"steps must be provided for remasking='{remasking}'")
    else:
        raise ValueError(f"Unknown remasking strategy: {remasking}")

    B, prompt_L = prompt.shape
    rollout_styles = _normalize_rollout_styles(rollout_compute_style, B)
    L = gen_length
    x = torch.full((B, L + prompt_L), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :prompt_L] = prompt
    steps_taken = torch.zeros((B,), dtype=torch.int32, device=x.device)
    num_blocks = L // block_length

    if attention_mask is not None:
        _attn_mask = torch.ones((B, L + prompt_L), dtype=torch.float, device=x.device)
        _attn_mask[:, :prompt_L] = attention_mask.float()
        if model_type == "Dream":
            _attn_mask = _attn_mask.unsqueeze(1).unsqueeze(-2) * _attn_mask.unsqueeze(
                1
            ).unsqueeze(-1)
        # Handle DDP-wrapped models
        model_dtype = model.module.dtype if hasattr(model, "module") else model.dtype
        _attn_mask = _attn_mask.to(model_dtype)
    else:
        _attn_mask = None

    # Strategy-specific state
    record_policy_data = policy is not None
    sampling_history = [] if record_policy_data else None
    policy_extra_feature_names = policy_extra_feature_names or []

    if enable_verifier and tokenizer is None:
        raise ValueError("tokenizer must be provided when enable_verifier=True")
    if enable_posthoc_continuation and tokenizer is None:
        raise ValueError("tokenizer must be provided for post-hoc continuation")
    if enable_verifier and verifier_type != "math":
        raise ValueError(f"Unsupported verifier_type: {verifier_type}")
    verifier = MathVerifier() if enable_verifier else None
    online_verifier_schedule = verifier_schedule
    if (
        enable_verifier
        and enable_remasking
        and remask_strategy == "verifier_answer"
        and verifier_schedule == "final_only"
    ):
        online_verifier_schedule = "after_answer_detected"
    verifier_remask = (
        VerifierAnswerRemask(
            max_remasks_per_sample=max_remasks_per_sample,
            remask_cooldown_steps=remask_cooldown_steps,
            min_steps_before_remask=min_steps_before_remask,
            verifier_threshold=verifier_threshold,
        )
        if enable_remasking and remask_strategy == "verifier_answer"
        else None
    )
    remask_states = [RemaskState() for _ in range(B)]
    verifier_calls = [0 for _ in range(B)]
    remasked_token_count = [0 for _ in range(B)]
    generated_text_before_remask = [None for _ in range(B)]
    generated_text_after_remask = [None for _ in range(B)]
    last_verifier_results = [None for _ in range(B)]
    answer_span_records = [None for _ in range(B)]
    answer_span_indicator = torch.zeros((B, L), dtype=torch.float, device=x.device)
    last_confidence = torch.zeros((B, L), dtype=torch.float, device=x.device)
    last_entropy = torch.zeros((B, L), dtype=torch.float, device=x.device)
    step_traces = [[] for _ in range(B)] if log_step_traces else None
    early_generated_text = [None for _ in range(B)]
    early_nfe = [None for _ in range(B)]
    patient_cap_applied_count = [0 for _ in range(B)]
    patient_active_step_count = [0 for _ in range(B)]
    patient_selected_before_sum = [0 for _ in range(B)]
    patient_selected_after_sum = [0 for _ in range(B)]
    delayed_low_confidence_count_sum = [0 for _ in range(B)]
    delayed_final_window_count_sum = [0 for _ in range(B)]
    patient_selected_before_delay_sum = [0 for _ in range(B)]
    patient_selected_after_delay_sum = [0 for _ in range(B)]
    final_window_min_step_values = [None for _ in range(B)]

    max_steps = L
    if remasking in ["low_confidence", "random"]:
        assert steps is not None and steps <= L
        tokens_per_step = L // steps
        max_steps = steps
    target_budget_values = _target_budget_values(target_budget, max_steps, B, x.device)
    probe_steps = torch.clamp(
        (target_budget_values.float() * float(deliberation_probe_frac)).long(),
        min=1,
    )
    step_limit = _generation_step_limit(
        target_budget=target_budget,
        max_steps=max_steps,
        batch_size=B,
        device=x.device,
        hard_stop_at_target_budget=hard_stop_at_target_budget,
        hard_generation_budget=hard_generation_budget,
    )

    policy_type = None
    if policy is not None:
        policy_type = (
            policy.module.policy_type
            if hasattr(policy, "module")
            else policy.policy_type
        )

    for num_block in range(num_blocks):
        start_idx = num_block * block_length
        end_idx = start_idx + block_length
        block_slice = slice(start_idx, end_idx)
        block_index = torch.zeros(L, dtype=torch.bool, device=x.device)
        block_index[start_idx:end_idx] = True

        for _ in range(block_length):
            generation_part = x[:, prompt_L:]
            active_budget = steps_taken < step_limit
            mask_index = (generation_part == mask_id) & active_budget.unsqueeze(-1)
            block_mask_index = mask_index[:, block_index]  # (B, BL)

            if (~block_mask_index).all():
                break

            model_output = model(
                x,
                attention_mask=_attn_mask,
                output_hidden_states=(policy_type == "dit_hidden"),
            )

            # Handle Dream model logit shifting
            # Dream: logits at position i predict token i+1
            # For generated tokens at [P, P+1, ..., P+L-1], we need logits at [P-1, P, ..., P+L-2]
            if model_type == "Dream":
                logits = model_output.logits[
                    :, prompt_L - 1 : -1
                ]  # Include last prompt pos, exclude last gen pos
            else:
                logits = model_output.logits[
                    :, prompt_L:
                ]  # Just slice to generation portion

            # Apply Gumbel noise
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Compute softmax once (needed by all strategies)
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
            entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(
                dim=-1
            )
            last_confidence = confidence.detach().float()
            last_entropy = entropy.detach().float()

            # Get unmask decisions based on strategy
            if remasking == "policy":
                unmask, sampling_data = _policy_unmask_decisions(
                    mask_index,
                    block_mask_index,
                    probs,
                    x0,
                    steps_taken,
                    block_slice,
                    L,
                    policy,
                    policy_type,
                    sampling_mode,
                    full_context,
                    confidences_top_p,
                    model_output,
                    prompt_L,
                    dpls_stop_logit,
                    temperature_policy,
                    target_budget,
                    policy_extra_feature_names,
                    entropy,
                    [_verifier_to_dict(result) for result in last_verifier_results],
                    torch.tensor(
                        [state.remask_count for state in remask_states],
                        device=x.device,
                    ),
                    answer_span_indicator,
                    target_budget_values=target_budget_values,
                    rollout_compute_style=rollout_styles,
                    patient_unmask_logit_bias=patient_unmask_logit_bias,
                    patient_policy_temperature=patient_policy_temperature,
                    patient_global_slowdown=patient_global_slowdown,
                    patient_max_unmask_fraction=patient_max_unmask_fraction,
                    patient_min_steps_frac=patient_min_steps_frac,
                    patient_delay_low_confidence=patient_delay_low_confidence,
                    patient_low_confidence_quantile=patient_low_confidence_quantile,
                    patient_delay_final_window=patient_delay_final_window,
                    patient_final_window_tokens=patient_final_window_tokens,
                    patient_final_window_min_steps_frac=(
                        patient_final_window_min_steps_frac
                    ),
                )
                sampling_history.append(sampling_data)
                for batch_idx, stat in enumerate(
                    sampling_data.get("patient_stats", [])
                ):
                    if stat["rollout_compute_style"] != "patient":
                        continue
                    if stat["patient_active_step"]:
                        patient_active_step_count[batch_idx] += 1
                        patient_selected_before_sum[batch_idx] += stat[
                            "selected_unmask_count_before_cap"
                        ]
                        patient_selected_after_sum[batch_idx] += stat[
                            "selected_unmask_count_after_cap"
                        ]
                        delayed_low_confidence_count_sum[batch_idx] += stat[
                            "delayed_low_confidence_count"
                        ]
                        delayed_final_window_count_sum[batch_idx] += stat[
                            "delayed_final_window_count"
                        ]
                        patient_selected_before_delay_sum[batch_idx] += stat[
                            "patient_selected_before_delay"
                        ]
                        patient_selected_after_delay_sum[batch_idx] += stat[
                            "patient_selected_after_delay"
                        ]
                        final_window_min_step_values[batch_idx] = stat[
                            "final_window_min_step"
                        ]
                    if stat["patient_cap_applied"]:
                        patient_cap_applied_count[batch_idx] += 1

            elif remasking == "fastdllm":
                unmask = _confidence_threshold_unmask(
                    block_mask_index, probs, block_slice, thres
                )
                if policy is not None:
                    sampling_data = _record_policy_data(
                        mask_index,
                        block_mask_index,
                        probs,
                        steps_taken,
                        block_slice,
                        L,
                        policy,
                        policy_type,
                        full_context,
                        confidences_top_p,
                        model_output,
                        prompt_L,
                        temperature_policy,
                        unmask,
                        target_budget,
                        policy_extra_feature_names,
                        entropy,
                        [_verifier_to_dict(result) for result in last_verifier_results],
                        torch.tensor(
                            [state.remask_count for state in remask_states],
                            device=x.device,
                        ),
                        answer_span_indicator,
                    )
                    sampling_history.append(sampling_data)

            elif remasking in ["low_confidence", "random"]:
                unmask = _fixed_step_unmask_decisions(
                    block_mask_index,
                    probs,
                    x0,
                    block_slice,
                    tokens_per_step,
                    remasking,
                )

            # Apply unmasking
            x[:, prompt_L:] = torch.where(unmask, x0, generation_part)

            # Update steps taken: only count steps for batch elements that had work to do
            active_this_step = block_mask_index.any(dim=-1)
            current_steps = steps_taken + active_this_step.int()

            if enable_deliberation_probe and tokenizer is not None:
                for batch_idx in range(B):
                    if not bool(active_this_step[batch_idx].item()):
                        continue
                    if early_generated_text[batch_idx] is not None:
                        continue
                    if int(current_steps[batch_idx].item()) >= int(
                        probe_steps[batch_idx].item()
                    ):
                        early_generated_text[batch_idx] = _decode_generation(
                            tokenizer,
                            x[batch_idx, prompt_L:],
                        )
                        early_nfe[batch_idx] = int(current_steps[batch_idx].item())

            if verifier is not None:
                for batch_idx in range(B):
                    if not bool(active_this_step[batch_idx].item()):
                        continue
                    decoded = _decode_generation(tokenizer, x[batch_idx, prompt_L:])
                    current_step = int(current_steps[batch_idx].item())
                    if not _should_run_verifier(
                        online_verifier_schedule,
                        current_step,
                        num_block,
                        num_blocks,
                        decoded,
                        verifier_every_k_steps,
                    ):
                        continue
                    verifier_result = verifier.verify(decoded)
                    verifier_calls[batch_idx] += 1
                    last_verifier_results[batch_idx] = verifier_result
                    span = find_final_answer_span(
                        tokenizer=tokenizer,
                        decoded_text=decoded,
                        generated_token_ids=x[batch_idx, prompt_L:],
                        verifier_result=verifier_result,
                        confidences=confidence[batch_idx],
                        min_token_index=0,
                        special_token_ids={mask_id},
                        remask_span_mode=remask_answer_span_mode,
                    )
                    answer_span_records[batch_idx] = (
                        span.to_dict() if span is not None else None
                    )
                    answer_span_indicator[batch_idx].zero_()
                    if span is not None:
                        answer_span_indicator[batch_idx, span.token_indices] = 1.0

                    remask_decision = None
                    if verifier_remask is not None:
                        remask_decision = verifier_remask.select(
                            verifier_result=verifier_result,
                            span=span,
                            state=remask_states[batch_idx],
                            current_step=current_step,
                            prompt_token_count=0,
                        )
                        if remask_decision.should_remask:
                            if generated_text_before_remask[batch_idx] is None:
                                generated_text_before_remask[batch_idx] = decoded
                            local_indices = torch.tensor(
                                remask_decision.token_indices,
                                dtype=torch.long,
                                device=x.device,
                            )
                            x[batch_idx, prompt_L + local_indices] = mask_id
                            verifier_remask.record(
                                remask_states[batch_idx],
                                remask_decision,
                                current_step,
                            )
                            remasked_token_count[batch_idx] += len(
                                remask_decision.token_indices
                            )
                            generated_text_after_remask[batch_idx] = _decode_generation(
                                tokenizer, x[batch_idx, prompt_L:]
                            )

                    if step_traces is not None:
                        step_traces[batch_idx].append(
                            {
                                "step": current_step,
                                "block": num_block,
                                "verifier_score": verifier_result.verifier_score,
                                "parse_ok": verifier_result.parse_ok,
                                "format_ok": verifier_result.format_ok,
                                "arithmetic_ok": verifier_result.arithmetic_ok,
                                "answer_span_token_indices": span.token_indices
                                if span is not None
                                else [],
                                "remask_indices": remask_decision.token_indices
                                if remask_decision is not None
                                else [],
                                "remask_reason": remask_decision.reason
                                if remask_decision is not None
                                else "none",
                            }
                        )

            steps_taken = current_steps

    if verifier is not None and verifier_schedule != "none":
        for batch_idx in range(B):
            decoded = _decode_generation(tokenizer, x[batch_idx, prompt_L:])
            verifier_result = verifier.verify(decoded)
            verifier_calls[batch_idx] += 1
            last_verifier_results[batch_idx] = verifier_result
            span = find_final_answer_span(
                tokenizer=tokenizer,
                decoded_text=decoded,
                generated_token_ids=x[batch_idx, prompt_L:],
                verifier_result=verifier_result,
                confidences=last_confidence[batch_idx],
                min_token_index=0,
                special_token_ids={mask_id},
                remask_span_mode=remask_answer_span_mode,
            )
            answer_span_records[batch_idx] = span.to_dict() if span is not None else None
            answer_span_indicator[batch_idx].zero_()
            if span is not None:
                answer_span_indicator[batch_idx, span.token_indices] = 1.0

    continuation_results_by_batch: list[list[dict[str, Any]]] = [[] for _ in range(B)]
    continuation_triggered = [False for _ in range(B)]
    continuation_trigger_reasons_by_batch = ["" for _ in range(B)]
    continuation_remasked_indices_by_batch: list[list[int]] = [[] for _ in range(B)]
    if enable_posthoc_continuation:
        special_ids = _tokenizer_special_ids(tokenizer, {mask_id})
        continuation_steps = list(continuation_steps or [])
        for batch_idx in range(B):
            generation_tokens = x[batch_idx, prompt_L:]
            base_text = _decode_generation(tokenizer, generation_tokens)
            verifier_dict = _verifier_to_dict(last_verifier_results[batch_idx])
            span_dict = answer_span_records[batch_idx]
            reward_source = extract_gsm_answer_with_source(
                base_text,
                mode="answer_span",
            ).source
            repetition = compute_repetition_metrics(
                base_text,
                enable_repetition_penalty=True,
            )
            reasons = (
                ["always_for_analysis"]
                if continuation_force_for_analysis
                else continuation_trigger_reasons(
                    trigger=continuation_trigger,
                    verifier_result=verifier_dict,
                    answer_span=span_dict,
                    reward_answer_source=reward_source,
                    repetition_penalty=repetition.penalty,
                    confidence_threshold=continuation_confidence_threshold,
                    verifier_threshold=continuation_verifier_threshold,
                )
            )
            continuation_triggered[batch_idx] = bool(reasons)
            continuation_trigger_reasons_by_batch[batch_idx] = "|".join(reasons)
            if not reasons:
                continue

            remask_indices = select_continuation_remask_indices(
                mode=continuation_remask_mode,
                generated_tokens=generation_tokens,
                answer_span_indices=span_dict.get("token_indices")
                if span_dict
                else None,
                confidences=last_confidence[batch_idx],
                mask_id=mask_id,
                special_token_ids=special_ids,
                final_window_tokens=continuation_final_window_tokens,
                low_confidence_quantile=continuation_low_confidence_quantile,
            )
            continuation_remasked_indices_by_batch[batch_idx] = remask_indices
            sample_target_budget = int(target_budget_values[batch_idx].item())
            sample_base_nfe = int(steps_taken[batch_idx].item())
            for k in continuation_steps:
                k = int(k)
                planned_over_budget = sample_base_nfe + k > sample_target_budget
                remaining_budget = sample_target_budget - sample_base_nfe
                allowed = (
                    not planned_over_budget
                    and remaining_budget >= int(continuation_min_remaining_budget)
                )
                if not allowed and not continuation_use_oracle_for_analysis:
                    continue
                result = _run_continuation_candidate(
                    model=model,
                    base_sequence=x[batch_idx],
                    prompt_L=prompt_L,
                    attention_mask=_attn_mask[batch_idx : batch_idx + 1]
                    if _attn_mask is not None
                    else None,
                    model_type=model_type,
                    temperature=temperature,
                    mask_id=mask_id,
                    tokenizer=tokenizer,
                    verifier=verifier,
                    remask_indices=remask_indices,
                    extra_steps=k,
                    base_nfe=sample_base_nfe,
                    last_confidence=last_confidence[batch_idx],
                    last_entropy=last_entropy[batch_idx],
                    remask_answer_span_mode=remask_answer_span_mode,
                )
                result.update(
                    {
                        "continuation_k": k,
                        "continuation_over_budget": bool(planned_over_budget),
                        "continuation_triggered": True,
                        "continuation_trigger_reason": "|".join(reasons),
                        "continuation_remask_mode": continuation_remask_mode,
                        "continuation_remasked_token_indices": remask_indices,
                        "continuation_remasked_token_count": len(remask_indices),
                    }
                )
                continuation_results_by_batch[batch_idx].append(result)

    step_limit_values = step_limit.to(device=x.device, dtype=torch.long)

    metadata = []
    for batch_idx in range(B):
        generation_tokens = x[batch_idx, prompt_L:]
        final_text = (
            _decode_generation(tokenizer, generation_tokens)
            if tokenizer is not None
            else None
        )
        conf_stats = _tensor_stats(last_confidence[batch_idx])
        ent_stats = _tensor_stats(last_entropy[batch_idx])
        verifier_dict = _verifier_to_dict(last_verifier_results[batch_idx])
        base_span_record = answer_span_records[batch_idx]
        sample_target_budget = int(target_budget_values[batch_idx].item())
        sample_step_limit = int(step_limit_values[batch_idx].item())
        sample_nfe = int(steps_taken[batch_idx].item())
        active_patient_steps = patient_active_step_count[batch_idx]
        avg_selected_before = (
            patient_selected_before_sum[batch_idx] / active_patient_steps
            if active_patient_steps
            else None
        )
        avg_selected_after = (
            patient_selected_after_sum[batch_idx] / active_patient_steps
            if active_patient_steps
            else None
        )
        avg_selected_before_delay = (
            patient_selected_before_delay_sum[batch_idx] / active_patient_steps
            if active_patient_steps
            else None
        )
        avg_selected_after_delay = (
            patient_selected_after_delay_sum[batch_idx] / active_patient_steps
            if active_patient_steps
            else None
        )
        masked_token_count = int((generation_tokens == mask_id).sum().item())
        masked_token_fraction = masked_token_count / max(1, generation_tokens.numel())
        still_masked = masked_token_count > 0
        eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer else None
        has_eos = (
            bool((generation_tokens == int(eos_token_id)).any().item())
            if eos_token_id is not None
            else None
        )
        if still_masked and sample_nfe >= sample_step_limit:
            stop_reason = (
                "target_budget_exhausted"
                if hard_stop_at_target_budget
                else "generation_budget_exhausted"
            )
        elif not still_masked:
            stop_reason = "all_unmasked"
        else:
            stop_reason = "max_steps_reached"
        metadata.append(
            {
                "generated_text_before_remask": generated_text_before_remask[
                    batch_idx
                ],
                "generated_text_after_remask": generated_text_after_remask[batch_idx],
                "generated_text_final": final_text,
                "base_generated_text": final_text,
                "early_generated_text": early_generated_text[batch_idx],
                "early_nfe": early_nfe[batch_idx],
                "final_nfe": sample_nfe,
                "base_nfe": sample_nfe,
                "nfe_used": sample_nfe,
                "actual_nfe": sample_nfe,
                "target_budget": sample_target_budget,
                "max_steps": max_steps,
                "step_limit": sample_step_limit,
                "hard_stop_at_target_budget": hard_stop_at_target_budget,
                "hard_generation_budget": hard_generation_budget,
                "budget_is_hard_cap": hard_stop_at_target_budget,
                "budget_hard_forced": hard_stop_at_target_budget,
                "budget_error": sample_nfe - sample_target_budget,
                "rollout_compute_style": rollout_styles[batch_idx],
                "patient_global_slowdown": patient_global_slowdown,
                "patient_delay_low_confidence": patient_delay_low_confidence,
                "patient_delay_final_window": patient_delay_final_window,
                "patient_min_steps": int(
                    float(patient_min_steps_frac) * sample_target_budget
                ),
                "patient_cap_applied": patient_cap_applied_count[batch_idx] > 0,
                "patient_cap_applied_count": patient_cap_applied_count[batch_idx],
                "selected_unmask_count_before_cap": avg_selected_before,
                "selected_unmask_count_after_cap": avg_selected_after,
                "delayed_low_confidence_count": delayed_low_confidence_count_sum[
                    batch_idx
                ],
                "delayed_final_window_count": delayed_final_window_count_sum[
                    batch_idx
                ],
                "patient_selected_before_delay": avg_selected_before_delay,
                "patient_selected_after_delay": avg_selected_after_delay,
                "final_window_min_step": final_window_min_step_values[batch_idx],
                "verifier_calls": verifier_calls[batch_idx],
                "remask_count": remask_states[batch_idx].remask_count,
                "remasked_token_count": remasked_token_count[batch_idx],
                "masked_token_count": masked_token_count,
                "masked_token_fraction": masked_token_fraction,
                "has_eos": has_eos,
                "final_verifier_score": verifier_dict["verifier_score"]
                if verifier_dict
                else None,
                "parse_ok": verifier_dict["parse_ok"] if verifier_dict else None,
                "base_parse_ok": verifier_dict["parse_ok"] if verifier_dict else None,
                "format_ok": verifier_dict["format_ok"] if verifier_dict else None,
                "base_format_ok": verifier_dict["format_ok"] if verifier_dict else None,
                "arithmetic_ok": verifier_dict["arithmetic_ok"]
                if verifier_dict
                else None,
                "base_arithmetic_ok": verifier_dict["arithmetic_ok"]
                if verifier_dict
                else None,
                "parsed_answer": verifier_dict["final_answer"] if verifier_dict else None,
                "base_parsed_answer": verifier_dict["final_answer"]
                if verifier_dict
                else None,
                "answer_span_text": verifier_dict["final_answer_span_text"]
                if verifier_dict
                else None,
                "base_answer_span": base_span_record,
                "answer_span_token_indices": answer_span_records[batch_idx][
                    "token_indices"
                ]
                if answer_span_records[batch_idx]
                else [],
                "answer_span_mean_confidence": answer_span_records[batch_idx][
                    "mean_confidence"
                ]
                if answer_span_records[batch_idx]
                else None,
                "base_answer_span_confidence": base_span_record["mean_confidence"]
                if base_span_record
                else None,
                "answer_span_min_confidence": answer_span_records[batch_idx][
                    "min_confidence"
                ]
                if answer_span_records[batch_idx]
                else None,
                "base_answer_span_entropy": _tensor_stats(
                    last_entropy[batch_idx, base_span_record["token_indices"]]
                )["mean"]
                if base_span_record and base_span_record.get("token_indices")
                else None,
                "mean_confidence": conf_stats["mean"],
                "min_confidence": conf_stats["min"],
                "entropy_stats": ent_stats,
                "stop_reason": stop_reason,
                "verifier_result": verifier_dict,
                "base_verifier_score": verifier_dict["verifier_score"]
                if verifier_dict
                else None,
                "enable_posthoc_continuation": enable_posthoc_continuation,
                "continuation_triggered": continuation_triggered[batch_idx],
                "continuation_trigger_reason": (
                    continuation_trigger_reasons_by_batch[batch_idx]
                ),
                "continuation_remask_mode": continuation_remask_mode,
                "continuation_remasked_token_indices": (
                    continuation_remasked_indices_by_batch[batch_idx]
                ),
                "continuation_remasked_token_count": len(
                    continuation_remasked_indices_by_batch[batch_idx]
                ),
                "continuation_results": continuation_results_by_batch[batch_idx],
                "continuation_steps": continuation_steps
                if enable_posthoc_continuation
                else [],
                "continuation_use_oracle_for_analysis": (
                    continuation_use_oracle_for_analysis
                ),
                "continuation_force_for_analysis": continuation_force_for_analysis,
                "continuation_select_best_k_oracle": (
                    continuation_select_best_k_oracle
                ),
            }
        )

    # Prepare metadata for gradient steps/loss computation
    if record_policy_data:
        generation_part = x[:, prompt_L:]
        still_masked = (generation_part == mask_id).any(dim=-1)

        if sampling_history:
            # Stack all sampling data for training
            sampling_inputs = torch.stack(
                [h["sampling_inputs"] for h in sampling_history], dim=1
            )
            samples = torch.stack([h["samples"] for h in sampling_history], dim=1)
            sampling_masks = torch.stack(
                [h["sampling_masks"] for h in sampling_history], dim=1
            )

            # Stack policy inputs
            policy_input_columns = zip(*[h["policy_inputs"] for h in sampling_history])
            policy_inputs_result = tuple(
                torch.stack(col, dim=1) for col in policy_input_columns
            )
        else:
            sampling_inputs = samples = sampling_masks = None
            policy_inputs_result = None

        return GenerationResult(
            sequences=x,
            steps_taken=steps_taken,
            sampling_inputs=sampling_inputs,
            samples=samples,
            sampling_masks=sampling_masks,
            policy_inputs=policy_inputs_result,
            still_masked=still_masked,
            metadata=metadata,
            step_traces=step_traces,
        )
    else:
        return GenerationResult(
            sequences=x,
            steps_taken=steps_taken,
            metadata=metadata,
            step_traces=step_traces,
        )


def _get_masks(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    block_slice: slice,
    full_context: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_mask = mask_index if full_context else block_mask_index

    if full_context:
        # Policy sees full sequence (B, L), but we only sample in current block
        sampling_mask = torch.zeros_like(mask_index)
        sampling_mask[:, block_slice] = block_mask_index
    else:
        # Policy sees only block (B, BL), sample from same positions
        sampling_mask = policy_mask

    return policy_mask, sampling_mask


def apply_patient_unmask_controls(
    samples: torch.Tensor,
    *,
    sampling_logits: torch.Tensor,
    sampling_mask: torch.Tensor,
    full_mask_index: torch.Tensor,
    confidence: torch.Tensor | None = None,
    full_context: bool = True,
    block_slice: slice | None = None,
    gen_length: int | None = None,
    steps_taken: torch.Tensor,
    target_budget_values: torch.Tensor,
    rollout_compute_style: list[str],
    patient_global_slowdown: bool = True,
    patient_max_unmask_fraction: float = 1.0,
    patient_min_steps_frac: float = 0.0,
    patient_delay_low_confidence: bool = False,
    patient_low_confidence_quantile: float = 0.3,
    patient_delay_final_window: bool = False,
    patient_final_window_tokens: int = 32,
    patient_final_window_min_steps_frac: float = 0.6,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Apply patient rollout delays while preserving normal actions."""
    adjusted = samples.clone()
    stats: list[dict[str, Any]] = []
    block_start = 0 if block_slice is None else int(block_slice.start or 0)
    total_gen_length = int(gen_length if gen_length is not None else samples.shape[1])
    for batch_idx, style in enumerate(rollout_compute_style):
        selected_before = int(adjusted[batch_idx].sum().item())
        selected_after = selected_before
        cap_applied = False
        delayed_low_confidence_count = 0
        delayed_final_window_count = 0
        target_budget = int(target_budget_values[batch_idx].item())
        patient_min_steps = int(float(patient_min_steps_frac) * target_budget)
        final_window_min_step = int(
            float(patient_final_window_min_steps_frac) * target_budget
        )
        current_step = int(steps_taken[batch_idx].item())
        patient_selected_before_delay = selected_before

        if style == "patient" and selected_before > 0:
            if (
                patient_delay_final_window
                and current_step < final_window_min_step
                and int(patient_final_window_tokens) > 0
            ):
                selected_positions = adjusted[batch_idx].nonzero(as_tuple=True)[0]
                global_positions = (
                    selected_positions
                    if full_context
                    else selected_positions + block_start
                )
                final_start = max(0, total_gen_length - int(patient_final_window_tokens))
                final_positions = selected_positions[global_positions >= final_start]
                if final_positions.numel() > 0:
                    adjusted[batch_idx, final_positions] = False
                    delayed_final_window_count = int(final_positions.numel())

            if patient_delay_low_confidence:
                selected_positions = adjusted[batch_idx].nonzero(as_tuple=True)[0]
                low_conf_fraction = float(patient_low_confidence_quantile)
                if (
                    selected_positions.numel() > 1
                    and 0.0 < low_conf_fraction < 1.0
                    and current_step < max(target_budget - 2, 1)
                ):
                    delay_count = min(
                        selected_positions.numel() - 1,
                        max(1, math.ceil(selected_positions.numel() * low_conf_fraction)),
                    )
                    confidence_source = (
                        confidence if confidence is not None else sampling_logits
                    )
                    selected_confidence = confidence_source[
                        batch_idx,
                        selected_positions,
                    ]
                    delay_local = torch.topk(
                        selected_confidence,
                        k=int(delay_count),
                        largest=False,
                    ).indices
                    delay_positions = selected_positions[delay_local]
                    adjusted[batch_idx, delay_positions] = False
                    delayed_low_confidence_count = int(delay_positions.numel())

            if patient_global_slowdown:
                selected_before_cap = int(adjusted[batch_idx].sum().item())
                if selected_before_cap > 0:
                    remaining_masks = int(full_mask_index[batch_idx].sum().item())
                    max_unmask_now = selected_before_cap
                    if 0.0 < float(patient_max_unmask_fraction) < 1.0:
                        max_from_fraction = max(
                            1,
                            math.ceil(
                                remaining_masks * float(patient_max_unmask_fraction)
                            ),
                        )
                        max_unmask_now = min(max_unmask_now, max_from_fraction)

                    if patient_min_steps > 0 and current_step < patient_min_steps:
                        remaining_min_steps = max(patient_min_steps - current_step, 1)
                        max_from_min_steps = max(
                            1,
                            math.ceil(remaining_masks / remaining_min_steps),
                        )
                        max_unmask_now = min(max_unmask_now, max_from_min_steps)

                    if selected_before_cap > max_unmask_now:
                        selected_positions = adjusted[batch_idx].nonzero(as_tuple=True)[0]
                        selected_scores = sampling_logits[batch_idx, selected_positions]
                        keep_count = max(1, int(max_unmask_now))
                        keep_local = torch.topk(selected_scores, k=keep_count).indices
                        keep_positions = selected_positions[keep_local]
                        adjusted[batch_idx].zero_()
                        adjusted[batch_idx, keep_positions] = True
                        cap_applied = True

            selected_after = int(adjusted[batch_idx].sum().item())

        stats.append(
            {
                "rollout_compute_style": style,
                "patient_global_slowdown": patient_global_slowdown,
                "patient_delay_low_confidence": patient_delay_low_confidence,
                "patient_delay_final_window": patient_delay_final_window,
                "patient_min_steps": patient_min_steps,
                "patient_cap_applied": cap_applied,
                "selected_unmask_count_before_cap": selected_before,
                "selected_unmask_count_after_cap": selected_after,
                "delayed_low_confidence_count": delayed_low_confidence_count,
                "delayed_final_window_count": delayed_final_window_count,
                "patient_selected_before_delay": patient_selected_before_delay,
                "patient_selected_after_delay": selected_after,
                "final_window_min_step": final_window_min_step,
                "patient_active_step": bool(sampling_mask[batch_idx].any().item()),
            }
        )
    return adjusted, stats


def _compute_policy_logits(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    temperature_policy: float,
    target_budget: int | torch.Tensor | None = None,
    policy_extra_feature_names: list[str] | None = None,
    entropy: torch.Tensor | None = None,
    verifier_states: list[dict[str, Any] | None] | None = None,
    remask_counts: torch.Tensor | None = None,
    answer_span_indicator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    """Compute policy logits and masks.

    :return: (policy_logits, policy_mask, sampling_mask, policy_inputs)
    """
    per_batch_timestep = steps_taken.unsqueeze(-1) * (1 / L)
    policy_mask, sampling_mask = _get_masks(
        mask_index, block_mask_index, block_slice, full_context
    )

    topk_result = probs.topk(confidences_top_p, dim=-1)
    c_max_input = (
        topk_result.values if full_context else topk_result.values[:, block_slice]
    )
    if policy_extra_feature_names:
        confidence = probs.max(dim=-1).values
        if entropy is None:
            entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(
                dim=-1
            )
        extra_features = build_policy_extra_features(
            feature_names=policy_extra_feature_names,
            mask_index=mask_index,
            confidence=confidence,
            entropy=entropy,
            steps_taken=steps_taken,
            max_steps=L,
            target_budget=target_budget,
            verifier_states=verifier_states,
            remask_counts=remask_counts,
            answer_span_indicator=answer_span_indicator,
        )
        if extra_features is not None:
            extra_input = extra_features if full_context else extra_features[:, block_slice]
            c_max_input = torch.cat([c_max_input, extra_input.to(c_max_input.dtype)], dim=-1)

    if policy_type == "dit_hidden":
        hidden_states = model_output.hidden_states[-1]
        hidden_states_input = (
            hidden_states[:, prompt_L:, :]
            if full_context
            else hidden_states[
                :, prompt_L + block_slice.start : prompt_L + block_slice.stop, :
            ]
        )
        policy_inputs = (policy_mask, hidden_states_input, per_batch_timestep)
    elif policy_type == "dit_confidence":
        policy_inputs = (policy_mask, c_max_input, per_batch_timestep)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    policy_logits = policy(*policy_inputs)

    # Apply temperature scaling
    if temperature_policy != 1.0:
        policy_logits = policy_logits / temperature_policy

    return policy_logits, policy_mask, sampling_mask, policy_inputs


def _policy_unmask_decisions(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    x0: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    sampling_mode: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    dpls_stop_logit: float = 0.0,
    temperature_policy: float = 1.0,
    target_budget: int | torch.Tensor | None = None,
    policy_extra_feature_names: list[str] | None = None,
    entropy: torch.Tensor | None = None,
    verifier_states: list[dict[str, Any] | None] | None = None,
    remask_counts: torch.Tensor | None = None,
    answer_span_indicator: torch.Tensor | None = None,
    target_budget_values: torch.Tensor | None = None,
    rollout_compute_style: list[str] | None = None,
    patient_unmask_logit_bias: float = 0.0,
    patient_policy_temperature: float = 1.0,
    patient_max_unmask_fraction: float = 1.0,
    patient_min_steps_frac: float = 0.0,
    patient_global_slowdown: bool = True,
    patient_delay_low_confidence: bool = False,
    patient_low_confidence_quantile: float = 0.3,
    patient_delay_final_window: bool = False,
    patient_final_window_tokens: int = 32,
    patient_final_window_min_steps_frac: float = 0.6,
) -> tuple[torch.Tensor, dict]:
    policy_logits, _, sampling_mask, policy_inputs = _compute_policy_logits(
        mask_index,
        block_mask_index,
        probs,
        steps_taken,
        block_slice,
        L,
        policy,
        policy_type,
        full_context,
        confidences_top_p,
        model_output,
        prompt_L,
        temperature_policy,
        target_budget,
        policy_extra_feature_names,
        entropy,
        verifier_states,
        remask_counts,
        answer_span_indicator,
    )
    rollout_compute_style = rollout_compute_style or ["normal"] * policy_logits.size(0)
    target_budget_values = (
        target_budget_values
        if target_budget_values is not None
        else _target_budget_values(
            target_budget,
            L,
            policy_logits.size(0),
            policy_logits.device,
        )
    )
    sampling_logits = policy_logits
    patient_rows = torch.tensor(
        [style == "patient" for style in rollout_compute_style],
        dtype=torch.bool,
        device=policy_logits.device,
    )
    if patient_rows.any() and patient_global_slowdown:
        sampling_logits = policy_logits.clone()
        patient_logits = sampling_logits[patient_rows]
        patient_logits = (
            patient_logits + float(patient_unmask_logit_bias)
        ) / max(float(patient_policy_temperature), 1e-6)
        sampling_logits[patient_rows] = patient_logits

    # Sample based on mode (using sampling_mask which is gated to current block)
    if sampling_mode == "bernoulli":
        b = bernoulli_sample(utilities=sampling_logits, mask_index=sampling_mask)
        b, patient_stats = apply_patient_unmask_controls(
            b,
            sampling_logits=sampling_logits,
            sampling_mask=sampling_mask,
            full_mask_index=mask_index,
            confidence=probs.max(dim=-1).values if full_context else probs.max(dim=-1).values[:, block_slice],
            full_context=full_context,
            block_slice=block_slice,
            gen_length=L,
            steps_taken=steps_taken,
            target_budget_values=target_budget_values,
            rollout_compute_style=rollout_compute_style,
            patient_global_slowdown=patient_global_slowdown,
            patient_max_unmask_fraction=patient_max_unmask_fraction,
            patient_min_steps_frac=patient_min_steps_frac,
            patient_delay_low_confidence=patient_delay_low_confidence,
            patient_low_confidence_quantile=patient_low_confidence_quantile,
            patient_delay_final_window=patient_delay_final_window,
            patient_final_window_tokens=patient_final_window_tokens,
            patient_final_window_min_steps_frac=patient_final_window_min_steps_frac,
        )
        samples_for_loglik = b
    elif sampling_mode == "bernoulli-argmax":
        b = bernoulli_sample(utilities=sampling_logits, mask_index=sampling_mask)
        # For batch items where nothing was selected, force unmask at argmax
        no_selection = b.sum(dim=-1) == 0
        if no_selection.any():
            masked_logits = sampling_logits.clone()
            masked_logits[~sampling_mask] = -torch.inf
            force_idx = torch.argmax(masked_logits, dim=-1)
            batch_indices = torch.arange(b.shape[0], device=b.device)[no_selection]
            b[batch_indices, force_idx[no_selection]] = True
        b, patient_stats = apply_patient_unmask_controls(
            b,
            sampling_logits=sampling_logits,
            sampling_mask=sampling_mask,
            full_mask_index=mask_index,
            confidence=probs.max(dim=-1).values if full_context else probs.max(dim=-1).values[:, block_slice],
            full_context=full_context,
            block_slice=block_slice,
            gen_length=L,
            steps_taken=steps_taken,
            target_budget_values=target_budget_values,
            rollout_compute_style=rollout_compute_style,
            patient_global_slowdown=patient_global_slowdown,
            patient_max_unmask_fraction=patient_max_unmask_fraction,
            patient_min_steps_frac=patient_min_steps_frac,
            patient_delay_low_confidence=patient_delay_low_confidence,
            patient_low_confidence_quantile=patient_low_confidence_quantile,
            patient_delay_final_window=patient_delay_final_window,
            patient_final_window_tokens=patient_final_window_tokens,
            patient_final_window_min_steps_frac=patient_final_window_min_steps_frac,
        )
        samples_for_loglik = b
    elif sampling_mode == "dpls":
        dpls_sequences, b = dpls_sample(
            utilities=sampling_logits,
            stop_logit=dpls_stop_logit,
            mask_index=sampling_mask,
        )
        patient_stats = [
            {
                "rollout_compute_style": style,
                "patient_global_slowdown": patient_global_slowdown,
                "patient_delay_low_confidence": patient_delay_low_confidence,
                "patient_delay_final_window": patient_delay_final_window,
                "patient_min_steps": int(
                    float(patient_min_steps_frac)
                    * int(target_budget_values[idx].item())
                ),
                "patient_cap_applied": False,
                "selected_unmask_count_before_cap": int(b[idx].sum().item()),
                "selected_unmask_count_after_cap": int(b[idx].sum().item()),
                "delayed_low_confidence_count": 0,
                "delayed_final_window_count": 0,
                "patient_selected_before_delay": int(b[idx].sum().item()),
                "patient_selected_after_delay": int(b[idx].sum().item()),
                "final_window_min_step": int(
                    float(patient_final_window_min_steps_frac)
                    * int(target_budget_values[idx].item())
                ),
                "patient_active_step": bool(sampling_mask[idx].any().item()),
            }
            for idx, style in enumerate(rollout_compute_style)
        ]
        samples_for_loglik = dpls_sequences
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    # Convert to sequence-level (always gate to current block)
    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    if full_context:
        unmask[:, block_slice] = b[:, block_slice]
    else:
        unmask[:, block_slice] = b

    sampling_data = {
        # TODO: Patient rollout actions are sampled from a mildly biased behavior
        # policy, but logprob is computed under the base Bernoulli policy for the
        # effective action. Keep patient bonuses small; normal rollout remains the
        # main on-policy signal.
        "sampling_inputs": policy_logits.detach(),
        "samples": samples_for_loglik.detach(),
        "sampling_masks": sampling_mask.detach(),
        "patient_stats": patient_stats,
        "policy_inputs": tuple(
            pi.detach() if isinstance(pi, torch.Tensor) else pi for pi in policy_inputs
        ),
    }

    return unmask, sampling_data


def _confidence_threshold_unmask(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    block_slice: slice,
    thres: float | torch.Tensor,
) -> torch.Tensor:
    confidence = probs.max(dim=-1).values

    # Only consider masked positions in current block
    confidence_masked = confidence[:, block_slice].clone()
    confidence_masked[~block_mask_index] = -torch.inf

    unmask_local = confidence_masked > thres
    has_valid = block_mask_index.any(dim=-1)
    needs_force = has_valid & ~unmask_local.any(dim=-1)
    if needs_force.any():
        force_idx = torch.argmax(confidence_masked, dim=-1)
        batch_indices = torch.arange(unmask_local.shape[0], device=probs.device)
        unmask_local[batch_indices[needs_force], force_idx[needs_force]] = True

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = unmask_local
    return unmask


def _record_policy_data(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    temperature_policy: float,
    unmask: torch.Tensor,
    target_budget: int | torch.Tensor | None = None,
    policy_extra_feature_names: list[str] | None = None,
    entropy: torch.Tensor | None = None,
    verifier_states: list[dict[str, Any] | None] | None = None,
    remask_counts: torch.Tensor | None = None,
    answer_span_indicator: torch.Tensor | None = None,
) -> dict:
    policy_logits, policy_mask, _, policy_inputs = _compute_policy_logits(
        mask_index,
        block_mask_index,
        probs,
        steps_taken,
        block_slice,
        L,
        policy,
        policy_type,
        full_context,
        confidences_top_p,
        model_output,
        prompt_L,
        temperature_policy,
        target_budget,
        policy_extra_feature_names,
        entropy,
        verifier_states,
        remask_counts,
        answer_span_indicator,
    )

    samples = unmask if full_context else unmask[:, block_slice]

    # ES (Expert Steering) special behavior: save policy_mask (not sampling_mask) so the
    # model learns to mimic the confidence thresholding in a block-agnostic way
    return {
        "sampling_inputs": policy_logits.detach().clone(),
        "samples": samples.detach().clone(),
        "sampling_masks": policy_mask.detach().clone(),
        "policy_inputs": tuple(
            pi.detach().clone() if isinstance(pi, torch.Tensor) else pi
            for pi in policy_inputs
        ),
    }


def _fixed_step_unmask_decisions(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    x0: torch.Tensor,
    block_slice: slice,
    tokens_per_step: int,
    mode: str,
) -> torch.Tensor:
    B = block_mask_index.shape[0]
    block_size = block_slice.stop - block_slice.start

    if mode == "low_confidence":
        confidence_block = torch.gather(
            probs[:, block_slice], dim=-1, index=x0[:, block_slice].unsqueeze(-1)
        ).squeeze(-1)
    elif mode == "random":
        confidence_block = torch.rand((B, block_size), device=x0.device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    confidence_masked = torch.where(block_mask_index, confidence_block, -torch.inf)

    num_masked_block = block_mask_index.sum(dim=-1)
    k = torch.clamp(num_masked_block, max=tokens_per_step)
    max_k = k.max().item()

    if max_k == 0:
        unmask = torch.zeros(
            (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
        )
        return unmask

    _, topk_indices = torch.topk(confidence_masked, k=max_k, dim=-1)

    positions = torch.arange(max_k, device=x0.device).unsqueeze(0).expand(B, -1)
    valid_mask = positions < k.unsqueeze(-1)

    unmask_local = torch.zeros_like(block_mask_index, dtype=torch.bool)
    unmask_local.scatter_(1, topk_indices, valid_mask)

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = unmask_local
    return unmask
